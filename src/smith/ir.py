from .dtype import (MatrixType, bfloat16, bool_, float16, float32, int2, int4, int8, int16, int32, int64, uint2, uint4, uint8,
                    uint16, uint32, uint64)

RANK = {bool_: 0, int8: 1, uint8: 1, int16: 2, uint16: 2, int32: 3, uint32: 3, int64: 4, uint64: 4,
        float16: 5, bfloat16: 5, float32: 6}
ARITHMETIC = ("add", "sub", "mul", "truediv", "floordiv", "mod")
BITWISE = ("and", "or")
SHIFTS = ("shl", "shr")
COMPARISONS = ("less", "less_equal", "greater", "greater_equal")
EXTREMA = ("maximum", "minimum")
MATH = ("exp", "log", "sqrt", "rsqrt")
BINARY = ARITHMETIC + BITWISE + SHIFTS + COMPARISONS + EXTREMA
RECREATED = ("const", "argument", "buffer", "shared", "program_id", "program_count", "lane")
LEAVES = RECREATED + ("capture", "loop_index", "carried")
EFFECTS = ("store", "barrier", "copy")
MATRIX = ("matrix_fill", "matrix_load", "matrix_mma", "matrix_element", "matrix_make")
PACKED_ELEMENTS = {int2: int8, uint2: uint8, int4: int8, uint4: uint8}
INTEGERS = {("int", 8): int8, ("int", 16): int16, ("int", 32): int32, ("int", 64): int64,
            ("uint", 8): uint8, ("uint", 16): uint16, ("uint", 32): uint32, ("uint", 64): uint64}


def _scalars(*nodes):
    for node in nodes:
        if isinstance(node.dtype, MatrixType):
            raise TypeError("simdgroup matrices only support the matrix operations")


def promote(*dtypes):
    floats = {dtype for dtype in dtypes if dtype.kind == "float"}
    if floats:
        if floats == {float16, bfloat16}:
            return float32
        return max(floats, key=RANK.__getitem__)
    integers = [dtype for dtype in dtypes if dtype.kind in ("int", "uint")]
    if not integers:
        return int32
    kind = "int" if any(dtype.kind == "int" for dtype in integers) else "uint"
    return INTEGERS[(kind, max(dtype.bits for dtype in integers))]


class Node:
    __slots__ = ("op", "inputs", "arg", "dtype", "index", "graph")

    def __init__(self, op, inputs, arg, dtype, index, graph):
        self.op = op
        self.inputs = inputs
        self.arg = arg
        self.dtype = dtype
        self.index = index
        self.graph = graph

    def __repr__(self):
        return f"Node(%{self.index} {self.op} {self.arg!r} {self.dtype})"


class LoopBody:
    def __init__(self, graph, index, carried, outputs):
        self.graph = graph
        self.index = index
        self.carried = carried
        self.outputs = outputs
        self.captures = list(graph.captures)
        self.has_stores = any(node.op in EFFECTS or (node.op == "loop" and node.arg.has_stores)
                              for node in graph.nodes)


class Graph:
    def __init__(self, parent=None):
        self.nodes = []
        self.table = {}
        self.version = 0
        self.parent = parent
        self.captures = []
        self.captured = {}

    def _own(self, node):
        if node.graph is self:
            return node
        if node.op in RECREATED:
            return self.node(node.op, (), node.arg, node.dtype)
        if self.parent is None:
            raise ValueError("a value from another graph cannot be used here")
        if id(node) not in self.captured:
            outer = self.parent._own(node)
            leaf = self.node("capture", (), len(self.captures), node.dtype, unique=True)
            self.captures.append(outer)
            self.captured[id(node)] = leaf
        return self.captured[id(node)]

    def node(self, op, inputs=(), arg=None, dtype=None, unique=False):
        inputs = tuple(self._own(i) for i in inputs)
        key = (op, tuple(i.index for i in inputs), arg, dtype)
        if not unique and key in self.table:
            return self.table[key]
        node = Node(op, inputs, arg, dtype, len(self.nodes), self)
        self.nodes.append(node)
        if not unique:
            self.table[key] = node
        return node

    def const(self, value):
        if isinstance(value, bool):
            dtype = bool_
        elif isinstance(value, float):
            dtype = float32
        elif isinstance(value, int):
            dtype = int32
        else:
            raise TypeError(f"cannot use {type(value).__name__} as a constant")
        return self.node("const", arg=value, dtype=dtype)

    def argument(self, name, dtype):
        return self.node("argument", arg=name, dtype=dtype)

    def buffer(self, name, dtype):
        return self.node("buffer", arg=name, dtype=dtype)

    def shared(self, name, dtype, count):
        return self.node("shared", arg=(name, count), dtype=dtype)

    def barrier(self):
        self.version += 1
        return self.node("barrier", arg=self.version, unique=True)

    def program_id(self, axis):
        return self.node("program_id", arg=axis, dtype=int32)

    def program_count(self, axis):
        return self.node("program_count", arg=axis, dtype=int32)

    def lane(self):
        return self.node("lane", dtype=int32)

    def cast(self, node, dtype):
        if node.dtype == dtype:
            return node
        _scalars(node)
        return self.node("cast", (node,), dtype=dtype)

    def binary(self, op, left, right):
        _scalars(left, right)
        if op in COMPARISONS:
            operand, result = promote(left.dtype, right.dtype), bool_
        elif op == "truediv":
            operand = result = float32
        elif op in BITWISE and left.dtype == bool_ and right.dtype == bool_:
            operand = result = bool_
        else:
            operand = result = promote(left.dtype, right.dtype)
        if op in BITWISE + SHIFTS and operand.kind == "float":
            raise TypeError(f"{op} is not defined for float values")
        return self.node(op, (self.cast(left, operand), self.cast(right, operand)), dtype=result)

    def unary(self, op, value):
        _scalars(value)
        dtype = value.dtype if value.dtype.kind == "float" else float32
        return self.node(op, (self.cast(value, dtype),), dtype=dtype)

    def shuffle_xor(self, value, mask):
        return self.node("shuffle_xor", (value,), arg=mask, dtype=value.dtype)

    def where(self, condition, left, right):
        _scalars(condition, left, right)
        dtype = promote(left.dtype, right.dtype)
        return self.node("where", (self.cast(condition, bool_), self.cast(left, dtype), self.cast(right, dtype)),
                         dtype=dtype)

    def negate(self, value):
        _scalars(value)
        operand = promote(value.dtype)
        return self.node("neg", (self.cast(value, operand),), dtype=operand)

    def _load_packed(self, buffer, offset, mask, other):
        dtype = buffer.dtype
        per_byte = 8 // dtype.bits
        field_mask = self.const((1 << dtype.bits) - 1)
        position = self.cast(offset, int32)
        index = self.binary("shr", position, self.const(per_byte.bit_length() - 1))
        shift = self.binary("shl", self.binary("and", position, self.const(per_byte - 1)),
                            self.const(dtype.bits.bit_length() - 1))
        inputs = [buffer, index]
        if mask is not None:
            fallback = self.binary("shl", self.binary("and", self.cast(other, int32), field_mask), shift)
            inputs += [self.cast(mask, bool_), self.cast(fallback, uint8)]
        raw = self.node("load", inputs, arg=self.version, dtype=uint8)
        field = self.binary("and", self.binary("shr", raw, shift), field_mask)
        if dtype.kind == "int":
            sign = self.binary("shl", self.binary("and", field, self.const(1 << (dtype.bits - 1))), self.const(1))
            field = self.binary("sub", field, sign)
        return self.cast(field, PACKED_ELEMENTS[dtype])

    def load(self, buffer, offset, mask, other):
        if buffer.dtype in PACKED_ELEMENTS:
            return self._load_packed(buffer, offset, mask, other)
        inputs = [buffer, self.cast(offset, int32)]
        if mask is not None:
            inputs += [self.cast(mask, bool_), self.cast(other, buffer.dtype)]
        return self.node("load", inputs, arg=self.version, dtype=buffer.dtype)

    def store(self, buffer, offset, value, mask):
        if buffer.dtype in PACKED_ELEMENTS:
            raise TypeError(f"cannot store into a packed {buffer.dtype.name} buffer: "
                            "compute two values per lane and store one packed byte into a uint8 buffer")
        inputs = [buffer, self.cast(offset, int32), self.cast(value, buffer.dtype)]
        if mask is not None:
            inputs.append(self.cast(mask, bool_))
        self.version += 1
        return self.node("store", inputs, arg=self.version, unique=True)

    def copy(self, destination, destination_offset, source, source_offset, width):
        if destination.op != "shared" or source.op != "buffer":
            raise TypeError("copy_to_shared copies from a kernel buffer into shared memory")
        if destination.dtype != source.dtype:
            raise TypeError(f"copy_to_shared needs matching element types, got {destination.dtype} and {source.dtype}")
        if destination.dtype in PACKED_ELEMENTS or destination.dtype.bits % 8 or destination.dtype.bits * width not in (32, 64, 128):
            raise TypeError("copy_to_shared moves 4, 8 or 16 bytes per thread: width * element size must be one of those")
        inputs = (destination, self.cast(destination_offset, int32), source, self.cast(source_offset, int32))
        self.version += 1
        return self.node("copy", inputs, arg=width, unique=True)

    def matrix_fill(self, value, element):
        if element.kind != "float":
            raise TypeError("simdgroup matrices hold float values")
        return self.node("matrix_fill", (self.cast(value, element),), dtype=MatrixType(element))

    def matrix_load(self, buffer, offset, stride, transpose):
        if buffer.dtype.kind != "float":
            raise TypeError("simdgroup matrices can only be loaded from float buffers")
        inputs = (buffer, self.cast(offset, int32), self.cast(stride, int32))
        return self.node("matrix_load", inputs, arg=(bool(transpose), self.version), dtype=MatrixType(buffer.dtype))

    def matrix_mma(self, left, right, accumulator):
        for matrix in (left, right, accumulator):
            if not isinstance(matrix.dtype, MatrixType):
                raise TypeError("matrix_mma needs matrices")
        if left.dtype != right.dtype:
            raise TypeError(f"matrix_mma inputs must have the same element type, got {left.dtype} and {right.dtype}")
        return self.node("matrix_mma", (left, right, accumulator), dtype=accumulator.dtype)

    def matrix_element(self, matrix, index):
        if not isinstance(matrix.dtype, MatrixType) or index not in (0, 1):
            raise TypeError("matrix_element reads element 0 or 1 of a matrix")
        return self.node("matrix_element", (matrix,), arg=index, dtype=matrix.dtype.element)

    def matrix_make(self, first, second, element=None):
        element = element or first.dtype
        if element.kind != "float":
            raise TypeError("simdgroup matrices hold float values")
        return self.node("matrix_make", (self.cast(first, element), self.cast(second, element)), dtype=MatrixType(element))

    def loop(self, start, end, inits, body):
        inputs = (self.cast(start, int32), self.cast(end, int32), *inits, *body.captures)
        loop = self.node("loop", inputs, arg=body, unique=True)
        if body.has_stores:
            self.version += 1
        return [self.node("loop_result", (loop,), arg=slot, dtype=init.dtype) for slot, init in enumerate(inits)]


def walk(graph):
    for node in graph.nodes:
        yield node
        if node.op == "loop":
            yield from walk(node.arg.graph)


def live_nodes(graph, roots=()):
    live = set()
    stack = [node for node in graph.nodes if node.op in EFFECTS or (node.op == "loop" and node.arg.has_stores)]
    stack.extend(roots)
    while stack:
        node = stack.pop()
        if node.index in live:
            continue
        live.add(node.index)
        stack.extend(node.inputs)
    return live


def format_graph(graph):
    live = live_nodes(graph)
    lines = []
    for node in graph.nodes:
        if node.index not in live:
            continue
        inputs = " ".join(f"%{i.index}" for i in node.inputs)
        arg = "" if node.arg is None else f"[{node.arg}]"
        dtype = "" if node.dtype is None else f" : {node.dtype.name}"
        lines.append(f"%{node.index} = {node.op}{arg} {inputs}{dtype}".replace("  ", " "))
    return "\n".join(lines)
