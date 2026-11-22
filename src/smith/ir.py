from .dtype import bool_, float32, int32, uint32

RANK = {bool_: 0, uint32: 1, int32: 2, float32: 3}
ARITHMETIC = ("add", "sub", "mul", "truediv", "floordiv", "mod")
BITWISE = ("and", "or")
COMPARISONS = ("less", "less_equal", "greater", "greater_equal")
BINARY = ARITHMETIC + BITWISE + COMPARISONS
LEAVES = ("const", "argument", "buffer", "program_id", "program_count", "lane")


def promote(*dtypes):
    result = max(dtypes, key=RANK.__getitem__)
    return result if RANK[result] >= RANK[int32] else int32


class Node:
    __slots__ = ("op", "inputs", "arg", "dtype", "index")

    def __init__(self, op, inputs, arg, dtype, index):
        self.op = op
        self.inputs = inputs
        self.arg = arg
        self.dtype = dtype
        self.index = index

    def __repr__(self):
        return f"Node(%{self.index} {self.op} {self.arg!r} {self.dtype})"


class Graph:
    def __init__(self):
        self.nodes = []
        self.table = {}
        self.version = 0

    def node(self, op, inputs=(), arg=None, dtype=None, unique=False):
        inputs = tuple(inputs)
        key = (op, tuple(i.index for i in inputs), arg, dtype)
        if not unique and key in self.table:
            return self.table[key]
        node = Node(op, inputs, arg, dtype, len(self.nodes))
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

    def program_id(self, axis):
        return self.node("program_id", arg=axis, dtype=int32)

    def program_count(self, axis):
        return self.node("program_count", arg=axis, dtype=int32)

    def lane(self):
        return self.node("lane", dtype=int32)

    def cast(self, node, dtype):
        if node.dtype == dtype:
            return node
        return self.node("cast", (node,), dtype=dtype)

    def binary(self, op, left, right):
        if op in COMPARISONS:
            operand, result = promote(left.dtype, right.dtype), bool_
        elif op == "truediv":
            operand = result = float32
        elif op in BITWISE and left.dtype == bool_ and right.dtype == bool_:
            operand = result = bool_
        else:
            operand = result = promote(left.dtype, right.dtype)
        if op in BITWISE and operand == float32:
            raise TypeError(f"{op} is not defined for float values")
        return self.node(op, (self.cast(left, operand), self.cast(right, operand)), dtype=result)

    def negate(self, value):
        operand = promote(value.dtype)
        return self.node("neg", (self.cast(value, operand),), dtype=operand)

    def load(self, buffer, offset, mask, other):
        inputs = [buffer, self.cast(offset, int32)]
        if mask is not None:
            inputs += [self.cast(mask, bool_), self.cast(other, buffer.dtype)]
        return self.node("load", inputs, arg=self.version, dtype=buffer.dtype)

    def store(self, buffer, offset, value, mask):
        inputs = [buffer, self.cast(offset, int32), self.cast(value, buffer.dtype)]
        if mask is not None:
            inputs.append(self.cast(mask, bool_))
        self.version += 1
        return self.node("store", inputs, arg=self.version, unique=True)


def live_nodes(graph):
    live = set()
    stack = [node for node in graph.nodes if node.op == "store"]
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
