from .dtype import DType, float32, int32
from .ir import Graph, LoopBody, Node, walk
from .ops import Pointer, _ctx
from .passes import simplify
from .render import render_metal


def _graph():
    return _ctx["tracer"].graph


class Symbol:
    def __init__(self, node):
        self.node = node

    def _binary(self, op, other, reflected=False):
        other = _lift(other)
        left, right = (other, self.node) if reflected else (self.node, other)
        return Symbol(_graph().binary(op, left, right))

    def __add__(self, other): return self._binary("add", other)
    def __radd__(self, other): return self._binary("add", other, True)
    def __sub__(self, other): return self._binary("sub", other)
    def __rsub__(self, other): return self._binary("sub", other, True)
    def __mul__(self, other): return self._binary("mul", other)
    def __rmul__(self, other): return self._binary("mul", other, True)
    def __truediv__(self, other): return self._binary("truediv", other)
    def __rtruediv__(self, other): return self._binary("truediv", other, True)
    def __floordiv__(self, other): return self._binary("floordiv", other)
    def __rfloordiv__(self, other): return self._binary("floordiv", other, True)
    def __mod__(self, other): return self._binary("mod", other)
    def __rmod__(self, other): return self._binary("mod", other, True)
    def __and__(self, other): return self._binary("and", other)
    def __rand__(self, other): return self._binary("and", other, True)
    def __or__(self, other): return self._binary("or", other)
    def __ror__(self, other): return self._binary("or", other, True)
    def __lshift__(self, other): return self._binary("shl", other)
    def __rlshift__(self, other): return self._binary("shl", other, True)
    def __rshift__(self, other): return self._binary("shr", other)
    def __rrshift__(self, other): return self._binary("shr", other, True)
    def __lt__(self, other): return self._binary("less", other)
    def __le__(self, other): return self._binary("less_equal", other)
    def __gt__(self, other): return self._binary("greater", other)
    def __ge__(self, other): return self._binary("greater_equal", other)
    def __neg__(self): return Symbol(_graph().negate(self.node))

    def __bool__(self):
        raise TypeError("a traced value has no Python truth value: use a mask instead of if/and/or")


def _lift(value):
    if isinstance(value, Symbol):
        return value.node
    if isinstance(value, Node):
        return value
    return _graph().const(value)


class TracedBuffer:
    def __init__(self, name, dtype, node):
        self.name = name
        self.dtype = dtype
        self.node = node


class Tracer:
    def __init__(self):
        self.graph = Graph()
        self.lanes = None
        self.shared_count = 0

    def _symbol(self, node):
        return Symbol(node)

    def unary(self, op, value):
        return self._symbol(self.graph.unary(op, _lift(value)))

    def binary(self, op, left, right):
        return self._symbol(self.graph.binary(op, _lift(left), _lift(right)))

    def matrix_fill(self, value, element):
        return self._symbol(self.graph.matrix_fill(_lift(value), element))

    def matrix_load(self, pointer, stride, transpose):
        return self._symbol(self.graph.matrix_load(pointer.buffer.node, _lift(pointer.offset), _lift(stride), transpose))

    def matrix_mma(self, left, right, accumulator):
        return self._symbol(self.graph.matrix_mma(_lift(left), _lift(right), _lift(accumulator)))

    def matrix_elements(self, matrix):
        node = _lift(matrix)
        return self._symbol(self.graph.matrix_element(node, 0)), self._symbol(self.graph.matrix_element(node, 1))

    def matrix_make(self, first, second, element):
        return self._symbol(self.graph.matrix_make(_lift(first), _lift(second), element))

    def shuffle_xor(self, value, mask):
        return self._symbol(self.graph.shuffle_xor(_lift(value), mask))

    def where(self, condition, left, right):
        return self._symbol(self.graph.where(_lift(condition), _lift(left), _lift(right)))

    def cast(self, value, dtype):
        return self._symbol(self.graph.cast(_lift(value), dtype))

    def loop(self, count, init, body):
        outer = self.graph
        inits = [_lift(value) for value in init]
        child = Graph(parent=outer)
        index = child.node("loop_index", dtype=int32)
        carried = [child.node("carried", arg=slot, dtype=node.dtype) for slot, node in enumerate(inits)]
        self.graph = child
        try:
            results = body(Symbol(index), *(Symbol(node) for node in carried))
            results = tuple(results) if isinstance(results, (tuple, list)) else (results,)
            if len(results) != len(inits):
                raise ValueError(f"loop body returned {len(results)} values for {len(inits)} carried values")
            outputs = [child.cast(_lift(value), node.dtype) for value, node in zip(results, carried)]
        finally:
            self.graph = outer
        finished = outer.loop(outer.const(0), _lift(count), inits, LoopBody(child, index, carried, outputs))
        return tuple(Symbol(node) for node in finished)

    def shared(self, dtype, count):
        name = f"shared_{self.shared_count}"
        self.shared_count += 1
        return Pointer(TracedBuffer(name, dtype, self.graph.shared(name, dtype, count)), 0)

    def barrier(self):
        self.graph.barrier()

    def copy_to_shared(self, destination, source, width):
        self.graph.copy(destination.buffer.node, _lift(destination.offset), source.buffer.node, _lift(source.offset), width)

    def program_id(self, axis):
        return self._symbol(self.graph.program_id(axis))

    def num_programs(self, axis):
        return self._symbol(self.graph.program_count(axis))

    def arange(self, start, end):
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError("arange bounds must be Python integers or constexpr values")
        length = end - start
        if self.lanes not in (None, length):
            raise ValueError(f"every arange in a kernel must have the same length, got {self.lanes} and {length}")
        self.lanes = length
        lane = self.graph.lane()
        return self._symbol(lane if start == 0 else self.graph.binary("add", lane, self.graph.const(start)))

    def load(self, pointer, mask, other):
        offset = _lift(pointer.offset)
        mask = None if mask is None else _lift(mask)
        return self._symbol(self.graph.load(pointer.buffer.node, offset, mask, _lift(other)))

    def store(self, pointer, value, mask):
        offset = _lift(pointer.offset)
        mask = None if mask is None else _lift(mask)
        self.graph.store(pointer.buffer.node, offset, _lift(value), mask)


class BufferSpec:
    def __init__(self, dtype, count=None):
        self.dtype = dtype
        self.count = count


def is_buffer(value):
    return isinstance(getattr(value, "dtype", None), DType)


class Program:
    def __init__(self, source, name, lanes, runtime_names, graph, buffers, scalars, traced=None, constexprs=None):
        self.source = source
        self.traced = traced
        self.constexprs = constexprs or {}
        self.name = name
        self.lanes = lanes
        self.runtime_names = runtime_names
        self.graph = graph
        self.buffers = buffers
        self.scalars = scalars
        self.uses_matrices = any(n.op.startswith("matrix_") for n in walk(graph))
        read = [n.inputs[0] for n in walk(graph) if n.op in ("load", "matrix_load")] + [n.inputs[2] for n in walk(graph) if n.op == "copy"]
        self.loaded = frozenset(source.arg for source in read if source.op == "buffer")
        self.stored = frozenset(n.inputs[0].arg for n in walk(graph) if n.op == "store" and n.inputs[0].op == "buffer")


RESERVED_NAMES = frozenset({
    "double", "float", "half", "bfloat", "int", "uint", "short", "ushort", "char", "uchar", "long", "ulong", "bool",
    "void", "kernel", "vertex", "fragment", "device", "constant", "threadgroup", "thread", "main", "struct", "class",
    "namespace", "using", "if", "else", "for", "while", "do", "return", "switch", "case", "default", "template",
})


def trace_kernel(function, values, constexpr_names) -> Program:
    if function.__name__ in RESERVED_NAMES:
        raise ValueError(f"{function.__name__!r} is a reserved word in Metal and cannot name a kernel: rename the function")
    tracer = Tracer()
    parameters, runtime_names, call = [], [], {}
    buffers, scalars = {}, {}
    for name, value in values.items():
        slot = len(runtime_names)
        if name in constexpr_names:
            call[name] = value
            continue
        if is_buffer(value):
            buffers[name] = value.dtype
            node = tracer.graph.buffer(name, value.dtype)
            call[name] = Pointer(TracedBuffer(name, value.dtype, node), 0)
            parameters.append(f"device {value.dtype.msl}* {name} [[buffer({slot})]]")
        elif type(value) is int:
            scalars[name] = int32
            call[name] = Symbol(tracer.graph.argument(name, int32))
            parameters.append(f"constant int& {name} [[buffer({slot})]]")
        elif type(value) is float:
            scalars[name] = float32
            call[name] = Symbol(tracer.graph.argument(name, float32))
            parameters.append(f"constant float& {name} [[buffer({slot})]]")
        else:
            raise TypeError(f"unsupported kernel argument {name!r} of type {type(value).__name__}")
        runtime_names.append(name)

    _ctx["tracer"] = tracer
    try:
        function(**call)
    finally:
        _ctx["tracer"] = None

    parameters += [
        "uint3 program_id [[threadgroup_position_in_grid]]",
        "uint3 lane_id [[thread_position_in_threadgroup]]",
        "uint3 program_count [[threadgroups_per_grid]]",
    ]
    graph = simplify(tracer.graph)
    name = function.__name__
    return Program(render_metal(graph, name, parameters), name, tracer.lanes or 1, runtime_names, graph, buffers, scalars,
                   tracer.graph, {key: values[key] for key in constexpr_names if key in values})
