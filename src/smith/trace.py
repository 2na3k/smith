from .dtype import DType, float32, int32
from .ir import Graph, Node
from .ops import Pointer, _ctx
from .passes import simplify
from .render import render_metal


class Symbol:
    def __init__(self, node, graph):
        self.node = node
        self.graph = graph

    def _binary(self, op, other, reflected=False):
        other = _lift(self.graph, other)
        left, right = (other, self.node) if reflected else (self.node, other)
        return Symbol(self.graph.binary(op, left, right), self.graph)

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
    def __lt__(self, other): return self._binary("less", other)
    def __le__(self, other): return self._binary("less_equal", other)
    def __gt__(self, other): return self._binary("greater", other)
    def __ge__(self, other): return self._binary("greater_equal", other)
    def __neg__(self): return Symbol(self.graph.negate(self.node), self.graph)

    def __bool__(self):
        raise TypeError("a traced value has no Python truth value: use a mask instead of if/and/or")


def _lift(graph, value):
    if isinstance(value, Symbol):
        return value.node
    if isinstance(value, Node):
        return value
    return graph.const(value)


class TracedBuffer:
    def __init__(self, name, dtype, node):
        self.name = name
        self.dtype = dtype
        self.node = node


class Tracer:
    def __init__(self):
        self.graph = Graph()
        self.lanes = None

    def _symbol(self, node):
        return Symbol(node, self.graph)

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
        offset = _lift(self.graph, pointer.offset)
        mask = None if mask is None else _lift(self.graph, mask)
        return self._symbol(self.graph.load(pointer.buffer.node, offset, mask, _lift(self.graph, other)))

    def store(self, pointer, value, mask):
        offset = _lift(self.graph, pointer.offset)
        mask = None if mask is None else _lift(self.graph, mask)
        self.graph.store(pointer.buffer.node, offset, _lift(self.graph, value), mask)


class BufferSpec:
    def __init__(self, dtype, count=None):
        self.dtype = dtype
        self.count = count


def is_buffer(value):
    return isinstance(getattr(value, "dtype", None), DType)


class Program:
    def __init__(self, source, name, lanes, runtime_names, graph, buffers, scalars):
        self.source = source
        self.name = name
        self.lanes = lanes
        self.runtime_names = runtime_names
        self.graph = graph
        self.buffers = buffers
        self.scalars = scalars
        self.loaded = frozenset(n.inputs[0].arg for n in graph.nodes if n.op == "load")
        self.stored = frozenset(n.inputs[0].arg for n in graph.nodes if n.op == "store")


def trace_kernel(function, values, constexpr_names) -> Program:
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
            call[name] = Symbol(tracer.graph.argument(name, int32), tracer.graph)
            parameters.append(f"constant int& {name} [[buffer({slot})]]")
        elif type(value) is float:
            scalars[name] = float32
            call[name] = Symbol(tracer.graph.argument(name, float32), tracer.graph)
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
    return Program(render_metal(graph, name, parameters), name, tracer.lanes or 1, runtime_names, graph, buffers, scalars)
