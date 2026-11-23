"""Kernel-side primitives. Pure Python, no dependencies.

A "vector" is a Vec (a list of lane values). Pointers are Pointer(buffer, offset) where
offset is an int or a Vec. The launcher in jit.py sets the program context before each run.
"""
import math
import operator

_ctx = {"pid": (0, 0, 0), "grid": (1, 1, 1), "tracer": None}


class Vec:
    def __init__(self, data):
        self.data = list(data)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return f"Vec({self.data})"

    def _bin(self, other, f):
        if isinstance(other, Vec):
            return Vec(f(a, b) for a, b in zip(self.data, other.data))
        return Vec(f(a, other) for a in self.data)

    def _rbin(self, other, f):
        return Vec(f(other, a) for a in self.data)

    def __add__(self, o): return self._bin(o, operator.add)
    def __radd__(self, o): return self._rbin(o, operator.add)
    def __sub__(self, o): return self._bin(o, operator.sub)
    def __rsub__(self, o): return self._rbin(o, operator.sub)
    def __mul__(self, o): return self._bin(o, operator.mul)
    def __rmul__(self, o): return self._rbin(o, operator.mul)
    def __truediv__(self, o): return self._bin(o, operator.truediv)
    def __floordiv__(self, o): return self._bin(o, operator.floordiv)
    def __mod__(self, o): return self._bin(o, operator.mod)
    def __and__(self, o): return self._bin(o, operator.and_)
    def __or__(self, o): return self._bin(o, operator.or_)
    def __lshift__(self, o): return self._bin(o, operator.lshift)
    def __rshift__(self, o): return self._bin(o, operator.rshift)
    def __rlshift__(self, o): return self._rbin(o, operator.lshift)
    def __rrshift__(self, o): return self._rbin(o, operator.rshift)
    def __lt__(self, o): return self._bin(o, operator.lt)
    def __le__(self, o): return self._bin(o, operator.le)
    def __gt__(self, o): return self._bin(o, operator.gt)
    def __ge__(self, o): return self._bin(o, operator.ge)
    def __neg__(self): return Vec(-a for a in self.data)


class Pointer:
    def __init__(self, buffer, offset=0):
        self.buffer = buffer
        self.offset = offset

    def __add__(self, o):
        return Pointer(self.buffer, self.offset + o)

    __radd__ = __add__


def program_id(axis: int = 0) -> int:
    if _ctx["tracer"] is not None:
        return _ctx["tracer"].program_id(axis)
    return _ctx["pid"][axis]


def num_programs(axis: int = 0) -> int:
    if _ctx["tracer"] is not None:
        return _ctx["tracer"].num_programs(axis)
    return _ctx["grid"][axis]


def arange(start: int, end: int) -> Vec:
    if _ctx["tracer"] is not None:
        return _ctx["tracer"].arange(start, end)
    return Vec(range(start, end))


def shared(dtype, count: int) -> Pointer:
    if _ctx["tracer"] is not None:
        return _ctx["tracer"].shared(dtype, count)
    return Pointer([0] * count, 0)


def barrier() -> None:
    if _ctx["tracer"] is not None:
        _ctx["tracer"].barrier()


def copy_to_shared(destination, source, width: int) -> None:
    if _ctx["tracer"] is None:
        raise NotImplementedError("copy_to_shared needs a traced kernel launch")
    _ctx["tracer"].copy_to_shared(destination, source, width)


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _lanes(x, n):
    return x.data if isinstance(x, Vec) else [x] * n


def load(pointer: Pointer, mask=None, other=0):
    if _ctx["tracer"] is not None:
        return _ctx["tracer"].load(pointer, mask, other)
    if isinstance(pointer.offset, Vec):
        n = len(pointer.offset)
        return Vec(pointer.buffer[o] if ok else other for o, ok in zip(pointer.offset.data, _lanes(True if mask is None else mask, n)))
    ok = True if mask is None else (mask.data[0] if isinstance(mask, Vec) else mask)
    return pointer.buffer[pointer.offset] if ok else other


def store(pointer: Pointer, value, mask=None) -> None:
    if _ctx["tracer"] is not None:
        _ctx["tracer"].store(pointer, value, mask)
        return
    if isinstance(pointer.offset, Vec):
        n = len(pointer.offset)
        for o, v, ok in zip(pointer.offset.data, _lanes(value, n), _lanes(True if mask is None else mask, n)):
            if ok:
                pointer.buffer[o] = v
        return
    ok = True if mask is None else (mask.data[0] if isinstance(mask, Vec) else mask)
    if ok:
        pointer.buffer[pointer.offset] = value.data[0] if isinstance(value, Vec) else value


def _map(function, value):
    if isinstance(value, Vec):
        return Vec(function(a) for a in value.data)
    return function(value)


def _combine(function, left, right):
    if isinstance(left, Vec):
        return left._bin(right, function)
    if isinstance(right, Vec):
        return right._rbin(left, function)
    return function(left, right)


def _exp(value):
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def _log(value):
    if value == 0:
        return -math.inf
    return math.log(value) if value > 0 else math.nan


def _sqrt(value):
    return math.sqrt(value) if value >= 0 else math.nan


def _rsqrt(value):
    root = _sqrt(value)
    return 1.0 / root if root else math.inf


def exp(value):
    if _ctx["tracer"] is not None:
        return _ctx["tracer"].unary("exp", value)
    return _map(_exp, value)


def log(value):
    if _ctx["tracer"] is not None:
        return _ctx["tracer"].unary("log", value)
    return _map(_log, value)


def sqrt(value):
    if _ctx["tracer"] is not None:
        return _ctx["tracer"].unary("sqrt", value)
    return _map(_sqrt, value)


def rsqrt(value):
    if _ctx["tracer"] is not None:
        return _ctx["tracer"].unary("rsqrt", value)
    return _map(_rsqrt, value)


def maximum(left, right):
    if _ctx["tracer"] is not None:
        return _ctx["tracer"].binary("maximum", left, right)
    return _combine(max, left, right)


def minimum(left, right):
    if _ctx["tracer"] is not None:
        return _ctx["tracer"].binary("minimum", left, right)
    return _combine(min, left, right)


def _matrix_unavailable():
    raise NotImplementedError("simdgroup matrices need a Metal device: launch the kernel with Buffers")


def matrix_fill(value, element):
    if _ctx["tracer"] is None:
        _matrix_unavailable()
    return _ctx["tracer"].matrix_fill(value, element)


def matrix_load(pointer, stride, transpose=False):
    if _ctx["tracer"] is None:
        _matrix_unavailable()
    return _ctx["tracer"].matrix_load(pointer, stride, transpose)


def matrix_mma(left, right, accumulator):
    if _ctx["tracer"] is None:
        _matrix_unavailable()
    return _ctx["tracer"].matrix_mma(left, right, accumulator)


def matrix_elements(matrix):
    if _ctx["tracer"] is None:
        _matrix_unavailable()
    return _ctx["tracer"].matrix_elements(matrix)


def matrix_make(first, second, element=None):
    if _ctx["tracer"] is None:
        _matrix_unavailable()
    return _ctx["tracer"].matrix_make(first, second, element)


def shuffle_xor(value, mask):
    if _ctx["tracer"] is not None:
        return _ctx["tracer"].shuffle_xor(value, mask)
    if isinstance(value, Vec):
        return Vec(value.data[index ^ mask] for index in range(len(value.data)))
    return value


def where(condition, left, right):
    if _ctx["tracer"] is not None:
        return _ctx["tracer"].where(condition, left, right)
    if isinstance(condition, Vec):
        length = len(condition)
        return Vec(a if flag else b for flag, a, b in zip(condition.data, _lanes(left, length), _lanes(right, length)))
    return left if condition else right


def cast(value, dtype):
    if _ctx["tracer"] is not None:
        return _ctx["tracer"].cast(value, dtype)
    convert = float if dtype.kind == "float" else bool if dtype.kind == "bool" else int
    return _map(convert, value)


def loop(count, init, body):
    if _ctx["tracer"] is not None:
        return _ctx["tracer"].loop(count, init, body)
    carried = tuple(init)
    if isinstance(count, Vec):
        for index in range(max(count.data, default=0)):
            result = body(index, *carried)
            result = tuple(result) if isinstance(result, (tuple, list)) else (result,)
            active = Vec(index < limit for limit in count.data)
            carried = tuple(where(active, new, old) for new, old in zip(result, carried))
        return carried
    for index in range(count):
        result = body(index, *carried)
        carried = tuple(result) if isinstance(result, (tuple, list)) else (result,)
    return carried
