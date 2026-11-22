"""Kernel-side primitives. Pure Python, no dependencies.

A "vector" is a Vec (a list of lane values). Pointers are Pointer(buffer, offset) where
offset is an int or a Vec. The launcher in jit.py sets the program context before each run.
"""
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
