import ctypes
import math
import struct
from dataclasses import dataclass

_INTEGER_CODES = {8: "B", 16: "H", 32: "I", 64: "Q"}
_SIGNED_CODES = {8: "b", 16: "h", 32: "i", 64: "q"}
_FLOAT_CODES = {16: "e", 32: "f"}
_UNSUPPORTED = {"float64": "float64 is not supported by Metal; use float32 instead"}


@dataclass(frozen=True)
class DType:
    name: str
    msl: str
    size: int
    ctype: type | None
    kind: str
    bits: int

    def __repr__(self) -> str:
        return f"smith.{self.name}"

    def pack(self, values) -> bytes:
        values = list(values)
        if self.name == "bfloat16":
            return b"".join(struct.pack("<H", _bfloat_bits(value)) for value in values)
        if self.kind == "float":
            return b"".join(_pack_float(_FLOAT_CODES[self.bits], value) for value in values)
        if self.kind == "bool":
            return struct.pack(f"<{len(values)}?", *(bool(value) for value in values))
        mask = (1 << self.bits) - 1
        return struct.pack(f"<{len(values)}{_INTEGER_CODES[self.bits]}", *(int(value) & mask for value in values))

    def unpack(self, raw, count) -> list:
        raw = bytes(raw[:count * self.size])
        if self.name == "bfloat16":
            return [_bfloat_value(bits) for (bits,) in struct.iter_unpack("<H", raw)]
        if self.kind == "float":
            return list(struct.unpack(f"<{count}{_FLOAT_CODES[self.bits]}", raw))
        if self.kind == "bool":
            return list(struct.unpack(f"<{count}?", raw))
        if self.kind == "int":
            values = struct.unpack(f"<{count}{_SIGNED_CODES[self.bits]}", raw)
            return list(values)
        return list(struct.unpack(f"<{count}{_INTEGER_CODES[self.bits]}", raw))


def _pack_float(code, value):
    try:
        return struct.pack(f"<{code}", value)
    except OverflowError:
        return struct.pack(f"<{code}", math.copysign(math.inf, value))


def _bfloat_bits(value):
    bits = struct.unpack("<I", _pack_float("f", value))[0]
    if math.isnan(value):
        return 0x7FC0
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def _bfloat_value(bits):
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


float16 = DType("float16", "half", 2, None, "float", 16)
bfloat16 = DType("bfloat16", "bfloat", 2, None, "float", 16)
float32 = DType("float32", "float", 4, ctypes.c_float, "float", 32)
int8 = DType("int8", "char", 1, ctypes.c_int8, "int", 8)
int16 = DType("int16", "short", 2, ctypes.c_int16, "int", 16)
int32 = DType("int32", "int", 4, ctypes.c_int32, "int", 32)
int64 = DType("int64", "long", 8, ctypes.c_int64, "int", 64)
uint8 = DType("uint8", "uchar", 1, ctypes.c_uint8, "uint", 8)
uint16 = DType("uint16", "ushort", 2, ctypes.c_uint16, "uint", 16)
uint32 = DType("uint32", "uint", 4, ctypes.c_uint32, "uint", 32)
uint64 = DType("uint64", "ulong", 8, ctypes.c_uint64, "uint", 64)
bool_ = DType("bool", "bool", 1, ctypes.c_bool, "bool", 8)

ALL = (float16, bfloat16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool_)


def __getattr__(name):
    if name in _UNSUPPORTED:
        raise AttributeError(_UNSUPPORTED[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
