import ctypes
import math
import struct
from dataclasses import dataclass

_UNSUPPORTED = {"float64": "float64 is not supported by Metal; use float32 instead"}


@dataclass(frozen=True)
class DType:
    name: str
    msl: str
    size: float
    ctype: type | None
    kind: str
    bits: int
    encode: object
    decode: object

    def __repr__(self) -> str:
        return f"smith.{self.name}"

    def nbytes(self, count) -> int:
        return (count * self.bits + 7) // 8

    def pack(self, values) -> bytes:
        return self.encode(list(values))

    def unpack(self, raw, count) -> list:
        return self.decode(bytes(raw[:self.nbytes(count)]), count)


@dataclass(frozen=True)
class MatrixType:
    element: DType

    @property
    def name(self) -> str:
        return f"matrix_{self.element.name}"

    @property
    def msl(self) -> str:
        return f"simdgroup_{self.element.msl}8x8"

    @property
    def kind(self) -> str:
        return "matrix"

    def __repr__(self) -> str:
        return f"smith.{self.name}"


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


def _float_codec(code):
    def encode(values):
        return b"".join(_pack_float(code, value) for value in values)

    def decode(raw, count):
        return list(struct.unpack(f"<{count}{code}", raw))

    return encode, decode


def _bfloat_codec():
    def encode(values):
        return b"".join(struct.pack("<H", _bfloat_bits(value)) for value in values)

    def decode(raw, count):
        return [_bfloat_value(bits) for (bits,) in struct.iter_unpack("<H", raw)]

    return encode, decode


def _bool_codec():
    def encode(values):
        return struct.pack(f"<{len(values)}?", *(bool(value) for value in values))

    def decode(raw, count):
        return list(struct.unpack(f"<{count}?", raw))

    return encode, decode


def _whole_codec(code):
    unsigned = code.upper()
    mask = (1 << (8 * struct.calcsize(unsigned))) - 1

    def encode(values):
        return struct.pack(f"<{len(values)}{unsigned}", *(int(value) & mask for value in values))

    def decode(raw, count):
        return list(struct.unpack(f"<{count}{code}", raw))

    return encode, decode


def _packed_codec(bits, signed):
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)

    def encode(values):
        data = bytearray((len(values) + per_byte - 1) // per_byte)
        for index, value in enumerate(values):
            data[index // per_byte] |= (int(value) & mask) << ((index % per_byte) * bits)
        return bytes(data)

    def decode(raw, count):
        fields = [(raw[index // per_byte] >> ((index % per_byte) * bits)) & mask for index in range(count)]
        if signed:
            return [field - (1 << bits) if field >= sign_bit else field for field in fields]
        return fields

    return encode, decode


float16 = DType("float16", "half", 2, None, "float", 16, *_float_codec("e"))
bfloat16 = DType("bfloat16", "bfloat", 2, None, "float", 16, *_bfloat_codec())
float32 = DType("float32", "float", 4, ctypes.c_float, "float", 32, *_float_codec("f"))
int2 = DType("int2", "uchar", 0.25, None, "int", 2, *_packed_codec(2, True))
uint2 = DType("uint2", "uchar", 0.25, None, "uint", 2, *_packed_codec(2, False))
int4 = DType("int4", "uchar", 0.5, None, "int", 4, *_packed_codec(4, True))
uint4 = DType("uint4", "uchar", 0.5, None, "uint", 4, *_packed_codec(4, False))
int8 = DType("int8", "char", 1, ctypes.c_int8, "int", 8, *_whole_codec("b"))
int16 = DType("int16", "short", 2, ctypes.c_int16, "int", 16, *_whole_codec("h"))
int32 = DType("int32", "int", 4, ctypes.c_int32, "int", 32, *_whole_codec("i"))
int64 = DType("int64", "long", 8, ctypes.c_int64, "int", 64, *_whole_codec("q"))
uint8 = DType("uint8", "uchar", 1, ctypes.c_uint8, "uint", 8, *_whole_codec("B"))
uint16 = DType("uint16", "ushort", 2, ctypes.c_uint16, "uint", 16, *_whole_codec("H"))
uint32 = DType("uint32", "uint", 4, ctypes.c_uint32, "uint", 32, *_whole_codec("I"))
uint64 = DType("uint64", "ulong", 8, ctypes.c_uint64, "uint", 64, *_whole_codec("Q"))
bool_ = DType("bool", "bool", 1, ctypes.c_bool, "bool", 8, *_bool_codec())

ALL = (float16, bfloat16, float32, int2, uint2, int4, uint4, int8, int16, int32, int64,
       uint8, uint16, uint32, uint64, bool_)


def __getattr__(name):
    if name in _UNSUPPORTED:
        raise AttributeError(_UNSUPPORTED[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
