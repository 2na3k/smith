import ctypes
from dataclasses import dataclass


@dataclass(frozen=True)
class DType:
    name: str
    msl: str
    size: int
    ctype: type

    def __repr__(self) -> str:
        return f"smith.{self.name}"


float32 = DType("float32", "float", 4, ctypes.c_float)
int32 = DType("int32", "int", 4, ctypes.c_int32)
uint32 = DType("uint32", "uint", 4, ctypes.c_uint32)
bool_ = DType("bool", "bool", 1, ctypes.c_bool)

ALL = (float32, int32, uint32, bool_)
