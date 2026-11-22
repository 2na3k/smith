from . import dtype
from .compiler import Buffer, compile_msl, launch

ADD_MSL = """
#include <metal_stdlib>
using namespace metal;
kernel void add(device const float* x [[buffer(0)]],
                 device const float* y [[buffer(1)]],
                 device float* out     [[buffer(2)]],
                 uint gid [[thread_position_in_grid]]) {
    out[gid] = x[gid] + y[gid];
}
"""


def add(x: list[float], y: list[float]) -> list[float]:
    assert len(x) == len(y), "length mismatch"
    kernel = compile_msl(ADD_MSL, "add")
    bx = Buffer.from_list(dtype.float32, x)
    by = Buffer.from_list(dtype.float32, y)
    bo = Buffer(dtype.float32, len(x))
    launch(kernel, [bx, by, bo], grid=len(x))
    return bo.to_list()
