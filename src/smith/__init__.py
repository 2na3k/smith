from .autotune import autotune, Config, configs_from_grid
from .jit import jit, constexpr, synchronize
from .ops import (program_id, num_programs, arange, load, store, cdiv, Vec, Pointer, loop, exp, log, sqrt, rsqrt,
                  maximum, minimum, where, cast, shuffle_xor, matrix_fill, matrix_load,
                  matrix_mma, matrix_elements, matrix_make)
from .profiler import Profiler


def main() -> None:
    print("Hello from smith!")
