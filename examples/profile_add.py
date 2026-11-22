import sys

import smith as sth
from smith import dtype
from smith.profiler import Profiler

if sys.platform != "darwin":
    print("profile_add needs macOS with Metal")
    raise SystemExit(0)

from smith.compiler import Buffer


@sth.jit
def add(x_pointer, y_pointer, out_pointer, n, BLOCK: sth.constexpr = 256):
    offset = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    x = sth.load(x_pointer + offset, mask=mask)
    y = sth.load(y_pointer + offset, mask=mask)
    sth.store(out_pointer + offset, x + y, mask=mask)


if __name__ == "__main__":
    n = 1 << 18
    block = 256
    x = Buffer.from_list(dtype.float32, [float(i) for i in range(n)])
    y = Buffer.from_list(dtype.float32, [2.0 * i for i in range(n)])
    out = Buffer.from_list(dtype.float32, [0.0] * n)
    grid = lambda meta: (sth.cdiv(n, meta["BLOCK"]),)

    program = add.trace(x, y, out, n, BLOCK=block)
    with Profiler().track(program, {"n": n}) as profiler:
        for _ in range(10):
            add[grid](x, y, out, n, BLOCK=block)

    print(profiler.report())
    print()
    print("out[:4] =", out.to_list()[:4])
    print("out[-1] =", out.to_list()[-1], "(expected", 3.0 * (n - 1), ")")
