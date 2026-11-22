import time

import mlx.core as mx

import smith as sth
from smith import mlx_bridge


@sth.jit
def add(x_pointer, y_pointer, out_pointer, n, BLOCK: sth.constexpr = 1024):
    pid = sth.program_id(0)
    offset = pid * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    x = sth.load(x_pointer + offset, mask=mask)
    y = sth.load(y_pointer + offset, mask=mask)
    sth.store(out_pointer + offset, x + y, mask=mask)


add_mlx = mlx_bridge.kernel(add)


def timed(function, repeat=10):
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        mx.eval(function())
        best = min(best, time.perf_counter() - start)
    return best


if __name__ == "__main__":
    n = 1 << 24
    block = 1024
    x = mx.random.uniform(shape=(n,))
    y = mx.random.uniform(shape=(n,))
    out = mx.zeros((n,), dtype=mx.float32)
    grid = lambda meta: (sth.cdiv(n, meta["BLOCK"]),)

    chained = mx.multiply(add_mlx[grid](x, y, out, n, BLOCK=block), 2.0)
    expected = (x + y) * 2.0
    mx.eval(chained)
    print("max abs error vs mx.add:", mx.max(mx.abs(chained - expected)).item())

    smith_time = timed(lambda: add_mlx[grid](x, y, out, n, BLOCK=block))
    mlx_time = timed(lambda: mx.add(x, y))
    print(f"elements: {n}")
    print(f"smith kernel via mlx: {smith_time * 1e3:.3f} ms")
    print(f"mx.add:               {mlx_time * 1e3:.3f} ms")
