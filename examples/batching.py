import statistics
import sys
import time

import smith as sth
from smith import dtype
from smith.compiler import Buffer


@sth.jit
def add(x_pointer, y_pointer, out_pointer, n, BLOCK: sth.constexpr = 256):
    offset = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    x = sth.load(x_pointer + offset, mask=mask)
    y = sth.load(y_pointer + offset, mask=mask)
    sth.store(out_pointer + offset, x + y, mask=mask)


def measure(launches, batched, x, y, out, n):
    grid = lambda meta: (n // meta["BLOCK"],)

    def run():
        if batched:
            with sth.batch():
                for _ in range(launches):
                    add[grid](x, y, out, n, BLOCK=256)
        else:
            for _ in range(launches):
                add[grid](x, y, out, n, BLOCK=256)

    for _ in range(5):
        run()
        sth.synchronize()
    submit, total = [], []
    for _ in range(15):
        start = time.perf_counter()
        run()
        submitted = time.perf_counter()
        sth.synchronize()
        finished = time.perf_counter()
        submit.append((submitted - start) / launches * 1e6)
        total.append((finished - start) * 1e3)
    return statistics.median(submit), statistics.median(total)


if __name__ == "__main__":
    n = 1 << 16
    x = Buffer.from_list(dtype.float32, [1.0] * n)
    y = Buffer.from_list(dtype.float32, [2.0] * n)
    out = Buffer.from_list(dtype.float32, [0.0] * n)
    counts = [int(value) for value in sys.argv[1:]] or [8, 64, 256]
    print(f"{'launches':>9} {'mode':>9} {'host us / launch':>17} {'whole run ms':>13}")
    for launches in counts:
        for batched in (False, True):
            submit, total = measure(launches, batched, x, y, out, n)
            print(f"{launches:>9} {'batched' if batched else 'separate':>9} {submit:>17.1f} {total:>13.3f}")
    assert out.to_list()[:4] == [3.0] * 4
