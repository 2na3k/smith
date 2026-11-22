import smith as sth
from smith import dtype
from smith.autotune import autotune, configs_from_grid
from smith.compiler import Buffer


@autotune(configs_from_grid({"BLOCK": [32, 64, 128, 256, 512]}), key=("n",), warmup=2, repeat=5)
@sth.jit
def add(x_pointer, y_pointer, out_pointer, n, BLOCK: sth.constexpr = 32):
    pid = sth.program_id(0)
    offset = pid * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    x = sth.load(x_pointer + offset, mask=mask)
    y = sth.load(y_pointer + offset, mask=mask)
    sth.store(out_pointer + offset, x + y, mask=mask)


if __name__ == "__main__":
    n = 1 << 20
    x = Buffer.from_list(dtype.float32, [float(i) for i in range(n)])
    y = Buffer.from_list(dtype.float32, [2.0 * i for i in range(n)])
    out = Buffer.from_list(dtype.float32, [0.0] * n)

    grid = lambda meta: (sth.cdiv(n, meta["BLOCK"]),)
    add[grid](x, y, out, n)

    print(f"best config: {add.best_config}")
    for result in add.results:
        if result["seconds"] is None:
            print(f"  {result['config']}  skipped: {result['error']}")
        else:
            print(f"  {result['config']}  median gpu time {result['seconds'] * 1e6:10.1f} us")

    values = out.to_list()
    print("correct:", all(values[i] == 3.0 * i for i in range(n)))
