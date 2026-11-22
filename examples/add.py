import smith as sth


@sth.jit
def add(x_pointer, y_pointer, out_pointer, n, BLOCK: sth.constexpr = 4):
    pid = sth.program_id(0)
    offset = pid * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    x = sth.load(x_pointer + offset, mask=mask)
    y = sth.load(y_pointer + offset, mask=mask)
    sth.store(out_pointer + offset, x + y, mask=mask)


if __name__ == "__main__":
    n = 10
    x = list(range(n))
    y = [10 * i for i in range(n)]
    out = [0] * n

    grid = lambda meta: (sth.cdiv(n, meta["BLOCK"]),)
    add[grid](x, y, out, n, BLOCK=4)

    print("x   =", x)
    print("y   =", y)
    print("out =", out)
