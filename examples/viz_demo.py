import os
import random
import tempfile

os.environ.setdefault("SMITH_CACHE_DIR", tempfile.mkdtemp(prefix="smith-viz-"))

import smith as sth
from smith import dtype, viz
from smith.compiler import Buffer
from smith.kernels.attention import attention_matrix_grid, tuned_flash_attention_matrix

viz.enable()


@sth.jit
def add(x_pointer, y_pointer, out_pointer, n, BLOCK: sth.constexpr = 256):
    offset = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    x = sth.load(x_pointer + offset, mask=mask)
    y = sth.load(y_pointer + offset, mask=mask)
    sth.store(out_pointer + offset, x + y, mask=mask)


@sth.jit
def row_sum(x_pointer, out_pointer, rows, columns, BLOCK: sth.constexpr = 64):
    row = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    mask = row < rows
    (total,) = sth.loop(columns, (0.0,), lambda column, acc: (acc + sth.load(x_pointer + row * columns + column, mask=mask),))
    sth.store(out_pointer + row, total, mask=mask)


if __name__ == "__main__":
    generator = random.Random(0)
    n = 1 << 16
    x = Buffer.from_list(dtype.float32, [generator.random() for _ in range(n)])
    y = Buffer.from_list(dtype.float32, [generator.random() for _ in range(n)])
    out = Buffer.from_list(dtype.float32, [0.0] * n)
    for _ in range(8):
        add[lambda meta: (sth.cdiv(n, meta["BLOCK"]),)](x, y, out, n, BLOCK=256)

    with sth.batch():
        for _ in range(8):
            add[lambda meta: (sth.cdiv(n, meta["BLOCK"]),)](x, y, out, n, BLOCK=256)
    sth.synchronize()

    rows, columns = 4096, 64
    table = Buffer.from_list(dtype.float32, [generator.random() for _ in range(rows * columns)])
    sums = Buffer.from_list(dtype.float32, [0.0] * rows)
    for _ in range(5):
        row_sum[lambda meta: (sth.cdiv(rows, meta["BLOCK"]),)](table, sums, rows, columns, BLOCK=64)

    heads, length, head_dim = 4, 256, 64
    q, k, v = (Buffer.from_list(dtype.float32, [generator.gauss(0, 1) for _ in range(heads * length * head_dim)]) for _ in range(3))
    attention = Buffer.from_list(dtype.float32, [0.0] * (heads * length * head_dim))
    tuned_flash_attention_matrix[attention_matrix_grid(heads, length)](
        q, k, v, attention, head_dim ** -0.5, SEQ_Q=length, SEQ_KV=length, HEAD_DIM=head_dim, CAUSAL=False)
    for _ in range(5):
        tuned_flash_attention_matrix[attention_matrix_grid(heads, length)](
            q, k, v, attention, head_dim ** -0.5, SEQ_Q=length, SEQ_KV=length, HEAD_DIM=head_dim, CAUSAL=False)
    from smith.kernels.attention import flash_attention_matrix
    for _ in range(4):
        flash_attention_matrix[attention_matrix_grid(heads, length)](
            q, k, v, attention, head_dim ** -0.5, SEQ_Q=length, SEQ_KV=length, HEAD_DIM=head_dim, CAUSAL=False,
            SIMDGROUPS=8, ROW_BLOCKS=2)
    sth.synchronize()
    viz.serve()
