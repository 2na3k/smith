import statistics
import sys
import time

import mlx.core as mx

from smith import dtype, mlx_bridge
from smith.kernels.attention import (attention_grid, attention_matrix_grid, flash_attention, flash_attention_matrix,
                                     tuned_flash_attention, tuned_flash_attention_matrix)
from smith.compiler import Buffer

SHAPES = [(8, 512, 64), (8, 1024, 64), (8, 2048, 64)]
TYPES = [(mx.float32, dtype.float32), (mx.float16, dtype.float16)]


def median_time(function, warmup=15, repeat=25):
    for _ in range(warmup):
        mx.eval(function())
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        mx.eval(function())
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def plain_attention(q, k, v, scale, causal):
    scores = (q @ k.transpose(0, 2, 1)) * scale
    if causal:
        length = q.shape[1]
        mask = mx.triu(mx.full((length, length), -1e30, dtype=scores.dtype), k=1)
        scores = scores + mask
    return mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype) @ v


def fused_attention(q, k, v, scale, causal):
    batch_heads, length, head_dim = q.shape
    shape = (1, batch_heads, length, head_dim)
    mask = "causal" if causal else None
    result = mx.fast.scaled_dot_product_attention(q.reshape(shape), k.reshape(shape), v.reshape(shape), scale=scale, mask=mask)
    return result.reshape(q.shape)


def tune(tuner, grid, arguments, constants):
    tuner[grid](*arguments, **constants)
    return tuner.best_config.values, len([entry for entry in tuner.results if entry["seconds"] is not None])


def run(heads, length, head_dim, mlx_type, smith_type, causal):
    mx.random.seed(0)
    shape = (heads, length, head_dim)
    q, k, v = (mx.random.normal(shape).astype(mlx_type) for _ in range(3))
    mx.eval(q, k, v)
    scale = head_dim ** -0.5

    buffers = [mlx_bridge.buffer_from_mlx(array) for array in (q, k, v)]
    arguments = (*buffers, Buffer(smith_type, q.size), scale)
    constants = dict(SEQ_Q=length, SEQ_KV=length, HEAD_DIM=head_dim, CAUSAL=causal)
    scalar_best, scalar_count = tune(tuned_flash_attention, attention_grid(heads, length), arguments, constants)
    matrix_best, matrix_count = tune(tuned_flash_attention_matrix, attention_matrix_grid(heads, length), arguments, constants)

    template = mx.zeros(shape, dtype=mlx_type)
    scalar = mlx_bridge.kernel(flash_attention)
    matrix = mlx_bridge.kernel(flash_attention_matrix)
    run_scalar = lambda: scalar[attention_grid(heads, length)](q, k, v, template, scale, **constants, **scalar_best)
    run_matrix_default = lambda: matrix[attention_matrix_grid(heads, length)](q, k, v, template, scale, **constants)
    run_matrix = lambda: matrix[attention_matrix_grid(heads, length)](q, k, v, template, scale, **constants, **matrix_best)

    reference = fused_attention(q, k, v, scale, causal).astype(mx.float32)
    errors = [mx.max(mx.abs(function().astype(mx.float32) - reference)).item() for function in (run_scalar, run_matrix)]
    flops = 4 * heads * length * length * head_dim * (0.5 if causal else 1.0)

    timings = {
        "mlx plain ops": median_time(lambda: plain_attention(q, k, v, scale, causal)),
        "mlx fused sdpa": median_time(lambda: fused_attention(q, k, v, scale, causal)),
        "smith scalar": median_time(run_scalar),
        "smith matrix default": median_time(run_matrix_default),
        "smith matrix tuned": median_time(run_matrix),
    }
    print(f"\n{mlx_type} heads={heads} seq={length} dim={head_dim} causal={causal}  max error vs sdpa: "
          f"scalar {errors[0]:.1e}, matrix {errors[1]:.1e}")
    print(f"  tuned scalar {scalar_best} ({scalar_count} configs)\n  tuned matrix {matrix_best} ({matrix_count} configs)")
    for name, seconds in timings.items():
        print(f"  {name:22s} {seconds * 1e3:9.3f} ms  {flops / seconds / 1e12:7.3f} TFLOP/s")
    print(f"  matrix tuned vs fused sdpa: {timings['smith matrix tuned'] / timings['mlx fused sdpa']:.2f}x time   "
          f"matrix vs scalar: {timings['smith scalar'] / timings['smith matrix tuned']:.2f}x faster   "
          f"autotune gain: {timings['smith matrix default'] / timings['smith matrix tuned']:.2f}x")
    return timings


if __name__ == "__main__":
    shapes = SHAPES if len(sys.argv) == 1 else [tuple(int(x) for x in sys.argv[1:4])]
    for heads, length, head_dim in shapes:
        for mlx_type, smith_type in TYPES:
            for causal in (False, True):
                run(heads, length, head_dim, mlx_type, smith_type, causal)
