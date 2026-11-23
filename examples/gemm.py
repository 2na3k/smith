import ctypes
import statistics
import sys
import time

import mlx.core as mx

import smith as sth
from smith import dtype, mlx_bridge, viz
from smith.compiler import Buffer, compile_msl, launch, synchronize

SHAPES = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]
TYPES = [(mx.float16, dtype.float16, "half"), (mx.float32, dtype.float32, "float")]


@sth.jit
def gemm(a_pointer, b_pointer, c_pointer,
         M: sth.constexpr, N: sth.constexpr, K: sth.constexpr,
         SG_M: sth.constexpr = 2, SG_N: sth.constexpr = 2, TM: sth.constexpr = 4, TN: sth.constexpr = 4,
         UNROLL: sth.constexpr = 1, SWIZZLE: sth.constexpr = 0):
    if M % (SG_M * TM * 8) or N % (SG_N * TN * 8) or K % (8 * UNROLL):
        raise ValueError("M, N, K must be multiples of the block, and K of 8 * UNROLL")
    if (M // (SG_M * TM * 8)) % (1 << SWIZZLE):
        raise ValueError("SWIZZLE needs the number of row blocks to be a multiple of 1 << SWIZZLE")
    thread = sth.arange(0, SG_M * SG_N * 32)
    lane = thread % 32
    quad = lane // 4
    fragment_row = (quad & 4) + (lane // 2) % 4
    fragment_column = (quad & 2) * 2 + (lane % 2) * 2
    simd = thread // 32
    block_column = sth.program_id(0) >> SWIZZLE
    block_row = (sth.program_id(1) << SWIZZLE) + (sth.program_id(0) & ((1 << SWIZZLE) - 1))
    row0 = (block_row * SG_M + simd // SG_N) * (TM * 8)
    col0 = (block_column * SG_N + simd % SG_N) * (TN * 8)

    def step(block, *state):
        updated = list(state)
        for u in range(UNROLL):
            column = (block * UNROLL + u) * 8
            a_tiles = [sth.matrix_load(a_pointer + ((row0 + i * 8) * K + column), K) for i in range(TM)]
            b_tiles = [sth.matrix_load(b_pointer + (column * N + col0 + j * 8), N) for j in range(TN)]
            for i in range(TM):
                for j in range(TN):
                    updated[i * TN + j] = sth.matrix_mma(a_tiles[i], b_tiles[j], updated[i * TN + j])
        return tuple(updated)

    results = sth.loop(K // (8 * UNROLL), tuple(sth.matrix_fill(0.0, dtype.float32) for _ in range(TM * TN)), step)
    for i in range(TM):
        for j in range(TN):
            low, high = sth.matrix_elements(results[i * TN + j])
            base = (row0 + i * 8 + fragment_row) * N + col0 + j * 8 + fragment_column
            sth.store(c_pointer + base, low)
            sth.store(c_pointer + (base + 1), high)


@sth.jit
def gemm_shared(a_pointer, b_pointer, c_pointer,
                M: sth.constexpr, N: sth.constexpr, K: sth.constexpr,
                SG_M: sth.constexpr = 2, SG_N: sth.constexpr = 2, TM: sth.constexpr = 4, TN: sth.constexpr = 4,
                BK: sth.constexpr = 32, PAD: sth.constexpr = 0, DOUBLE: sth.constexpr = False, SERPENTINE: sth.constexpr = False):
    element = a_pointer.buffer.dtype
    width = 128 // element.bits
    threads = SG_M * SG_N * 32
    block_m, block_n = SG_M * TM * 8, SG_N * TN * 8
    a_stride, b_stride = BK + PAD, block_n + PAD
    a_size, b_size = block_m * a_stride, BK * b_stride
    buffers = 2 if DOUBLE else 1
    if M % block_m or N % block_n or K % BK or BK % 8:
        raise ValueError("M, N, K must be multiples of the block and BK a multiple of 8")
    if (block_m * BK) % (width * threads) or (BK * block_n) % (width * threads) or BK % width or block_n % width or PAD % width:
        raise ValueError("the tiles must split evenly into 16 byte copies across the threads")
    if buffers * (a_size + b_size) * element.bits // 8 > 32768:
        raise ValueError("the staged tiles exceed 32 KB of threadgroup memory")
    a_tile = sth.shared(element, buffers * a_size)
    b_tile = sth.shared(element, buffers * b_size)
    thread = sth.arange(0, threads)
    lane = thread % 32
    quad = lane // 4
    fragment_row = (quad & 4) + (lane // 2) % 4
    fragment_column = (quad & 2) * 2 + (lane % 2) * 2
    simd = thread // 32
    tile_row = sth.program_id(1) * block_m
    tile_column = sth.program_id(0) * block_n
    local_row = (simd // SG_N) * (TM * 8)
    local_column = (simd % SG_N) * (TN * 8)
    blocks = K // BK

    def stage(block, which):
        first = block * BK
        for chunk in range(block_m * BK // (width * threads)):
            position = (chunk * threads + thread) * width
            row, column = position // BK, position % BK
            sth.copy_to_shared(a_tile + (which * a_size + row * a_stride + column),
                               a_pointer + ((tile_row + row) * K + first + column), width)
        for chunk in range(BK * block_n // (width * threads)):
            position = (chunk * threads + thread) * width
            row, column = position // block_n, position % block_n
            sth.copy_to_shared(b_tile + (which * b_size + row * b_stride + column),
                               b_pointer + ((first + row) * N + tile_column + column), width)

    def multiply(which, state):
        updated = list(state)
        for part in range(BK // 8):
            a_tiles = [sth.matrix_load(a_tile + (which * a_size + (local_row + i * 8) * a_stride + part * 8), a_stride) for i in range(TM)]
            b_tiles = [sth.matrix_load(b_tile + (which * b_size + part * 8 * b_stride + local_column + j * 8), b_stride) for j in range(TN)]
            for i in range(TM):
                for step_j in range(TN):
                    j = TN - 1 - step_j if SERPENTINE and i % 2 else step_j
                    updated[i * TN + j] = sth.matrix_mma(a_tiles[i], b_tiles[j], updated[i * TN + j])
        return tuple(updated)

    zero = tuple(sth.matrix_fill(0.0, dtype.float32) for _ in range(TM * TN))
    if DOUBLE:
        stage(0, 0)
        sth.barrier()

        def pipelined(block, *state):
            stage(block + 1, (block + 1) % 2)
            updated = multiply(block % 2, state)
            sth.barrier()
            return updated

        results = multiply((blocks - 1) % 2, sth.loop(blocks - 1, zero, pipelined))
    else:
        def plain(block, *state):
            stage(block, 0)
            sth.barrier()
            updated = multiply(0, state)
            sth.barrier()
            return updated

        results = sth.loop(blocks, zero, plain)
    for i in range(TM):
        for j in range(TN):
            low, high = sth.matrix_elements(results[i * TN + j])
            base = (tile_row + local_row + i * 8 + fragment_row) * N + tile_column + local_column + j * 8 + fragment_column
            sth.store(c_pointer + base, low)
            sth.store(c_pointer + (base + 1), high)


def gemm_shared_grid(m, n):
    return lambda meta: (n // (meta["SG_N"] * meta["TN"] * 8), m // (meta["SG_M"] * meta["TM"] * 8))


CONFIGS = sth.configs_from_grid({"SG_M": [1, 2, 4], "SG_N": [1, 2, 4], "TM": [2, 4], "TN": [2, 4], "UNROLL": [1, 2, 4]})
tuned_gemm = sth.autotune(CONFIGS, key=["M", "N", "K"], warmup=10, repeat=20, fast_math=False)(gemm)
SHARED_CONFIGS = sth.configs_from_grid({"SG_M": [1, 2], "SG_N": [1, 2, 4], "TM": [2, 4], "TN": [2, 4], "BK": [16, 32], "DOUBLE": [False, True]})
tuned_gemm_shared = sth.autotune(SHARED_CONFIGS, key=["M", "N", "K"], warmup=10, repeat=20, fast_math=False)(gemm_shared)


def gemm_grid(m, n):
    swizzle = lambda meta: meta.get("SWIZZLE", 0)
    return lambda meta: ((n // (meta["SG_N"] * meta["TN"] * 8)) << swizzle(meta), (m // (meta["SG_M"] * meta["TM"] * 8)) >> swizzle(meta))


METAL_DIRECT = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
#define UNROLL _Pragma("clang loop unroll(full)")
kernel void metal_direct(device const T* a [[buffer(0)]], device const T* b [[buffer(1)]], device T* c [[buffer(2)]],
                         constant uint& N [[buffer(3)]], constant uint& K [[buffer(4)]],
                         uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint row0 = (group.y * 2 + simd / 2) * 32;
    const uint col0 = (group.x * 2 + simd % 2) * 32;
    simdgroup_matrix<float, 8, 8> acc[4][4];
    UNROLL
    for (uint i = 0; i < 4; i++)
        UNROLL
        for (uint j = 0; j < 4; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < K; k += 8) {
        simdgroup_matrix<T, 8, 8> left[4], right[4];
        UNROLL
        for (uint i = 0; i < 4; i++) simdgroup_load(left[i], a + (row0 + i * 8) * K + k, K);
        UNROLL
        for (uint j = 0; j < 4; j++) simdgroup_load(right[j], b + k * N + col0 + j * 8, N);
        UNROLL
        for (uint i = 0; i < 4; i++)
            UNROLL
            for (uint j = 0; j < 4; j++)
                simdgroup_multiply_accumulate(acc[i][j], left[i], right[j], acc[i][j]);
    }
    UNROLL
    for (uint i = 0; i < 4; i++)
        UNROLL
        for (uint j = 0; j < 4; j++) {
            simdgroup_matrix<T, 8, 8> out;
            out.thread_elements()[0] = T(acc[i][j].thread_elements()[0]);
            out.thread_elements()[1] = T(acc[i][j].thread_elements()[1]);
            simdgroup_store(out, c + (row0 + i * 8) * N + col0 + j * 8, N);
        }
}
"""

METAL_TILED = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
#define UNROLL _Pragma("clang loop unroll(full)")
constant constexpr uint BM = 64, BN = 64, BK = 16;
kernel void metal_tiled(device const T* a [[buffer(0)]], device const T* b [[buffer(1)]], device T* c [[buffer(2)]],
                        constant uint& N [[buffer(3)]], constant uint& K [[buffer(4)]],
                        uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                        uint thread_id [[thread_index_in_threadgroup]]) {
    threadgroup T tile_a[BM * BK];
    threadgroup T tile_b[BK * BN];
    const uint row_base = group.y * BM, col_base = group.x * BN;
    const uint sm = (simd / 2) * 32, sn = (simd % 2) * 32;
    simdgroup_matrix<float, 8, 8> acc[4][4];
    UNROLL
    for (uint i = 0; i < 4; i++)
        UNROLL
        for (uint j = 0; j < 4; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < K; k += BK) {
        for (uint e = thread_id; e < BM * BK; e += 128) tile_a[e] = a[(row_base + e / BK) * K + k + e % BK];
        for (uint e = thread_id; e < BK * BN; e += 128) tile_b[e] = b[(k + e / BN) * N + col_base + e % BN];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<T, 8, 8> left[4], right[4];
            UNROLL
            for (uint i = 0; i < 4; i++) simdgroup_load(left[i], tile_a + (sm + i * 8) * BK + kk, BK);
            UNROLL
            for (uint j = 0; j < 4; j++) simdgroup_load(right[j], tile_b + kk * BN + sn + j * 8, BN);
            UNROLL
            for (uint i = 0; i < 4; i++)
                UNROLL
                for (uint j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(acc[i][j], left[i], right[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    UNROLL
    for (uint i = 0; i < 4; i++)
        UNROLL
        for (uint j = 0; j < 4; j++) {
            simdgroup_matrix<T, 8, 8> out;
            out.thread_elements()[0] = T(acc[i][j].thread_elements()[0]);
            out.thread_elements()[1] = T(acc[i][j].thread_elements()[1]);
            simdgroup_store(out, c + (row_base + sm + i * 8) * N + col_base + sn + j * 8, N);
        }
}
"""


class MetalGemm:
    def __init__(self, source, name, ctype, tile_m, tile_n, threads):
        self.kernel = compile_msl(f"typedef {ctype} T;\n" + source, name, fast_math=False)
        self.tile_m, self.tile_n, self.threads = tile_m, tile_n, threads

    def __call__(self, a, b, c, m, n, k, wait=False):
        grid = (n // self.tile_n * self.threads, m // self.tile_m, 1)
        return launch(self.kernel, [a, b, c, ctypes.c_uint32(n), ctypes.c_uint32(k)], grid, (self.threads, 1, 1), wait=wait)


def median_time(function, warmup=15, repeat=25):
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def pipelined_time(make, count=8):
    return median_time(lambda: mx.eval([make() for _ in range(count)]), 6, 15) / count


def finished(event, wait):
    return event.wait() if wait else event


def native_gpu_time(function, warmup=10, repeat=20):
    for _ in range(warmup):
        function(wait=True)
    return statistics.median(function(wait=True).gpu_time for _ in range(repeat))


def tune(tuner, grid, a_buffer, b_buffer, c_buffer, constants):
    with viz.phase("tuning"):
        tuner[grid](a_buffer, b_buffer, c_buffer, **constants)
    return tuner.best_config.values


def run(m, n, k, mlx_type, smith_type, ctype):
    mx.random.seed(0)
    a = mx.random.normal((m, k)).astype(mlx_type)
    b = mx.random.normal((k, n)).astype(mlx_type)
    mx.eval(a, b)
    flops = 2 * m * n * k
    constants = dict(M=m, N=n, K=k)
    a_buffer, b_buffer, c_buffer = mlx_bridge.buffer_from_mlx(a), mlx_bridge.buffer_from_mlx(b), Buffer(smith_type, m * n)
    direct_grid, shared_grid = gemm_grid(m, n), gemm_shared_grid(m, n)
    direct_best = tune(tuned_gemm, direct_grid, a_buffer, b_buffer, c_buffer, constants)
    shared_best = tune(tuned_gemm_shared, shared_grid, a_buffer, b_buffer, c_buffer, constants)

    template = mx.zeros((m, n), dtype=mlx_type)
    direct, shared = mlx_bridge.kernel(gemm), mlx_bridge.kernel(gemm_shared)
    contenders = {
        "mlx matmul": lambda: a @ b,
        "smith direct tuned": lambda: direct[direct_grid](a, b, template, **constants, **direct_best),
        "smith shared tuned": lambda: shared[shared_grid](a, b, template, **constants, **shared_best),
    }
    expected = a.astype(mx.float32) @ b.astype(mx.float32)
    scale = max(1.0, mx.max(mx.abs(expected)).item())
    errors = {name: mx.max(mx.abs(function().astype(mx.float32) - expected)).item() / scale for name, function in contenders.items()}

    print(f"\n{mlx_type} M={m} N={n} K={k}")
    print(f"  tuned direct {direct_best}\n  tuned shared {shared_best}")
    print(f"  {'':22s} {'one call':>10s} {'pipelined':>10s} {'TFLOP/s':>8s} {'vs mlx':>7s}  rel error")
    singles, pipelines = {}, {}
    with viz.phase("benchmark"):
        for name, function in contenders.items():
            singles[name] = median_time(lambda: mx.eval(function()))
            pipelines[name] = pipelined_time(function)
            ratio = pipelines[name] / pipelines["mlx matmul"]
            print(f"  {name:22s} {singles[name] * 1e3:8.3f}ms {pipelines[name] * 1e3:8.3f}ms {flops / pipelines[name] / 1e12:8.2f} {ratio:6.3f}x  {errors[name]:.1e}")

        direct_metal = MetalGemm(METAL_DIRECT, "metal_direct", ctype, 64, 64, 128)
        tiled_metal = MetalGemm(METAL_TILED, "metal_tiled", ctype, 64, 64, 128)
        c_metal = Buffer(smith_type, m * n)
        natives = {
            "metal direct (raw MSL)": lambda wait=False: direct_metal(a_buffer, b_buffer, c_metal, m, n, k, wait),
            "metal tiled (raw MSL)": lambda wait=False: tiled_metal(a_buffer, b_buffer, c_metal, m, n, k, wait),
            "smith direct native": lambda wait=False: finished(gemm[direct_grid](a_buffer, b_buffer, c_buffer, **constants, **direct_best), wait),
            "smith shared native": lambda wait=False: finished(gemm_shared[shared_grid](a_buffer, b_buffer, c_buffer, **constants, **shared_best), wait),
        }
        print("  native launches, GPU time only:")
        for name, function in natives.items():
            seconds = native_gpu_time(function)
            print(f"  {name:22s} {seconds * 1e3:8.3f}ms {'':10s} {flops / seconds / 1e12:8.2f}")
    return direct_best, shared_best


def profile_once(m, n, k, mlx_type, smith_type, ctype, direct_best, shared_best):
    a = mx.random.normal((m, k)).astype(mlx_type)
    b = mx.random.normal((k, n)).astype(mlx_type)
    mx.eval(a, b)
    a_buffer, b_buffer = mlx_bridge.buffer_from_mlx(a), mlx_bridge.buffer_from_mlx(b)
    c_buffer, c_metal = Buffer(smith_type, m * n), Buffer(smith_type, m * n)
    direct_metal = MetalGemm(METAL_DIRECT, "metal_direct", ctype, 64, 64, 128)
    tiled_metal = MetalGemm(METAL_TILED, "metal_tiled", ctype, 64, 64, 128)
    for _ in range(3):
        with viz.span("mlx", "mx.matmul"):
            mx.eval(a @ b)
        direct_metal(a_buffer, b_buffer, c_metal, m, n, k, wait=True)
        tiled_metal(a_buffer, b_buffer, c_metal, m, n, k, wait=True)
        gemm[gemm_grid(m, n)](a_buffer, b_buffer, c_buffer, M=m, N=n, K=k, **direct_best).wait()
        gemm_shared[gemm_shared_grid(m, n)](a_buffer, b_buffer, c_buffer, M=m, N=n, K=k, **shared_best).wait()


if __name__ == "__main__":
    shapes = SHAPES if len(sys.argv) == 1 else [tuple(int(value) for value in sys.argv[1:4])]
    bests = {}
    for shape in shapes:
        for mlx_type, smith_type, ctype in TYPES:
            bests[(shape, ctype)] = run(*shape, mlx_type, smith_type, ctype)
    mlx_type, smith_type, ctype = TYPES[0]
    profile_once(*shapes[-1], mlx_type, smith_type, ctype, *bests[(shapes[-1], ctype)])
