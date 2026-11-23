import importlib.util
import os
import statistics
import sys
import time

import mlx.core as mx

import smith as sth
from smith import dtype, mlx_bridge
from smith.compiler import Buffer

NEGATIVE_LARGE = -1e30


@sth.jit
def sparse_mla(q_pointer, q_pe_pointer, kv_pointer, k_pe_pointer, index_pointer, out_pointer, kv_rows, scale,
               HEADS: sth.constexpr, SEQ_Q: sth.constexpr, RANK: sth.constexpr, ROPE: sth.constexpr, TOPK: sth.constexpr,
               BLOCK_M: sth.constexpr = 4, BLOCK_N: sth.constexpr = 1, SPLIT: sth.constexpr = 32):
    if TOPK % BLOCK_N:
        raise ValueError("TOPK must be a multiple of BLOCK_N")
    if SPLIT < 1 or SPLIT > 32 or SPLIT & (SPLIT - 1) or RANK % SPLIT or ROPE % SPLIT:
        raise ValueError("SPLIT must be a power of two up to 32 that divides RANK and ROPE")
    latent_dims = RANK // SPLIT
    rope_dims = ROPE // SPLIT
    thread = sth.arange(0, BLOCK_M * SPLIT)
    row = sth.program_id(0) * BLOCK_M + thread // SPLIT
    row_mask = row < HEADS * SEQ_Q
    query = row // HEADS
    head = row % HEADS
    lane = thread % SPLIT
    output_row = (head * SEQ_Q + query) * RANK
    rope_row = (head * SEQ_Q + query) * ROPE
    queries = [sth.cast(sth.load(q_pointer + (output_row + lane + d * SPLIT), mask=row_mask), dtype.float32) * scale
               for d in range(latent_dims)]
    rope_queries = [sth.cast(sth.load(q_pe_pointer + (rope_row + lane + d * SPLIT), mask=row_mask), dtype.float32) * scale
                    for d in range(rope_dims)]

    def step(block, running_max, running_sum, *accumulators):
        picks, valids, scores = [], [], []
        for k in range(BLOCK_N):
            pick = sth.load(index_pointer + (query * TOPK + block * BLOCK_N + k), mask=row_mask, other=-1)
            valid = (pick >= 0) & (pick < kv_rows)
            safe = sth.where(valid, pick, 0)
            score = 0.0
            for d in range(latent_dims):
                score = score + queries[d] * sth.cast(sth.load(kv_pointer + (safe * RANK + lane + d * SPLIT)), dtype.float32)
            for d in range(rope_dims):
                score = score + rope_queries[d] * sth.cast(sth.load(k_pe_pointer + (safe * ROPE + lane + d * SPLIT)), dtype.float32)
            distance = 1
            while distance < SPLIT:
                score = score + sth.shuffle_xor(score, distance)
                distance *= 2
            picks.append(safe)
            valids.append(valid)
            scores.append(sth.where(valid, score, NEGATIVE_LARGE))
        new_max = running_max
        for score in scores:
            new_max = sth.maximum(new_max, score)
        correction = sth.exp(running_max - new_max)
        probabilities = [sth.where(valids[k], sth.exp(scores[k] - new_max), 0.0) for k in range(BLOCK_N)]
        new_sum = running_sum * correction
        for probability in probabilities:
            new_sum = new_sum + probability
        updated = []
        for d in range(latent_dims):
            value = accumulators[d] * correction
            for k in range(BLOCK_N):
                value = value + probabilities[k] * sth.cast(sth.load(kv_pointer + (picks[k] * RANK + lane + d * SPLIT)), dtype.float32)
            updated.append(value)
        return (new_max, new_sum, *updated)

    results = sth.loop(TOPK // BLOCK_N, (NEGATIVE_LARGE, 0.0) + (0.0,) * latent_dims, step)
    total = results[1]
    for d in range(latent_dims):
        sth.store(out_pointer + (output_row + lane + d * SPLIT), sth.where(total > 0.0, results[2 + d] / total, 0.0), mask=row_mask)


@sth.jit
def sparse_mla_matrix(q_pointer, q_pe_pointer, kv_pointer, k_pe_pointer, index_pointer, out_pointer, kv_rows, scale,
                      HEADS: sth.constexpr, SEQ_Q: sth.constexpr, RANK: sth.constexpr, ROPE: sth.constexpr, TOPK: sth.constexpr,
                      WN: sth.constexpr = 4, QK_UNROLL: sth.constexpr = 1, PV_UNROLL: sth.constexpr = 1, FLOAT_P: sth.constexpr = False):
    if HEADS % 16 or ROPE % 8 or RANK % (8 * WN):
        raise ValueError("HEADS must be a multiple of 16, ROPE of 8 and RANK of 8 * WN")
    if (RANK // 8) % QK_UNROLL or (ROPE // 8) % QK_UNROLL or (WN * 4) % PV_UNROLL:
        raise ValueError("the unroll factors must divide the step counts")
    block_keys = 32 * WN
    if (16 * block_keys * (4 if FLOAT_P else kv_pointer.buffer.dtype.size) + 16 * block_keys) > 32768:
        raise ValueError("the shared probability tile does not fit in 32 KB of threadgroup memory")
    latent_width = RANK // WN
    tiles = latent_width // 8
    thread = sth.arange(0, 32 * WN)
    group = thread // 32
    lane = thread % 32
    quad = lane // 4
    fragment_row = (quad & 4) + (lane // 2) % 4
    fragment_column = (quad & 2) * 2 + (lane % 2) * 2
    head_base = sth.program_id(0) * 16
    query = sth.program_id(1)
    query_origin = (head_base * SEQ_Q + query) * RANK
    rope_origin = (head_base * SEQ_Q + query) * ROPE
    index_base = query * TOPK
    element_type = dtype.float32 if FLOAT_P else kv_pointer.buffer.dtype
    probabilities_buffer = sth.shared(element_type, 16 * block_keys)
    maximum_buffer = sth.shared(dtype.float32, 2 * block_keys)
    sum_buffer = sth.shared(dtype.float32, 2 * block_keys)

    def block_step(block, *state):
        running_max, running_sum, outputs = list(state[0:2]), list(state[2:4]), list(state[4:])
        first = block * block_keys
        own = first + group * 32
        key_rows = []
        for n in range(4):
            pair = []
            for e in range(2):
                position = own + (n * 8 + e) + fragment_column
                pick = sth.load(index_pointer + (index_base + position), mask=position < TOPK, other=-1)
                valid = (pick >= 0) & (pick < kv_rows)
                pair.append((valid, sth.where(valid, pick, 0)))
            key_rows.append(pair)

        def accumulate(steps, query_pointer, origin, head_stride, cache_pointer, cache_stride, scores):
            def body(step, *current):
                updated = list(current)
                for u in range(QK_UNROLL):
                    column = (step * QK_UNROLL + u) * 8
                    key_tiles = [sth.matrix_make(sth.load(cache_pointer + (key_rows[n][0][1] * cache_stride + column + fragment_row)),
                                                 sth.load(cache_pointer + (key_rows[n][1][1] * cache_stride + column + fragment_row)))
                                 for n in range(4)]
                    for i in range(2):
                        query_tile = sth.matrix_load(query_pointer + (origin + (i * 8 * head_stride + column)), head_stride)
                        for n in range(4):
                            updated[i * 4 + n] = sth.matrix_mma(query_tile, key_tiles[n], updated[i * 4 + n])
                return tuple(updated)

            return sth.loop(steps // QK_UNROLL, scores, body)

        scores = tuple(sth.matrix_fill(0.0, dtype.float32) for _ in range(8))
        scores = accumulate(RANK // 8, q_pointer, query_origin, SEQ_Q * RANK, kv_pointer, RANK, scores)
        scores = accumulate(ROPE // 8, q_pe_pointer, rope_origin, SEQ_Q * ROPE, k_pe_pointer, ROPE, scores)

        values = [[], []]
        for i in range(2):
            for n in range(4):
                low, high = sth.matrix_elements(scores[i * 4 + n])
                values[i].append(sth.where(key_rows[n][0][0], low * scale, NEGATIVE_LARGE))
                values[i].append(sth.where(key_rows[n][1][0], high * scale, NEGATIVE_LARGE))
        row_max = []
        for i in range(2):
            top = values[i][0]
            for value in values[i][1:]:
                top = sth.maximum(top, value)
            top = sth.maximum(top, sth.shuffle_xor(top, 1))
            top = sth.maximum(top, sth.shuffle_xor(top, 8))
            row_max.append(top)
            sth.store(maximum_buffer + (i * block_keys + group * 32 + lane), top)
        sth.barrier()
        new_max, factor = [], []
        for i in range(2):
            top = running_max[i]
            for w in range(WN):
                top = sth.maximum(top, sth.load(maximum_buffer + (i * block_keys + w * 32 + lane)))
            new_max.append(top)
            factor.append(sth.exp(running_max[i] - top))
        for i in range(2):
            partial = 0.0
            for n in range(4):
                for e in range(2):
                    probability = sth.where(key_rows[n][e][0], sth.exp(values[i][n * 2 + e] - new_max[i]), 0.0)
                    partial = partial + probability
                    sth.store(probabilities_buffer + ((i * 8 + fragment_row) * block_keys + group * 32 + (n * 8 + e) + fragment_column), probability)
            sth.store(sum_buffer + (i * block_keys + group * 32 + lane), partial)
        sth.barrier()
        new_sum = []
        for i in range(2):
            total = running_sum[i] * factor[i]
            for w in range(WN):
                total = total + sth.load(sum_buffer + (i * block_keys + w * 32 + lane))
            new_sum.append(total)
        scaled = []
        for i in range(2):
            for c in range(tiles):
                low, high = sth.matrix_elements(outputs[i * tiles + c])
                scaled.append(sth.matrix_make(low * factor[i], high * factor[i]))

        def accumulate_values(step, *current):
            updated = list(current)
            for u in range(PV_UNROLL):
                key_block = step * PV_UNROLL + u
                position = first + key_block * 8 + fragment_row
                pick = sth.load(index_pointer + (index_base + position), mask=position < TOPK, other=-1)
                valid = (pick >= 0) & (pick < kv_rows)
                row_offset = sth.where(valid, pick, 0) * RANK + group * latent_width + fragment_column
                probability_tiles = [sth.matrix_load(probabilities_buffer + (i * 8 * block_keys + key_block * 8), block_keys) for i in range(2)]
                for c in range(tiles):
                    value_tile = sth.matrix_make(sth.cast(sth.load(kv_pointer + (row_offset + c * 8)), element_type),
                                                 sth.cast(sth.load(kv_pointer + (row_offset + (c * 8 + 1))), element_type))
                    for i in range(2):
                        updated[i * tiles + c] = sth.matrix_mma(probability_tiles[i], value_tile, updated[i * tiles + c])
            return tuple(updated)

        outputs = sth.loop(WN * 4 // PV_UNROLL, tuple(scaled), accumulate_values)
        return (*new_max, *new_sum, *outputs)

    initial = (NEGATIVE_LARGE, NEGATIVE_LARGE, 0.0, 0.0) + tuple(sth.matrix_fill(0.0, dtype.float32) for _ in range(2 * tiles))
    results = sth.loop(-(-TOPK // block_keys), initial, block_step)
    for i in range(2):
        total = results[2 + i] + sth.shuffle_xor(results[2 + i], 1)
        total = total + sth.shuffle_xor(total, 8)
        for c in range(tiles):
            low, high = sth.matrix_elements(results[4 + i * tiles + c])
            base = ((head_base + (i * 8) + fragment_row) * SEQ_Q + query) * RANK + group * latent_width + (c * 8) + fragment_column
            sth.store(out_pointer + base, sth.where(total > 0.0, low / total, 0.0))
            sth.store(out_pointer + (base + 1), sth.where(total > 0.0, high / total, 0.0))


CONFIGS = sth.configs_from_grid({"BLOCK_M": [1, 2, 4, 8, 16], "BLOCK_N": [1, 2, 4], "SPLIT": [16, 32]})
tuned_sparse_mla = sth.autotune(CONFIGS, key=["HEADS", "SEQ_Q", "RANK", "ROPE", "TOPK"], warmup=10, repeat=20, fast_math=False)(sparse_mla)
MATRIX_CONFIGS = sth.configs_from_grid({"WN": [4, 8, 16], "QK_UNROLL": [1, 2, 4], "PV_UNROLL": [1, 2, 4]})
tuned_sparse_mla_matrix = sth.autotune(MATRIX_CONFIGS, key=["HEADS", "SEQ_Q", "RANK", "ROPE", "TOPK"], warmup=10, repeat=20,
                                       fast_math=False)(sparse_mla_matrix)


def sparse_mla_grid(heads, seq_q):
    return lambda meta: (-(-(heads * seq_q) // meta["BLOCK_M"]), 1)


def sparse_mla_matrix_grid(heads, seq_q):
    return lambda meta: (heads // 16, seq_q)


def reference(q, q_pe, kv, k_pe, indices, scale):
    heads, length, rank = q.shape
    rows = kv.shape[0]
    valid = (indices >= 0) & (indices < rows)
    safe = mx.where(valid, indices, 0)
    gathered_kv = kv[safe].astype(mx.float32)
    gathered_pe = k_pe[safe].astype(mx.float32)
    scores = mx.matmul(q.astype(mx.float32).transpose(1, 0, 2), gathered_kv.transpose(0, 2, 1))
    scores = scores + mx.matmul(q_pe.astype(mx.float32).transpose(1, 0, 2), gathered_pe.transpose(0, 2, 1))
    scores = mx.where(valid[:, None, :], scores * scale, NEGATIVE_LARGE)
    weights = mx.softmax(scores, axis=-1)
    weights = mx.where(valid[:, None, :], weights, 0.0)
    return mx.matmul(weights, gathered_kv).transpose(1, 0, 2).astype(q.dtype)


def load_mlx_lm_kernel():
    try:
        from mlx_lm.kernels.sparse_mla import sparse_mla_attention
        return sparse_mla_attention
    except ImportError:
        pass
    path = os.environ.get("SPARSE_MLA_REFERENCE")
    if path is None:
        return None
    spec = importlib.util.spec_from_file_location("sparse_mla_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sparse_mla_attention


def median_time(function, warmup=15, repeat=25):
    for _ in range(warmup):
        mx.eval(function())
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        mx.eval(function())
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def make_inputs(heads, length, rank, rope, rows, topk, mlx_type):
    mx.random.seed(0)
    q = mx.random.normal((heads, length, rank)).astype(mlx_type)
    q_pe = mx.random.normal((heads, length, rope)).astype(mlx_type)
    kv = mx.random.normal((rows, rank)).astype(mlx_type)
    k_pe = mx.random.normal((rows, rope)).astype(mlx_type)
    order = mx.argsort(mx.random.uniform(shape=(length, rows)), axis=-1)[:, :topk].astype(mx.int32)
    dropped = mx.random.uniform(shape=order.shape) < 0.1
    indices = mx.where(dropped, mx.array(-1, mx.int32), order)
    mx.eval(q, q_pe, kv, k_pe, indices)
    return q, q_pe, kv, k_pe, indices


def tune(tuner, grid, buffers, smith_type, rows, scale, constants):
    tuner[grid](*buffers, Buffer(smith_type, buffers[0].n), rows, scale, **constants)
    return tuner.best_config.values, len([entry for entry in tuner.results if entry["seconds"] is not None])


def run(heads, length, rank, rope, rows, topk, mlx_type, smith_type, mlx_lm_kernel):
    q, q_pe, kv, k_pe, indices = make_inputs(heads, length, rank, rope, rows, topk, mlx_type)
    scale = (rank + rope) ** -0.5
    constants = dict(HEADS=heads, SEQ_Q=length, RANK=rank, ROPE=rope, TOPK=topk)
    scalar_grid = sparse_mla_grid(heads, length)
    matrix_grid = sparse_mla_matrix_grid(heads, length)

    buffers = [mlx_bridge.buffer_from_mlx(array) for array in (q, q_pe, kv, k_pe, indices)]
    scalar_best, scalar_count = tune(tuned_sparse_mla, scalar_grid, buffers, smith_type, rows, scale, constants)
    matrix_best, matrix_count = tune(tuned_sparse_mla_matrix, matrix_grid, buffers, smith_type, rows, scale, constants)

    template = mx.zeros(q.shape, dtype=mlx_type)
    scalar = mlx_bridge.kernel(sparse_mla)
    matrix = mlx_bridge.kernel(sparse_mla_matrix)
    arguments = (q, q_pe, kv, k_pe, indices, template, rows, scale)
    run_reference = lambda: reference(q, q_pe, kv, k_pe, indices, scale)
    contenders = {
        "mlx plain ops (gather)": run_reference,
        "smith scalar tuned": lambda: scalar[scalar_grid](*arguments, **constants, **scalar_best),
        "smith matrix default": lambda: matrix[matrix_grid](*arguments, **constants),
        "smith matrix tuned": lambda: matrix[matrix_grid](*arguments, **constants, **matrix_best),
    }
    if mlx_lm_kernel is not None:
        fused = lambda: mlx_lm_kernel(q, q_pe, kv, k_pe, indices, scale, variant="simdgroup_matrix")
        if fused() is not None:
            contenders["mlx-lm simdgroup_matrix"] = fused

    expected = run_reference().astype(mx.float32)
    errors = {name: mx.max(mx.abs(function().astype(mx.float32) - expected)).item() for name, function in contenders.items()}
    flops = 2 * heads * length * topk * (2 * rank + rope)

    print(f"\n{mlx_type} heads={heads} queries={length} cache rows={rows} topk={topk} rank={rank} rope={rope}")
    print(f"  tuned scalar {scalar_best} ({scalar_count} configs)\n  tuned matrix {matrix_best} ({matrix_count} configs)")
    timings = {}
    for name, function in contenders.items():
        timings[name] = median_time(function)
        print(f"  {name:24s} {timings[name] * 1e3:9.3f} ms  {flops / timings[name] / 1e12:7.3f} TFLOP/s  max error vs plain ops {errors[name]:.1e}")
    if "mlx-lm simdgroup_matrix" in timings:
        print(f"  smith matrix tuned vs mlx-lm: {timings['smith matrix tuned'] / timings['mlx-lm simdgroup_matrix']:.2f}x time   "
              f"scalar tuned vs mlx-lm: {timings['smith scalar tuned'] / timings['mlx-lm simdgroup_matrix']:.2f}x time")
    print(f"  matrix vs scalar: {timings['smith scalar tuned'] / timings['smith matrix tuned']:.2f}x faster   "
          f"matrix autotune gain: {timings['smith matrix default'] / timings['smith matrix tuned']:.2f}x")


SHAPES = [(16, 128, 512, 64, 4096, 512), (32, 256, 512, 64, 4096, 1024), (64, 128, 512, 64, 8192, 2048)]
TYPES = [(mx.float16, dtype.float16), (mx.bfloat16, dtype.bfloat16)]

if __name__ == "__main__":
    mlx_lm_kernel = load_mlx_lm_kernel()
    if mlx_lm_kernel is None:
        print("mlx-lm's sparse_mla is not importable: install mlx-lm or set SPARSE_MLA_REFERENCE to a copy of mlx_lm/kernels/sparse_mla.py")
    shapes = SHAPES if len(sys.argv) == 1 else [tuple(int(value) for value in sys.argv[1:7])]
    for shape in shapes:
        for mlx_type, smith_type in TYPES:
            run(*shape, mlx_type, smith_type, mlx_lm_kernel)
