from ..autotune import autotune, configs_from_grid
from ..dtype import float32
from ..jit import constexpr, jit
from ..ops import (arange, cast, exp, load, loop, matrix_elements, matrix_fill, matrix_load, matrix_make, matrix_mma,
                   maximum, minimum, program_id, shuffle_xor, store, where)

NEGATIVE_LARGE = -1e30


@jit
def flash_attention(q_pointer, k_pointer, v_pointer, out_pointer, scale,
                    SEQ_Q: constexpr, SEQ_KV: constexpr, HEAD_DIM: constexpr, CAUSAL: constexpr = False,
                    BLOCK_M: constexpr = 32, BLOCK_N: constexpr = 1, SPLIT: constexpr = 1):
    if SEQ_KV % BLOCK_N:
        raise ValueError("SEQ_KV must be a multiple of BLOCK_N")
    if CAUSAL and SEQ_Q != SEQ_KV:
        raise ValueError("causal attention needs SEQ_Q == SEQ_KV")
    if SPLIT < 1 or SPLIT > 32 or SPLIT & (SPLIT - 1) or HEAD_DIM % SPLIT:
        raise ValueError("SPLIT must be a power of two up to 32 that divides HEAD_DIM")
    local_dims = HEAD_DIM // SPLIT
    batch_head = program_id(1)
    thread = arange(0, BLOCK_M * SPLIT)
    row = program_id(0) * BLOCK_M + thread // SPLIT
    row_mask = row < SEQ_Q
    dim_base = thread % SPLIT * local_dims
    q_base = (batch_head * SEQ_Q + row) * HEAD_DIM + dim_base
    kv_base = batch_head * SEQ_KV * HEAD_DIM + dim_base
    queries = [cast(load(q_pointer + q_base + d, mask=row_mask), float32) * scale for d in range(local_dims)]

    def step(block, running_max, running_sum, *accumulators):
        first_key = block * BLOCK_N
        scores = []
        for k in range(BLOCK_N):
            key_base = kv_base + (first_key + k) * HEAD_DIM
            score = 0.0
            for d in range(local_dims):
                score = score + queries[d] * cast(load(k_pointer + key_base + d), float32)
            distance = 1
            while distance < SPLIT:
                score = score + shuffle_xor(score, distance)
                distance *= 2
            if CAUSAL:
                score = where(first_key + k > row, NEGATIVE_LARGE, score)
            scores.append(score)
        new_max = running_max
        for score in scores:
            new_max = maximum(new_max, score)
        correction = exp(running_max - new_max)
        probabilities = [exp(score - new_max) for score in scores]
        new_sum = running_sum * correction
        for probability in probabilities:
            new_sum = new_sum + probability
        updated = []
        for d in range(local_dims):
            value = accumulators[d] * correction
            for k in range(BLOCK_N):
                value_base = kv_base + (first_key + k) * HEAD_DIM
                value = value + probabilities[k] * cast(load(v_pointer + value_base + d), float32)
            updated.append(value)
        return (new_max, new_sum, *updated)

    blocks = SEQ_KV // BLOCK_N
    if CAUSAL:
        blocks = minimum(row // BLOCK_N + 1, blocks)
    results = loop(blocks, (NEGATIVE_LARGE, 0.0) + (0.0,) * local_dims, step)
    for d in range(local_dims):
        store(out_pointer + q_base + d, results[2 + d] / results[1], mask=row_mask)


CONFIGS = configs_from_grid({"BLOCK_M": [4, 8, 16, 32, 64], "BLOCK_N": [1, 2], "SPLIT": [4, 8, 16, 32]})

tuned_flash_attention = autotune(CONFIGS, key=["SEQ_Q", "SEQ_KV", "HEAD_DIM", "CAUSAL"], warmup=10, repeat=20)(flash_attention)


@jit
def flash_attention_matrix(q_pointer, k_pointer, v_pointer, out_pointer, scale,
                           SEQ_Q: constexpr, SEQ_KV: constexpr, HEAD_DIM: constexpr, CAUSAL: constexpr = False,
                           SIMDGROUPS: constexpr = 4, KEY_BLOCKS: constexpr = 1, ROW_BLOCKS: constexpr = 1):
    if SEQ_Q % 8 or HEAD_DIM % 8 or SEQ_KV % (8 * KEY_BLOCKS):
        raise ValueError("SEQ_Q and HEAD_DIM must be multiples of 8 and SEQ_KV a multiple of 8 * KEY_BLOCKS")
    if CAUSAL and SEQ_Q != SEQ_KV:
        raise ValueError("causal attention needs SEQ_Q == SEQ_KV")
    if SIMDGROUPS < 1 or SIMDGROUPS > 32 or ROW_BLOCKS < 1:
        raise ValueError("SIMDGROUPS must be between 1 and 32 and ROW_BLOCKS at least 1")
    tiles = HEAD_DIM // 8
    stride = 2 + tiles
    batch_head = program_id(1)
    thread = arange(0, SIMDGROUPS * 32)
    position = thread % 32
    quad = position // 4
    fragment_row = (quad & 4) + (position // 2) % 4
    fragment_column = (quad & 2) * 2 + (position % 2) * 2
    first_block = (program_id(0) * SIMDGROUPS + thread // 32) * ROW_BLOCKS
    row_bases = [minimum((first_block + r) * 8, SEQ_Q - 8) for r in range(ROW_BLOCKS)]
    rows = [row_base + fragment_row for row_base in row_bases]
    kv_offset = batch_head * SEQ_KV * HEAD_DIM
    probability_type = v_pointer.buffer.dtype
    query_tiles = [[matrix_load(q_pointer + ((batch_head * SEQ_Q + row_base) * HEAD_DIM + tile * 8), HEAD_DIM)
                    for tile in range(tiles)] for row_base in row_bases]

    def step(block, *state):
        scores = [[] for _ in range(ROW_BLOCKS)]
        for part in range(KEY_BLOCKS):
            first_key = (block * KEY_BLOCKS + part) * 8
            accumulators = [matrix_fill(0.0, float32) for _ in range(ROW_BLOCKS)]
            for tile in range(tiles):
                key_tile = matrix_load(k_pointer + (kv_offset + first_key * HEAD_DIM + tile * 8), HEAD_DIM, True)
                for r in range(ROW_BLOCKS):
                    accumulators[r] = matrix_mma(query_tiles[r][tile], key_tile, accumulators[r])
            for r in range(ROW_BLOCKS):
                low, high = matrix_elements(accumulators[r])
                low, high = low * scale, high * scale
                if CAUSAL:
                    low = where(first_key + fragment_column > rows[r], NEGATIVE_LARGE, low)
                    high = where(first_key + fragment_column + 1 > rows[r], NEGATIVE_LARGE, high)
                scores[r].append((low, high))
        updated = []
        weights = []
        for r in range(ROW_BLOCKS):
            running_max, running_sum = state[r * stride], state[r * stride + 1]
            block_max = scores[r][0][0]
            for low, high in scores[r]:
                block_max = maximum(block_max, maximum(low, high))
            block_max = maximum(block_max, shuffle_xor(block_max, 1))
            block_max = maximum(block_max, shuffle_xor(block_max, 8))
            new_max = maximum(running_max, block_max)
            correction = exp(running_max - new_max)
            row_weights = [(exp(low - new_max), exp(high - new_max)) for low, high in scores[r]]
            new_sum = running_sum * correction
            for low, high in row_weights:
                new_sum = new_sum + low + high
            updated.extend([new_max, new_sum])
            for tile in range(tiles):
                low, high = matrix_elements(state[r * stride + 2 + tile])
                updated.append(matrix_make(low * correction, high * correction))
            weights.append(row_weights)
        for part in range(KEY_BLOCKS):
            first_key = (block * KEY_BLOCKS + part) * 8
            probabilities = [matrix_make(weights[r][part][0], weights[r][part][1], probability_type)
                             for r in range(ROW_BLOCKS)]
            for tile in range(tiles):
                value_tile = matrix_load(v_pointer + (kv_offset + first_key * HEAD_DIM + tile * 8), HEAD_DIM)
                for r in range(ROW_BLOCKS):
                    slot = r * stride + 2 + tile
                    updated[slot] = matrix_mma(probabilities[r], value_tile, updated[slot])
        return tuple(updated)

    blocks = SEQ_KV // (8 * KEY_BLOCKS)
    if CAUSAL:
        blocks = minimum((row_bases[-1] + 7) // (8 * KEY_BLOCKS) + 1, blocks)
    initial = ((NEGATIVE_LARGE, 0.0) + tuple(matrix_fill(0.0, float32) for _ in range(tiles))) * ROW_BLOCKS
    results = loop(blocks, initial, step)
    for r in range(ROW_BLOCKS):
        total = results[r * stride + 1] + shuffle_xor(results[r * stride + 1], 1)
        total = total + shuffle_xor(total, 8)
        for tile in range(tiles):
            low, high = matrix_elements(results[r * stride + 2 + tile])
            base = (batch_head * SEQ_Q + rows[r]) * HEAD_DIM + tile * 8 + fragment_column
            store(out_pointer + base, low / total)
            store(out_pointer + (base + 1), high / total)


def attention_grid(batch_heads, seq_q):
    return lambda meta: (-(-seq_q // meta["BLOCK_M"]), batch_heads)


MATRIX_CONFIGS = configs_from_grid({"SIMDGROUPS": [1, 2, 4, 8], "KEY_BLOCKS": [1, 2, 4], "ROW_BLOCKS": [1, 2, 4]})

tuned_flash_attention_matrix = autotune(MATRIX_CONFIGS, key=["SEQ_Q", "SEQ_KV", "HEAD_DIM", "CAUSAL"], warmup=10, repeat=20)(flash_attention_matrix)


def attention_matrix_grid(batch_heads, seq_q):
    return lambda meta: (-(-seq_q // (8 * meta["SIMDGROUPS"] * meta.get("ROW_BLOCKS", 1))), batch_heads)
