import importlib.util
import os
import sys
import tempfile
import unittest
from unittest import mock

import smith as sth
from smith.kernels.attention import (attention_grid, attention_matrix_grid, flash_attention, flash_attention_matrix,
                                     tuned_flash_attention, tuned_flash_attention_matrix)

HAS_MLX = importlib.util.find_spec("mlx") is not None
METAL = sys.platform == "darwin"

if HAS_MLX:
    import mlx.core as mx
    from smith import mlx_bridge

if HAS_MLX and METAL:
    from smith.compiler import Buffer


def make_inputs(batch_heads, seq, head_dim, element=None, seed=0):
    mx.random.seed(seed)
    element = element or mx.float32
    return [mx.random.normal((batch_heads, seq, head_dim)).astype(element) for _ in range(3)]


def sdpa(q, k, v, scale, causal):
    batch_heads, seq, head_dim = q.shape
    shape = (1, batch_heads, seq, head_dim)
    result = mx.fast.scaled_dot_product_attention(q.reshape(shape), k.reshape(shape), v.reshape(shape), scale=scale,
                                                  mask="causal" if causal else None)
    return result.reshape(q.shape)


def flat_list(array):
    return array.reshape(-1).tolist()


def maximum_error(left, right):
    return mx.max(mx.abs(left.astype(mx.float32) - right.astype(mx.float32))).item()


def run_metal(q, k, v, scale, causal, **config):
    batch_heads, seq, head_dim = q.shape
    out = Buffer(mlx_bridge.smith_dtype(q.dtype), q.size)
    inputs = [mlx_bridge.buffer_from_mlx(array) for array in (q, k, v)]
    flash_attention[attention_grid(batch_heads, seq)](*inputs, out, scale, SEQ_Q=seq, SEQ_KV=seq, HEAD_DIM=head_dim,
                                                      CAUSAL=causal, **config)
    sth.synchronize()
    return mlx_bridge.mlx_from_buffer(out, q.shape)


def run_matrix_metal(q, k, v, scale, causal, **config):
    batch_heads, seq, head_dim = q.shape
    out = Buffer(mlx_bridge.smith_dtype(q.dtype), q.size)
    inputs = [mlx_bridge.buffer_from_mlx(array) for array in (q, k, v)]
    flash_attention_matrix[attention_matrix_grid(batch_heads, seq)](*inputs, out, scale, SEQ_Q=seq, SEQ_KV=seq,
                                                                    HEAD_DIM=head_dim, CAUSAL=causal, **config)
    sth.synchronize()
    return mlx_bridge.mlx_from_buffer(out, q.shape)


class TestConfigValidation(unittest.TestCase):
    def test_block_must_divide_the_key_length(self):
        with self.assertRaises(ValueError):
            flash_attention[(1, 1)]([0.0] * 8, [0.0] * 8, [0.0] * 8, [0.0] * 8, 1.0, SEQ_Q=2, SEQ_KV=2, HEAD_DIM=4, BLOCK_N=4)

    def test_split_must_be_a_power_of_two_dividing_the_head_dim(self):
        with self.assertRaises(ValueError):
            flash_attention[(1, 1)]([0.0] * 8, [0.0] * 8, [0.0] * 8, [0.0] * 8, 1.0, SEQ_Q=2, SEQ_KV=2, HEAD_DIM=4, SPLIT=3)

    def test_causal_needs_square_attention(self):
        with self.assertRaises(ValueError):
            flash_attention[(1, 1)]([0.0] * 8, [0.0] * 8, [0.0] * 8, [0.0] * 8, 1.0, SEQ_Q=2, SEQ_KV=4, HEAD_DIM=4, CAUSAL=True)


class TestMatrixConfigValidation(unittest.TestCase):
    def test_sizes_must_be_multiples_of_eight(self):
        arguments = ([0.0] * 8, [0.0] * 8, [0.0] * 8, [0.0] * 8, 1.0)
        with self.assertRaises(ValueError):
            flash_attention_matrix[(1, 1)](*arguments, SEQ_Q=4, SEQ_KV=8, HEAD_DIM=8)
        with self.assertRaises(ValueError):
            flash_attention_matrix[(1, 1)](*arguments, SEQ_Q=8, SEQ_KV=8, HEAD_DIM=4)
        with self.assertRaises(ValueError):
            flash_attention_matrix[(1, 1)](*arguments, SEQ_Q=8, SEQ_KV=8, HEAD_DIM=8, KEY_BLOCKS=2)

    def test_causal_needs_square_attention(self):
        with self.assertRaises(ValueError):
            flash_attention_matrix[(1, 1)]([0.0] * 8, [0.0] * 8, [0.0] * 8, [0.0] * 8, 1.0, SEQ_Q=8, SEQ_KV=16,
                                           HEAD_DIM=8, CAUSAL=True)


@unittest.skipUnless(HAS_MLX, "mlx is the reference")
class TestInterpreter(unittest.TestCase):
    def test_matches_mlx_sdpa(self):
        q, k, v = make_inputs(2, 16, 8)
        scale = 8 ** -0.5
        for causal in (False, True):
            expected = flat_list(sdpa(q, k, v, scale, causal))
            out = [0.0] * len(expected)
            flash_attention[attention_grid(2, 16)](flat_list(q), flat_list(k), flat_list(v), out, scale, SEQ_Q=16, SEQ_KV=16,
                                                   HEAD_DIM=8, CAUSAL=causal, BLOCK_M=4, BLOCK_N=2, SPLIT=2)
            self.assertLess(max(abs(a - b) for a, b in zip(out, expected)), 1e-5, causal)


@unittest.skipUnless(METAL and HAS_MLX, "needs macOS and mlx")
class TestMetal(unittest.TestCase):
    def test_matches_mlx_sdpa_across_configs(self):
        q, k, v = make_inputs(2, 32, 16)
        scale = 16 ** -0.5
        for causal in (False, True):
            expected = sdpa(q, k, v, scale, causal)
            for config in ({"BLOCK_M": 8, "BLOCK_N": 1, "SPLIT": 1}, {"BLOCK_M": 4, "BLOCK_N": 2, "SPLIT": 4},
                           {"BLOCK_M": 2, "BLOCK_N": 4, "SPLIT": 16}):
                self.assertLess(maximum_error(run_metal(q, k, v, scale, causal, **config), expected), 1e-5, (causal, config))

    def test_sequence_length_not_a_multiple_of_the_block(self):
        q, k, v = make_inputs(1, 20, 8)
        scale = 8 ** -0.5
        result = run_metal(q, k, v, scale, False, BLOCK_M=8, BLOCK_N=2, SPLIT=2)
        self.assertLess(maximum_error(result, sdpa(q, k, v, scale, False)), 1e-5)

    def test_half_precision_inputs_accumulate_in_float32(self):
        q, k, v = make_inputs(1, 32, 16, element=mx.float16)
        scale = 16 ** -0.5
        result = run_metal(q, k, v, scale, False, BLOCK_M=4, BLOCK_N=2, SPLIT=4)
        self.assertLess(maximum_error(result, sdpa(q, k, v, scale, False)), 5e-3)

    def test_autotuner_picks_a_working_config(self):
        q, k, v = make_inputs(1, 32, 16)
        scale = 16 ** -0.5
        out = Buffer(mlx_bridge.smith_dtype(q.dtype), q.size)
        inputs = [mlx_bridge.buffer_from_mlx(array) for array in (q, k, v)]
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"SMITH_CACHE_DIR": directory}):
            tuned_flash_attention[attention_grid(1, 32)](*inputs, out, scale, SEQ_Q=32, SEQ_KV=32, HEAD_DIM=16, CAUSAL=False)
        sth.synchronize()
        self.assertIsNotNone(tuned_flash_attention.best_config)
        self.assertLess(maximum_error(mlx_bridge.mlx_from_buffer(out, q.shape), sdpa(q, k, v, scale, False)), 1e-5)

    def test_matrix_kernel_matches_mlx_sdpa_across_configs(self):
        q, k, v = make_inputs(2, 64, 16)
        scale = 16 ** -0.5
        for causal in (False, True):
            expected = sdpa(q, k, v, scale, causal)
            for config in ({"SIMDGROUPS": 1, "KEY_BLOCKS": 1, "ROW_BLOCKS": 1}, {"SIMDGROUPS": 4, "KEY_BLOCKS": 2, "ROW_BLOCKS": 1},
                           {"SIMDGROUPS": 2, "KEY_BLOCKS": 1, "ROW_BLOCKS": 2}, {"SIMDGROUPS": 2, "KEY_BLOCKS": 4, "ROW_BLOCKS": 4}):
                result = run_matrix_metal(q, k, v, scale, causal, **config)
                self.assertLess(maximum_error(result, expected), 1e-5, (causal, config))

    def test_matrix_kernel_handles_row_counts_that_do_not_fill_a_program(self):
        q, k, v = make_inputs(1, 40, 8)
        scale = 8 ** -0.5
        for causal in (False, True):
            result = run_matrix_metal(q, k, v, scale, causal, SIMDGROUPS=2, KEY_BLOCKS=1, ROW_BLOCKS=2)
            self.assertLess(maximum_error(result, sdpa(q, k, v, scale, causal)), 1e-5, causal)

    def test_matrix_kernel_half_precision(self):
        q, k, v = make_inputs(2, 64, 32, element=mx.float16)
        scale = 32 ** -0.5
        result = run_matrix_metal(q, k, v, scale, False, SIMDGROUPS=2, KEY_BLOCKS=2, ROW_BLOCKS=1)
        self.assertLess(maximum_error(result, sdpa(q, k, v, scale, False)), 5e-3)

    def test_matrix_autotuner_picks_a_working_config(self):
        q, k, v = make_inputs(1, 64, 16)
        scale = 16 ** -0.5
        out = Buffer(mlx_bridge.smith_dtype(q.dtype), q.size)
        inputs = [mlx_bridge.buffer_from_mlx(array) for array in (q, k, v)]
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"SMITH_CACHE_DIR": directory}):
            tuned_flash_attention_matrix[attention_matrix_grid(1, 64)](*inputs, out, scale, SEQ_Q=64, SEQ_KV=64,
                                                                       HEAD_DIM=16, CAUSAL=False)
        sth.synchronize()
        self.assertIsNotNone(tuned_flash_attention_matrix.best_config)
        self.assertLess(maximum_error(mlx_bridge.mlx_from_buffer(out, q.shape), sdpa(q, k, v, scale, False)), 1e-5)

    def test_matrix_kernel_runs_inside_mlx_through_the_bridge(self):
        heads, seq, head_dim = 4, 128, 64
        scale = head_dim ** -0.5
        bridged = mlx_bridge.kernel(flash_attention_matrix)
        for causal in (False, True):
            q, k, v = make_inputs(heads, seq, head_dim, seed=1)
            out = bridged[attention_matrix_grid(heads, seq)](q, k, v, mx.zeros(q.shape), scale, SEQ_Q=seq, SEQ_KV=seq,
                                                             HEAD_DIM=head_dim, CAUSAL=causal, SIMDGROUPS=4, KEY_BLOCKS=2)
            self.assertLess(maximum_error(out, sdpa(q, k, v, scale, causal)), 2e-5, causal)
        for _ in range(3):
            bridged[attention_matrix_grid(heads, seq)](q, k, v, mx.zeros(q.shape), scale, SEQ_Q=seq, SEQ_KV=seq,
                                                       HEAD_DIM=head_dim, CAUSAL=True, SIMDGROUPS=4, KEY_BLOCKS=2)
        self.assertEqual(len(bridged.bodies), 2)

    def test_runs_inside_mlx_through_the_bridge(self):
        heads, seq, head_dim = 4, 128, 64
        scale = head_dim ** -0.5
        bridged = mlx_bridge.kernel(flash_attention)
        for causal in (False, True):
            q, k, v = make_inputs(heads, seq, head_dim, seed=1)
            out = bridged[attention_grid(heads, seq)](q, k, v, mx.zeros(q.shape), scale, SEQ_Q=seq, SEQ_KV=seq,
                                                      HEAD_DIM=head_dim, CAUSAL=causal, BLOCK_M=8, BLOCK_N=2, SPLIT=8)
            self.assertLess(maximum_error(out, sdpa(q, k, v, scale, causal)), 2e-5, causal)


if __name__ == "__main__":
    unittest.main()
