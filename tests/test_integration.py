import importlib.util
import os
import sys
import tempfile
import unittest
from unittest import mock

import smith as sth
from smith import dtype

HAS_MLX = importlib.util.find_spec("mlx") is not None

if sys.platform == "darwin":
    from smith.compiler import Buffer


@sth.jit
def add(x_pointer, y_pointer, out_pointer, n, BLOCK: sth.constexpr = 4):
    offset = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    x = sth.load(x_pointer + offset, mask=mask)
    y = sth.load(y_pointer + offset, mask=mask)
    sth.store(out_pointer + offset, x + y, mask=mask)


def grid_for(n):
    return lambda meta: (-(-n // meta["BLOCK"]),)


@unittest.skipUnless(sys.platform == "darwin", "Metal requires macOS")
class TestExports(unittest.TestCase):
    def test_public_names(self):
        for name in ("jit", "constexpr", "synchronize", "autotune", "Config", "configs_from_grid", "Profiler"):
            self.assertTrue(hasattr(sth, name), name)

    def test_profiler_around_an_autotuned_half_precision_kernel(self):
        tuned = sth.autotune(sth.configs_from_grid({"BLOCK": [32, 64, 128]}), key=["n"])(add)
        n = 1000
        x = Buffer.from_list(dtype.float16, [float(i % 100) for i in range(n)])
        y = Buffer.from_list(dtype.float16, [1.0] * n)
        out = Buffer.from_list(dtype.float16, [0.0] * n)
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"SMITH_CACHE_DIR": directory}):
            with sth.Profiler() as profiler:
                tuned[grid_for(n)](x, y, out, n)
            self.assertEqual(out.to_list(), [float(i % 100) + 1.0 for i in range(n)])
        self.assertGreater(len(profiler.records), 0)
        self.assertIn("add", profiler.report())
        self.assertEqual(len(tuned.results), 3)


@unittest.skipUnless(sys.platform == "darwin" and HAS_MLX, "needs macOS and mlx")
class TestMlxRoundTrip(unittest.TestCase):
    def test_float16_and_bfloat16_through_mlx(self):
        import mlx.core as mx
        from smith import mlx_bridge
        add_mlx = mlx_bridge.kernel(add)
        n = 1000
        for mlx_type, smith_type in ((mx.float16, dtype.float16), (mx.bfloat16, dtype.bfloat16)):
            x = mx.arange(n).astype(mlx_type)
            y = mx.ones(n).astype(mlx_type)
            out = add_mlx[grid_for(n)](x, y, mx.zeros(n).astype(mlx_type), n, BLOCK=64)
            expected = x + y
            self.assertEqual(out.dtype, mlx_type)
            self.assertTrue(mx.array_equal(out, expected).item(), smith_type.name)

    def test_copy_fallback_for_half_precision(self):
        import mlx.core as mx
        from smith import mlx_bridge
        for mlx_type in (mx.float16, mx.bfloat16):
            source = (mx.arange(64) * 0.5).astype(mlx_type)
            mx.eval(source)
            buffer = mlx_bridge.buffer_from_mlx(source)
            back = mlx_bridge.mlx_from_buffer(buffer, source.shape)
            self.assertTrue(mx.array_equal(back, source).item())


if __name__ == "__main__":
    unittest.main()
