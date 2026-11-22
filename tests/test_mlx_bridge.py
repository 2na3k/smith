import importlib.util
import sys
import unittest

HAS_MLX = importlib.util.find_spec("mlx") is not None
HAS_METAL = sys.platform == "darwin" and HAS_MLX

if HAS_MLX:
    import mlx.core as mx
    import smith as sth
    from smith import mlx_bridge
    from smith.dtype import float32, int32, bool_
    from smith.compiler import Buffer

    @sth.jit
    def add(x_pointer, y_pointer, out_pointer, n, BLOCK: sth.constexpr = 4):
        pid = sth.program_id(0)
        offset = pid * BLOCK + sth.arange(0, BLOCK)
        mask = offset < n
        x = sth.load(x_pointer + offset, mask=mask)
        y = sth.load(y_pointer + offset, mask=mask)
        sth.store(out_pointer + offset, x + y, mask=mask)

    @sth.jit
    def scale(x_pointer, out_pointer, factor, n, BLOCK: sth.constexpr = 4):
        pid = sth.program_id(0)
        offset = pid * BLOCK + sth.arange(0, BLOCK)
        mask = offset < n
        x = sth.load(x_pointer + offset, mask=mask)
        sth.store(out_pointer + offset, x * factor, mask=mask)

    @sth.jit
    def copy_in_place(x_pointer, n, BLOCK: sth.constexpr = 4):
        pid = sth.program_id(0)
        offset = pid * BLOCK + sth.arange(0, BLOCK)
        mask = offset < n
        x = sth.load(x_pointer + offset, mask=mask)
        sth.store(x_pointer + offset, x, mask=mask)


def grid_for(n):
    return lambda meta: (-(-n // meta["BLOCK"]),)


@unittest.skipUnless(HAS_MLX, "mlx is not installed")
class TestDtypeMapping(unittest.TestCase):
    def test_float32_round_trip_names(self):
        self.assertEqual(mlx_bridge.smith_dtype(mx.float32), float32)
        self.assertEqual(mlx_bridge.smith_dtype(mx.int32), int32)
        self.assertEqual(mlx_bridge.smith_dtype(mx.bool_), bool_)

    def test_unknown_dtype_is_clear_error(self):
        with self.assertRaises(TypeError):
            mlx_bridge.smith_dtype(mx.complex64)

    def test_mlx_dtype_from_smith(self):
        self.assertEqual(mlx_bridge.mlx_dtype(float32), mx.float32)
        self.assertEqual(mlx_bridge.mlx_dtype(bool_), mx.bool_)


@unittest.skipUnless(HAS_METAL, "needs macOS with mlx")
class TestMLXKernel(unittest.TestCase):
    def setUp(self):
        self.add_mlx = mlx_bridge.kernel(add)

    def test_add_matches_mx_add(self):
        n = 1000
        x = mx.arange(n).astype(mx.float32)
        y = x * 10
        out = mx.zeros((n,), dtype=mx.float32)
        result = self.add_mlx[grid_for(n)](x, y, out, n, BLOCK=4)
        self.assertTrue(mx.array_equal(result, mx.add(x, y)).item())

    def test_result_is_mx_array(self):
        n = 8
        x = mx.ones((n,))
        result = self.add_mlx[grid_for(n)](x, x, mx.zeros((n,)), n)
        self.assertIsInstance(result, mx.array)
        self.assertEqual(result.shape, (n,))

    def test_composition_in_graph(self):
        n = 64
        x = mx.arange(n).astype(mx.float32)
        y = mx.ones((n,))
        total = mx.sum(self.add_mlx[grid_for(n)](x, y, mx.zeros((n,)), n))
        expected = mx.sum(mx.add(x, y))
        self.assertAlmostEqual(total.item(), expected.item(), places=3)
        chained = mx.multiply(self.add_mlx[grid_for(n)](x, y, mx.zeros((n,)), n), 2.0)
        self.assertTrue(mx.allclose(chained, mx.add(x, y) * 2.0).item())

    def test_mx_compile(self):
        n = 32
        x = mx.arange(n).astype(mx.float32)
        y = mx.ones((n,))

        def function(a, b):
            return mx.sum(self.add_mlx[grid_for(n)](a, b, mx.zeros((n,)), n))

        compiled = mx.compile(function)
        self.assertAlmostEqual(compiled(x, y).item(), function(x, y).item(), places=3)

    def test_scalar_argument(self):
        n = 10
        x = mx.arange(n).astype(mx.float32)
        scale_mlx = mlx_bridge.kernel(scale)
        result = scale_mlx[grid_for(n)](x, mx.zeros((n,)), 2.5, n, BLOCK=4)
        self.assertTrue(mx.allclose(result, x * 2.5).item())

    def test_integer_scalar_bounds_the_store(self):
        scale_mlx = mlx_bridge.kernel(scale)
        x = mx.arange(16).astype(mx.float32)
        result = scale_mlx[grid_for(16)](x, mx.zeros((16,)), 3.0, 5, BLOCK=4)
        self.assertEqual(result.tolist()[:5], [0.0, 3.0, 6.0, 9.0, 12.0])

    def test_masked_tail_with_init_value(self):
        n = 10
        x = mx.arange(n).astype(mx.float32)
        y = x * 10
        out = mx.zeros((16,), dtype=mx.float32)
        result = self.add_mlx[grid_for(16)](x, y, out, n, BLOCK=4, init_value=-1.0)
        values = result.tolist()
        self.assertEqual(values[:n], mx.add(x, y).tolist())
        self.assertEqual(values[n:], [-1.0] * 6)

    def test_error_on_in_out_buffer(self):
        bad = mlx_bridge.kernel(copy_in_place)
        x = mx.arange(8).astype(mx.float32)
        with self.assertRaises(ValueError):
            bad[grid_for(8)](x, 8)

    def test_kernel_requires_jit_function(self):
        with self.assertRaises(TypeError):
            mlx_bridge.kernel(lambda x: x)

    def test_call_without_grid_is_error(self):
        with self.assertRaises(TypeError):
            self.add_mlx(mx.zeros((4,)), mx.zeros((4,)), mx.zeros((4,)), 4)


@unittest.skipUnless(HAS_METAL, "needs macOS with mlx")
class TestCopyFallback(unittest.TestCase):
    def test_buffer_from_mlx_round_trip(self):
        array = mx.arange(12).astype(mx.float32)
        buffer = mlx_bridge.buffer_from_mlx(array)
        self.assertIsInstance(buffer, Buffer)
        back = mlx_bridge.mlx_from_buffer(buffer, (3, 4))
        self.assertEqual(back.shape, (3, 4))
        self.assertTrue(mx.array_equal(back.reshape(12), array).item())

    def test_smith_launch_on_mlx_data(self):
        n = 10
        x = mx.arange(n).astype(mx.float32)
        y = x * 10
        x_buffer = mlx_bridge.buffer_from_mlx(x)
        y_buffer = mlx_bridge.buffer_from_mlx(y)
        out_buffer = Buffer(float32, n)
        add[grid_for(n)](x_buffer, y_buffer, out_buffer, n, BLOCK=4)
        result = mlx_bridge.mlx_from_buffer(out_buffer, (n,))
        self.assertTrue(mx.allclose(result, mx.add(x, y)).item())

    def test_int32_round_trip(self):
        array = mx.array([1, -2, 3], dtype=mx.int32)
        back = mlx_bridge.mlx_from_buffer(mlx_bridge.buffer_from_mlx(array), (3,))
        self.assertEqual(back.tolist(), [1, -2, 3])

    def test_shape_mismatch_is_error(self):
        buffer = mlx_bridge.buffer_from_mlx(mx.zeros((4,)))
        with self.assertRaises(ValueError):
            mlx_bridge.mlx_from_buffer(buffer, (3,))


class TestImportIsolation(unittest.TestCase):
    def test_module_imports_without_mlx_loaded(self):
        import importlib
        module = importlib.import_module("smith.mlx_bridge")
        self.assertTrue(hasattr(module, "kernel"))


if __name__ == "__main__":
    unittest.main()
