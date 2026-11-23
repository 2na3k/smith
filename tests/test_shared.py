import random
import sys
import unittest

import smith as sth
from smith import dtype
from smith.trace import BufferSpec

if sys.platform == "darwin":
    from smith.compiler import Buffer


@sth.jit
def reverse_groups(x_pointer, out_pointer, GROUP: sth.constexpr = 64):
    lane = sth.arange(0, GROUP)
    base = sth.program_id(0) * GROUP
    tile = sth.shared(dtype.float32, GROUP)
    sth.store(tile + lane, sth.load(x_pointer + (base + lane)))
    sth.barrier()
    sth.store(out_pointer + (base + lane), sth.load(tile + (GROUP - 1 - lane)))


@sth.jit
def group_sum(x_pointer, out_pointer, GROUP: sth.constexpr = 64):
    lane = sth.arange(0, GROUP)
    base = sth.program_id(0) * GROUP
    tile = sth.shared(dtype.float32, GROUP)
    sth.store(tile + lane, sth.load(x_pointer + (base + lane)))
    sth.barrier()
    stride = GROUP // 2
    while stride:
        active = lane < stride
        partner = sth.load(tile + (lane + stride), mask=active)
        mine = sth.load(tile + lane, mask=active)
        sth.barrier()
        sth.store(tile + lane, mine + partner, mask=active)
        sth.barrier()
        stride //= 2
    sth.store(out_pointer + (sth.program_id(0) + lane * 0), sth.load(tile + 0), mask=lane < 1)


@sth.jit
def looped_rotation(x_pointer, out_pointer, ROUNDS: sth.constexpr = 3, GROUP: sth.constexpr = 32):
    lane = sth.arange(0, GROUP)
    tile = sth.shared(dtype.float32, GROUP)
    sth.store(tile + lane, sth.load(x_pointer + lane))
    sth.barrier()

    def step(round_index, value):
        sth.barrier()
        sth.store(tile + lane, value)
        sth.barrier()
        return sth.load(tile + ((lane + 1) % GROUP))

    (result,) = sth.loop(ROUNDS, (sth.load(tile + lane),), step)
    sth.store(out_pointer + lane, result)


@sth.jit
def tile_through_shared(a_pointer, out_pointer):
    lane = sth.arange(0, 32)
    quad = lane // 4
    row, column = (quad & 4) + (lane // 2) % 4, (quad & 2) * 2 + (lane % 2) * 2
    tile = sth.shared(dtype.float32, 64)
    for part in range(2):
        sth.store(tile + (lane + part * 32), sth.load(a_pointer + (lane + part * 32)))
    sth.barrier()
    matrix = sth.matrix_load(tile + 0, 8)
    identity = sth.matrix_fill(0.0, dtype.float32)
    result = sth.matrix_mma(matrix, sth.matrix_load(tile + 0, 8), identity)
    low, high = sth.matrix_elements(result)
    sth.store(out_pointer + (row * 8 + column), low)
    sth.store(out_pointer + (row * 8 + column + 1), high)


class TestSharedTracing(unittest.TestCase):
    def trace(self, function, **constants):
        values = {name: BufferSpec(dtype.float32) for name in function.signature.parameters
                  if name.endswith("_pointer")}
        return function.trace(*values.values(), **constants)

    def test_source_declares_threadgroup_memory_and_barrier(self):
        source = self.trace(reverse_groups).source
        self.assertIn("threadgroup float shared_0[64];", source)
        self.assertIn("threadgroup_barrier(mem_flags::mem_threadgroup);", source)

    def test_shared_memory_is_not_a_kernel_argument(self):
        program = self.trace(reverse_groups)
        self.assertEqual(program.runtime_names, ["x_pointer", "out_pointer"])
        self.assertEqual(program.loaded, frozenset({"x_pointer"}))
        self.assertEqual(program.stored, frozenset({"out_pointer"}))

    def test_two_shared_arrays_get_distinct_names(self):
        @sth.jit
        def two(x_pointer, out_pointer):
            lane = sth.arange(0, 8)
            first = sth.shared(dtype.float32, 8)
            second = sth.shared(dtype.float32, 8)
            sth.store(first + lane, sth.load(x_pointer + lane))
            sth.store(second + lane, sth.load(first + lane))
            sth.barrier()
            sth.store(out_pointer + lane, sth.load(second + lane))

        source = self.trace(two).source
        self.assertIn("shared_0[8]", source)
        self.assertIn("shared_1[8]", source)

    def test_barrier_separates_loads_that_would_otherwise_merge(self):
        @sth.jit
        def twice(x_pointer, out_pointer):
            lane = sth.arange(0, 8)
            tile = sth.shared(dtype.float32, 8)
            sth.store(tile + lane, sth.load(x_pointer + lane))
            before = sth.load(tile + 0)
            sth.barrier()
            sth.store(tile + lane, 5.0)
            sth.barrier()
            after = sth.load(tile + 0)
            sth.store(out_pointer + lane, before + after)

        source = self.trace(twice).source
        self.assertEqual(source.count("= shared_0[0];"), 2)
        self.assertEqual(source.count("threadgroup_barrier"), 2)

    def test_barrier_inside_a_loop_is_kept(self):
        source = self.trace(looped_rotation).source
        self.assertEqual(source.count("threadgroup_barrier"), 3)


class TestSharedInterpreter(unittest.TestCase):
    def test_reverse_in_the_interpreter(self):
        x = [float(i) for i in range(128)]
        out = [0.0] * 128
        reverse_groups[(2,)](x, out, GROUP=64)
        self.assertEqual(out, x[63::-1] + x[127:63:-1])


@unittest.skipUnless(sys.platform == "darwin", "Metal requires macOS")
class TestSharedMetal(unittest.TestCase):
    def test_reverse_groups(self):
        x = [float(i) for i in range(256)]
        buffer, out = Buffer.from_list(dtype.float32, x), Buffer.from_list(dtype.float32, [0.0] * 256)
        reverse_groups[(4,)](buffer, out, GROUP=64)
        expected = []
        for group in range(4):
            expected += x[group * 64:(group + 1) * 64][::-1]
        self.assertEqual(out.to_list(), expected)

    def test_tree_reduction_with_barriers(self):
        generator = random.Random(1)
        x = [float(generator.randint(-8, 8)) for _ in range(256)]
        buffer, out = Buffer.from_list(dtype.float32, x), Buffer.from_list(dtype.float32, [0.0] * 4)
        group_sum[(4,)](buffer, out, GROUP=64)
        self.assertEqual(out.to_list(), [sum(x[g * 64:(g + 1) * 64]) for g in range(4)])

    def test_barriers_inside_a_loop(self):
        x = [float(i) for i in range(32)]
        buffer, out = Buffer.from_list(dtype.float32, x), Buffer.from_list(dtype.float32, [0.0] * 32)
        looped_rotation[(1,)](buffer, out, ROUNDS=3)
        self.assertEqual(out.to_list(), [x[(i + 3) % 32] for i in range(32)])

    def test_matrix_load_from_shared_memory(self):
        generator = random.Random(2)
        a = [generator.uniform(-1, 1) for _ in range(64)]
        buffer, out = Buffer.from_list(dtype.float32, a), Buffer.from_list(dtype.float32, [0.0] * 64)
        tile_through_shared[(1,)](buffer, out)
        rows = [a[r * 8:(r + 1) * 8] for r in range(8)]
        expected = [sum(rows[i][k] * rows[k][j] for k in range(8)) for i in range(8) for j in range(8)]
        self.assertLess(max(abs(x - y) for x, y in zip(out.to_list(), expected)), 1e-4)


@unittest.skipUnless(sys.platform == "darwin", "Metal requires macOS")
class TestSharedBridge(unittest.TestCase):
    def test_reverse_groups_through_mlx(self):
        try:
            import mlx.core as mx
        except ImportError:
            self.skipTest("mlx is not installed")
        from smith import mlx_bridge
        x = mx.arange(256).astype(mx.float32)
        result = mlx_bridge.kernel(reverse_groups)[(4,)](x, mx.zeros(256, dtype=mx.float32), GROUP=64)
        expected = []
        for group in range(4):
            expected += list(range(group * 64, (group + 1) * 64))[::-1]
        self.assertEqual(result.tolist(), [float(value) for value in expected])


if __name__ == "__main__":
    unittest.main()
