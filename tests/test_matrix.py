import random
import sys
import unittest

import smith as sth
from smith import dtype
from smith.ir import Graph
from smith.passes import simplify
from smith.trace import BufferSpec

if sys.platform == "darwin":
    from smith.compiler import Buffer


def fragment():
    lane = sth.arange(0, 32)
    quad = lane // 4
    return (quad & 4) + (lane // 2) % 4, (quad & 2) * 2 + (lane % 2) * 2


@sth.jit
def matmul(a_pointer, b_pointer, out_pointer, TRANSPOSE_B: sth.constexpr = False):
    row, column = fragment()
    a = sth.matrix_load(a_pointer + 0, 8)
    b = sth.matrix_load(b_pointer + 0, 8, TRANSPOSE_B)
    c = sth.matrix_mma(a, b, sth.matrix_fill(0.0, dtype.float32))
    low, high = sth.matrix_elements(c)
    sth.store(out_pointer + row * 8 + column, low)
    sth.store(out_pointer + row * 8 + column + 1, high)


@sth.jit
def rebuild(a_pointer, out_pointer):
    row, column = fragment()
    a = sth.matrix_load(a_pointer + 0, 8)
    low, high = sth.matrix_elements(a)
    doubled = sth.matrix_make(low * 2.0, high * 2.0)
    result = sth.matrix_mma(doubled, sth.matrix_load(a_pointer + 0, 8), sth.matrix_fill(1.0, dtype.float32))
    first, second = sth.matrix_elements(result)
    sth.store(out_pointer + row * 8 + column, first)
    sth.store(out_pointer + row * 8 + column + 1, second)


@sth.jit
def tiled(a_pointer, b_pointer, out_pointer, K: sth.constexpr = 16):
    row, column = fragment()
    accumulator = sth.matrix_fill(0.0, dtype.float32)
    for block in range(K // 8):
        accumulator = sth.matrix_mma(sth.matrix_load(a_pointer + block * 8, K),
                                     sth.matrix_load(b_pointer + block * 8 * 8, 8), accumulator)
    low, high = sth.matrix_elements(accumulator)
    sth.store(out_pointer + row * 8 + column, low)
    sth.store(out_pointer + row * 8 + column + 1, high)


def matrices():
    generator = random.Random(0)
    return [generator.uniform(-1, 1) for _ in range(64)], [generator.uniform(-1, 1) for _ in range(64)]


def as_rows(flat, width=8):
    return [flat[r * width:(r + 1) * width] for r in range(len(flat) // width)]


def product(x, y):
    return [[sum(x[i][k] * y[k][j] for k in range(len(y))) for j in range(len(y[0]))] for i in range(len(x))]


def transpose(x):
    return [list(column) for column in zip(*x)]


def flatten(rows):
    return [value for row in rows for value in row]


def maximum_error(left, right):
    return max(abs(a - b) for a, b in zip(left, right))


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()
        self.buffer = self.graph.buffer("data", dtype.float16)

    def test_matrices_reject_scalar_arithmetic(self):
        matrix = self.graph.matrix_fill(self.graph.const(0.0), dtype.float32)
        with self.assertRaises(TypeError):
            self.graph.binary("add", matrix, self.graph.const(1.0))
        with self.assertRaises(TypeError):
            self.graph.cast(matrix, dtype.float16)

    def test_mma_inputs_must_share_an_element_type(self):
        half = self.graph.matrix_load(self.buffer, self.graph.const(0), self.graph.const(8), False)
        single = self.graph.matrix_fill(self.graph.const(0.0), dtype.float32)
        with self.assertRaises(TypeError):
            self.graph.matrix_mma(half, single, single)

    def test_mixed_precision_accumulation_is_allowed(self):
        half = self.graph.matrix_load(self.buffer, self.graph.const(0), self.graph.const(8), False)
        accumulator = self.graph.matrix_fill(self.graph.const(0.0), dtype.float32)
        self.assertEqual(self.graph.matrix_mma(half, half, accumulator).dtype, accumulator.dtype)

    def test_only_float_buffers_can_be_loaded_as_matrices(self):
        integers = self.graph.buffer("integers", dtype.int32)
        with self.assertRaises(TypeError):
            self.graph.matrix_load(integers, self.graph.const(0), self.graph.const(8), False)

    def test_reading_an_element_of_a_rebuilt_matrix_is_folded(self):
        low, high = self.graph.const(1.0), self.graph.const(2.0)
        rebuilt = self.graph.matrix_make(low, high)
        element = self.graph.matrix_element(rebuilt, 1)
        out = self.graph.buffer("out", dtype.float32)
        self.graph.store(out, self.graph.lane(), element, None)
        result = simplify(self.graph)
        self.assertEqual([node.op for node in result.nodes if node.op.startswith("matrix_")], ["matrix_make"])
        stored = next(node for node in result.nodes if node.op == "store")
        self.assertEqual(stored.inputs[2].op, "const")

    def test_matrix_element_index_is_checked(self):
        matrix = self.graph.matrix_fill(self.graph.const(0.0), dtype.float32)
        with self.assertRaises(TypeError):
            self.graph.matrix_element(matrix, 2)


class TestTracing(unittest.TestCase):
    def test_source_uses_the_simdgroup_api(self):
        program = matmul.trace(BufferSpec(dtype.float16), BufferSpec(dtype.float16), BufferSpec(dtype.float32), TRANSPOSE_B=True)
        self.assertIn("#include <metal_simdgroup_matrix>", program.source)
        self.assertIn("simdgroup_load(", program.source)
        self.assertIn("ulong2(0, 0), true)", program.source)
        self.assertIn("simdgroup_multiply_accumulate(", program.source)
        self.assertIn("make_filled_simdgroup_matrix<float, 8, 8>", program.source)
        self.assertTrue(program.uses_matrices)

    def test_matrix_loads_count_as_buffer_reads(self):
        program = matmul.trace(BufferSpec(dtype.float32), BufferSpec(dtype.float32), BufferSpec(dtype.float32))
        self.assertEqual(program.loaded, {"a_pointer", "b_pointer"})
        self.assertEqual(program.stored, {"out_pointer"})

    def test_scalar_kernels_do_not_use_matrices(self):
        @sth.jit
        def copy(x_pointer, out_pointer):
            lane = sth.arange(0, 8)
            sth.store(out_pointer + lane, sth.load(x_pointer + lane))

        self.assertFalse(copy.trace(BufferSpec(dtype.float32), BufferSpec(dtype.float32)).uses_matrices)

    def test_interpreter_reports_that_matrices_need_a_device(self):
        with self.assertRaises(NotImplementedError):
            matmul[(1,)]([0.0] * 64, [0.0] * 64, [0.0] * 64)


@unittest.skipUnless(sys.platform == "darwin", "Metal requires macOS")
class TestMetal(unittest.TestCase):
    def run_matmul(self, a_type, b_type, a, b, **config):
        out = Buffer.from_list(dtype.float32, [0.0] * 64)
        matmul[(1,)](Buffer.from_list(a_type, a), Buffer.from_list(b_type, b), out, **config)
        return out.to_list()

    def test_float32_product(self):
        a, b = matrices()
        self.assertLess(maximum_error(self.run_matmul(dtype.float32, dtype.float32, a, b),
                                      flatten(product(as_rows(a), as_rows(b)))), 1e-5)

    def test_transposed_load(self):
        a, b = matrices()
        expected = flatten(product(as_rows(a), transpose(as_rows(b))))
        self.assertLess(maximum_error(self.run_matmul(dtype.float32, dtype.float32, a, b, TRANSPOSE_B=True), expected), 1e-5)

    def test_half_inputs_with_float_accumulator(self):
        a, b = matrices()
        rounded = lambda values: [dtype.float16.unpack(dtype.float16.pack([v]), 1)[0] for v in values]
        expected = flatten(product(as_rows(rounded(a)), as_rows(rounded(b))))
        self.assertLess(maximum_error(self.run_matmul(dtype.float16, dtype.float16, a, b), expected), 1e-5)

    def test_elements_round_trip_through_matrix_make(self):
        a, _ = matrices()
        out = Buffer.from_list(dtype.float32, [0.0] * 64)
        rebuild[(1,)](Buffer.from_list(dtype.float32, a), out)
        expected = [[1.0 + 2.0 * value for value in row] for row in product(as_rows(a), as_rows(a))]
        self.assertLess(maximum_error(out.to_list(), flatten(expected)), 1e-5)

    def test_accumulating_over_tiles_in_a_python_loop(self):
        generator = random.Random(1)
        a = [generator.uniform(-1, 1) for _ in range(8 * 16)]
        b = [generator.uniform(-1, 1) for _ in range(16 * 8)]
        out = Buffer.from_list(dtype.float32, [0.0] * 64)
        tiled[(1,)](Buffer.from_list(dtype.float32, a), Buffer.from_list(dtype.float32, b), out, K=16)
        expected = flatten(product(as_rows(a, 16), as_rows(b, 8)))
        self.assertLess(maximum_error(out.to_list(), expected), 1e-5)


if __name__ == "__main__":
    unittest.main()
