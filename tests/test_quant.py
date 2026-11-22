import importlib.util
import random
import sys
import unittest

import smith as sth
from smith import dtype, quant
from smith.kernels.dequantize import dequantize_kernel
from smith.trace import BufferSpec

HAS_MLX = importlib.util.find_spec("mlx") is not None
METAL = sys.platform == "darwin"

if METAL:
    from smith.compiler import Buffer


@sth.jit
def widen(w_pointer, out_pointer, n, BLOCK: sth.constexpr = 8):
    offset = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    sth.store(out_pointer + offset, sth.load(w_pointer + offset, mask=mask, other=-1), mask=mask)


@sth.jit
def pack_pairs(x_pointer, out_pointer, n, BLOCK: sth.constexpr = 8):
    pair = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    mask = pair < n
    low = sth.load(x_pointer + pair * 2, mask=mask) & 15
    high = sth.load(x_pointer + pair * 2 + 1, mask=mask) & 15
    sth.store(out_pointer + pair, low | (high << 4), mask=mask)


def random_values(count, seed=0):
    generator = random.Random(seed)
    return [generator.uniform(-3.0, 3.0) for _ in range(count)]


class TestPacking(unittest.TestCase):
    def test_nibble_order_is_low_first(self):
        self.assertEqual(dtype.uint4.pack([1, 2, 3, 15]).hex(), "21f3")
        self.assertEqual(dtype.uint4.unpack(bytes.fromhex("21f3"), 4), [1, 2, 3, 15])

    def test_signed_nibbles_round_trip(self):
        values = [-8, -1, 0, 7, 3, -5, 6]
        self.assertEqual(dtype.int4.unpack(dtype.int4.pack(values), len(values)), values)

    def test_two_bit_fields_pack_four_per_byte(self):
        self.assertEqual(dtype.uint2.pack([0, 1, 2, 3, 3]).hex(), "e403")
        self.assertEqual(dtype.int2.unpack(dtype.int2.pack([-2, -1, 0, 1, 1]), 5), [-2, -1, 0, 1, 1])
        self.assertEqual(dtype.int2.nbytes(5), 2)

    def test_every_dtype_round_trips_through_its_own_codec(self):
        samples = {"float": [0.5, -2.0, 3.0], "int": [0, 1, -1], "uint": [0, 1, 1], "bool": [True, False, True]}
        for candidate in dtype.ALL:
            values = samples[candidate.kind]
            self.assertEqual(candidate.unpack(candidate.pack(values), len(values)), values, candidate.name)

    def test_byte_count_rounds_up(self):
        self.assertEqual([dtype.int4.nbytes(count) for count in (0, 1, 2, 3, 9)], [0, 1, 1, 2, 5])

    def test_words_match_dtype_packing(self):
        codes = [index % 16 for index in range(40)]
        words = quant.pack_words(codes, 4)
        self.assertEqual(quant.words_to_bytes(words)[:20], dtype.uint4.pack(codes))
        self.assertEqual(quant.unpack_words(words, 40, 4), codes)


class TestQuantize(unittest.TestCase):
    def test_error_is_bounded_by_half_a_step(self):
        values = random_values(256)
        words, scales, biases = quant.quantize(values, bits=4, group_size=64)
        restored = quant.dequantize(words, scales, biases, len(values), bits=4, group_size=64)
        for index, (original, value) in enumerate(zip(values, restored)):
            self.assertLessEqual(abs(original - value), scales[index // 64] / 2 + 1e-9)

    def test_more_bits_means_less_error(self):
        values = random_values(256, seed=1)
        errors = []
        for bits in (2, 4, 8):
            words, scales, biases = quant.quantize(values, bits=bits, group_size=64)
            restored = quant.dequantize(words, scales, biases, len(values), bits=bits, group_size=64)
            errors.append(max(abs(a - b) for a, b in zip(values, restored)))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])

    def test_constant_group_is_exact(self):
        words, scales, biases = quant.quantize([2.5] * 64, bits=4, group_size=64)
        self.assertEqual(quant.dequantize(words, scales, biases, 64, bits=4, group_size=64), [2.5] * 64)


class TestTracing(unittest.TestCase):
    def test_packed_load_lowers_to_shift_and_mask(self):
        program = widen.trace(BufferSpec(dtype.int4), BufferSpec(dtype.int8), 9, BLOCK=8)
        source = program.source
        self.assertIn("device uchar* w_pointer", source)
        self.assertIn(">>", source)
        self.assertIn("& 15", source)
        self.assertEqual(program.buffers["w_pointer"], dtype.int4)

    def test_unsigned_packed_load_has_no_sign_extension(self):
        signed = widen.trace(BufferSpec(dtype.int4), BufferSpec(dtype.int8), 9, BLOCK=8).source
        unsigned = widen.trace(BufferSpec(dtype.uint4), BufferSpec(dtype.uint8), 9, BLOCK=8).source
        self.assertGreater(signed.count("\n"), unsigned.count("\n"))

    def test_store_into_packed_buffer_is_rejected(self):
        with self.assertRaises(TypeError):
            widen.trace(BufferSpec(dtype.int8), BufferSpec(dtype.int4), 9, BLOCK=8)

    def test_shifts_fold_and_simplify(self):
        from smith.ir import Graph, format_graph
        from smith.passes import simplify
        graph = Graph()
        out = graph.buffer("out", dtype.int32)
        lane = graph.lane()
        folded = graph.binary("shl", graph.const(3), graph.const(2))
        kept = graph.binary("shr", lane, graph.const(0))
        graph.store(out, kept, folded, None)
        text = format_graph(simplify(graph))
        self.assertNotIn("shl", text)
        self.assertNotIn("shr", text)
        self.assertIn("const[12]", text)

    def test_shift_on_float_is_rejected(self):
        from smith.ir import Graph
        graph = Graph()
        with self.assertRaises(TypeError):
            graph.binary("shr", graph.const(1.0), graph.const(2.0))


class TestInterpreter(unittest.TestCase):
    def test_dequantize_kernel_in_pure_python(self):
        values = random_values(128, seed=2)
        words, scales, biases = quant.quantize(values, bits=4, group_size=64)
        out = [0.0] * 128
        dequantize_kernel[(2,)](words, scales, biases, out, 128, BITS=4, GROUP=64, BLOCK=64)
        expected = quant.dequantize(words, scales, biases, 128, bits=4, group_size=64)
        for value, reference in zip(out, expected):
            self.assertAlmostEqual(value, reference, places=9)


@unittest.skipUnless(METAL, "Metal requires macOS")
class TestMetal(unittest.TestCase):
    def test_int4_buffer_reads_signed_values(self):
        values = [-8, -1, 0, 7, 3, -5, 6, 2, 1]
        out = Buffer.from_list(dtype.int8, [99] * 12)
        widen[(2,)](Buffer.from_list(dtype.int4, values), out, len(values), BLOCK=8)
        self.assertEqual(out.to_list(), values + [99, 99, 99])

    def test_uint4_buffer_reads_unsigned_values(self):
        values = [0, 15, 7, 8, 1, 14, 3]
        out = Buffer.from_list(dtype.uint8, [99] * 8)
        widen[(1,)](Buffer.from_list(dtype.uint4, values), out, len(values), BLOCK=8)
        self.assertEqual(out.to_list(), values + [99])

    def test_int2_buffer_reads_signed_values(self):
        values = [-2, -1, 0, 1, 1, -2, 0]
        out = Buffer.from_list(dtype.int8, [99] * 8)
        widen[(1,)](Buffer.from_list(dtype.int2, values), out, len(values), BLOCK=8)
        self.assertEqual(out.to_list(), values + [99])

    def test_buffer_uses_half_a_byte_per_element(self):
        self.assertEqual(Buffer(dtype.int4, 1001).nbytes, 501)

    def test_kernel_packs_two_values_per_byte(self):
        values = [3, 12, 7, 0, 15, 9, 1, 2]
        out = Buffer.from_list(dtype.uint8, [0] * 4)
        pack_pairs[(1,)](Buffer.from_list(dtype.uint8, values), out, 4, BLOCK=8)
        self.assertEqual(out.to_list(), [3 | (12 << 4), 7, 15 | (9 << 4), 1 | (2 << 4)])
        self.assertEqual(dtype.uint4.unpack(bytes(out.to_list()), 8), values)

    def test_dequantize_kernel_matches_host(self):
        for bits in (2, 4, 8):
            values = random_values(256, seed=bits)
            words, scales, biases = quant.quantize(values, bits=bits, group_size=64)
            expected = quant.dequantize(words, scales, biases, 256, bits=bits, group_size=64)
            out = Buffer.from_list(dtype.float32, [0.0] * 256)
            dequantize_kernel[(4,)](
                Buffer.from_list(dtype.uint32, words), Buffer.from_list(dtype.float32, scales),
                Buffer.from_list(dtype.float32, biases), out, 256, BITS=bits, GROUP=64, BLOCK=64)
            for value, reference in zip(out.to_list(), expected):
                self.assertAlmostEqual(value, reference, places=4)


@unittest.skipUnless(METAL and HAS_MLX, "needs macOS and mlx")
class TestMlxQuantization(unittest.TestCase):
    def test_dequantize_mlx_quantized_weights(self):
        import mlx.core as mx
        from smith import mlx_bridge
        dequantize_mlx = mlx_bridge.kernel(dequantize_kernel)
        for weight_type in (mx.float32, mx.float16):
            weights = mx.random.normal((16, 128)).astype(weight_type)
            packed, scales, biases = mx.quantize(weights, group_size=64, bits=4)
            expected = mx.dequantize(packed, scales, biases, group_size=64, bits=4)
            count = weights.size
            out = dequantize_mlx[lambda meta: (-(-count // meta["BLOCK"]),)](
                packed, scales, biases, mx.zeros(weights.shape, dtype=weight_type), count, BITS=4, GROUP=64, BLOCK=64)
            tolerance = 1e-5 if weight_type == mx.float32 else 2e-3
            self.assertTrue(mx.allclose(out, expected, atol=tolerance, rtol=tolerance).item(), weight_type)


if __name__ == "__main__":
    unittest.main()
