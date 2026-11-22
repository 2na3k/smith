import struct
import sys
import unittest

import smith as sth
from smith import dtype, render
from smith.ir import Graph, promote
from smith.passes import simplify

if sys.platform == "darwin":
    from smith.compiler import Buffer, CompileError, compile_msl
    

def wrap(value, bits, signed):
    modulus = 1 << bits
    value %= modulus
    if signed and value >= modulus // 2:
        value -= modulus
    return value


@sth.jit
def add_typed(x_pointer, y_pointer, out_pointer, n, BLOCK: sth.constexpr = 4):
    offset = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    x = sth.load(x_pointer + offset, mask=mask)
    y = sth.load(y_pointer + offset, mask=mask)
    sth.store(out_pointer + offset, x + y, mask=mask)


class TestDTypeTable(unittest.TestCase):
    def test_existing_dtypes_keep_their_spelling(self):
        self.assertEqual((dtype.float32.msl, dtype.float32.size, dtype.float32.ctype),
                         ("float", 4, dtype.ctypes.c_float))
        self.assertEqual((dtype.int32.msl, dtype.int32.size), ("int", 4))
        self.assertEqual((dtype.uint32.msl, dtype.uint32.size), ("uint", 4))
        self.assertEqual((dtype.bool_.msl, dtype.bool_.size), ("bool", 1))

    def test_new_dtype_spellings(self):
        expected = {
            "float16": ("half", 2), "bfloat16": ("bfloat", 2), "int8": ("char", 1), "int16": ("short", 2),
            "int64": ("long", 8), "uint8": ("uchar", 1), "uint16": ("ushort", 2), "uint64": ("ulong", 8),
        }
        for name, (msl, size) in expected.items():
            value = getattr(dtype, name)
            self.assertEqual((value.name, value.msl, value.size), (name, msl, size))

    def test_kind_and_bits(self):
        self.assertEqual((dtype.float16.kind, dtype.float16.bits), ("float", 16))
        self.assertEqual((dtype.bfloat16.kind, dtype.bfloat16.bits), ("float", 16))
        self.assertEqual((dtype.float32.kind, dtype.float32.bits), ("float", 32))
        self.assertEqual((dtype.int8.kind, dtype.int8.bits), ("int", 8))
        self.assertEqual((dtype.int64.kind, dtype.int64.bits), ("int", 64))
        self.assertEqual((dtype.uint16.kind, dtype.uint16.bits), ("uint", 16))
        self.assertEqual((dtype.uint64.kind, dtype.uint64.bits), ("uint", 64))
        self.assertEqual((dtype.bool_.kind, dtype.bool_.bits), ("bool", 8))

    def test_half_types_have_no_ctype(self):
        self.assertIsNone(dtype.float16.ctype)
        self.assertIsNone(dtype.bfloat16.ctype)

    def test_all_lists_every_dtype(self):
        self.assertEqual(len(dtype.ALL), len(set(dtype.ALL)))
        self.assertEqual(set(dtype.ALL), {getattr(dtype, name) for name in
                         ("float16", "bfloat16", "float32", "int8", "int16", "int32", "int64",
                          "uint8", "uint16", "uint32", "uint64", "bool_")})

    def test_float64_is_a_clear_error(self):
        with self.assertRaises(AttributeError) as context:
            dtype.float64
        self.assertIn("float64 is not supported by Metal", str(context.exception))
        self.assertIsNone(getattr(dtype, "float64", None))


class TestPackUnpack(unittest.TestCase):
    def roundtrip(self, value, values):
        self.assertEqual(value.unpack(value.pack(values), len(values)), values)

    def test_integer_roundtrip(self):
        self.roundtrip(dtype.int8, [-128, -1, 0, 127])
        self.roundtrip(dtype.int16, [-32768, -2, 0, 32767])
        self.roundtrip(dtype.int32, [-2**31, -1, 0, 2**31 - 1])
        self.roundtrip(dtype.int64, [-2**63, -5, 0, 2**63 - 1])
        self.roundtrip(dtype.uint8, [0, 1, 255])
        self.roundtrip(dtype.uint16, [0, 65535])
        self.roundtrip(dtype.uint32, [0, 2**32 - 1])
        self.roundtrip(dtype.uint64, [0, 2**64 - 1])

    def test_bool_roundtrip(self):
        self.roundtrip(dtype.bool_, [True, False, True])

    def test_float_roundtrip_of_exact_values(self):
        self.roundtrip(dtype.float32, [0.0, -1.5, 2.25, 2.0**100])
        self.roundtrip(dtype.float16, [0.0, -1.5, 2.25, 65504.0])
        self.roundtrip(dtype.bfloat16, [0.0, -1.5, 2.25, 3.0 * 2**100])

    def test_integers_wrap_on_pack(self):
        self.assertEqual(dtype.int8.unpack(dtype.int8.pack([200, -129]), 2), [-56, 127])
        self.assertEqual(dtype.uint8.unpack(dtype.uint8.pack([300, -1]), 2), [44, 255])
        self.assertEqual(dtype.uint64.unpack(dtype.uint64.pack([-1]), 1), [2**64 - 1])

    def test_float16_layout_and_overflow(self):
        self.assertEqual(dtype.float16.pack([1.5]), struct.pack("<e", 1.5))
        self.assertEqual(dtype.float16.unpack(dtype.float16.pack([70000.0]), 1), [float("inf")])
        self.assertEqual(dtype.float16.unpack(dtype.float16.pack([float("-inf")]), 1), [float("-inf")])

    def test_bfloat16_is_top_half_of_float32_with_ties_to_even(self):
        self.assertEqual(dtype.bfloat16.pack([1.0]), struct.pack("<H", 0x3F80))
        self.assertEqual(dtype.bfloat16.unpack(struct.pack("<H", 0x4049), 1), [3.140625])
        self.assertEqual(dtype.bfloat16.unpack(dtype.bfloat16.pack([1.0 + 2**-8]), 1), [1.0])
        self.assertEqual(dtype.bfloat16.unpack(dtype.bfloat16.pack([1.0 + 3 * 2**-8]), 1), [1.0 + 2**-6])

    def test_bfloat16_special_values(self):
        values = dtype.bfloat16.unpack(dtype.bfloat16.pack([float("inf"), float("-inf"), float("nan")]), 3)
        self.assertEqual(values[:2], [float("inf"), float("-inf")])
        self.assertTrue(values[2] != values[2])


class TestPromotion(unittest.TestCase):
    def test_existing_rules_are_unchanged(self):
        self.assertIs(promote(dtype.int32, dtype.float32), dtype.float32)
        self.assertIs(promote(dtype.uint32, dtype.int32), dtype.int32)
        self.assertIs(promote(dtype.bool_, dtype.bool_), dtype.int32)
        self.assertIs(promote(dtype.bool_, dtype.uint32), dtype.uint32)

    def test_integers_widen_within_their_kind(self):
        self.assertIs(promote(dtype.int8, dtype.int16), dtype.int16)
        self.assertIs(promote(dtype.int16, dtype.int64), dtype.int64)
        self.assertIs(promote(dtype.uint8, dtype.uint32), dtype.uint32)
        self.assertIs(promote(dtype.uint8, dtype.uint8), dtype.uint8)

    def test_mixed_signedness_gives_the_signed_type_of_the_widest_width(self):
        self.assertIs(promote(dtype.uint64, dtype.int8), dtype.int64)
        self.assertIs(promote(dtype.uint16, dtype.int32), dtype.int32)

    def test_integer_with_float_gives_the_float_type(self):
        self.assertIs(promote(dtype.int8, dtype.float16), dtype.float16)
        self.assertIs(promote(dtype.int64, dtype.bfloat16), dtype.bfloat16)
        self.assertIs(promote(dtype.uint32, dtype.float32), dtype.float32)

    def test_float16_with_bfloat16_gives_float32(self):
        self.assertIs(promote(dtype.float16, dtype.bfloat16), dtype.float32)
        self.assertIs(promote(dtype.bfloat16, dtype.float16, dtype.int8), dtype.float32)

    def test_widest_float_wins(self):
        self.assertIs(promote(dtype.float16, dtype.float32), dtype.float32)
        self.assertIs(promote(dtype.bfloat16, dtype.float16), dtype.float32)


class TestLiteralRendering(unittest.TestCase):
    def literal(self, value, dtype_value):
        graph = Graph()
        return render._literal(graph.node("const", (), value, dtype_value))

    def test_literal_suffixes_per_dtype(self):
        self.assertEqual(self.literal(1.5, dtype.float16), "1.5h")
        self.assertEqual(self.literal(1.5, dtype.float32), "1.5f")
        self.assertEqual(self.literal(2, dtype.uint32), "2u")
        self.assertEqual(self.literal(3, dtype.int64), "3l")
        self.assertEqual(self.literal(2**64 - 1, dtype.uint64), "18446744073709551615ul")
        self.assertEqual(self.literal(7, dtype.int32), "7")

    def test_small_integers_and_bfloat_use_a_cast(self):
        self.assertEqual(self.literal(3, dtype.int8), "char(3)")
        self.assertEqual(self.literal(200, dtype.uint8), "uchar(200)")
        self.assertEqual(self.literal(1.5, dtype.bfloat16), "bfloat(1.5f)")

    def test_negative_literals_are_parenthesized(self):
        self.assertEqual(self.literal(-2, dtype.int8), "(char(-2))")
        self.assertEqual(self.literal(-1.5, dtype.float16), "(-1.5h)")
        self.assertEqual(self.literal(-3, dtype.int64), "(-3l)")
        self.assertEqual(self.literal(-2**63, dtype.int64), "(-9223372036854775807l - 1l)")

    def test_bool_literals(self):
        self.assertEqual(self.literal(True, dtype.bool_), "true")
        self.assertEqual(self.literal(False, dtype.bool_), "false")


class TestConstantFolding(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()
        self.out = self.graph.buffer("out", dtype.int8)

    def folded_constants(self, value):
        self.graph.store(self.out, self.graph.lane(), value, None)
        return [(n.arg, n.dtype) for n in simplify(self.graph).nodes if n.op == "const"]

    def test_casts_wrap_per_bit_width(self):
        self.assertIn((-56, dtype.int8), self.folded_constants(self.graph.cast(self.graph.const(200), dtype.int8)))

    def test_unsigned_casts_wrap(self):
        constants = self.folded_constants(self.graph.cast(self.graph.const(300), dtype.uint8))
        self.assertIn((44, dtype.uint8), constants)

    def test_arithmetic_wraps_in_its_own_width(self):
        left = self.graph.cast(self.graph.const(127), dtype.int8)
        right = self.graph.cast(self.graph.const(1), dtype.int8)
        constants = self.folded_constants(self.graph.binary("add", left, right))
        self.assertIn((-128, dtype.int8), constants)

    def test_unsigned_64_bit_wraps(self):
        constant = self.graph.cast(self.graph.const(-1), dtype.uint64)
        self.assertIn((2**64 - 1, dtype.uint64), self.folded_constants(constant))


class TestPromotedGraph(unittest.TestCase):
    def test_mixed_families_promote_in_the_graph(self):
        graph = Graph()
        total = graph.binary("add", graph.buffer("x", dtype.int32), graph.buffer("y", dtype.float16))
        self.assertIs(total.dtype, dtype.float16)


@unittest.skipUnless(sys.platform == "darwin", "Metal requires macOS")
class TestDTypeMetal(unittest.TestCase):
    def run_add(self, value_dtype, x, y):
        n = len(x)
        x_buffer = Buffer.from_list(value_dtype, x)
        y_buffer = Buffer.from_list(value_dtype, y)
        out_buffer = Buffer(value_dtype, n)
        grid = lambda meta: (sth.cdiv(n, meta["BLOCK"]),)
        program = add_typed.trace(x_buffer, y_buffer, out_buffer, n, BLOCK=4)
        try:
            compile_msl(program.source, program.name)
        except CompileError as error:
            self.skipTest(f"{value_dtype.name} does not compile on this Metal target: {error}")
        add_typed[grid](x_buffer, y_buffer, out_buffer, n, BLOCK=4)
        return out_buffer.to_list()

    def test_buffer_roundtrip_for_every_dtype(self):
        samples = {
            "float16": [0.0, -1.5, 2.25, 65504.0], "bfloat16": [0.0, -1.5, 2.25, 3.0 * 2**100],
            "float32": [0.0, -1.5, 2.25, 2.0**100], "int8": [-128, -1, 0, 127], "int16": [-32768, 5, 0, 32767],
            "int32": [-2**31, -1, 0, 2**31 - 1], "int64": [-2**63, -5, 0, 2**63 - 1],
            "uint8": [0, 1, 128, 255], "uint16": [0, 65535, 3, 4], "uint32": [0, 2**32 - 1, 7, 8],
            "uint64": [0, 2**64 - 1, 9, 10], "bool_": [True, False, True, False],
        }
        for name, values in samples.items():
            with self.subTest(dtype=name):
                value = getattr(dtype, name)
                self.assertEqual(Buffer.from_list(value, values).to_list(), values)

    def test_float_families_add(self):
        for value_dtype in (dtype.float32, dtype.float16, dtype.bfloat16):
            with self.subTest(dtype=value_dtype.name):
                x = [0.5 * i for i in range(10)]
                y = [1.25 * i - 3.0 for i in range(10)]
                self.assertEqual(self.run_add(value_dtype, x, y), [a + b for a, b in zip(x, y)])

    def test_signed_integer_families_add_and_wrap(self):
        for value_dtype, bits in ((dtype.int8, 8), (dtype.int16, 16), (dtype.int32, 32), (dtype.int64, 64)):
            with self.subTest(dtype=value_dtype.name):
                x = [wrap(100 * i - 300, bits, True) for i in range(10)]
                y = [wrap(2**(bits - 2) - 60 * i, bits, True) for i in range(10)]
                expected = [wrap(a + b, bits, True) for a, b in zip(x, y)]
                self.assertEqual(self.run_add(value_dtype, x, y), expected)

    def test_unsigned_integer_families_add_and_wrap(self):
        for value_dtype, bits in ((dtype.uint8, 8), (dtype.uint16, 16), (dtype.uint32, 32), (dtype.uint64, 64)):
            with self.subTest(dtype=value_dtype.name):
                x = [(2**bits - 1) - 7 * i for i in range(10)]
                y = [3 * i + 200 for i in range(10)]
                expected = [wrap(a + b, bits, False) for a, b in zip(x, y)]
                self.assertEqual(self.run_add(value_dtype, x, y), expected)

    def test_mixed_integer_and_float16_promotes_to_float16(self):
        n = 10
        x = [i - 4 for i in range(n)]
        y = [0.25 * i for i in range(n)]
        x_buffer = Buffer.from_list(dtype.int32, x)
        y_buffer = Buffer.from_list(dtype.float16, y)
        out_buffer = Buffer(dtype.float16, n)
        grid = lambda meta: (sth.cdiv(n, meta["BLOCK"]),)
        add_typed[grid](x_buffer, y_buffer, out_buffer, n, BLOCK=4)
        self.assertEqual(out_buffer.to_list(), [a + b for a, b in zip(x, y)])


if __name__ == "__main__":
    unittest.main()
