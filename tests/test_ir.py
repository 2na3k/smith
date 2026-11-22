import unittest

from smith.dtype import bfloat16, bool_, float16, float32, int32
from smith.ir import Graph, format_graph
from smith.passes import simplify
from smith.render import render_metal


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()
        self.x = self.graph.buffer("x", float32)
        self.out = self.graph.buffer("out", float32)

    def test_identical_nodes_are_shared(self):
        a = self.graph.program_id(0)
        b = self.graph.lane()
        self.assertIs(self.graph.binary("add", a, b), self.graph.binary("add", a, b))

    def test_loads_are_shared_until_a_store(self):
        offset = self.graph.lane()
        first = self.graph.load(self.x, offset, None, None)
        self.assertIs(first, self.graph.load(self.x, offset, None, None))
        self.graph.store(self.out, offset, self.graph.const(1.0), None)
        self.assertIsNot(first, self.graph.load(self.x, offset, None, None))

    def test_mixed_arithmetic_inserts_casts(self):
        total = self.graph.binary("add", self.graph.lane(), self.graph.const(1.5))
        self.assertEqual(total.dtype, float32)
        self.assertEqual(total.inputs[0].op, "cast")

    def test_true_division_is_float(self):
        result = self.graph.binary("truediv", self.graph.lane(), self.graph.const(2))
        self.assertEqual(result.dtype, float32)

    def test_comparison_is_bool(self):
        self.assertEqual(self.graph.binary("less", self.graph.lane(), self.graph.const(3)).dtype, bool_)

    def test_bitwise_on_float_is_rejected(self):
        with self.assertRaises(TypeError):
            self.graph.binary("and", self.graph.const(1.0), self.graph.const(2.0))

    def test_bitwise_on_half_precision_is_rejected(self):
        for dtype in (float16, bfloat16):
            value = self.graph.cast(self.graph.const(1.0), dtype)
            with self.assertRaises(TypeError):
                self.graph.binary("or", value, value)


class TestSimplify(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()
        self.x = self.graph.buffer("x", float32)
        self.out = self.graph.buffer("out", float32)

    def test_identities_are_removed(self):
        offset = self.graph.binary("add", self.graph.lane(), self.graph.const(0))
        value = self.graph.binary("mul", self.graph.load(self.x, offset, None, None), self.graph.const(1.0))
        self.graph.store(self.out, offset, value, None)
        text = format_graph(simplify(self.graph))
        self.assertNotIn(" add", text)
        self.assertNotIn(" mul", text)

    def test_integer_constants_are_folded(self):
        offset = self.graph.binary("add", self.graph.const(2), self.graph.const(3))
        self.graph.store(self.out, offset, self.graph.const(1.0), None)
        result = simplify(self.graph)
        self.assertTrue(any(n.op == "const" and n.arg == 5 and n.dtype == int32 for n in result.nodes))
        self.assertNotIn(" add", format_graph(result))

    def test_constant_casts_are_folded(self):
        value = self.graph.cast(self.graph.const(3), float32)
        self.graph.store(self.out, self.graph.lane(), value, None)
        result = simplify(self.graph)
        self.assertTrue(any(n.op == "const" and n.arg == 3.0 and n.dtype == float32 for n in result.nodes))
        self.assertNotIn("cast", format_graph(result))

    def test_always_true_mask_is_dropped(self):
        mask = self.graph.binary("less", self.graph.const(1), self.graph.const(2))
        self.graph.store(self.out, self.graph.lane(), self.graph.const(1.0), mask)
        stores = [n for n in simplify(self.graph).nodes if n.op == "store"]
        self.assertEqual(len(stores), 1)
        self.assertEqual(len(stores[0].inputs), 3)

    def test_always_false_mask_removes_the_store(self):
        mask = self.graph.binary("less", self.graph.const(2), self.graph.const(1))
        self.graph.store(self.out, self.graph.lane(), self.graph.const(1.0), mask)
        self.assertEqual([n for n in simplify(self.graph).nodes if n.op == "store"], [])

    def test_double_negation_is_removed(self):
        lane = self.graph.lane()
        value = self.graph.negate(self.graph.negate(lane))
        self.graph.store(self.out, lane, self.graph.cast(value, float32), None)
        self.assertNotIn("neg", format_graph(simplify(self.graph)))


class TestRender(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()
        self.x = self.graph.buffer("x", float32)
        self.out = self.graph.buffer("out", float32)
        self.parameters = ["device float* x [[buffer(0)]]", "device float* out [[buffer(1)]]"]

    def render(self):
        return render_metal(simplify(self.graph), "kernel_name", self.parameters)

    def test_masked_store_and_shared_values(self):
        offset = self.graph.binary("add", self.graph.program_id(0), self.graph.lane())
        mask = self.graph.binary("less", offset, self.graph.const(10))
        value = self.graph.load(self.x, offset, mask, self.graph.const(0))
        self.graph.store(self.out, offset, value, mask)
        source = self.render()
        self.assertIn("kernel void kernel_name(", source)
        self.assertIn("if (value_2)", source)
        self.assertIn(": 0.0f", source)
        self.assertEqual(source.count("int(program_id.x)"), 1)

    def test_unused_loads_are_not_rendered(self):
        lane = self.graph.lane()
        self.graph.load(self.x, lane, None, None)
        self.graph.store(self.out, lane, self.graph.const(2.0), None)
        self.assertNotIn("x[", self.render())

    def test_negative_literals_are_parenthesized(self):
        self.graph.store(self.out, self.graph.lane(), self.graph.negate(self.graph.const(2.0)), None)
        self.assertNotIn("--", self.render())


if __name__ == "__main__":
    unittest.main()
