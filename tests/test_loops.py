import math
import sys
import unittest

import smith as sth
from smith import dtype
from smith.ir import Graph, format_graph, walk
from smith.layout import evaluate
from smith.passes import simplify
from smith.trace import BufferSpec

if sys.platform == "darwin":
    from smith.compiler import Buffer


@sth.jit
def row_sum(x_pointer, out_pointer, rows, columns, BLOCK: sth.constexpr = 8):
    row = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    mask = row < rows
    (total,) = sth.loop(columns, (0.0,), lambda column, acc: (acc + sth.load(x_pointer + row * columns + column, mask=mask),))
    sth.store(out_pointer + row, total, mask=mask)


@sth.jit
def fibonacci(out_pointer, steps, BLOCK: sth.constexpr = 4):
    lane = sth.arange(0, BLOCK)
    a, b = sth.loop(steps, (0, 1), lambda i, a, b: (b, a + b))
    sth.store(out_pointer + lane, a + lane)


@sth.jit
def nested(out_pointer, BLOCK: sth.constexpr = 4):
    lane = sth.arange(0, BLOCK)
    (outer,) = sth.loop(3, (0,), lambda i, total: sth.loop(4, (total,), lambda j, inner: (inner + i * 10 + j + lane,)))
    sth.store(out_pointer + lane, outer)


@sth.jit
def store_in_loop(out_pointer, n, BLOCK: sth.constexpr = 4):
    lane = sth.arange(0, BLOCK)

    def body(step, marker):
        sth.store(out_pointer + step * BLOCK + lane, marker + step)
        return (marker + 1,)

    sth.loop(n, (100,), body)


@sth.jit
def per_lane_trip_count(out_pointer, BLOCK: sth.constexpr = 8):
    lane = sth.arange(0, BLOCK)
    (count,) = sth.loop(lane, (0,), lambda i, total: (total + 1,))
    sth.store(out_pointer + lane, count)


@sth.jit
def math_pieces(x_pointer, out_pointer, n, BLOCK: sth.constexpr = 8):
    offset = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    x = sth.load(x_pointer + offset, mask=mask)
    y = sth.where(x > 0.0, sth.exp(sth.minimum(x, 2.0)), sth.maximum(x, -1.0))
    sth.store(out_pointer + offset, y + sth.sqrt(sth.cast(offset, dtype.float32)) + sth.log(sth.exp(x)), mask=mask)


@sth.jit
def butterfly(x_pointer, out_pointer, BLOCK: sth.constexpr = 8):
    lane = sth.arange(0, BLOCK)
    value = sth.load(x_pointer + lane)
    value = value + sth.shuffle_xor(value, 1)
    value = value + sth.shuffle_xor(value, 2)
    value = value + sth.shuffle_xor(value, 4)
    sth.store(out_pointer + lane, value)


class TestInterpreter(unittest.TestCase):
    def test_row_sum(self):
        data = [float(i % 5) for i in range(70)]
        out = [0.0] * 10
        row_sum[(2,)](data, out, 10, 7, BLOCK=8)
        self.assertEqual(out, [sum(data[r * 7:(r + 1) * 7]) for r in range(10)])

    def test_carried_values_swap(self):
        out = [0] * 4
        fibonacci[(1,)](out, 10, BLOCK=4)
        self.assertEqual(out, [55, 56, 57, 58])

    def test_nested_loops(self):
        out = [0] * 4
        nested[(1,)](out, BLOCK=4)
        self.assertEqual(out, [138, 150, 162, 174])

    def test_per_lane_trip_count(self):
        out = [0] * 8
        per_lane_trip_count[(1,)](out, BLOCK=8)
        self.assertEqual(out, list(range(8)))

    def test_butterfly_sums_groups(self):
        out = [0.0] * 8
        butterfly[(1,)]([float(i) for i in range(8)], out, BLOCK=8)
        self.assertEqual(out, [28.0] * 8)

    def test_math_ops(self):
        xs = [-2.0, -0.5, 0.0, 0.5, 1.0, 3.0, 5.0, -3.0]
        out = [0.0] * 8
        math_pieces[(1,)](xs, out, 8, BLOCK=8)
        expected = [(math.exp(min(x, 2.0)) if x > 0 else max(x, -1.0)) + math.sqrt(i) + x for i, x in enumerate(xs)]
        for value, reference in zip(out, expected):
            self.assertAlmostEqual(value, reference, places=9)


class TestGraph(unittest.TestCase):
    def test_loop_buffers_are_reported(self):
        program = row_sum.trace(BufferSpec(dtype.float32), BufferSpec(dtype.float32), 10, 7, BLOCK=8)
        self.assertEqual(program.loaded, {"x_pointer"})
        self.assertEqual(program.stored, {"out_pointer"})

    def test_loop_renders_a_for_statement_with_carried_values(self):
        source = row_sum.trace(BufferSpec(dtype.float32), BufferSpec(dtype.float32), 10, 7, BLOCK=8).source
        self.assertIn("for (int index_", source)
        self.assertIn("carried_", source)

    def test_values_from_outside_are_captured(self):
        program = row_sum.trace(BufferSpec(dtype.float32), BufferSpec(dtype.float32), 10, 7, BLOCK=8)
        loops = [node for node in program.graph.nodes if node.op == "loop"]
        self.assertEqual(len(loops), 1)
        self.assertGreater(len(loops[0].arg.captures), 0)
        self.assertTrue(any(node.op == "load" for node in walk(program.graph)))

    def test_constants_are_folded_inside_loop_bodies(self):
        graph = Graph()
        out = graph.buffer("out", dtype.int32)
        lane = graph.lane()

        from smith.ir import LoopBody
        child = Graph(parent=graph)
        index = child.node("loop_index", dtype=dtype.int32)
        carried = child.node("carried", arg=0, dtype=dtype.int32)
        doubled = child.binary("add", child.binary("mul", index, child.const(1)), child.const(0))
        output = child.binary("add", carried, doubled)
        (total,) = graph.loop(graph.const(0), graph.const(5), [graph.const(0)], LoopBody(child, index, [carried], [output]))
        graph.store(out, lane, total, None)
        text = format_graph(simplify(graph))
        loop = next(node for node in simplify(graph).nodes if node.op == "loop")
        inner = format_graph_all(loop.arg.graph)
        self.assertNotIn(" mul", inner)
        self.assertEqual(inner.count(" add"), 1)
        self.assertIn("loop", text)

    def test_layout_analysis_rejects_loops(self):
        program = row_sum.trace(BufferSpec(dtype.float32), BufferSpec(dtype.float32), 10, 7, BLOCK=8)
        with self.assertRaises(NotImplementedError):
            evaluate(program, (1,), {"rows": 10, "columns": 7})

    def test_body_arity_mismatch_is_rejected(self):
        @sth.jit
        def broken(out_pointer):
            sth.loop(3, (0, 1), lambda i, a, b: (a,))

        with self.assertRaises(ValueError):
            broken.trace(BufferSpec(dtype.int32))

    def test_bitwise_shuffle_keeps_dtype(self):
        graph = Graph()
        value = graph.cast(graph.const(1.0), dtype.float16)
        self.assertEqual(graph.shuffle_xor(value, 1).dtype, dtype.float16)


def format_graph_all(graph):
    return "\n".join(f"%{node.index} = {node.op}{'' if node.arg is None else f'[{node.arg}]'}" for node in graph.nodes)


@unittest.skipUnless(sys.platform == "darwin", "Metal requires macOS")
class TestMetal(unittest.TestCase):
    def test_row_sum_with_runtime_trip_count(self):
        data = [float(i % 5) for i in range(70)]
        out = Buffer.from_list(dtype.float32, [0.0] * 10)
        row_sum[(2,)](Buffer.from_list(dtype.float32, data), out, 10, 7, BLOCK=8)
        self.assertEqual(out.to_list(), [sum(data[r * 7:(r + 1) * 7]) for r in range(10)])

    def test_swap_nested_and_per_lane_loops_match_interpreter(self):
        for kernel, arguments, size, kind in ((fibonacci, (10,), 4, dtype.int32), (nested, (), 4, dtype.int32),
                                              (per_lane_trip_count, (), 8, dtype.int32)):
            pure = [0] * size
            kernel[(1,)](pure, *arguments, BLOCK=size)
            out = Buffer.from_list(kind, [0] * size)
            kernel[(1,)](out, *arguments, BLOCK=size)
            self.assertEqual(out.to_list(), pure, kernel.__name__)

    def test_stores_inside_a_loop(self):
        out = Buffer.from_list(dtype.int32, [0] * 12)
        store_in_loop[(1,)](out, 3, BLOCK=4)
        self.assertEqual(out.to_list(), [100] * 4 + [102] * 4 + [104] * 4)

    def test_math_ops_match_python(self):
        xs = [-2.0, -0.5, 0.0, 0.5, 1.0, 3.0, 5.0, -3.0]
        out = Buffer.from_list(dtype.float32, [0.0] * 8)
        math_pieces[(1,)](Buffer.from_list(dtype.float32, xs), out, 8, BLOCK=8)
        expected = [(math.exp(min(x, 2.0)) if x > 0 else max(x, -1.0)) + math.sqrt(i) + x for i, x in enumerate(xs)]
        for value, reference in zip(out.to_list(), expected):
            self.assertAlmostEqual(value, reference, places=4)

    def test_butterfly_shuffle(self):
        out = Buffer.from_list(dtype.float32, [0.0] * 8)
        butterfly[(1,)](Buffer.from_list(dtype.float32, [float(i) for i in range(8)]), out, BLOCK=8)
        self.assertEqual(out.to_list(), [28.0] * 8)


if __name__ == "__main__":
    unittest.main()
