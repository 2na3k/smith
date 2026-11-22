import os
import tempfile
import unittest

import smith as sth
from smith import dtype
from smith.layout import analyze, ascii_map, element_map, evaluate, graph_to_dot, layout_html
from smith.trace import BufferSpec


@sth.jit
def add(x_pointer, y_pointer, out_pointer, n, BLOCK: sth.constexpr = 4):
    offset = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    x = sth.load(x_pointer + offset, mask=mask)
    y = sth.load(y_pointer + offset, mask=mask)
    sth.store(out_pointer + offset, x + y, mask=mask)


@sth.jit
def strided(out_pointer, BLOCK: sth.constexpr = 4):
    offset = sth.program_id(0) * BLOCK * 2 + sth.arange(0, BLOCK) * 2
    sth.store(out_pointer + offset, 1.0)


@sth.jit
def irregular(out_pointer, BLOCK: sth.constexpr = 4):
    offset = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK) * sth.arange(0, BLOCK)
    sth.store(out_pointer + offset, 1.0)


@sth.jit
def collide(out_pointer, BLOCK: sth.constexpr = 4):
    sth.store(out_pointer + sth.arange(0, BLOCK), 1.0)


@sth.jit
def copy_unmasked(x_pointer, out_pointer, BLOCK: sth.constexpr = 4):
    offset = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    sth.store(out_pointer + offset, sth.load(x_pointer + offset))


@sth.jit
def write_head(out_pointer, BLOCK: sth.constexpr = 4):
    offset = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    mask = offset < 4
    sth.store(out_pointer + offset, 1.0, mask=mask)


@sth.jit
def gather(x_pointer, index_pointer, out_pointer, BLOCK: sth.constexpr = 4):
    offset = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    source = sth.load(index_pointer + offset)
    sth.store(out_pointer + offset, sth.load(x_pointer + source))


def add_program(n=10, block=4):
    spec = BufferSpec(dtype.float32)
    return add.trace(spec, spec, spec, n, BLOCK=block)


class TestEvaluate(unittest.TestCase):
    def test_access_counts_and_masks(self):
        evaluation = evaluate(add_program(), (3,), {"n": 10}, {"x_pointer": 10, "y_pointer": 10, "out_pointer": 10})
        self.assertEqual(len(evaluation.accesses), 36)
        stores = [access for access in evaluation.accesses if access.kind == "store"]
        self.assertEqual(len(stores), 12)
        self.assertEqual(sum(1 for access in stores if access.active), 10)
        self.assertEqual(stores[4].program, (1, 0, 0))
        self.assertEqual(stores[4].lane, 0)
        self.assertEqual(stores[4].offset, 4)

    def test_values_follow_buffer_contents(self):
        contents = {"x_pointer": [float(i) for i in range(10)], "y_pointer": [10.0 * i for i in range(10)],
                    "out_pointer": [0.0] * 10}
        evaluation = evaluate(add_program(), (3,), {"n": 10}, contents)
        values = [access.value for access in evaluation.accesses if access.kind == "store"][:10]
        self.assertEqual(values, [11.0 * i for i in range(10)])

    def test_missing_scalar_is_an_error(self):
        with self.assertRaises(ValueError):
            evaluate(add_program(), (3,), {}, {})

    def test_indirect_offsets_need_contents(self):
        spec = BufferSpec(dtype.float32)
        program = gather.trace(spec, BufferSpec(dtype.int32), spec, BLOCK=4)
        known = evaluate(program, (2,), None, {"x_pointer": [10.0 * i for i in range(8)],
                                                 "index_pointer": [3, 2, 1, 0, 7, 6, 5, 4],
                                                 "out_pointer": 8})
        stores = [access for access in known.accesses if access.kind == "store"]
        self.assertEqual([access.value for access in stores], [30.0, 20.0, 10.0, 0.0, 70.0, 60.0, 50.0, 40.0])
        unknown = evaluate(program, (2,), None, {"index_pointer": 8, "out_pointer": 8, "x_pointer": 8})
        loads = [access for access in unknown.accesses if access.buffer == "x_pointer"]
        self.assertTrue(loads and all(access.offset is None for access in loads))


class TestAnalyze(unittest.TestCase):
    def test_add_is_coalesced_and_clean(self):
        evaluation = evaluate(add_program(), (3,), {"n": 10}, {"x_pointer": 10, "y_pointer": 10, "out_pointer": 10})
        analysis = analyze(evaluation)
        self.assertTrue(analysis.clean)
        self.assertEqual(len(analysis.summaries), 3)
        self.assertTrue(all(summary.lane_stride == 1 and summary.coalesced for summary in analysis.summaries))

    def test_strided_access_warns(self):
        spec = BufferSpec(dtype.float32)
        evaluation = evaluate(strided.trace(spec, BLOCK=4), (2,), None, {"out_pointer": 16})
        analysis = analyze(evaluation)
        self.assertEqual([warning.kind for warning in analysis.warnings], ["strided"])
        self.assertEqual(analysis.warnings[0].details["stride"], 2)
        self.assertFalse(analysis.summaries[0].coalesced)

    def test_irregular_access_warns(self):
        spec = BufferSpec(dtype.float32)
        evaluation = evaluate(irregular.trace(spec, BLOCK=4), (1,), None, {"out_pointer": 16})
        kinds = [warning.kind for warning in analyze(evaluation).warnings]
        self.assertIn("irregular_stride", kinds)

    def test_overlapping_writes_warn(self):
        spec = BufferSpec(dtype.float32)
        evaluation = evaluate(collide.trace(spec, BLOCK=4), (3,), None, {"out_pointer": 4})
        warnings = [warning for warning in analyze(evaluation).warnings if warning.kind == "overlapping_writes"]
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].details["count"], 4)
        self.assertEqual(warnings[0].details["max_writers"], 3)

    def test_out_of_range_offsets_warn(self):
        spec = BufferSpec(dtype.float32)
        evaluation = evaluate(copy_unmasked.trace(spec, spec, BLOCK=4), (3,), None, {"x_pointer": 12, "out_pointer": 10})
        warnings = analyze(evaluation).warnings
        self.assertEqual([warning.kind for warning in warnings], ["out_of_range"])
        self.assertEqual(warnings[0].buffer, "out_pointer")
        self.assertEqual(warnings[0].details["count"], 2)
        self.assertEqual(warnings[0].details["max_offset"], 11)


class TestLayout(unittest.TestCase):
    def test_element_map_tracks_owning_programs(self):
        evaluation = evaluate(add_program(), (3,), {"n": 10}, {"x_pointer": 10, "y_pointer": 10, "out_pointer": 10})
        emap = element_map(evaluation, "out_pointer")
        self.assertEqual(emap.length, 10)
        self.assertEqual([programs[0] for programs in emap.programs], [0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
        self.assertEqual(emap.threads, [1] * 10)

    def test_ascii_map_shows_programs_and_untouched(self):
        evaluation = evaluate(add_program(), (3,), {"n": 10}, {"x_pointer": 10, "y_pointer": 10, "out_pointer": 10})
        text = ascii_map(evaluation, "out_pointer")
        self.assertIn("0 | 0000111122", text)
        self.assertIn("untouched 0, shared 0", text)

    def test_ascii_map_marks_overlap(self):
        spec = BufferSpec(dtype.float32)
        evaluation = evaluate(collide.trace(spec, BLOCK=4), (3,), None, {"out_pointer": 4})
        text = ascii_map(evaluation, "out_pointer")
        self.assertIn("****", text)
        self.assertIn("3333", text)
        self.assertIn("written by several threads 4", text)

    def test_untouched_elements_are_reported(self):
        spec = BufferSpec(dtype.float32)
        evaluation = evaluate(write_head.trace(spec, BLOCK=4), (2,), None, {"out_pointer": 8})
        self.assertEqual(element_map(evaluation, "out_pointer").threads, [1, 1, 1, 1, 0, 0, 0, 0])
        self.assertIn("untouched 4, shared 0", ascii_map(evaluation, "out_pointer"))

    def test_html_is_standalone_svg(self):
        spec = BufferSpec(dtype.float32)
        evaluation = evaluate(collide.trace(spec, BLOCK=4), (3,), None, {"out_pointer": 4})
        document = layout_html(evaluation, "out_pointer")
        self.assertIn("<svg", document)
        self.assertIn("<rect", document)
        self.assertIn("#d62728", document)
        for forbidden in ("<script", "<link", "src=", "href=", "@import", "url("):
            self.assertNotIn(forbidden, document)

    def test_html_is_written_to_path(self):
        evaluation = evaluate(add_program(), (3,), {"n": 10}, {"x_pointer": 10, "y_pointer": 10, "out_pointer": 10})
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "layout.html")
            layout_html(evaluation, "out_pointer", path=path)
            with open(path, encoding="utf-8") as handle:
                self.assertTrue(handle.read().startswith("<!DOCTYPE html>"))

    def test_graph_to_dot(self):
        text = graph_to_dot(add_program())
        self.assertTrue(text.startswith("digraph kernel {"))
        self.assertIn(" -> ", text)
        self.assertIn("store", text)
        self.assertTrue(text.rstrip().endswith("}"))


if __name__ == "__main__":
    unittest.main()
