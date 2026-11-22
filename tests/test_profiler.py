import json
import sys
import types
import unittest

import smith as sth
from smith import dtype
from smith.profiler import LaunchRecord, Profiler, estimate_bytes
from smith.trace import BufferSpec

if sys.platform == "darwin":
    from smith.compiler import Buffer


@sth.jit
def add(x_pointer, y_pointer, out_pointer, n, BLOCK: sth.constexpr = 4):
    offset = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    x = sth.load(x_pointer + offset, mask=mask)
    y = sth.load(y_pointer + offset, mask=mask)
    sth.store(out_pointer + offset, x + y, mask=mask)


@sth.jit
def scale_twice(x_pointer, out_pointer, n, BLOCK: sth.constexpr = 4):
    offset = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    sth.store(out_pointer + offset, sth.load(x_pointer + offset, mask=mask) * 2.0, mask=mask)


def add_program(n=10, block=4):
    spec = BufferSpec(dtype.float32)
    return add.trace(spec, spec, spec, n, BLOCK=block)


def fake_record(name, gpu_seconds, bytes_moved, exact=True):
    return LaunchRecord(name, (12, 1, 1), (4, 1, 1), 0.0, types.SimpleNamespace(gpu_time=gpu_seconds),
                        bytes_moved, exact)


class TestEstimateBytes(unittest.TestCase):
    def test_upper_bound_counts_every_thread(self):
        total, exact = estimate_bytes(add_program(), (12, 1, 1))
        self.assertEqual(total, 12 * 3 * 4)
        self.assertFalse(exact)

    def test_exact_counts_only_active_threads(self):
        total, exact = estimate_bytes(add_program(), (12, 1, 1), {"n": 10})
        self.assertEqual(total, 10 * 3 * 4)
        self.assertTrue(exact)

    def test_missing_scalar_raises(self):
        with self.assertRaises(ValueError):
            estimate_bytes(add_program(), (12, 1, 1), {})


class TestReport(unittest.TestCase):
    def setUp(self):
        self.profiler = Profiler()
        self.profiler.records.extend([
            fake_record("add", 0.001, 120),
            fake_record("add", 0.003, 120),
            fake_record("double", 0.002, None),
        ])

    def test_summary_per_kernel(self):
        rows = {row["name"]: row for row in self.profiler.summary()}
        self.assertEqual(rows["add"]["launches"], 2)
        self.assertAlmostEqual(rows["add"]["total_seconds"], 0.004)
        self.assertAlmostEqual(rows["add"]["mean_seconds"], 0.002)
        self.assertAlmostEqual(rows["add"]["min_seconds"], 0.001)
        self.assertAlmostEqual(rows["add"]["max_seconds"], 0.003)
        self.assertEqual(rows["add"]["bytes_moved"], 240)
        self.assertAlmostEqual(rows["add"]["gb_per_s"], 240 / 0.004 / 1e9)
        self.assertIsNone(rows["double"]["bytes_moved"])
        self.assertIsNone(rows["double"]["gb_per_s"])

    def test_report_is_a_table(self):
        text = self.profiler.report()
        self.assertIn("kernel", text.splitlines()[0])
        self.assertIn("add", text)
        self.assertIn("double", text)

    def test_empty_report(self):
        self.assertEqual(Profiler().report(), "no launches recorded")

    def test_to_dict_is_json(self):
        data = json.loads(json.dumps(self.profiler.to_dict()))
        self.assertEqual(len(data["launches"]), 3)
        self.assertEqual(data["launches"][0]["grid"], [12, 1, 1])
        self.assertEqual({row["name"] for row in data["kernels"]}, {"add", "double"})


@unittest.skipIf(sys.platform == "darwin", "only checks the non-Metal platform error")
class TestProfilerWithoutMetal(unittest.TestCase):
    def test_entering_requires_macos(self):
        with self.assertRaises(RuntimeError):
            with Profiler():
                pass


@unittest.skipUnless(sys.platform == "darwin", "Metal requires macOS")
class TestProfilerMetal(unittest.TestCase):
    def test_records_tracked_launches_with_exact_bytes(self):
        n = 10
        x = Buffer.from_list(dtype.float32, [float(i) for i in range(n)])
        y = Buffer.from_list(dtype.float32, [10.0 * i for i in range(n)])
        out = Buffer.from_list(dtype.float32, [0.0] * n)
        program = add.trace(x, y, out, n, BLOCK=4)
        grid = (sth.cdiv(n, 4),)
        with Profiler().track(program, {"n": n}) as profiler:
            add[grid](x, y, out, n, BLOCK=4)
            add[grid](x, y, out, n, BLOCK=4)
        self.assertEqual(len(profiler.records), 2)
        self.assertEqual([record.name for record in profiler.records], ["add", "add"])
        self.assertEqual(profiler.records[0].bytes_moved, 120)
        self.assertTrue(profiler.records[0].bytes_exact)
        self.assertGreaterEqual(profiler.records[0].gpu_time, 0.0)
        self.assertEqual(out.to_list(), [11.0 * i for i in range(n)])
        self.assertIn("add", profiler.report())
        json.dumps(profiler.to_dict())

    def test_untracked_kernels_have_no_byte_estimate(self):
        n = 8
        x = Buffer.from_list(dtype.float32, [1.0] * n)
        out = Buffer.from_list(dtype.float32, [0.0] * n)
        with Profiler() as profiler:
            scale_twice[(2,)](x, out, n, BLOCK=4)
        self.assertEqual(len(profiler.records), 1)
        self.assertEqual(profiler.records[0].name, "scale_twice")
        self.assertIsNone(profiler.records[0].bytes_moved)
        self.assertEqual(out.to_list(), [2.0] * n)


if __name__ == "__main__":
    unittest.main()
