import unittest

import smith as sth
from smith import dtype, viz
from smith.trace import BufferSpec
from smith.viz import opprofile


@sth.jit
def counted_sum(x_pointer, out_pointer, STEPS: sth.constexpr = 8, BLOCK: sth.constexpr = 8):
    row = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    (total,) = sth.loop(STEPS, (0.0,), lambda column, acc: (acc + sth.exp(sth.load(x_pointer + row * STEPS + column)),))
    sth.store(out_pointer + row, total)


@sth.jit
def runtime_sum(x_pointer, out_pointer, columns, BLOCK: sth.constexpr = 8):
    row = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    (total,) = sth.loop(columns, (0.0,), lambda column, acc: (acc + sth.load(x_pointer + row * columns + column),))
    sth.store(out_pointer + row, total)


@sth.jit
def staged(x_pointer, out_pointer, BLOCK: sth.constexpr = 32):
    tile = sth.shared(dtype.float32, BLOCK)
    lane = sth.arange(0, BLOCK)
    sth.store(tile + lane, sth.load(x_pointer + lane))
    sth.barrier()
    sth.store(out_pointer + lane, sth.load(tile + (BLOCK - 1 - lane)))


class TestOpProfile(unittest.TestCase):
    def setUp(self):
        viz.disable()
        viz.reset()
        viz.enable()

    def tearDown(self):
        viz.disable()
        viz.reset()

    def profile(self, function, *args, **kwargs):
        function.trace(*args, **kwargs)
        return viz.program_detail(0)["op_profile"]

    def test_loop_trip_counts_scale_the_phase_cost(self):
        spec = BufferSpec(dtype.float32)
        result = self.profile(counted_sum, spec, spec, STEPS=8)
        loop = next(phase for phase in result["phases"] if phase["trips"] == 8)
        self.assertFalse(loop["estimated"])
        self.assertEqual(loop["categories"]["special math"]["count"], 8)
        self.assertEqual(loop["categories"]["global load"]["count"], 8)
        viz.reset()
        viz.enable()
        double = self.profile(counted_sum, spec, spec, STEPS=16)
        loop_double = next(phase for phase in double["phases"] if phase["trips"] == 16)
        self.assertAlmostEqual(loop_double["cost"], 2 * loop["cost"])

    def test_unknown_trip_counts_are_flagged_as_estimates(self):
        spec = BufferSpec(dtype.float32)
        result = self.profile(runtime_sum, spec, spec, 4)
        self.assertTrue(any(phase["estimated"] for phase in result["phases"]))

    def test_shared_memory_and_barriers_have_their_own_categories(self):
        spec = BufferSpec(dtype.float32)
        result = self.profile(staged, spec, spec)
        self.assertIn("shared memory", result["categories"])
        self.assertEqual(result["categories"]["barrier"]["count"], 1)
        self.assertIn("global load", result["categories"])
        self.assertIn("global store", result["categories"])

    def test_phase_costs_add_up_to_the_total(self):
        spec = BufferSpec(dtype.float32)
        result = self.profile(counted_sum, spec, spec)
        self.assertAlmostEqual(sum(phase["cost"] for phase in result["phases"]), result["total"])
        self.assertAlmostEqual(sum(entry["cost"] for entry in result["categories"].values()), result["total"])
        for phase in result["phases"]:
            self.assertIn(phase["dominant"], opprofile.CATEGORIES)

    def test_snapshot_carries_the_profile(self):
        spec = BufferSpec(dtype.float32)
        counted_sum.trace(spec, spec)
        snapshot = viz.snapshot()
        self.assertEqual(snapshot["programs"][0]["op_profile"]["total"], viz.program_detail(0)["op_profile"]["total"])


if __name__ == "__main__":
    unittest.main()
