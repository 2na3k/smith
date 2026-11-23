import sys
import unittest

import smith as sth
from smith import dtype

if sys.platform == "darwin":
    from smith import compiler
    from smith.compiler import Buffer


@sth.jit
def scale(x_pointer, out_pointer, alpha, n, BLOCK: sth.constexpr = 64):
    offset = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    sth.store(out_pointer + offset, sth.load(x_pointer + offset, mask=mask) * alpha, mask=mask)


@sth.jit
def add(x_pointer, y_pointer, out_pointer, n, BLOCK: sth.constexpr = 64):
    offset = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    sth.store(out_pointer + offset, sth.load(x_pointer + offset, mask=mask) + sth.load(y_pointer + offset, mask=mask), mask=mask)


def grid_for(n):
    return lambda meta: (-(-n // meta["BLOCK"]),)


def floats(n, value):
    return Buffer.from_list(dtype.float32, [value] * n)


@unittest.skipUnless(sys.platform == "darwin", "Metal requires macOS")
class TestBatch(unittest.TestCase):
    n = 1000

    def test_dependent_kernels_run_in_order(self):
        x, a, b, c = floats(self.n, 1.0), floats(self.n, 0.0), floats(self.n, 0.0), floats(self.n, 0.0)
        with sth.batch():
            scale[grid_for(self.n)](x, a, 2.0, self.n)
            scale[grid_for(self.n)](a, b, 3.0, self.n)
            scale[grid_for(self.n)](b, c, 5.0, self.n)
        self.assertEqual(c.to_list(), [30.0] * self.n)

    def test_changing_buffers_and_scalars_between_launches_are_rebound(self):
        x = floats(self.n, 1.0)
        outputs = [floats(self.n, 0.0) for _ in range(4)]
        with sth.batch():
            for index, out in enumerate(outputs):
                scale[grid_for(self.n)](x, out, float(index + 2), self.n)
        for index, out in enumerate(outputs):
            self.assertEqual(out.to_list(), [float(index + 2)] * self.n, index)

    def test_same_arguments_repeated_still_run_every_time(self):
        x, out = floats(self.n, 1.0), floats(self.n, 0.0)
        counter = floats(self.n, 0.0)
        with sth.batch():
            for _ in range(5):
                add[grid_for(self.n)](counter, x, counter, self.n)
        self.assertEqual(counter.to_list(), [5.0] * self.n)

    def test_different_kernels_alternating_in_one_batch(self):
        x, y, s, total = floats(self.n, 2.0), floats(self.n, 3.0), floats(self.n, 0.0), floats(self.n, 0.0)
        with sth.batch():
            add[grid_for(self.n)](x, y, s, self.n)
            scale[grid_for(self.n)](s, total, 10.0, self.n)
            add[grid_for(self.n)](total, x, total, self.n)
        self.assertEqual(total.to_list(), [52.0] * self.n)

    def test_launches_in_a_batch_share_one_command_buffer(self):
        x, out = floats(self.n, 1.0), floats(self.n, 0.0)
        with sth.batch():
            events = [scale[grid_for(self.n)](x, out, 2.0, self.n) for _ in range(6)]
            self.assertTrue(all(event.submission is events[0].submission for event in events))
            self.assertFalse(events[0].submission.committed)
            self.assertTrue(all(event.submitted is None for event in events))
        self.assertTrue(events[0].submission.committed)
        self.assertTrue(all(event.submitted is not None for event in events))
        self.assertTrue(all(event.batched for event in events))
        sth.synchronize()

    def test_unbatched_launches_each_get_their_own_command_buffer(self):
        x, out = floats(self.n, 1.0), floats(self.n, 0.0)
        first = scale[grid_for(self.n)](x, out, 2.0, self.n)
        second = scale[grid_for(self.n)](x, out, 2.0, self.n)
        self.assertIsNot(first.submission, second.submission)
        self.assertFalse(first.batched)
        sth.synchronize()

    def test_reading_a_buffer_inside_a_batch_flushes_it(self):
        x, out = floats(self.n, 1.0), floats(self.n, 0.0)
        with sth.batch():
            scale[grid_for(self.n)](x, out, 4.0, self.n)
            self.assertEqual(out.to_list(), [4.0] * self.n)
            scale[grid_for(self.n)](out, out, 2.0, self.n)
        self.assertEqual(out.to_list(), [8.0] * self.n)

    def test_waiting_on_a_launch_inside_a_batch_flushes_and_continues(self):
        x, a, b = floats(self.n, 1.0), floats(self.n, 0.0), floats(self.n, 0.0)
        with sth.batch():
            first = scale[grid_for(self.n)](x, a, 2.0, self.n)
            first.wait()
            self.assertTrue(first.submission.committed)
            second = scale[grid_for(self.n)](a, b, 3.0, self.n)
            self.assertIsNot(first.submission, second.submission)
        self.assertEqual(b.to_list(), [6.0] * self.n)

    def test_synchronize_inside_a_batch_flushes_it(self):
        x, out = floats(self.n, 1.0), floats(self.n, 0.0)
        with sth.batch():
            event = scale[grid_for(self.n)](x, out, 3.0, self.n)
            sth.synchronize()
            self.assertTrue(event.done)
        self.assertEqual(out.to_list(), [3.0] * self.n)

    def test_batches_cannot_be_nested(self):
        with sth.batch():
            with self.assertRaises(RuntimeError):
                with sth.batch():
                    pass

    def test_work_encoded_before_an_exception_still_runs(self):
        x, out = floats(self.n, 1.0), floats(self.n, 0.0)
        with self.assertRaises(ValueError):
            with sth.batch():
                scale[grid_for(self.n)](x, out, 7.0, self.n)
                raise ValueError("stop")
        self.assertEqual(out.to_list(), [7.0] * self.n)

    def test_a_batch_is_usable_again_after_it_ends(self):
        x, out = floats(self.n, 1.0), floats(self.n, 0.0)
        for factor in (2.0, 3.0):
            with sth.batch():
                scale[grid_for(self.n)](x, out, factor, self.n)
            self.assertEqual(out.to_list(), [factor] * self.n)

    def test_gpu_time_of_a_batch_is_split_across_its_kernels(self):
        x, out = floats(self.n, 1.0), floats(self.n, 0.0)
        with sth.batch():
            events = [scale[grid_for(self.n)](x, out, 2.0, self.n) for _ in range(4)]
        start, end = events[0].gpu_interval
        self.assertAlmostEqual(events[0].gpu_time * 4, end - start)
        self.assertTrue(all(event.gpu_interval == events[0].gpu_interval for event in events))

    def test_observers_see_every_launch(self):
        seen = []
        compiler.add_launch_observer(seen.append)
        try:
            x, out = floats(self.n, 1.0), floats(self.n, 0.0)
            with sth.batch():
                for _ in range(3):
                    scale[grid_for(self.n)](x, out, 2.0, self.n)
            sth.synchronize()
        finally:
            compiler.remove_launch_observer(seen.append)
        self.assertEqual(len(seen), 3)

    def test_many_launches_exceeding_the_queue_depth(self):
        x, out = floats(self.n, 1.0), floats(self.n, 0.0)
        for _ in range(3):
            with sth.batch():
                for _ in range(200):
                    add[grid_for(self.n)](out, x, out, self.n)
        self.assertEqual(out.to_list(), [600.0] * self.n)


class TestPortable(unittest.TestCase):
    def test_batch_is_a_no_op_for_python_lists(self):
        x, out = [1.0, 2.0, 3.0, 4.0], [0.0] * 4
        with sth.batch():
            scale[(1,)](x, out, 2.0, 4, BLOCK=4)
        self.assertEqual(out, [2.0, 4.0, 6.0, 8.0])


if __name__ == "__main__":
    unittest.main()
