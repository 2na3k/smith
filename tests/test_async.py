import sys
import unittest

import smith as sth
from smith import dtype
from smith.trace import BufferSpec

if sys.platform == "darwin":
    from smith import compiler
    from smith.compiler import Buffer


@sth.jit
def scale(x_pointer, out_pointer, alpha, n, BLOCK: sth.constexpr = 64):
    offset = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    x = sth.load(x_pointer + offset, mask=mask)
    sth.store(out_pointer + offset, x * alpha, mask=mask)


@unittest.skipUnless(sys.platform == "darwin", "Metal requires macOS")
class TestAsync(unittest.TestCase):
    def test_launch_returns_an_event(self):
        n = 256
        x = Buffer.from_list(dtype.float32, [1.0] * n)
        out = Buffer.from_list(dtype.float32, [0.0] * n)
        event = scale[(n // 64,)](x, out, 2.0, n, BLOCK=64)
        self.assertIsInstance(event, compiler.Launch)
        event.wait()
        self.assertTrue(event.done)
        self.assertGreaterEqual(event.gpu_time, 0.0)
        self.assertEqual(out.to_list(), [2.0] * n)

    def test_dependent_launches_without_waiting(self):
        n = 4096
        x = Buffer.from_list(dtype.float32, [float(i) for i in range(n)])
        middle = Buffer.from_list(dtype.float32, [0.0] * n)
        out = Buffer.from_list(dtype.float32, [0.0] * n)
        for _ in range(20):
            scale[(n // 64,)](x, middle, 2.0, n, BLOCK=64)
            scale[(n // 64,)](middle, out, 3.0, n, BLOCK=64)
        sth.synchronize()
        self.assertEqual(out.to_list(), [6.0 * i for i in range(n)])

    def test_to_list_synchronizes(self):
        n = 1024
        x = Buffer.from_list(dtype.float32, [1.0] * n)
        out = Buffer.from_list(dtype.float32, [0.0] * n)
        for _ in range(50):
            scale[(n // 64,)](x, out, 5.0, n, BLOCK=64)
        self.assertEqual(out.to_list(), [5.0] * n)

    def test_observer_sees_launches(self):
        seen = []
        compiler.add_launch_observer(seen.append)
        try:
            x = Buffer.from_list(dtype.float32, [1.0] * 64)
            out = Buffer.from_list(dtype.float32, [0.0] * 64)
            scale[(1,)](x, out, 2.0, 64, BLOCK=64)
            sth.synchronize()
        finally:
            compiler.remove_launch_observer(seen.append)
        self.assertEqual([event.kernel.name for event in seen], ["scale"])


class TestTraceApi(unittest.TestCase):
    def test_trace_without_allocating_buffers(self):
        program = scale.trace(BufferSpec(dtype.float32), BufferSpec(dtype.float32), 2.0, 128, BLOCK=64)
        self.assertEqual(program.lanes, 64)
        self.assertEqual(list(program.buffers), ["x_pointer", "out_pointer"])
        self.assertEqual(list(program.scalars), ["alpha", "n"])
        self.assertEqual(program.loaded, {"x_pointer"})
        self.assertEqual(program.stored, {"out_pointer"})
        self.assertIn("kernel void scale(", program.source)


if __name__ == "__main__":
    unittest.main()
