import sys
import unittest

import smith as sth
from smith import dtype

if sys.platform == "darwin":
    from smith.compiler import Buffer


@sth.jit
def add(x_pointer, y_pointer, out_pointer, n, BLOCK: sth.constexpr = 4):
    pid = sth.program_id(0)
    offset = pid * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    x = sth.load(x_pointer + offset, mask=mask)
    y = sth.load(y_pointer + offset, mask=mask)
    sth.store(out_pointer + offset, x + y, mask=mask)


@sth.jit
def scale(x_pointer, out_pointer, alpha, n, BLOCK: sth.constexpr = 4):
    offset = sth.program_id(0) * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    x = sth.load(x_pointer + offset, mask=mask)
    sth.store(out_pointer + offset, x * alpha, mask=mask)


@sth.jit
def coordinates(out_pointer, width, BLOCK: sth.constexpr = 1):
    x = sth.program_id(0)
    y = sth.program_id(1)
    offset = y * width + x + sth.arange(0, BLOCK)
    sth.store(out_pointer + offset, x * 100 + y)


@sth.jit
def halve(out_pointer, BLOCK: sth.constexpr = 4):
    offset = sth.arange(0, BLOCK)
    sth.store(out_pointer + offset, offset / 2)


@sth.jit
def integer_halve(out_pointer, BLOCK: sth.constexpr = 4):
    offset = sth.arange(0, BLOCK)
    sth.store(out_pointer + offset, offset // 2)


@sth.jit
def branching(out_pointer, n):
    if sth.program_id(0) < n:
        sth.store(out_pointer + 0, 1)


@unittest.skipUnless(sys.platform == "darwin", "Metal requires macOS")
class TestJITMetal(unittest.TestCase):
    def test_add_not_multiple_of_block(self):
        n = 10
        x = [float(i) for i in range(n)]
        y = [10.0 * i for i in range(n)]
        out = Buffer.from_list(dtype.float32, [0.0] * n)
        grid = lambda meta: (sth.cdiv(n, meta["BLOCK"]),)
        add[grid](Buffer.from_list(dtype.float32, x), Buffer.from_list(dtype.float32, y), out, n, BLOCK=4)
        self.assertEqual(out.to_list(), [a + b for a, b in zip(x, y)])

    def test_masked_tail_untouched(self):
        out = Buffer.from_list(dtype.float32, [-1.0] * 8)
        x = Buffer.from_list(dtype.float32, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        y = Buffer.from_list(dtype.float32, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        add[(1,)](x, y, out, 3, BLOCK=8)
        self.assertEqual(out.to_list(), [2.0, 3.0, 4.0, -1.0, -1.0, -1.0, -1.0, -1.0])

    def test_float_scalar_argument(self):
        x = Buffer.from_list(dtype.float32, [float(i) for i in range(10)])
        out = Buffer.from_list(dtype.float32, [-1.0] * 12)
        scale[(3,)](x, out, 2.0, 10, BLOCK=4)
        self.assertEqual(out.to_list(), [2.0 * i for i in range(10)] + [-1.0, -1.0])

    def test_two_dimensional_grid(self):
        out = Buffer.from_list(dtype.int32, [0] * 6)
        coordinates[(3, 2)](out, 3, BLOCK=1)
        self.assertEqual(out.to_list(), [0, 100, 200, 1, 101, 201])

    def test_true_division_gives_floats(self):
        out = Buffer.from_list(dtype.float32, [0.0] * 4)
        halve[(1,)](out, BLOCK=4)
        self.assertEqual(out.to_list(), [0.0, 0.5, 1.0, 1.5])

    def test_floor_division_stays_integer(self):
        out = Buffer.from_list(dtype.int32, [0] * 4)
        integer_halve[(1,)](out, BLOCK=4)
        self.assertEqual(out.to_list(), [0, 0, 1, 1])

    def test_repeated_launch_reuses_program(self):
        x = Buffer.from_list(dtype.float32, [1.0, 2.0])
        out = Buffer.from_list(dtype.float32, [0.0, 0.0])
        scale[(1,)](x, out, 3.0, 2, BLOCK=2)
        traced = len(scale.programs)
        for _ in range(3):
            scale[(1,)](x, out, 3.0, 2, BLOCK=2)
        self.assertEqual(out.to_list(), [3.0, 6.0])
        self.assertEqual(len(scale.programs), traced)

    def test_python_branch_on_traced_value_raises(self):
        out = Buffer.from_list(dtype.int32, [0])
        with self.assertRaises(TypeError):
            branching[(1,)](out, 1)


if __name__ == "__main__":
    unittest.main()
