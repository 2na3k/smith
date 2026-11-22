import unittest

import smith as sth


@sth.jit
def add(x_pointer, y_pointer, out_pointer, n, BLOCK: sth.constexpr = 4):
    pid = sth.program_id(0)
    offset = pid * BLOCK + sth.arange(0, BLOCK)
    mask = offset < n
    x = sth.load(x_pointer + offset, mask=mask)
    y = sth.load(y_pointer + offset, mask=mask)
    sth.store(out_pointer + offset, x + y, mask=mask)


class TestAdd(unittest.TestCase):
    def test_add_not_multiple_of_block(self):
        n = 10
        x = list(range(n))
        y = [10 * i for i in range(n)]
        out = [0] * n
        grid = lambda meta: (sth.cdiv(n, meta["BLOCK"]),)
        add[grid](x, y, out, n, BLOCK=4)
        self.assertEqual(out, [a + b for a, b in zip(x, y)])

    def test_masked_tail_untouched(self):
        n = 3
        out = [-1] * 8
        add[(1,)](
            [1, 2, 3, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            out, n, BLOCK=8,
        )
        self.assertEqual(out, [2, 3, 4, -1, -1, -1, -1, -1])


if __name__ == "__main__":
    unittest.main()
