import sys
import unittest

if sys.platform == "darwin":
    from smith.kernels import add


@unittest.skipUnless(sys.platform == "darwin", "Metal requires macOS")
class TestAddMetal(unittest.TestCase):
    def test_add(self):
        x = [float(i) for i in range(1000)]
        y = [10.0 * i for i in range(1000)]
        out = add(x, y)
        self.assertEqual(out, [a + b for a, b in zip(x, y)])

    def test_add_small(self):
        self.assertEqual(add([1.5, 2.0], [0.25, -1.0]), [1.75, 1.0])


if __name__ == "__main__":
    unittest.main()
