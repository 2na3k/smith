import sys
import unittest

from smith import dtype

if sys.platform == "darwin":
    from smith.compiler import CompileError, compile_msl

ADD_ONE = """
#include <metal_stdlib>
using namespace metal;
kernel void add_one(device float* x [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    x[gid] = x[gid] + 1.0f;
}
"""


@unittest.skipUnless(sys.platform == "darwin", "Metal requires macOS")
class TestCompiler(unittest.TestCase):
    def test_compile_and_cache(self):
        k = compile_msl(ADD_ONE, "add_one")
        self.assertTrue(k.pipeline)
        self.assertIs(k, compile_msl(ADD_ONE, "add_one"))

    def test_bad_source_raises(self):
        with self.assertRaises(CompileError):
            compile_msl("kernel void broken( {", "broken")

    def test_missing_entry_raises(self):
        with self.assertRaises(CompileError):
            compile_msl(ADD_ONE, "does_not_exist")

    def test_dtype_spelling(self):
        self.assertEqual(dtype.float32.msl, "float")
        self.assertEqual(dtype.float32.size, 4)


if __name__ == "__main__":
    unittest.main()
