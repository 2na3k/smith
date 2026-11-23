import sys
import unittest

import smith as sth
from smith import dtype
from smith.trace import BufferSpec

if sys.platform == "darwin":
    from smith.compiler import Buffer


@sth.jit
def staged_copy(source_pointer, out_pointer, WIDTH: sth.constexpr = 4, BLOCK: sth.constexpr = 32):
    tile = sth.shared(dtype.float16, BLOCK * WIDTH)
    lane = sth.arange(0, BLOCK)
    sth.copy_to_shared(tile + lane * WIDTH, source_pointer + (sth.program_id(0) * BLOCK * WIDTH + lane * WIDTH), WIDTH)
    sth.barrier()
    for element in range(WIDTH):
        sth.store(out_pointer + (sth.program_id(0) * BLOCK * WIDTH + lane * WIDTH + element), sth.load(tile + (lane * WIDTH + element)))


class TestCopyToShared(unittest.TestCase):
    def test_renders_a_vector_copy_between_the_address_spaces(self):
        spec = BufferSpec(dtype.float16)
        program = staged_copy.trace(spec, spec, WIDTH=8)
        self.assertIn("*(threadgroup uint4*)(&shared_0[", program.source)
        self.assertIn("*(device const uint4*)(&source_pointer[", program.source)
        self.assertIn("source_pointer", program.loaded)

    def test_the_vector_width_follows_the_bytes_moved(self):
        spec = BufferSpec(dtype.float16)
        self.assertIn("uint2*", staged_copy.trace(spec, spec, WIDTH=4).source)
        self.assertIn("uint*", staged_copy.trace(spec, spec, WIDTH=2).source)

    def test_unsupported_widths_are_rejected(self):
        spec = BufferSpec(dtype.float16)
        with self.assertRaises(TypeError):
            staged_copy.trace(spec, spec, WIDTH=3)
        with self.assertRaises(TypeError):
            staged_copy.trace(spec, spec, WIDTH=16)

    def test_copy_requires_a_shared_destination_and_buffer_source(self):
        @sth.jit
        def wrong(source_pointer, out_pointer):
            sth.copy_to_shared(out_pointer, source_pointer, 4)

        spec = BufferSpec(dtype.float16)
        with self.assertRaises(TypeError):
            wrong.trace(spec, spec)

    @unittest.skipUnless(sys.platform == "darwin", "needs Metal")
    def test_copies_data_through_shared_memory_on_the_gpu(self):
        count = 64 * 8
        source = Buffer.from_list(dtype.float16, [float(i % 100) for i in range(count)])
        out = Buffer.from_list(dtype.float16, [0.0] * count)
        staged_copy[(2,)](source, out, WIDTH=8, BLOCK=32)
        self.assertEqual(out.to_list(), source.to_list())


if __name__ == "__main__":
    unittest.main()
