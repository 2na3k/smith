from ..jit import constexpr, jit
from ..ops import arange, load, program_id, store


@jit
def dequantize_kernel(packed_pointer, scales_pointer, biases_pointer, out_pointer, n,
                      BITS: constexpr = 4, GROUP: constexpr = 64, BLOCK: constexpr = 64):
    per_word = 32 // BITS
    offset = program_id(0) * BLOCK + arange(0, BLOCK)
    mask = offset < n
    word = load(packed_pointer + offset // per_word, mask=mask)
    code = (word >> ((offset % per_word) * BITS)) & ((1 << BITS) - 1)
    scale = load(scales_pointer + offset // GROUP, mask=mask)
    bias = load(biases_pointer + offset // GROUP, mask=mask)
    store(out_pointer + offset, code * scale + bias, mask=mask)
