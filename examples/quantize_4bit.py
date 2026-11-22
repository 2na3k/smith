import random

from smith import dtype, quant
from smith.kernels.dequantize import dequantize_kernel
from smith.compiler import Buffer

count, group_size = 4096, 64
generator = random.Random(0)
weights = [generator.gauss(0.0, 1.0) for _ in range(count)]

words, scales, biases = quant.quantize(weights, bits=4, group_size=group_size)
out = Buffer.from_list(dtype.float32, [0.0] * count)
dequantize_kernel[lambda meta: (-(-count // meta["BLOCK"]),)](
    Buffer.from_list(dtype.uint32, words),
    Buffer.from_list(dtype.float32, scales),
    Buffer.from_list(dtype.float32, biases),
    out, count, BITS=4, GROUP=group_size, BLOCK=64)

restored = out.to_list()
error = max(abs(a - b) for a, b in zip(weights, restored))
packed_bytes = len(words) * 4 + (len(scales) + len(biases)) * 4
print(f"float32 weights: {count * 4} bytes")
print(f"4-bit packed:    {packed_bytes} bytes ({packed_bytes / (count * 4):.1%}) including scales and biases")
print(f"max abs error:   {error:.4f} (step size per group <= {max(scales):.4f})")

codes = [index % 16 - 8 for index in range(count)]
packed = Buffer.from_list(dtype.int4, codes)
print(f"int4 buffer:     {packed.nbytes} bytes for {count} elements, round trip ok: {packed.to_list() == codes}")
