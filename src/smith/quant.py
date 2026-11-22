import struct


def pack_words(codes, bits=4):
    per_word = 32 // bits
    padded = list(codes) + [0] * (-len(codes) % per_word)
    mask = (1 << bits) - 1
    words = []
    for start in range(0, len(padded), per_word):
        word = 0
        for slot, code in enumerate(padded[start:start + per_word]):
            word |= (code & mask) << (slot * bits)
        words.append(word)
    return words


def unpack_words(words, count, bits=4):
    per_word = 32 // bits
    mask = (1 << bits) - 1
    return [(words[index // per_word] >> ((index % per_word) * bits)) & mask for index in range(count)]


def quantize(values, bits=4, group_size=64):
    if bits not in (1, 2, 4, 8):
        raise ValueError("bits must be 1, 2, 4 or 8")
    levels = (1 << bits) - 1
    codes, scales, biases = [], [], []
    for start in range(0, len(values), group_size):
        group = values[start:start + group_size]
        low, high = min(group), max(group)
        scale = (high - low) / levels or 1.0
        scales.append(scale)
        biases.append(low)
        codes.extend(min(levels, max(0, round((value - low) / scale))) for value in group)
    return pack_words(codes, bits), scales, biases


def dequantize(words, scales, biases, count, bits=4, group_size=64):
    codes = unpack_words(words, count, bits)
    return [code * scales[index // group_size] + biases[index // group_size] for index, code in enumerate(codes)]


def words_to_bytes(words):
    return struct.pack(f"<{len(words)}I", *words)
