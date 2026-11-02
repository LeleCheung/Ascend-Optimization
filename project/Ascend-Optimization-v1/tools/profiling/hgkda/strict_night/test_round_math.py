"""独立验证快速舍入公式的数学边界；设备生成代码仍须完整KGS和补测。"""
import math
import struct
import unittest


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def reference(value):
    if not math.isfinite(value):
        return value
    return math.copysign(float(round(value)), value)


def proposed(value):
    absolute = abs(value)
    rounded = absolute if absolute >= 8388608.0 else f32(f32(absolute + 8388608.0) - 8388608.0)
    return math.copysign(rounded, value)


class RoundMathTests(unittest.TestCase):
    def check_value(self, value):
        actual, expected = proposed(value), reference(value)
        if math.isnan(expected):
            self.assertTrue(math.isnan(actual))
        else:
            self.assertEqual(actual, expected, repr(value))
            if expected == 0:
                self.assertEqual(math.copysign(1, actual), math.copysign(1, expected))

    def test_every_fp16_value_and_signed_zero(self):
        for bits in range(65536):
            self.check_value(struct.unpack('e', struct.pack('H', bits))[0])

    def test_every_bf16_value(self):
        for bits in range(65536):
            self.check_value(struct.unpack('f', struct.pack('I', bits << 16))[0])

    def test_fp32_ties_large_integers_and_adjacent_values(self):
        for value in [0, .5, 1.5, 2.5, 3.5, 1023.5, 2047.5, 65535.5,
                      4194303.5, 4194304.5, 4194305., 8388607.5, 8388608., 16777216.]:
            bits = struct.unpack('I', struct.pack('f', value))[0]
            for adjacent in range(max(0, bits-2), bits+3):
                x = struct.unpack('f', struct.pack('I', adjacent))[0]
                self.check_value(x)
                self.check_value(-x)


if __name__ == '__main__':
    unittest.main()
