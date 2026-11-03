"""穷举FP16输入的CPU舍入模型；不代替NPU数学实现或真机正确性。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), '已有模型结果不覆盖'
    bits = np.arange(65536, dtype=np.uint16)
    x = bits.view(np.float16)
    with np.errstate(all='ignore'):
        reference = np.expm1(x.astype(np.float64)).astype(np.float16)
        raw = np.subtract(np.exp(x), np.float16(1), dtype=np.float16)
        series = np.multiply(x, np.add(np.float16(1),
            np.multiply(np.float16(.5), x, dtype=np.float16), dtype=np.float16), dtype=np.float16)
        actual = np.where(np.abs(x) < np.float16(.125), series, raw)
        actual = np.where(x == np.float16(0), x, actual).astype(np.float16)
        passed = np.isclose(actual, reference, rtol=.015, atol=0, equal_nan=True)
        finite = np.isfinite(reference) & (reference != 0) & np.isfinite(actual)
        relative = np.abs(actual[finite].astype(np.float64) - reference[finite].astype(np.float64)) / np.abs(reference[finite].astype(np.float64))
    zeros = x == 0
    zero_sign = bool(np.array_equal(np.signbit(actual[zeros]), np.signbit(reference[zeros])))
    result = dict(status='passed' if passed.all() and zero_sign else 'failed',
        domain='all_65536_FP16_bit_patterns', total=65536, passed=int(passed.sum()),
        rtol=.015, atol=0, equal_nan=True,
        max_relative_error=float(relative.max()), negative_zero_preserved=zero_sign,
        failing_input_bits=[int(v) for v in bits[~passed][:32]],
        reference='CPU float64 expm1 then FP16 rounding',
        formula_model='CPU numpy FP16 exp, multiply, add and subtract',
        numpy_version=np.__version__, checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        device_executed=False, performance_measured=False,
        scope='只验证理想FP16逐步舍入公式；未验证NPU的exp近似、融合、特殊值及实际存储。真机正式评测与补测单独报告。')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({k: result[k] for k in ['status', 'passed', 'total', 'max_relative_error', 'negative_zero_preserved']}))


if __name__ == '__main__':
    main()
