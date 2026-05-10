import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mini_compressor.fake_quant_linear import FakeQuantLinear
from mini_compressor.schemes import W8A8, W4A16


def test_from_float_shape():
    linear = nn.Linear(64, 32)
    q = FakeQuantLinear.from_float(linear, W8A8)
    assert q.weight.shape == linear.weight.shape
    assert q.in_features == 64
    assert q.out_features == 32
    print("test_from_float_shape: PASS")


def test_forward_without_scale():
    linear = nn.Linear(64, 32)
    q = FakeQuantLinear.from_float(linear, W8A8)
    x = torch.randn(2, 64)
    out = q(x)
    assert out.shape == (2, 32)
    print("test_forward_without_scale: PASS")


def test_forward_matches_fp_before_calibration():
    linear = nn.Linear(64, 32)
    q = FakeQuantLinear.from_float(linear, W4A16)
    x = torch.randn(2, 64)
    fp_out = linear(x)
    q_out = q(x)
    assert torch.allclose(fp_out, q_out), "scale 없으면 FP output과 동일해야 함"
    print("test_forward_matches_fp_before_calibration: PASS")


if __name__ == "__main__":
    test_from_float_shape()
    test_forward_without_scale()
    test_forward_matches_fp_before_calibration()
    print("\n모든 테스트 통과")
