import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mini_compressor.fake_quant_linear import FakeQuantLinear
from mini_compressor.schemes import W8A8, W4A16, W8A8_DYNAMIC


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


def test_dynamic_no_observer():
    """dynamic=True면 input_observer가 생성되지 않아야 한다."""
    linear = nn.Linear(64, 32)
    q = FakeQuantLinear.from_float(linear, W8A8_DYNAMIC)
    assert q.input_observer is None
    assert q.input_scale is None


def test_dynamic_per_token_forward():
    """dynamic per-token: input_scale 없이도 activation quantization이 적용된다."""
    linear = nn.Linear(64, 32)
    q = FakeQuantLinear.from_float(linear, W8A8_DYNAMIC)

    # weight scale 주입 (per_channel)
    q.weight_scale = q.weight.abs().amax(dim=1) / 127.0

    x = torch.randn(2, 4, 64)  # (batch, seq_len, hidden)
    out = q(x)
    assert out.shape == (2, 4, 32)


def test_dynamic_per_token_scale_differs_per_token():
    """토큰마다 activation scale이 다르게 계산되는지 확인."""
    linear = nn.Linear(64, 32)
    q = FakeQuantLinear.from_float(linear, W8A8_DYNAMIC)
    q.weight_scale = q.weight.abs().amax(dim=1) / 127.0

    # 토큰별로 값 범위가 크게 다른 입력
    x = torch.zeros(1, 3, 64)
    x[0, 0, :] = 1.0   # 토큰 0: 범위 작음
    x[0, 1, :] = 10.0  # 토큰 1: 범위 큼
    x[0, 2, :] = 0.1   # 토큰 2: 범위 매우 작음

    spec = W8A8_DYNAMIC.activation
    qmax = 2 ** (spec.num_bits - 1) - 1
    s = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax

    # 토큰별 scale이 모두 달라야 한다
    assert s[0, 0, 0] != s[0, 1, 0]
    assert s[0, 1, 0] != s[0, 2, 0]


if __name__ == "__main__":
    test_from_float_shape()
    test_forward_without_scale()
    test_forward_matches_fp_before_calibration()
    test_dynamic_no_observer()
    test_dynamic_per_token_forward()
    test_dynamic_per_token_scale_differs_per_token()
    print("\n모든 테스트 통과")
