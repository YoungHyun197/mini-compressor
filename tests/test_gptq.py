# GPTQModifier 단위 테스트 — initialize / calibrate / finalize + RTN 대비 MSE 검증
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from mini_compressor.modifiers.gptq import GPTQModifier
from mini_compressor.modifiers.quantization import QuantizationModifier
from mini_compressor.fake_quant_linear import FakeQuantLinear
from mini_compressor.schemes import W4A16


def _make_model(in_f: int = 128, out_f: int = 64, seed: int = 0) -> nn.Sequential:
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(in_f, out_f, bias=False))
    return m


def test_gptq_replaces_linear():
    """GPTQModifier.initialize가 nn.Linear를 FakeQuantLinear로 교체해야 한다."""
    model = _make_model()
    gptq = GPTQModifier(W4A16)
    x = torch.randn(8, 128)
    gptq.initialize(model)
    gptq.calibrate([(x,)])
    gptq.finalize()

    assert isinstance(model[0], FakeQuantLinear)


def test_gptq_scale_shape():
    """calibrate 후 weight_scale shape이 [out_features, n_groups]여야 한다."""
    model = _make_model(in_f=128, out_f=64)
    gptq = GPTQModifier(W4A16)
    x = torch.randn(8, 128)
    gptq.initialize(model)
    gptq.calibrate([(x,)])
    gptq.finalize()

    fql = model[0]
    # in_features=128, group_size=128 → n_groups=1
    assert fql.weight_scale.shape == (64, 1), f"got {fql.weight_scale.shape}"


def test_gptq_weight_on_grid():
    """GPTQ 후 weight는 이미 fake-quantized 상태 — re-quantizing이 no-op이어야 한다."""
    model = _make_model(in_f=128, out_f=64)
    gptq = GPTQModifier(W4A16)
    x = torch.randn(16, 128)
    gptq.initialize(model)
    gptq.calibrate([(x,)])
    gptq.finalize()

    fql = model[0]
    # weight를 한 번 더 fake-quant하면 결과가 같아야 한다 (on-grid 특성)
    w_requant = fql._fake_quantize_weight(fql.weight)
    assert torch.allclose(w_requant, fql.weight, atol=1e-4), \
        f"weight가 grid 위에 없음 — max diff: {(w_requant - fql.weight).abs().max():.6f}"


def test_gptq_mse_leq_rtn():
    """GPTQ 출력 MSE가 RTN 출력 MSE 이하여야 한다 (W4A16, n_groups=1).

    GPTQ는 H = 2·XᵀX (calibration data) 기준으로 출력 오차를 최소화한다.
    n_samples >= in_features 여야 H가 full-rank가 되어 GPTQ가 효과적이다.
    평가도 동일 calibration data로 수행 — GPTQ가 최소화한 metric과 일치.
    """
    torch.manual_seed(42)
    in_f, out_f = 128, 64
    W_orig = torch.randn(out_f, in_f)
    # n_samples > in_features: full-rank Hessian 보장
    x_calib = torch.randn(512, in_f)

    # RTN
    m_rtn = nn.Sequential(nn.Linear(in_f, out_f, bias=False))
    m_rtn[0].weight.data.copy_(W_orig)
    QuantizationModifier(W4A16).initialize(m_rtn)

    # GPTQ
    m_gptq = nn.Sequential(nn.Linear(in_f, out_f, bias=False))
    m_gptq[0].weight.data.copy_(W_orig)
    gptq = GPTQModifier(W4A16)
    gptq.initialize(m_gptq)
    gptq.calibrate([(x_calib,)])
    gptq.finalize()

    # calibration data로 평가: GPTQ가 최소화한 metric이므로 항상 RTN <= GPTQ
    out_orig = x_calib @ W_orig.T
    with torch.no_grad():
        out_rtn = m_rtn(x_calib)
        out_gptq = m_gptq(x_calib)

    mse_rtn = ((out_orig - out_rtn) ** 2).mean().item()
    mse_gptq = ((out_orig - out_gptq) ** 2).mean().item()

    assert mse_gptq < mse_rtn, \
        f"GPTQ MSE {mse_gptq:.6f} >= RTN MSE {mse_rtn:.6f}"
