# AWQModifier 수치 동등성 + Compressor chain 통합 검증 테스트
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import torch
import torch.nn as nn

from mini_compressor import Compressor, AWQModifier, QuantizationModifier
from mini_compressor.modifiers.awq import _quant_error, _int4_fake_quant
from mini_compressor.schemes import W4A16


class _MiniDecoderLayer(nn.Module):
    """Qwen3/LLaMA 스타일의 최소 decoder block — AWQ pair 검증용."""

    def __init__(self, hidden_size: int = 32, intermediate_size: int = 64):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.self_attn.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.self_attn.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.self_attn.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.mlp.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.mlp.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        h = self.input_layernorm(x)
        q = self.self_attn.q_proj(h)
        k = self.self_attn.k_proj(h)
        v = self.self_attn.v_proj(h)
        attn_out = self.self_attn.o_proj(q + k + v)
        x = x + attn_out

        h = self.post_attention_layernorm(x)
        mlp_out = self.mlp.down_proj(torch.relu(self.mlp.gate_proj(h)) * self.mlp.up_proj(h))
        return x + mlp_out


def test_awq_preserves_forward_output():
    """AWQModifier 단독 적용 후 (양자화 없이) forward 출력이 원본과 1e-3 이내 일치해야 한다.

    y = x @ W.T = (x/s) @ (W*s).T — norm.weight /= s, linear.weight *= s 등가 변환.
    """
    torch.manual_seed(0)
    model = _MiniDecoderLayer(hidden_size=32, intermediate_size=64)
    model.eval()

    x = torch.randn(4, 8, 32)
    with torch.no_grad():
        y_orig = model(x)

    awq = AWQModifier(n_grid=10, group_size=32)
    awq.initialize(model)
    awq.calibrate([x for _ in range(2)])
    awq.finalize()

    with torch.no_grad():
        y_after = model(x)

    assert torch.allclose(y_orig, y_after, atol=1e-3, rtol=1e-3), (
        f"AWQ 등가 변환 실패 — max abs diff = {(y_orig - y_after).abs().max().item()}"
    )


def test_awq_preserves_forward_with_layernorm_bias():
    """LayerNorm bias가 0이 아닌 경우에도 AWQ 등가 변환이 유지되어야 한다."""
    torch.manual_seed(1)
    model = _MiniDecoderLayer(hidden_size=32, intermediate_size=64)
    with torch.no_grad():
        model.input_layernorm.bias.copy_(torch.randn(32) * 0.1)
        model.post_attention_layernorm.bias.copy_(torch.randn(32) * 0.1)
    model.eval()

    x = torch.randn(4, 8, 32)
    with torch.no_grad():
        y_orig = model(x)

    awq = AWQModifier(n_grid=10, group_size=32)
    awq.initialize(model)
    awq.calibrate([x for _ in range(2)])
    awq.finalize()

    with torch.no_grad():
        y_after = model(x)

    assert torch.allclose(y_orig, y_after, atol=1e-3, rtol=1e-3), (
        f"LayerNorm bias 흡수 누락 — max abs diff = {(y_orig - y_after).abs().max().item()}"
    )


def test_awq_initialize_required():
    """initialize() 없이 calibrate() 호출 시 RuntimeError 발생."""
    awq = AWQModifier()
    try:
        awq.calibrate([torch.randn(2, 4)])
        assert False, "RuntimeError가 발생해야 함"
    except RuntimeError:
        pass


def test_awq_empty_dataloader():
    """빈 dataloader로 calibrate() 호출 시 ValueError 발생."""
    torch.manual_seed(0)
    model = _MiniDecoderLayer(hidden_size=32, intermediate_size=64)
    awq = AWQModifier()
    awq.initialize(model)
    try:
        awq.calibrate([])
        assert False, "ValueError가 발생해야 함"
    except ValueError:
        pass
    finally:
        awq.finalize()


def test_awq_no_pairs_model():
    """smooth pair가 없는 모델에서 calibrate()는 아무 변경 없이 통과해야 한다."""
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8))
    x = torch.randn(2, 16)

    original_weights = [p.clone() for p in model.parameters()]

    awq = AWQModifier()
    awq.initialize(model)
    awq.calibrate([x])
    awq.finalize()

    for orig, p in zip(original_weights, model.parameters()):
        assert torch.equal(orig, p), "pair 없는 모델의 weight이 변경됨"


def test_awq_compressor_chain():
    """Compressor([AWQModifier(), QuantizationModifier(W4A16)]) chain이 동작해야 한다."""
    torch.manual_seed(0)
    model = _MiniDecoderLayer(hidden_size=128, intermediate_size=256)
    model.eval()
    x = torch.randn(4, 8, 128)

    compressor = Compressor([
        AWQModifier(n_grid=5, group_size=128),
        QuantizationModifier(W4A16, targets=["Linear"]),
    ])
    compressor.compress(model, dataloader=[x for _ in range(2)])

    from mini_compressor.fake_quant_linear import FakeQuantLinear
    fq_count = sum(1 for m in model.modules() if isinstance(m, FakeQuantLinear))
    assert fq_count == 7, f"expected 7 FakeQuantLinear, got {fq_count}"

    with torch.no_grad():
        y = model(x)
    assert torch.isfinite(y).all(), "AWQ+W4A16 forward에서 NaN/Inf 발생"


def test_int4_fake_quant_roundtrip():
    """_int4_fake_quant: 제로 weight는 변화 없어야 하고 qmax=7 범위를 초과하지 않아야 한다."""
    w = torch.zeros(4, 128)
    w_q = _int4_fake_quant(w, group_size=128)
    assert torch.equal(w, w_q)

    w2 = torch.randn(4, 128)
    w_q2 = _int4_fake_quant(w2, group_size=128)
    # scale = amax / 7 → quantized values in [-7, 7] → dequant magnitude ≤ amax
    assert (w_q2.abs() <= w2.abs().amax() + 1e-5).all()


if __name__ == "__main__":
    test_awq_preserves_forward_output()
    test_awq_preserves_forward_with_layernorm_bias()
    test_awq_initialize_required()
    test_awq_empty_dataloader()
    test_awq_no_pairs_model()
    test_awq_compressor_chain()
    test_int4_fake_quant_roundtrip()
    print("\n모든 AWQ 테스트 통과")
