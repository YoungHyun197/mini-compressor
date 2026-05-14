# SmoothQuantModifier 수치 동등성 + Compressor chain 통합 검증 테스트
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import torch
import torch.nn as nn

from mini_compressor import Compressor, QuantizationModifier, SmoothQuantModifier
from mini_compressor.modifiers.smoothquant import _find_smooth_pairs
from mini_compressor.schemes import W8A8


class _MiniDecoderLayer(nn.Module):
    """Qwen3/LLaMA 스타일의 최소 decoder block — input_layernorm + self_attn + post_attention_layernorm + mlp."""

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
        # 단순화된 forward — attention 없이 q_proj 출력만 흘려보내 SmoothQuant pair만 검증
        h = self.input_layernorm(x)
        q = self.self_attn.q_proj(h)
        k = self.self_attn.k_proj(h)
        v = self.self_attn.v_proj(h)
        attn_out = self.self_attn.o_proj(q + k + v)
        x = x + attn_out

        h = self.post_attention_layernorm(x)
        mlp_out = self.mlp.down_proj(torch.relu(self.mlp.gate_proj(h)) * self.mlp.up_proj(h))
        return x + mlp_out


def test_find_smooth_pairs_detects_attn_and_mlp():
    model = _MiniDecoderLayer()
    pairs = _find_smooth_pairs(model)
    # input_layernorm → q/k/v 그룹 1개, post_attention_layernorm → gate/up 그룹 1개
    assert len(pairs) == 2

    norms = [p[0] for p in pairs]
    assert model.input_layernorm in norms
    assert model.post_attention_layernorm in norms

    attn_group = next(p for p in pairs if p[0] is model.input_layernorm)
    assert {id(l) for l in attn_group[1]} == {
        id(model.self_attn.q_proj),
        id(model.self_attn.k_proj),
        id(model.self_attn.v_proj),
    }


def test_smooth_preserves_forward_output():
    """SmoothQuantModifier 단독 적용 후 (양자화 없이) forward 출력이 원본과 1e-3 이내 일치해야 한다."""
    torch.manual_seed(0)
    model = _MiniDecoderLayer(hidden_size=32, intermediate_size=64)
    model.eval()

    x = torch.randn(4, 8, 32)
    with torch.no_grad():
        y_orig = model(x)

    sq = SmoothQuantModifier(alpha=0.5)
    sq.initialize(model)
    sq.calibrate([x for _ in range(2)])
    sq.finalize()

    with torch.no_grad():
        y_smoothed = model(x)

    # smooth는 등가 변환 — 수치 차이는 float 누적 오차 수준
    assert torch.allclose(y_orig, y_smoothed, atol=1e-3, rtol=1e-3), (
        f"max abs diff = {(y_orig - y_smoothed).abs().max().item()}"
    )


def test_smooth_preserves_forward_with_layernorm_bias():
    """LayerNorm bias가 0이 아닌 경우에도 SmoothQuant 등가 변환이 유지되어야 한다.

    norm.weight만 /= s 하고 norm.bias는 두면 (beta/s != beta이므로) 출력이 어긋난다.
    이 테스트는 bias를 학습된 모델처럼 비제로로 초기화해 그 회귀를 막는다.
    """
    torch.manual_seed(0)
    model = _MiniDecoderLayer(hidden_size=32, intermediate_size=64)
    # LayerNorm bias를 비제로로 강제 — 학습된 LayerNorm 모델 (GPT-2, BERT, OPT 등) 시뮬레이션
    with torch.no_grad():
        model.input_layernorm.bias.copy_(torch.randn(32) * 0.1)
        model.post_attention_layernorm.bias.copy_(torch.randn(32) * 0.1)
    model.eval()

    x = torch.randn(4, 8, 32)
    with torch.no_grad():
        y_orig = model(x)

    sq = SmoothQuantModifier(alpha=0.5)
    sq.initialize(model)
    sq.calibrate([x for _ in range(2)])
    sq.finalize()

    with torch.no_grad():
        y_smoothed = model(x)

    assert torch.allclose(y_orig, y_smoothed, atol=1e-3, rtol=1e-3), (
        f"LayerNorm bias 흡수 누락 추정 — max abs diff = {(y_orig - y_smoothed).abs().max().item()}"
    )


def test_compressor_chain_smoothquant_then_w8a8():
    """Compressor([SmoothQuant, QuantizationModifier(W8A8)]) 신규 형태가 동작해야 한다."""
    torch.manual_seed(0)
    model = _MiniDecoderLayer(hidden_size=32, intermediate_size=64)
    model.eval()
    x = torch.randn(4, 8, 32)

    compressor = Compressor([
        SmoothQuantModifier(alpha=0.5),
        QuantizationModifier(W8A8, targets=["Linear"]),
    ])
    compressor.compress(model, dataloader=[x for _ in range(2)])

    # 모든 nn.Linear가 FakeQuantLinear로 교체되어야 함 (SmoothQuant 이후 Quantization 동작 확인)
    from mini_compressor.fake_quant_linear import FakeQuantLinear
    fq_count = sum(1 for m in model.modules() if isinstance(m, FakeQuantLinear))
    assert fq_count == 7, f"expected 7 FakeQuantLinear, got {fq_count}"

    # generate sanity — forward 한 번 돌려 NaN/exception 없는지 확인
    with torch.no_grad():
        y = model(x)
    assert torch.isfinite(y).all()


if __name__ == "__main__":
    test_find_smooth_pairs_detects_attn_and_mlp()
    test_smooth_preserves_forward_output()
    test_smooth_preserves_forward_with_layernorm_bias()
    test_compressor_chain_smoothquant_then_w8a8()
    print("\n모든 SmoothQuant 테스트 통과")
