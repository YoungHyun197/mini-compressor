# sequential calibration 단위 테스트 — scale 동등성, 미지원 구조 에러, 빈 dataloader 에러
import copy
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn

from mini_compressor.fake_quant_linear import FakeQuantLinear
from mini_compressor.modifiers import QuantizationModifier
from mini_compressor.schemes import W8A8


# ── 테스트용 최소 모델 (model.model.layers 구조) ──────────────────────────────

class _DecodeLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, hidden_states, **kwargs):
        return (self.fc(hidden_states),)


class _InnerModel(nn.Module):
    def __init__(self, dim: int, n_layers: int):
        super().__init__()
        self.embed = nn.Embedding(50, dim)
        self.layers = nn.ModuleList([_DecodeLayer(dim) for _ in range(n_layers)])

    def forward(self, input_ids, **kwargs):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)[0]
        return h


class _CausalLM(nn.Module):
    def __init__(self, dim: int = 16, n_layers: int = 2):
        super().__init__()
        self.model = _InnerModel(dim, n_layers)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids)


def _calib_data(n: int = 4, seq_len: int = 8) -> list:
    return [{"input_ids": torch.randint(0, 50, (1, seq_len))} for _ in range(n)]


# ── 테스트 ─────────────────────────────────────────────────────────────────────

def test_sequential_scale_equals_full():
    """sequential과 full-model calibration이 동일한 input_scale을 산출한다.

    두 경로 모두 calibration forward 중 input_scale=None → activation fake-quant 없음.
    따라서 각 FakeQuantLinear가 보는 activation 분포가 동일 → scale 수치 일치.
    """
    torch.manual_seed(42)
    model_full = _CausalLM()
    model_seq = copy.deepcopy(model_full)
    data = _calib_data()

    mod_full = QuantizationModifier(W8A8)
    mod_full.initialize(model_full)
    mod_full.calibrate(data, sequential=False)
    mod_full.finalize()

    mod_seq = QuantizationModifier(W8A8)
    mod_seq.initialize(model_seq)
    mod_seq.calibrate(data, sequential=True)
    mod_seq.finalize()

    fqls_full = [m for m in model_full.modules() if isinstance(m, FakeQuantLinear)]
    fqls_seq = [m for m in model_seq.modules() if isinstance(m, FakeQuantLinear)]

    assert len(fqls_full) > 0, "FakeQuantLinear 교체 실패"
    for fa, fb in zip(fqls_full, fqls_seq):
        assert fa.input_scale is not None
        assert fb.input_scale is not None
        assert torch.allclose(fa.input_scale, fb.input_scale, atol=1e-6), (
            f"scale mismatch: {fa.input_scale.item():.6f} vs {fb.input_scale.item():.6f}"
        )


def test_sequential_unsupported_model_raises():
    """model.model.layers 구조가 없는 모델에서 sequential=True는 RuntimeError를 낸다."""
    model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 8))
    mod = QuantizationModifier(W8A8)
    mod.initialize(model)

    with pytest.raises(RuntimeError, match="model.model.layers"):
        mod.calibrate(_calib_data(), sequential=True)


def test_sequential_empty_dataloader_raises():
    """빈 dataloader로 sequential calibration을 시도하면 ValueError를 낸다."""
    model = _CausalLM()
    mod = QuantizationModifier(W8A8)
    mod.initialize(model)

    with pytest.raises(ValueError):
        mod.calibrate([], sequential=True)
