# Compressor API 테스트 — from_recipe / compress / save round-trip
import os
import tempfile

import pytest
import torch
import torch.nn as nn

from mini_compressor import Compressor, QuantizationModifier, SmoothQuantModifier
from mini_compressor.fake_quant_linear import FakeQuantLinear
from mini_compressor.schemes import W8A8, W4A16, W8A8_DYNAMIC


class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        # in_features=256: W4A16 group_size=128 이상이어야 per_group reshape 가능
        self.proj = nn.Linear(256, 256, bias=False)
        self.lm_head = nn.Linear(256, 256, bias=False)

    def forward(self, x):
        return self.lm_head(self.proj(x))

    def save_pretrained(self, save_dir):
        from safetensors.torch import save_file
        os.makedirs(save_dir, exist_ok=True)
        save_file(self.state_dict(), os.path.join(save_dir, "model.safetensors"))


def test_compress_w4a16_replaces_linear():
    model = _TinyLM()
    compressor = Compressor.from_recipe("w4a16", targets=["Linear"], ignore=["lm_head"])
    compressor.compress(model)

    assert isinstance(model.proj, FakeQuantLinear), "proj은 FakeQuantLinear로 교체되어야 함"
    assert isinstance(model.lm_head, nn.Linear), "lm_head는 ignore 대상이므로 그대로여야 함"
    assert not isinstance(model.lm_head, FakeQuantLinear)


def test_compress_w4a16_scale_shape():
    model = _TinyLM()
    compressor = Compressor.from_recipe("w4a16", targets=["Linear"], ignore=["lm_head"])
    compressor.compress(model)

    proj = model.proj
    # per_group: scale shape은 [out_features, in_features // group_size].
    # in_features=256, group_size=128 → 그룹 2개. weight observer가 _to_units로 분할 처리.
    assert proj.weight_scale is not None


def test_compress_w8a8_with_calibration():
    model = _TinyLM()
    compressor = Compressor.from_recipe("w8a8", targets=["Linear"], ignore=["lm_head"])

    # _TinyLM.forward(x)는 positional arg이므로 tuple 형태로 전달
    dataloader_tuple = [(torch.randn(2, 256),) for _ in range(3)]
    compressor.compress(model, dataloader=dataloader_tuple)

    proj = model.proj
    assert isinstance(proj, FakeQuantLinear)
    assert proj.input_observer is None, "finalize 후 observer는 제거되어야 함"


def test_compress_w8a8_dynamic_no_calibration():
    """W8A8_DYNAMIC는 calibration 데이터 없이도 compress가 완료되어야 한다."""
    model = _TinyLM()
    compressor = Compressor.from_recipe("w8a8_dynamic", targets=["Linear"], ignore=["lm_head"])
    compressor.compress(model)  # dataloader 없이 호출

    proj = model.proj
    assert isinstance(proj, FakeQuantLinear)
    assert proj.input_observer is None
    assert proj.input_scale is None  # dynamic은 런타임 계산 — 사전 scale 없음

    # forward가 정상 동작하는지 확인
    x = torch.randn(2, 4, 256)
    out = model(x)
    assert out.shape == (2, 4, 256)


def test_compressor_save_creates_files():
    model = _TinyLM()
    compressor = Compressor.from_recipe("w4a16", targets=["Linear"], ignore=["lm_head"])
    compressor.compress(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        compressor.save(model, tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "quantization_config.json"))
        assert os.path.exists(os.path.join(tmpdir, "model.safetensors"))


def test_from_recipe_unknown_raises():
    with pytest.raises(ValueError, match="Unknown recipe"):
        Compressor.from_recipe("nonexistent")


def test_from_recipe_w8a8_smoothquant_builds_modifier_chain():
    """from_recipe가 [SmoothQuantModifier, QuantizationModifier] chain으로 펼쳐져야 한다."""
    compressor = Compressor.from_recipe(
        "w8a8_smoothquant", targets=["Linear"], ignore=["lm_head"]
    )
    assert len(compressor.modifiers) == 2
    assert isinstance(compressor.modifiers[0], SmoothQuantModifier)
    assert isinstance(compressor.modifiers[1], QuantizationModifier)
    # targets / ignore가 recipe 내부 QuantizationModifier로 전달돼야 함
    assert compressor.modifiers[1].targets == ["Linear"]
    assert compressor.modifiers[1].ignore == ["lm_head"]


def test_from_recipe_compress_replaces_linear():
    """from_recipe("w8a8_smoothquant")가 end-to-end로 compress까지 동작해야 한다.

    _TinyLM은 decoder 구조가 아니라 SmoothQuant pair가 0개 — SmoothQuantModifier는
    no-op이 되고, recipe 파이프라인은 그대로 QuantizationModifier까지 흐른다.
    """
    model = _TinyLM()
    compressor = Compressor.from_recipe(
        "w8a8_smoothquant", targets=["Linear"], ignore=["lm_head"]
    )
    dataloader_tuple = [(torch.randn(2, 256),) for _ in range(3)]
    compressor.compress(model, dataloader=dataloader_tuple)

    assert isinstance(model.proj, FakeQuantLinear)
    assert not isinstance(model.lm_head, FakeQuantLinear)
