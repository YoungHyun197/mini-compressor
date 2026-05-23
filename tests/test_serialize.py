# serialize.py round-trip 테스트 — save_pretrained / load_pretrained
import json
import os
import tempfile

import pytest
import torch
import torch.nn as nn

from mini_compressor.serialize import (
    _scheme_to_dict,
    _scheme_from_dict,
    save_pretrained,
    QUANT_CONFIG_FILENAME,
)
from mini_compressor.schemes import W8A8, W4A16


# --- _scheme_to_dict / _scheme_from_dict 단위 테스트 ---

def test_scheme_to_dict_w8a8():
    d = _scheme_to_dict(W8A8, ignore=["lm_head"])
    assert d["quant_type"] == "compressed-tensors"
    assert d["quantization_status"] == "calibrated"
    group = d["config_groups"]["group_0"]
    assert group["weights"]["strategy"] == "channel"
    assert group["weights"]["num_bits"] == 8
    assert group["weights"]["axis"] == 0
    assert group["input_activations"]["strategy"] == "tensor"
    assert group["input_activations"]["symmetric"] is False
    assert group["targets"] == ["Linear"]
    assert group["ignore"] == ["lm_head"]


def test_scheme_to_dict_w4a16():
    d = _scheme_to_dict(W4A16)
    group = d["config_groups"]["group_0"]
    assert group["weights"]["strategy"] == "group"
    assert group["weights"]["num_bits"] == 4
    assert group["weights"]["axis"] == 1
    assert group["input_activations"] is None
    assert "ignore" not in group


def test_scheme_roundtrip_w8a8():
    d = _scheme_to_dict(W8A8, ignore=["lm_head"])
    scheme, ignore = _scheme_from_dict(d)
    assert scheme.name == "w8a8"
    assert scheme.weight.num_bits == 8
    assert scheme.weight.granularity == "per_channel"
    assert scheme.activation is not None
    assert scheme.activation.granularity == "per_tensor"
    assert ignore == ["lm_head"]


def test_scheme_roundtrip_w4a16():
    d = _scheme_to_dict(W4A16)
    scheme, ignore = _scheme_from_dict(d)
    assert scheme.name == "w4a16"
    assert scheme.weight.num_bits == 4
    assert scheme.weight.granularity == "per_group"
    assert scheme.activation is None
    assert ignore is None


# --- save_pretrained + quantization_config.json 저장 확인 ---

class _TinyModel(nn.Module):
    """HF save_pretrained 없이 state_dict만 저장하는 소형 테스트 모델."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)
        self.lm_head = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.lm_head(self.fc(x))

    def save_pretrained(self, save_dir):
        from safetensors.torch import save_file
        os.makedirs(save_dir, exist_ok=True)
        save_file(self.state_dict(), os.path.join(save_dir, "model.safetensors"))


def test_save_writes_quant_config():
    model = _TinyModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        save_pretrained(model, tmpdir, scheme=W8A8, ignore=["lm_head"])
        config_path = os.path.join(tmpdir, QUANT_CONFIG_FILENAME)
        assert os.path.exists(config_path)
        with open(config_path) as f:
            d = json.load(f)
        assert d["quant_type"] == "compressed-tensors"
        assert d["config_groups"]["group_0"]["ignore"] == ["lm_head"]


def test_save_writes_safetensors():
    model = _TinyModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        save_pretrained(model, tmpdir, scheme=W4A16)
        assert os.path.exists(os.path.join(tmpdir, "model.safetensors"))
