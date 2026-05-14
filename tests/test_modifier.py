# QuantizationModifier initialize / calibrate / finalize 3단계 검증 테스트
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from mini_compressor.fake_quant_linear import FakeQuantLinear
from mini_compressor.modifiers import QuantizationModifier
from mini_compressor.schemes import W8A8, W4A16


class TinyModel(nn.Module):
    """ignore 테스트용 — named module이 필요한 경우 사용."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 32)
        self.lm_head = nn.Linear(32, 16)

    def forward(self, x):
        return self.lm_head(self.fc(x))


def test_initialize_replaces_linear():
    # nn.Linear가 FakeQuantLinear로 교체되고 weight/bias/scale shape이 올바른지 확인
    linear1 = nn.Linear(64, 32)
    linear2 = nn.Linear(32, 16)
    model = nn.Sequential(linear1, linear2)

    modifier = QuantizationModifier(W8A8)
    modifier.initialize(model)

    fql1, fql2 = model[0], model[1]
    assert isinstance(fql1, FakeQuantLinear)
    assert isinstance(fql2, FakeQuantLinear)

    assert torch.allclose(fql1.weight, linear1.weight)
    assert torch.allclose(fql2.weight, linear2.weight)
    assert torch.allclose(fql1.bias, linear1.bias)

    assert fql1.weight_scale.shape == (32,)
    assert fql2.weight_scale.shape == (16,)
    assert (fql1.weight_zero_point == 0).all()


def test_initialize_ignores_module():
    # ignore 목록에 있는 모듈은 nn.Linear로 유지되어야 함
    model = TinyModel()
    modifier = QuantizationModifier(W8A8, ignore=["lm_head"])
    modifier.initialize(model)

    assert isinstance(model.fc, FakeQuantLinear)
    assert type(model.lm_head) is nn.Linear


def test_initialize_w4a16_scale_shape():
    # W4A16 per_group scale shape이 [out_features, in_features // group_size]인지 확인
    model = nn.Sequential(nn.Linear(128, 64))
    modifier = QuantizationModifier(W4A16)
    modifier.initialize(model)

    fql = model[0]
    assert isinstance(fql, FakeQuantLinear)
    assert fql.weight_scale.shape == (64, 1)  # in_features=128, group_size=128


def test_calibrate_sets_input_scale():
    # W8A8 calibrate 후 input_scale이 채워지는지 확인
    model = nn.Sequential(nn.Linear(64, 32))
    modifier = QuantizationModifier(W8A8)
    modifier.initialize(model)

    dataloader = [torch.randn(2, 64) for _ in range(3)]
    modifier.calibrate(dataloader)

    fql = model[0]
    assert fql.input_scale is not None
    assert fql.input_scale.shape == torch.Size([])  # per_tensor scalar


def test_calibrate_skips_w4a16():
    # W4A16은 activation=None이므로 calibrate 후에도 input_scale이 None이어야 함
    model = nn.Sequential(nn.Linear(128, 64))
    modifier = QuantizationModifier(W4A16)
    modifier.initialize(model)

    dataloader = [torch.randn(2, 128) for _ in range(3)]
    modifier.calibrate(dataloader)

    assert model[0].input_scale is None


def test_finalize_removes_observer():
    # finalize 후 input_observer가 None으로 제거되는지 확인
    model = nn.Sequential(nn.Linear(64, 32))
    modifier = QuantizationModifier(W8A8)
    modifier.initialize(model)

    dataloader = [torch.randn(2, 64) for _ in range(3)]
    modifier.calibrate(dataloader)
    modifier.finalize()

    assert model[0].input_observer is None


if __name__ == "__main__":
    test_initialize_replaces_linear()
    test_initialize_ignores_module()
    test_initialize_w4a16_scale_shape()
    test_calibrate_sets_input_scale()
    test_calibrate_skips_w4a16()
    test_finalize_removes_observer()
    print("\n모든 테스트 통과")
