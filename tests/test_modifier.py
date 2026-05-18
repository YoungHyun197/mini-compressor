# QuantizationModifier initialize / calibrate / finalize 3단계 검증 테스트
import dataclasses
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


def test_weight_observer_method_selectable():
    """weight scale도 calibration_method(minmax/percentile/mse)를 반영한다."""
    torch.manual_seed(0)
    w = torch.randn(64, 128)

    def scale_of(method):
        model = nn.Sequential(nn.Linear(128, 64))
        model[0].weight.data.copy_(w)
        spec = dataclasses.replace(W8A8.weight, calibration_method=method)
        scheme = dataclasses.replace(W8A8, weight=spec)
        QuantizationModifier(scheme).initialize(model)
        return model[0].weight_scale

    s_mm = scale_of("minmax")
    s_pct = scale_of("percentile")
    s_mse = scale_of("mse")

    # 모두 per_channel → 출력 채널마다 scale 1개
    assert s_mm.shape == (64,) and s_pct.shape == (64,) and s_mse.shape == (64,)
    # percentile/mse는 outlier를 깎으므로 minmax보다 좁거나 같은 범위여야 한다
    assert (s_pct <= s_mm + 1e-6).all(), "percentile scale이 minmax보다 커선 안 된다"
    assert (s_mse <= s_mm + 1e-6).all(), "mse scale이 minmax보다 커선 안 된다"
    # percentile은 실제로 minmax와 달라야 한다 (방법이 weight에 반영됨)
    assert not torch.allclose(s_pct, s_mm), "percentile이 minmax와 동일 — 미반영"


def test_weight_observer_per_group():
    """per_group weight에도 observer 방법 선택이 동작하고 scale shape이 보존된다."""
    model = nn.Sequential(nn.Linear(128, 64))
    spec = dataclasses.replace(W4A16.weight, calibration_method="mse")
    scheme = dataclasses.replace(W4A16, weight=spec)
    QuantizationModifier(scheme).initialize(model)
    # in_features=128, group_size=128 → 그룹 1개
    assert model[0].weight_scale.shape == (64, 1)


if __name__ == "__main__":
    test_initialize_replaces_linear()
    test_initialize_ignores_module()
    test_initialize_w4a16_scale_shape()
    test_calibrate_sets_input_scale()
    test_calibrate_skips_w4a16()
    test_finalize_removes_observer()
    test_weight_observer_method_selectable()
    test_weight_observer_per_group()
    print("\n모든 테스트 통과")
