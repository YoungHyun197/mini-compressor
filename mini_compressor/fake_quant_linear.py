# nn.Linear를 fake quantization으로 교체하는 FakeQuantLinear 모듈
import torch
import torch.nn as nn
import torch.nn.functional as F
from .schemes import QuantizationScheme
from .observer import build_observer, BaseObserver


class FakeQuantLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        scheme: QuantizationScheme = None,
    ):
        super().__init__(in_features, out_features, bias)
        self.scheme = scheme
        self.register_buffer("weight_scale", None)
        self.register_buffer("weight_zero_point", None)
        # input_scale = None은 두 가지를 의미한다:
        #   (a) static calibration 전 — calibrate() 호출 후 채워짐
        #   (b) dynamic 또는 weight-only — 런타임 scale 계산이나 activation 양자화 없음
        # forward()는 scheme.activation을 먼저 확인하므로 (a)/(b) 구분 없이 동작이 맞다.
        self.register_buffer("input_scale", None)
        self.register_buffer("input_zero_point", None)

        # dynamic=True면 런타임 scale 계산 → observer 불필요
        # calibration 중에만 존재, finalize() 후 None으로 제거 → state_dict 오염 방지
        if scheme is not None and scheme.activation is not None and not scheme.activation.dynamic:
            self.input_observer: BaseObserver | None = build_observer(scheme.activation)
        else:
            self.input_observer = None

    @classmethod
    def from_float(cls, module: nn.Linear, scheme: QuantizationScheme) -> "FakeQuantLinear":
        q = cls(module.in_features, module.out_features, module.bias is not None, scheme)
        q.weight = module.weight
        q.bias = module.bias
        return q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._fake_quantize_weight(self.weight)
        if self.input_observer is not None:
            self.input_observer.update(x)
        if self.scheme is not None and self.scheme.activation is not None:
            is_dynamic = self.scheme.activation.dynamic
            if is_dynamic or self.input_scale is not None:
                x = self._fake_quantize_activation(x)
        return F.linear(x, weight, self.bias)

    def _fake_quantize_weight(self, w: torch.Tensor) -> torch.Tensor:
        if self.weight_scale is None:
            return w

        spec = self.scheme.weight

        if spec.dtype == "float8":
            # float8 (E4M3 / E5M2) fake quant 경로
            # Intended behavior:
            #   1. torch.float8_e4m3fn 또는 float8_e5m2 dtype으로 cast
            #   2. scale을 곱해 원래 range로 복원 (fake quant 시뮬레이션)
            #   3. PyTorch >= 2.1 + Hopper GPU(H100) 필요
            # Reference: https://arxiv.org/abs/2209.05433 (FP8 Formats for Deep Learning)
            raise NotImplementedError(
                "float8 weight quantization is not yet implemented. "
                "Requires PyTorch >= 2.1 and float8_e4m3fn/float8_e5m2 dtype support."
            )

        qmax = 2 ** (spec.num_bits - 1) - 1
        qmin = -qmax if spec.symmetric else -(2 ** (spec.num_bits - 1))
        s = self.weight_scale.to(w.dtype)
        zp = (self.weight_zero_point.to(w.dtype)
              if self.weight_zero_point is not None else torch.zeros_like(s))

        if spec.granularity == "per_channel":
            s = s.view(-1, *([1] * (w.dim() - 1)))
            zp = zp.view(-1, *([1] * (w.dim() - 1)))
            q = torch.clamp(torch.round(w / s + zp), qmin, qmax)
            return (q - zp) * s

        if spec.granularity == "per_group":
            return self._group_fake_quant(w, s, zp, spec.group_size, qmin, qmax)

        # per_tensor
        q = torch.clamp(torch.round(w / s + zp), qmin, qmax)
        return (q - zp) * s

    def _fake_quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        spec = self.scheme.activation

        if spec.dtype == "float8":
            # float8 activation fake quant 경로 — weight와 동일한 제약 조건 적용
            raise NotImplementedError(
                "float8 activation quantization is not yet implemented."
            )

        qmax = 2 ** (spec.num_bits - 1) - 1
        qmin = -qmax if spec.symmetric else -(2 ** (spec.num_bits - 1))

        if spec.dynamic:
            # 런타임 scale 계산 — calibration 불필요
            if spec.granularity == "per_token":
                # x: (..., seq_len, hidden_dim) → 토큰(마지막 dim 제외)마다 scale
                s = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
            else:
                # dynamic per_tensor
                s = x.detach().abs().amax().clamp(min=1e-8) / qmax
            zp = torch.zeros_like(s)
        else:
            if self.input_scale is None:
                return x
            s = self.input_scale.to(x.dtype)
            zp = (self.input_zero_point.to(x.dtype)
                  if self.input_zero_point is not None else torch.zeros_like(s))

        q = torch.clamp(torch.round(x / s + zp), qmin, qmax)
        return (q - zp) * s

    def _group_fake_quant(
        self,
        w: torch.Tensor,
        s: torch.Tensor,
        zp: torch.Tensor,
        group_size: int,
        qmin: int,
        qmax: int,
    ) -> torch.Tensor:
        rows, cols = w.shape
        w_grouped = w.reshape(rows, -1, group_size)
        s = s.reshape(rows, -1, 1)
        zp = zp.reshape(rows, -1, 1)
        q = torch.clamp(torch.round(w_grouped / s + zp), qmin, qmax)
        return ((q - zp) * s).reshape(rows, cols)
