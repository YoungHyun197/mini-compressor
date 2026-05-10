# nn.Linear를 fake quantization으로 교체하는 FakeQuantLinear 모듈
import torch
import torch.nn as nn
import torch.nn.functional as F
from .schemes import QuantizationScheme


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
        self.register_buffer("input_scale", None)

    @classmethod
    def from_float(cls, module: nn.Linear, scheme: QuantizationScheme) -> "FakeQuantLinear":
        q = cls(module.in_features, module.out_features, module.bias is not None, scheme)
        q.weight = module.weight
        q.bias = module.bias
        return q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._fake_quantize_weight(self.weight)
        if self.scheme is not None and self.scheme.activation is not None and self.input_scale is not None:
            x = self._fake_quantize_activation(x)
        return F.linear(x, weight, self.bias)

    def _fake_quantize_weight(self, w: torch.Tensor) -> torch.Tensor:
        if self.weight_scale is None:
            return w

        spec = self.scheme.weight
        qmax = 2 ** (spec.num_bits - 1) - 1
        qmin = -qmax if spec.symmetric else -(2 ** (spec.num_bits - 1))
        s = self.weight_scale
        zp = self.weight_zero_point if self.weight_zero_point is not None else torch.zeros_like(s)

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
        if self.input_scale is None:
            return x

        spec = self.scheme.activation
        qmax = 2 ** (spec.num_bits - 1) - 1
        qmin = -qmax if spec.symmetric else -(2 ** (spec.num_bits - 1))
        s = self.input_scale

        q = torch.clamp(torch.round(x / s), qmin, qmax)
        return q * s

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
