# QuantizationModifier — initialize / calibrate / finalize 3단계 양자화 파이프라인
from __future__ import annotations

import fnmatch
from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from .fake_quant_linear import FakeQuantLinear
from .schemes import QuantizationScheme, QuantizationSpec


class QuantizationModifier:
    def __init__(
        self,
        model: nn.Module,
        scheme: QuantizationScheme,
        targets: Optional[List[str]] = None,
        ignore: Optional[List[str]] = None,
    ):
        self.model = model
        self.scheme = scheme
        self.targets = targets      # None → 모든 nn.Linear 대상
        self.ignore = ignore or []

    def _should_replace(self, name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if name in self.ignore:
            return False
        if self.targets is None:
            return True
        return any(fnmatch.fnmatch(name, pat) for pat in self.targets)

    def initialize(self, compute_scales: bool = True) -> None:
        """nn.Linear → FakeQuantLinear 교체.

        compute_scales=True (기본): RTN으로 weight scale 즉시 계산 — 압축 흐름.
        compute_scales=False: 구조만 생성, scale은 state_dict 로드 후 채움 — load 흐름.
        """
        to_replace = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._should_replace(name, mod)
        ]

        for name, mod in to_replace:
            fql = FakeQuantLinear.from_float(mod, self.scheme)
            if compute_scales:
                fql.weight_scale, fql.weight_zero_point = _compute_weight_scale(
                    fql.weight, self.scheme.weight
                )
            *parent_path, attr = name.split(".")
            parent = self.model
            for part in parent_path:
                parent = getattr(parent, part)
            setattr(parent, attr, fql)

    def calibrate(self, dataloader: Iterable, num_samples: Optional[int] = None) -> None:
        """calibration forward pass로 activation scale 계산."""
        if self.scheme.activation is None:
            return

        self.model.eval()
        n = 0
        with torch.no_grad():
            for batch in dataloader:
                if num_samples is not None and n >= num_samples:
                    break
                if isinstance(batch, dict):
                    self.model(**batch)
                elif isinstance(batch, (list, tuple)):
                    self.model(*batch)
                else:
                    self.model(batch)
                n += 1

        spec = self.scheme.activation
        for mod in self.model.modules():
            if isinstance(mod, FakeQuantLinear) and mod.input_observer is not None:
                scale, zp = mod.input_observer.compute_scale_zp(spec)
                mod.input_scale = scale
                mod.input_zero_point = zp

    def finalize(self) -> None:
        """observer 제거, scale buffer만 남김."""
        for mod in self.model.modules():
            if isinstance(mod, FakeQuantLinear):
                mod.input_observer = None


def _compute_weight_scale(
    w: torch.Tensor, spec: QuantizationSpec
) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = 2 ** (spec.num_bits - 1) - 1

    if spec.granularity == "per_channel":
        # [out_features, in_features] → output channel별 scale
        max_val = w.detach().abs().amax(dim=1)
        scale = torch.clamp(max_val / qmax, min=1e-8)
        zero_point = torch.zeros_like(scale)

    elif spec.granularity == "per_group":
        # in_features 방향 group 분할 → [out_features, in_features // group_size]
        rows, cols = w.shape
        w_grouped = w.detach().reshape(rows, -1, spec.group_size)
        max_val = w_grouped.abs().amax(dim=2)
        scale = torch.clamp(max_val / qmax, min=1e-8)
        zero_point = torch.zeros_like(scale)

    else:  # per_tensor
        max_val = w.detach().abs().max()
        scale = torch.clamp(max_val / qmax, min=1e-8)
        zero_point = torch.zeros_like(scale)

    return scale, zero_point
