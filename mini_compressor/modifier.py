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

    def calibrate(
        self,
        dataloader: Iterable,
        num_samples: Optional[int] = None,
        sequential: bool = False,
    ) -> None:
        """calibration forward pass로 activation scale 계산.

        Args:
            dataloader: 캘리브레이션 배치 이터러블. 각 배치는 dict, tuple, Tensor 중 하나.
            num_samples: 사용할 최대 배치 수. None이면 전체 사용.
            sequential: True이면 layer별 순차 캘리브레이션 (메모리 효율). 미구현.

        Note:
            sequential=True는 현재 미구현입니다. 활성화 시 NotImplementedError를 발생시킵니다.
            구현 예정 동작: 각 FakeQuantLinear를 순서대로 활성화하고 나머지는 bypass하여
            GPU 메모리 사용량을 O(single_layer)로 줄입니다.
        """
        if sequential:
            raise NotImplementedError(
                "sequential calibration is not yet implemented. "
                "Use sequential=False (default) for full-model forward calibration."
            )

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

    def smooth(
        self,
        dataloader: Iterable,
        alpha: float = 0.5,
        num_samples: Optional[int] = None,
    ) -> None:
        """SmoothQuant: activation 분포를 weight 쪽으로 이관하여 양자화 오차 감소.

        Args:
            dataloader: activation 통계 수집용 캘리브레이션 배치 이터러블.
            alpha: 이관 강도 (0.0 = weight만 조정, 1.0 = activation만 조정). 기본값 0.5.
            num_samples: 통계 수집에 사용할 최대 배치 수.

        Intended behavior:
            1. 각 Linear layer 직전 activation의 channel-wise max를 수집한다.
            2. 직전 LayerNorm/RMSNorm의 weight와 Linear의 weight에 smooth factor
               s = max(|X|)^alpha / max(|W|)^(1-alpha) 를 적용한다.
               - norm.weight /= s  (activation 스케일 흡수)
               - linear.weight *= s  (weight 쪽으로 분포 이관)
            3. 변환 후 activation 분포가 평탄해져 per-tensor int8 양자화 오차가 줄어든다.

        Reference:
            SmoothQuant: https://arxiv.org/abs/2211.10438
        """
        raise NotImplementedError(
            "SmoothQuant is not yet implemented. "
            "Call initialize() → calibrate() → finalize() for standard PTQ."
        )

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
