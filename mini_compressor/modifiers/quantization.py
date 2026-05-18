# QuantizationMixin + QuantizationModifier — module replacement 공통 mixin과 RTN modifier
from __future__ import annotations

import fnmatch
from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from ..fake_quant_linear import FakeQuantLinear
from ..observer import build_observer
from ..schemes import QuantizationScheme
from .base import BaseModifier


class QuantizationMixin:
    """nn.Linear → FakeQuantLinear 교체 + finalize를 GPTQModifier와 공유하는 mixin.

    llm-compressor의 QuantizationMixin 패턴: QuantizationModifier와 GPTQModifier가
    동일한 module replacement 로직을 상속한다.
    initialize / finalize가 여기 구현되고, calibrate는 각 Modifier가 별도로 구현한다.
    """

    def __init__(
        self,
        scheme: QuantizationScheme,
        targets: Optional[List[str]] = None,
        ignore: Optional[List[str]] = None,
        compute_scales: bool = True,
    ):
        self.scheme = scheme
        self.targets = targets
        self.ignore = ignore or []
        self.compute_scales = compute_scales
        self.model: Optional[nn.Module] = None

    def _should_replace(self, name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if name in self.ignore:
            return False
        if self.targets is None:
            return True
        class_name = type(module).__name__
        return any(
            fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(class_name, pat)
            for pat in self.targets
        )

    def initialize(self, model: nn.Module) -> None:
        """nn.Linear → FakeQuantLinear 교체.

        compute_scales=True (기본): weight observer로 weight scale 즉시 산출 — RTN 흐름.
        compute_scales=False: 구조만 생성, scale은 calibrate에서 채움 — GPTQ 흐름.
        """
        self.model = model
        to_replace = [
            (name, mod)
            for name, mod in model.named_modules()
            if self._should_replace(name, mod)
        ]

        for name, mod in to_replace:
            fql = FakeQuantLinear.from_float(mod, self.scheme)
            if self.compute_scales:
                wobs = build_observer(self.scheme.weight).to(fql.weight.device)
                wobs.update(fql.weight.detach())
                w_scale, w_zp = wobs.compute_scale_zp()
                fql.weight_scale = w_scale.to(fql.weight.device)
                fql.weight_zero_point = w_zp.to(fql.weight.device)
            *parent_path, attr = name.split(".")
            parent = model
            for part in parent_path:
                parent = getattr(parent, part)
            setattr(parent, attr, fql)

    def finalize(self) -> None:
        """observer 제거, scale buffer만 남김."""
        assert self.model is not None, "initialize(model)을 먼저 호출해야 합니다."
        for mod in self.model.modules():
            if isinstance(mod, FakeQuantLinear):
                mod.input_observer = None


class QuantizationModifier(QuantizationMixin, BaseModifier):
    """nn.Linear를 FakeQuantLinear로 교체하고 observer로 scale을 산출한다.

    weight·activation 모두 동일한 observer 추상화를 거친다 — clip range는
    scheme.{weight,activation}.calibration_method(minmax/percentile/mse)가 정하고,
    rounding은 양쪽 다 round-to-nearest(RTN)다.

    Lifecycle:
        initialize(model): nn.Linear → FakeQuantLinear 교체 + weight observer로
                           weight scale 산출 (compute_scales=True 기본).
        calibrate(dataloader): activation observer로 input_scale/input_zero_point 계산
                               (scheme.activation이 None이거나 dynamic이면 no-op).
        finalize(): observer 제거, scale buffer만 남김.
    """

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

        if self.scheme.activation is None or self.scheme.activation.dynamic:
            return

        assert self.model is not None, "initialize(model)을 먼저 호출해야 합니다."

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

        for mod in self.model.modules():
            if isinstance(mod, FakeQuantLinear) and mod.input_observer is not None:
                mod.input_observer.sync()  # multi-GPU: rank 간 통계 동기화 (단일 GPU에선 no-op)
                scale, zp = mod.input_observer.compute_scale_zp()
                # scale/zp가 weight.device를 따라가도록 보정 — Percentile 등은 _data를
                # CPU로 모아 결과가 CPU에 남을 수 있음 (device_map="auto" 호환).
                mod.input_scale = scale.to(mod.weight.device)
                mod.input_zero_point = zp.to(mod.weight.device)
