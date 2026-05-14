# SmoothQuantModifier — activation 분포를 weight 쪽으로 이관하여 양자화 오차 감소
from __future__ import annotations

from typing import Iterable, Optional

import torch.nn as nn

from .base import BaseModifier


class SmoothQuantModifier(BaseModifier):
    """SmoothQuant: activation outlier를 weight에 흡수시켜 W8A8 static의 정확도를 끌어올린다.

    Lifecycle:
        initialize(model): smooth pair 탐색 (norm → linear group) + hook 등록.
        calibrate(dataloader): forward pass로 channel-wise activation max 수집 →
                               smooth factor s = x_max^alpha / w_max^(1-alpha) 계산 →
                               norm.weight /= s, linear.weight *= s 적용.
        finalize(): hook 해제 + 임시 상태 정리.

    Reference:
        SmoothQuant: https://arxiv.org/abs/2211.10438
    """

    def __init__(
        self,
        alpha: float = 0.5,
        num_samples: Optional[int] = None,
    ):
        self.alpha = alpha
        self.num_samples = num_samples

    def initialize(self, model: nn.Module) -> None:
        raise NotImplementedError("SmoothQuantModifier 실구현은 Task #11에서 추가됩니다.")

    def calibrate(
        self, dataloader: Iterable, num_samples: Optional[int] = None
    ) -> None:
        raise NotImplementedError("SmoothQuantModifier 실구현은 Task #11에서 추가됩니다.")

    def finalize(self) -> None:
        raise NotImplementedError("SmoothQuantModifier 실구현은 Task #11에서 추가됩니다.")
