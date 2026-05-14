# AWQModifier — activation magnitude 기반 weight per-channel scaling stub
from __future__ import annotations

from typing import Iterable, Optional

import torch.nn as nn

from .base import BaseModifier


class AWQModifier(BaseModifier):
    """AWQ: activation magnitude 기반 weight per-channel scaling으로 W4A16 정확도 향상.

    Lifecycle:
        initialize(model): smooth pair 탐색 + activation hook 등록 (SmoothQuantModifier와 유사).
        calibrate(dataloader): channel-wise activation magnitude 수집 → weight scaling 적용.
        finalize(): hook 해제.

    Intended behavior:
        1. 각 Linear 입력의 channel-wise activation magnitude를 수집한다:
           s_x = mean(|X|) per input-channel (calibration data 평균)
        2. per-channel scaling factor를 계산한다:
           s = s_x^alpha  (SmoothQuant와 달리 weight 분포는 고려하지 않음)
        3. weight에 s를 흡수한다: linear.weight *= diag(s)
        4. 직전 LayerNorm / embedding에 s^-1 을 흡수해 등가 변환을 유지한다.
        5. 이후 QuantizationModifier를 chain으로 붙여 RTN 양자화를 적용한다.

    Note:
        SmoothQuant와 달리 activation의 salient channel을 보호하는 방향으로 설계된다.
        Compressor([AWQModifier(...), QuantizationModifier(W4A16)]) 같은 형태로 사용.
        변경 파일: modifiers/awq.py 만.

    Reference:
        AWQ: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        alpha: float = 0.5,
        num_samples: Optional[int] = None,
    ):
        self.alpha = alpha
        self.num_samples = num_samples

    def initialize(self, model: nn.Module) -> None:
        raise NotImplementedError(
            "AWQ is not yet implemented. "
            "Use SmoothQuantModifier for SmoothQuant or QuantizationModifier for standard RTN."
        )

    def calibrate(
        self, dataloader: Iterable, num_samples: Optional[int] = None
    ) -> None:
        raise NotImplementedError(
            "AWQ is not yet implemented. "
            "Use SmoothQuantModifier for SmoothQuant or QuantizationModifier for standard RTN."
        )

    def finalize(self) -> None:
        raise NotImplementedError(
            "AWQ is not yet implemented. "
            "Use SmoothQuantModifier for SmoothQuant or QuantizationModifier for standard RTN."
        )
