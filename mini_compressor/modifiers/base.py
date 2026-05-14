# Modifier 공통 추상 인터페이스 — initialize / calibrate / finalize 3단계 lifecycle
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

import torch.nn as nn


class BaseModifier(ABC):
    """모든 modifier가 따라야 하는 lifecycle 인터페이스.

    Compressor가 modifier list를 받아 각 modifier에 대해
    initialize → calibrate → finalize 순서로 호출한다.

    설계 의도:
        llm-compressor의 modifier composition pattern을 차용한다.
        새 알고리즘 추가 시 BaseModifier 상속한 새 파일 하나만 추가하면 되도록
        변경 범위를 modifier 단위로 격리한다.
    """

    @abstractmethod
    def initialize(self, model: nn.Module) -> None:
        """model에 modifier가 동작하기 위한 구조를 준비한다.

        예: QuantizationModifier는 nn.Linear → FakeQuantLinear 교체.
            SmoothQuantModifier는 norm-linear pair 탐색 + hook 등록.
        """

    @abstractmethod
    def calibrate(
        self, dataloader: Iterable, num_samples: Optional[int] = None
    ) -> None:
        """calibration 데이터를 사용해 modifier의 핵심 동작을 수행한다.

        예: QuantizationModifier는 activation observer scale 계산.
            SmoothQuantModifier는 activation 통계 수집 후 weight 변형 적용.
        calibration이 필요 없는 modifier는 no-op로 둔다.
        """

    @abstractmethod
    def finalize(self) -> None:
        """modifier가 사용한 임시 상태를 정리한다.

        예: QuantizationModifier는 observer 제거.
            SmoothQuantModifier는 hook 해제.
        """
