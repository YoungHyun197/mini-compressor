# GPTQModifier — Hessian 기반 weight 업데이트 stub
from __future__ import annotations

from typing import Iterable, List, Optional

import torch.nn as nn

from ..schemes import QuantizationScheme
from .base import BaseModifier


class GPTQModifier(BaseModifier):
    """GPTQ: Hessian 기반 layer-wise weight 업데이트로 W4A16 정확도 향상.

    Lifecycle:
        initialize(model): nn.Linear → FakeQuantLinear 교체 (QuantizationModifier와 동일).
        calibrate(dataloader): RTN 대신 GPTQ 알고리즘으로 weight 업데이트.
        finalize(): hook/임시 상태 정리.

    Intended behavior:
        1. 각 FakeQuantLinear에 forward hook을 등록해 입력 X를 수집한다.
        2. H = 2 * X^T X (Hessian 근사)를 layer별로 계산한다.
        3. Cholesky 분해 H = L L^T 후 역행렬을 구한다.
        4. weight column을 block_size 단위로 순서대로 양자화하면서
           오차를 남은 column에 전파한다: W[:,j+1:] -= (err * H_inv[j, j+1:])
        5. 결과 weight를 FakeQuantLinear.weight에 직접 저장한다.

    Note:
        Compressor([GPTQModifier(...)]) 또는 SmoothQuant과 조합 가능:
        Compressor([SmoothQuantModifier(...), GPTQModifier(...)]).
        변경 파일: modifiers/gptq.py 만 (BaseModifier 상속).

    Reference:
        GPTQ: https://arxiv.org/abs/2210.17323
    """

    def __init__(
        self,
        scheme: QuantizationScheme,
        targets: Optional[List[str]] = None,
        ignore: Optional[List[str]] = None,
        block_size: int = 128,
    ):
        self.scheme = scheme
        self.targets = targets
        self.ignore = ignore or []
        self.block_size = block_size

    def initialize(self, model: nn.Module) -> None:
        raise NotImplementedError(
            "GPTQ is not yet implemented. "
            "Use QuantizationModifier for RTN-based quantization."
        )

    def calibrate(
        self, dataloader: Iterable, num_samples: Optional[int] = None
    ) -> None:
        raise NotImplementedError(
            "GPTQ is not yet implemented. "
            "Use QuantizationModifier for RTN-based quantization."
        )

    def finalize(self) -> None:
        raise NotImplementedError(
            "GPTQ is not yet implemented. "
            "Use QuantizationModifier for RTN-based quantization."
        )
