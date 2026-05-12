# Compressor — one-click 진입점 API (from_scheme → compress)
from __future__ import annotations

from typing import Iterable, List, Optional

import torch.nn as nn

from .modifier import QuantizationModifier
from .schemes import QuantizationScheme, SCHEME_REGISTRY
from .serialize import save_pretrained


class Compressor:
    def __init__(
        self,
        scheme: QuantizationScheme,
        targets: Optional[List[str]] = None,
        ignore: Optional[List[str]] = None,
    ):
        self.scheme = scheme
        self.targets = targets
        self.ignore = ignore

    @classmethod
    def from_scheme(
        cls,
        name: str,
        targets: Optional[List[str]] = None,
        ignore: Optional[List[str]] = None,
    ) -> "Compressor":
        """SCHEME_REGISTRY에서 scheme을 조회해 Compressor를 생성한다."""
        if name not in SCHEME_REGISTRY:
            raise ValueError(f"Unknown scheme '{name}'. Available: {list(SCHEME_REGISTRY)}")
        return cls(scheme=SCHEME_REGISTRY[name], targets=targets, ignore=ignore)

    def compress(
        self,
        model: nn.Module,
        dataloader: Optional[Iterable] = None,
        num_samples: Optional[int] = None,
    ) -> nn.Module:
        """initialize → calibrate → finalize 3단계를 순서대로 실행한다."""
        modifier = QuantizationModifier(model, self.scheme, targets=self.targets, ignore=self.ignore)
        modifier.initialize()
        modifier.calibrate(dataloader or [], num_samples=num_samples)
        modifier.finalize()
        return model

    def save(
        self,
        model: nn.Module,
        save_dir: str,
        tokenizer=None,
    ) -> None:
        """compress 완료 후 저장."""
        save_pretrained(
            model,
            save_dir,
            scheme=self.scheme,
            ignore=self.ignore,
            tokenizer=tokenizer,
        )
