# SmoothQuantModifier — activation 분포를 weight 쪽으로 이관하여 양자화 오차 감소
from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .base import BaseModifier
from ._pair_utils import _find_smooth_pairs, _collect_linears, _has_affine_weight


class SmoothQuantModifier(BaseModifier):
    """SmoothQuant: activation outlier를 weight에 흡수시켜 W8A8 static의 정확도를 끌어올린다.

    수식:
        y = x @ W.T = (x / s) @ (W * s).T
        s_j = max(|x_j|)^alpha / max(|w_j|)^(1-alpha)
    norm.weight /= s, linear.weight *= s 적용.
    forward 출력은 (양자화 없이는) 원본과 수치적으로 동일하고, activation 분포만 평탄해진다.

    Lifecycle:
        initialize(model): norm → linear group pair 자동 탐색 + forward pre-hook 등록.
        calibrate(dataloader): forward로 channel-wise activation max 누적 → s 계산 → weight 변형.
        finalize(): hook 해제 + 통계 buffer 정리.

    Reference:
        SmoothQuant: https://arxiv.org/abs/2211.10438
    """

    def __init__(
        self,
        alpha: float = 0.5,
        num_samples: Optional[int] = None,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.num_samples = num_samples
        self.model: Optional[nn.Module] = None
        self._pairs: List[Tuple[nn.Module, List[nn.Linear]]] = []
        self._hooks: list = []
        self._x_max: dict[int, torch.Tensor] = {}   # id(first_linear) → channel-wise abs max

    def initialize(self, model: nn.Module) -> None:
        """norm → linear group을 자동 탐색하고 첫 linear에 forward pre-hook을 단다."""
        self.model = model
        self._pairs = _find_smooth_pairs(model)
        if not self._pairs:
            return
        for norm, linears in self._pairs:
            first_linear = linears[0]
            self._x_max[id(first_linear)] = None
            hook = first_linear.register_forward_pre_hook(self._make_hook(first_linear))
            self._hooks.append(hook)

    def calibrate(
        self,
        dataloader: Iterable,
        num_samples: Optional[int] = None,
    ) -> None:
        """forward pass로 activation 통계 누적 후 smooth factor 계산 + weight 변형."""
        if self.model is None:
            raise RuntimeError("initialize(model)을 먼저 호출해야 합니다.")
        if not self._pairs:
            return

        limit = num_samples if num_samples is not None else self.num_samples

        self.model.eval()
        n = 0
        with torch.no_grad():
            for batch in dataloader:
                if limit is not None and n >= limit:
                    break
                if isinstance(batch, (dict, Mapping)):
                    self.model(**batch)
                elif isinstance(batch, (list, tuple)):
                    self.model(*batch)
                else:
                    self.model(batch)
                n += 1

        if n == 0:
            raise ValueError(
                "SmoothQuantModifier.calibrate(): dataloader가 비어있어 activation 통계를 수집할 수 없습니다."
            )

        for norm, linears in self._pairs:
            x_max = self._x_max[id(linears[0])]
            if x_max is None:
                continue
            w_max = torch.stack(
                [lin.weight.detach().abs().amax(dim=0) for lin in linears],
                dim=0,
            ).amax(dim=0)

            # float32로 계산해 수치 안정성 확보, 적용 시 원본 dtype으로 복귀
            x_max_f = x_max.to(torch.float32).clamp(min=1e-5)
            w_max_f = w_max.to(torch.float32).clamp(min=1e-5)
            s = (x_max_f.pow(self.alpha) / w_max_f.pow(1.0 - self.alpha)).clamp(min=1e-5)

            s_norm_dtype = s.to(dtype=norm.weight.dtype, device=norm.weight.device)
            norm.weight.data.div_(s_norm_dtype)
            # LayerNorm bias도 동일한 등가 변환에 포함되어야 한다.
            # y = gamma * x_hat + beta → (gamma/s) * x_hat + (beta/s) = y / s.
            # RMSNorm은 bias가 없으므로 이 분기가 작동 안 함 (Qwen3/LLaMA).
            if getattr(norm, "bias", None) is not None:
                norm.bias.data.div_(s_norm_dtype)
            for lin in linears:
                s_lin = s.to(dtype=lin.weight.dtype, device=lin.weight.device)
                lin.weight.data.mul_(s_lin.unsqueeze(0))

    def finalize(self) -> None:
        """hook 해제 + 통계 buffer 정리."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._x_max.clear()
        self._pairs = []

    def _make_hook(self, linear: nn.Linear):
        key = id(linear)

        def hook(_module, args):
            x = args[0] if isinstance(args, tuple) else args
            # x shape: (..., in_features) → 마지막 dim 기준 채널별 abs max
            x_abs = x.detach().abs()
            flat = x_abs.reshape(-1, x_abs.shape[-1])
            cur = flat.amax(dim=0)
            prev = self._x_max[key]
            self._x_max[key] = cur if prev is None else torch.maximum(prev, cur)

        return hook


