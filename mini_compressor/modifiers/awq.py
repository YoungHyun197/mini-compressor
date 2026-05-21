# AWQModifier — activation magnitude 기반 grid-search scaling으로 W4A16 정확도 향상
from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .base import BaseModifier
from ._pair_utils import _find_smooth_pairs


class AWQModifier(BaseModifier):
    """AWQ: activation magnitude 기반 grid-search scaling으로 W4A16 정확도 향상.

    수식:
        y = x @ W.T = (x / s) @ (W * s).T
        s_x = channel-wise mean(|X|)  (calibration 누적 평균)
        s = (s_x / mean(s_x))^alpha   (alpha는 grid search로 최적화)
    norm.weight /= s, linear.weight *= s 적용.

    SmoothQuant와의 차이:
        SmoothQuant: alpha=0.5 고정, s_x = max(|X|), weight 분포(w_max)도 반영
        AWQ: alpha grid search (0,1], s_x = mean(|X|), INT4 quantization error 직접 최소화

    Lifecycle:
        initialize(model): norm → linear group pair 자동 탐색 + activation 수집용 hook 등록.
        calibrate(dataloader): channel-wise activation mean 수집 → grid search → weight 변형.
        finalize(): hook 해제 + 통계 buffer 정리.

    Reference:
        AWQ: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        n_grid: int = 20,
        group_size: int = 128,
        num_samples: Optional[int] = None,
    ):
        if n_grid < 1:
            raise ValueError(f"n_grid must be >= 1, got {n_grid}")
        self.n_grid = n_grid
        self.group_size = group_size
        self.num_samples = num_samples
        self.model: Optional[nn.Module] = None
        self._pairs: List[Tuple[nn.Module, List[nn.Linear]]] = []
        self._hooks: list = []
        self._x_sum: dict = {}   # id(first_linear) → running sum of |x| per channel (float32)
        self._x_count: dict = {}  # id(first_linear) → total sample count

    def initialize(self, model: nn.Module) -> None:
        """norm → linear group을 자동 탐색하고 첫 linear에 forward pre-hook을 단다."""
        self.model = model
        self._pairs = _find_smooth_pairs(model)
        if not self._pairs:
            return
        for _norm, linears in self._pairs:
            fl = linears[0]
            key = id(fl)
            self._x_sum[key] = None
            self._x_count[key] = 0
            hook = fl.register_forward_pre_hook(self._make_hook(fl))
            self._hooks.append(hook)

    def calibrate(
        self,
        dataloader: Iterable,
        num_samples: Optional[int] = None,
    ) -> None:
        """forward pass로 activation mean 누적 → grid search → weight 변형."""
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
                "AWQModifier.calibrate(): dataloader가 비어있어 activation 통계를 수집할 수 없습니다."
            )

        for norm, linears in self._pairs:
            fl = linears[0]
            key = id(fl)
            x_sum = self._x_sum[key]
            x_count = self._x_count[key]
            if x_sum is None or x_count == 0:
                continue

            # channel-wise activation mean (float32 정밀도 유지)
            s_x = (x_sum / x_count).clamp(min=1e-5)
            s_x_mean = s_x.mean().clamp(min=1e-5)

            # grid search: alpha ∈ (0, 1], n_grid steps
            alphas = torch.linspace(0.0, 1.0, self.n_grid + 1)[1:]
            best_err = float("inf")
            best_s = None

            for alpha in alphas:
                s = (s_x / s_x_mean).pow(alpha.item()).clamp(min=1e-5)
                err = sum(
                    _quant_error(
                        lin.weight.detach().to(torch.float32), s, s_x, self.group_size
                    )
                    for lin in linears
                )
                if err < best_err:
                    best_err = err
                    best_s = s

            if best_s is None:
                continue

            # apply: norm.weight /= s, linear.weight *= s (in_features 차원 broadcast)
            s_norm = best_s.to(dtype=norm.weight.dtype, device=norm.weight.device)
            norm.weight.data.div_(s_norm)
            # LayerNorm bias도 동일한 등가 변환에 포함 (RMSNorm은 bias 없으므로 이 분기 미작동)
            if getattr(norm, "bias", None) is not None:
                norm.bias.data.div_(s_norm)
            for lin in linears:
                s_lin = best_s.to(dtype=lin.weight.dtype, device=lin.weight.device)
                lin.weight.data.mul_(s_lin.unsqueeze(0))

    def finalize(self) -> None:
        """hook 해제 + 통계 buffer 정리."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._x_sum.clear()
        self._x_count.clear()
        self._pairs = []

    def _make_hook(self, linear: nn.Linear):
        key = id(linear)

        def hook(_module, args):
            x = args[0] if isinstance(args, tuple) else args
            x_abs = x.detach().abs().reshape(-1, x.shape[-1])
            cur_sum = x_abs.sum(dim=0).to(torch.float32)
            cur_count = x_abs.shape[0]
            prev = self._x_sum[key]
            self._x_sum[key] = cur_sum if prev is None else prev + cur_sum
            self._x_count[key] = self._x_count[key] + cur_count

        return hook


def _int4_fake_quant(w: torch.Tensor, group_size: int) -> torch.Tensor:
    """per-group symmetric INT4 fake quantization — AWQ grid search 내부 전용."""
    rows, cols = w.shape
    actual_gs = group_size if cols % group_size == 0 else cols
    w_g = w.reshape(rows, -1, actual_gs)
    qmax = 7.0
    s_g = w_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    q = (w_g / s_g).round().clamp(-qmax, qmax)
    return (q * s_g).reshape(rows, cols)


def _quant_error(
    w: torch.Tensor,
    s: torch.Tensor,
    s_x: torch.Tensor,
    group_size: int,
) -> float:
    """s를 적용 후 INT4 fake quant → 복원 오차를 activation magnitude로 가중합산한다.

    입력:  w [out, in] float32, s [in] float32, s_x [in] float32
    출력:  scalar float (낮을수록 좋음)
    """
    w_scaled = w * s.unsqueeze(0)          # [out, in]: in-channel 별 scaling
    w_q = _int4_fake_quant(w_scaled, group_size)
    w_dq = w_q / s.unsqueeze(0)            # unscale: 실제 추론 값 복원
    return ((w_dq - w).pow(2) * s_x.unsqueeze(0)).mean().item()
