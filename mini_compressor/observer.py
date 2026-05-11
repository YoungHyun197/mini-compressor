# activation 통계 수집 및 scale/zero_point 계산 observer 모듈
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple

from .schemes import QuantizationSpec


class BaseObserver(nn.Module):
    """Observer 공통 인터페이스."""

    def update(self, x: torch.Tensor) -> None:
        raise NotImplementedError

    def compute_scale_zp(self, spec: QuantizationSpec) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    @staticmethod
    def _scale_zp_from_range(
        min_val: torch.Tensor,
        max_val: torch.Tensor,
        spec: QuantizationSpec,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """min/max → scale, zero_point 공통 계산 (zero 포함 보장)."""
        min_val = torch.minimum(min_val, torch.zeros_like(min_val))
        max_val = torch.maximum(max_val, torch.zeros_like(max_val))

        qmax = 2 ** (spec.num_bits - 1) - 1
        qmin = -qmax if spec.symmetric else -(2 ** (spec.num_bits - 1))

        scale = (max_val - min_val) / (qmax - qmin)
        scale = torch.clamp(scale, min=1e-8)

        if spec.symmetric:
            zero_point = torch.zeros_like(scale)
        else:
            zero_point = torch.clamp(
                torch.round(qmin - min_val / scale),
                qmin, qmax,
            )
        return scale, zero_point


class MinMaxObserver(BaseObserver):
    """running min/max 수집 — multi-GPU: all_reduce(MIN/MAX) 한 줄로 동기화 가능."""

    def __init__(self):
        super().__init__()
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def update(self, x: torch.Tensor) -> None:
        self.min_val = torch.minimum(self.min_val, x.detach().min())
        self.max_val = torch.maximum(self.max_val, x.detach().max())

    def compute_scale_zp(self, spec: QuantizationSpec) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._scale_zp_from_range(self.min_val, self.max_val, spec)

    def reset(self) -> None:
        self.min_val.fill_(float("inf"))
        self.max_val.fill_(float("-inf"))


class PercentileObserver(BaseObserver):
    """percentile 클리핑으로 outlier 제거 — multi-GPU: all_gather 후 전체 분포에서 계산."""

    def __init__(self, percentile: float = 99.9):
        super().__init__()
        self.percentile = percentile
        self._data: list[torch.Tensor] = []

    def update(self, x: torch.Tensor) -> None:
        self._data.append(x.detach().flatten().cpu())

    def compute_scale_zp(self, spec: QuantizationSpec) -> Tuple[torch.Tensor, torch.Tensor]:
        all_data = torch.cat(self._data)
        lower = (100.0 - self.percentile) / 100.0
        upper = self.percentile / 100.0
        min_val = torch.quantile(all_data, lower)
        max_val = torch.quantile(all_data, upper)
        return self._scale_zp_from_range(min_val, max_val, spec)

    def reset(self) -> None:
        self._data.clear()


class MSEObserver(BaseObserver):
    """grid-search로 MSE를 최소화하는 scale 탐색 — multi-GPU: 로컬 탐색 후 all_reduce(argmin)."""

    def __init__(self, num_grids: int = 100):
        super().__init__()
        self.num_grids = num_grids
        self._data: list[torch.Tensor] = []

    def update(self, x: torch.Tensor) -> None:
        self._data.append(x.detach().flatten().cpu())

    def compute_scale_zp(self, spec: QuantizationSpec) -> Tuple[torch.Tensor, torch.Tensor]:
        all_data = torch.cat(self._data)
        qmax = 2 ** (spec.num_bits - 1) - 1
        qmin = -qmax if spec.symmetric else -(2 ** (spec.num_bits - 1))

        min_val = torch.minimum(all_data.min(), torch.tensor(0.0))
        max_val = torch.maximum(all_data.max(), torch.tensor(0.0))

        best_scale, best_zp = self._scale_zp_from_range(min_val, max_val, spec)
        best_mse = float("inf")

        for alpha in torch.linspace(0.8, 1.0, self.num_grids):
            cmin = min_val * alpha
            cmax = max_val * alpha
            scale = torch.clamp((cmax - cmin) / (qmax - qmin), min=1e-8)

            if spec.symmetric:
                zp = torch.zeros_like(scale)
            else:
                zp = torch.clamp(torch.round(qmin - cmin / scale), qmin, qmax)

            q = torch.clamp(torch.round(all_data / scale + zp), qmin, qmax)
            mse = ((all_data - (q - zp) * scale) ** 2).mean().item()

            if mse < best_mse:
                best_mse = mse
                best_scale, best_zp = scale, zp

        return best_scale, best_zp

    def reset(self) -> None:
        self._data.clear()


class KLDivergenceObserver(BaseObserver):
    """histogram 기반 KL divergence 최소화로 clip range 탐색 — multi-GPU: histogram bins all_reduce(SUM)."""

    def __init__(self, num_bins: int = 2048):
        super().__init__()
        self.num_bins = num_bins
        self._data: list[torch.Tensor] = []

    def update(self, x: torch.Tensor) -> None:
        self._data.append(x.detach().flatten().cpu())

    def compute_scale_zp(self, spec: QuantizationSpec) -> Tuple[torch.Tensor, torch.Tensor]:
        all_data = torch.cat(self._data).float()
        qmax = 2 ** (spec.num_bits - 1) - 1
        qmin = -qmax if spec.symmetric else -(2 ** (spec.num_bits - 1))
        num_levels = qmax - qmin + 1

        abs_max = all_data.abs().max().item()
        if abs_max < 1e-8:
            return torch.tensor(1e-8), torch.zeros(1)

        hist = torch.histc(all_data, bins=self.num_bins, min=-abs_max, max=abs_max)
        hist = hist + 1e-8

        best_kl = float("inf")
        best_clip_max = abs_max

        for i in range(self.num_bins, self.num_bins // 2, -1):
            clip_max = abs_max * i / self.num_bins
            lo = self.num_bins - i if spec.symmetric else 0
            hi = i - 1

            p = hist.clone()
            p[lo] += p[:lo].sum()
            p[:lo] = 0
            p[hi] += p[hi + 1:].sum()
            p[hi + 1:] = 0
            p = p / p.sum()

            num_range_bins = hi - lo + 1
            bins_per_level = max(num_range_bins // num_levels, 1)
            q = torch.zeros_like(p)
            for lv in range(num_levels):
                s = lo + lv * bins_per_level
                e = min(s + bins_per_level, hi + 1)
                if s >= e:
                    continue
                level_sum = p[s:e].sum()
                nonzero = (p[s:e] > 1e-10).float().sum()
                if nonzero > 0:
                    q[s:e] = torch.where(
                        p[s:e] > 1e-10,
                        level_sum / nonzero,
                        torch.zeros_like(p[s:e]),
                    )

            total_q = q.sum()
            if total_q < 1e-8:
                continue
            q = q / total_q

            mask = (p > 1e-10) & (q > 1e-10)
            kl = (p[mask] * torch.log(p[mask] / q[mask])).sum().item()
            if kl < best_kl:
                best_kl = kl
                best_clip_max = clip_max

        min_val = torch.tensor(-best_clip_max if spec.symmetric else min(all_data.min().item(), 0.0))
        max_val = torch.tensor(best_clip_max)
        return self._scale_zp_from_range(min_val, max_val, spec)

    def reset(self) -> None:
        self._data.clear()


OBSERVER_REGISTRY: dict[str, type[BaseObserver]] = {
    "minmax": MinMaxObserver,
    "percentile": PercentileObserver,
    "mse": MSEObserver,
    "kl_divergence": KLDivergenceObserver,
}


def build_observer(calibration_method: str, **kwargs) -> BaseObserver:
    if calibration_method not in OBSERVER_REGISTRY:
        raise ValueError(
            f"Unknown calibration_method '{calibration_method}'. "
            f"Choose from {list(OBSERVER_REGISTRY)}"
        )
    return OBSERVER_REGISTRY[calibration_method](**kwargs)
