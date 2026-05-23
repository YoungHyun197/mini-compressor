# weight/activation 통계 수집 및 scale/zero_point 계산 observer 모듈
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Tuple

from .schemes import QuantizationSpec


def _dist_active() -> bool:
    """torch.distributed가 초기화돼 있고 world_size > 1일 때만 rank 동기화가 필요하다."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _to_units(t: torch.Tensor, spec: QuantizationSpec) -> torch.Tensor:
    """텐서를 양자화 단위 기준 (단위..., 단위내_원소) 형태로 재배열한다.

    scale은 단위 하나당 1개다. 마지막 축이 단위 내부 원소가 되도록 정리하면
    amin/amax/quantile(dim=-1) 한 줄로 granularity와 무관하게 단위별 통계를 뽑을 수 있다.
        per_tensor  : (N,)         — 단위 1개
        per_channel : (out, in)    — 출력 채널마다 단위
        per_group   : (out, G, gs) — 채널×그룹마다 단위
    """
    g = spec.granularity
    if g == "per_tensor":
        return t.reshape(-1)
    if g == "per_channel":
        return t.reshape(t.shape[0], -1)
    if g == "per_group":
        return t.reshape(t.shape[0], -1, spec.group_size)
    raise ValueError(
        f"observer가 지원하지 않는 granularity: '{g}'. "
        f"per_tensor / per_channel / per_group 중 하나여야 합니다."
    )


def _build_histogram(
    units: torch.Tensor,
    h_min: torch.Tensor,
    h_max: torch.Tensor,
    num_bins: int,
) -> torch.Tensor:
    """units (..., k) → histogram (..., num_bins) — GPU에서 직접 실행."""
    step = ((h_max - h_min) / num_bins).clamp(min=1e-8)
    idx = ((units - h_min.unsqueeze(-1)) / step.unsqueeze(-1)).floor().long()
    idx = idx.clamp(0, num_bins - 1)
    hist = torch.zeros(*units.shape[:-1], num_bins, dtype=torch.float32, device=units.device)
    hist.scatter_add_(-1, idx, torch.ones_like(units, dtype=torch.float32))
    return hist


def _resize_histogram(
    hist: torch.Tensor,
    old_min: torch.Tensor,
    old_max: torch.Tensor,
    new_min: torch.Tensor,
    new_max: torch.Tensor,
) -> torch.Tensor:
    """기존 histogram의 bin 카운트를 새 범위에 맞는 bin으로 선형 재분배한다.

    rank 간 범위가 다를 때 histogram을 공통 범위로 정렬한 뒤 all_reduce할 수 있도록 한다.
    """
    num_bins = hist.shape[-1]
    device = hist.device
    old_step = ((old_max - old_min) / num_bins).clamp(min=1e-8)
    centers = old_min.unsqueeze(-1) + (torch.arange(num_bins, device=device) + 0.5) * old_step.unsqueeze(-1)
    new_step = ((new_max - new_min) / num_bins).clamp(min=1e-8)
    new_idx = ((centers - new_min.unsqueeze(-1)) / new_step.unsqueeze(-1)).floor().long()
    new_idx = new_idx.clamp(0, num_bins - 1)
    new_hist = torch.zeros_like(hist)
    new_hist.scatter_add_(-1, new_idx, hist)
    return new_hist


class BaseObserver(nn.Module):
    """Observer 공통 인터페이스 — weight·activation이 동일 추상화를 공유한다.

    생성 시점에 spec(QuantizationSpec)을 받아 granularity를 인지하므로,
    per_tensor activation과 per_channel/per_group weight를 같은 클래스로 처리한다.
    weight는 정적 텐서라 update()를 1회 호출, activation은 calibration forward마다 호출한다.
    """

    def __init__(self, spec: QuantizationSpec):
        super().__init__()
        self.spec = spec

    def update(self, x: torch.Tensor) -> None:
        raise NotImplementedError

    def compute_scale_zp(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def sync(self) -> None:
        """Multi-GPU calibration에서 rank 간 통계를 동기화한다 (분산 환경 아니면 no-op)."""
        raise NotImplementedError

    def _scale_zp_from_range(
        self,
        min_val: torch.Tensor,
        max_val: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """min/max → scale, zero_point 공통 계산 (단위별 broadcast, zero 포함 보장)."""
        spec = self.spec
        min_val = torch.minimum(min_val, torch.zeros_like(min_val))
        max_val = torch.maximum(max_val, torch.zeros_like(max_val))

        qmax = 2 ** (spec.num_bits - 1) - 1
        qmin = -qmax if spec.symmetric else -(2 ** (spec.num_bits - 1))

        if spec.symmetric:
            # 대칭: 표현 범위가 [-qmax·s, qmax·s]이므로 s = max(|min|,|max|)/qmax.
            # (min-max 폭/2qmax이 아니다 — 한쪽 꼬리가 길면 그쪽이 잘린다.)
            abs_max = torch.maximum(max_val, -min_val)
            scale = torch.clamp(abs_max / qmax, min=1e-8)
            zero_point = torch.zeros_like(scale)
        else:
            scale = torch.clamp((max_val - min_val) / (qmax - qmin), min=1e-8)
            zero_point = torch.clamp(
                torch.round(qmin - min_val / scale), qmin, qmax,
            )
        return scale, zero_point


class MinMaxObserver(BaseObserver):
    """running min/max 수집 — multi-GPU: all_reduce(MIN/MAX) 한 줄로 동기화 가능.

    granularity에 따라 단위별 min/max 벡터를 유지한다 (per_tensor면 0-dim 스칼라).
    """

    def __init__(self, spec: QuantizationSpec):
        super().__init__(spec)
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def update(self, x: torch.Tensor) -> None:
        units = _to_units(x.detach(), self.spec)
        cur_min = units.amin(dim=-1)
        cur_max = units.amax(dim=-1)
        # 최초 update 시 0-dim inf 버퍼가 단위별 shape으로 broadcast된다.
        self.min_val = torch.minimum(self.min_val.to(cur_min.device), cur_min)
        self.max_val = torch.maximum(self.max_val.to(cur_max.device), cur_max)

    def compute_scale_zp(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._scale_zp_from_range(self.min_val, self.max_val)

    def reset(self) -> None:
        self.min_val.fill_(float("inf"))
        self.max_val.fill_(float("-inf"))

    def sync(self) -> None:
        """rank 간 min/max를 all_reduce로 병합 — min·max는 결합적이라 한 줄로 정확히 동기화된다."""
        if not _dist_active():
            return
        dist.all_reduce(self.min_val, op=dist.ReduceOp.MIN)
        dist.all_reduce(self.max_val, op=dist.ReduceOp.MAX)


class PercentileObserver(BaseObserver):
    """histogram 기반 percentile 클리핑 — GPU 상주, NCCL all_reduce(SUM) 동기화.

    raw tensor를 누적하지 않고 고정 크기(NUM_BINS) histogram을 GPU에 유지한다.
    메모리는 sample 수와 무관하게 O(units × NUM_BINS)로 고정된다.
    multi-GPU sync는 (1) global range all_reduce → (2) 로컬 histogram resize →
    (3) histogram all_reduce(SUM) 세 단계로 NCCL 통신만 사용한다.
    """

    NUM_BINS: int = 2048

    def __init__(self, spec: QuantizationSpec, percentile: float = 99.9):
        super().__init__(spec)
        self.percentile = percentile
        self.register_buffer("_hist", None)
        self.register_buffer("_hist_min", None)
        self.register_buffer("_hist_max", None)

    def update(self, x: torch.Tensor) -> None:
        x = x.detach()
        units = _to_units(x, self.spec)
        cur_min = units.amin(dim=-1)
        cur_max = units.amax(dim=-1)

        if self._hist is None:
            self._buffers["_hist_min"] = cur_min.clone()
            self._buffers["_hist_max"] = cur_max.clone()
            self._buffers["_hist"] = _build_histogram(units, cur_min, cur_max, self.NUM_BINS)
        else:
            new_min = torch.minimum(self._hist_min, cur_min)
            new_max = torch.maximum(self._hist_max, cur_max)
            if not (new_min.equal(self._hist_min) and new_max.equal(self._hist_max)):
                self._buffers["_hist"] = _resize_histogram(
                    self._hist, self._hist_min, self._hist_max, new_min, new_max
                )
                self._buffers["_hist_min"] = new_min
                self._buffers["_hist_max"] = new_max
            self._buffers["_hist"] = self._hist + _build_histogram(
                units, self._hist_min, self._hist_max, self.NUM_BINS
            )

    def compute_scale_zp(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._hist is None:
            raise RuntimeError("update()를 최소 한 번 호출한 뒤 compute_scale_zp()를 호출해야 합니다.")
        B = self.NUM_BINS
        step = ((self._hist_max - self._hist_min) / B).clamp(min=1e-8)
        # bin edges: (..., B+1)
        bin_edges = self._hist_min.unsqueeze(-1) + torch.arange(B + 1, device=self._hist.device) * step.unsqueeze(-1)

        total = self._hist.sum(dim=-1, keepdim=True).clamp(min=1)
        cdf = self._hist.cumsum(dim=-1) / total

        lower_frac = (100.0 - self.percentile) / 100.0
        upper_frac = self.percentile / 100.0

        lower_idx = (cdf < lower_frac).sum(dim=-1).clamp(0, B)
        upper_idx = (cdf <= upper_frac).sum(dim=-1).clamp(1, B)

        min_val = bin_edges.gather(-1, lower_idx.unsqueeze(-1)).squeeze(-1)
        max_val = bin_edges.gather(-1, upper_idx.unsqueeze(-1)).squeeze(-1)
        return self._scale_zp_from_range(min_val, max_val)

    def reset(self) -> None:
        self._buffers["_hist"] = None
        self._buffers["_hist_min"] = None
        self._buffers["_hist_max"] = None

    def sync(self) -> None:
        """histogram 기반 NCCL 동기화: range 합의 → resize → histogram SUM.

        (1) all_reduce(MIN/MAX)로 global range 확정
        (2) 로컬 histogram을 global range로 resize (bin 카운트 재분배)
        (3) all_reduce(SUM)으로 histogram 합산
        → 모든 rank가 동일한 전역 분포 histogram을 보유하게 된다.
        """
        if not _dist_active() or self._hist is None:
            return
        local_min = self._hist_min.clone()
        local_max = self._hist_max.clone()
        dist.all_reduce(self._hist_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(self._hist_max, op=dist.ReduceOp.MAX)
        self._buffers["_hist"] = _resize_histogram(
            self._hist, local_min, local_max, self._hist_min, self._hist_max
        )
        dist.all_reduce(self._hist, op=dist.ReduceOp.SUM)


class MSEObserver(BaseObserver):
    """histogram 기반 MSE grid-search — GPU 상주, NCCL all_reduce(SUM) 동기화.

    raw tensor 대신 histogram 위에서 bin center별 weighted MSE를 계산한다.
    sync 방식은 PercentileObserver와 동일: range all_reduce → resize → histogram SUM.
    """

    NUM_BINS: int = 2048

    def __init__(self, spec: QuantizationSpec, num_grids: int = 100):
        super().__init__(spec)
        self.num_grids = num_grids
        self.register_buffer("_hist", None)
        self.register_buffer("_hist_min", None)
        self.register_buffer("_hist_max", None)

    def update(self, x: torch.Tensor) -> None:
        x = x.detach()
        units = _to_units(x, self.spec)
        cur_min = units.amin(dim=-1)
        cur_max = units.amax(dim=-1)

        if self._hist is None:
            self._buffers["_hist_min"] = cur_min.clone()
            self._buffers["_hist_max"] = cur_max.clone()
            self._buffers["_hist"] = _build_histogram(units, cur_min, cur_max, self.NUM_BINS)
        else:
            new_min = torch.minimum(self._hist_min, cur_min)
            new_max = torch.maximum(self._hist_max, cur_max)
            if not (new_min.equal(self._hist_min) and new_max.equal(self._hist_max)):
                self._buffers["_hist"] = _resize_histogram(
                    self._hist, self._hist_min, self._hist_max, new_min, new_max
                )
                self._buffers["_hist_min"] = new_min
                self._buffers["_hist_max"] = new_max
            self._buffers["_hist"] = self._hist + _build_histogram(
                units, self._hist_min, self._hist_max, self.NUM_BINS
            )

    def compute_scale_zp(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._hist is None:
            raise RuntimeError("update()를 최소 한 번 호출한 뒤 compute_scale_zp()를 호출해야 합니다.")
        B = self.NUM_BINS
        device = self._hist.device

        qmax = 2 ** (self.spec.num_bits - 1) - 1
        qmin_q = -qmax if self.spec.symmetric else -(2 ** (self.spec.num_bits - 1))

        step = ((self._hist_max - self._hist_min) / B).clamp(min=1e-8)
        # bin centers: (..., B)
        centers = self._hist_min.unsqueeze(-1) + (torch.arange(B, device=device) + 0.5) * step.unsqueeze(-1)

        total = self._hist.sum(dim=-1, keepdim=True).clamp(min=1)
        weight = self._hist / total  # normalized (확률 가중치)

        best_scale, best_zp = self._scale_zp_from_range(self._hist_min, self._hist_max)
        best_err = torch.full(best_scale.shape, float("inf"), device=device)

        for alpha in torch.linspace(0.8, 1.0, self.num_grids).tolist():
            scale, zp = self._scale_zp_from_range(self._hist_min * alpha, self._hist_max * alpha)
            s = scale.unsqueeze(-1)
            z = zp.unsqueeze(-1)
            q = torch.clamp(torch.round(centers / s + z), qmin_q, qmax)
            recon = (q - z) * s
            err = ((centers - recon) ** 2 * weight).sum(dim=-1)

            better = err < best_err
            best_scale = torch.where(better, scale, best_scale)
            best_zp = torch.where(better, zp, best_zp)
            best_err = torch.where(better, err, best_err)

        return best_scale, best_zp

    def reset(self) -> None:
        self._buffers["_hist"] = None
        self._buffers["_hist_min"] = None
        self._buffers["_hist_max"] = None

    def sync(self) -> None:
        """PercentileObserver와 동일: range all_reduce → resize → histogram SUM."""
        if not _dist_active() or self._hist is None:
            return
        local_min = self._hist_min.clone()
        local_max = self._hist_max.clone()
        dist.all_reduce(self._hist_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(self._hist_max, op=dist.ReduceOp.MAX)
        self._buffers["_hist"] = _resize_histogram(
            self._hist, local_min, local_max, self._hist_min, self._hist_max
        )
        dist.all_reduce(self._hist, op=dist.ReduceOp.SUM)


OBSERVER_REGISTRY: dict[str, type[BaseObserver]] = {
    "minmax": MinMaxObserver,
    "percentile": PercentileObserver,
    "mse": MSEObserver,
}


def build_observer(spec: QuantizationSpec) -> BaseObserver:
    """spec.calibration_method에 맞는 observer를 생성한다 (spec 전체를 주입).

    weight·activation 어느 쪽이든 같은 진입점을 쓴다 — 차이는 spec.granularity뿐이다.
    """
    method = spec.calibration_method
    if method not in OBSERVER_REGISTRY:
        raise ValueError(
            f"Unknown calibration_method '{method}'. "
            f"Choose from {list(OBSERVER_REGISTRY)}"
        )
    return OBSERVER_REGISTRY[method](spec)
