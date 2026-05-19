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


def _sync_data(data: list[torch.Tensor]) -> list[torch.Tensor]:
    """rank별 raw 통계 데이터 리스트를 all_gather해 전역 리스트로 합친다.

    Percentile/MSE는 raw 데이터를 모아 compute_scale_zp에서 비결합적 계산
    (percentile·grid-search)을 한다. 부분 통계로는 병합이 불가능하므로
    raw 데이터를 전 rank가 공유해 동일한 전역 결과를 내도록 한다.
    분산 환경이 아니면 입력을 그대로 돌려준다 (no-op).
    """
    if not _dist_active():
        return data
    gathered: list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, data)
    return [t for rank_data in gathered for t in rank_data]


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
    """percentile 클리핑으로 outlier 제거 — multi-GPU: all_gather 후 전체 분포에서 계산."""

    def __init__(self, spec: QuantizationSpec, percentile: float = 99.9):
        super().__init__(spec)
        self.percentile = percentile
        self._data: list[torch.Tensor] = []

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().cpu()
        # per_tensor는 배치마다 shape이 달라 flatten해 모으고,
        # per_channel/per_group은 채널 구조를 보존해야 단위별 통계가 가능하다.
        self._data.append(x.flatten() if self.spec.granularity == "per_tensor" else x)

    def compute_scale_zp(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.spec.granularity == "per_tensor":
            raw = torch.cat(self._data)
        else:
            raw = torch.cat(self._data, dim=0)
        units = _to_units(raw, self.spec)
        lower = (100.0 - self.percentile) / 100.0
        upper = self.percentile / 100.0
        min_val = torch.quantile(units, lower, dim=-1)
        max_val = torch.quantile(units, upper, dim=-1)
        return self._scale_zp_from_range(min_val, max_val)

    def reset(self) -> None:
        self._data.clear()

    def sync(self) -> None:
        """rank별 raw 데이터를 all_gather해 전역 분포로 합친다."""
        self._data = _sync_data(self._data)


class MSEObserver(BaseObserver):
    """grid-search로 양자화 MSE를 최소화하는 clip range 탐색 — 단위마다 독립 탐색.

    multi-GPU: raw 데이터를 all_gather한 뒤 전역 분포에서 탐색.
    """

    def __init__(self, spec: QuantizationSpec, num_grids: int = 100):
        super().__init__(spec)
        self.num_grids = num_grids
        self._data: list[torch.Tensor] = []

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().cpu()
        self._data.append(x.flatten() if self.spec.granularity == "per_tensor" else x)

    def compute_scale_zp(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.spec.granularity == "per_tensor":
            raw = torch.cat(self._data)
        else:
            raw = torch.cat(self._data, dim=0)
        units = _to_units(raw, self.spec)  # (단위..., k)

        qmax = 2 ** (self.spec.num_bits - 1) - 1
        qmin = -qmax if self.spec.symmetric else -(2 ** (self.spec.num_bits - 1))

        umin = units.amin(dim=-1)
        umax = units.amax(dim=-1)

        best_scale, best_zp = self._scale_zp_from_range(umin, umax)
        best_err = torch.full_like(best_scale, float("inf"))

        # alpha로 단위별 min/max를 함께 줄여 clip range 후보를 만든다.
        for alpha in torch.linspace(0.8, 1.0, self.num_grids).tolist():
            scale, zp = self._scale_zp_from_range(umin * alpha, umax * alpha)
            s = scale.unsqueeze(-1)
            z = zp.unsqueeze(-1)
            q = torch.clamp(torch.round(units / s + z), qmin, qmax)
            err = ((units - (q - z) * s) ** 2).mean(dim=-1)

            better = err < best_err
            best_scale = torch.where(better, scale, best_scale)
            best_zp = torch.where(better, zp, best_zp)
            best_err = torch.where(better, err, best_err)

        return best_scale, best_zp

    def reset(self) -> None:
        self._data.clear()

    def sync(self) -> None:
        """rank별 raw 데이터를 all_gather해 전역 분포로 합친다."""
        self._data = _sync_data(self._data)


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
