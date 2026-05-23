# Multi-GPU observer sync 검증 — gloo 백엔드 2-프로세스로 all_reduce 실검증
import dataclasses
import os
import socket
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mini_compressor.observer import build_observer
from mini_compressor.schemes import W8A8

_WORLD = 2
_ACT_SPEC = W8A8.activation  # int8, per_tensor, asymmetric
_METHODS = ["minmax", "percentile", "mse"]


def _spec(method: str):
    """per_tensor activation spec에 calibration_method만 바꿔 끼운다."""
    return dataclasses.replace(_ACT_SPEC, calibration_method=method)


def _find_free_port() -> int:
    try:
        s = socket.socket()
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port
    except (PermissionError, OSError) as e:
        pytest.skip(f"소켓 생성 불가 (sandbox 환경): {e}")


def _rank_data(rank: int) -> torch.Tensor:
    """rank마다 결정적으로 다른 calibration 데이터 — 모든 worker가 재현 가능.

    (rank + 1) 배율로 rank별 분포 폭을 다르게 해, sync가 빠지면 결과가 어긋나도록 한다.
    """
    g = torch.Generator().manual_seed(1000 + rank)
    return torch.randn(512, generator=g) * (rank + 1)


def _run_observer_sync(rank: int, port: int) -> None:
    """spawn된 각 프로세스: 모든 observer의 분산 sync 결과가 단일 프로세스 전역 결과와 일치하는지 검증.

    MinMax: min/max는 결합적이므로 all_reduce(MIN/MAX)로 정확히 일치.
    Percentile/MSE: histogram 기반이므로 단일 프로세스와 동일한 연산 경로를 거쳐
    all_reduce(SUM) 후 결과가 일치한다. FP 누적 차이를 고려해 atol을 약간 완화.
    """
    dist.init_process_group(
        backend="gloo", rank=rank, world_size=_WORLD,
        init_method=f"tcp://127.0.0.1:{port}",
    )
    try:
        for method in _METHODS:
            spec = _spec(method)
            # 분산 경로 — 각 rank는 자기 데이터만 update 후 sync
            obs = build_observer(spec)
            obs.update(_rank_data(rank))
            obs.sync()
            scale, zp = obs.compute_scale_zp()

            # 기준 경로 — 전체 rank 데이터를 단일 observer로
            ref = build_observer(spec)
            for r in range(_WORLD):
                ref.update(_rank_data(r))
            ref_scale, ref_zp = ref.compute_scale_zp()

            atol = 1e-5 if method == "minmax" else 1e-4
            assert torch.allclose(scale, ref_scale, atol=atol), (
                f"{method} rank{rank}: scale {scale} != ref {ref_scale}"
            )
            assert torch.allclose(zp, ref_zp, atol=atol), (
                f"{method} rank{rank}: zp {zp} != ref {ref_zp}"
            )
    finally:
        dist.destroy_process_group()


def test_observer_sync_matches_single_process():
    """gloo 2-프로세스 sync 결과가 전체 데이터 단일 프로세스 결과와 일치해야 한다 (4개 observer)."""
    if not dist.is_available():
        pytest.skip("torch.distributed 미지원 빌드")
    mp.spawn(_run_observer_sync, args=(_find_free_port(),), nprocs=_WORLD, join=True)


def test_sync_noop_without_distributed():
    """분산 init 없이 sync() 호출 시 통계가 그대로 유지돼야 한다 (단일 GPU no-op)."""
    obs = build_observer(_ACT_SPEC)  # calibration_method 기본값 minmax
    obs.update(torch.tensor([-2.0, 3.0]))
    obs.sync()
    assert obs.min_val.item() == -2.0
    assert obs.max_val.item() == 3.0
