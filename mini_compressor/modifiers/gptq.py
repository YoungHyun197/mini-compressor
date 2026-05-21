# GPTQModifier — Hessian 기반 layer-wise weight 업데이트 (GPTQ 실구현)
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from ..fake_quant_linear import FakeQuantLinear
from ..schemes import QuantizationScheme, QuantizationSpec
from .base import BaseModifier
from .quantization import QuantizationMixin


class GPTQModifier(QuantizationMixin, BaseModifier):
    """GPTQ: Hessian 기반 layer-wise weight 업데이트로 W4A16 정확도 향상.

    RTN보다 낮은 quantization error를 달성하는 post-training quantization 알고리즘.
    calibration forward pass로 각 layer의 입력 Hessian H = 2·XᵀX를 수집하고,
    column 단위로 양자화 오차를 이후 column에 전파해 per-layer reconstruction error를 최소화한다.

    Lifecycle:
        initialize(model): nn.Linear → FakeQuantLinear 교체 (scale 미산출 — GPTQ에서 채움).
        calibrate(dataloader): Hessian 수집 후 layer별 GPTQ로 weight + scale 산출.
        finalize(): QuantizationMixin.finalize — observer 제거.

    Reference:
        GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
        https://arxiv.org/abs/2210.17323
    """

    def __init__(
        self,
        scheme: QuantizationScheme,
        targets: Optional[List[str]] = None,
        ignore: Optional[List[str]] = None,
        dampening_frac: float = 0.01,
    ):
        super().__init__(scheme, targets, ignore, compute_scales=False)
        self.dampening_frac = dampening_frac

    def calibrate(
        self,
        dataloader: Iterable,
        num_samples: Optional[int] = None,
    ) -> None:
        """Hessian을 수집하고 layer별 GPTQ로 weight를 업데이트한다.

        Args:
            dataloader: 캘리브레이션 배치. 각 배치는 dict, tuple, Tensor 중 하나.
            num_samples: 사용할 최대 배치 수. None이면 전체 사용.
        """
        if self.model is None:
            raise RuntimeError("initialize(model)을 먼저 호출해야 합니다.")

        weight_spec = self.scheme.weight
        if weight_spec.granularity != "per_group":
            raise ValueError(
                f"GPTQModifier는 per_group weight만 지원합니다 (요청: '{weight_spec.granularity}'). "
                "W4A16 scheme을 사용하세요."
            )

        # layer별 Hessian 수집
        hessians: dict[int, torch.Tensor] = {}
        handles: list = []

        for mod in self.model.modules():
            if isinstance(mod, FakeQuantLinear):
                mid = id(mod)
                hessians[mid] = torch.zeros(
                    mod.in_features, mod.in_features,
                    device=mod.weight.device, dtype=torch.float32,
                )
                handles.append(
                    mod.register_forward_pre_hook(_make_hessian_hook(mid, hessians))
                )

        if not hessians:
            return  # 교체 대상 없음

        dataloader = list(dataloader)
        try:
            self.model.eval()
            n = 0
            with torch.no_grad():
                for batch in dataloader:
                    if num_samples is not None and n >= num_samples:
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
                    "GPTQModifier는 calibration 데이터가 필요합니다. "
                    "dataloader가 비어있거나 num_samples=0입니다."
                )
        finally:
            for h in handles:
                h.remove()

        # layer별 GPTQ 적용
        for mod in self.model.modules():
            if isinstance(mod, FakeQuantLinear):
                H = hessians[id(mod)]
                _gptq_quantize(mod, H, weight_spec, self.dampening_frac)

        hessians.clear()


def _make_hessian_hook(mid: int, hessians: dict):
    """forward pre-hook: 입력 X를 받아 H += 2·XᵀX를 누적한다."""
    def hook(module: nn.Module, args: tuple) -> None:
        x = args[0].detach().float()
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        hessians[mid] += 2.0 * x.T @ x
    return hook


def _gptq_quantize(
    module: FakeQuantLinear,
    H: torch.Tensor,
    weight_spec: QuantizationSpec,
    dampening_frac: float,
) -> None:
    """한 FakeQuantLinear의 weight를 GPTQ 알고리즘으로 업데이트한다.

    원논문의 per-layer 공식을 그대로 따른다:
      1. dead column 처리 + Hessian dampening
      2. Cholesky로 H⁻¹ upper triangular 계산
      3. group 단위 column loop: 양자화 → 오차 전파
    """
    device = module.weight.device
    W = module.weight.data.float().clone()  # [out, in] — float32 작업 사본
    out_features, in_features = W.shape
    group_size = weight_spec.group_size
    n_groups = in_features // group_size

    qmax = 2 ** (weight_spec.num_bits - 1) - 1
    qmin = -qmax if weight_spec.symmetric else -(2 ** (weight_spec.num_bits - 1))

    H = H.to(device)

    # dead column: 한 번도 활성화되지 않은 입력 채널 → weight를 0으로, H 대각을 1로
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    # Hessian dampening + Cholesky inverse
    damp = dampening_frac * torch.diag(H).mean()
    H.diagonal().add_(damp)

    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        Hinv_upper = torch.linalg.cholesky(Hinv, upper=True)
    except torch.linalg.LinAlgError:
        # 수치 불안정 fallback: 추가 regularization
        H += torch.eye(in_features, device=device) * damp
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        Hinv_upper = torch.linalg.cholesky(Hinv, upper=True)

    Q = torch.zeros_like(W)  # 양자화 결과 (dequantized float)
    scale = torch.zeros(out_features, n_groups, device=device, dtype=torch.float32)
    zp = torch.zeros(out_features, n_groups, device=device, dtype=torch.float32)

    for g in range(n_groups):
        c0, c1 = g * group_size, (g + 1) * group_size

        Wg = W[:, c0:c1]          # [out, gs] — 이 group의 현재 (부분 보정된) weight
        Hinv_g = Hinv_upper[c0:c1, c0:c1]  # [gs, gs]

        # group scale: absmax per output channel (현재 working weight 기준)
        if weight_spec.symmetric:
            w_abs_max = Wg.abs().amax(dim=-1).clamp(min=1e-8)  # [out]
            s_g = w_abs_max / qmax                              # [out]
            z_g = torch.zeros_like(s_g)
        else:
            w_min = torch.minimum(Wg.amin(dim=-1), torch.zeros(out_features, device=device))
            w_max = torch.maximum(Wg.amax(dim=-1), torch.zeros(out_features, device=device))
            s_g = ((w_max - w_min) / (qmax - qmin)).clamp(min=1e-8)
            z_g = torch.clamp(torch.round(qmin - w_min / s_g), qmin, qmax)

        scale[:, g] = s_g
        zp[:, g] = z_g

        Err_g = torch.zeros_like(Wg)  # [out, gs] — group 내 오차 누적

        for j in range(c1 - c0):
            col = c0 + j
            w_col = W[:, col]           # [out]
            d = Hinv_g[j, j]            # scalar

            q_int = torch.clamp(torch.round(w_col / s_g + z_g), qmin, qmax)
            q_fake = (q_int - z_g) * s_g   # [out]

            Q[:, col] = q_fake
            err = (w_col - q_fake) / d  # [out]
            Err_g[:, j] = err

            # intra-group 오차 전파: 남은 columns에 분산
            if j + 1 < (c1 - c0):
                W[:, col + 1:c1] -= err.unsqueeze(1) * Hinv_g[j, j + 1:].unsqueeze(0)

        # inter-group 오차 전파: 이후 groups에 분산
        if c1 < in_features:
            W[:, c1:] -= Err_g @ Hinv_upper[c0:c1, c1:]

    # 결과 저장
    module.weight.data = Q.to(module.weight.dtype)
    module.weight_scale = scale
    module.weight_zero_point = zp
