# QuantizationMixin + QuantizationModifier — module replacement 공통 mixin과 RTN modifier
from __future__ import annotations

import fnmatch
from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from ..fake_quant_linear import FakeQuantLinear
from ..observer import build_observer
from ..schemes import QuantizationScheme
from .base import BaseModifier


class QuantizationMixin:
    """nn.Linear → FakeQuantLinear 교체 + finalize를 GPTQModifier와 공유하는 mixin.

    llm-compressor의 QuantizationMixin 패턴: QuantizationModifier와 GPTQModifier가
    동일한 module replacement 로직을 상속한다.
    initialize / finalize가 여기 구현되고, calibrate는 각 Modifier가 별도로 구현한다.
    """

    def __init__(
        self,
        scheme: QuantizationScheme,
        targets: Optional[List[str]] = None,
        ignore: Optional[List[str]] = None,
        compute_scales: bool = True,
    ):
        self.scheme = scheme
        self.targets = targets
        self.ignore = ignore or []
        self.compute_scales = compute_scales
        self.model: Optional[nn.Module] = None

    def _should_replace(self, name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        class_name = type(module).__name__
        if any(fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(class_name, pat)
               for pat in self.ignore):
            return False
        if self.targets is None:
            return True
        return any(
            fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(class_name, pat)
            for pat in self.targets
        )

    def initialize(self, model: nn.Module) -> None:
        """nn.Linear → FakeQuantLinear 교체.

        compute_scales=True (기본): weight observer로 weight scale 즉시 산출 — RTN 흐름.
        compute_scales=False: 구조만 생성, scale은 calibrate에서 채움 — GPTQ 흐름.
        """
        self.model = model
        to_replace = [
            (name, mod)
            for name, mod in model.named_modules()
            if self._should_replace(name, mod)
        ]

        for name, mod in to_replace:
            if (self.scheme.weight.granularity == "per_group"
                    and self.scheme.weight.group_size is not None
                    and mod.in_features % self.scheme.weight.group_size != 0):
                raise ValueError(
                    f"'{name}': in_features={mod.in_features}가 "
                    f"group_size={self.scheme.weight.group_size}의 배수가 아닙니다. "
                    "per_group 양자화는 in_features % group_size == 0을 요구합니다."
                )
            fql = FakeQuantLinear.from_float(mod, self.scheme)
            if self.compute_scales:
                wobs = build_observer(self.scheme.weight).to(fql.weight.device)
                wobs.update(fql.weight.detach())
                w_scale, w_zp = wobs.compute_scale_zp()
                fql.weight_scale = w_scale.to(fql.weight.device)
                fql.weight_zero_point = w_zp.to(fql.weight.device)
            *parent_path, attr = name.split(".")
            parent = model
            for part in parent_path:
                parent = getattr(parent, part)
            setattr(parent, attr, fql)

    def finalize(self) -> None:
        """observer 제거, scale buffer만 남김."""
        if self.model is None:
            raise RuntimeError("initialize(model)을 먼저 호출해야 합니다.")
        for mod in self.model.modules():
            if isinstance(mod, FakeQuantLinear):
                mod.input_observer = None


class QuantizationModifier(QuantizationMixin, BaseModifier):
    """nn.Linear를 FakeQuantLinear로 교체하고 observer로 scale을 산출한다.

    weight·activation 모두 동일한 observer 추상화를 거친다 — clip range는
    scheme.{weight,activation}.calibration_method(minmax/percentile/mse)가 정하고,
    rounding은 양쪽 다 round-to-nearest(RTN)다.

    Lifecycle:
        initialize(model): nn.Linear → FakeQuantLinear 교체 + weight observer로
                           weight scale 산출 (compute_scales=True 기본).
        calibrate(dataloader): activation observer로 input_scale/input_zero_point 계산
                               (scheme.activation이 None이거나 dynamic이면 no-op).
        finalize(): observer 제거, scale buffer만 남김.
    """

    def calibrate(
        self,
        dataloader: Iterable,
        num_samples: Optional[int] = None,
        sequential: bool = False,
    ) -> None:
        """calibration forward pass로 activation scale 계산.

        Args:
            dataloader: 캘리브레이션 배치 이터러블. 각 배치는 dict, tuple, Tensor 중 하나.
            num_samples: 사용할 최대 배치 수. None이면 전체 사용.
            sequential: True이면 layer별 순차 캘리브레이션.
                        model.model.layers 구조 필요 (Qwen3/LLaMA 등 HF CausalLM 표준).
                        peak GPU 메모리를 O(single_layer)로 줄인다.
        """
        if self.scheme.activation is None or self.scheme.activation.dynamic:
            return

        if self.model is None:
            raise RuntimeError("initialize(model)을 먼저 호출해야 합니다.")

        if sequential:
            self._calibrate_sequential(dataloader, num_samples)
            return

        self.model.eval()
        n = 0
        with torch.no_grad():
            for batch in dataloader:
                if num_samples is not None and n >= num_samples:
                    break
                if isinstance(batch, dict):
                    self.model(**batch)
                elif isinstance(batch, (list, tuple)):
                    self.model(*batch)
                else:
                    self.model(batch)
                n += 1

        for mod in self.model.modules():
            if isinstance(mod, FakeQuantLinear) and mod.input_observer is not None:
                mod.input_observer.sync()  # multi-GPU: rank 간 통계 동기화 (단일 GPU에선 no-op)
                scale, zp = mod.input_observer.compute_scale_zp()
                # scale/zp가 weight.device를 따라가도록 보정 — Percentile 등은 _data를
                # CPU로 모아 결과가 CPU에 남을 수 있음 (device_map="auto" 호환).
                mod.input_scale = scale.to(mod.weight.device)
                mod.input_zero_point = zp.to(mod.weight.device)

    def _calibrate_sequential(
        self,
        dataloader: Iterable,
        num_samples: Optional[int] = None,
    ) -> None:
        """layer별 순차 calibration — decoder layer 하나씩 GPU에 올려 peak 메모리를 줄인다.

        지원 구조: model.model.layers (Qwen3/LLaMA 등 HF CausalLM 표준).
        해당 구조가 아니면 RuntimeError를 발생시킨다.

        동작:
            1. decoder layers 전체를 CPU로 내린다 (embedding/norm은 유지).
            2. layers[0].forward를 임시 대체해 배치마다 embedding 출력(hidden_states + kwargs)을
               CPU에 캐시한 뒤 _Abort 예외로 나머지 forward를 중단한다.
            3. 각 decoder layer를 compute_device로 올려 캐시된 hidden_states로 forward →
               observer 수집 → scale 확정 → CPU 반환.
            4. 모든 layer를 원래 device로 복귀.

        scale 동등성:
            full-model calibration과 동일한 데이터·알고리즘이므로 input_scale 값이 일치한다.
            calibration forward 중 input_scale=None → activation fake-quant 없음 (양쪽 동일).
        """
        inner = getattr(self.model, "model", None)
        if inner is None or not hasattr(inner, "layers"):
            raise RuntimeError(
                "_calibrate_sequential은 model.model.layers 구조만 지원합니다. "
                "sequential=False를 사용하세요."
            )

        layers = list(inner.layers)

        # GPU가 있으면 CUDA, 없으면 현재 모델 device를 compute_device로 사용
        compute_device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else next(self.model.parameters()).device
        )

        # 배치 수집
        batches: list = []
        for i, batch in enumerate(dataloader):
            if num_samples is not None and i >= num_samples:
                break
            batches.append(batch)
        if not batches:
            raise ValueError(
                "_calibrate_sequential: dataloader가 비어있어 activation 통계를 수집할 수 없습니다."
            )

        # 1. decoder layers를 CPU로 내리고 원래 device를 기록
        layer_devices = [
            next((p.device for p in layer.parameters()), torch.device("cpu"))
            for layer in layers
        ]
        for layer in layers:
            layer.cpu()
        if compute_device.type == "cuda":
            torch.cuda.empty_cache()

        # 2. layers[0] 입력 캡처
        #    layers[0].forward를 임시 대체 → (args, kwargs)를 CPU 텐서로 저장 후 _Abort 발생.
        #    embedding + pre-processing(causal_mask, position_ids 등)이 compute되고 나서
        #    layers[0]에 진입하는 순간 캡처하므로 모델 구조와 무관하게 동작한다.
        class _Abort(Exception):
            pass

        captured: list = []
        orig_fwd = layers[0].forward

        def _capture_fwd(*args, **kwargs):
            captured.append((
                tuple(a.detach().cpu() if isinstance(a, torch.Tensor) else a for a in args),
                {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                 for k, v in kwargs.items()},
            ))
            raise _Abort()

        layers[0].forward = _capture_fwd
        self.model.eval()
        with torch.no_grad():
            for batch in batches:
                try:
                    if isinstance(batch, dict):
                        self.model(**batch)
                    elif isinstance(batch, (list, tuple)):
                        self.model(*batch)
                    else:
                        self.model(batch)
                except _Abort:
                    pass
        layers[0].forward = orig_fwd

        if len(captured) != len(batches):
            raise RuntimeError(
                f"layers[0] 입력 캡처 실패: {len(captured)}/{len(batches)} 배치. "
                "model.model.layers[0].forward가 호출되지 않았는지 확인하세요."
            )

        # 3. layer별 순차 처리
        for layer in layers:
            layer.to(compute_device)
            next_captured: list = []

            with torch.no_grad():
                for args_cpu, kwargs_cpu in captured:
                    args_dev = tuple(
                        a.to(compute_device) if isinstance(a, torch.Tensor) else a
                        for a in args_cpu
                    )
                    kwargs_dev = {
                        k: v.to(compute_device) if isinstance(v, torch.Tensor) else v
                        for k, v in kwargs_cpu.items()
                    }
                    out = layer(*args_dev, **kwargs_dev)
                    out_h = out[0] if isinstance(out, tuple) else out
                    # position_ids, attention_mask 등 kwargs는 레이어 간 불변 → 재사용
                    next_captured.append(((out_h.detach().cpu(),), kwargs_cpu))

            # 이 layer의 scale 확정 (layer가 compute_device에 있는 동안)
            for mod in layer.modules():
                if isinstance(mod, FakeQuantLinear) and mod.input_observer is not None:
                    mod.input_observer.sync()
                    scale, zp = mod.input_observer.compute_scale_zp()
                    mod.input_scale = scale.to(mod.weight.device)
                    mod.input_zero_point = zp.to(mod.weight.device)
                    mod.input_observer = None  # finalize()와 동일 — observer 즉시 해제

            layer.cpu()
            if compute_device.type == "cuda":
                torch.cuda.empty_cache()
            captured = next_captured

        # 4. layers를 원래 device로 복귀
        for layer, dev in zip(layers, layer_devices):
            layer.to(dev)
