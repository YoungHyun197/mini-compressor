# serialize — save_pretrained / load_pretrained / quantization_config.json (HF 호환)
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerBase

from .modifier import QuantizationModifier
from .schemes import QuantizationScheme, QuantizationSpec, SCHEME_REGISTRY

_GRANULARITY_TO_STRATEGY = {
    "per_tensor": "tensor",
    "per_channel": "channel",
    "per_group": "group",
}

_STRATEGY_TO_GRANULARITY = {v: k for k, v in _GRANULARITY_TO_STRATEGY.items()}

QUANT_CONFIG_FILENAME = "quantization_config.json"


def _spec_to_dict(spec: QuantizationSpec) -> dict:
    d = {
        "num_bits": spec.num_bits,
        "type": spec.dtype,
        "symmetric": spec.symmetric,
        "strategy": _GRANULARITY_TO_STRATEGY[spec.granularity],
        "dynamic": spec.dynamic,
    }
    if spec.group_size is not None:
        d["group_size"] = spec.group_size
    return d


def _spec_from_dict(d: dict) -> QuantizationSpec:
    granularity = _STRATEGY_TO_GRANULARITY[d["strategy"]]
    return QuantizationSpec(
        num_bits=d["num_bits"],
        dtype=d.get("type", "int"),
        symmetric=d["symmetric"],
        granularity=granularity,
        group_size=d.get("group_size"),
        dynamic=d.get("dynamic", False),
    )


def _scheme_to_dict(
    scheme: QuantizationScheme,
    ignore: Optional[List[str]] = None,
) -> dict:
    group: dict = {
        "weights": _spec_to_dict(scheme.weight),
        "input_activations": _spec_to_dict(scheme.activation) if scheme.activation else None,
        "targets": ["Linear"],
    }
    if ignore:
        group["ignore"] = ignore
    return {
        "quant_type": "compressed-tensors",
        "quantization_status": "calibrated",
        "config_groups": {"group_0": group},
    }


def _scheme_from_dict(d: dict) -> tuple[QuantizationScheme, Optional[List[str]]]:
    group = d["config_groups"]["group_0"]
    weight_spec = _spec_from_dict(group["weights"])
    act_raw = group.get("input_activations")
    act_spec = _spec_from_dict(act_raw) if act_raw else None
    ignore = group.get("ignore")

    # SCHEME_REGISTRY에서 name 매칭 시도, 없으면 "custom"으로 이름 부여
    name = "custom"
    for k, v in SCHEME_REGISTRY.items():
        if (
            v.weight.num_bits == weight_spec.num_bits
            and v.weight.granularity == weight_spec.granularity
            and (v.activation is None) == (act_spec is None)
        ):
            name = k
            break

    scheme = QuantizationScheme(name=name, weight=weight_spec, activation=act_spec)
    return scheme, ignore


def save_pretrained(
    model: nn.Module,
    save_dir: str,
    scheme: QuantizationScheme,
    ignore: Optional[List[str]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> None:
    """모델 + quantization_config.json 저장. tokenizer 있으면 함께 저장."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        model.save_pretrained(save_dir)

        config_dict = _scheme_to_dict(scheme, ignore=ignore)
        config_path = os.path.join(save_dir, QUANT_CONFIG_FILENAME)
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)


def load_pretrained(save_dir: str) -> nn.Module:
    """저장된 디렉토리에서 quantized model 복원.

    흐름:
    1. config.json에서 원본 model_id 읽기 (HF가 _name_or_path에 자동 저장)
    2. base model 생성 (float16)
    3. quantization_config.json에서 scheme + ignore 복원
    4. modifier.initialize(compute_scales=False) — 구조만 생성, scale 계산 안 함
    5. saved state_dict 로드 — weight + scale buffer 주입
    """
    # 1. 원본 model_id 복원
    hf_config = AutoConfig.from_pretrained(save_dir)
    model_id = hf_config._name_or_path

    # 2. base model 생성 (FakeQuantLinear 없는 원본 구조)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    )

    # 3. scheme + ignore 복원
    config_path = os.path.join(save_dir, QUANT_CONFIG_FILENAME)
    with open(config_path) as f:
        config_dict = json.load(f)
    scheme, ignore = _scheme_from_dict(config_dict)

    # 4. 구조만 생성 (scale은 state_dict에서 채울 것이므로 RTN 계산 생략)
    modifier = QuantizationModifier(model, scheme, ignore=ignore)
    modifier.initialize(compute_scales=False)

    # 5. saved weight + scale buffer 주입
    # HF save_pretrained는 safetensors 형식으로 저장함
    from safetensors.torch import load_file

    safetensors_path = os.path.join(save_dir, "model.safetensors")
    saved_state = load_file(safetensors_path, device="cpu")
    model.load_state_dict(saved_state)

    return model
