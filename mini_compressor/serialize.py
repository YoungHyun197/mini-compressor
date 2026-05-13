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
    "per_token": "token",
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
        # save_pretrained 이전에 읽어야 함 — 호출 후 config.json의 _name_or_path가 save_dir로 덮어써짐
        model_id = getattr(getattr(model, "config", None), "_name_or_path", None)

        model.save_pretrained(save_dir)

        config_dict = _scheme_to_dict(scheme, ignore=ignore)
        if model_id:
            config_dict["base_model_name_or_path"] = model_id
        config_path = os.path.join(save_dir, QUANT_CONFIG_FILENAME)
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)


def load_pretrained(save_dir: str) -> nn.Module:
    """저장된 디렉토리에서 quantized model 복원.

    흐름:
    1. quantization_config.json → scheme + ignore 복원
    2. from_pretrained(model_id)으로 base model 로드
       ※ from_pretrained(save_dir) 불가: safetensors에 weight_scale 등 추가 키 존재
       ※ from_config만 쓰면 inv_freq 등 persistent=False 버퍼가 float32로 초기화되어
         원본(float16) 대비 1.78e-04 오차 → attention 전 레이어에 걸쳐 logit 4.19 차이
    3. FakeQuantLinear로 교체 (구조만, scale 계산 안 함)
    4. safetensors 로드
    5. weight + scale buffer 직접 주입
       ※ load_state_dict 미사용: None buffer는 local_state에서 제외되어 copy_() 미호출
    """
    from safetensors.torch import load_file

    # 1. scheme + ignore 복원
    config_path = os.path.join(save_dir, QUANT_CONFIG_FILENAME)
    with open(config_path) as f:
        config_dict = json.load(f)
    scheme, ignore = _scheme_from_dict(config_dict)

    # 2. base model 로드 (inv_freq dtype 보존 목적 — from_config + .to() 대신 from_pretrained 사용)
    # quantization_config.json에 원본 model_id가 있으면 우선 사용.
    # 없으면 config.json의 _name_or_path를 읽는데, save_pretrained 이후엔 save_dir로 덮어써져
    # from_pretrained(save_dir)가 호출되어 weight_scale 등 추가 키가 UNEXPECTED 경고를 냄.
    model_id = config_dict.get("base_model_name_or_path")
    if not model_id:
        hf_config = AutoConfig.from_pretrained(save_dir)
        model_id = hf_config._name_or_path
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

    # 3. FakeQuantLinear 구조 생성 (scale 계산 생략)
    modifier = QuantizationModifier(model, scheme, ignore=ignore)
    modifier.initialize(compute_scales=False)

    # load 흐름에서는 observer 불필요 — finalize()와 동일 상태로 맞춤
    from .fake_quant_linear import FakeQuantLinear as _FQL
    for mod in model.modules():
        if isinstance(mod, _FQL):
            mod.input_observer = None

    # 4. safetensors 로드
    saved_state = load_file(os.path.join(save_dir, "model.safetensors"), device="cpu")

    # 5. weight + buffer 전체 직접 주입
    # load_state_dict 미사용: None buffer는 local_state에서 제외되어 copy_()가 호출되지 않음.
    # saved_state를 직접 순회하여 parameter는 data.copy_(), buffer는 _buffers 직접 할당.
    for key, tensor in saved_state.items():
        parts = key.split(".")
        mod = model
        for part in parts[:-1]:
            mod = getattr(mod, part)
        attr = parts[-1]
        if attr in mod._parameters and mod._parameters[attr] is not None:
            mod._parameters[attr].data.copy_(tensor)
        elif attr in mod._buffers:
            mod._buffers[attr] = tensor

    return model
