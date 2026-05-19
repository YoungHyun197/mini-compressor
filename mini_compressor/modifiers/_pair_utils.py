# norm → linear 페어 탐색 유틸 — SmoothQuantModifier, AWQModifier 공통
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


def _find_smooth_pairs(model: nn.Module) -> List[Tuple[nn.Module, List[nn.Linear]]]:
    """Qwen3/LLaMA 공통 패턴으로 (norm, [linear,...]) 페어를 자동 탐색한다.

    탐색 규칙 (각 decoder layer 후보 모듈마다):
      - input_layernorm + self_attn.{q_proj, k_proj, v_proj}
      - post_attention_layernorm + mlp.{gate_proj, up_proj}
    각 그룹의 linear들은 모두 nn.Linear이고 weight가 존재할 때만 페어에 포함된다.
    """
    pairs: List[Tuple[nn.Module, List[nn.Linear]]] = []
    for module in model.modules():
        if hasattr(module, "input_layernorm") and hasattr(module, "self_attn"):
            norm = module.input_layernorm
            attn = module.self_attn
            linears = _collect_linears(attn, ["q_proj", "k_proj", "v_proj"])
            if linears and _has_affine_weight(norm):
                pairs.append((norm, linears))
        if hasattr(module, "post_attention_layernorm") and hasattr(module, "mlp"):
            norm = module.post_attention_layernorm
            mlp = module.mlp
            linears = _collect_linears(mlp, ["gate_proj", "up_proj"])
            if linears and _has_affine_weight(norm):
                pairs.append((norm, linears))
    return pairs


def _collect_linears(parent: nn.Module, names: List[str]) -> List[nn.Linear]:
    out = []
    for n in names:
        mod = getattr(parent, n, None)
        if isinstance(mod, nn.Linear):
            out.append(mod)
    return out


def _has_affine_weight(module: nn.Module) -> bool:
    return hasattr(module, "weight") and isinstance(module.weight, torch.Tensor)
