# Compressor — modifier list를 받아 lifecycle을 순차 실행하는 one-click 진입점
from __future__ import annotations

from typing import Iterable, List, Optional

import torch.nn as nn

from .fake_quant_linear import FakeQuantLinear
from .modifiers import BaseModifier, QuantizationMixin, QuantizationModifier
from .recipes import RECIPE_REGISTRY
from .schemes import QuantizationScheme
from .serialize import save_pretrained


class Compressor:
    """modifier list를 받아 각 modifier에 initialize → calibrate → finalize를 순차 호출한다.

    Usage:
        # recipe preset 한 줄 — 단일 RTN도, 알고리즘 chain도 recipe 이름으로
        compressor = Compressor.from_recipe("w8a8", targets=["Linear"], ignore=["lm_head"])
        compressor = Compressor.from_recipe("w8a8_smoothquant", ignore=["lm_head"])

        # modifier list 직접 구성 (alpha 등 세부 제어)
        compressor = Compressor([
            SmoothQuantModifier(alpha=0.5),
            QuantizationModifier(scheme=W8A8, targets=["Linear"], ignore=["lm_head"]),
        ])

        compressor.compress(model, dataloader)
        compressor.save(model, "./out", tokenizer=tokenizer)
    """

    def __init__(self, modifiers: List[BaseModifier], recipe_name: Optional[str] = None):
        if not modifiers:
            raise ValueError("modifiers는 최소 1개 이상이어야 합니다.")
        self.modifiers = modifiers
        self.recipe_name = recipe_name

    @classmethod
    def from_recipe(
        cls,
        name: str,
        targets: Optional[List[str]] = None,
        ignore: Optional[List[str]] = None,
    ) -> "Compressor":
        """RECIPE_REGISTRY 이름으로 modifier 파이프라인을 펼쳐 Compressor를 생성한다.

        Compressor의 유일한 preset 진입점이다. 단일 RTN scheme은 modifier 1개짜리
        recipe로, SmoothQuant 같은 알고리즘은 여러 modifier가 chain된 recipe로 —
        composition 패턴 위에 얹은 선언적 진입점이다. targets / ignore는 recipe
        내부의 QuantizationModifier로 전달된다.
        """
        if name not in RECIPE_REGISTRY:
            raise ValueError(f"Unknown recipe '{name}'. Available: {list(RECIPE_REGISTRY)}")
        return cls(RECIPE_REGISTRY[name](targets, ignore), recipe_name=name)

    def compress(
        self,
        model: nn.Module,
        dataloader: Optional[Iterable] = None,
        num_samples: Optional[int] = None,
    ) -> nn.Module:
        """각 modifier에 대해 initialize → calibrate → finalize를 순차 실행한다.

        modifier 사이에 calibration 데이터가 공유되므로 dataloader는 한 번만 전달한다.
        calibrate() 도중 예외가 발생해도 finalize()를 반드시 호출해 hook 등 임시 상태를 정리한다.
        """
        data = list(dataloader) if dataloader is not None else []
        for modifier in self.modifiers:
            modifier.initialize(model)
            try:
                modifier.calibrate(data, num_samples=num_samples)
            finally:
                modifier.finalize()
        return model

    def save(
        self,
        model: nn.Module,
        save_dir: str,
        tokenizer=None,
    ) -> None:
        """modifier list에서 QuantizationModifier를 찾아 그 scheme/ignore로 저장한다."""
        quant_mod = self._find_quantization_modifier()
        _validate_compressed(model, quant_mod.scheme)
        save_pretrained(
            model,
            save_dir,
            scheme=quant_mod.scheme,
            ignore=quant_mod.ignore,
            tokenizer=tokenizer,
            recipe_name=self.recipe_name,
        )

    def save_to_hub(
        self,
        model: nn.Module,
        repo_id: str,
        tokenizer=None,
        private: bool = True,
        commit_message: str = "Upload quantized model",
    ) -> str:
        """compress 완료 후 HuggingFace Hub에 업로드한다.

        Args:
            model: compress() 완료된 모델.
            repo_id: HF Hub 저장소 ID. 예: "username/qwen3-w4a16".
            tokenizer: 함께 업로드할 tokenizer. None이면 생략.
            private: True이면 private 저장소로 생성. 기본값 True.
            commit_message: Hub commit 메시지.

        Returns:
            업로드된 HuggingFace Hub URL.

        Note:
            huggingface_hub 패키지와 HF_TOKEN 환경변수 또는 `huggingface-cli login` 필요.

        Usage:
            url = compressor.save_to_hub(model, "username/qwen3-w4a16", tokenizer=tokenizer)
        """
        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            raise ImportError(
                "save_to_hub()에는 huggingface_hub 패키지가 필요합니다. "
                "pip install huggingface_hub 로 설치하세요."
            ) from e

        import tempfile

        api = HfApi()
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            self.save(model, tmpdir, tokenizer=tokenizer)
            _write_model_card(tmpdir, model, repo_id, self._find_quantization_modifier(), self.recipe_name)
            commit_info = api.upload_folder(
                repo_id=repo_id,
                folder_path=tmpdir,
                commit_message=commit_message,
            )

        return commit_info.commit_url

    def _find_quantization_modifier(self) -> QuantizationMixin:
        """modifier list에서 첫 번째 QuantizationMixin 인스턴스를 반환한다.

        복수의 QuantizationMixin이 있으면 list 순서상 첫 번째의 scheme/ignore로 저장된다.
        현재 recipe는 QuantizationMixin을 최대 하나만 포함하므로 이 동작이 항상 의도와 일치한다.
        """
        for m in self.modifiers:
            if isinstance(m, QuantizationMixin):
                return m
        raise ValueError(
            "save()는 modifier list에 QuantizationModifier 또는 GPTQModifier가 포함되어 있을 때만 호출 가능합니다. "
            f"현재 modifier 종류: {[type(m).__name__ for m in self.modifiers]}"
        )


def _validate_compressed(model: nn.Module, scheme: QuantizationScheme) -> None:
    """compress() 없이 save()를 호출한 경우를 감지해 ValueError를 발생시킨다."""
    fql_modules = [m for m in model.modules() if isinstance(m, FakeQuantLinear)]
    if not fql_modules:
        raise ValueError(
            "save()를 호출하기 전에 compress()를 먼저 호출해야 합니다. "
            "모델에 FakeQuantLinear가 없습니다."
        )
    if scheme.activation is not None and not scheme.activation.dynamic:
        missing = [m for m in fql_modules if m.input_scale is None]
        if missing:
            raise ValueError(
                f"Static activation scheme이지만 {len(missing)}개 레이어의 input_scale이 None입니다. "
                "calibration dataloader를 전달하지 않았거나 calibration이 완료되지 않았습니다."
            )


def _write_model_card(
    save_dir: str,
    model: nn.Module,
    repo_id: str,
    quant_mod: QuantizationMixin,
    recipe_name: Optional[str] = None,
) -> None:
    """HuggingFace Hub용 README.md(모델 카드)를 save_dir에 생성한다."""
    import os

    scheme = quant_mod.scheme
    base_model = getattr(getattr(model, "config", None), "_name_or_path", "unknown")
    w_bits = scheme.weight.num_bits
    a_bits = scheme.activation.num_bits if scheme.activation else "fp16"
    w_gran = scheme.weight.granularity.replace("per_", "per-")
    recipe_name = recipe_name or scheme.name

    ignore_note = ""
    if quant_mod.ignore:
        ignore_note = f"\n- **ignore**: `{quant_mod.ignore}`"

    card = f"""---
base_model: {base_model}
library_name: mini-compressor
tags:
  - quantization
  - compressed-tensors
  - mini-compressor
---

# {repo_id.split("/")[-1]}

Post-training quantization of [{base_model}](https://huggingface.co/{base_model}) using [mini-compressor](https://github.com/your-username/mini-compressor).

## Quantization Details

| Property | Value |
|----------|-------|
| Recipe | `{recipe_name}` |
| Weight bits | W{w_bits} ({w_gran}) |
| Activation bits | A{a_bits} |
| Format | compressed-tensors (fake-quantized) |
| Status | calibrated |{ignore_note}

## Usage

```python
from mini_compressor import load_pretrained

model = load_pretrained("{repo_id}")
```

Or with the full pipeline:

```python
from mini_compressor import Compressor
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("{base_model}")
model = Compressor.from_recipe("{recipe_name}").compress(
    model, dataloader=calibration_data
)
```

## Notes

This model uses **fake quantization** (float16 weights + quantization error simulation).
Actual INT{w_bits} packing and kernel dispatch require a compatible runtime (e.g., vLLM with compressed-tensors support).
"""

    with open(os.path.join(save_dir, "README.md"), "w") as f:
        f.write(card)
