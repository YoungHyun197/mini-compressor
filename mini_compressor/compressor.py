# Compressor — modifier list를 받아 lifecycle을 순차 실행하는 one-click 진입점
from __future__ import annotations

from typing import Iterable, List, Optional

import torch.nn as nn

from .modifiers import BaseModifier, QuantizationMixin, QuantizationModifier
from .recipes import RECIPE_REGISTRY
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

    def __init__(self, modifiers: List[BaseModifier]):
        if not modifiers:
            raise ValueError("modifiers는 최소 1개 이상이어야 합니다.")
        self.modifiers = modifiers

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
        return cls(RECIPE_REGISTRY[name](targets, ignore))

    def compress(
        self,
        model: nn.Module,
        dataloader: Optional[Iterable] = None,
        num_samples: Optional[int] = None,
    ) -> nn.Module:
        """각 modifier에 대해 initialize → calibrate → finalize를 순차 실행한다.

        modifier 사이에 calibration 데이터가 공유되므로 dataloader는 한 번만 전달한다.
        """
        data = list(dataloader) if dataloader is not None else []
        for modifier in self.modifiers:
            modifier.initialize(model)
            modifier.calibrate(data, num_samples=num_samples)
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
        save_pretrained(
            model,
            save_dir,
            scheme=quant_mod.scheme,
            ignore=quant_mod.ignore,
            tokenizer=tokenizer,
        )

    def save_to_hub(
        self,
        model: nn.Module,
        repo_id: str,
        tokenizer=None,
        private: bool = True,
        commit_message: str = "Upload quantized model",
    ) -> None:
        """compress 완료 후 HuggingFace Hub에 업로드한다.

        Args:
            model: compress() 완료된 모델.
            repo_id: HF Hub 저장소 ID. 예: "username/qwen3-w4a16".
            tokenizer: 함께 업로드할 tokenizer. None이면 생략.
            private: True이면 private 저장소로 생성. 기본값 True.
            commit_message: Hub commit 메시지.

        Intended behavior:
            1. 임시 디렉토리에 save() 호출 (safetensors + quantization_config.json).
            2. huggingface_hub.HfApi().upload_folder()로 임시 디렉토리 전체를 업로드한다.
            3. 임시 디렉토리를 정리한다.
            모델 파일 구조가 이미 HF 호환이므로 upload_folder 한 번으로 완료된다.

        Note:
            huggingface_hub 패키지와 HF_TOKEN 환경변수 또는 `huggingface-cli login` 필요.

        Usage:
            compressor.save_to_hub(model, "username/qwen3-w4a16", tokenizer=tokenizer)
        """
        raise NotImplementedError(
            "save_to_hub is not yet implemented. "
            "Use save(model, save_dir) to save locally, then upload manually with "
            "huggingface_hub.upload_folder(repo_id=..., folder_path=save_dir)."
        )

    def _find_quantization_modifier(self) -> QuantizationMixin:
        for m in self.modifiers:
            if isinstance(m, QuantizationMixin):
                return m
        raise ValueError(
            "save()는 modifier list에 QuantizationModifier 또는 GPTQModifier가 포함되어 있을 때만 호출 가능합니다. "
            f"현재 modifier 종류: {[type(m).__name__ for m in self.modifiers]}"
        )
