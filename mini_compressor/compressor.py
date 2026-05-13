# Compressor — one-click 진입점 API (from_scheme → compress)
from __future__ import annotations

from typing import Iterable, List, Optional

import torch.nn as nn

from .modifier import QuantizationModifier
from .schemes import QuantizationScheme, SCHEME_REGISTRY
from .serialize import save_pretrained


class Compressor:
    def __init__(
        self,
        scheme: QuantizationScheme,
        targets: Optional[List[str]] = None,
        ignore: Optional[List[str]] = None,
    ):
        self.scheme = scheme
        self.targets = targets
        self.ignore = ignore

    @classmethod
    def from_scheme(
        cls,
        name: str,
        targets: Optional[List[str]] = None,
        ignore: Optional[List[str]] = None,
    ) -> "Compressor":
        """SCHEME_REGISTRY에서 scheme을 조회해 Compressor를 생성한다."""
        if name not in SCHEME_REGISTRY:
            raise ValueError(f"Unknown scheme '{name}'. Available: {list(SCHEME_REGISTRY)}")
        return cls(scheme=SCHEME_REGISTRY[name], targets=targets, ignore=ignore)

    def compress(
        self,
        model: nn.Module,
        dataloader: Optional[Iterable] = None,
        num_samples: Optional[int] = None,
    ) -> nn.Module:
        """initialize → calibrate → finalize 3단계를 순서대로 실행한다."""
        modifier = QuantizationModifier(model, self.scheme, targets=self.targets, ignore=self.ignore)
        modifier.initialize()
        modifier.calibrate(dataloader or [], num_samples=num_samples)
        modifier.finalize()
        return model

    def save(
        self,
        model: nn.Module,
        save_dir: str,
        tokenizer=None,
    ) -> None:
        """compress 완료 후 로컬 디렉토리에 저장."""
        save_pretrained(
            model,
            save_dir,
            scheme=self.scheme,
            ignore=self.ignore,
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
