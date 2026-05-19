# save_to_hub — HfApi를 mock하여 upload 흐름과 모델 카드 생성을 검증한다
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from mini_compressor import Compressor


class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(256, 256, bias=False)
        self.lm_head = nn.Linear(256, 256, bias=False)

    def forward(self, x):
        return self.lm_head(self.proj(x))

    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        save_file(self.state_dict(), os.path.join(save_dir, "model.safetensors"))


@pytest.fixture()
def compressed_model_and_compressor():
    model = _TinyLM()
    compressor = Compressor.from_recipe("w4a16", targets=["Linear"], ignore=["lm_head"])
    compressor.compress(model)
    return model, compressor


def _make_mock_api():
    api = MagicMock()
    commit = MagicMock()
    commit.commit_url = "https://huggingface.co/user/repo/commit/abc123"
    api.upload_folder.return_value = commit
    return api


def test_save_to_hub_calls_create_repo_and_upload(compressed_model_and_compressor):
    model, compressor = compressed_model_and_compressor
    mock_api = _make_mock_api()

    with patch("huggingface_hub.HfApi", return_value=mock_api):
        url = compressor.save_to_hub(model, "user/test-repo")

    mock_api.create_repo.assert_called_once_with(
        repo_id="user/test-repo", private=True, exist_ok=True
    )
    mock_api.upload_folder.assert_called_once()
    call_kwargs = mock_api.upload_folder.call_args
    assert call_kwargs.kwargs["repo_id"] == "user/test-repo"
    assert url == "https://huggingface.co/user/repo/commit/abc123"


def test_save_to_hub_private_false(compressed_model_and_compressor):
    model, compressor = compressed_model_and_compressor
    mock_api = _make_mock_api()

    with patch("huggingface_hub.HfApi", return_value=mock_api):
        compressor.save_to_hub(model, "user/pub-repo", private=False)

    mock_api.create_repo.assert_called_once_with(
        repo_id="user/pub-repo", private=False, exist_ok=True
    )


def test_save_to_hub_uploads_required_files(compressed_model_and_compressor):
    """upload_folder에 전달된 folder_path에 필수 파일이 존재해야 한다."""
    model, compressor = compressed_model_and_compressor
    captured = {}

    def _capture_upload(**kwargs):
        captured["folder_path"] = kwargs["folder_path"]
        # upload_folder 호출 시점에 tmpdir은 아직 살아있으므로 파일 목록 확인 가능
        captured["files"] = os.listdir(kwargs["folder_path"])
        result = MagicMock()
        result.commit_url = "https://huggingface.co/x"
        return result

    mock_api = _make_mock_api()
    mock_api.upload_folder.side_effect = _capture_upload

    with patch("huggingface_hub.HfApi", return_value=mock_api):
        compressor.save_to_hub(model, "user/test-repo")

    assert "model.safetensors" in captured["files"]
    assert "quantization_config.json" in captured["files"]
    assert "README.md" in captured["files"]


def test_save_to_hub_model_card_content(compressed_model_and_compressor):
    """생성된 모델 카드에 recipe 이름과 필수 섹션이 포함되어야 한다."""
    model, compressor = compressed_model_and_compressor
    captured_card = {}

    def _capture_upload(**kwargs):
        readme_path = os.path.join(kwargs["folder_path"], "README.md")
        with open(readme_path) as f:
            captured_card["content"] = f.read()
        result = MagicMock()
        result.commit_url = "https://huggingface.co/x"
        return result

    mock_api = _make_mock_api()
    mock_api.upload_folder.side_effect = _capture_upload

    with patch("huggingface_hub.HfApi", return_value=mock_api):
        compressor.save_to_hub(model, "user/qwen3-w4a16")

    card = captured_card["content"]
    assert "w4a16" in card          # recipe name
    assert "W4" in card             # weight bits
    assert "compressed-tensors" in card
    assert "## Usage" in card


def test_save_to_hub_missing_huggingface_hub(compressed_model_and_compressor):
    """huggingface_hub 미설치 시 ImportError가 발생해야 한다."""
    model, compressor = compressed_model_and_compressor

    with patch.dict("sys.modules", {"huggingface_hub": None}):
        with pytest.raises(ImportError, match="huggingface_hub"):
            compressor.save_to_hub(model, "user/test-repo")
