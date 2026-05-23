# calibration 데이터 로딩 유틸 — wikitext-2 / C4 에서 무작위 시퀀스 샘플링
from __future__ import annotations

import torch


def get_calibration_data(
    tokenizer,
    dataset: str = "wikitext2",
    n_samples: int = 128,
    seq_len: int = 512,
    seed: int = 42,
    device: str = "cpu",
) -> list[dict]:
    """calibration용 token sequence 배치를 반환한다.

    Args:
        dataset: "wikitext2" 또는 "c4".
        n_samples: 반환할 배치 수.
        seq_len: 각 배치의 시퀀스 길이 (토큰 수).
        seed: 재현성을 위한 random seed.
        device: 반환 tensor를 올릴 device.

    Returns:
        [{"input_ids": Tensor[1, seq_len]}, ...] 길이 n_samples인 리스트.
    """
    from datasets import load_dataset

    if dataset == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(x for x in ds["text"] if x.strip())
    elif dataset == "c4":
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        rows = [row["text"] for _, row in zip(range(n_samples * 10), ds)]
        text = "\n\n".join(rows)
    else:
        raise ValueError(f"지원하지 않는 dataset: {dataset!r}. 'wikitext2' 또는 'c4' 중 선택.")

    tokens = tokenizer(text, return_tensors="pt").input_ids[0]

    gen = torch.Generator()
    gen.manual_seed(seed)

    batches = []
    for _ in range(n_samples):
        start = torch.randint(0, len(tokens) - seq_len, (1,), generator=gen).item()
        chunk = tokens[start : start + seq_len].unsqueeze(0).to(device)
        batches.append({"input_ids": chunk})
    return batches
