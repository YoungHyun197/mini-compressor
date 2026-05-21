# mini-compressor 데모 — W4A16 / W4A16-GPTQ / W8A8 / W8A8-dynamic / W8A8+SmoothQuant 압축·생성·저장 round-trip
"""
사용법:
    python demo.py                        # 기본: 다섯 scheme 생성 비교
    python demo.py --save /tmp/demo_save  # W4A16 저장 + 로드 round-trip 추가
    python demo.py --ppl                  # wikitext-2 perplexity 측정 추가 (시간 소요)
"""
import argparse
import math
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_compressor import (
    Compressor,
    load_pretrained,
)

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT   = "The key advantage of quantization is"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def load_model(model_id):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE)
    model.eval()
    return model


def generate(model, tokenizer, prompt=PROMPT, max_new_tokens=40):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def _load_gptq_calib(
    tokenizer, n_seqs: int = 8, seq_len: int = 512, seed: int = 42
) -> list:
    """GPTQ Hessian 수집용 wikitext-2 train 시퀀스를 준비한다.

    n_seqs * seq_len 개의 토큰을 제공해 largest layer in_features >= hidden_size 조건을 충족한다.
    """
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(x for x in ds["text"] if x.strip())
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]

    gen = torch.Generator()
    gen.manual_seed(seed)
    batches = []
    for _ in range(n_seqs):
        start = torch.randint(0, len(tokens) - seq_len, (1,), generator=gen).item()
        chunk = tokens[start : start + seq_len].unsqueeze(0).to(DEVICE)
        batches.append({"input_ids": chunk})
    return batches


def compute_perplexity(model, encodings, stride=512, max_len=2048):
    """sliding window perplexity (HuggingFace 공식 방식)."""
    input_ids = encodings.input_ids.to(DEVICE)
    seq_len   = input_ids.shape[1]
    nlls, prev_end = [], 0
    for begin in range(0, seq_len, stride):
        end        = min(begin + max_len, seq_len)
        target_len = end - prev_end
        chunk      = input_ids[:, begin:end]
        labels     = chunk.clone()
        labels[:, :-target_len] = -100
        with torch.no_grad():
            loss = model(chunk, labels=labels).loss
        nlls.append(loss.item() * target_len)
        prev_end = end
        if end == seq_len:
            break
    return math.exp(sum(nlls) / prev_end)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID, help=f"HuggingFace 모델 ID (기본: {MODEL_ID})")
    parser.add_argument("--save", metavar="DIR",  help="W4A16 저장 후 load_pretrained round-trip 확인")
    parser.add_argument("--ppl",  action="store_true", help="wikitext-2 perplexity 측정")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ppl_results = {}

    # --ppl 시 encodings를 미리 준비 (모델마다 재사용)
    encodings = None
    if args.ppl:
        from datasets import load_dataset
        print("[PPL] wikitext-2-raw-v1 토크나이즈 중...")
        dataset   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text      = "\n\n".join(dataset["text"])
        encodings = tokenizer(text, return_tensors="pt")

    # ── 1. FP16 baseline ──────────────────────────────────────────────────────
    print("\n[1/6] FP16 baseline")
    model = load_model(args.model)
    text_fp = generate(model, tokenizer)
    print(f"  {text_fp}")
    if args.ppl:
        print("  PPL 측정 중...")
        ppl_results["FP16 (baseline)"] = compute_perplexity(model, encodings)
        print(f"  PPL = {ppl_results['FP16 (baseline)']:.2f}")
    del model
    torch.cuda.empty_cache()

    # ── 2. W4A16 ─────────────────────────────────────────────────────────────
    print("\n[2/6] W4A16 — weight-only INT4 (RTN, calibration 불필요)")
    model_w4 = load_model(args.model)
    Compressor.from_recipe("w4a16", targets=["Linear"], ignore=["lm_head"]).compress(model_w4)
    text_w4 = generate(model_w4, tokenizer)
    print(f"  {text_w4}")
    if args.ppl:
        print("  PPL 측정 중...")
        ppl_results["W4A16 RTN"] = compute_perplexity(model_w4, encodings)
        print(f"  PPL = {ppl_results['W4A16 RTN']:.2f}")

    if args.save:
        print(f"\n  → {args.save} 에 저장 중...")
        compressor_w4 = Compressor.from_recipe("w4a16", targets=["Linear"], ignore=["lm_head"])
        compressor_w4.save(model_w4, args.save, tokenizer=tokenizer)
        print(f"  저장 파일: {os.listdir(args.save)}")

        print("  → load_pretrained 로드 중...")
        model_loaded = load_pretrained(args.save).to(DEVICE)
        model_loaded.eval()
        text_loaded = generate(model_loaded, tokenizer)
        print(f"  [load] {text_loaded}")
        print(f"  round-trip 일치: {text_w4 == text_loaded}")
        del model_loaded

    del model_w4
    torch.cuda.empty_cache()

    # ── 3. W4A16 GPTQ ─────────────────────────────────────────────────────────
    print("\n[3/6] W4A16 GPTQ — Hessian 기반 오차 전파 (wikitext-2 train calibration)")
    model_gptq = load_model(args.model)
    print("  캘리브레이션 데이터 준비 중...")
    gptq_calib = _load_gptq_calib(tokenizer)
    Compressor.from_recipe("w4a16_gptq", targets=["Linear"], ignore=["lm_head"]).compress(
        model_gptq, dataloader=gptq_calib
    )
    text_gptq = generate(model_gptq, tokenizer)
    print(f"  {text_gptq}")
    if args.ppl:
        print("  PPL 측정 중...")
        ppl_results["W4A16 GPTQ"] = compute_perplexity(model_gptq, encodings)
        print(f"  PPL = {ppl_results['W4A16 GPTQ']:.2f}")
    del model_gptq
    torch.cuda.empty_cache()

    # ── 4. W8A8 static ────────────────────────────────────────────────────────
    print("\n[4/6] W8A8 static — weight+activation INT8 (MinMax calibration)")
    model_w8 = load_model(args.model)
    calib_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Quantization reduces model size by representing weights in lower precision.",
        "Large language models require significant computational resources.",
        "Neural networks learn representations through gradient descent.",
        "Attention mechanisms allow models to focus on relevant input tokens.",
    ]
    calib_inputs = [
        {k: v.to(DEVICE) for k, v in tokenizer(t, return_tensors="pt").items()}
        for t in calib_texts
    ]
    Compressor.from_recipe("w8a8", targets=["Linear"], ignore=["lm_head"]).compress(
        model_w8, dataloader=calib_inputs
    )
    text_w8 = generate(model_w8, tokenizer)
    print(f"  {text_w8}")
    if args.ppl:
        print("  PPL 측정 중...")
        ppl_results["W8A8 static"] = compute_perplexity(model_w8, encodings)
        print(f"  PPL = {ppl_results['W8A8 static']:.2f}")
    del model_w8
    torch.cuda.empty_cache()

    # ── 5. W8A8 dynamic ───────────────────────────────────────────────────────
    print("\n[5/6] W8A8 dynamic — per-token INT8 (calibration 불필요)")
    model_dyn = load_model(args.model)
    Compressor.from_recipe("w8a8_dynamic", targets=["Linear"], ignore=["lm_head"]).compress(model_dyn)
    text_dyn = generate(model_dyn, tokenizer)
    print(f"  {text_dyn}")
    if args.ppl:
        print("  PPL 측정 중...")
        ppl_results["W8A8 dynamic"] = compute_perplexity(model_dyn, encodings)
        print(f"  PPL = {ppl_results['W8A8 dynamic']:.2f}")
    del model_dyn
    torch.cuda.empty_cache()

    # ── 6. W8A8 + SmoothQuant ─────────────────────────────────────────────────
    print("\n[6/6] W8A8 + SmoothQuant — activation 분포 평탄화 후 W8A8 dynamic")
    model_sq = load_model(args.model)
    Compressor.from_recipe(
        "w8a8_smoothquant", targets=["Linear"], ignore=["lm_head"]
    ).compress(model_sq, dataloader=calib_inputs)
    text_sq = generate(model_sq, tokenizer)
    print(f"  {text_sq}")
    if args.ppl:
        print("  PPL 측정 중...")
        ppl_results["W8A8 + SmoothQuant"] = compute_perplexity(model_sq, encodings)
        print(f"  PPL = {ppl_results['W8A8 + SmoothQuant']:.2f}")
    del model_sq
    torch.cuda.empty_cache()

    # ── perplexity 요약 표 ─────────────────────────────────────────────────────
    if args.ppl:
        ppl_fp = ppl_results["FP16 (baseline)"]
        print(f"\n{'Scheme':<22} {'PPL':>8}  {'Δ vs FP16':>10}")
        print("-" * 45)
        for name, ppl in ppl_results.items():
            delta = "—" if name == "FP16 (baseline)" else f"+{ppl - ppl_fp:.2f}"
            print(f"{name:<22} {ppl:>8.2f}  {delta:>10}")

    # ── generate 결과 요약 ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  [FP16]              {text_fp[:80]}")
    print(f"  [W4A16 RTN]         {text_w4[:80]}")
    print(f"  [W4A16 GPTQ]        {text_gptq[:80]}")
    print(f"  [W8A8 static]       {text_w8[:80]}")
    print(f"  [W8A8 dynamic]      {text_dyn[:80]}")
    print(f"  [W8A8 + SmoothQuant]{text_sq[:80]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
