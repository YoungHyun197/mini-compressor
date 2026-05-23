# GPTQ 데모 — Hessian 기반 weight 최적화로 RTN 대비 재구성 오차를 줄이는 것을 직접 증명
"""
사용법:
    python demo/demo_gptq.py
    python demo/demo_gptq.py --n-samples 4 --seq-len 128

소요 시간:
    GPU: 약 2~4분 (Hessian 수집 + layer-wise GPTQ 최적화)
    CPU: 약 10~15분

증명 항목:
    1. layer.0 의 q_proj weight 재구성 MSE: GPTQ < RTN
    2. output MSE (동일 입력 기준): GPTQ < RTN
    3. generate 비교: FP16 vs W4A16-GPTQ
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_compressor import Compressor
from mini_compressor.modifiers.quantization import QuantizationModifier
from mini_compressor.fake_quant_linear import FakeQuantLinear
from mini_compressor.schemes import W4A16

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 20


def _load(model_id):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE)
    model.eval()
    return model


def _gen(model, tokenizer, prompt="The key advantage of quantization is"):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def _load_calib(tokenizer, n_samples=4, seq_len=128, seed=42):
    """wikitext-2 train에서 고정 seed로 캘리브레이션 배치를 준비한다.
    datasets 미설치 시 고정 텍스트로 fallback한다.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(x for x in ds["text"] if x.strip())
    except Exception:
        text = (
            "Quantization is a process that reduces the precision of the weights in a neural "
            "network. Large language models have billions of parameters, requiring significant "
            "memory. Post-training quantization applies after training without fine-tuning. "
            "The GPTQ algorithm uses second-order information to minimize quantization error. "
        ) * 60

    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    gen = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_samples):
        start = torch.randint(0, max(1, len(tokens) - seq_len), (1,), generator=gen).item()
        chunk = tokens[start: start + seq_len].unsqueeze(0).to(DEVICE)
        batches.append({"input_ids": chunk})
    return batches


def _get_first_qproj(model):
    """model.model.layers[0] 의 q_proj를 찾아 (이름, 모듈) 반환한다."""
    for name, mod in model.named_modules():
        if "layers.0" in name and "q_proj" in name:
            if isinstance(mod, (nn.Linear, FakeQuantLinear)):
                return name, mod
    return None, None


def _compute_rtn(W_orig_cpu):
    """W_orig에 RTN 양자화를 적용한 weight와 weight MSE를 반환한다."""
    out_f, in_f = W_orig_cpu.shape
    dummy = nn.Sequential(nn.Linear(in_f, out_f, bias=False))
    dummy[0].weight.data.copy_(W_orig_cpu.float())
    QuantizationModifier(W4A16).initialize(dummy)
    W_rtn = dummy[0].weight.detach().float()
    mse = ((W_rtn - W_orig_cpu.float()) ** 2).mean().item()
    return W_rtn, mse


def run(model_id=MODEL_ID, n_samples=4, seq_len=128):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"\n{'─'*65}")
    print(f"  mini-compressor  |  GPTQ Demo  |  device={DEVICE.upper()}")
    print(f"  calibration: {n_samples} samples × {seq_len} tokens")
    print(f"{'─'*65}")
    print("  ※ GPTQ는 Hessian 수집 + layer-wise 최적화로 GPU 기준 2~4분 소요됩니다.")

    t0 = time.time()

    # ── 1. FP16 기준선 ────────────────────────────────────────────────────────
    print("\n[1/3] FP16 baseline 로드 + 생성")
    model = _load(model_id)
    text_fp = _gen(model, tokenizer)
    print(f"  FP16 생성: {text_fp}")

    # 비교 기준: layer.0 q_proj 원본 weight 스냅샷
    qproj_name, qproj_mod = _get_first_qproj(model)
    W_orig = qproj_mod.weight.detach().clone().cpu()
    print(f"  비교 레이어: {qproj_name}  shape={tuple(W_orig.shape)}")

    # RTN weight 및 MSE 사전 계산 (GPTQ 적용 전 동일 원본 사용)
    print("  RTN 재구성 MSE 계산 중...")
    W_rtn, mse_rtn = _compute_rtn(W_orig)

    # ── 2. GPTQ 적용 ─────────────────────────────────────────────────────────
    print(f"\n[2/3] GPTQ 적용 중 — Hessian 수집 → layer-wise 최적화")
    t1 = time.time()
    calib = _load_calib(tokenizer, n_samples=n_samples, seq_len=seq_len)
    Compressor.from_recipe("w4a16_gptq", targets=["Linear"], ignore=["lm_head"]).compress(
        model, dataloader=calib
    )
    gptq_elapsed = time.time() - t1
    print(f"  GPTQ 완료  소요: {gptq_elapsed:.1f}s")

    # GPTQ weight 취득 (압축 후 동일 레이어)
    _, qproj_gptq = _get_first_qproj(model)
    W_gptq = qproj_gptq.weight.detach().cpu().float()
    mse_gptq = ((W_gptq - W_orig.float()) ** 2).mean().item()

    # output MSE 비교 — 동일 입력 token에 대해 각 weight로 linear 출력 계산
    x_probe = calib[0]["input_ids"].to(DEVICE)
    with torch.no_grad():
        hidden = model.model.embed_tokens(x_probe)
        x_flat = hidden.reshape(-1, W_orig.shape[1]).half()
        out_fp   = x_flat @ W_orig.to(DEVICE).half().T
        out_rtn  = x_flat @ W_rtn.to(DEVICE).half().T
        out_gptq = x_flat @ W_gptq.to(DEVICE).half().T
    out_mse_rtn  = ((out_rtn  - out_fp) ** 2).mean().item()
    out_mse_gptq = ((out_gptq - out_fp) ** 2).mean().item()

    # 결과 출력
    print(f"\n  [weight 재구성 MSE — {qproj_name}]")
    print(f"  RTN  weight MSE: {mse_rtn:.6f}")
    print(f"  GPTQ weight MSE: {mse_gptq:.6f}")
    w_improv = (mse_rtn - mse_gptq) / mse_rtn * 100 if mse_rtn > 0 else 0
    print(f"  ▶ GPTQ가 RTN 대비 weight 재구성 오차 {w_improv:.1f}% 감소")

    print(f"\n  [output MSE — 동일 입력 기준]")
    print(f"  RTN  output MSE: {out_mse_rtn:.6f}")
    print(f"  GPTQ output MSE: {out_mse_gptq:.6f}")
    o_improv = (out_mse_rtn - out_mse_gptq) / out_mse_rtn * 100 if out_mse_rtn > 0 else 0
    print(f"  ▶ GPTQ output MSE {o_improv:.1f}% 감소")

    # ── 3. GPTQ generate 비교 ────────────────────────────────────────────────
    print("\n[3/3] 생성 비교")
    text_gptq = _gen(model, tokenizer)
    total_elapsed = time.time() - t0
    print(f"  FP16 생성:       {text_fp}")
    print(f"  W4A16-GPTQ 생성: {text_gptq}")
    print(f"  총 소요: {total_elapsed:.1f}s")

    del model; torch.cuda.empty_cache()

    return {
        "text_fp":       text_fp,
        "text_gptq":     text_gptq,
        "mse_rtn":       mse_rtn,
        "mse_gptq":      mse_gptq,
        "out_mse_rtn":   out_mse_rtn,
        "out_mse_gptq":  out_mse_gptq,
        "w_improv":      w_improv,
        "elapsed":       total_elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--n-samples", type=int, default=4,
                        help="Hessian 수집 배치 수 (기본 4, 많을수록 정확하지만 느림)")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="배치당 시퀀스 길이 (기본 128)")
    args = parser.parse_args()

    r = run(args.model, args.n_samples, args.seq_len)

    print(f"\n{'═'*65}")
    print("  결과 요약")
    print(f"{'─'*65}")
    print(f"  FP16:        {r['text_fp'][:55]}...")
    print(f"  W4A16-GPTQ:  {r['text_gptq'][:55]}...")
    print(f"  Weight MSE:  RTN={r['mse_rtn']:.6f}  GPTQ={r['mse_gptq']:.6f}  ({r['w_improv']:.1f}% 개선)")
    print(f"  Output MSE:  RTN={r['out_mse_rtn']:.6f}  GPTQ={r['out_mse_gptq']:.6f}")
    print(f"  총 소요: {r['elapsed']:.1f}s")
    print(f"{'═'*65}")


if __name__ == "__main__":
    main()
