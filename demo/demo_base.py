# W4A16 RTN · W8A8 dynamic 기본 압축 데모 — calibration 불필요, ~30초 완료
"""
사용법:
    python demo/demo_base.py
    python demo/demo_base.py --model Qwen/Qwen3-0.6B

증명 항목:
    1. nn.Linear → FakeQuantLinear 교체 (layer type 직접 확인)
    2. weight_scale shape — per-group 양자화 검증
    3. input_scale is None — dynamic scheme은 사전 scale 없음
    4. generate가 여전히 동작 — 모델 기능 유지
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_compressor import Compressor
from mini_compressor.fake_quant_linear import FakeQuantLinear

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "The key advantage of quantization is"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 20


def _load(model_id):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE)
    model.eval()
    return model


def _gen(model, tokenizer, prompt=PROMPT):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def _count(model, cls):
    return sum(1 for m in model.modules() if isinstance(m, cls))


def run(model_id=MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    results = []

    print(f"\n{'─'*65}")
    print(f"  mini-compressor  |  Base Demo  |  device={DEVICE.upper()}")
    print(f"{'─'*65}")

    # ── 1. FP16 baseline ────────────────────────────────────────────────────
    print("\n[1/3] FP16 baseline 로드 중...")
    t0 = time.time()
    model = _load(model_id)
    n_linear = _count(model, nn.Linear)
    text_fp = _gen(model, tokenizer)
    print(f"  nn.Linear 레이어 수: {n_linear}")
    print(f"  생성: {text_fp}")
    results.append({"scheme": "FP16 baseline", "text": text_fp,
                    "elapsed": time.time() - t0, "replaced": 0, "n_linear": n_linear})
    del model; torch.cuda.empty_cache()

    # ── 2. W4A16 RTN ────────────────────────────────────────────────────────
    print("\n[2/3] W4A16 RTN — weight-only INT4, calibration 불필요")
    t0 = time.time()
    model = _load(model_id)

    print("  압축 전: proj layer 타입 =", type(model.model.layers[0].self_attn.q_proj).__name__)
    Compressor.from_recipe("w4a16", targets=["Linear"], ignore=["lm_head"]).compress(model)
    print("  압축 후: proj layer 타입 =", type(model.model.layers[0].self_attn.q_proj).__name__)

    n_fql = _count(model, FakeQuantLinear)
    sample = next(m for m in model.modules() if isinstance(m, FakeQuantLinear))
    print(f"  FakeQuantLinear 교체: {n_fql}/{n_linear} 레이어")
    print(f"  weight_scale shape: {sample.weight_scale.shape}  (per-group, group_size=128)")
    print(f"  input_scale: {sample.input_scale}  (weight-only → activation scale 없음)")

    text_w4 = _gen(model, tokenizer)
    print(f"  생성: {text_w4}")
    elapsed = time.time() - t0
    print(f"  소요: {elapsed:.1f}s")
    results.append({"scheme": "W4A16 RTN", "text": text_w4,
                    "elapsed": elapsed, "replaced": n_fql, "n_linear": n_linear})
    del model; torch.cuda.empty_cache()

    # ── 3. W8A8 dynamic ─────────────────────────────────────────────────────
    print("\n[3/3] W8A8 dynamic — weight+activation INT8, calibration 불필요")
    t0 = time.time()
    model = _load(model_id)
    Compressor.from_recipe("w8a8_dynamic", targets=["Linear"], ignore=["lm_head"]).compress(model)

    n_fql = _count(model, FakeQuantLinear)
    sample = next(m for m in model.modules() if isinstance(m, FakeQuantLinear))
    print(f"  FakeQuantLinear 교체: {n_fql}/{n_linear} 레이어")
    print(f"  weight_scale shape: {sample.weight_scale.shape}  (per-channel weight)")
    print(f"  input_scale: {sample.input_scale}  (dynamic → 런타임에 per-token 계산)")

    text_dyn = _gen(model, tokenizer)
    print(f"  생성: {text_dyn}")
    elapsed = time.time() - t0
    print(f"  소요: {elapsed:.1f}s")
    results.append({"scheme": "W8A8 dynamic", "text": text_dyn,
                    "elapsed": elapsed, "replaced": n_fql, "n_linear": n_linear})
    del model; torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    results = run(args.model)

    print(f"\n{'═'*65}")
    print(f"  {'Scheme':<22} {'교체':>6}  {'소요':>7}  생성 (앞 50자)")
    print(f"{'─'*65}")
    for r in results:
        repl = "-" if r["replaced"] == 0 else f"{r['replaced']}/{r['n_linear']}"
        print(f"  {r['scheme']:<22} {repl:>6}  {r['elapsed']:>6.1f}s  {r['text'][:50]}...")
    print(f"{'═'*65}")


if __name__ == "__main__":
    main()
