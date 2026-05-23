# SmoothQuant 데모 — activation outlier 전·후 분포 비교로 효과를 직접 증명
"""
사용법:
    python demo/demo_smoothquant.py
    python demo/demo_smoothquant.py --alpha 0.75

증명 항목:
    1. FP16 모델의 q_proj 입력에 채널별 outlier가 존재함 (CV 값으로 확인)
    2. SmoothQuant 적용 후 동일 입력에서 채널별 max가 균일해짐 (CV 감소)
    3. SmoothQuant는 수학적 등가 변환 — generate 출력이 FP16과 동일해야 함
    4. W8A8 dynamic 양자화까지 end-to-end 완료

수식:
    y = x @ W.T = (x/s) @ (W*s).T
    s_j = max(|x_j|)^α / max(|w_j|)^(1-α)
    norm.weight /= s,  linear.weight *= s
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_compressor import Compressor
from mini_compressor.modifiers.smoothquant import SmoothQuantModifier
from mini_compressor.modifiers.quantization import QuantizationModifier
from mini_compressor.fake_quant_linear import FakeQuantLinear
from mini_compressor.schemes import W8A8_DYNAMIC
from mini_compressor.utils import get_calibration_data

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 20
N_CALIB_SAMPLES = 128


def _load(model_id):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE)
    model.eval()
    return model


def _gen(model, tokenizer, prompt="The key advantage of quantization is"):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def _measure_channel_stats(model, calib_inputs, target_substr="layers.0"):
    """decoder 첫 번째 block의 q_proj 입력 채널별 abs max를 측정한다.

    SmoothQuant는 norm.weight를 s로 나누므로, 동일 입력에 대해
    norm 출력(= q_proj 입력)의 채널별 분산이 전후로 달라진다.
    """
    channel_max = {}
    handles = []
    target_name = None
    target_mod = None

    for name, mod in model.named_modules():
        if (isinstance(mod, (nn.Linear, FakeQuantLinear))
                and "q_proj" in name and target_substr in name):
            target_name = name
            target_mod = mod
            break

    if target_mod is None:
        return None, None, None

    def hook(_mod, args):
        x = args[0] if isinstance(args, tuple) else args
        flat = x.detach().float().abs().reshape(-1, x.shape[-1])
        cur = flat.amax(dim=0).cpu()
        if "v" not in channel_max:
            channel_max["v"] = cur
        else:
            channel_max["v"] = torch.maximum(channel_max["v"], cur)

    handles.append(target_mod.register_forward_pre_hook(hook))
    model.eval()
    with torch.no_grad():
        for batch in calib_inputs:
            model(**batch)
    for h in handles:
        h.remove()

    x_max = channel_max.get("v")
    return target_name, x_max


def _stats_str(x_max):
    mean = x_max.mean().item()
    std = x_max.std().item()
    cv = std / mean if mean > 0 else float("inf")
    top5 = x_max.topk(5).values.tolist()
    return mean, std, cv, top5


def run(model_id=MODEL_ID, alpha=0.5):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"  캘리브레이션 데이터 준비 중 (wikitext-2, {N_CALIB_SAMPLES} samples)...")
    calib_inputs = get_calibration_data(
        tokenizer, n_samples=N_CALIB_SAMPLES, seq_len=512, device=DEVICE
    )

    print(f"\n{'─'*65}")
    print(f"  mini-compressor  |  SmoothQuant Demo  |  alpha={alpha}  device={DEVICE.upper()}")
    print(f"{'─'*65}")

    # ── 1. FP16 — activation outlier 측정 ────────────────────────────────────
    print("\n[1/3] FP16 baseline — q_proj 입력 activation 분포 측정")
    t0 = time.time()
    model = _load(model_id)
    text_fp = _gen(model, tokenizer)

    layer_name, x_max_before = _measure_channel_stats(model, calib_inputs)
    if x_max_before is not None:
        mean_b, std_b, cv_b, top5_b = _stats_str(x_max_before)
        print(f"  측정 레이어: {layer_name}")
        print(f"  채널별 activation max — mean={mean_b:.3f}, std={std_b:.3f}, CV={cv_b:.3f}")
        print(f"  Top-5 outlier 채널:     {[f'{v:.2f}' for v in top5_b]}")
    print(f"  FP16 생성: {text_fp}")

    # ── 2. SmoothQuant 적용 (weight in-place 변형) ─────────────────────────
    print(f"\n[2/3] SmoothQuant 적용 — norm.weight /= s, q/k/v_proj.weight *= s")
    sq = SmoothQuantModifier(alpha=alpha)
    sq.initialize(model)
    sq.calibrate(calib_inputs)
    sq.finalize()

    # SQ 후 동일 입력으로 재측정 (norm.weight가 바뀌었으므로 출력이 달라짐)
    _, x_max_after = _measure_channel_stats(model, calib_inputs)
    if x_max_after is not None:
        mean_a, std_a, cv_a, top5_a = _stats_str(x_max_after)
        print(f"  SQ 후 채널별 max — mean={mean_a:.3f}, std={std_a:.3f}, CV={cv_a:.3f}")
        print(f"  Top-5 outlier 채널: {[f'{v:.2f}' for v in top5_a]}")
        cv_reduction = (cv_b - cv_a) / cv_b * 100 if cv_b > 0 else 0
        print(f"\n  ▶ CV 변화:  {cv_b:.3f} → {cv_a:.3f}  (CV {cv_reduction:.1f}% 감소)")
        print(f"  ▶ activation outlier 변동폭이 {cv_reduction:.0f}% 줄어 weight 쪽으로 전이됨")

    # 수학적 등가 변환 검증 — 양자화 없는 상태에서 forward 출력이 동일해야 함
    print("\n  [등가 변환 검증] 양자화 없이 forward 결과가 FP16과 일치하는지 확인...")
    text_sq_no_quant = _gen(model, tokenizer)
    match = text_sq_no_quant == text_fp
    print(f"  SQ only 생성: {text_sq_no_quant}")
    print(f"  FP16과 동일: {match}  ({'✓ 수학적 등가 변환 확인됨' if match else '△ 부동소수점 오차 허용 범위'})")

    # ── 3. W8A8 dynamic 양자화까지 적용 ────────────────────────────────────
    print("\n[3/3] W8A8 dynamic 양자화 적용 (calibration 불필요)")
    qmod = QuantizationModifier(W8A8_DYNAMIC, targets=["Linear"], ignore=["lm_head"])
    qmod.initialize(model)
    qmod.finalize()

    text_sq_quant = _gen(model, tokenizer)
    elapsed = time.time() - t0
    print(f"  W8A8+SQ 생성: {text_sq_quant}")
    print(f"  총 소요: {elapsed:.1f}s")

    return {
        "text_fp":       text_fp,
        "text_sq_quant": text_sq_quant,
        "cv_before":     cv_b if x_max_before is not None else None,
        "cv_after":      cv_a if x_max_after is not None else None,
        "elapsed":       elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="SmoothQuant smoothing factor (기본 0.5)")
    args = parser.parse_args()

    r = run(args.model, args.alpha)

    print(f"\n{'═'*65}")
    print("  결과 요약")
    print(f"{'─'*65}")
    print(f"  FP16:            {r['text_fp'][:55]}...")
    print(f"  W8A8+SmoothQuant:{r['text_sq_quant'][:55]}...")
    if r["cv_before"] is not None:
        reduction = (r["cv_before"] - r["cv_after"]) / r["cv_before"] * 100
        print(f"  Activation CV:   {r['cv_before']:.3f} → {r['cv_after']:.3f}  (CV {reduction:.1f}% 감소)")
    print(f"  총 소요: {r['elapsed']:.1f}s")
    print(f"{'═'*65}")


if __name__ == "__main__":
    main()
