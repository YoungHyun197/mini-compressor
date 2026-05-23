# 전체 압축 알고리즘 통합 데모 — base / SmoothQuant / GPTQ 순차 실행 후 비교 표 출력
"""
사용법:
    python demo/demo_all.py
    python demo/demo_all.py --model Qwen/Qwen3-0.6B
    python demo/demo_all.py --skip-gptq   # GPTQ 생략 (빠른 확인용)

소요 시간:
    GPU 기준: base ~30s + SmoothQuant ~45s + GPTQ ~3min = 총 약 5분
    --skip-gptq 사용 시: 약 75초

각 데모가 end-to-end로 독립 실행되므로
    python demo/demo_base.py
    python demo/demo_smoothquant.py
    python demo/demo_gptq.py
로 개별 실행도 가능합니다.
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo_base        import run as run_base
from demo_smoothquant import run as run_sq
from demo_gptq        import run as run_gptq

MODEL_ID = "Qwen/Qwen3-0.6B"


def _sep(char="─", width=70):
    print(char * width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--skip-gptq", action="store_true",
                        help="GPTQ 데모 생략 (시간 절약)")
    parser.add_argument("--n-samples", type=int, default=4,
                        help="GPTQ Hessian 수집 배치 수")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="GPTQ 배치당 시퀀스 길이")
    args = parser.parse_args()

    _sep("═")
    print("  mini-compressor  |  All-in-One Demo")
    _sep("═")

    all_start = time.time()
    summary = []

    # ── Base demo ────────────────────────────────────────────────────────────
    _sep()
    print("  STEP 1/3  Base Demo — W4A16 RTN · W8A8 dynamic (calibration 불필요)")
    _sep()
    base_results = run_base(args.model)
    for r in base_results:
        summary.append({
            "scheme":  r["scheme"],
            "text":    r["text"],
            "elapsed": r.get("elapsed", 0),
            "note":    f"교체 {r['replaced']}/{r['n_linear']}" if r["replaced"] else "baseline",
        })

    # ── SmoothQuant demo ─────────────────────────────────────────────────────
    _sep()
    print("  STEP 2/3  SmoothQuant Demo — activation outlier 전·후 분포 비교")
    _sep()
    sq_result = run_sq(args.model)
    cv_note = ""
    if sq_result["cv_before"] is not None:
        reduction = (sq_result["cv_before"] - sq_result["cv_after"]) / sq_result["cv_before"] * 100
        cv_note = f"CV {sq_result['cv_before']:.3f}→{sq_result['cv_after']:.3f} ({reduction:.1f}% 감소)"
    summary.append({
        "scheme":  "W8A8+SmoothQuant",
        "text":    sq_result["text_sq_quant"],
        "elapsed": sq_result["elapsed"],
        "note":    cv_note,
    })

    # ── GPTQ demo ────────────────────────────────────────────────────────────
    gptq_result = None
    if not args.skip_gptq:
        _sep()
        print("  STEP 3/3  GPTQ Demo — Hessian 기반 weight 최적화")
        _sep()
        gptq_result = run_gptq(args.model, args.n_samples, args.seq_len)
        w_improv = gptq_result.get("w_improv", 0)
        summary.append({
            "scheme":  "W4A16-GPTQ",
            "text":    gptq_result["text_gptq"],
            "elapsed": gptq_result["elapsed"],
            "note":    f"weight MSE {w_improv:.1f}% < RTN",
        })
    else:
        print("\n  [STEP 3/3 GPTQ 생략 — --skip-gptq 옵션]")

    total_elapsed = time.time() - all_start

    # ── 최종 비교 표 ─────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  최종 비교 요약")
    print(f"{'─'*70}")
    print(f"  {'Scheme':<22} {'소요':>7}  {'비고':<28}  생성 (앞 40자)")
    print(f"{'─'*70}")
    for r in summary:
        print(f"  {r['scheme']:<22} {r['elapsed']:>6.1f}s  {r['note']:<28}  {r['text'][:40]}...")
    print(f"{'─'*70}")

    # 알고리즘별 핵심 특성 한 줄 요약
    print()
    print("  알고리즘 특성 요약:")
    print("  ┌─────────────────────┬──────────────────────────────────────┐")
    print("  │ W4A16 RTN           │ calibration 없음, 즉시 적용 가능     │")
    print("  │ W8A8 dynamic        │ calibration 없음, per-token scale     │")
    print("  │ W8A8 + SmoothQuant  │ activation outlier → weight 전이      │")
    print("  │ W4A16 GPTQ          │ Hessian 기반, RTN 대비 MSE 감소      │")
    print("  └─────────────────────┴──────────────────────────────────────┘")

    if gptq_result is not None:
        print()
        print("  GPTQ 증명 수치:")
        print(f"    weight MSE: RTN={gptq_result['mse_rtn']:.6f}  GPTQ={gptq_result['mse_gptq']:.6f}  ({gptq_result['w_improv']:.1f}% 개선)")
        out_i = (gptq_result["out_mse_rtn"] - gptq_result["out_mse_gptq"]) / gptq_result["out_mse_rtn"] * 100
        print(f"    output MSE: RTN={gptq_result['out_mse_rtn']:.6f}  GPTQ={gptq_result['out_mse_gptq']:.6f}  ({out_i:.1f}% 개선)")

    if sq_result["cv_before"] is not None:
        print()
        print("  SmoothQuant 증명 수치:")
        reduction = (sq_result["cv_before"] - sq_result["cv_after"]) / sq_result["cv_before"] * 100
        print(f"    activation CV (q_proj 입력): {sq_result['cv_before']:.3f} → {sq_result['cv_after']:.3f}  (CV {reduction:.1f}% 감소)")

    print(f"\n  총 소요: {total_elapsed:.1f}s")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
