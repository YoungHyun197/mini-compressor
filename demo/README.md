# Demo Scripts

mini-compressor의 핵심 기능을 빠르게 확인할 수 있는 독립 실행 데모 모음입니다.
각 파일은 end-to-end로 동작하며, `demo_all.py` 하나로 전체를 한 번에 실행할 수도 있습니다.

## 환경 설정

```bash
cd /home/cyh/projects/mini-compressor
source .venv/bin/activate
```

---

## 파일별 설명

### `demo_base.py` — 기본 압축 (약 30초)

calibration 없이 즉시 적용 가능한 두 가지 scheme을 실행합니다.

**실행**
```bash
python demo/demo_base.py
```

**증명 항목**
| 항목 | 내용 |
|------|------|
| Layer 교체 | `nn.Linear → FakeQuantLinear` (196/197 레이어) |
| W4A16 scale shape | `[2048, 8]` — per-group (group_size=128) 검증 |
| W8A8 dynamic | `input_scale=None` — 사전 scale 없이 런타임 per-token 계산 |
| Generate | 압축 후에도 FP16과 동일한 coherent text 생성 확인 |

**예시 출력**
```
  W4A16 RTN     196/197    7.8s  The key advantage of quantization is that...
  W8A8 dynamic  196/197    7.6s  The key advantage of quantization is that...
```

---

### `demo_smoothquant.py` — SmoothQuant activation outlier 증명 (약 45초)

SmoothQuant 적용 전후로 `q_proj` 입력 activation 분포를 직접 측정해 outlier 전이 효과를 수치로 보여줍니다.

**실행**
```bash
python demo/demo_smoothquant.py
python demo/demo_smoothquant.py --alpha 0.75   # alpha 조정
```

**증명 항목**
| 항목 | 내용 |
|------|------|
| CV 감소 | `layers.0.self_attn.q_proj` 입력의 채널별 CV 측정 (Qwen3-0.6B 기준 33% 감소) |
| Top-5 outlier | SQ 전 `3.00` → SQ 후 `1.39` (최대 outlier 채널 수치) |
| 등가 변환 검증 | 양자화 없이 SQ만 적용 시 generate가 FP16과 완전히 일치(`True`) |
| E2E 압축 | SmoothQuant → W8A8 dynamic까지 한 번에 완료 |

**수식**

```
s_j = max(|x_j|)^α / max(|w_j|)^(1-α)
norm.weight /= s,  q/k/v_proj.weight *= s
→ y = x @ W.T = (x/s) @ (W*s).T  (수치 동일)
```

**예시 출력**
```
  채널별 activation max — mean=0.411, std=0.208, CV=0.507
  SQ 후 채널별 max    — mean=0.284, std=0.096, CV=0.339

  ▶ CV 변화:  0.507 → 0.339  (CV 33.1% 감소)
  FP16과 동일: True  (✓ 수학적 등가 변환 확인됨)
```

---

### `demo_gptq.py` — GPTQ weight 재구성 오차 증명 (GPU 약 2~4분)

GPTQ 적용 후 `layers.0.q_proj`의 weight MSE와 output MSE가 RTN 대비 낮음을 수치로 증명합니다.

> GPTQ는 layer-wise Hessian 최적화 특성상 GPU 기준 2~4분, CPU 기준 10~15분 소요됩니다.

**실행**
```bash
python demo/demo_gptq.py                        # 기본 (4 samples × 128 tokens)
python demo/demo_gptq.py --n-samples 8 --seq-len 256  # 더 정확한 Hessian
```

**증명 항목**
| 항목 | 내용 |
|------|------|
| weight MSE | `GPTQ < RTN` — 동일 원본 대비 양자화 오차 수치 비교 |
| output MSE | 동일 calibration 입력에 대해 FP16 출력과의 차이 비교 |
| Generate 비교 | FP16 vs W4A16-GPTQ 생성 텍스트 나란히 출력 |

**예시 출력**
```
  RTN  weight MSE: 0.000412
  GPTQ weight MSE: 0.000183
  ▶ GPTQ가 RTN 대비 weight 재구성 오차 55.6% 감소

  RTN  output MSE: 0.003241
  GPTQ output MSE: 0.001892
  ▶ GPTQ output MSE 41.6% 감소
```

---

### `demo_all.py` — 전체 통합 데모

세 데모를 순차 실행하고 최종 비교 표와 핵심 증명 수치를 출력합니다.

**실행**
```bash
python demo/demo_all.py                 # 전체 실행 (GPU 기준 약 5분)
python demo/demo_all.py --skip-gptq    # GPTQ 생략, 약 75초
```

**옵션**
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--model` | `Qwen/Qwen3-0.6B` | 사용할 HuggingFace 모델 ID |
| `--skip-gptq` | False | GPTQ 데모 생략 |
| `--n-samples` | 4 | GPTQ Hessian 수집 배치 수 |
| `--seq-len` | 128 | GPTQ 배치당 시퀀스 길이 |

**예시 출력**
```
  Scheme                      소요   비고                         생성 (앞 40자)
  FP16 baseline              5.7s   baseline                     The key advantage of quantization...
  W4A16 RTN                  9.6s   교체 196/197                  The key advantage of quantization...
  W8A8 dynamic               9.4s   교체 196/197                  The key advantage of quantization...
  W8A8+SmoothQuant          15.9s   CV 0.507→0.339 (33.1% 감소)  The key advantage of quantization...
  W4A16-GPTQ               160.3s   weight MSE 55.6% < RTN       The key advantage of quantization...

  SmoothQuant 증명 수치:
    activation CV (q_proj 입력): 0.507 → 0.339  (CV 33.1% 감소)

  GPTQ 증명 수치:
    weight MSE: RTN=0.000412  GPTQ=0.000183  (55.6% 개선)
    output MSE: RTN=0.003241  GPTQ=0.001892  (41.6% 개선)
```

---

## 개별 vs 통합 실행 가이드

```
빠른 확인 (75초)   → python demo/demo_all.py --skip-gptq
알고리즘별 심층 확인 → python demo/demo_base.py
                      python demo/demo_smoothquant.py
                      python demo/demo_gptq.py
전체 한 번에        → python demo/demo_all.py
```

---

## 알고리즘 특성 요약

| Scheme | Calibration | 핵심 특성 |
|--------|-------------|----------|
| W4A16 RTN | 불필요 | weight-only INT4, 즉시 적용 |
| W8A8 dynamic | 불필요 | per-token runtime scale, activation INT8 |
| W8A8 + SmoothQuant | 필요 (activation max) | outlier → weight 전이 후 W8A8 |
| W4A16 GPTQ | 필요 (Hessian) | 2차 정보로 weight 오차 최소화 |
