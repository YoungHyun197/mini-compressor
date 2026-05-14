# mini-compressor

PyTorch 기반 LLM post-training quantization 라이브러리입니다. HuggingFace 모델의 W4A16(INT4), W8A8(INT8) 양자화를 one-click API로 수행하고, compressed-tensors 포맷으로 저장·복원합니다.

> **현재 개발 진행 중입니다.** W4A16 RTN, W8A8 static/dynamic, **W8A8 + SmoothQuant**가 구현되어 있으며, GPTQ·AWQ 등 고급 알고리즘은 구현 예정입니다.

---

## 특징

- **One-click API** — `Compressor.from_scheme("w8a8").compress(model, dataloader)` 한 줄로 압축
- **HuggingFace 호환** — `safetensors` + compressed-tensors 포맷으로 저장·복원
- **Fake Quantization** — float16 weight를 유지한 채 quantization 오차 시뮬레이션 (런타임 검증 전 정확도 평가 단계)
- **다양한 Calibration** — MinMax, Percentile, MSE, KL-Divergence observer 지원
- **아키텍처 독립** — `fnmatch` 패턴 기반 targets/ignore로 모델 구조 비종속

---

## 아키텍처

```
mini_compressor/
├── schemes.py             — QuantizationSpec, QuantizationScheme, W8A8/W4A16 프리셋
├── observer.py            — activation 통계 수집 (MinMax / Percentile / MSE / KL-Divergence)
├── fake_quant_linear.py   — nn.Linear 교체 모듈 (flat buffer: weight_scale, input_scale)
├── modifiers/             — 알고리즘별 Modifier 클래스 (composition pattern)
│   ├── base.py            — BaseModifier 추상 인터페이스 (initialize / calibrate / finalize)
│   ├── quantization.py    — QuantizationModifier (RTN — W4A16 / W8A8 static / dynamic)
│   ├── smoothquant.py     — SmoothQuantModifier (activation 분포 평탄화)
│   ├── gptq.py            — GPTQModifier (stub)
│   └── awq.py             — AWQModifier (stub)
├── compressor.py          — modifier list를 받아 lifecycle을 순차 실행하는 진입점
└── serialize.py           — save_pretrained / load_pretrained (compressed-tensors 호환)

demo.py                    — W4A16 / W8A8 / W8A8-dynamic / W8A8+SmoothQuant 압축·생성·저장 end-to-end 데모
```

압축 흐름은 [llm-compressor](https://github.com/vllm-project/llm-compressor)의 **modifier composition pattern**과 `initialize → calibrate → finalize` lifecycle을, module replacement는 [AMD Quark](https://quark.docs.amd.com/) 방식을, 저장 포맷은 llm-compressor의 compressed-tensors 스펙을 따릅니다. 새 알고리즘 추가 시 `BaseModifier` 상속한 새 파일 하나만 추가하면 됩니다.

---

## 실행 환경

| 항목 | 버전 |
|------|------|
| Python | >= 3.11 |
| PyTorch | >= 2.0 |
| transformers | >= 4.40 |
| safetensors | >= 0.4 |
| CUDA | 11.8 이상 권장 (CPU 환경에서도 동작) |

GPU 사용을 권장하지만 모든 연산은 CPU에서 동작합니다. CI는 CPU(`torch` whl) 환경으로 검증합니다.

---

## 설치

```bash
pip install -e ".[dev]"
```

`[dev]`는 pytest를 추가로 설치합니다. `pip install -e .`로 테스트 도구 없이 라이브러리만 설치할 수도 있습니다.

---

## 빠른 시작

### W4A16 — weight-only INT4 (RTN)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from mini_compressor import Compressor

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype="float16")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

compressor = Compressor.from_scheme("w4a16", targets=["Linear"], ignore=["lm_head"])
compressor.compress(model)          # W4A16은 RTN — calibration 데이터 불필요
compressor.save(model, "./qwen3-w4a16", tokenizer=tokenizer)
```

### W8A8 — weight + activation INT8

```python
compressor = Compressor.from_scheme("w8a8", targets=["Linear"], ignore=["lm_head"])

calib_inputs = [
    tokenizer(text, return_tensors="pt")
    for text in calib_texts
]
compressor.compress(model, dataloader=calib_inputs)
compressor.save(model, "./qwen3-w8a8", tokenizer=tokenizer)
```

### targets — 특정 레이어 타입만 양자화

```python
# Linear만 양자화하고 싶을 때 (기본값과 동일)
compressor = Compressor.from_scheme("w8a8", targets=["Linear"], ignore=["lm_head"])

# 특정 레이어만 지정하고 싶을 때 (fnmatch 패턴 지원)
compressor = Compressor.from_scheme("w8a8", targets=["model.layers.*.self_attn.*"])

# targets=None (기본값): 모든 nn.Linear 대상
compressor = Compressor.from_scheme("w8a8", ignore=["lm_head"])
```

`targets`와 `ignore`는 독립적으로 동작하며 `ignore`가 우선합니다. `targets=None`이면 모든 `nn.Linear`가 대상입니다.

### W8A8 + SmoothQuant — activation 분포 평탄화

W8A8 static의 정확도 손실은 주로 activation outlier에서 옵니다. SmoothQuant는 outlier를 weight 쪽으로 옮겨 per-tensor int8 양자화 오차를 줄입니다.

```python
from mini_compressor import (
    Compressor,
    QuantizationModifier,
    SmoothQuantModifier,
    W8A8,
)

compressor = Compressor([
    SmoothQuantModifier(alpha=0.5),
    QuantizationModifier(W8A8, targets=["Linear"], ignore=["lm_head"]),
])
compressor.compress(model, dataloader=calib_inputs)
compressor.save(model, "./qwen3-w8a8-smq", tokenizer=tokenizer)
```

`Compressor`는 modifier list를 받아 각 modifier에 `initialize → calibrate → finalize`를 순차 호출합니다. SmoothQuant + W8A8, AWQ + W4A16 같은 알고리즘 chain을 list 순서로 자연스럽게 표현합니다.

### 저장된 모델 불러오기

```python
from mini_compressor import load_pretrained

model = load_pretrained("./qwen3-w8a8")
```

`load_pretrained`는 저장 디렉토리의 `config.json`에서 원본 모델 ID(`_name_or_path`)를 읽어 HuggingFace cache 또는 네트워크에서 베이스 모델을 로드한 뒤, `model.safetensors`의 weight와 scale을 주입합니다. **원본 모델이 HF cache에 있거나 네트워크 접근이 가능해야 합니다.**

---

## 지원 Scheme

| Scheme | Weight | Activation | Calibration |
|--------|--------|------------|-------------|
| `w4a16` | INT4, per-group (group\_size=128), symmetric | — | RTN (불필요) |
| `w8a8` | INT8, per-channel, symmetric | INT8, per-tensor, asymmetric | MinMax (static) |
| `w8a8_dynamic` | INT8, per-channel, symmetric | INT8, per-token, symmetric | 불필요 (런타임 계산) |

`w8a8_dynamic` 사용 예시.

```python
compressor = Compressor.from_scheme("w8a8_dynamic", targets=["Linear"], ignore=["lm_head"])
compressor.compress(model)  # calibration 데이터 불필요
```

### QuantizationSpec 두 축 설계

`granularity`와 `dynamic`은 독립적인 두 축입니다.

| 축 | 의미 | 값 |
|----|------|----|
| `granularity` | scale이 커버하는 차원 | `"per_tensor"` / `"per_channel"` / `"per_group"` / `"per_token"` |
| `dynamic` | scale 계산 시점 | `False` (calibration, 정적) / `True` (inference, 동적) |

`per_token + dynamic=True` 조합이 per-token dynamic quantization입니다. 시퀀스 길이가 가변이므로 `per_token`은 실질적으로 `dynamic=True`와 함께 씁니다.

### Calibration Observer

`QuantizationSpec`의 `calibration_method` 필드로 선택합니다. `dynamic=True`인 scheme은 calibration이 필요 없으므로 observer가 생성되지 않습니다.

| 키 | 방식 | 비고 |
|----|------|------|
| `"minmax"` | running min/max 누적 | 빠름, outlier에 민감 |
| `"percentile"` | percentile clip | outlier 제거, 하이퍼파라미터 필요 |
| `"mse"` | alpha grid-search로 MSE 최소화 | 정확하지만 느림 |
| `"kl_divergence"` | histogram 기반 KL divergence 최소화 | TensorRT 방식 |

---

## HuggingFace 호환 저장 포맷

`save_pretrained`는 아래 두 파일을 생성합니다.

```
save_dir/
├── model.safetensors        — float16 weight + scale buffer
├── config.json              — HF 모델 설정
├── quantization_config.json — compressed-tensors 스펙 (아래 예시 참고)
└── tokenizer.*              — tokenizer 파일 (tokenizer 전달 시)
```

`quantization_config.json` 예시 (W8A8):

```json
{
  "quant_type": "compressed-tensors",
  "quantization_status": "calibrated",
  "config_groups": {
    "group_0": {
      "weights": {
        "num_bits": 8, "type": "int", "symmetric": true, "strategy": "channel"
      },
      "input_activations": {
        "num_bits": 8, "type": "int", "symmetric": false, "strategy": "tensor"
      },
      "targets": ["Linear"],
      "ignore": ["lm_head"]
    }
  }
}
```

> `"quantization_status": "calibrated"` — 현재는 fake quantization 단계입니다. weight는 float16으로 저장됩니다. `"compressed"` (실제 INT4/INT8 패킹)는 컴파일러·런타임이 처리하는 다음 단계로, 툴체인 레이어 간 책임을 명확히 분리합니다.

---

## 검증 결과

### Perplexity (wikitext-2-raw-v1, Qwen3-0.6B)

측정 환경 두 가지를 분리해 표기합니다.

**(A) `python demo.py --ppl` — calibration 5 샘플 (재현 가능한 기본 데모 조건)**

| Scheme | PPL | Δ vs FP16 |
|--------|----:|----------:|
| FP16 (baseline) | 18.16 | — |
| W4A16 RTN | 25.89 | +7.73 |
| W8A8 static | 25.01 | +6.85 |
| **W8A8 + SmoothQuant** | **23.67** | **+5.51** |
| W8A8 dynamic | **18.48** | **+0.32** |

W8A8 static 25.01 → W8A8 + SmoothQuant 23.67 — 같은 calibration 조건에서 **-1.34 개선**. per-tensor static의 본질적 한계 안에서 SmoothQuant이 activation outlier를 weight로 이관해 만든 차이.

**(B) `notebooks/milestone11_perplexity.ipynb` — calibration 128 샘플 (대규모 calibration)**

| Scheme | PPL | Δ vs FP16 |
|--------|----:|----------:|
| FP16 (baseline) | 18.16 | — |
| W4A16 RTN | 25.89 | +7.73 |
| W8A8 static | 27.75 | +9.59 |
| W8A8 dynamic | 18.48 | +0.32 |

> **흥미로운 관찰**: 128 샘플(B)이 5 샘플(A)보다 W8A8 static PPL이 더 나쁨 (27.75 vs 25.01). MinMax observer가 더 큰 데이터셋에서 outlier에 더 많이 노출돼 per-tensor scale을 과도하게 넓게 잡는 현상 — SmoothQuant 도입 동기를 정확히 보여주는 부수 결과.

측정 방식 공통: sliding window (stride=512, max\_len=2048), wikitext-2 test split.

> **W8A8 dynamic이 FP16에 근접한 이유**: 토큰별로 runtime scale을 계산하므로 activation outlier에 영향을 받지 않습니다.

### Round-trip 일치 확인 (Qwen3-0.6B)

`notebooks/milestone8_round_trip.ipynb` — compress → save → load → generate 동일 출력 확인:

```
[W4A16]        The key advantage of quantization is that it can reduce the number of bits needed...
[W4A16 load]   The key advantage of quantization is that it can reduce the number of bits needed...  ✓ 일치
[W8A8]         The key advantage of quantization is that it can reduce the number of bits required...
[W8A8 load]    The key advantage of quantization is that it can reduce the number of bits required... ✓ 일치
```

---

## 데모 실행

```bash
# 기본: 네 scheme 압축 후 generate 결과 비교 (W4A16 / W8A8 / W8A8-dynamic / W8A8+SmoothQuant)
python demo.py

# W4A16 저장 + load_pretrained round-trip 확인
python demo.py --save /tmp/demo_save

# wikitext-2 perplexity 측정 추가 (시간 소요)
python demo.py --ppl

# 전체 옵션 동시 실행
python demo.py --save /tmp/demo_save --ppl
```

---

## 테스트

```bash
pytest tests/ -v
```

| 파일 | 내용 |
|------|------|
| `test_fake_quant_linear.py` | per-tensor / per-channel / per-group fake quant 수치 검증 |
| `test_modifier.py` | QuantizationModifier initialize / calibrate / finalize, scale shape |
| `test_smoothquant.py` | SmoothQuant pair 탐색, 등가 변환 수치 검증, Compressor chain 통합 |
| `test_serialize.py` | scheme ↔ dict 변환 round-trip, 파일 생성 확인 |
| `test_compressor.py` | Compressor API, from\_scheme, compress, save |

---

## 구현 현황

### 완료

- [x] Fake quantization (W4A16 RTN, W8A8 static, W8A8 dynamic per-token)
- [x] 4종 calibration observer (MinMax, Percentile, MSE, KL-Divergence)
- [x] compressed-tensors 호환 save / load + round-trip 일치 확인 (Qwen3-0.6B)
- [x] Compressor one-click API + modifier composition (BaseModifier + per-algorithm class)
- [x] **SmoothQuant** (`SmoothQuantModifier`) — activation 분포 평탄화로 W8A8 정확도 향상
- [x] End-to-End 데모 (`demo.py`) — 네 scheme (W4A16 / W8A8 / W8A8-dynamic / W8A8+SmoothQuant) compress → generate → save → load 전체 흐름
- [x] 단위 테스트 27개, CI 통과

### 진행 중

아래 기능은 인터페이스(stub)만 정의되어 있으며 호출 시 `NotImplementedError`를 발생시킵니다. 각 stub에는 입출력 타입, docstring, 의도된 동작이 명세되어 있습니다.

- [ ] **GPTQ** (`GPTQModifier`) — Hessian 기반 weight update로 W4A16 정확도 향상
- [ ] **AWQ** (`AWQModifier`) — activation magnitude 기반 per-channel scaling으로 W4A16 정확도 향상
- [ ] **Sequential calibration** (`QuantizationModifier.calibrate(sequential=True)`) — layer-by-layer 순차 캘리브레이션 (GPU 메모리 효율화)
- [ ] **Float8** (`QuantizationSpec(dtype="float8")`) — E4M3/E5M2 fake quant 경로 (PyTorch >= 2.1 필요)
- [ ] **Multi-GPU 지원** — `BaseObserver.sync()` all-reduce 동기화, `device_map="auto"` 호환 검증 (rank 0 저장 가드만 구현됨)
- [ ] **HuggingFace Hub 업로드** (`Compressor.save_to_hub()`) — 로컬 저장 후 Hub push
- [ ] **Multi-model 검증** — LLaMA-3.2-1B 등 다른 아키텍처에서 동작 확인
- [ ] **W8A8 + SmoothQuant perplexity 측정** — `python demo.py --ppl` 로 측정 후 README 표 갱신

---

## 주요 설계 결정

### Flat buffer 방식
`weight_scale`, `input_scale` 등을 `FakeQuantLinear`의 직속 buffer로 보유합니다. `_weight_quantizer.scale` 같은 중첩 경로 없이 state\_dict key가 단순해져 HF save/load 매핑이 명확합니다.

### load_pretrained에서 `from_pretrained` 사용 이유
`from_config().to(float16)` 방식은 `inv_freq` 같은 `persistent=False` buffer를 float16으로 변환하지만, 원본 모델은 float32로 유지합니다. 이 1.78e-04 차이가 28개 attention layer를 거치면 logit 오차 4.19로 증폭됩니다. `from_pretrained(model_id, torch_dtype=float16)`으로 dtype 일관성을 확보합니다.

### `quantization_status: "calibrated"` vs `"compressed"`
현재 저장 상태(`calibrated`)는 float16 weight + scale buffer를 함께 저장한 fake quant 상태입니다. 실제 INT 패킹(`compressed`)은 컴파일러 단에서 처리하며, 이 툴은 그 이전 단계인 정확도 검증에 집중합니다.

### `load_state_dict` 미사용 이유
PyTorch의 `_load_from_state_dict`는 None buffer를 `local_state`에서 제외하여 `copy_()`를 호출하지 않습니다. `initialize(compute_scales=False)` 후 scale buffer가 None인 상태에서는 scale 주입이 무음으로 실패합니다. 이를 우회하기 위해 saved\_state를 직접 순회하여 `_parameters`와 `_buffers`에 직접 할당합니다.
