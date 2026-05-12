# mini-compressor

PyTorch 기반 LLM post-training quantization 라이브러리입니다. HuggingFace 모델의 W4A16(INT4), W8A8(INT8) 양자화를 one-click API로 수행하고, compressed-tensors 포맷으로 저장·복원합니다.

> **현재 개발 진행 중입니다.** SmoothQuant, GPTQ 등 고급 알고리즘은 구현 예정입니다.

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
├── schemes.py           — QuantizationSpec, QuantizationScheme, W8A8/W4A16 프리셋
├── observer.py          — activation 통계 수집 (MinMax / Percentile / MSE / KL-Divergence)
├── fake_quant_linear.py — nn.Linear 교체 모듈 (flat buffer: weight_scale, input_scale)
├── modifier.py          — initialize → calibrate → finalize 3단계 파이프라인
├── compressor.py        — one-click 진입점 (Compressor API)
└── serialize.py         — save_pretrained / load_pretrained (compressed-tensors 호환)
```

압축 흐름은 [AMD Quark](https://quark.docs.amd.com/)의 `initialize → calibrate → finalize` 패턴을, 저장 포맷은 [llm-compressor](https://github.com/vllm-project/llm-compressor)의 compressed-tensors 스펙을 따릅니다.

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

compressor = Compressor.from_scheme("w4a16", ignore=["lm_head"])
compressor.compress(model)          # W4A16은 RTN — calibration 데이터 불필요
compressor.save(model, "./qwen3-w4a16", tokenizer=tokenizer)
```

### W8A8 — weight + activation INT8

```python
compressor = Compressor.from_scheme("w8a8", ignore=["lm_head"])

calib_inputs = [
    tokenizer(text, return_tensors="pt")
    for text in calib_texts
]
compressor.compress(model, dataloader=calib_inputs)
compressor.save(model, "./qwen3-w8a8", tokenizer=tokenizer)
```

### 저장된 모델 불러오기

```python
from mini_compressor import load_pretrained

model = load_pretrained("./qwen3-w8a8")
```

---

## 지원 Scheme

| Scheme | Weight | Activation | 알고리즘 |
|--------|--------|------------|---------|
| `w4a16` | INT4, per-group (group\_size=128), symmetric | — | RTN |
| `w8a8` | INT8, per-channel, symmetric | INT8, per-tensor, asymmetric | MinMax calibration |

### Calibration Observer

`QuantizationSpec`의 `calibration_method` 필드로 선택합니다.

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

`notebooks/milestone8_round_trip.ipynb` — Qwen3-0.6B round-trip 테스트:

```
[FP]           The key advantage of quantization is that it can reduce the number of bits...
[W4A16]        The key advantage of quantization is that it can reduce the number of bits needed...
[W4A16 load]   The key advantage of quantization is that it can reduce the number of bits needed...  ✓ 일치
[W8A8]         The key advantage of quantization is that it can reduce the number of bits required...
[W8A8 load]    The key advantage of quantization is that it can reduce the number of bits required... ✓ 일치
```

---

## 테스트

```bash
pytest tests/ -v
```

| 파일 | 내용 |
|------|------|
| `test_fake_quant_linear.py` | per-tensor / per-channel / per-group fake quant 수치 검증 |
| `test_modifier.py` | initialize / calibrate / finalize 파이프라인, scale shape |
| `test_serialize.py` | scheme ↔ dict 변환 round-trip, 파일 생성 확인 |
| `test_compressor.py` | Compressor API, from\_scheme, compress, save |

---

## 구현 현황

### 완료

- [x] Fake quantization (W4A16 RTN, W8A8 MinMax calibration)
- [x] 4종 calibration observer (MinMax, Percentile, MSE, KL-Divergence)
- [x] compressed-tensors 호환 save / load + round-trip 일치 확인 (Qwen3-0.6B)
- [x] Compressor one-click API
- [x] 단위 테스트 20개, CI 통과

### 진행 중

아래 기능은 인터페이스(stub)만 정의되어 있으며 호출 시 `NotImplementedError`를 발생시킵니다. 각 stub에는 입출력 타입, docstring, 의도된 동작이 명세되어 있습니다.

- [ ] **SmoothQuant** (`QuantizationModifier.smooth()`) — activation 분포 평탄화로 W8A8 정확도 향상
- [ ] **Sequential calibration** (`calibrate(sequential=True)`) — layer-by-layer 순차 캘리브레이션 (GPU 메모리 효율화)
- [ ] **GPTQ** — Hessian 기반 weight update로 W4A16 정확도 향상 (stub 미작성)
- [ ] **Multi-model 검증** — LLaMA-3.2-1B 등 다른 아키텍처에서 동작 확인
- [ ] **lm-eval perplexity 측정** — 알고리즘별 정량 비교

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
