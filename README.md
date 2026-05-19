# mini-compressor

`mini-compressor`는 HuggingFace causal language model을 대상으로 post-training quantization을 수행하는 PyTorch 기반 compression tool POC입니다. W4A16, W8A8 계열 recipe를 제공하고, fake quantization 기반 `model.generate()`, save/load round-trip, perplexity 평가까지 하나의 workflow로 연결합니다.

이 프로젝트는 실제 INT kernel 구현보다 **quantization unit, config, workflow, serialization 경계**를 명확히 설계하는 데 초점을 둡니다. 실제 packed INT weight와 backend-specific kernel layout은 후속 compiler/runtime 단계의 책임으로 분리했습니다.

---

## 프로젝트 구조

```text
mini_compressor/
├── schemes.py             # QuantizationSpec / QuantizationScheme / W4A16·W8A8 preset
├── fake_quant_linear.py   # nn.Linear 대체 모듈. weight/input scale을 flat buffer로 보유
├── observer.py            # MinMax / Percentile / MSE observer + distributed sync
├── recipes.py             # recipe 이름을 modifier pipeline으로 펼치는 registry
├── compressor.py          # 사용자-facing one-click API: from_recipe / compress / save
├── serialize.py           # save_pretrained / load_pretrained / quantization_config.json
└── modifiers/
    ├── base.py            # BaseModifier lifecycle: initialize / calibrate / finalize
    ├── quantization.py    # RTN W4A16 / W8A8, QuantizationMixin module replacement
    ├── smoothquant.py     # SmoothQuant: activation outlier를 weight로 이관
    ├── gptq.py            # GPTQ: Hessian 기반 W4A16 weight 최적화
    ├── awq.py             # AWQ-style activation-aware INT4 scaling POC
    └── _pair_utils.py     # SmoothQuant / AWQ 공통 norm-linear pair 탐색 유틸

tests/                     # 수식, lifecycle, serialization, sync 검증
notebooks/                 # milestone별 실험 및 검증 기록
demo.py                    # generate / save-load / perplexity end-to-end demo
```

핵심 책임 분리는 다음과 같습니다.

| 레이어 | 파일 | 역할 |
|--------|------|------|
| Quantization config | `schemes.py` | bit-width, dtype, granularity, dynamic 여부를 표현 |
| Quantization unit | `fake_quant_linear.py` | `nn.Linear`를 fake quant 가능한 module로 교체 |
| Calibration | `observer.py` | weight/activation 통계 수집과 scale/zero-point 계산 |
| Algorithm | `modifiers/` | RTN, SmoothQuant, GPTQ, AWQ-style scaling 구현 |
| User API | `compressor.py`, `recipes.py` | recipe 기반 one-click compression |
| Artifact | `serialize.py` | HF-style 저장/복원과 quantization metadata 관리 |

---

## 주요 기능

- **Recipe 기반 API**: `Compressor.from_recipe("w8a8_smoothquant").compress(...)`
- **Composable modifier**: RTN, SmoothQuant, GPTQ, AWQ-style scaling이 같은 lifecycle을 공유
- **Fake-quantized generation**: quantized model도 표준 PyTorch `model.generate()` 경로로 실행
- **Observer 선택 가능**: MinMax, Percentile, MSE 지원
- **Static/Dynamic activation quantization**: W8A8 static과 per-token dynamic W8A8 제공
- **HF-style artifact**: `model.safetensors`, `config.json`, tokenizer, `quantization_config.json` 저장
- **Large-model calibration 고려**: layer-wise sequential calibration 지원
- **Distributed calibration 고려**: observer 통계 multi-process sync 지원

---

## 실행 환경

| 항목 | 요구사항 / 검증 환경 |
|------|----------------------|
| Python | `>= 3.11` |
| PyTorch | `>= 2.0` |
| transformers | `>= 4.40` |
| safetensors | `>= 0.4` |
| CUDA | 선택사항. 모델 규모 demo는 CUDA 11.8+ 권장 |

custom CUDA kernel은 필요하지 않습니다. fake quantization은 PyTorch 기본 연산으로 동작하므로 CPU에서도 실행 가능합니다. CI는 CPU PyTorch 환경에서 test suite를 실행합니다.

---

## 설치

```bash
pip install -e ".[dev]"
```

라이브러리만 설치하려면:

```bash
pip install -e .
```

---

## 빠른 시작

### W4A16 RTN

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mini_compressor import Compressor

model_id = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

compressor = Compressor.from_recipe("w4a16", targets=["Linear"], ignore=["lm_head"])
compressor.compress(model)          # W4A16은 RTN — calibration 데이터 불필요
compressor.save(model, "./qwen3-w4a16", tokenizer=tokenizer)
```

### W8A8 Static

```python
calib_inputs = [
    tokenizer("Quantization reduces model size.", return_tensors="pt"),
    tokenizer("Large language models are expensive to serve.", return_tensors="pt"),
]

compressor = Compressor.from_recipe("w8a8", targets=["Linear"], ignore=["lm_head"])
compressor.compress(model, dataloader=calib_inputs)
compressor.save(model, "./qwen3-w8a8", tokenizer=tokenizer)
```

### W8A8 + SmoothQuant (per-token dynamic)

```python
# SmoothQuant weight 변환은 calibration data 필요, activation은 dynamic이므로 불필요
compressor = Compressor.from_recipe("w8a8_smoothquant", ignore=["lm_head"])
compressor.compress(model, dataloader=calib_inputs)
```

### targets / ignore

`targets`와 `ignore`는 모듈 경로와 클래스명 모두에 fnmatch 패턴 매칭합니다.

```python
# 클래스명 기반 — 모든 nn.Linear 대상 (기본)
compressor = Compressor.from_recipe("w8a8", targets=["Linear"], ignore=["lm_head"])

# 모듈 경로 패턴 — attention projection만 지정
compressor = Compressor.from_recipe("w8a8", targets=["model.layers.*.self_attn.*"])

# targets=None (기본값): 모든 nn.Linear 대상
compressor = Compressor.from_recipe("w8a8", ignore=["lm_head"])
```

`ignore`가 `targets`보다 우선합니다.

### 저장된 모델 로드

```python
from mini_compressor import load_pretrained

model = load_pretrained("./qwen3-w8a8")
```

원본 모델이 HuggingFace cache에 있거나 네트워크 접근이 가능해야 합니다.

---

## 지원 Recipe

`Compressor.from_recipe(name)`이 유일한 preset 진입점입니다.

| Recipe | Pipeline | Calibration | 설명 |
|--------|----------|-------------|------|
| `w4a16` | RTN | 불필요 | INT4 weight-only, per-group 128 |
| `w4a16_gptq` | GPTQ | 필요 | Hessian 기반 INT4 weight 최적화 |
| `w4a16_awq` | AWQ-style scaling → RTN | 필요 | activation-aware INT4 scaling POC |
| `w8a8` | RTN | 필요 | INT8 weight/activation, static activation scale |
| `w8a8_dynamic` | RTN | 불필요 | INT8 weight + per-token dynamic activation scale |
| `w8a8_smoothquant` | SmoothQuant → W8A8 dynamic RTN | 필요 (weight 변환용) | activation outlier smoothing + per-token dynamic INT8 |

### Modifier 직접 조합

Recipe는 modifier list를 감싼 convenience layer입니다. `alpha` 조정이나 static activation 조합 등 세부 제어가 필요하면 직접 구성합니다.

```python
from mini_compressor import Compressor, QuantizationModifier, SmoothQuantModifier, W8A8

compressor = Compressor([
    SmoothQuantModifier(alpha=0.5),
    QuantizationModifier(W8A8, targets=["Linear"], ignore=["lm_head"]),
])
compressor.compress(model, dataloader=calib_inputs)
```

---

## Quantization 설계

### Quantization Unit

대상 `nn.Linear` module을 `FakeQuantLinear`로 교체합니다.

```text
nn.Linear
   ↓
FakeQuantLinear(weight, bias, weight_scale, input_scale, ...)
```

scale과 zero-point는 flat buffer로 저장합니다.

```text
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.q_proj.weight_scale
model.layers.0.self_attn.q_proj.weight_zero_point
model.layers.0.self_attn.q_proj.input_scale
model.layers.0.self_attn.q_proj.input_zero_point
```

`_weight_quantizer.scale` 같은 중첩 key를 만들지 않아 HF-style state dict 복원이 단순합니다.

### Quantization Config

Quantization 설정은 `QuantizationSpec`과 `QuantizationScheme`으로 나눕니다.

```python
QuantizationSpec(
    num_bits=8,
    symmetric=True,
    granularity="per_channel",
    dtype="int",
    dynamic=False,
    calibration_method="minmax",
)
```

| 필드 | 의미 |
|------|------|
| `num_bits` | quantized bit-width |
| `dtype` | quantized numeric family. 현재 `int`, float8 stub |
| `symmetric` | symmetric / asymmetric quantization 여부 |
| `granularity` | `per_tensor`, `per_channel`, `per_group`, `per_token` |
| `dynamic` | scale을 runtime에 계산하는지 여부 |
| `calibration_method` | static scale 계산에 사용할 observer |

`granularity`와 `dynamic`은 독립 축입니다. W8A8 dynamic activation은 `granularity="per_token"`과 `dynamic=True`의 조합으로 표현됩니다.

### 구현 결정 사항

**Flat buffer** — `weight_scale`, `input_scale` 등을 `FakeQuantLinear`의 직속 buffer로 보유합니다. `_weight_quantizer.scale` 같은 중첩 경로 없이 state\_dict key가 단순해져 HF save/load 매핑이 명확합니다.

**`load_pretrained`에서 `from_pretrained` 사용** — `from_config().to(float16)` 방식은 `inv_freq` 같은 `persistent=False` buffer를 float16으로 변환하지 않습니다. 이 미세한 dtype 차이(1.78e-04)가 28개 attention layer를 거치면 logit 오차 4.19로 증폭됩니다. `from_pretrained(model_id, torch_dtype=float16)`으로 dtype 일관성을 확보합니다.

**`quantization_status: "calibrated"`로 끊은 이유** — 어떤 INT layout으로 패킹하느냐는 HW 데이터 경로에 의해 결정됩니다. 컴파일러가 결정해야 할 정보를 양자화 도구가 미리 박으면 HW별로 다시 변환해야 합니다. 이 도구는 정확도 검증(`calibrated`) 단계까지만 책임지고, 실제 INT 패킹(`compressed`)은 컴파일러·런타임에 위임합니다.

**`load_state_dict` 미사용** — PyTorch `_load_from_state_dict`는 None buffer를 `copy_()` 대상에서 제외합니다. `initialize(compute_scales=False)` 후 scale buffer가 None인 상태에서는 scale 주입이 무음으로 실패합니다. 이를 우회하기 위해 saved\_state를 직접 순회하여 `_parameters`와 `_buffers`에 직접 할당합니다.

---

## 압축 Workflow

모든 알고리즘은 같은 modifier lifecycle을 따릅니다.

```text
initialize(model)
    module 교체
    임시 상태 준비

calibrate(dataloader)
    통계 수집
    scale 계산
    알고리즘별 transform 적용

finalize()
    observer / hook 제거
    저장 가능한 buffer만 유지
```

이 lifecycle을 공유하는 modifier:

- `QuantizationModifier`: RTN W4A16 / W8A8
- `SmoothQuantModifier`: activation smoothing
- `GPTQModifier`: Hessian 기반 W4A16 최적화
- `AWQModifier`: activation-aware scaling POC

새 알고리즘 추가 시 `BaseModifier`를 상속한 새 파일 하나만 추가하면 됩니다.

---

## Observer

weight와 activation calibration은 같은 observer 추상화를 사용합니다. observer가 정하는 것은 **clip range**이고, rounding은 어느 경우든 round-to-nearest(RTN)입니다. `dynamic=True`인 activation은 런타임 scale을 쓰므로 observer가 생성되지 않습니다.

| Method | Key | 지원 범위 |
|--------|-----|-----------|
| MinMax | `"minmax"` | weight / activation |
| Percentile | `"percentile"` | weight / activation |
| MSE | `"mse"` | weight / activation |

분산 환경에서는 observer별 통계 성질에 맞춰 sync 방식을 다르게 둡니다.

| Observer | Sync 방식 |
|----------|-----------|
| MinMax | `all_reduce(MIN/MAX)` |
| Percentile | `all_gather_object(raw_data)` |
| MSE | `all_gather_object(raw_data)` |

Percentile, MSE는 rank별 부분 통계만으로 전역 threshold를 정확히 복원할 수 없기 때문에 raw data gather를 사용합니다.

---

## 저장 포맷

`compressor.save()`는 다음 artifact를 저장합니다.

```text
save_dir/
├── model.safetensors        — float16 weight + scale buffer
├── config.json              — HF 모델 설정
├── quantization_config.json — compressed-tensors 스펙
└── tokenizer.*              — tokenizer 파일
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

`quantization_status="calibrated"`는 weight가 float16으로 유지된 fake quant 상태를 의미합니다. 실제 packed INT artifact는 backend-specific packing 이후 `"compressed"` 상태로 분리하는 것이 자연스럽습니다.

---

## 평가 결과

### Perplexity (wikitext-2-raw-v1, Qwen3-0.6B)

측정: sliding-window, stride 512, max length 2048

| Scheme | PPL | Δ vs FP16 |
|--------|----:|----------:|
| FP16 baseline | 18.16 | — |
| W4A16 RTN | 25.89 | +7.73 |
| **W4A16 GPTQ** | **20.96** | **+2.80** |
| W8A8 static | 25.01 | +6.85 |
| W8A8 + SmoothQuant (static activation 기준) | 23.67 | +5.51 |
| **W8A8 dynamic** | **18.48** | **+0.32** |

- **W8A8 dynamic**이 FP16에 근접하는 이유: 토큰별 runtime scale을 계산하므로 activation outlier의 영향을 받지 않습니다.
- **SmoothQuant** 행은 static activation 조합 기준 측정값입니다. 현재 `w8a8_smoothquant` recipe는 per-token dynamic activation을 기본으로 사용합니다.
- **GPTQ**는 Hessian 기반 오차 전파로 W4A16 RTN 대비 -4.93 개선됩니다.

### Round-trip 일치 확인 (Qwen3-0.6B)

`compress → save → load → generate` 동일 출력 확인 (W4A16, W8A8).

### Multi-model 검증 — TinyLlama-1.1B (LLaMA 아키텍처)

| 항목 | 결과 |
|------|------|
| 모델 | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (`model_type=llama`, 22층, GQA 32:4) |
| Recipe | `w4a16`, `w8a8`, `w8a8_dynamic`, `w8a8_smoothquant` 4종 compress → generate 정상 |
| Linear replacement | `lm_head` 제외 154개 교체 |
| SmoothQuant pair 탐색 | 44개 (RMSNorm 기반, GQA 구조 그대로 반영) |
| 라이브러리 코드 수정 | 없음 |

---

## 데모 실행

```bash
# FP16과 quantized recipe generate 비교
python demo.py

# W4A16 저장 후 load_pretrained round-trip 확인
python demo.py --save /tmp/mini_compressor_demo

# wikitext-2 perplexity 측정
python demo.py --ppl

# 다른 HF causal LM에서 실행 (아키텍처 비종속 확인)
python demo.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

---

## 테스트

```bash
pytest tests/ -v
```

| Test file | 검증 내용 |
|-----------|----------|
| `test_fake_quant_linear.py` | fake quant 수식, dynamic activation |
| `test_modifier.py` | RTN lifecycle, scale shape |
| `test_smoothquant.py` | SmoothQuant pair 탐색, 등가 변환 |
| `test_gptq.py` | GPTQ scale shape, on-grid weight, RTN 대비 MSE |
| `test_awq.py` | AWQ-style scaling interface, 등가 변환 |
| `test_serialize.py` | config round-trip, artifact 생성 |
| `test_compressor.py` | recipe API, one-click workflow |
| `test_sequential_calib.py` | sequential calibration |
| `test_observer_sync.py` | 2-process gloo observer sync |

---

## 한계

| 항목 | 현재 상태 |
|------|-----------|
| Real INT packing | 미구현. fake quant only |
| 실제 latency / memory benchmark | 미측정 |
| Float8 execution | 의도된 동작을 명세한 stub |
| Hub upload helper | stub |
| Tensor Parallelism | 범위 외 |
| 실제 2-GPU `device_map="auto"` 실행 | 하드웨어 한계로 미측정 |

AWQ는 activation-aware INT4 scaling POC로 구현되어 있습니다. 정확도 개선 수치는 SmoothQuant와 GPTQ 측정 결과를 기준으로 합니다.

---

## 요약

`mini-compressor`는 backend-specific INT packing 이전 단계에서 quantization scheme과 algorithm pipeline을 검증하기 위한 compact LLM compression library입니다. 핵심 초점은 module replacement, typed quantization config, composable modifier workflow, explicit serialization metadata, measurable accuracy validation입니다.
