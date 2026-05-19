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
| GPU | Qwen3-0.6B / TinyLlama demo, perplexity 측정 시 권장 |
| CPU | unit test와 core quantization logic 실행 가능 |

비고:

- custom CUDA kernel은 필요하지 않습니다.
- fake quantization은 PyTorch 기본 연산으로 동작하므로 CPU에서도 실행 가능합니다.
- full model demo와 perplexity 평가는 GPU에서 실행하는 것을 권장합니다.
- CI는 CPU PyTorch 환경에서 test suite를 실행하도록 구성되어 있습니다.

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

compressor = Compressor.from_recipe(
    "w4a16",
    targets=["Linear"],
    ignore=["lm_head"],
)
compressor.compress(model)
compressor.save(model, "./qwen3-w4a16", tokenizer=tokenizer)
```

### W8A8 Static

```python
calib_inputs = [
    tokenizer("Quantization reduces model size.", return_tensors="pt"),
    tokenizer("Large language models are expensive to serve.", return_tensors="pt"),
]

compressor = Compressor.from_recipe(
    "w8a8",
    targets=["Linear"],
    ignore=["lm_head"],
)
compressor.compress(model, dataloader=calib_inputs)
compressor.save(model, "./qwen3-w8a8", tokenizer=tokenizer)
```

### W8A8 + SmoothQuant

```python
compressor = Compressor.from_recipe("w8a8_smoothquant", ignore=["lm_head"])
compressor.compress(model, dataloader=calib_inputs)
```

### 저장된 모델 로드

```python
from mini_compressor import load_pretrained

model = load_pretrained("./qwen3-w8a8")
```

`load_pretrained()`는 원본 HuggingFace base model을 다시 로드한 뒤 저장된 quantized weight와 scale buffer를 주입합니다. 따라서 원본 모델이 HuggingFace cache에 있거나 네트워크 접근이 가능해야 합니다.

---

## 지원 Recipe

`Compressor.from_recipe(name)`이 유일한 preset 진입점입니다. 단일 RTN quantization과 여러 단계의 알고리즘 chain을 모두 recipe로 표현합니다.

| Recipe | Pipeline | Calibration | 설명 |
|--------|----------|-------------|------|
| `w4a16` | RTN | 불필요 | INT4 weight-only, per-group 128 |
| `w4a16_gptq` | GPTQ | 필요 | Hessian 기반 INT4 weight 최적화 |
| `w4a16_awq` | AWQ-style scaling → RTN | 필요 | activation-aware INT4 scaling POC |
| `w8a8` | RTN | 필요 | INT8 weight/activation, static activation scale |
| `w8a8_dynamic` | RTN | 불필요 | INT8 weight + per-token dynamic activation scale |
| `w8a8_smoothquant` | SmoothQuant → RTN | 필요 | activation outlier smoothing 후 static W8A8 |

### Modifier 직접 조합

Recipe는 modifier list를 감싼 convenience layer입니다.

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

이 구조는 `_weight_quantizer.scale` 같은 중첩 key를 만들지 않아 HF-style state dict 복원이 단순합니다.

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

`granularity`와 `dynamic`은 독립 축입니다. 예를 들어 W8A8 dynamic activation은 `granularity="per_token"`과 `dynamic=True`의 조합으로 표현됩니다.

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

---

## Observer

weight와 activation calibration은 같은 observer 추상화를 사용합니다.

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
├── model.safetensors
├── config.json
├── quantization_config.json
└── tokenizer.*
```

`quantization_config.json` 예시:

```json
{
  "quant_type": "compressed-tensors",
  "quantization_status": "calibrated",
  "config_groups": {
    "group_0": {
      "weights": {
        "num_bits": 8,
        "type": "int",
        "symmetric": true,
        "strategy": "channel",
        "dynamic": false
      },
      "input_activations": {
        "num_bits": 8,
        "type": "int",
        "symmetric": false,
        "strategy": "tensor",
        "dynamic": false
      },
      "targets": ["Linear"],
      "ignore": ["lm_head"]
    }
  }
}
```

`quantization_status="calibrated"`는 fake quantized state를 의미합니다. weight는 floating-point tensor로 유지되고, scale metadata가 함께 저장됩니다. 실제 packed INT artifact는 backend-specific packing 이후 `"compressed"` 상태로 분리하는 것이 자연스럽습니다.

---

## 평가 결과

### Perplexity

Model: `Qwen/Qwen3-0.6B`  
Dataset: `wikitext-2-raw-v1` test split  
Evaluation: sliding-window perplexity, stride 512, max length 2048

| Scheme | PPL | Δ vs FP16 |
|--------|----:|----------:|
| FP16 baseline | 18.16 | - |
| W4A16 RTN | 25.89 | +7.73 |
| **W4A16 GPTQ** | **20.96** | **+2.80** |
| W8A8 static | 25.01 | +6.85 |
| **W8A8 + SmoothQuant** | **23.67** | **+5.51** |
| **W8A8 dynamic** | **18.48** | **+0.32** |

해석:

- W8A8 static의 손실은 activation scale에 크게 민감합니다.
- W8A8 dynamic은 token별 runtime scale을 사용해 FP16에 가깝게 유지됩니다.
- SmoothQuant는 activation outlier를 weight로 이관해 static W8A8을 개선합니다.
- GPTQ는 Hessian 정보를 이용한 error propagation으로 W4A16 RTN 대비 손실을 크게 줄입니다.

### Round-trip

Qwen3-0.6B에서 `compress → save → load → generate`를 검증했습니다.

- W4A16 RTN
- W8A8 static

### Multi-model Validation

`TinyLlama/TinyLlama-1.1B-Chat-v1.0`에서도 같은 API로 검증했습니다.

| 항목 | 결과 |
|------|------|
| Recipe | `w4a16`, `w8a8`, `w8a8_dynamic`, `w8a8_smoothquant` |
| Linear replacement | `lm_head` 제외 154개 Linear 교체 |
| SmoothQuant pair | 44개 norm-linear pair 탐색 |
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

# 다른 HF causal LM에서 실행
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
| Real INT packing | 미구현 |
| 실제 latency / memory benchmark | 미측정. fake quant only |
| Float8 execution | 의도된 동작을 명세한 stub |
| Hub upload helper | stub |
| Tensor Parallelism | 범위 외 |
| 실제 2-GPU `device_map="auto"` 실행 | 하드웨어 한계로 미측정 |

AWQ는 activation-aware INT4 scaling POC로 구현되어 있습니다. 현재 README의 주요 정확도 개선 수치는 SmoothQuant와 GPTQ 측정 결과를 기준으로 합니다.

---

## 요약

`mini-compressor`는 backend-specific INT packing 이전 단계에서 quantization scheme과 algorithm pipeline을 검증하기 위한 compact LLM compression library입니다. 핵심 초점은 module replacement, typed quantization config, composable modifier workflow, explicit serialization metadata, measurable accuracy validation입니다.
