# mini-compressor context notes

설계 결정과 그 근거를 세션 간 이어받기 위한 기록. 코드 외부에서 이유를 추적할 수 없는 결정만 기록한다.

---

## 파일 구조

```
mini_compressor/
  schemes.py             — QuantizationSpec, QuantizationScheme, W8A8/W4A16/W8A8_DYNAMIC, SCHEME_REGISTRY
  observer.py            — BaseObserver(+sync stub), MinMax/Percentile/MSE/KLDivergence observer
  fake_quant_linear.py   — FakeQuantLinear (flat buffer, dynamic 분기, float8 stub)
  modifiers/             — modifier composition pattern (llm-compressor 정렬)
    base.py              — BaseModifier 추상 인터페이스 (initialize/calibrate/finalize)
    quantization.py      — QuantizationModifier (RTN)
    smoothquant.py       — SmoothQuantModifier (실구현)
    gptq.py              — GPTQModifier (stub)
    awq.py               — AWQModifier (stub)
  compressor.py          — Compressor (modifier list 수용, from_scheme/compress/save, save_to_hub stub)
  serialize.py           — save_pretrained / load_pretrained (compressed-tensors 호환)

demo.py                  — W4A16 / W8A8 / W8A8-dynamic / W8A8+SmoothQuant end-to-end 데모 (--save, --ppl 옵션)
```

---

## 설계 결정

### 1. FakeQuantLinear — flat buffer 방식

`weight_scale`, `weight_zero_point`, `input_scale`, `input_zero_point`를 모두 직속 buffer로 보유.

- **이유**: `_weight_quantizer.scale` 같은 중첩 경로 대신 flat key로 state_dict를 단순하게 유지. HF save/load 시 key 매핑이 명확해짐.
- **참고**: Furiosa llm-compressor 스타일.

### 2. observer — FakeQuantLinear가 소유

`FakeQuantLinear.__init__`에서 `input_observer`를 생성하고, `forward()`에서 `observer.update(x)` 호출.

- **이유**: modifier가 observer를 직접 관리하지 않아도 됨. `calibrate()`는 model forward만 돌리면 각 layer가 알아서 통계 수집.
- **결과**: modifier.calibrate()는 forward loop + 루프 후 scale 채우기만 담당.

### 3. QuantizationModifier — initialize / calibrate / finalize 3단계

원래 PROGRESS.md 기준 M5(calibrate)와 M7(리팩토링)이 분리되어 있었으나, 어차피 calibrate를 제대로 쓰려면 initialize가 필요하므로 M5+M7 병합해서 한번에 구현.

- M6(W4A16 RTN)은 modifier 완성 후 진행.
- M7은 별도 milestone으로 두지 않음.

### 4. weight scale 계산 (RTN)

| granularity | 수식 | scale shape |
|-------------|------|-------------|
| per_channel | `max(\|w\|, dim=1) / qmax` | `[out_features]` |
| per_group   | `max(\|w_grouped\|, dim=2) / qmax` | `[out_features, in_features // group_size]` |
| per_tensor  | `max(\|w\|) / qmax` | scalar |

- symmetric이므로 zero_point = 0.
- `_compute_weight_scale()`은 modifier.py 모듈 수준 함수로 분리 (QuantizationModifier 메서드 아님).

### 5. targets / ignore 우선순위

`ignore`가 `targets`보다 우선. `ignore`에 있으면 targets 매칭 여부 무관하게 제외.

`targets=None`이면 모든 `nn.Linear` 대상.

### 6. calibrate() — activation=None이면 early return

W4A16는 `scheme.activation = None`이므로 calibrate()에서 즉시 return.

### 7. finalize() — input_observer를 None으로 설정

`del` 대신 `None` 대입. state_dict 오염 방지 (observer는 buffer가 아닌 일반 attribute이므로 state_dict에는 포함 안 됨).

---

## observer.py 설계

### Observer 종류

| 클래스 | calibration_method 키 | 방식 |
|--------|----------------------|------|
| MinMaxObserver | `"minmax"` | running min/max 누적 |
| PercentileObserver | `"percentile"` | 전체 데이터 수집 후 percentile clip |
| MSEObserver | `"mse"` | alpha grid-search로 MSE 최소화 |
| KLDivergenceObserver | `"kl_divergence"` | histogram 기반 KL divergence 최소화 |

### 핵심 설계 결정

- **observer 선택 방식**: `QuantizationSpec.calibration_method` 문자열 → `build_observer()`가 `OBSERVER_REGISTRY`에서 클래스 조회. modifier나 FakeQuantLinear가 Observer 클래스를 직접 import하지 않아도 됨.

- **MinMaxObserver의 buffer**: `min_val`, `max_val`을 `register_buffer`로 등록. `model.to(device)` 시 자동으로 같은 device로 이동.

- **PercentileObserver / MSEObserver / KLDivergence**: raw data를 CPU list로 수집 (`_data: list[torch.Tensor]`). GPU 메모리 절약 목적. compute 시점에 `torch.cat`으로 합침.

- **`_scale_zp_from_range()` 공통 메서드**: min/max → scale, zero_point 계산 로직을 BaseObserver static method로 분리. 모든 observer가 공유.

- **zero 포함 보장**: `min_val = min(min_val, 0)`, `max_val = max(max_val, 0)`. 실수값 0이 반드시 표현 가능한 범위 내에 들어오도록 강제.

### scale / zero_point 계산 공식

```
scale = (max_val - min_val) / (qmax - qmin)

symmetric:   zero_point = 0
asymmetric:  zero_point = clamp(round(qmin - min_val / scale), qmin, qmax)
```

W8A8 activation (asymmetric int8): qmin=-128, qmax=127

---

## W8A8 스펙 요약

```
weight:     per_channel, int8, symmetric, axis=0
activation: per_tensor,  int8, asymmetric, calibration=minmax
```

## W4A16 스펙 요약

```
weight:     per_group, int4, symmetric, group_size=128, axis=1
activation: None
```

---

## 완료된 Milestone 요약

| Milestone | 핵심 파일 | 상태 |
|-----------|-----------|------|
| M1 | notebooks/milestone1_model_load.ipynb | 완료 |
| M2 | (notebook에서 확인) | 완료 |
| M3 | fake_quant_linear.py (_fake_quantize_*) | 완료 |
| M4 | schemes.py, fake_quant_linear.py, tests/test_fake_quant_linear.py | 완료 |
| M5+M7 | observer.py, modifier.py | 완료 |
| M8 | modifier.py, serialize.py, compressor.py, tests/ | 완료 |
| M8-5 | schemes.py(W8A8_DYNAMIC), fake_quant_linear.py(dynamic 분기), modifier.py | 완료 |
| M9 | README.md | 완료 |
| M11 | notebooks/milestone11_perplexity.ipynb, README perplexity 표 | 완료 |
| M12 | demo.py | 완료 |

단위 테스트: 24개 통과 (CI 연동)

---

### 8. M6 W4A16 RTN → M5 통합

원래 M6에서 구현 예정이던 W4A16 RTN per-group fake quant가 M5 구현 시점에 이미 완성됨.

- `_group_fake_quant()`: fake_quant_linear.py에 구현
- `_compute_weight_scale()` per_group 분기: modifier.py에 구현
- `test_initialize_w4a16_scale_shape`: scale shape 검증 통과

M6은 SmoothQuant(6-A) / GPTQ(6-B) 확장 milestone으로 재정의. RTN 관련 항목은 M5 완료로 처리.

---

## 미완료 / 다음 할 일

- [ ] M6-A: SmoothQuant (`modifier.smooth()` stub → 구현)
- [ ] M6-B: GPTQ (`modifier.gptq()` stub → 구현)
- [ ] M13: Multi-GPU — Observer all-reduce, device_map="auto" 호환 검증

---

## M8 설계 결정 (2026-05-12 discussion)

### 9. Compressor API — one-click 방식

```python
compressor = Compressor.from_scheme("w8a8", ignore=["lm_head"])
compressor.compress(model, dataloader)  # initialize → calibrate → finalize 자동
```

llm-compressor의 `oneshot()`, Quark의 `quantizer.quantize_model()` 모두 원클릭 진입점을 제공함. 발표 데모 관점에서도 한 줄이 강함.

내부에서 `QuantizationModifier`를 생성하고 3단계를 순서대로 호출. Modifier 구조는 그대로 유지.

### 10. serialize.py — 책임 분리

`QuantizationScheme → dict` 직렬화 로직은 `serialize.py` 내부 함수로 처리.

- `schemes.py`는 "무엇을 양자화할 것인가"만 담당
- `serialize.py`가 "어떻게 저장하는가" 담당
- `schemes.py`에 `to_dict()` / `from_dict()` 추가하지 않음

### 11. quantization_config.json — compressed-tensors 포맷 준수

Furiosa-llm이 HF 호환 + compressed-tensors 지원 예정이므로 해당 스펙을 그대로 따름.

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

결정 사항:
- `quant_type`: `"compressed-tensors"` (Furiosa-llm 로드 시 자동 호환 목표)
- `quantization_status`: `"calibrated"` — fake quant는 weight가 float16 그대로. `"compressed"`는 real int packing 완료 시.
- `targets`: `["Linear"]` — 클래스 이름. fnmatch 패턴 아님. compressed-tensors 스펙 준수.
- `input_activations`: W4A16처럼 activation 없으면 `null` 명시. 필드 생략하지 않음.
- 저장 위치: `{save_dir}/quantization_config.json` 별도 파일. `config.json` 안에 포함하지 않음 (llm-compressor 방식).

### 12. save_pretrained / load_pretrained 시그니처

```python
# save
save_pretrained(model, save_dir, tokenizer=None)
# → model.save_pretrained(save_dir)  (safetensors + config.json 자동 생성)
# → quantization_config.json 별도 저장
# → tokenizer 있으면 tokenizer.save_pretrained(save_dir)

# load
load_pretrained(save_dir)
# → quantization_config.json → scheme, ignore 복원
# → AutoModelForCausalLM.from_pretrained(config._name_or_path) — inv_freq dtype 보존 목적
# → modifier.initialize(compute_scales=False) — 구조만 생성, scale 계산 안함
# → input_observer = None (W8A8: initialize가 observer를 재생성하므로 명시적 제거)
# → saved_state 직접 주입 루프 — weight는 data.copy_(), buffer는 _buffers 직접 할당
```

llm-compressor의 `_process_model_before_weight_loading()` (구조 생성) / weight 로딩 / `_process_model_after_weight_loading()` 패턴과 동일 철학.

### 14. load_pretrained — load_state_dict 미사용 이유

PyTorch의 `_load_from_state_dict`는 `local_state = {k: v for ... if v is not None}` 필터링으로 None buffer를 건너뜀. `initialize(compute_scales=False)` 후 `weight_scale` 등이 None이므로 `copy_()`가 호출되지 않아 safetensors 값이 주입되지 않음.

해결: `saved_state`를 직접 순회하여 parameter는 `data.copy_()`, buffer는 `mod._buffers[attr] = tensor`로 직접 할당.

### 15. load_pretrained — from_config 대신 from_pretrained 사용 이유

`from_config(hf_config).to(torch.float16)`은 `inv_freq` 같은 `persistent=False` 버퍼를 float16으로 변환. 반면 원본 `from_pretrained(..., torch_dtype=float16)`은 이 버퍼를 float32로 유지 (HF가 non-persistent buffer에 dtype 미적용).

결과: inv_freq에 1.78e-04 오차 발생 → Qwen3-0.6B 28개 attention 레이어를 거쳐 logit max diff 4.19로 증폭. round-trip 실패.

해결: `from_pretrained(model_id, torch_dtype=float16)` 사용. weights는 step 5에서 safetensors로 덮어씌우므로 이중 로딩이지만 dtype 일관성 확보.

### 16. load_pretrained — W8A8 observer 재활성화 문제

`initialize(compute_scales=False)`는 `scheme.activation is not None`이면 MinMaxObserver를 새로 생성. 원본은 `finalize()`가 이미 `input_observer = None`으로 제거한 상태.

결과: 로드된 모델에 observer 서브모듈이 남아 `forward()`에서 `update(x)` 호출 → `min_val`, `max_val` buffer 394개 추가, 연산 흐름 변경.

해결: `initialize(compute_scales=False)` 직후 모든 FQL에 대해 `mod.input_observer = None` 명시적 설정.

### 13. modifier.initialize() — compute_scales 파라미터 추가

load 흐름에서 scale 재계산 낭비를 없애기 위해 `initialize(compute_scales=True)` 파라미터 추가.

```python
def initialize(self, compute_scales: bool = True) -> None:
    for name, mod in to_replace:
        fql = FakeQuantLinear.from_float(mod, self.scheme)
        if compute_scales:
            fql.weight_scale, fql.weight_zero_point = _compute_weight_scale(...)
        setattr(parent, attr, fql)
```

- 압축 흐름: `modifier.initialize()` (기본값 True, 기존 동작 그대로)
- load 흐름: `modifier.initialize(compute_scales=False)` (구조만, scale은 state_dict에서 채움)
- llm-compressor, Quark 모두 구조 생성과 scale 로딩을 분리하는 동일 패턴 사용

---

## per-token dynamic quantization 설계 결정 (2026-05-12)

### 17. granularity와 dynamic — 두 축 분리 설계

**배경**: per-token dynamic quantization 구현 시 "granularity에 per_token을 추가" vs "`dynamic=True`로만 표현" 중 선택.

**결정**: llm-compressor(compressed-tensors), AMD Quark와 동일하게 **두 축 분리**로 설계.

- `granularity` = scale이 커버하는 *공간적 차원* (per_tensor / per_channel / per_group / per_token)
- `dynamic` = scale 계산 *시점* (False: calibration 정적 / True: inference 동적)

**이유**: 두 개념이 독립적이기 때문.
- `dynamic=True`만 있으면 per-tensor dynamic인지 per-token dynamic인지 표현 불가
- `per_token` granularity만 있으면 정적/동적 여부를 표현 불가 (per_token은 시퀀스 길이 가변 → 사실상 dynamic과 묶임)

**외부 인터페이스**: 두 축 조합은 verbosity가 높으므로 preset으로 노출.
```python
W8A8_DYNAMIC = QuantizationScheme(
    weight=QuantizationSpec(granularity="per_channel", ...),
    activation=QuantizationSpec(granularity="per_token", dynamic=True, ...),
)
Compressor.from_scheme("w8a8_dynamic")  # 사용자는 preset만 알면 됨
```

### 18. dynamic=True 시 observer 미생성 및 calibrate() early return

`dynamic=True`인 scheme은 calibration이 필요 없으므로:
- `FakeQuantLinear.__init__`: `scheme.activation.dynamic`이 True이면 observer 생성하지 않음
- `modifier.calibrate()`: `scheme.activation.dynamic`이 True이면 early return
- `FakeQuantLinear.forward()`: `is_dynamic=True`이면 `input_scale` 없어도 `_fake_quantize_activation()` 호출

scale 계산 위치 — `_fake_quantize_activation()` 내부에서 분기:
```python
if spec.dynamic:
    if spec.granularity == "per_token":
        s = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    else:  # per_tensor dynamic
        s = x.detach().abs().amax().clamp(min=1e-8) / qmax
    zp = torch.zeros_like(s)  # dynamic은 symmetric → zero_point=0
else:
    s = self.input_scale  # static: calibration에서 사전 계산
    zp = self.input_zero_point or zeros_like(s)
```

dynamic은 symmetric=True로 강제 — min/max 두 값 계산 오버헤드를 피하고, 하드웨어 최적화와 정렬.

---

## 추가 설계 결정 (2026-05-13)

### 19. targets — 모듈 경로 + 클래스명 양쪽 매칭

**문제**: `targets=["Linear"]` 전달 시 `fnmatch.fnmatch("proj", "Linear")`가 False → 교체 없음.

**원인**: `_should_replace`에서 `name`(모듈 경로, 예: "proj")에만 fnmatch를 적용함.

**결정**: 클래스명(`type(module).__name__`)에도 fnmatch를 적용해 OR 조건으로 처리.

```python
class_name = type(module).__name__
return any(
    fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(class_name, pat)
    for pat in self.targets
)
```

- `targets=["Linear"]` → 클래스명 "Linear" 매칭 → 모든 `nn.Linear` 교체
- `targets=["model.layers.*.self_attn.*"]` → 경로 패턴 매칭 → 기존 동작 그대로
- 두 방식이 하나의 targets 리스트에 혼재 가능

### 20. save_pretrained — base_model_name_or_path 저장

**문제**: `model.save_pretrained(save_dir)` 호출 후 HuggingFace가 `config.json`의 `_name_or_path`를 save_dir 절대경로로 덮어씀. 이후 `load_pretrained`에서 `hf_config._name_or_path`를 읽으면 로컬 경로가 반환되어 `from_pretrained(local_path)`가 호출되고, safetensors의 `weight_scale` 등 추가 키를 UNEXPECTED로 경고함.

**결정**: `save_pretrained` 호출 전에 원본 `model_id`를 읽어 `quantization_config.json`에 `base_model_name_or_path` 필드로 저장. `load_pretrained`는 이 필드를 우선 사용하고, 없으면 `hf_config._name_or_path`로 fallback.

```python
# save 시: model.save_pretrained 전에 읽어야 함
model_id = getattr(getattr(model, "config", None), "_name_or_path", None)
# quantization_config.json에 포함
config_dict["base_model_name_or_path"] = model_id

# load 시
model_id = config_dict.get("base_model_name_or_path")
if not model_id:
    hf_config = AutoConfig.from_pretrained(save_dir)
    model_id = hf_config._name_or_path
```

결과: `from_pretrained("Qwen/Qwen3-0.6B")` 로 HF cache에서 로드 → UNEXPECTED 경고 없음.

### 21. stub 확장 목록 (인터페이스 정의 완료, 구현 예정)

| stub | 파일 | 위치 | 설명 |
|------|------|------|------|
| `GPTQModifier` | modifiers/gptq.py | 클래스 | Hessian 기반 W4A16. BaseModifier 상속 |
| `AWQModifier` | modifiers/awq.py | 클래스 | activation magnitude 기반 W4A16. BaseModifier 상속 |
| `BaseObserver.sync()` | observer.py | 메서드 | multi-GPU all-reduce 자리 표시 |
| `save_to_hub()` | compressor.py | 메서드 | HF Hub 업로드. 파일 구조가 이미 HF 호환 |
| float8 경로 | fake_quant_linear.py | 분기 | `spec.dtype == "float8"` 시 NotImplementedError |

모든 stub은 입출력 타입, 의도된 동작, 참고 논문을 docstring에 명세함.

---

## 2026-05-14 — Modifier composition 리팩토링 + SmoothQuant 실구현

### 22. modifier.py → modifiers/ 디렉토리 (composition pattern)

**문제**: 원래 `QuantizationModifier` 한 클래스에 `smooth/gptq/awq` 메서드가 박혀 있어 god-class 형태.

**해결**: llm-compressor의 modifier composition pattern으로 재구성.

| 변경 전 | 변경 후 |
|---------|---------|
| `modifier.py:QuantizationModifier.smooth()` | `modifiers/smoothquant.py:SmoothQuantModifier` |
| `modifier.py:QuantizationModifier.gptq()` | `modifiers/gptq.py:GPTQModifier` |
| `modifier.py:QuantizationModifier.awq()` | `modifiers/awq.py:AWQModifier` |
| `Compressor.compress(model, dataloader)` 내부에서 단일 modifier | `Compressor(modifiers=[...])` 가 list 순회 |

**핵심 변경.**
- `BaseModifier` 추상 인터페이스 (`modifiers/base.py`): `initialize(model) / calibrate(dataloader, num_samples) / finalize()`.
- `QuantizationModifier.__init__(scheme, ...)` — `compute_scales`를 생성자 인자로 이동. `initialize(model)`이 model을 받는 형태.
- `Compressor.__init__(modifiers: list[BaseModifier])` — list 수용.
- `Compressor.from_scheme(name, ...)` — 단일 `QuantizationModifier`로 감싸서 list 생성. **backward compat 유지**.
- `Compressor.save()` — modifier list에서 `QuantizationModifier` 인스턴스를 찾아 그 scheme/ignore로 `save_pretrained` 호출.
- `serialize.py:load_pretrained` — 새 시그니처에 맞춰 `QuantizationModifier(scheme, ignore=..., compute_scales=False)` + `modifier.initialize(model)`로 갱신.

**근거 (왜 composition pattern?):**
- 조합 표현: `[SmoothQuantModifier, QuantizationModifier(W4A16)]`, `[SmoothQuantModifier, GPTQModifier(W4A16)]` 같은 chain을 list 순서로 자연스럽게 표현.
- 변경 범위 제한: 새 알고리즘 = `modifiers/<algo>.py` 한 파일 추가. 기존 modifier 무수정.
- 면접 답변 강화: road_map 17.3, 19번 질문 ("새 알고리즘 추가 시 변경 범위") 답변이 "새 파일 추가만"으로 강화됨.

### 23. SmoothQuantModifier 알고리즘 상세

**적용 대상 자동 탐색** (`_find_smooth_pairs`):
- `*.input_layernorm` → `*.self_attn.{q_proj, k_proj, v_proj}` (그룹 단위 smooth)
- `*.post_attention_layernorm` → `*.mlp.{gate_proj, up_proj}` (그룹 단위 smooth)
- `o_proj`, `down_proj`는 직전이 norm이 아니므로 표준 SmoothQuant에서 제외 (논문도 동일)

**Hook 기반 통계 수집**: `register_forward_pre_hook`으로 첫 linear의 입력 `x`를 받아 채널별 abs max 누적.

**Smooth factor 적용**:
- `s = (x_max.pow(alpha) / w_max.pow(1-alpha)).clamp(min=1e-5)` (float32 계산)
- `norm.weight.data /= s` (등가 변환)
- `linear.weight.data *= s.unsqueeze(0)` (in_features 차원 broadcast)

**수치 검증 (test_smoothquant.py:test_smooth_preserves_forward_output)**:
SmoothQuant 단독 적용 후 (양자화 없이) forward 출력이 원본과 1e-3 이내 일치.
등가 변환이므로 수학적으로 동일, 실제 차이는 float 누적 오차 수준.

### 24. Compressor.save() — modifier list에서 scheme 추출

기존: `Compressor`가 `scheme/ignore`를 직접 보유.
신규: modifier list에서 `QuantizationModifier` 인스턴스를 찾아 그 scheme/ignore 사용 (`_find_quantization_modifier`).

**이유**: modifier list가 사용자 입력의 단일 진실 출처(single source of truth)가 되어 Compressor 내부에 별도 scheme 상태를 두지 않음.

**에러 가드**: modifier list에 `QuantizationModifier`가 없으면 `save()` 호출 시 명시적 에러.

### 25. LayerNorm bias 흡수 누락 fix

**문제**: 초기 `SmoothQuantModifier.calibrate()`는 `norm.weight /= s`만 적용. LayerNorm은 affine 두 개(`weight=gamma`, `bias=beta`)를 갖는데 bias는 그대로 두면 등가 변환이 깨진다.

수식:
```
y = gamma * x_hat + beta
y_new = (gamma/s) * x_hat + (beta/s) = y / s    ← 둘 다 나눠야 등가
```

**Qwen3/LLaMA는 RMSNorm(bias 없음)이라 문제가 드러나지 않았다.** 초기 등가 검증 테스트(`test_smooth_preserves_forward_output`)도 `nn.LayerNorm` 기본 초기화(bias=0)를 써서 false-positive로 통과했다.

**Fix**: `getattr(norm, "bias", None) is not None`이면 `norm.bias /= s`도 적용.

**회귀 테스트 추가** (`test_smooth_preserves_forward_with_layernorm_bias`): LayerNorm bias를 비제로로 강제 초기화한 모델에서 등가 변환 유지 확인. 학습된 LayerNorm 모델(GPT-2, BERT, OPT 등) 시뮬레이션.

### 26. SmoothQuant PPL 측정 결과 및 calibration 샘플 효과 관찰

**측정 결과** (Qwen3-0.6B, wikitext-2, sliding window stride=512 max_len=2048):

| Scheme | demo (5 샘플) | milestone11 (128 샘플) |
|--------|--------------:|----------------------:|
| FP16 | 18.16 | 18.16 |
| W4A16 RTN | 25.89 | 25.89 |
| W8A8 static | 25.01 | 27.75 |
| W8A8 dynamic | 18.48 | 18.48 |
| W8A8 + SmoothQuant | **23.67** | (미측정) |

**관찰 1 — refactor 회귀 없음**: W4A16, FP16, dynamic 모두 기존 측정값과 정확히 일치. modifier composition 리팩토링이 RTN 동작을 깨뜨리지 않음.

**관찰 2 — SmoothQuant 효과**: 동일 calibration 조건(5 샘플)에서 W8A8 static 25.01 → SmoothQuant 23.67. **-1.34 개선**. per-tensor static의 본질적 한계 안에서 의미 있는 차이.

**관찰 3 — calibration 샘플 효과 (비직관적)**: 같은 W8A8 static인데 128 샘플(27.75) > 5 샘플(25.01). MinMax observer는 본질적으로 outlier에 민감 — 더 큰 데이터셋에서 outlier에 더 많이 노출되어 per-tensor scale을 과도하게 넓게 잡는다.
- **함의**: MinMax observer는 calibration sample을 늘려도 PPL이 개선되지 않을 수 있다. SmoothQuant이나 percentile/MSE observer가 본질 해결책.
- 발표 자료에서 SmoothQuant 동기를 설명할 때 이 관찰을 함께 제시 가능.

---

## 2026-05-17 — Recipe preset 레이어

### 27. from_recipe + RECIPE_REGISTRY — composition 위의 선언적 preset

**문제**: SmoothQuant 도입 후 RTN 경로는 `from_scheme("w8a8")` 한 줄 preset이 유지되지만, SmoothQuant는 `Compressor([SmoothQuantModifier(0.5), QuantizationModifier(W8A8)])` 조립형으로만 닿는다. Quark식 "이름만 알면 되는" 낮은 진입장벽이 SmoothQuant까지 확장되지 않은 갭.

**결정**: composition 패턴 위에 선언적 preset 레이어를 얹는다. preset(declarative)과 composition(조립)은 배타가 아니라 레이어 관계 — llm-compressor도 named recipe가 내부에서 modifier 리스트로 펼쳐진다.

| 레이어 | 진입점 | 다루는 것 |
|--------|--------|-----------|
| 선언 (declarative) | `from_recipe(name)` / `from_scheme(name)` | 이름 |
| 조립 (composition) | `Compressor([modifier, ...])` | modifier 리스트 |

**구현**:
- `recipes.py` 신규 — `RECIPE_REGISTRY: Dict[str, RecipeFactory]`. factory는 `(targets, ignore) → List[BaseModifier]`. modifier가 hook·통계 등 내부 상태를 가지므로 **호출마다 새 인스턴스 생성** (registry 값은 미리 만든 instance가 아니라 callable).
- `Compressor.from_recipe(name, targets, ignore)` — `from_scheme`과 나란히. targets/ignore는 recipe 내부 QuantizationModifier로 전달.
- `scheme`(수치 포맷, SCHEME_REGISTRY)과 `recipe`(알고리즘 파이프라인, RECIPE_REGISTRY)를 별도 레지스트리로 분리 — 책임 분리 유지.
- 현재 recipe: `w8a8_smoothquant` 1개. 알고리즘이 늘면 항목 추가.

**근거**: 면접 질문 "SmoothQuant 도입으로 Quark식 preset 장점이 사라졌나"의 답 — 사라진 게 아니라 확장 안 했던 갭이고 레지스트리 하나로 메워진다. composition을 택해도 declarative preset과 양립한다는 걸 코드로 입증.

### 28. from_scheme 제거 — preset 진입점을 from_recipe 하나로 통일

> #27은 `from_scheme`과 `from_recipe`를 나란히 뒀으나, 아래에서 진입점을 하나로 통일하며 `from_scheme`을 제거함. #27의 "별도 레지스트리로 분리" 부분은 이 결정으로 갱신됨.

**배경**: #27 직후, "값이 늘어날 때 scheme/recipe 두 진입점이 scalable한가" 검토.

**진단**: 중복은 "두 레지스트리"가 아니라 "두 진입점"에 있었다. `SCHEME_REGISTRY`가 (1) scheme 카탈로그 (2) 진입점 디스패치 두 역할을 겸했고, (2)가 `from_recipe`와 겹쳤다.

**결정**: 진입점을 `from_recipe` 하나로 통일, `from_scheme` 제거.
- `RECIPE_REGISTRY`에 단일 RTN(`w4a16`/`w8a8`/`w8a8_dynamic`)을 modifier 1개짜리 recipe로 추가 (`_rtn(scheme)` factory helper).
- `SCHEME_REGISTRY`(dict)와 `W8A8` 등 scheme 객체는 잔존 — `serialize.py`가 scheme→name 매칭(`serialize.py:80`)에 쓰는 데이터 카탈로그. 진입점 역할만 벗음.
- scheme = recipe 내부의 수치 포맷 building block, recipe = 유일한 사용자 진입점.

**근거**: 값이 늘어도 진입점은 영원히 하나 → API 표면적 불변, 이름 충돌 불가, "이건 scheme이냐 recipe냐" 판단을 사용자에게 안 떠넘김. llm-compressor도 recipe가 진입점, quantization config는 modifier 내부 부품인 구조.

**마이그레이션**: `demo.py`(4곳), `tests/test_compressor.py`(5곳 + `test_from_scheme_unknown_raises` 제거), README → `from_recipe`. 단위 테스트 30개 통과.

**미수정**: `notebooks/milestone8_round_trip.ipynb`·`milestone11_perplexity.ipynb`은 `from_scheme`을 쓰지만 과거 실행 기록(출력 보존)이라 손대지 않음 — 재실행 시점에 갱신.

---

## 2026-05-17 — M13 Multi-GPU observer sync

### 29. observer sync — MinMax는 all_reduce, 나머지 셋은 all_gather

**문제**: multi-GPU DataParallel calibration에서 각 rank가 자기 배치만 보므로 observer 통계가 rank별로 부분적. `compute_scale_zp` 전에 rank 간 동기화 필요.

**결정**: observer 종류에 따라 두 방식.
- `MinMaxObserver.sync()` — `all_reduce(min_val, MIN)` / `all_reduce(max_val, MAX)`. min·max는 결합법칙이 성립해 부분 결과를 정확히 병합. 통신량 O(1). road_map 원칙 2("MinMax 통계를 tensor 연산으로 유지")가 노린 결과.
- `Percentile/MSE/KL.sync()` — `all_gather_object`로 raw `_data`를 전 rank에 공유. 이 셋은 percentile·grid-search·histogram 등 비결합적 계산이라 부분 통계 병합 불가. (a) raw data 공유 vs (b) 알고리즘 재구조화 중, (b)는 동작 코드 3개를 갈아엎음 → (a) 채택. `compute_scale_zp` 무수정 + `sync()`만 추가. 한계: 메모리 rank배 → 대규모는 (b)가 future work.
- stub docstring의 "MSE: all_reduce(argmin)"은 부정확(rank별 argmin이 달라 병합 안 됨) — all_gather로 바로잡음.

**no-op 보장**: `_dist_active()`(`dist.is_initialized() and world_size>1`)가 아니면 `sync()`는 즉시 return. 단일 GPU 경로(기존 테스트·demo·PPL) 무변경.

**호출 지점**: `QuantizationModifier.calibrate()` — forward 루프 종료 후, `compute_scale_zp` 직전.

**검증**: 2-GPU 하드웨어 없음 → `gloo` 백엔드(CPU 분산)로 `torch.multiprocessing.spawn` 2-프로세스. `tests/test_observer_sync.py`: rank별 다른 데이터 → sync → 결과가 전체 데이터 단일 프로세스 결과와 일치 확인(4개 observer). 검증 불가 영역은 `device_map="auto"` 물리 cross-GPU 배치뿐.

### 30. device_map 감사 — input_scale을 weight.device로 보정

**13-2 감사 결과**: `weight_scale`은 weight에서 계산돼 device를 따라감. MinMax observer는 FakeQuantLinear 서브모듈이라 buffer도 따라감. 그러나 Percentile/MSE/KL은 `update()`에서 `_data`를 `.cpu()`로 모아 `input_scale`이 CPU에 남음 → GPU 모델 forward 시 device mismatch.

**fix**: `calibrate()`에서 `mod.input_scale = scale.to(mod.weight.device)` (zp 동일). 단일 CPU/GPU 모두 안전.

## 2026-05-17 — M6-D 멀티모델 검증

### 31. Llama-3.2-1B → TinyLlama-1.1B 대체

**문제**: M6-D 타깃 `meta-llama/Llama-3.2-1B`이 HF gated이고 토큰에 접근 권한 없음(403).

**결정**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`로 대체. `model_type=llama` — 같은 LLaMA 아키텍처(RMSNorm, GQA 32:4)라 "non-Qwen 아키텍처 검증" 목적은 동일하게 충족. 오히려 GQA 비율(8:1)이 Qwen3보다 커서 SmoothQuant GQA 경로를 더 강하게 실증.

**검증 결과** (라이브러리 코드 0줄 수정):
- `_find_smooth_pairs` 44개 페어 자동 탐색 (22층×2) — `input_layernorm`/`post_attention_layernorm` 이름 규칙이 LLaMA 공통이라 그대로 동작. attn 페어 dims `[(2048,2048),(2048,256),(2048,256)]` — GQA(q_proj out > k/v_proj out)가 페어에 그대로 반영.
- `targets`/`ignore` model-agnostic — `lm_head`만 제외, 154개 Linear 교체.
- 4개 recipe 모두 compress → generate 정상.

**산출물**: `demo.py`에 `--model` 인자 추가(end-to-end 데모를 모델 비종속화), `notebooks/milestone6d_llama_validation.ipynb`(실행 결과 포함). 라이브러리 코드 변경 없음 — 이게 M6-D의 핵심 증거.

### 작업 위치 (2026-05-17 갱신 — M6-D)

- main: `dc9582a` (M13 PR #4까지 머지 완료). working tree에 M6-D 작업 uncommitted.
  - uncommitted: `demo.py`(`--model` 인자), `notebooks/milestone6d_llama_validation.ipynb`(신규), `README`/`PROGRESS`/`road_map`/`context-notes`. 라이브러리 코드(`mini_compressor/`) 무변경.
- 단위 테스트 32개 통과 (라이브러리 무변경이라 회귀 없음).

### 다음 작업

- git: M6-D 작업을 `feature/llama-validation` 브랜치로 커밋 → PR → main 머지.
- 코드 후보: M6-B GPTQ 실제 구현.
