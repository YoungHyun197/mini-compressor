# mini-compressor context notes

설계 결정과 그 근거를 세션 간 이어받기 위한 기록. 코드 외부에서 이유를 추적할 수 없는 결정만 기록한다.

---

## 파일 구조

```
mini_compressor/
  schemes.py           — QuantizationSpec, QuantizationScheme, W8A8, W4A16, SCHEME_REGISTRY
  observer.py          — BaseObserver, MinMaxObserver, PercentileObserver, MSEObserver, KLDivergenceObserver
  fake_quant_linear.py — FakeQuantLinear (nn.Linear 교체 대상)
  modifier.py          — QuantizationModifier (initialize / calibrate / finalize)
  compressor.py        — stub (Milestone 8)
  serialize.py         — stub (Milestone 9)
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
| M5+M7 | observer.py, modifier.py | 완료 (20개 테스트 통과) |
| M8 | modifier.py, serialize.py, compressor.py, tests/test_serialize.py, tests/test_compressor.py | 완료 (20개 테스트 통과) |

---

### 8. M6 W4A16 RTN → M5 통합

원래 M6에서 구현 예정이던 W4A16 RTN per-group fake quant가 M5 구현 시점에 이미 완성됨.

- `_group_fake_quant()`: fake_quant_linear.py에 구현
- `_compute_weight_scale()` per_group 분기: modifier.py에 구현
- `test_initialize_w4a16_scale_shape`: scale shape 검증 통과

M6은 SmoothQuant(6-A) / GPTQ(6-B) 확장 milestone으로 재정의. RTN 관련 항목은 M5 완료로 처리.

---

## 미완료 / 다음 할 일

- [ ] M6-A: SmoothQuant (시간 허락 시)
- [ ] M6-B: GPTQ (시간 허락 시)
- [ ] M8: modifier.py 수정 + serialize.py + compressor.py

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
