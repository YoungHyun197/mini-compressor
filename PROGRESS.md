# mini-compressor 구현 진행 상황

## 현재 날짜: 2026-05-10

---

## 프로젝트 구조 (완료)

- [x] `mini_compressor/__init__.py`
- [x] `mini_compressor/schemes.py` — QuantizationSpec, QuantizationScheme, W8A8, W4A16
- [x] `mini_compressor/fake_quant_linear.py` — FakeQuantLinear (flat buffer 방식)
- [x] `mini_compressor/modifier.py` — stub (Milestone 7에서 구현 예정)
- [x] `mini_compressor/compressor.py` — stub (Milestone 8에서 구현 예정)
- [x] `mini_compressor/serialize.py` — stub (Milestone 9에서 구현 예정)
- [x] `tests/test_fake_quant_linear.py` — 3개 기본 테스트
- [x] `notebooks/` 디렉토리

---

## Milestone 진행 상황

### Milestone 1 — Qwen3-0.6B 모델 로드
```
Qwen/Qwen3-0.6B load
tokenizer load
FP model.generate() 확인
```
- [x] `notebooks/` 또는 스크립트에서 Qwen3-0.6B 로드 확인
- [x] FP16 model.generate() 동작 확인

---

### Milestone 2 — Linear 목록 출력 + ignore 확인
```
모든 nn.Linear 이름 출력
ignore=["lm_head"] 적용 확인
```
- [x] `named_modules()` 순회로 Linear 이름 출력
- [x] `lm_head` 제외 로직 확인

---

### Milestone 3 — FakeQuantizer 구현
```
FakeQuantizer 구현
per-tensor symmetric weight fake quant
```
- [x] `FakeQuantLinear._fake_quantize_weight()` 구현 (per_tensor, per_channel, per_group)
- [x] `FakeQuantLinear._fake_quantize_activation()` 구현

---

### Milestone 4 — FakeQuantLinear 구현 + generate 확인
```
FakeQuantLinear 구현
기존 nn.Linear weight를 복사
forward에서 fake quant weight 사용
generate 확인
```
- [x] `schemes.py` 구현 (QuantizationSpec, QuantizationScheme, W8A8, W4A16)
- [x] `FakeQuantLinear.from_float()` 구현
- [x] `FakeQuantLinear.forward()` 구현
- [x] `tests/test_fake_quant_linear.py` 3개 테스트 통과 확인
- [x] Qwen3에 Linear replacement 적용 후 model.generate() 확인

---

### Milestone 5 — W8A8 + W4A16 RTN (observer + calibration + per-group)
```
W8A8 / W4A16 추가
activation observer 추가 (minmax / percentile / mse / kl_divergence 선택 가능)
calibration forward pass
activation scale 고정
per-group weight fake quant (W4A16)
generate 확인
```
> **M6 W4A16 RTN 통합**: per-group fake quant가 M5 구현 시점에 이미 완성되어 M5로 병합.
> M6은 SmoothQuant / GPTQ 확장 milestone으로 재정의.

- [x] `mini_compressor/observer.py` 구현
  - [x] `BaseObserver` — 공통 인터페이스 (update / compute_scale_zp)
  - [x] `MinMaxObserver` — min/max 수집 + zero 포함 보장 (기본값)
  - [x] `PercentileObserver` — percentile 클리핑 (기본값 99.9th)
  - [x] `MSEObserver` — grid-search로 MSE 최소 scale 탐색
  - [x] `KLDivergenceObserver` — histogram 기반 KL divergence 최소화
- [x] `QuantizationSpec`에 `calibration_method: str = "minmax"` 필드 추가
- [x] `FakeQuantLinear`가 `calibration_method`에 따라 observer 인스턴스화
- [x] `QuantizationModifier` 구현 (M5+M7 병합)
  - [x] `initialize()` — nn.Linear → FakeQuantLinear 교체 + weight scale RTN 계산 (per_channel / per_group / per_tensor)
  - [x] `calibrate()` — observer forward pass로 scale/zp 계산, activation=None 시 early return
  - [x] `finalize()` — observer 제거, scale buffer만 남김
- [x] `input_scale`, `input_zero_point` buffer 채우기
- [x] `_group_fake_quant()` — W4A16 per-group weight fake quant
- [x] W4A16 weight_scale shape `[out_features, in_features // group_size]` 검증
- [x] `tests/test_modifier.py` 작성 (6개 케이스 통과)
- [x] W8A8 scheme으로 end-to-end generate 확인 (notebooks/milestone5_w8a8_e2e.ipynb)

---

### Milestone 6 — 고급 압축 기법 확장 (시간 허락 시)
> W4A16 RTN은 M5에 통합 완료. M6은 SmoothQuant / GPTQ 확장만 담당.

### Milestone 6-A — SmoothQuant (시간 허락 시)
```
modifier.py에 smooth() 단계 추가
per-channel scaling factor s 계산 (α=0.5)
FakeQuantLinear.weight에 s 흡수
W8A8 + SmoothQuant generate 확인
RTN W8A8 대비 perplexity 비교
```
변경 파일: `modifier.py` 만

- [ ] `modifier.py`에 `smooth()` 메서드 구현
  - [ ] calibration forward로 activation max 통계 수집
  - [ ] `s = max(|X|)^α / max(|W|)^(1-α)` 계산 (α=0.5)
  - [ ] `FakeQuantLinear.weight *= diag(s)` 흡수
- [ ] W8A8 + SmoothQuant generate 확인
- [ ] lm-eval perplexity 비교 (RTN vs SmoothQuant)

---

### Milestone 6-B — GPTQ (시간 허락 시)
```
modifier.py calibrate()에 GPTQ 분기 추가
layer별 Hessian + column-wise weight 최적화
W4A16 + GPTQ generate 확인
RTN W4A16 대비 perplexity 비교
```
변경 파일: `modifier.py` 만

- [ ] `modifier.py`에 GPTQ 분기 구현
  - [ ] layer별 Hessian 계산 (calibration data)
  - [ ] weight column 순서대로 양자화 + 오차 보상
  - [ ] 결과 weight를 `FakeQuantLinear.weight`에 저장
- [ ] W4A16 + GPTQ generate 확인
- [ ] lm-eval perplexity 비교 (RTN vs GPTQ)

---

### Milestone 6-C — Sequential Calibration (시간 허락 시)
```
modifier.py calibrate()에 sequential=True 모드 추가
layer 단위 GPU offload calibration
대형 모델에서 전체 forward 불가 시 사용
```
변경 파일: `modifier.py` 만

- [ ] `calibrate(sequential=False)` 기본 인터페이스 확정
- [ ] `_calibrate_sequential()` 구현
  - [ ] embedding 출력까지 CPU 캐시 수집
  - [ ] layer별 GPU 이동 → calibration → CPU offload 루프
  - [ ] 각 layer의 scale buffer 채우기
- [ ] Qwen3-0.6B에서 sequential 모드 동작 확인

---

### Milestone 6-D — 멀티모델 검증 (시간 허락 시)
```
LLaMA 계열에서 W8A8 / W4A16 동작 확인
targets/ignore 패턴 model-agnostic 검증
```

- [ ] `meta-llama/Llama-3.2-1B` W8A8 RTN generate 확인
- [ ] `meta-llama/Llama-3.2-1B` W4A16 RTN generate 확인
- [ ] targets 패턴이 Qwen3/LLaMA 공통으로 동작함을 확인
- [ ] SmoothQuant norm layer 탐색 (`smooth_norm_pattern` config) 적용

---

### Milestone 7 — QuantizationModifier 리팩토링
> **M5와 병합 완료.** initialize / calibrate / finalize 모두 M5에서 구현됨. 별도 진행 없음.

- [x] `initialize()` — nn.Linear → FakeQuantLinear 교체 (module replacement)
- [x] `calibrate()` — calibration dataloader forward, observer scale 계산
- [x] `finalize()` — observer 제거, scale buffer만 남김

---

### Milestone 8 — HF 호환 save/load + Compressor API
```
serialize.py 구현 → compressor.py 구현 순서
compressed-tensors 포맷 호환 quantization_config.json
Compressor.from_scheme("w8a8").compress(model, dataloader) 원클릭 API
```

#### 8-1. modifier.py 수정
- [x] `initialize(compute_scales: bool = True)` 파라미터 추가
  - True (기본): 기존 동작 그대로 (weight scale RTN 계산)
  - False: FakeQuantLinear 구조만 생성, scale 계산 생략 (load 흐름 전용)

#### 8-2. serialize.py 구현
- [x] `_scheme_to_dict(scheme)` — QuantizationScheme → compressed-tensors 포맷 dict
- [x] `_scheme_from_dict(d)` — dict → QuantizationScheme 복원
- [x] `save_pretrained(model, save_dir, scheme, ignore=None, tokenizer=None)`
  - [x] `model.save_pretrained(save_dir)` — safetensors + config.json 자동 생성
  - [x] `quantization_config.json` 별도 저장 (compressed-tensors 포맷)
  - [x] tokenizer 있으면 `tokenizer.save_pretrained(save_dir)`
- [x] `load_pretrained(save_dir)`
  - [x] `quantization_config.json` → scheme, ignore 복원
  - [x] `AutoModelForCausalLM.from_pretrained(model_id)` — inv_freq dtype 보존용
  - [x] `modifier.initialize(compute_scales=False)` — 구조만 생성
  - [x] `input_observer = None` — finalize()와 동일 상태로 맞춤 (W8A8 observer 재활성화 방지)
  - [x] 직접 state 주입 루프 — load_state_dict 미사용 (None buffer skip 문제 우회)
- [x] `tests/test_serialize.py` 6개 테스트 통과

#### 8-3. compressor.py 구현
- [x] `Compressor.from_scheme(scheme_name, ignore=None)` — SCHEME_REGISTRY 조회
- [x] `Compressor.compress(model, dataloader, num_samples=None)`
  - [x] `modifier.initialize()` → `modifier.calibrate()` → `modifier.finalize()` 순서 호출
  - [x] model 반환 (in-place 수정이지만 체이닝 가능하도록)
- [x] `Compressor.save(model, save_dir, tokenizer=None)`
- [x] `tests/test_compressor.py` 5개 테스트 통과

#### 8-4. 검증
- [x] scheme dict round-trip (W8A8, W4A16)
- [x] `quantization_config.json` 내용 확인 (calibrated 상태, null input_activations 등)
- [x] save → safetensors 파일 생성 확인
- [x] HF 모델(Qwen3-0.6B): `compress → save → load → generate` 왕복 일치 확인
  - W4A16 round-trip: True
  - W8A8 round-trip: True
- [x] CI 통과 (`transformers`, `safetensors` pyproject.toml dependencies 추가)

---

### Milestone 9 — README
```
README 작성 (설치법, 실행법, 지원 scheme, 설계 설명, limitation)
```
> save_pretrained / quantization_config.json은 M8에서 완료됨.

- [ ] `README.md` 작성 — 설계 철학 + 사용법 + limitation

---

### Milestone 10 — 발표자료 + trade-off 정리
```
발표자료 작성
trade-off 정리
known limitation 정리
```
- [ ] 설계 철학 슬라이드
- [ ] 핵심 3문답 답변 준비 (추상화 단위 / Config 표현 / scheme 확장 범위)
- [ ] 각 결정의 근거 + trade-off
- [ ] known limitation (fake quant vs real, packing 없음 등)

---

### Milestone 11 — lm-eval 벤치마크
```
lm-eval-harness 연동
FP baseline 측정 (wikitext-2 perplexity / lambada accuracy)
W8A8 / W4A16 측정 + 비교 표 → README 반영
```
- [ ] `lm-eval` 설치 및 연동 확인
- [ ] FP Qwen3-0.6B perplexity 측정 (baseline)
- [ ] W8A8 RTN perplexity 측정
- [ ] W4A16 RTN perplexity 측정
- [ ] W8A8 SmoothQuant perplexity 측정 (Milestone 6-A 완료 시)
- [ ] W4A16 GPTQ perplexity 측정 (Milestone 6-B 완료 시)
- [ ] 비교 표 README에 추가

---

### Milestone 12 — End-to-End Demo 노트북
```
notebooks/demo.ipynb 작성
Compressor.from_scheme() → generate → save → lm-eval 전체 flow 재현
노트북 한 파일에서 발표 demo 가능한 상태로 완성
```
- [ ] `notebooks/demo.ipynb` 작성
- [ ] W8A8: compress → generate → save → reload 전체 flow 확인
- [ ] W4A16: compress → generate 확인
- [ ] lm-eval 수치 노트북 내 출력
- [ ] quantization_config.json 확인 셀 포함

---

## Git + CI/CD 상태

### Git 초기화
- [x] GitHub 저장소 생성 (public) — https://github.com/YoungHyun197/mini-compressor
- [x] `git init` + `git remote add origin`
- [x] `.gitignore` 설정
- [x] 첫 커밋 + push — `chore: initial project scaffold`

### CI/CD 적용 현황
> CI 항목은 각 Milestone 완료 시점에 필요성을 논의 후 채택한다. 임의로 추가하지 않는다.

| Milestone | CI 논의 시점 | 채택 여부 | 내용 |
|-----------|------------|---------|------|
| Milestone 4 완료 후 | 2026-05-10 | ✅ 채택 | pytest unit test — `.github/workflows/ci.yml` |
| Milestone 7 완료 후 | 2026-05-11 | ✅ 채택 | test_modifier.py — 6개 케이스, pytest tests/에 자동 포함 |
| Milestone 9 완료 후 | - | 미정 | save/load round-trip 테스트 |

---

## Known Limitations (미지원 기능)

| 기능 | 상태 | 비고 |
|------|------|------|
| per-token dynamic quantization | **구현 완료** | `W8A8_DYNAMIC` preset — `granularity="per_token"` + `dynamic=True`. 런타임에 토큰별 scale 계산, calibration 불필요 |
| real INT 패킹 (`quantization_status: "compressed"`) | 미지원 | 현재는 fake quant 단계 — weight는 float16 저장. 실제 INT4/INT8 패킹은 컴파일러 단 담당 |
| SmoothQuant | stub만 존재 | `QuantizationModifier.smooth()` — NotImplementedError |
| GPTQ | 미구현 | stub도 없음 |
| Sequential calibration | stub만 존재 | `calibrate(sequential=True)` — NotImplementedError |
| Multi-model 검증 | 미진행 | Qwen3-0.6B 외 LLaMA 등 미확인 |

---

## 현재 작업 위치

> **Milestone 1–8 + README 완료**

**다음 할 일.**
1. Milestone 6-A: SmoothQuant 구현
2. Milestone 11: lm-eval perplexity 측정

---

## 설계 철학 요약

| 레이어 | 출처 | 설명 |
|--------|------|------|
| 사용자 API | HF / llm-compressor | `save_pretrained`, `quantization_config.json` |
| 내부 동작 | AMD Quark | module replacement, initialize→calibrate→finalize |
| finalize 방식 | llm-compressor | observer 제거, scale buffer만 남김 |
| weight 포맷 | Furiosa 정렬 | fake quant 상태 float16 저장, real packing은 compiler가 담당 |

---

## 핵심 구현 결정사항

1. **FakeQuantLinear**: `nn.Linear` 교체, flat buffer (`weight_scale`, `input_scale`) 직속 보유
2. **Config**: `QuantizationSpec` + `QuantizationScheme` 2-tier
3. **Serialization**:
   - `quantization_config.json`: HF compressed-tensors 스펙
   - state_dict key: `weight_scale` (NOT `_weight_quantizer.scale`)
   - weight key: `weight` 유지 (float16, qweight 아님)
4. **멀티모델**: fnmatch 패턴 기반 target/ignore, 모델 아키텍처 비종속
5. **Sequential calibration**: `calibrate(sequential=True)` 플래그, `modifier.py` 내부 분기

---

## Multi-GPU 가산점 요소

> **구현 전 반드시 별도 논의 후 결정**. 아래는 설계 고려사항만 기록.

| 고려사항 | 내용 |
|---------|------|
| DataParallel calibration | Observer 통계 all-reduce 동기화 필요 |
| Tensor Parallelism | shard된 weight의 scale 계산/merge 필요 |
| Sequential + Multi-GPU | device placement 로직 추가 |
| state_dict 저장 | rank 0 저장 또는 shard별 저장 방식 선택 |

영향 파일: `modifier.py`, `compressor.py`, `serialize.py`  
`FakeQuantLinear`, `schemes.py` 변경 없음
