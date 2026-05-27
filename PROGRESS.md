# mini-compressor 구현 진행 상황

## 현재 날짜: 2026-05-23

---

## 프로젝트 구조 (완료)

- [x] `mini_compressor/schemes.py` — QuantizationSpec, QuantizationScheme, SCHEME_REGISTRY
- [x] `mini_compressor/fake_quant_linear.py` — FakeQuantLinear (flat buffer 방식)
- [x] `mini_compressor/observer.py` — 4종 observer (weight·activation 공용, granularity-aware) + multi-GPU sync
- [x] `mini_compressor/modifiers/` — BaseModifier + 알고리즘별 클래스 (composition pattern)
- [x] `mini_compressor/recipes.py` — RECIPE_REGISTRY (preset 진입점)
- [x] `mini_compressor/compressor.py` — Compressor (modifier list 진입점)
- [x] `mini_compressor/serialize.py` — save_pretrained / load_pretrained
- [x] `tests/` — 9개 파일, 49개 케이스
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
activation observer 추가 (minmax / percentile / mse 선택 가능)
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
- [x] `QuantizationSpec`에 `calibration_method: str = "minmax"` 필드 추가
- [x] `FakeQuantLinear`가 `calibration_method`에 따라 observer 인스턴스화
- [x] `QuantizationModifier` 구현 (M5+M7 병합)
  - [x] `initialize()` — nn.Linear → FakeQuantLinear 교체 + weight observer로 weight scale 산출 (per_channel / per_group / per_tensor)
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

### Milestone 6-A — SmoothQuant ✅ 완료
```
modifier composition 리팩토링 + SmoothQuantModifier 실구현
norm → linear group 자동 탐색
per-channel scaling factor s 계산 (α=0.5)
norm.weight /= s, linear.weight *= s 적용
Compressor([SmoothQuantModifier, QuantizationModifier(W8A8)]) chain 동작
```
변경 파일: `mini_compressor/modifiers/` (신규 디렉토리), `compressor.py`, `serialize.py`, tests, `demo.py`

- [x] `modifier.py` → `modifiers/` 디렉토리 분리 (BaseModifier + 알고리즘별 클래스)
- [x] `SmoothQuantModifier` 실구현 (`modifiers/smoothquant.py`)
  - [x] norm-linear pair 자동 탐색 (`input_layernorm`→q/k/v_proj, `post_attention_layernorm`→gate/up_proj)
  - [x] forward pre-hook으로 channel-wise activation abs max 수집
  - [x] `s = max(|X|)^α / max(|W|)^(1-α)` 계산 (α=0.5 기본)
  - [x] `norm.weight /= s`, `linear.weight *= s` 적용
- [x] `Compressor` API 갱신 — modifier list 수용 (`Compressor([SmoothQuantModifier, QuantizationModifier(W8A8)])`)
- [x] backward compatibility 유지 (이후 6-A 후속에서 `from_recipe`로 통일하며 `from_scheme` 제거)
- [x] 동등성 단위 테스트 3개 추가 (`tests/test_smoothquant.py`)
- [x] `demo.py`에 W8A8+SmoothQuant 옵션 추가
- [x] W8A8 + SmoothQuant Qwen3-0.6B PPL 측정 (`python eval.py --ppl`)
  - W8A8 static 25.01 → W8A8 + SmoothQuant (dynamic) 21.08 (-3.93 개선)
  - refactor 회귀 확인 — W4A16/FP16/dynamic 기존 측정값과 일치

### Milestone 6-A 후속 — Recipe preset 레이어 ✅ 완료

```
composition 패턴 위에 Quark식 선언적 preset 레이어 추가
모든 preset을 단일 진입점 from_recipe + RECIPE_REGISTRY로 통일 (from_scheme 제거)
```
변경 파일: `mini_compressor/recipes.py` (신규), `compressor.py`, `__init__.py`, `tests/test_compressor.py`, `demo.py`

- [x] `recipes.py` 신규 — `RECIPE_REGISTRY` (이름 → modifier 리스트 factory)
- [x] `Compressor.from_recipe(name, targets, ignore)` — 유일한 preset 진입점
- [x] 단일 RTN(`w4a16`/`w8a8`/`w8a8_dynamic`)도 modifier 1개짜리 recipe로 흡수
- [x] `w8a8_smoothquant` recipe — `[SmoothQuantModifier(0.5), QuantizationModifier(W8A8)]`
- [x] `Compressor.from_recipe` 제거 — 진입점 중복 해소 (`SCHEME_REGISTRY`는 `serialize.py`용 카탈로그로 잔존)
- [x] `__init__.py` export 갱신, `demo.py` / `tests` `from_recipe` 마이그레이션 (단위 테스트 30개 통과)

---

### Milestone 6-B — GPTQ + AWQ ✅ 완료
```
modifiers/gptq.py GPTQModifier 실구현
modifiers/awq.py AWQModifier 실구현 (activation magnitude grid-search)
layer별 Hessian + column-wise weight 최적화 (GPTQ)
activation mean 기반 per-channel scale + grid search alpha (AWQ)
unit test 7개 (AWQ) + 4개 (GPTQ) 통과
```
변경 파일: `modifiers/_pair_utils.py` (신규), `modifiers/gptq.py` (stub → 실구현),
`modifiers/awq.py` (stub → 실구현), `modifiers/smoothquant.py` (pair_utils 분리),
`modifiers/quantization.py` (QuantizationMixin 분리), `modifiers/__init__.py`, `compressor.py`, `recipes.py`

- [x] `QuantizationMixin` 추출 (`modifiers/quantization.py`) — llm-compressor 패턴
  - `initialize` / `_should_replace` / `finalize` 공통 mixin으로 분리
  - `GPTQModifier`와 `QuantizationModifier`가 mixin 상속으로 module replacement 공유
- [x] `GPTQModifier(QuantizationMixin, BaseModifier)` 실구현 (`modifiers/gptq.py`)
  - forward pre-hook으로 layer별 Hessian `H = 2·XᵀX` 누적
  - dead column 처리 + dampening + Cholesky inverse (`H → H⁻¹ → Hinv_upper`)
  - per-group block-column loop: absmax scale → column 양자화 → intra/inter-group 오차 전파
  - 결과 `Q`(dequantized float)를 `weight.data`에, `scale`을 `weight_scale`에 저장
- [x] `w4a16_gptq` recipe 추가 (`recipes.py`)
- [x] `Compressor._find_quantization_modifier` → `isinstance(m, QuantizationMixin)` — GPTQ save 지원
- [x] `tests/test_gptq.py` 4개 테스트 통과
  - replaces_linear, scale_shape, weight_on_grid, mse_leq_rtn
- [x] W4A16 GPTQ Qwen3-0.6B wikitext-2 perplexity 측정 — 20.96 (RTN 25.89 대비 -4.93)

**AWQ 추가 구현**:
- [x] `modifiers/_pair_utils.py` 신규 — `_find_smooth_pairs`, `_collect_linears`, `_has_affine_weight` 공통 유틸 (SQ+AWQ 공유)
- [x] `modifiers/smoothquant.py` 수정 — `_pair_utils` import로 교체
- [x] `AWQModifier` 실구현 (`modifiers/awq.py`)
  - forward pre-hook으로 channel-wise activation mean 누적 (`s_x = mean(|X|)`)
  - grid search: alpha ∈ linspace(0,1,n_grid+1)[1:] → `s = (s_x / mean(s_x))^alpha`
  - INT4 per-group fake quant (`_int4_fake_quant`) + activation-weighted error 최소화 (`_quant_error`)
  - best_s로 `norm.weight /= s`, `linear.weight *= s` 적용
- [x] `w4a16_awq` recipe 추가 (`recipes.py`)
- [x] `tests/test_awq.py` 7개 테스트 통과
  - preserves_forward_output, preserves_forward_with_layernorm_bias, initialize_required, empty_dataloader, no_pairs_model, compressor_chain, int4_fake_quant_roundtrip

---

### Milestone 6-C — Sequential Calibration ✅ 완료
```
QuantizationModifier.calibrate()에 sequential=True 모드 추가
layer 단위 GPU offload calibration
대형 모델에서 전체 forward 불가 시 사용
```
변경 파일: `modifiers/quantization.py`, `tests/test_sequential_calib.py` (신규)

- [x] `calibrate(sequential=False)` 기본 인터페이스 확정 (파라미터 추가 + stub)
- [x] `_calibrate_sequential()` 구현
  - [x] layers[0].forward 임시 대체로 embedding 출력(hidden_states + kwargs) CPU 캐시 수집
  - [x] layer별 compute_device 이동 → forward (observer 수집) → scale 확정 → CPU offload 루프
  - [x] 각 layer의 scale buffer 채우기 + input_observer 즉시 해제
- [x] 단위 테스트 3개 통과 (scale 동등성 · 미지원 구조 에러 · 빈 dataloader 에러)
- [x] Qwen3-0.6B 구조 지원 확인 (model.model.layers 패턴 — 아키텍처 비종속)

---

### Milestone 6-D — 멀티모델 검증 ✅ 완료 (TinyLlama-1.1B)
```
LLaMA 계열에서 recipe 동작 확인
targets/ignore 패턴 model-agnostic 검증
```
> `meta-llama/Llama-3.2-1B`은 HF gated(접근 미승인) → 동일 아키텍처(`model_type=llama`)인
> `TinyLlama/TinyLlama-1.1B-Chat-v1.0`로 대체. 검증 가치 동일.

- [x] TinyLlama-1.1B에서 `w4a16`/`w8a8`/`w8a8_dynamic`/`w8a8_smoothquant` 4종 compress → generate 확인
- [x] `targets`/`ignore` 패턴이 Qwen3/LLaMA 공통 동작 — `lm_head` 제외, 154개 Linear 교체
- [x] SmoothQuant `_find_smooth_pairs`가 LLaMA에서 44개 페어 자동 탐색 (RMSNorm, GQA 32:4)
- [x] `eval.py`에 `--model` 인자 추가 — end-to-end 평가 데모 model-agnostic
- [x] `notebooks/milestone6d_llama_validation.ipynb` 검증 노트북 (실행 결과 포함)

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
Compressor.from_recipe("w8a8").compress(model, dataloader) 원클릭 API
```

#### 8-1. modifier.py 수정
- [x] `initialize(compute_scales: bool = True)` 파라미터 추가
  - True (기본): 기존 동작 그대로 (weight observer로 weight scale 산출)
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
  - [x] `base_model_name_or_path`를 `quantization_config.json`에 저장 — `save_pretrained` 후 `config.json`의 `_name_or_path`가 save_dir로 덮어써져 UNEXPECTED key 경고가 뜨는 문제 수정
- [x] `tests/test_serialize.py` 6개 테스트 통과

#### 8-3. compressor.py 구현
- [x] `Compressor.from_recipe(scheme_name, targets=None, ignore=None)` — SCHEME_REGISTRY 조회, targets 파라미터 노출
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

#### 8-5. per-token dynamic quantization 추가
- [x] `QuantizationSpec.granularity`에 `"per_token"` 추가
- [x] `W8A8_DYNAMIC` preset — `per_channel` weight + `per_token dynamic` activation
- [x] `FakeQuantLinear`: dynamic=True 시 observer 미생성, 런타임 scale 계산 분기
- [x] `modifier.calibrate()`: dynamic scheme early return
- [x] `serialize.py`: `"per_token" → "token"` granularity 매핑 추가
- [x] `tests/test_fake_quant_linear.py` 3개 테스트 추가 (observer 미생성, forward, per-token 독립성)
- [x] `tests/test_compressor.py` 1개 테스트 추가 (dataloader 없이 compress)

#### 8-6. stub 명세
- [x] `QuantizationModifier.smooth()` — SmoothQuant stub (NotImplementedError + docstring)
- [x] `calibrate(sequential=True)` — Sequential calibration stub (NotImplementedError + docstring)

---

### Milestone 9 — README
```
README 작성 (설치법, 실행법, 지원 scheme, 설계 설명, limitation)
```
> save_pretrained / quantization_config.json은 M8에서 완료됨.

- [x] `README.md` 작성 — 설계 철학 + 사용법 + limitation

---

### Milestone 10 — 발표자료 + trade-off 정리
```
발표자료 작성
trade-off 정리
known limitation 정리
```
> 산출물. `presentation/slides.md` (13슬라이드 + Q&A backup), `presentation/script.md` (30분 발표 스크립트 + Q&A 풀텍스트).

- [x] 설계 철학 슬라이드 (Slide 2 "Why this design?", Slide 3 OSS 차용 매핑)
- [x] 핵심 3문답 답변 준비 (Slide 4-6: 추상화 단위 / Config 표현 / scheme 확장 범위)
- [x] 각 결정의 근거 + trade-off (Slide 9 책임 분리, Slide 12 trade-off 표)
- [x] known limitation (Slide 12 — fake quant 한계, packing 미지원, multi-model 미검증)

---

### Milestone 11 — lm-eval 벤치마크
```
lm-eval-harness 연동
FP baseline 측정 (wikitext-2 perplexity / lambada accuracy)
W8A8 / W4A16 측정 + 비교 표 → README 반영
```
- [x] lm-eval 설치 (`notebooks/milestone11_perplexity.ipynb` 작성 완료)
- [x] 측정 방식: sliding window perplexity (HF 공식 방식, wikitext-2-raw-v1 test split)
- [x] FP16 baseline perplexity 측정 — 18.16
- [x] W4A16 RTN perplexity 측정 — 25.89 (+7.73)
- [x] W8A8 dynamic perplexity 측정 — 18.48 (+0.32)
- [x] W8A8 static perplexity 측정 — 29.71 (+11.55) _(calibration: wikitext-2 128 samples. MinMax outlier 민감도로 dynamic 대비 큰 열화)_
- [x] W8A8 + SmoothQuant perplexity 측정 — 18.33 (+0.17) _(calibration: wikitext-2 128 samples. 전체 scheme 중 최고 성능)_
- [x] W4A16 GPTQ perplexity 측정 — 20.96 (+2.80) ← RTN 25.89 대비 -4.93 개선
- [x] 비교 표 README에 추가

---

### Milestone 13 — Multi-GPU 지원
```
DDP calibration: Observer 통계 all-reduce 동기화
device_map="auto" 호환 검증 (pipeline parallel)
rank 0 저장 가드 (serialize.py) — 이미 적용됨
검증 코드 작성
```

#### 13-1. DataParallel calibration ✅ 완료
- [x] rank 0 저장 가드 (`serialize.py:103`) — 이미 구현됨
- [x] `MinMaxObserver.sync()` — `all_reduce(MIN/MAX)` (min·max는 결합적 → 한 줄로 정확 병합)
- [x] `PercentileObserver.sync()` — range all_reduce → histogram resize → all_reduce(SUM). GPU histogram 기반, NCCL 전용
- [x] `MSEObserver.sync()` — range all_reduce → histogram resize → all_reduce(SUM). GPU histogram 기반, NCCL 전용
      (histogram 기반으로 raw data 교환 없이 NCCL 통신만 사용)
- [x] `QuantizationModifier.calibrate()`에서 `compute_scale_zp` 직전 `sync()` 호출

#### 13-2. device_map="auto" 호환 ✅ 코드 감사 완료
- [x] scale이 `weight.device`를 따라가는지 감사 — `weight_scale`은 weight에서 계산돼 OK,
      `input_scale` device 갭은 histogram 리팩토링으로 근본 해소 (GPU 상주). `calibrate()`의 `.to(mod.weight.device)` 보정은 호환성 유지
- [ ] 실제 2-GPU `device_map="auto"` round-trip 실측 — 2-GPU 하드웨어 없음 (한계)

#### 13-3. Tensor Parallelism (범위 외 — 별도 논의)
- [ ] shard된 weight에서 scale 계산/merge 방식 결정
- [ ] 구현 여부는 진행 상황에 따라 판단

#### 13-4. 검증 코드 ✅ 완료
- [x] `tests/test_observer_sync.py` — gloo 2-프로세스로 4개 observer sync 실검증
      (분산 결과 == 전체 데이터 단일 프로세스 결과)
- [x] sync() no-op 검증 (분산 init 없이 호출 시 통계 불변)
- [ ] 2-GPU `device_map="auto"` round-trip — 하드웨어 한계로 미실행

---

### Milestone 12 — End-to-End Demo
```
demo_quick.py / eval.py 작성
세 scheme(W4A16 / W8A8 / W8A8-dynamic) compress → generate → save → load 전체 flow
```
- [x] `demo_quick.py` 작성 — 빠른 fake-quant generate smoke demo
- [x] `eval.py` 작성
  - [x] W4A16: compress → generate 확인
  - [x] W8A8 static: compress (calibration wikitext-2 128 samples) → generate 확인
  - [x] W8A8 dynamic: compress → generate 확인
  - [x] `--save DIR`: W4A16 save → load_pretrained → generate round-trip 확인
  - [x] `--ppl`: wikitext-2 sliding window perplexity 측정
  - [x] 세 옵션 동시 실행 가능 (`--save DIR --ppl`)

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
| **SmoothQuant** | **구현 완료** | `SmoothQuantModifier` — `Compressor.from_recipe("w8a8_smoothquant")` 또는 modifier list 직접 구성. activation 분포 평탄화 |
| real INT 패킹 (`quantization_status: "compressed"`) | 미지원 | 현재는 fake quant 단계 — weight는 float16 저장. 실제 INT4/INT8 패킹은 컴파일러 단 담당 |
| GPTQ | **구현 완료** | `GPTQModifier` — Hessian 기반 W4A16 weight 최적화. `Compressor.from_recipe("w4a16_gptq")` |
| AWQ | **구현 완료** | `AWQModifier` — activation magnitude grid-search W4A16, `from_recipe("w4a16_awq")` |
| Sequential calibration | **구현 완료** | `calibrate(sequential=True)` — layer별 GPU offload, model.model.layers 구조 지원 |
| Float8 | stub 있음 | `_fake_quantize_weight/activation()` dtype=="float8" 분기 — NotImplementedError |
| Multi-GPU Observer 동기화 | **구현 완료** | MinMax: all_reduce(MIN/MAX). Percentile/MSE: histogram 기반 range all_reduce → resize → all_reduce(SUM). 모두 NCCL. gloo 2-proc 검증. |
| HuggingFace Hub 업로드 | **구현 완료** | `Compressor.save_to_hub()` — 모델 카드 자동 생성 + `HfApi.upload_folder` |
| Multi-model 검증 | **완료** | TinyLlama-1.1B (LLaMA 아키텍처) — 라이브러리 코드 수정 없이 4 recipe 동작 |

---

## 2026-05-23 — calibration 데이터 개선 + torch cu124 전환

- **`mini_compressor/utils.py` 신규**: `get_calibration_data(tokenizer, dataset, n_samples, seq_len)` — wikitext-2/C4 train split에서 n_samples×seq_len 토큰 시퀀스 무작위 샘플링. GPTQ의 `_load_gptq_calib()`과 동일 방식을 공용 유틸로 일반화.
- **`eval.py` / `demo/demo_smoothquant.py`**: W8A8 static, W8A8+SmoothQuant calibration을 5개 하드코딩 문장 → wikitext-2 128 samples로 교체. 전체 scheme이 동일 품질 calibration 데이터 사용.
- **torch 2.12.0+cu130 → 2.6.0+cu124**: RTX 3060 Ti 드라이버(CUDA 12.6)와 호환되지 않던 cu130 빌드를 cu124로 교체. `pyproject.toml`에 pytorch-cu124 index 추가, `uv.lock` 갱신. nvidia-cudnn-cu12 등 바이너리 패키지 재설치로 CUDA 정상 동작 확인.
- **PPL 재측정 결과**: W8A8+SmoothQuant 21.08 → 18.33 (FP16 대비 +0.17로 개선), W8A8 static 29.71 (MinMax outlier 민감도).

---

## 2026-05-19 — 코드 품질 개선 (버그 수정 + 주석)

- **`Compressor.compress()` hook 잔류 버그 수정**: `calibrate()` 예외 발생 시 `finalize()`가 호출되지 않아
  `SmoothQuantModifier`의 forward hook이 모델에 잔류하는 버그. `try/finally`로 감싸 예외 시에도
  `finalize()` 실행 보장.
- **`assert` → `RuntimeError` 통일**: `QuantizationMixin.finalize()`와 `QuantizationModifier.calibrate()`의
  `assert self.model is not None`을 `RuntimeError`로 교체. `python -O` 플래그로 비활성화되는
  `assert`는 API 계약으로 부적절. `SmoothQuantModifier`(기존 `RuntimeError`)와 일관성 확보.
- **`FakeQuantLinear.input_scale = None` 이중 의미 주석**: (a) static calibration 전, (b) dynamic·weight-only.
  `forward()`가 `scheme.activation` 먼저 체크해 동작은 맞지만 가독성 개선 목적.
- **`_find_quantization_modifier()` docstring 보완**: "첫 번째 QuantizationMixin 선택" 규칙과 복수 인스턴스 시
  동작을 명시.

---

## 2026-05-18 — fake_quant_linear dtype 일관성 버그 수정

- **버그**: `weight_scale`/`input_scale`이 float32로 저장되어 float16 weight/activation과
  연산 시 float32로 업캐스트 → F.linear(float16, float32) 또는 (float32, float32) 혼용.
  W4A16 GPTQ는 scale이 명시적 float32라 `RuntimeError: c10::Half != float`로 즉시 실패.
  W8A8 static/SmoothQuant는 F.linear 결과가 float32로 계산되어 PPL이 부정확.
- **수정**: `_fake_quantize_weight`와 `_fake_quantize_activation`에서 scale을
  `s.to(w.dtype)` / `s.to(x.dtype)`으로 캐스팅 → 연산 dtype이 모델 dtype(float16)으로 통일.
- **영향**: W4A16 GPTQ 동작. W8A8 static 27.75 → 25.01, SmoothQuant 19.28 → 23.67 (float16 시뮬레이션이 더 정확).
  W4A16 RTN·W8A8 dynamic은 값 불변 (float16 scale이라 이미 정상).

---

## 2026-05-18 — weight observer 통합 (granularity-aware)

- [x] observer를 granularity-aware로 통합 — weight·activation이 동일 추상화 공유 (llm-compressor / AMD Quark 표준 패턴)
- [x] `BaseObserver(spec)` 생성자에 spec 주입, `_to_units` 헬퍼로 per_tensor/per_channel/per_group 단위 정리
- [x] `_compute_weight_scale` 삭제 → `QuantizationModifier.initialize()`가 weight observer를 1회 호출
- [x] weight도 `calibration_method`(minmax / percentile / mse) 선택 가능
- [x] `_scale_zp_from_range` 대칭 분기 교정 (`max(|min|,|max|)/qmax`) — weight minmax는 기존 `absmax/qmax`와 수치 동일
- [x] 단위 테스트 35개 통과 (weight observer 3개 추가)

---

## 현재 작업 위치

> **Milestone 1–13 + M6-B(GPTQ+AWQ) + M6-C + M6-D 완료**
> M13 중 Tensor Parallelism(13-3)·실 2-GPU 실측은 범위 외 / 하드웨어 한계.

**다음 할 일.**
1. Float8 stub → 실구현 (선택)

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
5. **Sequential calibration**: `calibrate(sequential=True)` 플래그, `modifiers/quantization.py` 내부 분기

---

## 과제 명세서 가점 요소 현황

| 가점 항목 | 상태 | 비고 |
|-----------|------|------|
| serialization 포맷 설계 (quantization config 등) | **완료** | compressed-tensors 포맷 호환 `quantization_config.json`, `save_pretrained`/`load_pretrained` 구현 |
| HuggingFace Hub model card upload 가능한 구조 | **완료** | `save_to_hub(model, repo_id)` — model card 자동 생성 + `HfApi.upload_folder`. `pip install mini-compressor[hub]`로 선택 의존성 설치 |
| Multi-GPU 고려 | **완료** | M13 — observer `sync()` 3종 모두 NCCL all_reduce. Percentile/MSE는 histogram 기반으로 GPU-native sync. gloo 2-proc 검증 |

---

## Multi-GPU 가산점 요소

> **M13에서 구현 완료** — observer `sync()` 4종 + gloo 2-proc 검증. 아래는 항목별 상태.

| 고려사항 | 상태 |
|---------|------|
| DataParallel calibration | ✅ observer all_reduce / all_gather 동기화 |
| Tensor Parallelism | 범위 외 — shard된 weight의 scale merge는 별도 난제 |
| Sequential + Multi-GPU | 미진행 — device placement 로직 추가 필요 |
| state_dict 저장 | ✅ rank 0 저장 가드 구현됨 (`serialize.py`) |

영향 파일: `observer.py` (sync), `modifiers/quantization.py` (sync 호출), `serialize.py` (rank 0 가드)
