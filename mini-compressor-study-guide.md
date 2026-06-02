# mini-compressor 학습 가이드

부제 — 코드로 배우는 LLM Compression Tool 설계

이 문서는 `github.com/YoungHyun197/mini-compressor` 저장소 하나만 가진 학생이, 코드와 주석을 따라 읽으며 설계 철학·트레이드오프·사용자/개발자 관점을 습득하도록 안내한다. 이 가이드를 끝까지 소화하면 LLM compression tool 설계 리뷰의 심화 점검 항목에 코드 근거를 들어 답할 수 있어야 한다.

읽는 법 — 각 절이 지목하는 파일을 직접 열고, 명시된 함수·클래스를 본문과 함께 읽어라. 가이드만 읽고 코드를 안 보면 절반만 얻는다.

## 이 프로젝트의 본질

이 프로젝트는 "양자화 알고리즘 하나를 구현하는 것"에 그치지 않는다. **LLM compression tool을 어떻게 설계할 것인가**를 검증하는 POC다. 설계 초점은 다음에 있다.

- Quantization 대상을 어떤 단위로 추상화했는가.
- Quantization config를 어떻게 표현했는가.
- 새로운 scheme·알고리즘이 들어오면 변경 범위가 얼마나 작은가.
- load → prepare → calibrate → save → generate workflow를 어떻게 설계했는가.
- HuggingFace / compressed-tensors / runtime 호환을 어느 정도 고려했는가.
- fake quant POC라도 real quantization·NPU backend로 확장 가능한 구조인가.

그래서 이 가이드는 "알고리즘 수식"보다 **"왜 이 파일이 이렇게 생겼는가"**에 집중한다.

## 전체 구조 — 의존성이 곧 학습 순서

```
mini_compressor/
  schemes.py            무엇을 양자화 (config)
  fake_quant_linear.py  어디에 양자화 (단위)
  observer.py           activation을 어떻게 측정 (통계)
  modifiers/
    base.py             알고리즘 공통 계약 (lifecycle)
    quantization.py     RTN 알고리즘 (핵심 흐름)
    smoothquant.py      두 번째 알고리즘 (composition 입증)
    gptq.py / awq.py    고급 PTQ 알고리즘 구현
  recipes.py            preset 레이어 (declarative)
  compressor.py         one-click 진입점
  serialize.py          HF 호환 저장/복원
demo_quick.py           빠른 fake-quant generate 확인
eval.py                 generate / save-load / perplexity 평가
tests/                  설계 주장의 코드 증거
```

권장 읽기 순서는 **의존성 방향**과 같다. 데이터(config) → 단위(module) → 통계(observer) → 알고리즘(modifier) → 조립(recipe·compressor) → 입출력(serialize) → 사용(demo·eval·test). 아래에 있는 것이 위에 있는 것을 사용하므로, 위에서부터 읽어야 막힘이 없다.

1. `schemes.py`
2. `fake_quant_linear.py`
3. `observer.py`
4. `modifiers/base.py`
5. `modifiers/quantization.py`
6. `modifiers/smoothquant.py`
7. `recipes.py`
8. `compressor.py`
9. `serialize.py`
10. `modifiers/gptq.py`, `modifiers/awq.py`
11. `demo_quick.py`, `eval.py`, `tests/`

# 1. schemes.py — 무엇을 양자화하는가

**역할** — 양자화 명세를 if-else 코드가 아니라 선언적 dataclass로 표현한다.

**무엇을 볼까** — `QuantizationSpec`의 필드, `QuantizationScheme`, 프리셋 `W8A8` / `W4A16` / `W8A8_DYNAMIC`, 그리고 `SCHEME_REGISTRY`.

**습득할 개념**

- **2-tier config.** `QuantizationSpec`은 텐서 하나에 대한 수치 명세(bit, symmetric, granularity, dtype, group_size, axis, dynamic, calibration_method)다. `QuantizationScheme`은 그 Spec을 weight와 activation 두 개로 묶은 레이어 단위 정책이다. weight와 activation에 같은 Spec 타입을 재사용한다.
- **두 개의 독립 축.** `granularity`(scale이 커버하는 공간 차원 — per_tensor/per_channel/per_group/per_token)와 `dynamic`(scale 계산 시점 — 정적 calibration / 동적 runtime)은 서로 독립이다. 이 둘을 한 필드로 합치면 "per-token static" 같은 조합을 표현할 수 없다.
- **activation=None의 의미.** `QuantizationScheme.activation`이 `None`이면 weight-only 양자화(W4A16)다. 이 한 값으로 W8A8 계열과 W4A16이 자연스럽게 갈린다.

**왜 이렇게 설계했나** — config를 코드가 아니라 데이터로 둔다. `FakeQuantLinear`와 modifier는 scheme 객체를 런타임에 *읽어서* 동작하므로, 새 수치 포맷은 `schemes.py`에 dataclass 한 줄을 추가하면 끝난다. 모듈 코드는 건드리지 않는다.

**트레이드오프** — 선언적 dataclass는 유연하지만 (1) 스키마가 표현하지 못하는 조합(예: 레이어별 mixed-precision)이 생기면 결국 필드를 늘려야 하고, (2) dataclass에는 검증 로직이 없어 잘못된 조합(group_size 없는 per_group 등)을 런타임까지 못 잡는다.

**사용자 관점** — preset 이름(`"w8a8"`)만 알면 된다. 내부 Spec 필드를 몰라도 된다. **개발자 관점** — 새 수치 포맷 추가가 `schemes.py` 단일 파일로 제한된다. 이것이 뒤에 나올 "scheme 확장 범위" 설계 설명의 근거다.

# 2. fake_quant_linear.py — 어디에 양자화하는가

**역할** — quantization unit. `nn.Linear`를 대체하는 `FakeQuantLinear` 모듈.

**무엇을 볼까** — `__init__`의 `register_buffer` 4개, `from_float`, `forward`, `_fake_quantize_weight`(per_channel/per_group/per_tensor 분기와 float8 stub), `_fake_quantize_activation`(dynamic 분기), `_group_fake_quant`.

**습득할 개념**

- **왜 quantization unit이 nn.Linear인가.** LLM의 계산량 대부분이 attention·MLP 내부의 Linear에 집중된다. `nn.Linear`를 통째로 교체하면 모델 코드를 수정하지 않고 양자화를 끼워넣을 수 있고, `model.generate()` 경로에 자연스럽게 들어가며, scale/zero_point를 모듈 buffer로 저장할 수 있다.
- **fake quantization 수식.** `q = clamp(round(x/s + zp), qmin, qmax)`, `x_hat = (q - zp) * s`. 결과는 여전히 float 텐서지만 값이 양자화 그리드 위에 놓인다. 실제 dtype은 float16 그대로다.
- **flat buffer.** `weight_scale`, `weight_zero_point`, `input_scale`, `input_zero_point`를 중첩 경로(`_weight_quantizer.scale`) 없이 모듈 직속 buffer로 둔다. state_dict의 key가 평평해져 HF `from_pretrained` 로드 시 key 매핑이 명확하다.
- **dynamic 분기.** `_fake_quantize_activation`에서 `spec.dynamic`이면 calibration 없이 runtime에 scale을 계산한다(per_token은 마지막 dim 기준 amax).

**왜 이렇게 설계했나** — module replacement는 Quark에서 차용한 방식이다. hook을 거는 대신 부모 모듈을 찾아 자식을 `setattr`로 교체한다. hook 방식보다 state_dict 호환이 자연스럽다. flat buffer는 그 호환성을 끝까지 밀어붙인 결과다.

**트레이드오프**

- **fake quant의 본질적 한계.** 실제 INT 커널이 아니므로 메모리 절감도 latency 향상도 없다. 양자화 *오차*만 시뮬레이션한다. POC에서 workflow와 정확도 검증이 목적이므로 의도된 선택이다.
- **nn.Linear 단위.** Linear가 아닌 모듈(embedding, lm_head 등)은 예외 처리가 필요하다. attention 내부 연산(softmax 등)은 양자화 대상이 아니다.
- 대안인 tensor-level quantizer는 유연하지만 state_dict에 scale이 독립 key로 안 남고, decoder-layer wrapper는 모델 아키텍처 의존성이 커진다. nn.Linear 교체가 granularity와 호환성의 실용적 균형점이다.

**사용자 관점** — 교체는 투명하다. `generate()`가 그대로 동작한다. **개발자 관점** — 새 dtype(float8 등) 지원은 `_fake_quantize_weight/activation`의 분기 추가로 끝나고, scheme 정의 레이어는 안 바뀐다.

# 3. observer.py — activation을 어떻게 측정하는가

**역할** — calibration 동안 activation 통계를 모아 scale/zero_point를 계산한다.

**무엇을 볼까** — `BaseObserver`(update / compute_scale_zp / reset / sync), 네 가지 observer, `_scale_zp_from_range`(zero 포함 보장), `build_observer`와 `OBSERVER_REGISTRY`, 그리고 `sync`와 헬퍼 `_dist_active` / `_sync_data`.

**습득할 개념**

- **왜 activation은 calibration이 필요한가.** weight는 offline에 고정돼 있어 언제든 값을 본다. activation은 입력에 따라 변하므로, 대표 입력 몇 개를 흘려 분포를 미리 관찰해야 정적 scale을 정할 수 있다.
- **observer 세 종류.** MinMax(running min/max), Percentile(분포 클리핑), MSE(grid-search). MinMax는 통계를 `register_buffer`된 tensor로 누적하고, 나머지 둘은 raw activation을 CPU 리스트(`_data`)로 모은다.
- **observer를 누가 소유하나.** `FakeQuantLinear`가 `input_observer`를 소유하고 `forward`에서 `update(x)`를 호출한다. 그래서 modifier의 `calibrate`는 모델 forward만 돌리면 각 레이어가 알아서 통계를 모은다.
- **multi-GPU sync.** `sync()`는 rank 간 통계를 합친다. MinMax는 `all_reduce(MIN/MAX)` — min·max는 결합법칙이 성립해 부분 결과를 정확히 병합한다. Percentile/MSE는 `all_gather_object`로 raw `_data`를 공유 — percentile·grid-search는 비결합적이라 부분 통계만으론 병합할 수 없기 때문이다. 분산 환경이 아니면 `sync()`는 즉시 return(no-op)한다.

**왜 이렇게 설계했나** — observer를 문자열(`calibration_method`)로 고르고 `OBSERVER_REGISTRY`에서 조회하므로, modifier도 FakeQuantLinear도 Observer 클래스를 직접 import하지 않는다. MinMax의 통계를 Python scalar가 아니라 tensor 연산으로 유지한 것도 의도적이다 — 나중에 `dist.all_reduce` 한 줄로 multi-GPU 동기화가 완성되도록.

**트레이드오프**

- MinMax는 빠르지만 outlier에 민감하다. calibration 샘플을 늘릴수록 극단값에 더 노출돼 per-tensor scale이 과도하게 넓어진다 — 이것이 SmoothQuant를 도입하는 직접적 동기다.
- Percentile/MSE는 더 정확하지만 raw data를 메모리에 들고 있어야 한다.
- multi-GPU에서 raw data를 `all_gather`하는 방식은 정확하고 수술적이지만 메모리가 rank 배로 늘어난다. 대규모 calibration에서는 관측자 알고리즘을 "부분 통계 → 동기화 → 마무리"로 재구조화하는 편이 낫다.

**사용자 관점** — `calibration_method` 문자열 하나로 관측자를 바꾼다. **개발자 관점** — 새 observer는 클래스 하나 + registry 등록 한 줄이다.

# 4. modifiers/base.py — 알고리즘의 공통 계약

**역할** — 모든 압축 알고리즘이 따르는 추상 인터페이스.

**무엇을 볼까** — `BaseModifier` ABC와 추상 메서드 `initialize(model)` / `calibrate(dataloader)` / `finalize()`.

**습득할 개념** — **3단계 lifecycle.** initialize는 모델 구조를 바꾸고(예: Linear→FakeQuantLinear 교체), calibrate는 데이터를 흘려 통계를 모으고, finalize는 임시 자원을 정리한다. 알고리즘마다 세 단계의 *내용*은 다르지만 *인터페이스*는 같다.

**왜 이렇게 설계했나** — 이 공통 계약 덕분에 `Compressor`는 modifier가 RTN인지 SmoothQuant인지 *몰라도* 리스트를 순회하며 같은 세 메서드만 호출하면 된다. lifecycle을 API 레벨에서 명시적으로 분리한 것은 llm-compressor식이다. Quark는 `quantize()` 한 번에 일괄 처리한다 — mini-compressor는 디버깅과 조합을 위해 명시적 분리를 택했다.

**트레이드오프** — 어떤 알고리즘에는 3단계가 과한 구조일 수 있다(예: 순수 weight-only RTN은 calibrate가 거의 빈 함수다). 하지만 알고리즘을 조합하고 단계별로 검증하기에는 명시적 단계가 훨씬 명확하다.

# 5. modifiers/quantization.py — RTN, 핵심 흐름

**역할** — Round-to-Nearest 양자화 modifier. 이 파일이 압축 흐름의 심장이다.

**무엇을 볼까** — `__init__`(scheme / targets / ignore / compute_scales), `_should_replace`, `initialize`의 module replacement 루프, `calibrate`, `finalize`, 모듈 함수 `_compute_weight_scale`.

**습득할 개념**

- **module replacement 메커니즘.** `initialize`는 `named_modules()`를 순회하며 교체 대상 Linear를 모은 뒤, 각각에 대해 부모 모듈을 찾아 `setattr(parent, attr, FakeQuantLinear)`로 자식을 갈아끼운다. 이것이 "모델 코드 수정 없는 양자화"의 실체다.
- **targets / ignore가 model-agnostic인 이유.** `_should_replace`는 모듈 경로와 클래스 이름 양쪽에 `fnmatch`를 적용한다. `targets=["Linear"]`는 클래스명으로, `targets=["model.layers.*.self_attn.*"]`는 경로 패턴으로 매칭된다. `ignore`가 항상 우선한다. 특정 모델 구조에 하드코딩하지 않으므로 Qwen3든 LLaMA든 같은 코드로 동작한다.
- **compute_scales 분리.** `initialize(compute_scales=True)`는 압축 흐름 — 구조를 만들면서 weight scale을 RTN으로 즉시 계산한다. `compute_scales=False`는 load 흐름 — 구조만 만들고 scale은 저장된 state_dict에서 채운다. 같은 코드가 두 흐름을 모두 지원한다.
- **calibrate의 순서.** forward로 통계를 모으고 → `observer.sync()`(multi-GPU 동기화, 단일 GPU면 no-op) → `compute_scale_zp` → `input_scale`을 `weight.device`로 옮겨 저장. 마지막 `.to(device)`는 Percentile 등이 `_data`를 CPU에 모아 scale이 CPU에 남는 문제를 막는다.

**왜 이렇게 설계했나** — `_compute_weight_scale`을 클래스 메서드가 아니라 모듈 수준 함수로 분리해 테스트와 재사용을 쉽게 했다. compute_scales 파라미터는 "구조 생성"과 "scale 로딩"을 분리하는 llm-compressor·Quark 공통 패턴이다.

**트레이드오프** — RTN은 각 weight를 독립적으로 가장 가까운 그리드 값에 반올림한다. 단순하고 빠르지만, weight 간 상호작용을 고려하지 않아 GPTQ 같은 오차 보상 기법보다 정확도가 낮다. 또 calibrate가 전체 모델 forward를 한 번에 돌리므로 대형 모델에서는 메모리가 문제다 — 이를 위한 sequential calibration이 stub으로 명세돼 있다.

**사용자 관점** — `from_recipe` 뒤에 가려져 직접 만질 일이 없다. **개발자 관점** — 이 파일이 "새 알고리즘을 어떻게 쓰는가"의 템플릿이다. GPTQModifier도 이 구조(BaseModifier 상속, 3단계)를 따른다.

# 6. modifiers/smoothquant.py — composition을 입증하는 두 번째 알고리즘

**역할** — activation outlier를 weight로 옮겨 W8A8 정확도를 끌어올리는 SmoothQuant. 그리고 "새 알고리즘 = 새 파일 하나"라는 설계 주장의 코드 증거.

**무엇을 볼까** — `SmoothQuantModifier`, `_find_smooth_pairs`(norm-linear 페어 자동 탐색), `_make_hook`(forward pre-hook), `calibrate`의 smooth factor 계산과 weight 변형.

**습득할 개념**

- **SmoothQuant 수식.** `y = x @ W.T = (x/s) @ (W·s).T`. 양쪽에 같은 per-input-channel 벡터 `s`를 나누고 곱하므로 등가 변환이다. `s_j = max(|x_j|)^α / max(|w_j|)^(1-α)` (α=0.5 기본). activation의 큰 outlier를 `s`로 나눠 평탄하게 만들고, 그만큼을 weight가 흡수한다.
- **offline weight fusing.** `s`를 직전 norm의 weight에 `1/s`로, linear의 weight에 `s`로 미리 곱해 넣는다. runtime에 추가 연산이 0이다 — 양자화 전처리로 끝난다.
- **norm bias 흡수.** LayerNorm은 `weight(γ)`와 `bias(β)` 둘 다 affine이다. `y = γ·x̂ + β`를 `s`로 나누려면 `γ`뿐 아니라 `β`도 나눠야 등가가 유지된다. RMSNorm(Qwen3·LLaMA)은 bias가 없어 이 분기가 작동하지 않지만, LayerNorm 모델(GPT-2 등)에서는 반드시 필요하다.
- **norm-linear 페어 자동 탐색.** `_find_smooth_pairs`는 `input_layernorm`→`self_attn.{q,k,v}_proj`, `post_attention_layernorm`→`mlp.{gate,up}_proj`를 이름으로 찾는다. q/k/v는 같은 입력을 받으므로 한 그룹으로 묶어 하나의 `s`를 공유한다. `o_proj`·`down_proj`는 직전이 norm이 아니라 제외한다(논문도 동일).

**왜 이렇게 설계했나** — 핵심은 이 파일이 추가될 때 무엇이 *안* 바뀌었는지다. SmoothQuant를 구현하면서 `schemes.py`·`fake_quant_linear.py`·`observer.py`·`serialize.py`는 한 줄도 바뀌지 않았다. 알고리즘 추가가 `modifiers/` 디렉토리에 파일 하나 추가로 끝난다 — 이것이 composition 패턴의 실증이다.

**트레이드오프** — 페어 탐색이 attribute 이름 기반 duck typing이다. Qwen3·LLaMA 명명 규칙에만 맞고, GPT-2처럼 q/k/v가 하나로 fused된 모델은 못 잡는다. 더 견고하게 하려면 그래프 추적이나 config 기반 매핑이 필요하다. 또 hook으로 raw activation을 모으므로 calibration 중 메모리를 쓴다.

**사용자 관점** — `from_recipe("w8a8_smoothquant")` 한 줄. **개발자 관점** — 새 알고리즘의 변경 범위가 파일 하나임을 눈으로 확인하라. GPTQ를 추가해도 똑같다.

# 7. recipes.py — declarative preset 레이어

**역할** — composition 패턴 위에 얹은 선언적 preset 레이어.

**무엇을 볼까** — `RECIPE_REGISTRY`, `_rtn` factory, `_w8a8_smoothquant`, 타입 별칭 `RecipeFactory`.

**습득할 개념**

- **scheme과 recipe의 차이.** scheme은 *수치 포맷*(W8A8의 bit·granularity)이고 recipe는 *알고리즘 파이프라인*(modifier들의 순서)이다. 단일 RTN(`w8a8`)은 modifier 1개짜리 recipe, SmoothQuant(`w8a8_smoothquant`)는 `[SmoothQuant, Quantization]` chain recipe다.
- **factory가 callable인 이유.** registry 값은 미리 만든 modifier 인스턴스가 아니라 함수다. modifier는 hook·통계 같은 내부 상태를 갖기 때문에, `from_recipe`를 호출할 때마다 새 인스턴스를 만들어야 한다.
- **두 레이어의 관계.** preset(선언)과 composition(조립)은 배타가 아니라 레이어다. `RECIPE_REGISTRY`는 composition 위에 얹혀 "이름만 알면 되는" 진입점을 제공한다.

**왜 이렇게 설계했나** — 원래는 단일 scheme용 `from_scheme`과 chain용 `from_recipe`가 따로 있었다. 검토 결과 중복은 "두 레지스트리"가 아니라 "두 진입점"에 있었고, 진입점을 `from_recipe` 하나로 통일했다. 단일 RTN도 modifier 1개짜리 recipe로 흡수된다. `SCHEME_REGISTRY`는 사라지지 않고 `serialize.py`가 scheme→이름 매칭에 쓰는 데이터 카탈로그로 남았다.

**트레이드오프** — recipe가 Python factory(코드)다. llm-compressor의 YAML recipe나 Quark의 직렬화 가능한 QuantConfig보다 *선언성*이 약하다. `alpha=0.5` 같은 값이 코드에 박혀 있어, 세부 튜닝은 `Compressor([...])` 직접 조립으로 내려가야 한다.

**사용자 관점** — 이름 하나(`from_recipe("w8a8_smoothquant")`). **개발자 관점** — 새 파이프라인은 `RECIPE_REGISTRY`에 항목 한 줄이다.

# 8. compressor.py — one-click 진입점

**역할** — modifier 리스트를 받아 lifecycle을 순차 실행하는 사용자 진입점.

**무엇을 볼까** — `__init__`(modifier list), `from_recipe`, `compress`, `save`, `_find_quantization_modifier`, `save_to_hub` stub.

**습득할 개념**

- **modifier 리스트가 단일 진실 출처.** `Compressor`는 scheme을 따로 들고 있지 않는다. `save`가 필요할 때 modifier 리스트에서 `QuantizationModifier`를 찾아 그 scheme을 쓴다.
- **compress의 동작.** 각 modifier에 대해 `initialize → calibrate → finalize`를 순서대로 호출한다. 리스트 순서가 곧 알고리즘 chain 순서다 — SmoothQuant가 먼저 weight를 변형하고, 그 위에서 Quantization이 RTN을 돈다.

**왜 이렇게 설계했나** — `oneshot()`(llm-compressor), `quantize_model()`(Quark) 모두 원클릭 진입점을 제공한다. 한 줄 API는 데모에서도 강력하다.

**트레이드오프** — 각 modifier를 완전히(initialize~finalize) 끝낸 뒤 다음으로 넘어간다. llm-compressor의 진짜 recipe 스케줄러처럼 여러 modifier가 한 calibration 패스를 공유하지는 않는다. PTQ 범위에서는 충분하지만 QAT까지 가면 이 부분은 재설계가 필요하다 — 의도적으로 단순화한 지점이다.

**사용자 관점** — `Compressor.from_recipe(name).compress(model)`. **개발자 관점** — 새 알고리즘을 추가해도 `Compressor` 자체는 거의 바뀌지 않는다.

# 9. serialize.py — HF 호환 저장과 복원

**역할** — 양자화된 모델을 compressed-tensors 포맷으로 저장하고 되살린다.

**무엇을 볼까** — `_scheme_to_dict` / `_scheme_from_dict`, `save_pretrained`(rank 0 가드, `base_model_name_or_path`), `load_pretrained`의 5단계 흐름.

**습득할 개념**

- **자체 포맷을 만들지 않는다.** `quantization_config.json`은 HF compressed-tensors 스펙(`config_groups`, `quant_type`)을 그대로 따른다. vLLM 등 runtime이 그대로 로드할 수 있는 포맷이 목표다.
- **calibrated vs compressed.** `quantization_status: "calibrated"`는 fake quant 상태 — weight가 float16 그대로다. 실제 INT 패킹(`"compressed"`)은 컴파일러·런타임이 처리하는 다음 단계다. 툴체인 레이어 간 책임 분리다.
- **왜 from_pretrained인가.** `load_pretrained`는 `from_config().to(float16)`이 아니라 원본 `from_pretrained(model_id)`로 base 모델을 만든다. 전자는 `inv_freq` 같은 non-persistent buffer를 float16으로 바꿔버려, 28개 attention 레이어를 거치며 logit 오차가 증폭되고 round-trip이 깨진다.
- **왜 load_state_dict를 안 쓰나.** PyTorch의 `_load_from_state_dict`는 `None` buffer를 건너뛴다. `initialize(compute_scales=False)` 직후 scale buffer가 `None`이라 `copy_()`가 호출되지 않아 값 주입이 조용히 실패한다. 그래서 저장된 state를 직접 순회하며 parameter와 buffer에 할당한다.

**왜 이렇게 설계했나** — scheme→dict 직렬화 로직을 `schemes.py`가 아니라 `serialize.py`에 둔다. `schemes.py`는 "무엇을 양자화", `serialize.py`는 "어떻게 저장"으로 책임을 분리해, 포맷 요구가 바뀌면 `serialize.py` 하나만 고친다.

**트레이드오프** — fake quant라 weight를 float16으로 저장하므로 저장 파일 크기는 줄지 않는다. 또 `load_pretrained`가 base 모델을 다시 받아야 하므로 HF 캐시나 네트워크 접근이 필요하다.

**사용자 관점** — `compressor.save(...)` / `load_pretrained(dir)`. **개발자 관점** — 새 직렬화 요구는 `serialize.py` 한 파일로 제한된다.

# 10. gptq.py / awq.py — stub의 의미

**역할** — W4A16이라는 같은 최종 scheme을 더 좋은 weight update / activation-aware scaling으로 얻기 위한 고급 PTQ 알고리즘.

**무엇을 볼까** — `GPTQModifier`의 Hessian 수집과 error propagation, `AWQModifier`의 activation mean 수집과 scaling search, 그리고 둘 다 `BaseModifier` lifecycle 안에 들어가는 방식.

**습득할 개념** — 새 알고리즘은 scheme 정의를 다시 만드는 일이 아니라, 필요한 통계와 model update를 `Modifier` 안에 격리하는 일이다. GPTQ는 Hessian, AWQ는 activation mean과 grid search가 필요하지만 `Compressor` orchestration은 그대로 유지된다.

# 11. demo_quick.py / eval.py / tests/ — 사용과 증거

**demo_quick.py** — 제출 요구사항에 맞춘 빠른 fake-quant generate smoke demo. 기본 `w8a8` static을 포함해 `--recipe`로 `RECIPE_REGISTRY`의 모든 recipe를 선택할 수 있고, baseline generate와 fake-quant generate 문장을 바로 보여준다.

**eval.py** — 평가용 데모. `--model` 인자로 모델 비종속이다(`Qwen3-0.6B` 기본, `TinyLlama` 등으로 교체 가능). FP16·W4A16·W4A16-GPTQ·W8A8·W8A8-dynamic·W8A8+SmoothQuant를 compress→generate하고, `--ppl`로 perplexity까지 측정한다.

**tests/** — 32개 단위 테스트. 단순 회귀 방지를 넘어 *설계 주장을 코드로 못박는다*. `test_smoothquant.py`는 SmoothQuant가 양자화 없이는 출력을 보존하는 등가 변환임을 검증하고, `test_observer_sync.py`는 gloo 2-프로세스로 multi-GPU 동기화를 실제로 돌려 검증한다.

**습득할 개념** — `generate()`가 동작한다는 것은 "양자화가 forward 경로를 깨뜨리지 않는다"는 검증이다. 테스트가 통과한다는 것은 "model-agnostic", "등가 변환" 같은 말로 한 주장이 코드로 증명됐다는 뜻이다.

# 12. 설계 철학 통합 — 하이브리드

mini-compressor는 한 OSS를 베끼지 않았다. 레이어별로 출처가 다른 **하이브리드**다.

| 레이어 | 출처 | 내용 |
|--------|------|------|
| 사용자 API · 저장 포맷 | HF / llm-compressor | save_pretrained, compressed-tensors |
| lifecycle 3단계 | llm-compressor | initialize / calibrate / finalize 명시적 분리 |
| 진입점 (recipe) | llm-compressor | named recipe → modifier 리스트 |
| module replacement | AMD Quark | parent setattr로 nn.Linear 교체 |
| 선언적 config 객체 | AMD Quark 정신 | QuantizationScheme 2-tier |
| weight 포맷 | backend 경계 분리 | fake quant float16, real packing은 compiler/runtime 단계 |

**composition vs config-dispatch.** llm-compressor는 알고리즘마다 Modifier 클래스를 두고 recipe로 조합한다(composition). Quark는 중앙 config 객체를 두고 quantizer가 분기한다(config-dispatch). 전자는 확장이 깔끔하지만 사용자가 조합을 알아야 하고, 후자는 쓰기 쉽지만 확장 시 중앙을 건드린다. mini-compressor는 composition을 메인으로 택하되, "어떤 수치 포맷이냐"는 Quark식 선언적 `QuantizationScheme`으로 분리한 하이브리드다.

**핵심 설계 항목** — 설계 검토에서 가장 집중적으로 묻는 부분이다.

1. **추상화 단위는?** `nn.Linear`를 대체하는 `FakeQuantLinear`. LLM 계산이 Linear에 집중되고, module replacement가 state_dict 호환과 generate 경로 통합에 자연스럽기 때문.
2. **config는 어떻게 표현했나?** `QuantizationSpec`(텐서 수준) + `QuantizationScheme`(weight·activation 쌍)의 2-tier. 새 scheme은 dataclass 한 줄.
3. **새것이 들어오면 변경 범위는?** 새 *수치 포맷*은 `schemes.py` 한 파일. 새 *알고리즘*은 `modifiers/`에 파일 하나 추가, 기존 modifier 무수정.

# 13. 트레이드오프 종합

| 결정 | 얻은 것 | 내준 것 |
|------|---------|---------|
| fake quant | 구현 단순, generate 그대로 동작, 정확도 검증 가능 | 실제 speedup·메모리 절감 없음 |
| nn.Linear 단위 | state_dict 호환, generate 통합, model-agnostic | Linear 외 모듈 예외 처리 |
| per-tensor activation static | runtime scale 계산 비용 0 | outlier에 취약 (W8A8 PPL 손실) |
| MinMax observer 기본 | 빠름, all_reduce 동기화 쉬움 | outlier 민감, 샘플 늘려도 개선 안 될 수 있음 |
| composition 패턴 | 알고리즘 추가가 파일 하나 | 사용자가 조합을 알아야 함 (→ recipe로 완화) |
| recipe = Python factory | 구현 단순, 코드로 표현 | YAML/config보다 선언성·직렬화성 약함 |
| name-based pair 탐색 | 단순, 디버깅 쉬움 | Qwen3/LLaMA 명명에만 동작 |
| multi-GPU all_gather sync | compute 코드 무수정, 정확 | 메모리 rank 배 |
| 순차 modifier 실행 | 흐름 단순, 디버깅 명확 | calibration 패스 공유 안 함 (QAT엔 재설계) |

# 14. 심화 점검 항목 대비

아래 항목을 코드 근거와 함께 설명할 수 있어야 한다.

**Q. 양자화 단위를 왜 nn.Linear로 잡았나? 대안과 트레이드오프는?**
LLM 계산량이 attention·MLP의 Linear에 집중되고, module replacement 방식이 모델 코드 수정 없이 generate 경로에 들어가며 scale을 buffer로 저장할 수 있다. tensor-level quantizer는 유연하지만 state_dict에 scale이 독립 key로 안 남고, decoder-layer wrapper는 모델별로 달라진다. nn.Linear 교체가 granularity와 호환성의 균형점이다.

**Q. fake quant의 한계는? real quant·NPU로 어떻게 확장하나?**
fake quant는 dtype이 float16이라 메모리·latency 이득이 없고 양자화 오차만 시뮬레이션한다. 다만 `FakeQuantLinear` 추상화와 `quantization_config.json` 메타데이터를 real packed weight·NPU 커널로 교체 가능하게 설계했다. `quantization_status`가 `"calibrated"`에서 `"compressed"`로 가는 것이 그 경계다.

**Q. W8A8 static이 dynamic보다 perplexity가 나쁜 이유는?**
static은 per-tensor scale 하나를 calibration으로 고정하는데, MinMax observer는 outlier에 민감해 scale이 과도하게 넓어진다. dynamic은 토큰마다 runtime scale을 계산해 outlier 영향을 받지 않는다. 측정에서도 W8A8 dynamic은 FP16에 근접(18.48 vs 18.16)했다.

**Q. SmoothQuant가 해결하는 문제는? 왜 norm으로 흡수하나?**
activation outlier가 per-tensor scale을 망친다. SmoothQuant는 `s`로 activation을 나눠 평탄하게 만들고 그만큼을 weight가 흡수한다. `x/s`를 만들려면 그 `x`를 생산하는 직전 norm의 weight에 `1/s`를 미리 곱해 넣어야 한다 — 그래서 norm이다. runtime 추가 연산이 0인 offline fusing이다.

**Q. GQA에서 SmoothQuant fuse 기준이 달라지나?**
표준 input-channel SmoothQuant에서는 안 바뀐다. smoothing 축은 input channel(hidden_size)이고 q/k/v가 모두 같은 norm 출력을 받으므로 in_features가 동일하다. GQA가 줄이는 것은 output(head) 축인데 SmoothQuant는 output 축을 건드리지 않는다. 코드에 GQA 전용 분기가 없는 이유다.

**Q. 새 scheme 추가와 새 algorithm 추가의 변경 범위 차이는?**
새 scheme(수치 포맷)은 `schemes.py`에 dataclass + registry 한 줄, 그리고 recipe 한 줄. 새 algorithm은 `modifiers/`에 `BaseModifier`를 상속한 파일 하나. 둘 다 기존 코드를 수정하지 않는다. SmoothQuant 구현이 이를 실증한다.

**Q. scheme과 recipe는 뭐가 다른가? 왜 진입점을 통일했나?**
scheme은 수치 포맷, recipe는 알고리즘 파이프라인이다. 원래 `from_scheme`/`from_recipe` 두 진입점이 있었는데, 중복은 진입점에 있었다. `from_recipe` 하나로 통일하고 단일 RTN도 modifier 1개짜리 recipe로 흡수했다. 값이 늘어도 진입점이 하나라 API 표면적이 안 늘고 이름 충돌이 불가능하다.

**Q. composition과 config-dispatch의 트레이드오프는?**
composition(llm-compressor)은 알고리즘 추가가 파일 하나로 깔끔하지만 사용자가 조합을 알아야 한다. config-dispatch(Quark)는 쓰기 쉽지만 확장 시 중앙 dispatcher를 건드린다. mini-compressor는 composition을 택하되 recipe 레이어로 진입장벽을 낮췄다.

**Q. lifecycle을 왜 3단계로 나눴나? finalize에서 observer를 왜 제거하나?**
initialize(구조 변경)·calibrate(통계)·finalize(정리)를 분리하면 Compressor가 알고리즘 종류를 몰라도 순회할 수 있고 단계별 디버깅이 된다. finalize는 `input_observer`를 `None`으로 만들어 state_dict에 observer 잔여물이 안 남게 한다 — 컴파일러에 넘길 깨끗한 weight+scale만 남긴다.

**Q. load에서 왜 from_pretrained를 쓰나? load_state_dict는 왜 안 쓰나?**
`from_config().to(float16)`은 `inv_freq` 같은 non-persistent buffer를 float16으로 바꿔 round-trip을 깨뜨린다. `from_pretrained`는 dtype 일관성을 지킨다. `load_state_dict`는 `None` buffer를 건너뛰는데, `compute_scales=False` 직후 scale buffer가 `None`이라 값 주입이 조용히 실패한다 — 그래서 직접 주입 루프를 쓴다.

**Q. multi-GPU observer sync에서 MinMax는 all_reduce, 나머지는 all_gather인 이유는?**
min·max는 결합법칙이 성립해 `all_reduce(MIN/MAX)` 한 줄로 부분 결과를 정확히 병합한다. percentile·grid-search는 비결합적이라 부분 통계만으론 전역 결과를 못 만든다. raw data를 `all_gather`해 모든 rank가 같은 전역 데이터로 계산하게 했다 — compute 코드를 수정하지 않는 수술적 선택이다.

**Q. weight는 per-channel/per-group, activation은 per-tensor인 이유는?**
weight는 offline 고정이라 channel·group 단위 scale을 저장해도 overhead가 작다. activation은 입력마다 변하고 토큰마다 계산되므로 scale의 계산·저장·적용 비용이 중요하다. 그래서 activation은 per-tensor(static) 또는 per-token(dynamic) 중 runtime 비용과 정확도를 보고 고른다.

**Q. quantization_status의 calibrated와 compressed 차이는?**
`calibrated`는 fake quant 상태 — weight가 float16, scale buffer가 함께 저장된다. `compressed`는 실제 INT 패킹이 끝난 상태다. mini-compressor는 정확도 검증 단계(calibrated)에 집중하고, real packing은 컴파일러·런타임의 책임으로 분리했다.

# 15. 심화 점검 항목 — 추가 20선

앞 절의 13문에 더해, 코드를 깊게 읽어야 답할 수 있는 20문을 더한다. 모든 답은 실제 파일·함수에 근거를 둔다.

**Q. per-group 양자화는 왜 in_features 축(axis=1)으로 그룹을 나누나? weight_scale shape는?**
`weight`는 `[out_features, in_features]`다. per_group은 한 출력 뉴런이 받는 입력 weight들을 `group_size`(128)개씩 묶어 그룹마다 scale 하나를 둔다 — 그래서 in_features 방향 분할이 자연스럽다. `_compute_weight_scale`은 weight를 `[out, in//group_size, group_size]`로 reshape하고 `dim=2`에서 `amax`를 취해, scale shape는 `[out_features, in_features // group_size]`가 된다. per_channel(축 0, scale `[out_features]`)보다 촘촘해 INT4의 좁은 범위를 보완한다.

**Q. weight는 symmetric, activation은 asymmetric으로 둔 이유는?**
weight 분포는 0을 중심으로 대칭에 가까워 symmetric(zero_point=0 고정)이 효율적이다 — zero_point 저장·연산을 생략한다. activation은 ReLU·GELU 등 비선형을 거쳐 한쪽으로 치우치므로 asymmetric이 양자화 범위를 더 알뜰하게 쓴다. 그래서 W8A8은 weight per_channel symmetric + activation per_tensor asymmetric이다. 단 dynamic activation은 symmetric으로 강제한다 — 토큰마다 min/max 두 값을 계산하는 오버헤드를 피하고 하드웨어 최적화와 정렬하기 위해서다.

**Q. zero_point는 언제 0이고 언제 계산하나? scale 계산에서 0을 범위에 강제 포함하는 이유는?**
symmetric이면 `zero_point=0`, asymmetric이면 `clamp(round(qmin - min/scale), qmin, qmax)`로 계산한다. `_scale_zp_from_range`는 항상 `min_val = min(min_val, 0)`, `max_val = max(max_val, 0)`으로 0을 범위에 강제 포함한다 — 실수값 0이 양자화 그리드 위에 정확히 표현돼야 padding·마스킹 같은 곳의 0이 복원 시에도 정확히 0이 되기 때문이다.

**Q. observer를 modifier가 아니라 FakeQuantLinear가 소유하게 한 이유는?**
`FakeQuantLinear`가 `input_observer`를 들고 `forward`에서 `observer.update(x)`를 호출한다. 그래서 modifier의 `calibrate`는 모델 forward만 돌리면 각 레이어가 알아서 통계를 모은다. 책임이 깔끔하게 나뉜다 — modifier는 "forward 루프 + 루프 후 scale 채우기"만 담당하고, observer 생명주기는 레이어가 안다. modifier가 observer를 일일이 추적할 필요가 없다.

**Q. calibration 샘플을 5개에서 128개로 늘렸더니 W8A8 static perplexity가 오히려 나빠졌다. 왜인가?**
MinMax observer는 본질적으로 outlier에 민감하다. 샘플이 많아질수록 극단값에 더 노출돼 per-tensor scale이 과도하게 넓어지고, 그러면 대부분의 정상 값에 배정되는 양자화 그리드가 성겨진다. 실측에서 5샘플 25.01 < 128샘플 27.75였다. 함의 — MinMax는 calibration을 늘려도 개선이 보장되지 않는다. percentile/MSE observer나 SmoothQuant가 본질적 해법이다.

**Q. SmoothQuant의 alpha(α)는 무엇을 조절하나? α가 0이나 1이면?**
`s = max(|x|)^α / max(|w|)^(1-α)`에서 α는 "activation의 부담을 weight로 얼마나 옮길지"를 정한다. α=1이면 `s ≈ x_max`라 activation outlier를 전부 weight로 떠넘겨 weight가 과부하된다. α=0이면 거의 안 옮긴다. α=0.5는 activation과 weight가 양자화 난이도를 절반씩 나눠 갖는 균형점이다 — 한쪽만 쉬워지고 다른 쪽이 깨지면 의미가 없으므로 중간값을 기본으로 둔다.

**Q. SmoothQuant에서 o_proj·down_proj는 왜 제외하나?**
SmoothQuant는 `x/s`를 만들려고 그 `x`를 생산하는 직전 norm의 weight에 `1/s`를 흡수시킨다. 그런데 o_proj의 입력은 attention 출력이고 down_proj의 입력은 gate·up의 곱이다 — 둘 다 직전이 norm이 아니다. 흡수시킬 norm.weight가 없으므로 표준 SmoothQuant 대상에서 제외한다(논문도 동일).

**Q. q/k/v를 한 그룹으로 묶어 하나의 s를 공유하는 이유는?**
q_proj·k_proj·v_proj는 모두 같은 `input_layernorm` 출력을 입력으로 받는다. norm.weight는 하나뿐이라 거기 흡수시킬 `s`도 하나여야 한다. 만약 q/k/v가 각자 다른 s를 쓰면 norm은 하나의 `1/s`만 흡수할 수 있어 등가 변환이 깨진다. 그래서 그룹 내 세 linear의 `w_max`를 함께 보고 단일 `s`를 만든다.

**Q. GPTQ는 RTN의 어떤 한계를 보완하나? Hessian이 왜 필요한가?**
RTN은 각 weight를 독립적으로 가장 가까운 그리드 값에 반올림한다 — weight 간 상호작용을 무시한다. GPTQ는 한 weight를 양자화하며 생긴 오차를 아직 양자화하지 않은 다른 weight들에 보상한다. 어떤 weight가 출력에 더 민감한지를 layer 입력의 Hessian(2차 곡률 ≈ 입력 공분산 XᵀX, calibration 데이터로 추정)으로 측정해, 민감한 방향의 오차를 우선 보정한다.

**Q. AWQ와 SmoothQuant는 둘 다 채널별 scaling인데 무엇이 다른가?**
목적이 다르다. SmoothQuant는 activation outlier를 weight로 옮겨 **activation 양자화**(W8A8의 A8)를 쉽게 만드는 게 목표다. AWQ는 activation magnitude로 *중요한 weight 채널*을 식별해 그 채널을 scaling으로 보호함으로써 **weight 양자화**(W4의 W4) 정확도를 높이는 게 목표다. SmoothQuant는 A8 타깃, AWQ는 W4 타깃 — 그래서 mini-compressor에서도 SmoothQuant는 W8A8 recipe에, AWQ stub은 W4A16 계열에 연결된다.

**Q. targets/ignore를 fnmatch로 한 것과 FX graph 추적 방식의 트레이드오프는?**
fnmatch는 모듈 이름 문자열 매칭이다 — 단순하고 의존성이 없으며 디버깅이 쉽지만 명명 규칙에 의존한다. FX graph 추적은 실제 연산 그래프를 분석해 이름과 무관하게 구조를 파악하지만, 동적 제어흐름·커스텀 op에서 trace가 깨지기 쉽고 복잡하다. POC는 fnmatch를 택했다 — Linear는 어느 모델이든 `nn.Linear`라 클래스명 매칭이 견고하고, 세밀한 지정이 필요하면 경로 패턴을 쓰면 된다.

**Q. 구조 생성과 scale 계산을 분리한(compute_scales) 이유는? load 흐름에서의 이점은?**
`initialize(compute_scales=True)`는 압축 흐름 — 구조를 만들며 RTN scale을 즉시 계산한다. `False`는 load 흐름 — 구조만 만들고 scale은 저장된 state_dict에서 채운다. load에서 scale을 재계산하면 낭비일 뿐 아니라, load 시점의 weight는 이미 fake-quant로 변형돼 있어 원본 기준으로 계산한 저장값과 달라질 수 있다. 구조 생성과 값 로딩의 분리는 llm-compressor·Quark 공통 패턴이다.

**Q. from_scheme을 제거했는데 SCHEME_REGISTRY는 왜 남겼나?**
`from_scheme`(진입점)은 제거됐지만 `SCHEME_REGISTRY`(dict)는 데이터 카탈로그로 남았다. `serialize.py`의 `_scheme_from_dict`가 저장된 `quantization_config.json`을 다시 scheme 객체로 복원할 때, 알려진 scheme들과 대조해 이름(`"w8a8"` 등)을 붙이는 데 쓴다. 즉 "진입점" 역할만 벗고 "이름↔scheme 카탈로그" 역할은 유지한다 — 한 객체가 쓰던 두 모자 중 하나만 벗긴 것이다.

**Q. recipe가 Python factory(코드)인 것의 한계는? YAML recipe라면 무엇이 좋아지나?**
`RECIPE_REGISTRY`의 항목은 Python 함수다 — `alpha=0.5` 같은 값이 코드에 박혀 있고 recipe 자체를 직렬화(저장·전송)할 수 없다. llm-compressor식 YAML recipe라면 코드 수정 없이 파일만 바꿔 실험하고, recipe를 모델과 함께 저장하며, 비개발자도 편집할 수 있다. 대신 YAML은 파싱·검증 레이어가 필요하다. POC는 단순함을 위해 코드 factory를 택했고, 세부 튜닝은 `Compressor([...])` 직접 조립으로 내려가게 했다.

**Q. multi-GPU를 gloo 2-process로 검증한 것이 실제 2-GPU 검증을 대체하나?**
완전 대체는 아니다. gloo 2-process는 `all_reduce`/`all_gather` **로직의 정확성** — 분산 결과가 전체 데이터를 단일 프로세스로 돌린 결과와 일치하는지 — 를 진짜로 검증한다. 하지만 실제 2-GPU의 device 배치, NCCL 백엔드 동작, cross-GPU 메모리는 검증하지 못한다. 정직한 표현은 "동기화 알고리즘은 검증됐고, 물리적 하드웨어 통합은 미검증"이다.

**Q. quantization_config.json을 config.json에 합치지 않고 별도 파일로 둔 이유는?**
`config.json`은 HF 모델 아키텍처 설정이고 양자화 메타데이터는 별개 관심사다. 별도 파일로 두면 모델 config를 오염시키지 않고(책임 분리), llm-compressor·compressed-tensors가 쓰는 방식과 일치하며, 양자화되지 않은 모델과도 포맷이 호환된다. 새로운 직렬화 요구가 생기면 수정 범위가 `serialize.py` 하나로 제한된다.

**Q. fake quant 상태의 weight를 `weight` key로 두고 `qweight`로 바꾸지 않은 이유는?**
fake quant 상태는 weight dtype이 float16 그대로다. `weight` key로 저장하면 HF `from_pretrained` 로드 경로가 수정 없이 동작한다. `qweight`(packed int)로 분리하는 것은 real int packing을 구현할 때다 — 그때는 dtype·shape가 달라 별도 key가 필요하다. POC 범위에서는 `weight` key 유지가 호환성에 유리하다.

**Q. modifier를 순차 실행하는(각자 initialize~finalize 완료 후 다음) 설계의 한계는? QAT로 가면?**
`Compressor.compress`는 각 modifier를 완전히 끝낸 뒤 다음으로 넘어간다. PTQ에서는 충분하다 — SmoothQuant가 weight를 변형하고 그 위에서 Quantization이 RTN을 도는 순차 의존이 자연스럽다. 하지만 QAT(학습 중 양자화)로 가면 여러 modifier가 같은 학습 루프와 forward 패스를 공유하고 epoch 중간에 끼어들어야 한다. 그때는 llm-compressor식 recipe 스케줄러(modifier가 학습 스텝에 훅으로 참여)로 재설계가 필요하다 — 의도적으로 단순화한 지점이다.

**Q. Sequential calibration은 어떤 문제를 푸나? 변경 범위는 어디인가?**
대형 모델(7B+)은 전체 forward를 한 번에 GPU에 올릴 수 없다. Sequential calibration은 layer를 하나씩 GPU에 올려 calibrate하고 CPU로 offload해, 메모리 사용을 O(전체 모델)에서 O(단일 layer)로 줄인다. 변경 범위는 `modifiers/quantization.py`의 `calibrate()` 내부 분기(full forward vs layer-by-layer)뿐이다 — `FakeQuantLinear`·`schemes.py`·`serialize.py`는 건드리지 않는다. 현재는 stub으로 인터페이스만 확정돼 있다.

**Q. 이 툴을 NPU backend로 확장한다면 무엇을 교체하고 무엇을 유지하나?**
유지 — config 레이어(`QuantizationScheme`), workflow(initialize/calibrate/finalize), 직렬화 메타데이터(`quantization_config.json`), targets/ignore. 교체 — `FakeQuantLinear._fake_quantize_*`의 fake quant 연산을 real packed weight + NPU 커널 호출로, `quantization_status`를 `"calibrated"`에서 `"compressed"`로, weight key를 `weight`에서 `qweight`로. 핵심은 `FakeQuantLinear` 추상화와 메타데이터가 이 교체를 흡수하도록 설계됐다는 점이다 — 상위 레이어(scheme·modifier·compressor·serialize 진입점)는 바뀌지 않는다.

# 마치며

이 가이드의 모든 절은 결국 한 문장으로 모인다 — **"새 scheme·새 알고리즘·새 모델이 들어와도 변경 범위를 작게 유지하는 구조"**. 코드를 읽을 때마다 "이 파일이 바뀌면 무엇이 같이 바뀌는가"를 물어라. 답이 "거의 없다"에 수렴한다면, 그 설계 의도를 제대로 읽은 것이다.
