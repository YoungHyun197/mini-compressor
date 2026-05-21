# mini-compressor 학습 가이드

이 문서는 양자화 도구를 처음 공부하는 학생이 `mini-compressor`의 파이썬 파일을 어떤 순서로 읽으면 좋은지, 그리고 AMD Quark와 llm-compressor의 설계 철학과 비교했을 때 이 프로젝트에서 무엇을 배울 수 있는지 정리한 학습용 문서다.

기준은 현재 레포지토리의 실제 코드다. 따라서 문서나 과거 메모에 있었더라도 현재 코드에 없는 기능은 "현재 미구현"으로 분리해서 설명한다. 예를 들어 observer는 현재 `minmax`, `percentile`, `mse`만 구현되어 있고 KL observer는 없다.

## 먼저 알아야 할 큰 그림

`mini-compressor`는 Hugging Face PyTorch 모델의 `nn.Linear`를 `FakeQuantLinear`로 교체하고, recipe와 modifier 조합으로 RTN, SmoothQuant, GPTQ, AWQ 스타일의 post-training quantization 흐름을 실험할 수 있게 만든 작은 양자화 프레임워크다.

이 프로젝트의 핵심은 "실제 INT packed kernel을 제공하는 런타임"이 아니라 "양자화 도구가 어떤 구조로 설계되는지"다. 즉, 아래 질문을 코드로 답한다.

- 어떤 layer를 양자화할지 어떻게 선택하는가?
- weight scale과 activation scale은 누가 계산하는가?
- calibration 데이터는 어떤 lifecycle에서 흘러가는가?
- 알고리즘은 monolithic 함수가 아니라 modifier로 어떻게 분리되는가?
- 저장 포맷은 어떻게 Hugging Face 생태계와 이어지는가?
- GPTQ, AWQ, SmoothQuant 같은 알고리즘을 하나의 압축 pipeline에 어떻게 끼워 넣는가?

## 추천 학습 순서

처음부터 `demo.py`를 읽으면 전체 흐름은 보이지만 내부 설계가 잘 보이지 않는다. 반대로 `GPTQModifier`부터 읽으면 수학과 lifecycle이 섞여 부담이 크다. 아래 순서가 가장 안정적이다.

| 순서 | 파일 | 먼저 배울 것 |
|---:|---|---|
| 1 | `mini_compressor/schemes.py` | 양자화 설정을 어떤 데이터 구조로 표현하는지 |
| 2 | `mini_compressor/fake_quant_linear.py` | 실제 `nn.Linear`가 어떤 모듈로 바뀌는지 |
| 3 | `mini_compressor/observer.py` | scale/zero-point를 어떻게 계산하는지 |
| 4 | `mini_compressor/modifiers/base.py` | 모든 알고리즘이 따르는 lifecycle |
| 5 | `mini_compressor/modifiers/__init__.py` | modifier package의 public export 경계 |
| 6 | `mini_compressor/modifiers/quantization.py` | RTN, module replacement, activation calibration |
| 7 | `mini_compressor/compressor.py` | 여러 modifier를 한 번에 실행하는 orchestration |
| 8 | `mini_compressor/recipes.py` | 사용자 친화적인 preset 이름을 modifier pipeline으로 바꾸는 방식 |
| 9 | `mini_compressor/modifiers/_pair_utils.py` | SmoothQuant/AWQ가 의존하는 norm-linear pair 탐색 |
| 10 | `mini_compressor/modifiers/smoothquant.py` | activation outlier를 weight로 흡수하는 등가 변환 |
| 11 | `mini_compressor/modifiers/gptq.py` | Hessian 기반 weight-only INT4 보정 |
| 12 | `mini_compressor/modifiers/awq.py` | activation-aware scaling과 grid search |
| 13 | `mini_compressor/serialize.py` | `quantization_config.json`과 저장/로드 |
| 14 | `mini_compressor/__init__.py` | 외부에 노출하는 public API |
| 15 | `demo.py` | 실제 사용 흐름, generation, PPL, round-trip |
| 16 | `tests/*.py` | 설계 의도를 회귀 테스트로 어떻게 고정했는지 |

학습할 때는 "코드가 무엇을 하는가"보다 "왜 이 책임이 이 파일에 있는가"를 계속 묻는 것이 좋다. 양자화 도구의 완성도는 수식 하나보다 lifecycle, target 선택, calibration, 저장 포맷, 에러 처리에서 드러난다.

## AMD Quark와 llm-compressor를 먼저 이해하기

### AMD Quark란?

AMD Quark는 PyTorch와 ONNX 모델을 대상으로 양자화와 export를 제공하는 모델 최적화 도구다. 공식 문서 기준으로 PyTorch LLM flow에서는 원본 모델 로드, calibration dataloader 준비, quantization configuration 설정, 모델 모듈의 in-place quantized module replacement, export 단계가 주요 흐름이다. ONNX flow와 PyTorch flow를 모두 다루며, AMD 하드웨어 배포와 export 관점이 강하다.

Quark의 설계 철학은 다음과 같이 이해하면 된다.

- **하드웨어/배포 지향**: 어떤 backend에서 실행할 것인지가 중요하다.
- **configuration 중심**: 사용자는 quantization config와 algorithm config를 정의한다.
- **in-place module replacement**: float module을 quantized module로 바꿔 실제 추론 구조에 가깝게 만든다.
- **export가 중요한 단계**: 양자화 결과는 배포 가능한 형식으로 내보내야 의미가 있다.
- **알고리즘보다 runtime boundary가 선명함**: weight, activation, KV cache, FP8/INT8/INT4 같은 선택지가 실제 backend와 연결된다.

`mini-compressor`가 Quark에서 배울 수 있는 부분은 "양자화는 단순히 텐서를 round하는 함수가 아니라 모델 구조를 바꾸는 작업"이라는 점이다. `QuantizationMixin.initialize()`가 `nn.Linear`를 `FakeQuantLinear`로 바꾸는 방식이 이 철학과 닮아 있다.

### llm-compressor란?

llm-compressor는 vLLM ecosystem과 가까운 LLM compression 도구로, recipe와 modifier 중심의 one-shot PTQ 흐름을 제공한다. 공식 문서 기준 `oneshot`은 Hugging Face `transformers` 모델을 로드하고, recipe-defined modifier를 적용하며, GPTQ/AWQ/SmoothQuant/QuantizationModifier 같은 calibration 기반 압축 알고리즘을 실행하고 저장할 수 있는 entrypoint다.

llm-compressor의 설계 철학은 다음과 같다.

- **recipe 중심**: 사용자는 어떤 modifier를 어떤 순서로 적용할지 recipe로 선언한다.
- **modifier composition**: GPTQ, AWQ, SmoothQuant, QuantizationModifier 같은 알고리즘을 독립된 modifier로 조합한다.
- **one-shot pipeline**: pretrained model에 calibration을 한 번 수행하고 압축 결과를 저장한다.
- **Hugging Face 친화성**: 모델 로딩, tokenizer, 저장, `compressed-tensors` metadata와 연결된다.
- **확장성**: 새 알고리즘을 전체 도구에 억지로 끼우지 않고 modifier 하나로 추가하는 방향이다.

`mini-compressor`가 llm-compressor에서 가장 잘 가져온 부분은 `BaseModifier`와 `Compressor`의 lifecycle이다. `initialize -> calibrate -> finalize` 흐름은 작은 프로젝트에도 매우 큰 설계상 이점이 있다. hook 등록, calibration forward, observer 제거 같은 작업을 알고리즘별로 격리할 수 있기 때문이다.

### Quark와 llm-compressor의 핵심 차이

| 관점 | AMD Quark | llm-compressor | mini-compressor의 위치 |
|---|---|---|---|
| 주된 관점 | 하드웨어 배포와 export | recipe 기반 LLM compression | 학습 가능한 compact POC |
| 사용자 진입점 | Quantizer/config/export flow | `oneshot`, recipe, modifier | `Compressor.from_recipe()` |
| 모델 변경 방식 | quantized module로 in-place replacement | modifier가 모델을 변형 | `nn.Linear -> FakeQuantLinear` |
| 알고리즘 조합 | config와 quantizer에 통합 | modifier list로 조합 | recipe registry가 modifier list 반환 |
| 저장/배포 | ONNX, safetensors 등 backend 고려 | HF/유사 compressed-tensors 흐름 | `quantization_config.json` + safetensors |
| 강한 장점 | backend-aware, deployment-ready | composable, LLM-focused | 구조가 작고 읽기 쉬움 |
| 약한 부분 | 내부 구조 학습은 부담 | framework abstraction이 큼 | real packing/runtime kernel 없음 |

학생이 이 프로젝트에서 배워야 할 점은 "Quark와 llm-compressor 중 하나를 흉내냈다"가 아니다. 더 정확히는 다음과 같다.

- Quark처럼 `nn.Linear`를 실제 양자화 모듈로 교체하는 모델 구조 변환을 채택했다.
- llm-compressor처럼 알고리즘을 modifier로 분리하고 recipe로 조합했다.
- 대형 프레임워크가 숨기는 scale 계산, observer, hook, save/load 문제를 작은 코드로 노출했다.
- 대신 production tool이 제공하는 real INT packing, backend-specific kernel, 대규모 모델 template, 완전한 schema compatibility는 의도적으로 줄였다.

## 파일별 상세 해설

### 1. `mini_compressor/schemes.py`

이 파일은 양자화 정책을 표현하는 가장 작은 단위다. `QuantizationSpec`은 weight나 activation 하나의 양자화 방식을 표현하고, `QuantizationScheme`은 weight spec과 activation spec을 묶어 하나의 recipe-level scheme으로 만든다.

핵심 구조:

- `QuantizationSpec`
  - `num_bits`: 4bit, 8bit 등 bit-width
  - `dtype`: 현재는 주로 `"int"`, float8은 stub 수준
  - `symmetric`: symmetric quantization 여부
  - `granularity`: `per_tensor`, `per_channel`, `per_group`, `per_token`
  - `group_size`: per-group weight quantization에서 사용
  - `dynamic`: activation scale을 calibration이 아니라 forward마다 계산할지
  - `calibration_method`: `minmax`, `percentile`, `mse`
- `QuantizationScheme`
  - `name`
  - `weight`
  - `activation`
- registry
  - `W8A8`
  - `W4A16`
  - `W8A8_DYNAMIC`
  - `SCHEME_REGISTRY`

Quark와 비교하면, 이 파일은 Quark의 quantization configuration을 매우 작은 dataclass로 축소한 형태다. Quark는 backend, export, algorithm config, dtype support까지 더 넓은 configuration surface를 갖지만, 이 프로젝트는 학생이 반드시 이해해야 하는 최소 축만 남겼다. 즉 "몇 bit인가", "weight와 activation 중 무엇을 양자화하는가", "scale 단위는 무엇인가", "calibration이 필요한가"가 핵심이다.

llm-compressor와 비교하면, 이 파일은 recipe 안에서 modifier가 참조하는 scheme 역할에 가깝다. 다만 llm-compressor는 YAML/JSON recipe와 더 복잡한 target group, compressed-tensors schema를 가진다. `mini-compressor`는 Python dataclass로 바로 읽히게 만들었다.

이 설계의 장점:

- frozen dataclass라 설정이 중간에 바뀌지 않는다.
- weight spec과 activation spec이 같은 타입을 공유한다.
- `W4A16`, `W8A8`, `W8A8_DYNAMIC`이 코드에서 명시적으로 보인다.
- observer, fake quant, serialize가 모두 같은 spec을 읽기 때문에 설정 흐름이 추적 가능하다.

trade-off:

- 문자열 기반 필드가 많아 잘못된 값이 런타임까지 갈 수 있다.
- `calibration_method`가 spec 안에 있어 weight와 activation 통계 정책은 표현되지만, calibration dataset 정책과 lifecycle 정책은 표현하지 못한다.
- `SCHEME_REGISTRY`는 단순 name registry라 hybrid quantization처럼 layer별로 다른 scheme을 주는 데 부족하다.
- KV cache quantization, output activation quantization, FP8 format 세부 선택, packing format은 아직 표현하지 않는다.

개선한다면:

- `Literal` 또는 `Enum`으로 `granularity`, `dtype`, `calibration_method`를 엄격히 제한한다.
- `CalibrationSpec`을 분리해서 percentile 값, MSE grid 수, sample 수를 설정으로 둔다.
- `KVCacheQuantizationSpec`을 추가해 K/V별 dtype, granularity, dynamic/static 여부를 표현한다.
- `HybridQuantizationScheme`을 만들어 layer name pattern별 scheme을 지정한다.
- schema version을 두어 저장 포맷과 code config가 함께 진화하게 만든다.

학생이 해볼 실습:

- `W4A8` scheme을 추가하고 observer/fake quant가 어디까지 동작하는지 확인한다.
- `W8A8_DYNAMIC`에서 activation granularity를 `per_tensor`로 바꾸면 어떤 forward scale이 계산되는지 추적한다.
- `group_size=64`로 바꿨을 때 테스트가 왜 통과하거나 실패하는지 본다.

### 2. `mini_compressor/fake_quant_linear.py`

이 파일은 실제로 모델 안에 들어가는 양자화 모듈이다. `FakeQuantLinear`는 `nn.Linear`를 상속하고, weight와 activation을 fake quantize한 뒤 `F.linear()`를 호출한다.

핵심 흐름:

1. `from_float()`가 기존 `nn.Linear`의 weight와 bias를 가져온다.
2. `weight_scale`, `weight_zero_point`, `input_scale`, `input_zero_point`를 buffer로 등록한다.
3. static activation quantization이면 `input_observer`를 만든다.
4. forward에서 weight fake quantization을 수행한다.
5. calibration 중이면 observer가 입력 activation을 update한다.
6. dynamic activation이면 forward마다 activation scale을 계산한다.
7. static activation이면 calibration 이후 저장된 `input_scale`로 activation fake quantization을 수행한다.

Quark와 비교하면, 이 파일은 Quark의 quantized module replacement 개념을 가장 직접적으로 보여준다. float module을 그대로 둔 채 외부에서 tensor만 바꾸는 것이 아니라, 모델 내부 module class가 바뀐다. 이 방식은 export나 backend mapping을 생각할 때 중요하다.

llm-compressor와 비교하면, 이 파일은 modifier가 바꿔 끼울 quantized layer 역할이다. llm-compressor의 production 구현은 실제 quantization metadata, compressed tensor, kernel compatibility를 더 엄격히 다루지만, 여기서는 fake quant math가 코드 안에 드러나 있어 학습에 좋다.

이 설계의 장점:

- `nn.Linear`와 같은 인터페이스라 기존 forward graph에 자연스럽게 들어간다.
- scale이 없으면 원래 FP output과 동일하게 동작한다.
- weight-only, static activation, dynamic activation을 하나의 class에서 처리한다.
- observer는 calibration 기간에만 존재하고 finalize 후 제거된다.
- per-channel, per-group, per-token dynamic activation을 실제 shape 변환으로 확인할 수 있다.

trade-off:

- `FakeQuantLinear`는 packed INT module이 아니다. weight는 여전히 float tensor이고 quantization error만 시뮬레이션한다.
- `input_scale=None`이 여러 의미를 가진다. static calibration 전, dynamic activation, weight-only 상태를 모두 표현한다.
- subclassing `nn.Linear`는 편하지만, production에서는 wrapper나 backend-specific module이 더 명확할 수 있다.
- float8 branch는 설명만 있고 `NotImplementedError`다.
- per-group weight는 `in_features % group_size == 0`을 전제로 한다.

개선한다면:

- fake quant module과 packed quant module을 분리한다.
- `input_scale`의 상태를 enum이나 explicit flag로 표현한다.
- FP8 fake quant를 실제 PyTorch float8 cast 기반으로 구현한다.
- group remainder를 허용할지, 엄격히 금지할지 전체 코드에서 일관되게 정한다.
- `extra_repr()`를 구현해 출력 시 scheme과 scale 상태가 보이게 한다.

학생이 해볼 실습:

- `FakeQuantLinear.from_float()` 전후로 `state_dict()` key가 어떻게 바뀌는지 확인한다.
- `weight_scale=None`일 때와 scale이 있을 때 output MSE를 비교한다.
- `W8A8_DYNAMIC`에서 token별 scale이 실제로 다른지 breakpoint로 본다.

### 3. `mini_compressor/observer.py`

observer는 calibration 데이터에서 quantization range를 추정하는 모듈이다. weight observer는 보통 한 번 update하고, activation observer는 calibration forward마다 update한다.

현재 구현된 observer:

- `MinMaxObserver`
  - 단위별 min/max를 모은다.
  - multi-GPU에서는 `all_reduce(MIN/MAX)`로 정확히 병합 가능하다.
- `PercentileObserver`
  - raw data를 모아 percentile clipping range를 계산한다.
  - outlier에 덜 민감하다.
- `MSEObserver`
  - clip range를 grid search하여 fake quant reconstruction MSE가 낮은 range를 찾는다.
  - percentile보다 비싸지만 RTN scale 선택의 질을 높일 수 있다.

핵심 helper:

- `_to_units()`
  - `per_tensor`, `per_channel`, `per_group`을 같은 통계 계산 형태로 바꾼다.
- `_scale_zp_from_range()`
  - min/max를 scale과 zero-point로 바꾼다.
- `_sync_data()`
  - percentile/MSE처럼 raw data가 필요한 observer를 분산 환경에서 동기화한다.

Quark와 비교하면, observer는 calibration config의 실제 실행부에 해당한다. Quark 같은 도구는 사용자에게 observer 세부를 덜 노출하지만, 내부적으로는 calibration range 추정이 품질에 큰 영향을 준다. 이 프로젝트는 그 내부를 읽기 쉬운 형태로 보여준다.

llm-compressor와 비교하면, modifier calibration 중 activation observer가 scale을 산출하는 구조가 유사하다. 다만 llm-compressor는 더 많은 scheme과 target group, distributed/session 처리, 저장 metadata를 통합한다.

이 설계의 장점:

- weight와 activation이 같은 observer abstraction을 공유한다.
- granularity별 통계 처리 로직이 `_to_units()`에 모여 있다.
- minmax는 결합적 통계라 분산 동기화가 효율적이다.
- percentile/MSE는 raw data all-gather로 정확성을 우선한다.

trade-off:

- Percentile/MSE는 raw tensor를 CPU list에 저장하므로 calibration sample이 커지면 메모리를 많이 쓴다.
- 현재 KL divergence observer는 없다.
- empty data에 대한 방어가 observer 내부보다는 호출자 쪽에 의존한다.
- per-token observer는 static observer 관점에서는 지원하지 않고 dynamic forward 경로에서 처리된다.

개선한다면:

- histogram 기반 KL observer를 추가한다.
- percentile/MSE의 raw data 저장을 histogram, reservoir sampling, t-digest 같은 요약 구조로 바꾼다.
- observer별 메모리 사용량과 sample 수를 리포트한다.
- observer state serialization을 지원해 calibration 재사용이 가능하게 한다.
- 분산 환경에서 raw all-gather 대신 histogram merge를 지원한다.

학생이 해볼 실습:

- 같은 weight에 대해 minmax, percentile, mse scale을 비교한다.
- outlier 하나를 추가했을 때 W8A8 output error가 어떻게 바뀌는지 본다.
- `test_observer_sync.py`를 읽고 왜 minmax와 percentile/MSE의 sync 방식이 다른지 설명해본다.

### 4. `mini_compressor/modifiers/base.py`

이 파일은 모든 modifier가 따라야 하는 최소 인터페이스를 정의한다.

세 단계 lifecycle:

- `initialize(model)`
  - 모델 구조를 바꾸거나 hook을 등록한다.
- `calibrate(dataloader, num_samples)`
  - calibration 데이터를 사용해 통계 수집, scale 계산, 알고리즘 적용을 수행한다.
- `finalize()`
  - hook, observer, 임시 상태를 정리한다.

llm-compressor와 가장 직접적으로 닮은 부분이다. 알고리즘이 많아질수록 `compress()` 하나에 if문을 쌓는 방식은 유지보수가 어렵다. `BaseModifier`를 두면 새 알고리즘을 파일 하나와 class 하나로 확장할 수 있다.

Quark와 비교하면, Quark의 quantizer 내부 phase를 더 작은 Python interface로 드러낸 형태다. Quark는 사용자가 phase를 직접 만지지 않아도 되게 더 통합되어 있지만, 학습용으로는 이 interface가 훨씬 명확하다.

이 설계의 장점:

- 새 알고리즘 추가 지점이 명확하다.
- calibration이 필요한 알고리즘과 필요 없는 알고리즘을 같은 pipeline에 넣을 수 있다.
- hook cleanup을 `finalize()`로 통일할 수 있다.

trade-off:

- state machine이 없어 `initialize()` 전 `calibrate()` 같은 misuse를 각 modifier가 직접 막아야 한다.
- modifier 간 dependency를 표현하지 않는다. 예를 들어 AWQ 다음 QuantizationModifier가 와야 한다는 규칙은 recipe discipline에 의존한다.
- `finalize()` 실패나 중복 호출 정책이 명확하지 않다.

개선한다면:

- `initialized`, `calibrated`, `finalized` 상태를 BaseModifier에서 관리한다.
- modifier dependency validation을 추가한다.
- modifier별 리포트 객체를 반환하게 한다.
- `dry_run(model)`으로 어떤 module이 바뀔지 미리 보여준다.

### 5. `mini_compressor/modifiers/__init__.py`

이 파일은 `modifiers` package에서 외부로 노출할 class를 모아 re-export한다.

현재 노출 항목:

- `BaseModifier`
- `QuantizationMixin`
- `QuantizationModifier`
- `SmoothQuantModifier`
- `GPTQModifier`
- `AWQModifier`

이 파일은 구현 로직은 거의 없지만, library 설계에서는 중요하다. 사용자가 `mini_compressor.modifiers.quantization` 같은 내부 경로를 기억하지 않아도 `from mini_compressor.modifiers import GPTQModifier`처럼 import할 수 있게 만든다.

llm-compressor와 비교하면 modifier namespace를 정리하는 작은 API layer다. production library에서는 어떤 class가 public API이고 어떤 helper가 internal인지 명확히 해야 한다. 이 프로젝트에서는 학습 편의상 `QuantizationMixin`까지 노출되어 있는데, 실제 배포 library라면 mixin은 내부 구현으로 숨길지 고민할 수 있다.

Quark와 비교하면 사용자가 직접 algorithm class를 import해 조합하는 surface가 더 열려 있다. Quark는 high-level quantizer/config 사용성이 더 강하고, 이 프로젝트는 modifier 조합 실험성이 더 강하다.

이 설계의 장점:

- modifier 추가 시 package-level import 경로가 정리된다.
- `__all__`이 있어 의도한 public symbol이 명확하다.
- `recipes.py`와 외부 사용자 코드가 짧은 import path를 쓸 수 있다.

trade-off:

- mixin까지 public API처럼 보일 수 있다.
- modifier 수가 늘어나면 import side effect나 dependency 비용을 관리해야 한다.
- helper인 `_pair_utils.py`는 의도적으로 노출하지 않는데, 이것이 public/private boundary를 보여준다.

개선한다면:

- `QuantizationMixin`을 public export에서 제거할지 검토한다.
- experimental modifier는 별도 namespace로 분리한다.
- package-level docstring에 modifier별 사용 목적을 짧게 적는다.

### 6. `mini_compressor/modifiers/quantization.py`

이 파일은 프로젝트의 중심이다. `QuantizationMixin`은 `nn.Linear -> FakeQuantLinear` 교체 로직을 제공하고, `QuantizationModifier`는 RTN 기반 scale 계산과 activation calibration을 수행한다.

핵심 구조:

- `QuantizationMixin`
  - `_should_replace()`: target/ignore pattern과 class name을 보고 교체 여부 결정
  - `initialize()`: target `nn.Linear`를 `FakeQuantLinear`로 교체
  - weight observer로 weight scale 계산
  - group size divisibility 검증
  - `finalize()`: observer 제거
- `QuantizationModifier`
  - `calibrate()`: static activation quantization을 위한 forward pass
  - `sequential=True`: `model.model.layers` 구조에서 layer-wise calibration

Quark와 비교하면, in-place module replacement와 scale 산출이 Quark식 quantizer flow와 닮았다. 특히 target module을 찾아 실제 module을 바꾼다는 점이 중요하다.

llm-compressor와 비교하면, `QuantizationMixin`이라는 이름 그대로 modifier 간 공통 module replacement를 공유하는 방식이 닮았다. `GPTQModifier`도 이 mixin을 사용해 replacement는 공유하고 calibration만 다르게 구현한다.

이 설계의 장점:

- RTN과 GPTQ가 같은 replacement 코드를 공유한다.
- `targets`, `ignore`가 pattern 기반이라 사용자 제어가 가능하다.
- weight scale은 initialize 시점에 바로 계산하므로 weight-only quantization은 calibration 없이 끝난다.
- static activation quantization은 observer를 통해 calibration 이후 scale을 확정한다.
- sequential calibration은 큰 모델에서 peak memory를 줄이기 위한 현실적인 고민을 담고 있다.

trade-off:

- `dataloader`를 외부에서 list로 만들거나 `Compressor.compress()`에서 list화하기 때문에 streaming dataloader 장점이 줄어든다.
- sequential calibration은 `model.model.layers` 구조에 강하게 의존한다.
- layer 입력을 잡기 위해 forward monkey patch와 `_Abort` 예외를 사용한다. 실용적이지만 fragile하다.
- replacement가 in-place라 원본 모델을 보존하려면 사용자가 deepcopy해야 한다.
- target/ignore dry-run report가 없어 실수로 너무 많은 layer를 바꿔도 바로 알기 어렵다.

개선한다면:

- target matching 결과를 표로 반환하는 `plan()` 또는 `dry_run()`을 추가한다.
- sequential calibration adapter를 모델 architecture별로 분리한다.
- `model.model.layers` 외에 `transformer.h`, `gpt_neox.layers` 같은 구조를 지원한다.
- dataloader를 list로 고정하지 않고 재사용 가능한 calibration cache abstraction을 둔다.
- group remainder 처리 정책을 `FakeQuantLinear`, `AWQ`, validation에서 일관화한다.

학생이 해볼 실습:

- `ignore=["lm_head"]`가 name match로 어떻게 동작하는지 확인한다.
- `targets=["Linear"]`와 `targets=["*.q_proj"]`의 차이를 실험한다.
- sequential calibration과 full calibration의 scale이 왜 같은지 `test_sequential_calib.py`를 따라가며 설명한다.

### 7. `mini_compressor/compressor.py`

`Compressor`는 사용자가 직접 만나는 one-click entrypoint다. modifier list를 받아 `initialize -> calibrate -> finalize`를 순서대로 실행한다.

핵심 API:

- `Compressor(modifiers)`
  - 직접 modifier list를 넣는 방식
- `Compressor.from_recipe(name, targets, ignore)`
  - registry 이름으로 modifier pipeline 생성
- `compress(model, dataloader, num_samples)`
  - 각 modifier를 순차 실행
- `save(model, save_dir, tokenizer)`
  - quantization modifier를 찾아 저장
- `save_to_hub(...)`
  - HF Hub 업로드와 모델 카드 생성

llm-compressor와 비교하면 `oneshot`의 축소판이다. llm-compressor의 `oneshot`은 모델 로딩, recipe 적용, calibration, 저장까지 큰 workflow를 제공한다. 이 프로젝트에서는 모델 로딩은 사용자가 하고, compression lifecycle과 save를 `Compressor`가 담당한다.

Quark와 비교하면 `ModelQuantizer.quantize_model()` 같은 high-level quantizer entrypoint의 작은 버전이다. 다만 backend export나 hardware validation은 없다.

이 설계의 장점:

- 사용자가 내부 modifier lifecycle을 몰라도 `from_recipe().compress()`로 시작할 수 있다.
- `try/finally`로 calibration 중 예외가 나도 `finalize()`가 호출된다.
- direct modifier list를 허용해 실험성이 좋다.
- `save_to_hub()`까지 있어 artifact publishing 흐름을 보여준다.

trade-off:

- `compress()`가 `dataloader`를 `list()`로 물리화한다. 큰 calibration dataset에서는 메모리 부담이 된다.
- `save()`는 첫 번째 `QuantizationMixin`만 찾는다. 미래에 hybrid/multiple quant modifier가 생기면 불충분하다.
- `_write_model_card()` 안의 `_pipeline` 참조는 현재 실제로 채워지는 구조가 아니므로 확장 여지가 있다.
- 모델 로드, tokenizer, dataset preprocess까지 통합하지는 않는다.

개선한다면:

- `CompressionReport`를 반환해 교체 layer 수, scale shape, calibration sample 수, recipe 정보를 담는다.
- dataloader materialization을 선택 사항으로 바꾼다.
- modifier dependency/order validation을 추가한다.
- multiple quantization groups를 저장할 수 있게 `save()`를 확장한다.
- `from_recipe()`가 Python registry뿐 아니라 YAML/JSON recipe를 받을 수 있게 한다.

### 8. `mini_compressor/recipes.py`

recipe는 사람이 쓰기 쉬운 이름을 modifier pipeline으로 바꾸는 파일이다.

현재 registry:

- `"w4a16"` -> `QuantizationModifier(W4A16)`
- `"w4a16_gptq"` -> `GPTQModifier(W4A16)`
- `"w4a16_awq"` -> `AWQModifier` 후 `QuantizationModifier(W4A16)`
- `"w8a8"` -> `QuantizationModifier(W8A8)`
- `"w8a8_dynamic"` -> `QuantizationModifier(W8A8_DYNAMIC)`
- `"w8a8_smoothquant"` -> `SmoothQuantModifier` 후 `QuantizationModifier(W8A8_DYNAMIC)`

llm-compressor와 비교하면 이 파일은 recipe system의 아주 작은 Python 버전이다. llm-compressor는 YAML/JSON recipe, modifier parameter, order validation, session integration이 있지만, 이 프로젝트는 dictionary registry 하나로 학습 난도를 낮췄다.

Quark와 비교하면 Quark의 preset/config recipe와도 닮았다. 다만 Quark는 backend와 export format까지 config에 연결되는 반면, 이 파일은 modifier pipeline만 표현한다.

이 설계의 장점:

- recipe 이름만 보면 어떤 알고리즘 조합인지 알 수 있다.
- modifier는 상태를 가지므로 factory가 매번 새 인스턴스를 만드는 점이 좋다.
- SmoothQuant/AWQ처럼 "전처리 알고리즘 + 최종 quantization" 조합을 자연스럽게 표현한다.

trade-off:

- Python code를 수정해야 recipe를 추가할 수 있다.
- parameter override가 제한적이다. 예를 들어 AWQ `n_grid`, SmoothQuant `alpha`를 recipe call에서 바꾸기 어렵다.
- layer별 hybrid quantization 정책은 표현하지 못한다.

개선한다면:

- YAML recipe parser를 추가한다.
- modifier parameter override를 지원한다.
- recipe validation을 넣어 AWQ/SmoothQuant 뒤에 QuantizationModifier가 없으면 경고한다.
- hybrid recipe를 추가해 attention은 W8A8, MLP는 W4A16 같은 구성을 표현한다.

### 9. `mini_compressor/modifiers/_pair_utils.py`

이 파일은 SmoothQuant와 AWQ가 공유하는 norm-linear pair 탐색 유틸이다.

현재 탐색 규칙:

- `input_layernorm + self_attn.{q_proj,k_proj,v_proj}`
- `post_attention_layernorm + mlp.{gate_proj,up_proj}`
- affine weight가 있는 norm만 대상으로 한다.

Quark와 비교하면 architecture template의 가장 작은 형태다. Quark 같은 production tool은 모델 타입별 template이나 graph pattern을 더 체계적으로 가진다. 여기서는 Qwen/LLaMA 계열 decoder block naming convention을 직접 사용한다.

llm-compressor와 비교하면 target pattern과 model-specific modifier support의 축소판이다. llm-compressor는 더 다양한 모델과 modifier target 설정을 다룰 수 있지만, 이 프로젝트는 SmoothQuant/AWQ에 필요한 핵심 pair만 추출한다.

이 설계의 장점:

- SmoothQuant와 AWQ가 pair discovery를 중복 구현하지 않는다.
- 코드가 짧아 어떤 구조를 지원하는지 명확하다.
- Qwen/LLaMA 계열 핵심 projection을 바로 이해할 수 있다.

trade-off:

- architecture-specific naming에 강하게 묶여 있다.
- `o_proj`, `down_proj`는 pair에 포함하지 않는다. 알고리즘상 앞 norm과 직접 이어지는 projection만 다루려는 선택이다.
- GPT-2, OPT, Falcon, Mixtral/MoE 등은 추가 rule이 필요하다.

개선한다면:

- model config의 `model_type`별 pair template registry를 둔다.
- 사용자 정의 pair map을 받는다.
- 발견된 pair를 리포트로 출력한다.
- unsupported model에서 no-op이 아니라 informative warning을 선택적으로 제공한다.

### 10. `mini_compressor/modifiers/smoothquant.py`

SmoothQuant는 activation outlier를 weight 쪽으로 옮겨 activation quantization을 쉽게 만드는 알고리즘이다.

핵심 아이디어:

```text
y = x @ W.T
  = (x / s) @ (W * s).T
```

양자화 전에는 수학적으로 같은 연산이지만, activation range가 줄어들면 W8A8 activation quantization이 더 안정적일 수 있다.

현재 구현 흐름:

1. `_find_smooth_pairs()`로 norm과 linear group을 찾는다.
2. 첫 번째 linear에 forward pre-hook을 걸어 입력 activation의 channel-wise abs max를 수집한다.
3. calibration forward를 수행한다.
4. `x_max`와 `w_max`로 smooth factor `s`를 계산한다.
5. `norm.weight`, `norm.bias`를 `s`로 나눈다.
6. 연결된 linear weight에 `s`를 곱한다.
7. hook과 통계를 제거한다.

llm-compressor와 비교하면 SmoothQuant를 독립 modifier로 구현한 점이 유사하다. 그리고 recipe에서 `SmoothQuantModifier -> QuantizationModifier` 순서로 조합한다는 점이 중요하다.

Quark와 비교하면 calibration 기반 preprocessing algorithm이 quantization pipeline 앞단에 들어가는 구조가 닮았다. 다만 Quark는 backend/export와 결합된 더 큰 flow 안에서 이런 알고리즘을 다룬다.

이 설계의 장점:

- 양자화 전 equivalent transform을 테스트로 보장한다.
- LayerNorm bias까지 나누는 점이 좋다. bias를 빼먹으면 LayerNorm bias가 있는 모델에서 output equivalence가 깨진다.
- hook 등록과 해제가 modifier lifecycle에 잘 들어가 있다.

trade-off:

- `alpha`가 고정 parameter이고 자동 탐색은 없다.
- activation 통계는 max 기반이라 outlier에 민감할 수 있다.
- pair discovery가 Qwen/LLaMA naming에 의존한다.
- multi-GPU activation 통계 sync는 별도로 구현되어 있지 않다.

개선한다면:

- layer별 `alpha` search를 추가한다.
- max 대신 percentile activation 통계를 옵션으로 둔다.
- pair discovery를 architecture registry로 확장한다.
- SmoothQuant 적용 전후 activation range report를 만든다.
- W8A8 static과 dynamic 각각에서 실제 PPL 영향을 비교한다.

학생이 해볼 실습:

- `test_smoothquant.py`의 bias test를 지우고 어떤 회귀가 생기는지 생각해본다.
- `alpha=0.0`, `0.5`, `1.0`에서 weight range와 activation range가 어떻게 바뀌는지 출력한다.
- SmoothQuant만 적용했을 때 output이 거의 같고, quantization까지 적용하면 error가 달라지는 이유를 설명한다.

### 11. `mini_compressor/modifiers/gptq.py`

GPTQ는 weight-only low-bit quantization에서 RTN보다 낮은 reconstruction error를 얻기 위한 Hessian 기반 PTQ 알고리즘이다.

현재 구현 흐름:

1. `QuantizationMixin`으로 `nn.Linear`를 `FakeQuantLinear`로 교체한다.
2. weight scale은 initialize에서 계산하지 않는다. `compute_scales=False`다.
3. 각 `FakeQuantLinear`에 forward pre-hook을 걸어 입력 `X`를 수집하고 Hessian 근사 `H += 2 X^T X`를 누적한다.
4. calibration forward를 수행한다.
5. hook을 제거한다.
6. 각 layer별로 `_gptq_quantize()`를 실행한다.
7. Cholesky inverse를 이용해 column-wise quantization error를 뒤 column에 전파한다.
8. 최종 fake-quantized weight, per-group scale, zero-point를 module에 저장한다.

llm-compressor와 비교하면 `GPTQModifier`가 `QuantizationMixin`을 공유하고 calibration만 다르게 구현하는 구조가 특히 닮았다. GPTQ가 단순 scheme이 아니라 algorithm modifier라는 점을 잘 보여준다.

Quark와 비교하면 GPTQ는 Quark가 제공할 수 있는 PTQ 알고리즘 중 하나와 개념적으로 대응한다. 다만 Quark에서는 결과가 backend/export 경로까지 이어지는 반면, 여기서는 fake quantized module로 남는다.

이 설계의 장점:

- RTN과 같은 `FakeQuantLinear` 결과물을 사용하므로 후속 save/forward가 단순하다.
- hook cleanup이 `try/finally`로 되어 있어 calibration 실패 시에도 hook 누수를 줄인다.
- dead column, dampening, Cholesky fallback을 다룬다.
- 테스트에서 RTN 대비 MSE 개선을 검증한다.

trade-off:

- Hessian이 `in_features x in_features`라 layer가 커질수록 메모리 비용이 크다.
- block-wise GPTQ, act-order, true sequential layer-wise GPTQ 같은 production 최적화는 없다.
- per-group weight만 지원한다.
- dataloader를 list로 만든다.
- 실제 packed INT4 weight가 아니라 dequantized fake weight로 저장한다.

개선한다면:

- block size 기반 GPTQ를 구현해 메모리와 속도를 제어한다.
- activation order 옵션을 추가한다.
- sequential GPTQ를 추가해 큰 모델에서 layer별로 GPU 메모리를 관리한다.
- calibration sample 수와 Hessian rank condition을 리포트한다.
- GPTQ 후 layer별 reconstruction error를 출력한다.

학생이 해볼 실습:

- `test_gptq_mse_leq_rtn()`에서 calibration sample 수를 줄이면 GPTQ가 왜 불안정해질 수 있는지 확인한다.
- dampening fraction을 바꿔 Cholesky 안정성과 MSE 변화를 본다.
- per-channel scheme으로 GPTQ를 호출하면 왜 에러를 내는지 설명한다.

### 12. `mini_compressor/modifiers/awq.py`

AWQ는 activation magnitude를 보고 중요한 channel의 weight quantization error를 줄이는 방향으로 scale을 찾는 알고리즘이다.

현재 구현 흐름:

1. SmoothQuant와 같은 `_find_smooth_pairs()`를 사용한다.
2. 첫 번째 linear 입력 activation의 channel-wise mean absolute value를 누적한다.
3. `alpha`를 grid search한다.
4. 각 후보 scale에 대해 INT4 fake quant error를 계산한다.
5. activation magnitude로 가중한 quantization error가 가장 낮은 scale을 고른다.
6. `norm.weight`, `norm.bias`를 나누고 linear weight에 scale을 곱한다.
7. recipe에서는 이후 `QuantizationModifier(W4A16)`이 실제 RTN fake quantization을 수행한다.

SmoothQuant와의 차이:

- SmoothQuant는 max activation과 weight max를 사용하고 `alpha`를 보통 고정한다.
- AWQ는 mean activation magnitude를 사용하고 alpha grid search로 quantization error를 직접 본다.
- SmoothQuant는 W8A8 activation outlier 완화 성격이 강하고, AWQ는 W4A16 weight-only 정확도 개선 성격이 강하다.

Quark/llm-compressor와 비교하면, 이 파일은 algorithm modifier를 가볍게 구현한 좋은 예다. production AWQ는 더 복잡한 salient weight 보존, block/group 처리, reconstruction metric을 사용하지만, 여기서는 "activation-aware scaling을 modifier로 pipeline 앞에 둔다"는 구조가 핵심이다.

이 설계의 장점:

- SmoothQuant와 같은 pair discovery를 재사용한다.
- 양자화 전 equivalent transform 테스트가 있다.
- alpha grid search가 있어 고정 alpha보다 알고리즘 의도가 잘 보인다.
- W4A16 quantization과 분리되어 있어 AWQ 자체와 최종 quantization을 따로 이해할 수 있다.

trade-off:

- `_int4_fake_quant()`는 내부 helper에서 `cols % group_size != 0`이면 전체 cols를 group으로 쓰는 fallback이 있다. 반면 `QuantizationMixin`은 group_size divisibility를 엄격히 요구한다. 내부 grid search용이라 큰 문제는 아니지만 정책 일관성은 아쉽다.
- full AWQ 구현이라기보다 핵심 아이디어 POC에 가깝다.
- pair discovery가 제한적이다.
- PPL 기반 검증이나 실제 large model 검증은 demo 옵션에만 의존한다.

개선한다면:

- group_size fallback 정책을 전체 프로젝트와 맞춘다.
- salient weight protection을 더 명시적으로 구현한다.
- layer별 best alpha를 저장하고 리포트한다.
- AWQ 후 QuantizationModifier가 기대한 scale shape과 error를 검증한다.
- MoE expert별 AWQ scaling을 지원한다.

학생이 해볼 실습:

- `n_grid=1`, `5`, `20`에서 best error와 속도를 비교한다.
- activation 분포를 인위적으로 한 channel에 몰아넣고 best scale이 어떻게 변하는지 본다.
- AWQ만 적용했을 때 output equivalence가 유지되는 이유를 수식으로 설명한다.

### 13. `mini_compressor/serialize.py`

이 파일은 quantized model artifact를 저장하고 다시 로드하는 역할을 한다.

핵심 저장 흐름:

1. `model.save_pretrained(save_dir)`로 safetensors를 저장한다.
2. `_scheme_to_dict()`로 `quantization_config.json`을 만든다.
3. tokenizer가 있으면 함께 저장한다.
4. `base_model_name_or_path`를 기록한다.

핵심 로드 흐름:

1. `quantization_config.json`을 읽어 scheme과 ignore를 복원한다.
2. base model을 `AutoModelForCausalLM.from_pretrained()`로 다시 로드한다.
3. `QuantizationModifier(..., compute_scales=False)`로 구조만 `FakeQuantLinear`로 바꾼다.
4. 저장된 safetensors를 읽는다.
5. parameter는 `data.copy_()`, buffer는 `_buffers`에 직접 주입한다.

llm-compressor와 비교하면 `compressed-tensors` 계열 metadata를 매우 단순하게 흉내낸 형태다. llm-compressor는 vLLM과 더 엄격히 연결되는 format compatibility가 중요하지만, 여기서는 `quantization_config.json`이 어떤 정보를 가져야 하는지 학습하는 데 초점이 있다.

Quark와 비교하면 export 단계의 축소판이다. Quark는 배포 가능한 format과 backend compatibility가 더 중요하지만, 이 프로젝트는 HF 저장/로드 round-trip을 우선한다.

이 설계의 장점:

- 저장 포맷이 눈으로 읽히는 JSON이다.
- strategy name을 `tensor`, `channel`, `group`, `token`으로 바꿔 metadata 친화적으로 저장한다.
- `load_state_dict()`가 None buffer를 복원하지 못하는 문제를 직접 우회한다.
- `from_config()`가 dtype을 틀리게 만들 수 있는 문제를 피하고 base model을 다시 로드한다.

trade-off:

- base model에 접근 가능해야 로드할 수 있다. 완전히 self-contained artifact는 아니다.
- 여러 config group이나 hybrid quantization을 표현하지 못한다.
- 실제 packed weight가 아니라 fake quantized float weight를 저장한다.
- `quantization_status`는 calibrated로 고정되어 세부 상태를 표현하지 않는다.

개선한다면:

- JSON schema validation을 추가한다.
- multiple config groups를 지원한다.
- packed INT4/INT8 export format을 추가한다.
- base model이 없을 때 config + saved state만으로 로드하는 fallback을 만든다.
- model card와 quantization config의 내용이 항상 일치하는지 테스트한다.

학생이 해볼 실습:

- `save_pretrained()` 결과의 `model.safetensors` key를 확인한다.
- `load_pretrained()`에서 `_buffers` 직접 주입을 `load_state_dict()`로 바꾸면 어떤 문제가 생길지 생각한다.
- `quantization_config.json`에서 `strategy`를 바꾸면 `_scheme_from_dict()`가 어떻게 반응하는지 본다.

### 14. `mini_compressor/__init__.py`

이 파일은 외부 사용자가 import할 public API를 정리한다.

노출 API:

- `Compressor`
- `QuantizationModifier`
- `GPTQModifier`
- `AWQModifier`
- `SmoothQuantModifier`
- `BaseModifier`
- `QuantizationMixin`
- `W8A8`, `W4A16`, `W8A8_DYNAMIC`
- `QuantizationScheme`, `QuantizationSpec`
- `save_pretrained`, `load_pretrained`

이 파일은 작지만 중요하다. library 사용자는 내부 파일 구조를 몰라도 `from mini_compressor import Compressor`로 시작할 수 있어야 한다.

llm-compressor와 비교하면 package-level import surface를 정리하는 역할이다. production library는 public/private API boundary가 더 엄격해야 하지만, 이 프로젝트는 학습과 실험성을 위해 modifier들도 바로 노출한다.

개선한다면:

- `__version__`을 추가한다.
- `QuantizationMixin`은 내부 구현에 가까우므로 public export에서 뺄지 고민한다.
- import 순환이나 heavy dependency가 늘어나면 lazy import를 고려한다.

### 15. `demo.py`

`demo.py`는 전체 도구를 사용자가 어떻게 쓰는지 보여주는 실행 예제다.

현재 데모 흐름:

1. FP16 baseline 생성
2. W4A16 RTN
3. W4A16 GPTQ
4. W8A8 static
5. W8A8 dynamic
6. W8A8 + SmoothQuant
7. optional save/load round-trip
8. optional Wikitext-2 perplexity 측정

이 파일은 내부 설계를 공부한 뒤 마지막에 읽는 것이 좋다. 처음부터 읽으면 `Compressor.from_recipe()`가 마법처럼 보이지만, 앞 파일들을 보고 오면 recipe, modifier, observer, save/load가 연결되는 것이 보인다.

Quark와 비교하면 end-to-end quantize/export example에 해당한다. llm-compressor와 비교하면 `oneshot` 사용 예제를 직접 Python 코드로 풀어 쓴 형태다.

장점:

- 여러 recipe를 같은 prompt로 비교한다.
- W4A16 저장과 `load_pretrained()` round-trip을 확인한다.
- `--ppl` 옵션으로 generate뿐 아니라 perplexity도 본다.
- GPTQ calibration dataset과 static W8A8 calibration dataset이 다르다는 점을 보여준다.

trade-off:

- 데모는 네트워크와 GPU memory에 민감하다.
- calibration sample이 작아 정확도 비교를 일반화하기 어렵다.
- generation text 비교는 정성 지표이므로 PPL이나 task metric이 함께 필요하다.
- CLI 옵션이 아직 제한적이다.

개선한다면:

- calibration dataset path와 sample 수를 CLI로 받는다.
- recipe를 CLI에서 선택하게 한다.
- 결과를 JSON/CSV 리포트로 저장한다.
- 모델 로드 dtype, device_map, max_memory를 옵션화한다.
- AWQ/GPTQ/SmoothQuant의 layer별 통계 요약을 출력한다.

## 테스트 파일 학습 순서

테스트는 이 프로젝트의 숨은 설계 문서다. 어떤 회귀를 두려워했는지 알 수 있기 때문이다.

### 1. `tests/test_fake_quant_linear.py`

가장 먼저 읽을 테스트다. `FakeQuantLinear`가 `nn.Linear`를 대체하면서 shape, FP fallback, dynamic activation이 제대로 동작하는지 검증한다.

배울 점:

- scale이 없으면 FP output과 같아야 한다.
- dynamic activation은 observer 없이 forward에서 scale을 계산한다.
- per-token scale은 token마다 달라질 수 있다.

### 2. `tests/test_modifier.py`

`QuantizationModifier`의 initialize/calibrate/finalize를 검증한다.

배울 점:

- `ignore=["lm_head"]`가 실제로 module replacement를 막는다.
- W4A16은 activation scale이 없다.
- W8A8은 calibration 후 input scale이 생긴다.
- finalize 후 observer가 제거된다.
- weight observer method가 weight scale에도 반영된다.

### 3. `tests/test_compressor.py`

사용자 entrypoint 관점의 테스트다.

배울 점:

- `from_recipe()`가 올바른 modifier chain을 만든다.
- `compress()`가 end-to-end replacement와 calibration을 수행한다.
- `save()`가 필요한 파일을 만든다.
- SmoothQuant가 pair 없는 모델에서 no-op이어도 pipeline이 깨지지 않는다.

### 4. `tests/test_serialize.py`

저장 포맷의 최소 계약을 검증한다.

배울 점:

- `quantization_config.json`에 어떤 key가 필요한지 알 수 있다.
- W8A8과 W4A16이 metadata에서 어떻게 다르게 표현되는지 보인다.
- `_scheme_to_dict()`와 `_scheme_from_dict()` round-trip이 중요하다.

### 5. `tests/test_smoothquant.py`

SmoothQuant의 equivalent transform을 검증한다.

배울 점:

- pair discovery가 어떤 구조를 찾는지 알 수 있다.
- LayerNorm bias를 함께 나눠야 output equivalence가 유지된다.
- SmoothQuant 후 QuantizationModifier chain이 동작한다.

### 6. `tests/test_awq.py`

AWQ의 scaling과 Quantization chain을 검증한다.

배울 점:

- AWQ도 quantization 전에는 equivalent transform이어야 한다.
- empty dataloader와 initialize 누락을 에러로 처리한다.
- pair 없는 모델에서는 no-op이어야 한다.

### 7. `tests/test_gptq.py`

GPTQ의 핵심 품질 테스트다.

배울 점:

- GPTQ도 `FakeQuantLinear`로 replacement한다.
- GPTQ 후 weight는 quantization grid 위에 있어야 한다.
- calibration metric 기준으로 RTN보다 MSE가 낮아야 한다.

### 8. `tests/test_sequential_calib.py`

메모리 절약형 calibration 설계를 이해하는 테스트다.

배울 점:

- full calibration과 sequential calibration이 같은 scale을 낼 수 있다.
- `model.model.layers` 구조가 없으면 명시적으로 에러를 낸다.
- 빈 dataloader는 calibration 불가능하다.

### 9. `tests/test_observer_sync.py`

분산 calibration 통계 동기화를 검증한다.

배울 점:

- minmax는 all-reduce로 충분하다.
- percentile/MSE는 raw data를 all-gather해야 같은 결과가 나온다.
- distributed가 초기화되지 않으면 sync는 no-op이어야 한다.

### 10. `tests/test_hub.py`

HF Hub 업로드 흐름을 mock으로 검증한다.

배울 점:

- 외부 서비스를 직접 호출하지 않고도 upload flow를 테스트할 수 있다.
- model card에 recipe, bit-width, compressed-tensors 같은 핵심 정보가 들어가야 한다.
- optional dependency가 없을 때 error message가 명확해야 한다.

## 이 프로젝트에서 특히 잘 살린 설계 철학

첫째, "양자화 알고리즘"과 "도구 구조"를 분리했다. RTN, GPTQ, AWQ, SmoothQuant가 모두 한 함수에 들어 있지 않고 modifier로 나뉘어 있다. 이것은 llm-compressor식 확장성의 핵심이다.

둘째, module replacement를 실제로 수행한다. 단순히 `weight = fake_quant(weight)`만 하는 notebook 코드가 아니라 모델 내부의 `nn.Linear`가 `FakeQuantLinear`로 바뀐다. 이것은 Quark식 quantizer/export 관점을 학습하기 좋다.

셋째, weight와 activation scale 계산을 observer abstraction으로 통일했다. minmax, percentile, mse가 같은 interface를 쓰므로 새로운 observer를 추가하기 쉽다.

넷째, calibration lifecycle의 지저분한 부분을 숨기지 않았다. hook 등록/해제, device 이동, sequential calibration, distributed sync, save/load buffer 문제 같은 실제 도구 개발에서 부딪히는 문제가 코드에 드러난다.

다섯째, 테스트가 단순 shape check를 넘어 설계 의도를 검증한다. SmoothQuant/AWQ equivalent transform, GPTQ MSE 개선, observer sync, serialization round-trip이 그 예다.

## 설계상 감수한 trade-off

이 프로젝트는 production quantization runtime이 아니라 학습 가능한 compressor다. 그래서 다음 trade-off가 있다.

- real INT4/INT8 packed weight와 kernel dispatch는 없다.
- fake quantization이므로 memory compression 효과는 제한적이다.
- 지원 architecture는 Qwen/LLaMA 계열 naming에 많이 의존한다.
- recipe는 Python registry라 YAML 기반 외부 설정보다 단순하다.
- `quantization_config.json`은 compressed-tensors 스타일을 따르지만 완전한 production compatibility를 보장하지 않는다.
- hybrid quantization, KV cache quantization, MoE, VLM, TP/PP sharding은 설계 논의 수준이다.

이 trade-off는 나쁜 것이 아니다. 작은 프로젝트에서는 핵심 추상화를 선명하게 보여주는 것이 더 중요할 수 있다. 다만 면접이나 코드 리뷰에서는 "무엇을 의도적으로 줄였고, production으로 가려면 무엇을 추가해야 하는가"를 명확히 말해야 한다.

## 추가 기능을 개선할 때 고려할 점

### 1. Hybrid quantization

hybrid quantization은 layer나 module type별로 다른 scheme을 적용하는 것이다.

예:

- attention projection은 W8A8 dynamic
- MLP projection은 W4A16 GPTQ
- `lm_head`는 FP16 유지
- 일부 outlier layer만 W8A16 유지

현재 구조에서 필요한 변화:

- `QuantizationScheme` 하나만 받는 `QuantizationModifier`로는 부족하다.
- target pattern별 scheme mapping이 필요하다.
- `serialize.py`의 `config_groups`를 여러 개로 저장해야 한다.
- `Compressor.save()`가 첫 번째 quant modifier만 보는 구조를 바꿔야 한다.

주의할 점:

- 서로 다른 scheme이 같은 module에 동시에 match되면 우선순위가 필요하다.
- GPTQ/AWQ/SmoothQuant 같은 algorithm modifier와 hybrid scheme의 순서를 검증해야 한다.
- 저장 포맷에서 group별 targets/ignore가 명확해야 한다.

### 2. KV cache quantization

KV cache quantization은 weight quantization과 성격이 다르다. weight는 모델 parameter지만 KV cache는 generation 중 동적으로 쌓이는 activation state다.

추가하려면:

- `FakeQuantLinear`가 아니라 attention module 또는 cache object를 감싸야 한다.
- K/V별 scale granularity를 정해야 한다. 예를 들어 per-token, per-head, per-channel 중 무엇을 쓸지 결정해야 한다.
- prefill과 decode 단계의 cache shape과 update path를 알아야 한다.
- scale을 cache와 함께 저장할지, block별로 저장할지 결정해야 한다.

Quark 관점에서는 KV cache quantization이 backend/runtime과 강하게 연결된다. llm-compressor 관점에서는 recipe metadata에 KV cache quantization config를 표현할 수 있어야 한다. `mini-compressor`에서 구현한다면 먼저 "fake KV cache quantization hook"으로 시작하고, 이후 실제 runtime cache layout으로 확장하는 것이 좋다.

### 3. Real INT packing

fake quantization 다음 단계는 packed weight다.

필요한 변화:

- `FakeQuantLinear`와 별도 `PackedQuantLinear`를 만든다.
- INT4는 두 값을 한 byte에 packing해야 한다.
- group scale과 zero-point layout을 kernel이 기대하는 형태로 저장해야 한다.
- forward는 PyTorch `F.linear()`가 아니라 custom kernel 또는 dequantize-on-the-fly path가 필요하다.

주의할 점:

- packed format은 저장 포맷과 runtime kernel이 함께 정해져야 한다.
- 단순 packing만으로 속도가 빨라지지는 않는다. kernel dispatch가 핵심이다.

### 4. Architecture template

현재 `_pair_utils.py`와 sequential calibration은 Qwen/LLaMA 계열에 가깝다.

확장하려면:

- `model.config.model_type` 기반 template registry를 둔다.
- layer list path, norm-linear pair, attention projection name, MLP projection name을 template로 표현한다.
- unsupported model에서는 어떤 rule이 실패했는지 알려준다.

### 5. 더 좋은 report와 debugging 도구

양자화 도구는 사용자가 "정말 내가 원한 layer가 바뀌었나?"를 빨리 확인해야 한다.

추가하면 좋은 report:

- 교체된 module 수
- ignore된 module 목록
- scheme별 target group
- layer별 scale shape
- observer method
- calibration sample 수
- SmoothQuant/AWQ scale range
- GPTQ reconstruction error
- save artifact 목록

## 학생을 위한 학습 과제

아래 순서대로 직접 수정해보면 프로젝트 이해도가 빠르게 올라간다.

1. `W4A8` scheme을 추가하고 테스트를 하나 작성한다.
2. `QuantizationModifier.plan(model)`을 만들어 교체 예정 module을 표로 출력한다.
3. `PercentileObserver`의 percentile 값을 `QuantizationSpec`에서 조절 가능하게 만든다.
4. `SmoothQuantModifier`에 layer별 alpha report를 추가한다.
5. `GPTQModifier`에 block size 옵션을 추가한다.
6. `serialize.py`가 multiple config groups를 저장하도록 바꾼다.
7. hybrid quantization recipe를 하나 추가한다.
8. KV cache quantization을 위한 config dataclass만 먼저 설계한다.
9. fake INT4 packed format을 저장만 해보고, 로드는 아직 dequantize로 처리한다.
10. `demo.py` 결과를 JSON report로 저장한다.

## 면접이나 코드 리뷰에서 설명하면 좋은 포인트

이 프로젝트를 설명할 때 "작게 만들었다"보다 "어떤 경계를 의도적으로 세웠다"가 더 전문적으로 들린다.

좋은 설명:

- "양자화 수식 자체보다 도구화에 필요한 lifecycle을 먼저 분리했습니다."
- "Quark처럼 module replacement를 실제 수행하되, llm-compressor처럼 modifier composition으로 알고리즘을 확장하게 했습니다."
- "Fake quantization으로 kernel 개발을 제외하고, scale 추정, calibration, serialization, algorithm composition을 검증하는 데 집중했습니다."
- "SmoothQuant와 AWQ는 quantization 전 equivalent transform이므로 output equivalence test를 먼저 만들었습니다."
- "GPTQ는 RTN 대비 reconstruction MSE가 낮아야 한다는 property test로 검증했습니다."
- "Production으로 가려면 packed weight, backend kernel, richer schema, architecture template, KV cache quantization이 다음 단계입니다."

피해야 할 설명:

- "그냥 작은 버전으로 구현했습니다."
- "Quark/llm-compressor를 따라했습니다."
- "fake quant라 실제 압축은 아닙니다"에서 멈추는 설명.

더 나은 표현은 다음과 같다.

"이 프로젝트는 runtime kernel을 만드는 프로젝트가 아니라 quantization pipeline의 control plane을 구현한 것입니다. data plane인 packed kernel은 없지만, 어떤 module을 바꾸고, 어떤 통계로 scale을 만들고, 어떤 algorithm을 어떤 순서로 적용하고, 어떤 metadata로 저장할지를 작은 코드로 검증했습니다."

## 참고 링크

- AMD Quark Getting Started: https://quark.docs.amd.com/latest/basic_usage.html
- AMD ROCm model quantization guide: https://rocmdocs.amd.com/en/develop/how-to/rocm-for-ai/inference-optimization/model-quantization.html
- llm-compressor oneshot guide: https://docs.vllm.ai/projects/llm-compressor/en/latest/guides/entrypoints/oneshot/
- llm-compressor recipe API: https://docs.vllm.ai/projects/llm-compressor/en/latest/api/llmcompressor/recipe/recipe/
