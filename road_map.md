# FuriosaAI Quantization Engineer 과제 준비 로드맵

## 0. 전체 방향성

이 과제는 단순히 “LLM quantization 코드를 하나 구현하는 것”이 아니라, **LLM Compression Tool을 어떻게 설계할 것인가**를 보는 과제다.

핵심 평가 포인트는 다음이다.

- Quantization Unit을 무엇으로 정의할 것인가?
- Quantization config를 어떻게 표현할 것인가?
- 새로운 quantization scheme이 들어오면 변경 범위가 얼마나 작은가?
- `load → prepare → calibrate → convert → save → generate` workflow를 어떻게 설계했는가?
- HuggingFace / compressed-tensors / runtime compatibility를 어느 정도 고려했는가?
- fake quant 기반 POC라도 구조적으로 real quantization과 NPU backend로 확장 가능한가?

따라서 Claude Code, Cursor 같은 에이전트에게 한 번에 맡기기보다, 먼저 설계 지식을 쌓고 “내가 원하는 구조”를 정한 뒤 step-by-step으로 구현하는 것이 안전하다.

---

## 1. 최종 산출물부터 정의하기

먼저 과제의 목표물을 명확히 고정한다.

```text
입력 (타깃 모델):
  필수: Qwen/Qwen3-0.6B
  검증 권장: meta-llama/Llama-3.2-1B (다른 아키텍처에서도 동작함을 확인)
  설계 원칙: 특정 모델에 종속되지 않음 — fnmatch 패턴 기반 target/ignore

지원 양자화 기법 (최소 2개 요건 충족):
  필수 구현:
    1. RTN (Round-to-Nearest) — W8A8 scheme
         weight: per-channel observer(기본 MinMax), activation: per-tensor observer calibration
    2. RTN (Round-to-Nearest) — W4A16 scheme
         weight: per-group observer(기본 MinMax), activation: FP16 passthrough
       ※ W8A8과 W4A16은 동일 알고리즘(RTN)의 다른 config이므로,
         과제 요건 "2개 이상의 양자화 기법" 충족을 위해 아래 추가 기법 구현 권장

  시간 허락 시 추가:
    3. SmoothQuant — W8A8 + activation-weight channel rebalancing pre-processing
    4. GPTQ        — W4A16 + Hessian 기반 layer-wise weight 최적화

필수 demo:
  model.generate() 정상 동작

핵심 설계:
  QuantizationSpec + QuantizationScheme (2-tier config)
  QuantizationModifier (initialize → calibrate → finalize 3단계 — llm-compressor식 lifecycle)
    └ initialize() 내부의 module replacement 메커니즘은 Quark식
  FakeQuantLinear (module replacement + flat buffer — weight_scale / input_scale 직속 보유)
  HF 호환 save_pretrained + quantization_config.json

Scalability 설계:
  Sequential calibration: 대형 모델에서 전체 forward 불가 시
    → layer 단위 GPU offload calibration (modifier.py sequential 모드)
  Multi-GPU: 가산점 요소, 필수 아님
    → 구현 여부는 별도 논의 후 결정
```

과제의 본질은 최고 성능 알고리즘 구현이 아니라, **OSS 생태계와 호환 가능한 compression tool의 설계와 POC**다.

---

## 2. 학습 순서 요약

추천 학습 순서는 다음과 같다.

```text
1. 과제 요구사항 재정의
2. PyTorch nn.Module / state_dict / buffer / module replacement
3. quantization 기본 수식
4. fake quant vs real quant
5. QuantLinear 설계
6. QuantizationScheme / QuantizationModifier 설계
7. prepare-calibrate-convert-save workflow
8. llm-compressor 구조 이해
9. Quark template/preset 구조 이해
10. serialization / quant_config 이해
11. 최소 end-to-end 구현
12. README와 발표용 design rationale 정리
```

---

## 3. 1단계: PyTorch Module 조작 이해

가장 먼저 볼 것은 GPTQ/AWQ 논문이 아니라 **PyTorch module system**이다.

과제의 핵심 질문은 결국 다음이다.

> Quantization 대상을 어떤 단위로 추상화할 것인가?

구현상 반드시 알아야 할 개념은 다음이다.

```text
- nn.Module 구조
- named_modules()
- get_submodule()
- parent module 찾아서 child 교체하기
- register_buffer()
- state_dict에서 parameter와 buffer가 어떻게 저장되는지
- model.eval()
- torch.no_grad()
- dtype / device 이동
```

특히 `register_buffer()`는 중요하다.  
scale과 zero-point는 학습되는 parameter는 아니지만, 저장되어야 하는 값이다.

```python
self.register_buffer("weight_scale", scale)
self.register_buffer("weight_zero_point", zero_point)
```

이 단계의 목표는 다음이다.

```text
Qwen3 model 안의 모든 nn.Linear 이름을 출력할 수 있다.
특정 Linear를 내가 만든 wrapper로 교체할 수 있다.
교체 후 model.generate()가 깨지지 않는다.
```

---

## 4. 2단계: Quantization 기본 수식 정리

최소한 다음은 손으로 설명할 수 있어야 한다.

### Symmetric Quantization

```text
scale = max(abs(x)) / qmax
q = round(x / scale)
x_hat = q * scale
```

### Asymmetric Quantization

```text
scale = (x_max - x_min) / (qmax - qmin)
zero_point = qmin - round(x_min / scale)
q = round(x / scale + zero_point)
x_hat = (q - zero_point) * scale
```

### Granularity

```text
per-tensor:
  tensor 전체에 scale 하나

per-channel:
  output channel마다 scale 하나

per-group:
  group_size 단위로 scale 하나
```

면접에서 중요한 질문은 다음이다.

> 왜 weight는 per-channel/per-group을 쓰고, activation은 per-tensor를 쓰는가?

답변 방향:

```text
weight는 offline에서 고정되어 있으므로 channel/group 단위 scale을 저장해도 overhead가 작다.
activation은 runtime input에 따라 변하고 매 token마다 계산되므로 scale 계산/저장/적용 비용이 중요하다.
따라서 activation은 per-tensor 또는 dynamic/static 중 runtime cost와 accuracy를 보고 선택한다.
```

---

## 5. 3단계: Fake Quantization과 Real Quantization 차이

과제는 fake quantize 기반 generate 동작을 허용한다.  
즉, 반드시 int4 kernel이나 real packed weight를 구현할 필요는 없다.

### Fake Quantization

실제 tensor dtype은 FP16/FP32로 유지된다.

```python
q = torch.clamp(torch.round(x / scale), qmin, qmax)
x_fake = q * scale
```

결과는 float tensor이지만, 값은 quantized grid 위에 놓인다.

장점:

```text
- 구현이 쉽다
- model.generate()가 기존 PyTorch 연산으로 동작한다
- custom int4 kernel이 없어도 된다
- quantization effect를 시뮬레이션할 수 있다
```

단점:

```text
- 실제 memory saving 없음
- 실제 latency speedup 없음
- hardware-specific kernel 효과 검증 불가
```

면접 답변 예시:

```text
이번 POC에서는 end-to-end workflow와 software abstraction 검증이 목적이므로 fake quantization으로 generate 동작을 보장했습니다. 다만 serialization metadata와 QuantLinear abstraction은 추후 real packed weight나 NPU backend kernel로 교체 가능한 형태로 설계했습니다.
```

---

## 6. 4단계: Quantization Unit 설계

Quantization Unit은 크게 세 가지 선택지가 있다.

### A. Tensor 단위 Quantizer

```python
quantizer.quantize_tensor(weight)
```

장점:

```text
- 가장 단순
- scheme 구현이 독립적
- weight/activation 모두 재사용 가능
```

단점:

```text
- model 구조와 연결이 약함
- 어떤 layer에 적용했는지 추적이 어려움
```

### B. nn.Linear 대체 Module

```python
nn.Linear -> FakeQuantLinear 또는 QuantLinear
```

장점:

```text
- LLM의 주요 연산 단위와 잘 맞음
- generate path에 자연스럽게 들어감
- scale/zero_point를 module buffer로 저장 가능
- serialization이 명확함
```

단점:

```text
- Linear 외 module 확장 시 추가 구현 필요
- attention, embedding, lm_head 등 예외 처리 필요
```

### C. Decoder Layer Wrapper

```python
DecoderLayer -> QuantizedDecoderLayer
```

장점:

```text
- layer 전체 정책을 적용하기 좋음
- attention/MLP 단위 제어 가능
```

단점:

```text
- 모델 아키텍처 의존성이 커짐
- Qwen, LLaMA, Mistral마다 wrapper가 달라짐
```

### 추천 선택

과제에서는 **B. nn.Linear 대체 module**을 추천한다.

면접 답변 예시:

```text
저는 quantization unit을 nn.Linear 단위로 정의했습니다. LLM에서 projection, MLP, output projection의 대부분이 Linear로 구성되어 있고, weight/activation quantization을 적용하기에 가장 자연스러운 단위이기 때문입니다. Tensor 단위 quantizer는 내부 component로 두고, workflow 관점에서는 Linear module을 wrapper로 교체하는 방식을 택했습니다. 이렇게 하면 scale과 zero-point를 module buffer로 저장할 수 있고, save/load 및 generate path와도 잘 연결됩니다.
```

---

## 7. 5단계: Config / Scheme 설계

과제에서는 bit-width, dtype, granularity, symmetric 여부 등을 어떻게 config로 표현할지 봐야 한다.

추천 dataclass 구조:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class QuantizationSpec:
    num_bits: int
    symmetric: bool
    granularity: str  # "per_tensor", "per_channel", "per_group", "per_token"
    dtype: str        # "int", "float"
    group_size: Optional[int] = None
    axis: Optional[int] = None
    dynamic: bool = False  # True: 런타임 scale 계산 (calibration 불필요)

@dataclass
class QuantizationScheme:
    name: str
    weight: QuantizationSpec
    activation: Optional[QuantizationSpec] = None
```

### W8A8 예시

```python
W8A8 = QuantizationScheme(
    name="w8a8",
    weight=QuantizationSpec(
        num_bits=8,
        symmetric=True,
        granularity="per_channel",
        dtype="int",
        axis=0,
    ),
    activation=QuantizationSpec(
        num_bits=8,
        symmetric=False,
        granularity="per_tensor",
        dtype="int",
        dynamic=False,
    ),
)
```

### W4A16 예시

```python
W4A16 = QuantizationScheme(
    name="w4a16",
    weight=QuantizationSpec(
        num_bits=4,
        symmetric=True,
        granularity="per_group",
        dtype="int",
        group_size=128,
        axis=1,
    ),
    activation=None,
)
```

### W8A8_DYNAMIC 예시

```python
W8A8_DYNAMIC = QuantizationScheme(
    name="w8a8_dynamic",
    weight=QuantizationSpec(num_bits=8, symmetric=True, granularity="per_channel", dtype="int", axis=0),
    activation=QuantizationSpec(num_bits=8, symmetric=True, granularity="per_token", dtype="int", dynamic=True),
)
```

`granularity`(공간 차원)와 `dynamic`(계산 시점)은 독립 축. `per_token + dynamic=True` 조합이 per-token dynamic quantization.
- `dynamic=True`: `FakeQuantLinear.forward()`에서 런타임 scale 계산, observer 미생성, calibrate() early return
- `per_token`: `x.abs().amax(dim=-1, keepdim=True)` — 토큰(마지막 dim 제외)마다 개별 scale

새 scheme 추가 시 설명:

```text
새로운 scheme은 QuantizationScheme만 추가하고, 기존 workflow는 그대로 재사용합니다.
만약 새로운 quantization algorithm이 필요하면 BaseQuantizer를 상속한 구현체만 추가합니다.
```

---

## 8. 6단계: Workflow 설계

과제에서 요구하는 workflow는 다음이다.

```text
1. HuggingFace model ID로부터 모델 load
2. 모델에 quantization 구조 적용, prepare
3. quantization parameter 산출
4. quantized model 저장
```

구현에서는 이를 더 세분화한다.

```text
load
  ↓
initialize()   — nn.Linear → FakeQuantLinear 교체 + weight observer로 weight scale 계산
               — (Quark식 module replacement)
  ↓
[smooth()]     — [SmoothQuant 시] activation 통계 수집 → per-channel s 계산
               — weight에 s 흡수, FakeQuantLinear.weight 값 직접 수정
               — RTN 기본 구현에는 이 단계 없음
  ↓
calibrate()    — [RTN] activation observer forward, calibration_method별 scale 계산
               — [GPTQ] Hessian 기반 layer-wise weight 최적화 (RTN scale 계산 대체)
               — scheme.activation is None 분기 (W4A16은 activation skip)
               — [sequential 모드 ✅] layer 하나씩 GPU 올려 calibrate → CPU offload
                   model.model.layers 구조 감지 → layers[0].forward 임시 대체로
                   embedding 출력 캡처 → layer별 forward → scale 확정 → CPU 반환
                   calibrate(sequential=True) 플래그로 진입
  ↓
finalize()     — observer 제거, scale buffer만 남김 (깔끔한 state_dict)
  ↓
generate sanity check
  ↓
save()         — save_pretrained + quantization_config.json (HF 호환)
```

> **알고리즘별 변경 범위 요약** (modifier composition 리팩토링 이후 — 6-A에서 완료):  
> SmoothQuant → `modifiers/smoothquant.py` 새 파일 추가. 나머지 파일 변경 없음.  
> GPTQ → `modifiers/gptq.py` 새 파일 추가. 나머지 파일 변경 없음.  
> AWQ → `modifiers/awq.py` 실구현 + `_pair_utils.py` 공통 유틸 분리. `fake_quant_linear.py`, `schemes.py`, `serialize.py` 변경 없음.  
> 세 기법 모두 `FakeQuantLinear.forward()`, `schemes.py`, `serialize.py`는 건드리지 않는다.

### load

```text
HF model id로 tokenizer/model load
dtype, device 설정
```

### initialize()

```text
fnmatch 패턴으로 target / ignore module 탐색
nn.Linear → FakeQuantLinear 교체 (Quark식 module replacement)
FakeQuantLinear는 weight_scale / input_scale을 직속 flat buffer로 보유 (HF 호환)
```

### calibrate()

```text
activation이 있는 scheme (W8A8): calibration text forward pass → observer 통계 수집 → scale/zp 계산
weight-only scheme (W4A16): forward pass 없이 weight scale만 계산
scheme.activation is None 여부로 분기
```

#### Observer 종류 (QuantizationSpec.calibration_method로 선택)

| method | 통계 | scale/zp 계산 | Multi-GPU 동기화 |
|--------|------|--------------|----------------|
| `minmax` (기본) | running min/max | MinMax + zero 포함 보장 | `all_reduce(MIN/MAX)` |
| `percentile` | 전체 activation 수집 | 지정 percentile 클리핑 후 MinMax | `all_gather` → percentile |
| `mse` | grid-search | MSE 최소 scale 탐색 | 로컬 탐색 후 `all_reduce(argmin)` |
| `kl_divergence` | histogram bins | KL divergence 최소 clip range 탐색 | `all_reduce(SUM)` on bins |

**설계 결정:**
- `BaseObserver` 공통 인터페이스 (`update()` / `compute_scale_zp()`) — 생성 시 spec을 받아 granularity 인지
- `MinMaxObserver`가 기본값 — tensor 연산 유지로 multi-GPU all_reduce 확장 용이
- weight·activation이 동일 observer 공유 — activation observer는 `FakeQuantLinear`가, weight observer는 `QuantizationModifier.initialize()`가 `calibration_method`에 따라 인스턴스화
- `finalize()`에서 `module.input_observer = None` → state_dict 오염 방지
- percentile/kl은 메모리 heavy, mse는 grid-search 비용 → POC 기본값은 minmax
- weight(per_channel·per_group)는 minmax/percentile/mse 지원, kl_divergence는 채널별 히스토그램 비용 때문에 per_tensor(activation) 전용

### finalize()

```text
observer 제거 (llm-compressor식 cleanup)
scale/zero_point를 FakeQuantLinear buffer로 고정
state_dict에 observer 잔여물 남기지 않음
Furiosa compiler 입력으로 바로 사용 가능한 깔끔한 weight + scale buffer만 남김
```

> **예외 안전성**: `Compressor.compress()`는 `calibrate()`를 `try/finally`로 감싸 예외 발생 시에도
> `finalize()`를 반드시 호출한다. SmoothQuantModifier처럼 hook을 등록하는 modifier가 있을 때
> OOM 등으로 calibration이 중단돼도 hook이 모델에 잔류하지 않는다.

### generate sanity check

```text
model.generate()로 짧은 문장 생성
FP output과 비교해서 quantization degradation 확인
```

### save()

```text
model.save_pretrained()
tokenizer.save_pretrained()
quantization_config.json 저장 (HF 호환 포맷)
furiosa-llm / vllm이 그대로 로드 가능
```

---

## 9. 7단계: LLM Compressor와 Quark를 설계 관점에서 읽기

OSS 코드를 처음부터 전부 읽지 말고, “책임 분리” 관점에서 보면 된다.

### LLM Compressor에서 볼 것

```text
- Modifier가 무엇을 표현하는가?
- targets / ignore를 어떻게 받는가?
- oneshot이 어떤 workflow를 감싸는가?
- save_pretrained와 compressed-tensors가 어떻게 연결되는가?
```

내 코드에 반영할 결론:

```text
사용자 API는 HF 호환으로 간다.
save_pretrained + quantization_config.json이 표준 진입점.
```

### AMD Quark에서 볼 것

```text
- LLMTemplate이 왜 필요한가?
- model_type별 preset을 어떻게 제공하는가?
- QuantConfig와 ModelQuantizer 역할이 무엇인가?
- export를 별도 책임으로 분리하는가?
- ModelQuantizer.quantize() 내부에서 module replacement를 어떻게 구현하는가?
```

Quark의 진입점: ModelQuantizer.quantize() 하나가 module replacement + calibration + finalize를 일괄 처리한다.  
lifecycle 단계가 API 레벨에서 명시적으로 분리되지 않는 것이 Quark의 특징이다.

내 코드에 반영할 결론:

```text
Quark에서 차용하는 것: module replacement 메커니즘
  → nn.Linear를 FakeQuantLinear로 직접 교체하는 방식
  → parent module을 찾아 setattr로 child를 교체하는 구현 패턴

Quark에서 차용하지 않는 것: quantize() 일괄 처리 방식
  → 우리는 initialize / calibrate / finalize를 명시적으로 분리한다 (llm-compressor식)
```

### 추천 설계

```text
사용자-facing API:
  HF 호환 — save_pretrained / quantization_config.json
  (furiosa-llm, vllm이 그대로 로드 가능한 포맷)
  출처: llm-compressor / HF compressed-tensors 스펙

내부 core:
  module replacement 메커니즘  → Quark식
  lifecycle 3단계 구조         → llm-compressor식
    initialize(): nn.Linear → FakeQuantLinear 교체
    calibrate():  observer scale 계산
    finalize():   observer 제거, scale buffer 고정
  "Modifier" 클래스명          → llm-compressor 용어
```

즉, hybrid design이 가장 설득력 있다.

근거: Furiosa는 사용자에게 HF 표준 인터페이스를 노출하고 내부 컴파일러 단계에서
module replacement + explicit export 철학을 사용한다.
mini-compressor의 설계 방향이 이와 정렬되어 있음을 발표에서 직접 언급할 수 있다.

---

## 10. 8단계: Serialization 설계

### 설계 철학: HF 표준 포맷을 따른다

자체 포맷을 만들지 않는다. furiosa-llm / vllm이 그대로 로드 가능한 포맷이 목표이고,
그 표준은 HuggingFace의 `quantization_config` 스펙이다.

Furiosa llm 공식 문서에 HF 호환 우선순위 명시, compressed-tensors 지원 예정이 확인됨.
따라서 `quant_type: "compressed-tensors"` 포맷을 그대로 준수한다.

---

### 결정 1: quantization_config.json 포맷

HF 표준 스펙을 따른다. compressed-tensors 포맷을 준수한다.

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

**결정 근거:**
- `quant_type`: `"compressed-tensors"` — furiosa-llm 자동 호환 목표. `"mini-compressor"` 같은 자체 문자열은 쓰지 않는다.
- `quantization_status`: `"calibrated"` — fake quant 상태는 weight가 float16 그대로. `"compressed"`는 real int packing 완료 시 사용.
- `targets`: `["Linear"]` — 클래스 이름. fnmatch 패턴이 아닌 compressed-tensors 스펙 준수.
- `input_activations`: W4A16처럼 activation 없으면 `null` **명시**. 필드 생략하지 않는다.
- 저장 위치: `{save_dir}/quantization_config.json` 별도 파일. `config.json` 안에 포함하지 않음 (llm-compressor 방식).

---

### 결정 2: 책임 분리 — serialize.py

`QuantizationScheme → dict` 직렬화 로직은 `serialize.py` 내부 함수로 처리한다.

```text
schemes.py     — "무엇을 양자화할 것인가" (QuantizationSpec, QuantizationScheme 정의)
serialize.py   — "어떻게 저장하는가" (_scheme_to_dict, _scheme_from_dict)
```

`schemes.py`에 `to_dict()` / `from_dict()` 메서드를 추가하지 않는다.
책임이 한 파일에 집중되면 새로운 포맷 요구사항이 생겼을 때 수정 범위가 serialize.py 하나로 제한된다.

---

### 결정 3: Compressor — one-click 진입점 API

```python
compressor = Compressor.from_recipe("w8a8", ignore=["lm_head"])
compressor.compress(model, dataloader)  # initialize → calibrate → finalize 자동 실행
```

내부에서 `QuantizationModifier`를 생성하고 3단계를 순서대로 호출한다.
llm-compressor의 `oneshot()`, Quark의 `quantizer.quantize_model()` 모두 이 원클릭 진입점을 제공함.
발표 데모 관점에서도 한 줄 API가 설명력이 강하다.

> **후속 (recipe preset 레이어로 통일)**: 처음엔 `from_scheme`(단일 scheme)만 있었으나,
> SmoothQuant처럼 여러 modifier가 chain되는 알고리즘은 한 줄로 표현되지 않았다.
> `from_recipe(name)` + `RECIPE_REGISTRY`(`recipes.py`)를 **유일한 진입점**으로 두고
> `from_scheme`은 제거했다 — 단일 RTN scheme도 modifier 1개짜리 recipe로 흡수된다.
> composition 패턴 위에 Quark식 선언적 preset 레이어를 얹되 진입점은 하나로 유지한다.
> `scheme`(수치 포맷)은 recipe 내부 building block으로, `SCHEME_REGISTRY`는
> `serialize.py`의 scheme→name 매칭용 카탈로그로 남는다.

---

### 결정 4: save_pretrained / load_pretrained 시그니처

```python
# 저장
save_pretrained(model, save_dir, tokenizer=None)
# → model.save_pretrained(save_dir)  (safetensors + config.json 자동 생성)
# → quantization_config.json 별도 저장
# → tokenizer 있으면 tokenizer.save_pretrained(save_dir)

# 로드
load_pretrained(save_dir)
# → AutoConfig.from_pretrained(save_dir)로 model_id 꺼냄 (config._name_or_path)
# → AutoModelForCausalLM.from_pretrained(model_id)로 base model 생성
# → quantization_config.json 읽어 scheme, ignore 복원
# → modifier.initialize(compute_scales=False) — 구조만 생성, scale 계산 안 함
# → model.load_state_dict(saved_state_dict) — weight + scale buffer 주입
```

`config._name_or_path`에 대한 근거: `model.save_pretrained()` 호출 시 HF가 원본 model_id를
`config.json`에 자동으로 저장함. `AutoConfig.from_pretrained(save_dir)`로 복원 가능.

---

### 결정 5: initialize(compute_scales=False) — 구조/scale 분리 패턴

load 흐름에서 scale 재계산 낭비를 없애기 위해 `compute_scales` 파라미터를 추가한다.

```python
def initialize(self, compute_scales: bool = True) -> None:
    for name, mod in to_replace:
        fql = FakeQuantLinear.from_float(mod, self.scheme)
        if compute_scales:
            wobs = build_observer(self.scheme.weight)
            wobs.update(fql.weight.detach())
            fql.weight_scale, fql.weight_zero_point = wobs.compute_scale_zp()
        setattr(parent, attr, fql)
```

- **압축 흐름**: `modifier.initialize()` (기본값 True, 기존 동작 그대로)
- **load 흐름**: `modifier.initialize(compute_scales=False)` (구조만, scale은 state_dict에서 채움)

llm-compressor의 `_process_model_before_weight_loading()` (구조 생성, scale 계산 안 함) + weight 로딩 패턴과 동일 철학.
Quark도 구조 생성과 scale 로딩을 분리하는 동일 패턴을 사용.

---

### 결정 6: 구현 순서

```text
1. modifier.py — initialize(compute_scales=True) 파라미터 추가
2. serialize.py — _scheme_to_dict, _scheme_from_dict, save_pretrained, load_pretrained 구현
3. compressor.py — Compressor.from_scheme(), compress() 구현
4. round-trip 테스트 — W8A8, W4A16 compress → save → load → generate 검증
```

---

### 결정 7: state_dict key 구조

llm-compressor식을 따른다 — FakeQuantLinear 직속 buffer.

```text
# mini-compressor state_dict
model.layers.0.self_attn.q_proj.weight          # float16 (fake quant 상태)
model.layers.0.self_attn.q_proj.weight_scale    # buffer
model.layers.0.self_attn.q_proj.weight_zero_point  # buffer (symmetric이면 0)
model.layers.0.self_attn.q_proj.input_scale     # W8A8일 때만
```

Quark식 (`_weight_quantizer.scale`) 대신 flat key 구조를 택하는 이유:
HF `from_pretrained`로 로드할 때 submodule key 매핑 없이 바로 복원 가능하기 때문이다.

---

### 결정 8: fake quant 상태의 weight key

`weight` key를 그대로 유지한다. `qweight`로 분리하지 않는다.

fake quant는 dtype이 float16 그대로이므로 `weight`로 저장해도 HF 로드 경로가 깨지지 않는다.
`qweight`는 real int packing까지 구현할 때 도입한다 (현재 POC 범위 밖).

---

면접 답변 예시:

```text
HF 표준 quantization_config 포맷을 따랐습니다. compressed-tensors 스펙을 참조해
config_groups 구조로 scheme을 표현했고, state_dict는 scale/zero_point를
FakeQuantLinear 직속 buffer로 저장해 from_pretrained로 바로 복원 가능합니다.
fake quant 상태이므로 weight dtype은 float16을 유지했고,
real int packing은 Furiosa compiler 단계에서 처리됩니다.
```

---

## 11. 최소 구현 Milestone

### Milestone 1

```text
Qwen/Qwen3-0.6B load
tokenizer load
FP model.generate() 확인
```

### Milestone 2

```text
모든 nn.Linear 이름 출력
ignore=["lm_head"] 적용 확인
```

### Milestone 3

```text
FakeQuantizer 구현
per-tensor symmetric weight fake quant
```

### Milestone 4

```text
FakeQuantLinear 구현
기존 nn.Linear weight를 복사
forward에서 fake quant weight 사용
generate 확인
```

### Milestone 5

```text
W8A8 추가
activation observer 추가
calibration forward pass
activation scale 고정
generate 확인
```

### Milestone 6 ✅ W4A16 RTN — M5에 통합 완료 (2026-05-11)

```text
W4A16 RTN per-group weight fake quant → M5 구현 시점에 이미 완성
M6은 SmoothQuant / GPTQ 확장 milestone으로 재정의
```

### Milestone 6-A — SmoothQuant ✅ 완료 (feature/smoothquant 브랜치)

```text
modifier composition 리팩토링 (modifier.py → modifiers/ 디렉토리)
BaseModifier + 알고리즘별 클래스 (Quantization / SmoothQuant / GPTQ stub / AWQ stub)
Compressor가 modifier list 수용 → algorithm chain을 list 순서로 표현
SmoothQuantModifier 실구현:
  - norm-linear pair 자동 탐색 (input_layernorm→q/k/v_proj, post_attention_layernorm→gate/up_proj)
  - forward pre-hook으로 channel-wise activation abs max 수집
  - s = max(|X|)^α / max(|W|)^(1-α) 계산 (α=0.5 default)
  - norm.weight /= s, linear.weight *= s 적용
```

변경 파일: `mini_compressor/modifiers/` (신규), `compressor.py`, `serialize.py`, tests, `demo.py`  
변경 없음: `fake_quant_linear.py`, `schemes.py`, `observer.py`

> **설계 검증 효과**: 새 알고리즘 추가가 `modifiers/<algo>.py` 한 파일 추가로 끝난다는 점이 SmoothQuant 구현으로 입증됐다. road_map 17.3 / 19 답변 강도 상승.

### Milestone 6-B — GPTQ + AWQ ✅ 완료 (2026-05-18/19)

**GPTQ** (2026-05-18):
```text
modifiers/gptq.py GPTQModifier 실구현 (stub → 실구현)
QuantizationMixin 분리 (llm-compressor QuantizationMixin 패턴)
forward pre-hook으로 layer별 H = 2·XᵀX 누적
dead column + dampening + Cholesky inverse
per-group block-column 양자화 + intra/inter-group 오차 전파
w4a16_gptq recipe 추가
unit test 4개: replaces_linear / scale_shape / weight_on_grid / mse_leq_rtn
Qwen3-0.6B wikitext-2 PPL 측정 — GPTQ 20.96 vs RTN 25.89 (-4.93)
fake_quant_linear dtype 버그 수정 (scale float16 캐스팅 일관화)
```

**AWQ** (2026-05-19):
```text
modifiers/_pair_utils.py 신규 — SmoothQuant/AWQ 공통 pair 탐색 유틸
modifiers/awq.py AWQModifier 실구현 (stub → 실구현)
  - channel-wise activation mean 수집 (SmoothQuant의 max 대신 mean)
  - grid search alpha ∈ (0,1]: s = (s_x / mean(s_x))^alpha
  - INT4 per-group fake quant error 최소화
  - best_s → norm.weight /= s, linear.weight *= s (등가 변환)
w4a16_awq recipe 추가
unit test 7개: preserves_forward_output, layernorm_bias, initialize_required, empty_dataloader,
               no_pairs_model, compressor_chain, int4_fake_quant_roundtrip
```

변경 파일: `modifiers/_pair_utils.py` (신규), `modifiers/gptq.py`, `modifiers/awq.py`,
`modifiers/quantization.py`, `modifiers/smoothquant.py` (import 갱신),
`modifiers/__init__.py`, `compressor.py`, `recipes.py`,
`tests/test_gptq.py`, `tests/test_awq.py` (신규), `fake_quant_linear.py`, `demo.py`  
변경 없음: `schemes.py`, `serialize.py`  
테스트: 49개 통과 (7개 추가)

### 코드 품질 개선 ✅ 완료 (2026-05-19 — GPT 리뷰 반영)

```text
1. ignore fnmatch 버그 수정 (quantization.py)
   name in self.ignore (exact match) → fnmatch 패턴 매칭 (targets와 동일 방식)
2. group_size 배수 검증 추가 (quantization.py)
   initialize() 진입 시 in_features % group_size != 0 이면 ValueError
3. GPTQModifier hook try/finally (gptq.py)
   forward loop 예외 시 hook 잔류 방지
4. assert → RuntimeError (gptq.py)
   python -O 비활성화 방지, SmoothQuantModifier와 일관성
5. frozen=True (schemes.py)
   QuantizationSpec, QuantizationScheme 불변 객체로 명시
   slides.md "Scheme = frozen dataclass" 주장과 코드 일치
발표자료(slides.md, script.md) 최신화: from_recipe API, 구현 알고리즘 현황, 6종 PPL 표
```

### Milestone 7 ✅ (M5와 병합 완료 — 2026-05-11)

```text
QuantizationModifier로 리팩토링
initialize / calibrate / finalize 3단계 분리
scheme.activation is None 분기 확인
```

### Milestone 8

```text
8-1. modifier.py — initialize(compute_scales: bool = True) 파라미터 추가
8-2. serialize.py — _scheme_to_dict, _scheme_from_dict, save_pretrained, load_pretrained 구현
     - quantization_config.json: compressed-tensors 포맷, status="calibrated"
     - load_pretrained: config._name_or_path로 model_id 복원
     - initialize(compute_scales=False) 패턴 활용
8-3. compressor.py — Compressor.from_scheme("w8a8", ignore=["lm_head"]) + compress() 구현
     - 내부에서 QuantizationModifier 생성, initialize → calibrate → finalize 자동 실행
8-4. round-trip 테스트 — W8A8, W4A16 compress → save → load → generate 검증
```

### Milestone 9

```text
README 작성 (설치법, 실행법, 지원 scheme, 설계 설명, limitation)
```

### Milestone 10

```text
발표자료 작성
trade-off 정리
known limitation 정리
```

### Milestone 11 ✅ 완료

```text
lm-eval-harness 연동 ✓
Qwen3-0.6B FP baseline 측정 — 18.16 ✓
W8A8 static 측정 — 25.01 ✓
W4A16 RTN 측정 — 25.89 ✓
W8A8 dynamic 측정 — 18.48 ✓
W8A8 SmoothQuant 측정 — 23.67 ✓
W4A16 GPTQ 측정 — 20.96 ✓  (RTN 25.89 대비 -4.93)
FP vs RTN vs SmoothQuant vs GPTQ 비교 표 → README 반영 ✓
```

> 주의: fake quant는 실제 inference path를 통과하므로 perplexity 열화는 측정 가능하다.  
> real quantization 대비 latency / memory saving은 없으며, 이것을 limitation으로 명시한다.

### Milestone 6-C — Sequential Calibration (시간 허락 시)

```text
QuantizationModifier.calibrate()에 sequential=True 모드 추가
layer 하나씩 GPU에 올려 calibrate → CPU offload
전체 forward가 불가능한 대형 모델에서도 calibration 가능하게
Qwen3-0.6B로 동작 확인 (소형 모델에서도 sequential 모드가 동작함을 검증)
```

변경 파일: `modifiers/quantization.py` 만 (`calibrate()` 내부 sequential 분기 추가)  
파이프라인 위치: `calibrate()` 내부 — full forward vs layer-by-layer offload 분기

```python
# 예시 인터페이스
modifier.calibrate(model, dataloader, sequential=False)  # 기본: 전체 forward
modifier.calibrate(model, dataloader, sequential=True)   # 대형 모델: layer-by-layer
```

### Milestone 6-D — 멀티모델 검증 ✅ 완료 (TinyLlama-1.1B)

> `meta-llama/Llama-3.2-1B`은 HF gated(접근 미승인) → 동일 `model_type=llama`인
> `TinyLlama/TinyLlama-1.1B-Chat-v1.0`로 대체. 결과: 라이브러리 코드 0줄 수정으로
> 4개 recipe 동작, SmoothQuant pair 44개 자동 탐색(RMSNorm, GQA 32:4), `lm_head` ignore.
> 산출물: `demo.py --model` 인자, `notebooks/milestone6d_llama_validation.ipynb`.

```text
LLaMA 계열 에서 W8A8 / W4A16 동작 확인
targets / ignore 패턴이 모델 아키텍처에 종속되지 않음을 검증
SmoothQuant의 preceding norm layer 탐색이 LLaMA에서도 동작하는지 확인
```

모델 독립성 설계 포인트:
```text
targets: ["model.layers.*.self_attn.*_proj", "model.layers.*.mlp.*"]  → Qwen3/LLaMA 공통
ignore:  ["lm_head"]                                                   → 공통
SmoothQuant norm 탐색: 이름 기반 auto-detect 또는 model_config로 지정
```

### Milestone 12 — End-to-End Demo

```text
notebooks/demo.ipynb 작성
1. Compressor.from_scheme("w8a8").compress(model, dataloader)
2. model.generate() 동작 확인
3. save_pretrained → quantization_config.json 확인
4. lm-eval 수치 출력
전체 flow를 노트북 한 파일에서 재현 가능하게 만든다.
```

중요한 원칙:

```text
처음부터 예쁜 architecture로 시작하지 말고,
동작하는 작은 코드를 만든 뒤 abstraction으로 끌어올린다.
```

---

## 12. 15 Working Days 학습/구현 플랜

### Day 1: 과제 분석 + PyTorch module replacement

목표:

```text
Qwen3 load
Linear module list 출력
특정 Linear 교체 실험
```

### Day 2: quantization 수식 + FakeQuantizer

목표:

```text
per-tensor / per-channel fake quant 구현
unit test 작성
```

### Day 3: FakeQuantLinear

목표:

```text
nn.Linear -> FakeQuantLinear 교체
FP output과 shape 비교
generate 동작 확인
```

### Day 4: Scheme config 설계

목표:

```text
QuantizationSpec
QuantizationScheme
W8A8, W4A16 preset 정의
```

### Day 5: prepare workflow

목표:

```text
targets / ignore 기반 module replacement
QuantizationModifier 구현
```

### Day 6: calibration

목표:

```text
activation observer
calibration text forward
scale freeze
```

### Day 7: W8A8 end-to-end

목표:

```text
Qwen3 W8A8 fake quant
generate demo
```

### Day 8: W4A16 추가 (RTN)

목표:

```text
per-group weight fake quant (RTN)
activation FP 유지
generate demo
```

### Day 8-A: SmoothQuant (시간 허락 시)

목표:

```text
modifier.py smooth() 구현
per-channel scaling factor s 계산 (α=0.5)
FakeQuantLinear.weight에 s 흡수
W8A8 RTN 대비 perplexity 비교
```

### Day 8-B: GPTQ (시간 허락 시)

목표:

```text
modifier.py calibrate() GPTQ 분기 구현
layer별 Hessian + column-wise 최적화
W4A16 RTN 대비 perplexity 비교
```

### Day 9: save / serialization

목표:

```text
save_pretrained
quant_config.json
load script 또는 limitation 명시
```

### Day 10: HF 호환 API 완성

목표:

```text
Compressor.from_scheme("w8a8") 진입점 확정
quantization_config.json 포맷 HF 스펙 맞춤
save_pretrained / load 왕복 테스트
```

### Day 11: README 정리

목표:

```text
설치법
실행법
지원 scheme
설계 설명
limitation
```

### Day 12: test / sanity check + lm-eval

목표:

```text
unit test (pytest)
generate test
save/load test 가능한 범위
lm-eval-harness 연동 + FP / W8A8 / W4A16 perplexity 측정
비교 표 작성
```

### Day 13: End-to-End Demo 노트북

목표:

```text
notebooks/demo.ipynb 작성
Compressor.from_scheme() → generate → save → lm-eval 전체 flow 재현
노트북 한 파일에서 발표 demo 가능한 상태로 완성
```

### Day 13.5 (여유): CI/CD 점검

목표:

```text
채택한 CI 검증 항목 실제 동작 확인
.github/workflows/ 최종 정리
```

### Day 14: 발표자료 초안

목표:

```text
Problem
OSS study
Design (핵심 3문답: 추상화 단위 / Config 표현 / scheme 확장 범위)
Implementation
Demo (notebooks/demo.ipynb)
Trade-off
Future work
```

### Day 14.5: 예상 질문 답변

목표:

```text
왜 Linear unit?
왜 fake quant?
왜 initialize/calibrate/finalize 3단계로 나눴나?
왜 사용자 API를 HF 호환으로 맞췄나?
serialization은 어떻게 확장?
NPU 최적화와 어떻게 연결?
```

### Day 15: 리허설

목표:

```text
30분 발표 연습
코드 walkthrough 연습
Q&A 압박질문 대비
```

---

## 13. Scalability 설계

### 13-1. 멀티모델 지원 설계

**원칙: 모델 구조에 종속되지 않는다.**

현재 `named_modules()` + fnmatch 패턴 방식은 이미 model-agnostic이지만,
다음 두 지점이 모델 의존성을 유발할 수 있다.

**지점 1: target/ignore 패턴**

```python
# Bad — 모델 종속
ignore = ["model.embed_tokens", "lm_head"]

# Good — 모델 독립
ignore = ["lm_head"]               # 공통 패턴
targets = ["*.q_proj", "*.k_proj", "*.v_proj", "*.o_proj",
           "*.gate_proj", "*.up_proj", "*.down_proj"]  # Qwen3/LLaMA 공통 MLP 이름
```

Compressor API에서 `targets`/`ignore`를 사용자가 지정 가능하도록 설계한다.
기본값은 일반적인 LLM Linear layer 패턴으로 설정한다.

**지점 2: SmoothQuant의 preceding norm layer 탐색**

SmoothQuant는 Linear layer 직전의 RMSNorm/LayerNorm을 찾아야 한다.
모델마다 이름이 다르므로 두 가지 중 하나를 선택한다.

```text
옵션 A: auto-detect
  Linear layer parent를 타고 올라가며 norm layer 자동 탐색
  구현 복잡도 높음

옵션 B: model_config로 지정
  smooth_norm_pattern: "*.input_layernorm"  # Qwen3/LLaMA 공통
  사용자가 모델에 맞게 지정 가능
  구현 단순, 확장성 충분
```

추천: 옵션 B (simple over clever)

---

### 13-2. Sequential Calibration (Layer-by-Layer Offload)

**문제**: 대형 모델(7B+)은 전체 forward를 한 번에 GPU에서 실행할 수 없다.

**해결**: layer 하나씩 GPU에 올려 calibrate하고 CPU로 offload한다.

```text
일반 모드 (sequential=False):
  dataloader → 전체 model forward → observer가 각 layer에서 통계 수집

Sequential 모드 (sequential=True):
  Step 1: 첫 번째 layer 직전까지의 입력 수집 (hook)
           embedding + 초기 처리만 GPU에서 실행, 결과를 CPU에 캐시

  Step 2: For each layer i:
           layer[i]를 GPU로 이동
           캐시된 layer[i] 입력으로 calibration 실행 (scale 계산)
           layer[i]의 출력을 CPU에 캐시 → 다음 layer의 입력
           layer[i]를 CPU로 offload

  결과: 각 layer의 scale이 FakeQuantLinear buffer에 저장됨
```

**파이프라인 위치**: `modifier.py`의 `calibrate()` 내부 분기

```python
def calibrate(self, model, dataloader, sequential=False):
    if sequential:
        self._calibrate_sequential(model, dataloader)  # layer-by-layer
    else:
        self._calibrate_full(model, dataloader)        # 전체 forward
```

변경 파일: `modifier.py` 만  
`FakeQuantLinear`, `schemes.py`, `serialize.py` 변경 없음

> 참고: GPTQ 원 구현과 llm-compressor의 `sequential_update=True`가 이 방식을 사용한다.
> Milestone 6-C에서 구현한다.

---

### 13-3. Multi-GPU 지원 (가산점 요소)

**현재 상태: Milestone 13 완료 — observer `sync()` 4종 구현(MinMax는 all_reduce, Percentile/MSE/KL은 all_gather) + gloo 2-프로세스 검증. Tensor Parallelism과 실 2-GPU `device_map="auto"` 실측은 범위 외 / 하드웨어 한계.**

**고려해야 할 사항:**

```text
1. DataParallel calibration
   - 여러 GPU에 calibration batch를 분산
   - Observer 통계(min/max)를 all-reduce로 동기화해야 함
   - torch.distributed.all_reduce 필요

2. Tensor Parallelism (모델 병렬)
   - Linear weight가 GPU 간에 split된 경우
   - scale 계산 시 shard된 weight를 reconstitute하거나
     shard별로 scale을 계산하고 merge해야 함
   - 구현 복잡도 높음

3. Sequential calibration과 결합
   - 각 layer를 특정 GPU로 보내는 경우
   - device placement 로직이 추가로 필요

4. state_dict 저장
   - distributed 환경에서 rank 0만 저장하거나
     각 rank가 자신의 shard를 저장하는 방식 선택 필요
```

**영향 범위:**

```text
modifier.py:  calibrate() — observer all-reduce 동기화
compressor.py: device placement 로직
serialize.py:  distributed state_dict 처리
```

> 이 기능은 Milestone 6-D 이후, 시간이 충분할 때 별도 논의 후 진행한다.

### 13-4. Multi-GPU 확장을 위한 초기 설계 원칙

나중에 multi-GPU 확장 공사 비용을 줄이기 위해 **초기 구현 시점부터** 아래 원칙을 지킨다.  
지금 당장 분산 코드를 작성하는 게 아니라, 확장을 막는 패턴을 피하는 것이 목표다.

**원칙 1: device를 하드코딩하지 않는다**

```python
# Bad — device 가정이 코드에 박힘
scale = scale.to("cuda")

# Good — weight가 있는 device를 따라감
scale = scale.to(module.weight.device)
```

`FakeQuantLinear`의 buffer/scale은 항상 `weight.device`를 참조한다.  
`device_map="auto"`로 모델이 여러 GPU에 분산돼도 추가 수정 없이 동작한다.

**원칙 2: Observer 통계를 tensor 연산으로 유지한다**

```python
# Bad — Python scalar 연산
self.min_val = min(self.min_val, x.min().item())

# Good — tensor 연산 유지
self.min_val = torch.minimum(self.min_val, x.min())
self.max_val = torch.maximum(self.max_val, x.max())
# → 나중에 dist.all_reduce() 한 줄만 추가하면 multi-GPU 동기화 완성
```

**원칙 3: calibrate()는 device 인자를 받지 않는다**

```python
# Bad — 호출부가 device를 알아야 함
modifier.calibrate(model, dataloader, device="cuda:0")

# Good — model에서 device를 내부 참조
modifier.calibrate(model, dataloader)
# 내부: batch = batch.to(next(model.parameters()).device)
```

**원칙 4: save_pretrained에 rank guard 패턴 적용**

```python
# serialize.py 구현 시점에 적용
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    model.save_pretrained(save_dir)
```

single GPU / multi-GPU 환경 모두 동작하는 패턴이다.

**적용 시점 요약:**

| 원칙 | 적용 Milestone |
|------|--------------|
| device 하드코딩 금지 | Milestone 5 (Observer 구현 시) |
| Observer tensor 연산 | Milestone 5 (MinMaxObserver 구현 시) |
| calibrate() device 인자 제거 | Milestone 7 (modifier.py 구현 시) |
| rank guard | Milestone 9 (serialize.py 구현 시) |

---

## 14. Git + CI/CD 운영 방침

### Git 운영 원칙

1인 프로젝트이지만 GitHub을 통해 버전 관리하고 CI 경험을 쌓는다.

```text
초기화 시점: Milestone 1 완료 직후 (FP generate 확인되면 즉시 push)
브랜치 전략: feature/<milestone-name> 브랜치 → main PR 방식
커밋 단위: Milestone 하나 완료마다 최소 1 커밋
```

### CI/CD 채택 방침

**원칙: 임의로 CI를 짜지 않는다.**

- 작업 중 적절한 시점에 "어떤 CI 검증이 필요한가?" 물어본다
- 그 시점 코드 상태에 맞는 검증 방법을 **이유와 함께** 설명 받는다
- 채택 결정 후 `.github/workflows/` 에 추가한다
- 필요 없는 CI는 추가하지 않는다 (1인 프로젝트이므로 유지비용이 실익보다 크면 제외)

**예상 적용 시점과 검증 항목:**

| 시점 | 추천 CI 항목 | 이유 |
|------|------------|------|
| Milestone 4 완료 후 | `pytest tests/` | FakeQuantLinear forward 회귀 방지 |
| Milestone 7 완료 후 | modifier workflow 통합 테스트 | initialize/calibrate/finalize 3단계 API 안정성 |
| Milestone 9 완료 후 | save/load round-trip 테스트 | state_dict + quantization_config.json 무결성 |

> 실제 채택 여부와 내용은 각 시점에 별도 논의한다. 위 표는 참고용이다.

### CI 논의 방법

각 Milestone이 완료될 때, 다음과 같이 요청한다:

```text
"Milestone N이 완료됐어. 이 시점에 어떤 CI 검증을 추가하면 좋을지 이유와 함께 알려줘."
```

---

## 15. AI 에이전트를 쓰는 올바른 방식

### 나쁜 방식

```text
Furiosa quantization 과제 전체 구현해줘.
```

이렇게 하면 과제는 나올 수 있지만, 본인 설계가 아니어서 발표면접에서 취약하다.

### 좋은 방식

```text
내가 QuantizationScheme dataclass를 이렇게 설계했다.
이 설계에서 W8A8과 W4A16을 표현하기에 부족한 필드가 있는지 리뷰해줘.
```

```text
내 FakeQuantLinear forward가 model.generate에서 깨진다.
원인은 device/dtype mismatch로 보이는데, 이 함수만 디버깅해줘.
```

```text
이 workflow diagram을 기준으로 README의 design section을 작성해줘.
단, 구현하지 않은 부분은 limitation으로 명시해줘.
```

AI는 **작성자**가 아니라 **코드 리뷰어 / 디버거 / 문서화 보조자**로 쓰는 것이 좋다.

---

## 16. 반드시 답할 수 있어야 하는 질문

공부와 구현을 진행하면서 아래 질문에 답할 수 있어야 한다.

```text
1. 왜 quantization unit을 nn.Linear로 잡았나?
2. Tensor quantizer와 module wrapper의 차이는?
3. W8A8과 W4A16은 config상 어떻게 다른가?
4. per-channel과 per-group scale shape은 어떻게 달라지는가?
5. activation quantization은 왜 calibration이 필요한가?
6. fake quant와 real quant의 차이는?
7. generate가 동작한다는 것은 무엇을 검증하는가?
8. save_pretrained만으로 충분한가?
9. quant_config.json에는 무엇이 들어가야 하는가?
10. 새로운 scheme을 추가하면 어떤 파일을 수정해야 하는가?
11. module replacement 방식(Quark식)의 장점은 무엇인가? hook 방식과 무엇이 다른가?
12. finalize에서 observer를 제거하는 이유는? FrozenFakeQuantize 방식과 무엇이 다른가?
13. 사용자 API는 HF 호환, module replacement는 Quark식, lifecycle 구조는 llm-compressor식으로 가져갔는데 왜 이렇게 섞였는가? Furiosa 실무와 어떻게 정렬되는가?
14. NPU-specific optimization은 이 구조에서 어디에 들어갈 수 있는가?
15. 실제 speedup이 안 나는 fake quant POC의 한계는 무엇인가?
16. W8A8과 W4A16이 모두 RTN인데, 과제 요건 "2개 이상의 양자화 기법"을 어떻게 충족하는가?
17. SmoothQuant가 해결하는 문제는 무엇인가? RTN W8A8 대비 perplexity가 왜 개선되는가?
18. GPTQ가 RTN보다 나은 이유는? Hessian이 왜 필요한가?
19. SmoothQuant와 GPTQ 모두 `modifiers/`에 새 파일 하나 추가로 끝나는데, 이것이 설계상 무엇을 보여주는가? (`fake_quant_linear.py`, `schemes.py`, `serialize.py` 변경 없음으로 입증됨)
20. 대형 모델에서 전체 forward가 불가능할 때 어떻게 calibration을 수행하는가?
21. sequential calibration은 어떤 파일을 수정하는가? FakeQuantLinear나 schemes.py를 건드리는가?
22. 이 툴을 LLaMA 계열에 적용하려면 무엇을 바꿔야 하는가?
```

---

## 17. 발표 핵심 3문답 — 설계 의도와 trade-off

> 면접관이 가장 집중해서 보는 부분. 이 세 가지에 대해 막힘 없이 답할 수 있어야 한다.

---

### 문답 1: Quantization 대상을 어떤 단위로 추상화했는가?

**질문 형태:**
> "Quantization 대상을 어떤 단위로 추상화했나요? 그 선택의 근거는 무엇인가요?"

**답변 구조:**

```text
선택: nn.Linear를 대체하는 custom module (FakeQuantLinear)

근거:
  LLM의 계산량 대부분이 attention / MLP 내부의 Linear에 집중되어 있다.
  nn.Linear 단위로 교체하면 model 구조를 건드리지 않고 quantization을 적용할 수 있다.
  module replacement 방식이므로 model 코드 수정 없이 hook 방식보다 state_dict 호환이 자연스럽다.

대안들과 trade-off:
  tensor-level quantizer: 유연하지만 모든 텐서에 일일이 적용해야 함. state_dict에 scale이 독립 key로 남지 않음.
  layer-level wrapper (Decoder block 단위): 배치 처리 용이하지만 MHA attention와 MLP 구분 없이 묶여 granularity 부족.
  
결론:
  nn.Linear 교체가 granularity와 호환성 사이 가장 실용적인 균형점이다.
  Quark, GPTQ, AWQ 모두 이 수준에서 동작한다.
```

**핵심 문장:**
> "LLM에서 quantization의 실익은 Linear에 집중되어 있고, module replacement 방식이 state_dict 호환과 workflow 통합 관점에서 가장 자연스럽습니다."

---

### 문답 2: Config를 어떻게 표현했는가?

**질문 형태:**
> "Quantization config를 어떻게 표현하셨나요? bit-width, dtype, granularity, symmetric 여부를 어떻게 다루나요?"

**답변 구조:**

```text
2-tier 설계:

  QuantizationSpec (텐서 수준 명세):
    - num_bits: 양자화 비트폭 (4, 8 등)
    - symmetric: True → zero_point = 0 고정, False → asymmetric
    - granularity: "per_tensor" | "per_channel" | "per_group"
    - group_size: granularity == "per_group" 일 때만 유효
    - axis: per_channel scale의 기준 축 (weight는 axis=0, 즉 out_features 축)
    - dtype: "int" (추후 float8 등 확장 가능)
    - dynamic: True이면 runtime에 scale 계산 (activation용)

  QuantizationScheme (weight + activation 쌍):
    - weight: QuantizationSpec
    - activation: Optional[QuantizationSpec]
    - W8A8 → weight(8bit per_channel symmetric) + activation(8bit per_tensor asymmetric)
    - W4A16 → weight(4bit per_group symmetric) + activation(None, FP16 유지)

설계 의도:
  QuantizationSpec을 weight와 activation에 공통 적용하면서,
  activation=None이면 weight-only quantization으로 자연스럽게 분기.
  새로운 spec 추가 시 QuantizationScheme 한 줄이면 표현 완성.
```

**핵심 문장:**
> "QuantizationSpec은 단일 텐서에 적용되는 수치 명세이고, QuantizationScheme은 weight와 activation 쌍을 묶는 layer 단위 정책입니다. 이 2-tier 구조 덕분에 새로운 scheme은 dataclass 정의 한 번으로 추가됩니다."

---

### 문답 3: 새로운 scheme 추가 시 변경 범위는?

**질문 형태:**
> "새로운 quantization scheme이 들어오면 어떤 부분을 수정해야 하나요? 변경 범위가 얼마나 제한되나요?"

**답변 구조:**

```text
예시: W8A8FP (float8 activation) scheme을 추가하는 경우

수정 필요한 파일: mini_compressor/schemes.py 만

  추가 내용:
    W8A8FP = QuantizationScheme(
        name="w8a8fp",
        weight=QuantizationSpec(num_bits=8, dtype="int", symmetric=True, granularity="per_channel", axis=0),
        activation=QuantizationSpec(num_bits=8, dtype="float8", symmetric=True, granularity="per_tensor"),
    )
    SCHEME_REGISTRY["w8a8fp"] = W8A8FP

수정 불필요한 파일:
  - FakeQuantLinear: scheme 객체를 읽어서 동작하므로 변경 없음
  - QuantizationModifier: scheme 기반 분기 없이 initialize/calibrate/finalize 그대로
  - Compressor: from_recipe(name) + RECIPE_REGISTRY 구조 — recipe 한 줄 등록 외 진입점 코드 변경 없음
  - serialize: quantization_config.json 포맷은 QuantizationScheme을 직렬화하므로 변경 없음

단, dtype="float8" 실제 연산을 지원하려면:
  - FakeQuantLinear._fake_quantize_weight() 내 float8 경로 추가 필요
  - 이는 FakeQuantLinear의 연산 로직이지 scheme 정의 레이어 변경이 아님

변경 범위 요약:
  scheme 정의만 추가 → schemes.py 1개 파일
  새 dtype 연산 지원 → fake_quant_linear.py 1개 파일의 메서드 확장
  나머지 파일 (modifier, compressor, serialize) → 변경 없음
```

**핵심 문장:**
> "새로운 scheme은 schemes.py에 dataclass 한 줄 추가로 끝납니다. FakeQuantLinear는 scheme 객체를 런타임에 읽어서 동작하므로 scheme이 늘어나도 module 코드를 건드리지 않습니다. 변경 범위를 schemes.py로 제한하는 것이 이 2-tier 설계의 핵심 목표였습니다."

---

## 18. 발표에서 강조할 핵심 문장

```text
이번 과제에서는 단순히 특정 quantization algorithm을 구현하기보다, 새로운 scheme과 model architecture가 추가되어도 변경 범위를 제한할 수 있는 compression tool 구조를 설계하는 데 집중했습니다.
```

```text
내부 core의 module replacement 메커니즘은 Quark식으로 설계했고, lifecycle 구조(initialize/calibrate/finalize 3단계 명시적 분리)는 llm-compressor식을 따랐습니다. 사용자 API는 HF 호환 save_pretrained + quantization_config.json으로 맞췄습니다. Furiosa가 실제로 취하는 quantize → compile → serve 파이프라인과 정렬한 설계 결정입니다.
```

```text
Quantization Unit은 LLM의 주요 연산 단위인 nn.Linear로 정의했고, 내부적으로는 tensor-level quantizer를 사용하되 workflow 관점에서는 module replacement 방식으로 통합했습니다.
```

```text
현재 구현은 fake quant 기반이므로 실제 memory saving이나 latency speedup은 제공하지 않지만, prepare-calibrate-convert-save-generate workflow와 serialization metadata를 명확히 분리해 추후 real packed weight 및 NPU backend kernel로 확장 가능한 구조를 지향했습니다.
```

---

## 19. 최종 핵심

```text
먼저 내가 어떤 설계 철학으로 만들 것인지 정한다.
그 설계를 기준으로 AI에게 작은 단위 구현을 요청한다.
각 구현 단위마다 왜 그렇게 만들었는지 설명 가능해야 한다.
```

한 문장 요약:

> 이 과제의 핵심은 “양자화 알고리즘을 아는 사람”이 아니라, **LLM quantization을 실제 compression software stack으로 구조화할 수 있는 사람**임을 보여주는 것이다.
