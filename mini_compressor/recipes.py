# 양자화 preset(modifier 파이프라인)을 이름으로 정의하는 recipe 레지스트리 모듈
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .modifiers import AWQModifier, BaseModifier, GPTQModifier, QuantizationModifier, SmoothQuantModifier
from .schemes import W8A8, W4A16, W8A8_DYNAMIC, QuantizationScheme

# recipe factory: (targets, ignore) → modifier 리스트.
# modifier는 hook·activation 통계 등 내부 상태를 가지므로 호출마다 새 인스턴스를 만든다.
RecipeFactory = Callable[
    [Optional[List[str]], Optional[List[str]]], List[BaseModifier]
]


def _rtn(scheme: QuantizationScheme) -> RecipeFactory:
    """단일 scheme RTN recipe — modifier 1개짜리 파이프라인 factory를 만든다."""

    def factory(
        targets: Optional[List[str]], ignore: Optional[List[str]]
    ) -> List[BaseModifier]:
        return [QuantizationModifier(scheme, targets=targets, ignore=ignore)]

    return factory


def _w4a16_gptq(
    targets: Optional[List[str]], ignore: Optional[List[str]]
) -> List[BaseModifier]:
    """GPTQ W4A16 — Hessian 기반 오차 전파로 RTN보다 낮은 reconstruction error를 달성한다."""
    return [GPTQModifier(W4A16, targets=targets, ignore=ignore)]


def _w8a8_smoothquant(
    targets: Optional[List[str]], ignore: Optional[List[str]]
) -> List[BaseModifier]:
    """SmoothQuant(α=0.5) → W8A8 dynamic RTN.

    activation outlier를 weight로 흡수한 뒤, per-token dynamic int8로 양자화한다.
    dynamic activation은 런타임에 토큰별 scale을 계산하므로 calibration 불필요.
    """
    return [
        SmoothQuantModifier(alpha=0.5),
        QuantizationModifier(W8A8_DYNAMIC, targets=targets, ignore=ignore),
    ]


def _w4a16_awq(
    targets: Optional[List[str]], ignore: Optional[List[str]]
) -> List[BaseModifier]:
    """AWQ → W4A16 RTN. activation magnitude 기반 grid-search scaling 후 INT4 RTN 적용한다."""
    return [
        AWQModifier(n_grid=20, group_size=128),
        QuantizationModifier(W4A16, targets=targets, ignore=ignore),
    ]


# 이름 → recipe factory. Compressor.from_recipe()의 유일한 preset 진입점.
# 단일 RTN scheme은 modifier 1개짜리 recipe로, SmoothQuant 같은 알고리즘은 여러
# modifier가 chain된 recipe로 — 모든 preset을 하나의 레지스트리로 표현한다.
RECIPE_REGISTRY: Dict[str, RecipeFactory] = {
    "w4a16": _rtn(W4A16),
    "w4a16_gptq": _w4a16_gptq,
    "w4a16_awq": _w4a16_awq,
    "w8a8": _rtn(W8A8),
    "w8a8_dynamic": _rtn(W8A8_DYNAMIC),
    "w8a8_smoothquant": _w8a8_smoothquant,
}
