# 양자화 단위 명세(QuantizationSpec)와 scheme 프리셋(W8A8, W4A16)을 정의하는 모듈
from dataclasses import dataclass
from typing import Optional


@dataclass
class QuantizationSpec:
    num_bits: int
    symmetric: bool
    granularity: str        # "per_tensor" | "per_channel" | "per_group"
    dtype: str              # "int" | "float8"
    group_size: Optional[int] = None  # per_group일 때만 유효
    axis: Optional[int] = None        # per_channel일 때만 유효, 보통 0
    dynamic: bool = False
    calibration_method: str = "minmax"  # "minmax" | "percentile" | "mse" | "kl_divergence"


@dataclass
class QuantizationScheme:
    name: str
    weight: QuantizationSpec
    activation: Optional[QuantizationSpec] = None


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

W4A16 = QuantizationScheme(
    name="w4a16",
    weight=QuantizationSpec(
        num_bits=4,
        symmetric=True,
        granularity="per_group",
        dtype="int",
        group_size=128,
        axis=1,  # in_features 방향으로 group 분할 (Quark 기준)
    ),
    activation=None,
)

SCHEME_REGISTRY = {
    "w8a8": W8A8,
    "w4a16": W4A16,
}
