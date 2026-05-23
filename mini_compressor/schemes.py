# 양자화 단위 명세(QuantizationSpec)와 scheme 프리셋(W8A8, W4A16)을 정의하는 모듈
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class QuantizationSpec:
    num_bits: int
    symmetric: bool
    granularity: str        # "per_tensor" | "per_channel" | "per_group" | "per_token"
    dtype: str              # "int" | "float8"
    group_size: Optional[int] = None  # per_group일 때만 유효
    axis: Optional[int] = None        # per_channel일 때만 유효, 보통 0
    dynamic: bool = False             # True: 런타임 scale 계산 (calibration 불필요)
    calibration_method: str = "minmax"  # "minmax" | "percentile" | "mse"

    def __post_init__(self) -> None:
        valid_granularities = {"per_tensor", "per_channel", "per_group", "per_token"}
        if self.granularity not in valid_granularities:
            raise ValueError(
                f"Unknown granularity '{self.granularity}'. "
                f"Choose from {sorted(valid_granularities)}."
            )

        valid_dtypes = {"int", "float8"}
        if self.dtype not in valid_dtypes:
            raise ValueError(
                f"Unknown dtype '{self.dtype}'. Choose from {sorted(valid_dtypes)}."
            )

        valid_methods = {"minmax", "percentile", "mse"}
        if self.calibration_method not in valid_methods:
            raise ValueError(
                f"Unknown calibration_method '{self.calibration_method}'. "
                f"Choose from {sorted(valid_methods)}."
            )

        if self.num_bits <= 0:
            raise ValueError(f"num_bits must be positive, got {self.num_bits}.")

        if self.granularity == "per_group":
            if self.group_size is None or self.group_size <= 0:
                raise ValueError("per_group quantization requires positive group_size.")
        elif self.group_size is not None:
            raise ValueError("group_size is only valid for per_group quantization.")

        if self.granularity == "per_channel" and self.axis is None:
            raise ValueError("per_channel quantization requires axis.")

        if self.granularity == "per_token" and not self.dynamic:
            raise ValueError(
                "per_token activation quantization must be dynamic=True. "
                "Static per-token scale cannot be reused for unseen runtime tokens."
            )


@dataclass(frozen=True)
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

W8A8_DYNAMIC = QuantizationScheme(
    name="w8a8_dynamic",
    weight=QuantizationSpec(
        num_bits=8,
        symmetric=True,
        granularity="per_channel",
        dtype="int",
        axis=0,
    ),
    activation=QuantizationSpec(
        num_bits=8,
        symmetric=True,
        granularity="per_token",
        dtype="int",
        dynamic=True,
    ),
)

SCHEME_REGISTRY = {
    "w8a8": W8A8,
    "w4a16": W4A16,
    "w8a8_dynamic": W8A8_DYNAMIC,
}
