import pytest

from mini_compressor.schemes import QuantizationSpec


def test_per_token_static_rejected():
    with pytest.raises(ValueError, match="per_token.*dynamic=True"):
        QuantizationSpec(
            num_bits=8,
            symmetric=True,
            granularity="per_token",
            dtype="int",
            dynamic=False,
        )


def test_per_token_dynamic_allowed():
    spec = QuantizationSpec(
        num_bits=8,
        symmetric=True,
        granularity="per_token",
        dtype="int",
        dynamic=True,
    )
    assert spec.dynamic is True


def test_per_group_requires_group_size():
    with pytest.raises(ValueError, match="group_size"):
        QuantizationSpec(
            num_bits=4,
            symmetric=True,
            granularity="per_group",
            dtype="int",
        )


def test_group_size_only_valid_for_per_group():
    with pytest.raises(ValueError, match="group_size.*per_group"):
        QuantizationSpec(
            num_bits=8,
            symmetric=True,
            granularity="per_channel",
            dtype="int",
            axis=0,
            group_size=128,
        )


def test_per_channel_requires_axis():
    with pytest.raises(ValueError, match="axis"):
        QuantizationSpec(
            num_bits=8,
            symmetric=True,
            granularity="per_channel",
            dtype="int",
        )
