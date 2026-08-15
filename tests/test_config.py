"""Configuration validation tests."""

from __future__ import annotations

import pytest

from lunaris_dense_irl2 import LunarisDenseConfig


def test_debug_profile_is_valid() -> None:
    config = LunarisDenseConfig.debug()
    assert config.head_dim == 32
    assert config.attention_type(0) == "local"
    assert config.attention_type(1) == "global"
    assert config.attention_type(2) == "local"


def test_unknown_configuration_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown configuration"):
        LunarisDenseConfig.from_dict({"unknown_field": 1})


def test_invalid_head_layout_is_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        LunarisDenseConfig.debug(d_model=130)


def test_flash_attention_backend_is_valid() -> None:
    config = LunarisDenseConfig.debug(attention_backend="flash")
    assert config.attention_backend == "flash"
