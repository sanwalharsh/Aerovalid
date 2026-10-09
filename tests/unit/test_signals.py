from __future__ import annotations

import pytest

from aerovalid.core.errors import SignalRegistryError
from aerovalid.core.signals import get_signal, known_signals


@pytest.mark.req("FR-COR-004")
def test_known_signal_lookup() -> None:
    spec = get_signal("alt_m")
    assert spec.unit == "m"
    assert spec.default_tolerance == 5.0


@pytest.mark.req("FR-COR-004")
def test_unknown_signal_raises_with_clear_message() -> None:
    with pytest.raises(SignalRegistryError, match="Unknown signal 'not_a_signal'"):
        get_signal("not_a_signal")


@pytest.mark.req("FR-COR-004")
def test_known_signals_sorted_and_nonempty() -> None:
    names = known_signals()
    assert names == sorted(names)
    assert "alt_m" in names
    assert "vel_air_mps" in names


@pytest.mark.req("FR-COR-004")
def test_signal_spec_is_frozen() -> None:
    spec = get_signal("alt_m")
    with pytest.raises(Exception):  # noqa: B017 - pydantic frozen model raises its own error type
        spec.unit = "ft"
