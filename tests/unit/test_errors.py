from __future__ import annotations

from aerovalid.core.errors import (
    AeroValidError,
    ConfigError,
    ConvergenceError,
    DataQualityError,
    ManifestError,
    SchemaError,
    SignalRegistryError,
    SimulationError,
)


def test_all_errors_subclass_aerovaliderror() -> None:
    for cls in (
        ConfigError,
        SchemaError,
        SignalRegistryError,
        ManifestError,
        SimulationError,
        DataQualityError,
        ConvergenceError,
    ):
        assert issubclass(cls, AeroValidError)
        assert issubclass(cls, Exception)


def test_error_message_is_preserved() -> None:
    try:
        raise ConfigError("field 'study.seed' is invalid")
    except AeroValidError as exc:
        assert "study.seed" in str(exc)
