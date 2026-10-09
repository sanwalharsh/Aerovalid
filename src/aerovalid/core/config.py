"""Typed configuration (FR-COR-001). `StudyConfig` is the pydantic-validated root of every
AeroValidate run, loaded from YAML with optional `include:` composition and dotted-path
`--set key=value` overrides. Every field has a default so a bare `study:` section is enough
to get a runnable config (data spec §7).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aerovalid.core.errors import ConfigError


class _StrictModel(BaseModel):
    """Base for all config models: unknown keys are rejected so typos surface immediately."""

    model_config = ConfigDict(extra="forbid")


class StudyMeta(_StrictModel):
    name: str = "demo"
    seed: int = 20260101
    profile: str = "demo"


class VehicleRef(_StrictModel):
    ref: str = "generic_uav_v1"


class FaultsConfig(_StrictModel):
    probability_per_flight: float = 0.5
    library: str = "configs/faults/"


class GenerateConditions(_StrictModel):
    alt0_m: tuple[float, float] = (500.0, 6000.0)
    v0_mps: tuple[float, float] = (28.0, 60.0)
    mass_kg: tuple[float, float] = (22.0, 28.0)
    wind_mean_mps: tuple[float, float] = (0.0, 10.0)
    turb: list[str] = Field(default_factory=lambda: ["none", "light", "moderate"])
    dT_K: tuple[float, float] = (-10.0, 15.0)  # noqa: N815 -- spec key name, data spec §7


class GenerateConfig(_StrictModel):
    n_flights: int = 12
    scenarios: list[str] = Field(
        default_factory=lambda: ["mixed_mission", "climb_cruise", "turn_series", "doublet_series"]
    )
    conditions: GenerateConditions = Field(default_factory=GenerateConditions)
    sampling: str = "sobol"
    telemetry_rate_hz: float = 10.0
    truth_rate_hz: float = 100.0
    faults: FaultsConfig = Field(default_factory=FaultsConfig)


class HampelConfig(_StrictModel):
    window: int = 7
    n_sigma: float = 4.0


class SyncConfig(_StrictModel):
    method: str = "xcorr"
    ref_signals: list[str] = Field(default_factory=lambda: ["alt_m", "vel_air_mps"])
    max_lag_s: float = 3.0
    drift: bool = True


class SavitzkyGolayConfig(_StrictModel):
    window: int = 11
    order: int = 3


class PrepConfig(_StrictModel):
    grid_hz: float = 10.0
    max_gap_s: float = 0.5
    hampel: HampelConfig = Field(default_factory=HampelConfig)
    sync: SyncConfig = Field(default_factory=SyncConfig)
    sg: SavitzkyGolayConfig = Field(default_factory=SavitzkyGolayConfig)


class BootstrapConfig(_StrictModel):
    n: int = 2000
    block_s: float = 10.0


class ValidateConfig(_StrictModel):
    signals: list[str] = Field(
        default_factory=lambda: [
            "alt_m",
            "vel_air_mps",
            "gamma_rad",
            "chi_rad",
            "nz_g",
            "pitch_rad",
            "roll_rad",
            "x_m",
            "y_m",
        ]
    )
    horizons_s: list[float] = Field(default_factory=lambda: [1.0, 2.0, 5.0, 10.0])
    window_s: float = 10.0
    stride_s: float = 5.0
    weights: dict[str, float] = Field(default_factory=dict)
    tolerances: dict[str, float] = Field(default_factory=dict)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)


class CVConfig(_StrictModel):
    scheme: str = "loso"
    folds: int = 6


class ConformalConfig(_StrictModel):
    alpha: float = 0.1


class PredictConfig(_StrictModel):
    tasks: list[str] = Field(default_factory=lambda: ["residual", "hybrid", "forecast"])
    targets: list[str] = Field(default_factory=lambda: ["alt_m", "vel_air_mps"])
    horizons_s: list[float] = Field(default_factory=lambda: [1.0, 5.0, 10.0, 30.0])
    models: list[str] = Field(default_factory=lambda: ["ridge", "hgb", "rf"])
    seq: bool = False
    cv: CVConfig = Field(default_factory=CVConfig)
    conformal: ConformalConfig = Field(default_factory=ConformalConfig)


class EarlyWarningConfig(_StrictModel):
    horizon_s: float = 30.0
    p_star: float = 0.8


class DetectConfig(_StrictModel):
    target_far_per_hr: float = 1.0
    detectors: list[str] = Field(
        default_factory=lambda: ["zscore", "ewma", "cusum", "mahalanobis", "iforest", "pelt"]
    )
    early_warning: EarlyWarningConfig = Field(default_factory=EarlyWarningConfig)


class SobolConfig(_StrictModel):
    n: int = 1024


class RegimesConfig(_StrictModel):
    mach_bins: list[float] = Field(default_factory=lambda: [0.0, 0.12, 0.18, 0.25])


class IdentifyConfig(_StrictModel):
    params: list[str] = Field(default_factory=lambda: ["CL0", "CLa", "CD0", "k", "thrust_lapse_n"])
    bounds: dict[str, tuple[float, float]] = Field(default_factory=dict)
    sobol: SobolConfig = Field(default_factory=SobolConfig)
    regimes: RegimesConfig = Field(default_factory=RegimesConfig)
    revalidate: bool = True


class ReportConfig(_StrictModel):
    formats: list[str] = Field(default_factory=lambda: ["html"])
    pdf: bool = False


class StudyConfig(_StrictModel):
    """Root configuration for an AeroValidate study (data spec §7)."""

    study: StudyMeta = Field(default_factory=StudyMeta)
    vehicle: VehicleRef = Field(default_factory=VehicleRef)
    generate: GenerateConfig = Field(default_factory=GenerateConfig)
    variants: list[str] = Field(default_factory=lambda: ["V0", "V1", "V2", "V3", "V4", "V5"])
    preprocess: PrepConfig = Field(default_factory=PrepConfig)
    # Python attribute is `validation`, not `validate`: BaseModel.validate is a pydantic
    # v1-compat classmethod, and reusing its name breaks the mypy pydantic plugin (not
    # just a runtime warning). The YAML/CLI-facing key stays `validate` via alias
    # (data spec §7); see DECISIONS.md ADR-005.
    validation: ValidateConfig = Field(default_factory=ValidateConfig, alias="validate")
    predict: PredictConfig = Field(default_factory=PredictConfig)
    detect: DetectConfig = Field(default_factory=DetectConfig)
    identify: IdentifyConfig = Field(default_factory=IdentifyConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)


def _resolve_includes(raw: dict[str, Any], base_dir: Path, seen: frozenset[Path]) -> dict[str, Any]:
    """Merge any `include:` list of sub-config paths into `raw`, depth-first, raw wins."""
    include_paths = raw.pop("include", [])
    if isinstance(include_paths, str):
        include_paths = [include_paths]

    merged: dict[str, Any] = {}
    for rel_path in include_paths:
        include_path = (base_dir / rel_path).resolve()
        if include_path in seen:
            raise ConfigError(f"Circular `include:` detected at {include_path}")
        included_raw = _read_yaml(include_path)
        included_raw = _resolve_includes(included_raw, include_path.parent, seen | {include_path})
        merged = _deep_merge(merged, included_raw)

    return _deep_merge(merged, raw)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("r") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping, got {type(data).__name__}")
    return data


def _coerce(raw_value: str) -> object:
    """Parse a `--set` override's RHS as YAML so numbers/bools/lists work, falling back to str."""
    try:
        return yaml.safe_load(raw_value)
    except yaml.YAMLError:
        return raw_value


def _apply_override(raw: dict[str, Any], dotted_key: str, value: object) -> None:
    parts = dotted_key.split(".")
    node = raw
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def load_config(path: Path, overrides: Sequence[str] = ()) -> StudyConfig:
    """
    Load a YAML study config, resolve `include:` directives, apply `--set` overrides.

    Parameters
    ----------
    path : Path
        Path to the root YAML config file.
    overrides : list[str]
        Each entry is `"section.key=value"` (dotted path); `value` is parsed as YAML
        so ints/floats/bools/lists are typed correctly.

    Returns
    -------
    StudyConfig
        The fully validated, resolved configuration.

    Raises
    ------
    ConfigError
        If the file is missing, malformed, has circular includes, an override is
        malformed, or the resolved config fails pydantic validation -- the message
        names the offending field.
    """
    path = Path(path).resolve()
    raw = _read_yaml(path)
    raw = _resolve_includes(raw, path.parent, frozenset({path}))

    for override in overrides:
        if "=" not in override:
            raise ConfigError(f"Malformed --set override (expected key=value): {override!r}")
        dotted_key, _, raw_value = override.partition("=")
        _apply_override(raw, dotted_key.strip(), _coerce(raw_value.strip()))

    try:
        return StudyConfig.model_validate(raw)
    except ValidationError as exc:
        fields = "; ".join(".".join(str(p) for p in err["loc"]) for err in exc.errors())
        raise ConfigError(
            f"Config validation failed in {path} for field(s): {fields}\n{exc}"
        ) from exc
