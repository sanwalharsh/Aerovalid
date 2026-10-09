"""Signal registry (FR-COR-004): canonical name -> unit, frame, description, default
tolerance, plausible range. Authoritative source for QC, plotting, and metrics; used by
`core/schema.py` to know which signal columns a canonical table may carry.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from aerovalid.core.errors import SignalRegistryError


class SignalSpec(BaseModel):
    """Metadata for one canonical signal column (data spec §4.1)."""

    model_config = ConfigDict(frozen=True)

    name: str
    symbol: str
    unit: str
    frame: str
    description: str
    default_tolerance: float | None
    plausible_range: tuple[float, float] | None = None


_REGISTRY: dict[str, SignalSpec] = {
    spec.name: spec
    for spec in (
        SignalSpec(
            name="x_m",
            symbol="x",
            unit="m",
            frame="local NE plane from start",
            description="Local north position",
            default_tolerance=15.0,
        ),
        SignalSpec(
            name="y_m",
            symbol="y",
            unit="m",
            frame="local NE plane from start",
            description="Local east position",
            default_tolerance=15.0,
        ),
        SignalSpec(
            name="alt_m",
            symbol="h",
            unit="m",
            frame="MSL, up positive",
            description="Altitude",
            default_tolerance=5.0,
            plausible_range=(-500.0, 20000.0),
        ),
        SignalSpec(
            name="vel_air_mps",
            symbol="V_a",
            unit="m/s",
            frame="body/wind",
            description="True airspeed",
            default_tolerance=1.5,
            plausible_range=(0.0, 400.0),
        ),
        SignalSpec(
            name="vel_gnd_mps",
            symbol="V_g",
            unit="m/s",
            frame="inertial",
            description="Ground speed",
            default_tolerance=1.5,
            plausible_range=(0.0, 400.0),
        ),
        SignalSpec(
            name="gamma_rad",
            symbol="gamma",
            unit="rad",
            frame="inertial",
            description="Flight-path angle",
            default_tolerance=0.02,
            plausible_range=(-1.6, 1.6),
        ),
        SignalSpec(
            name="chi_rad",
            symbol="chi",
            unit="rad",
            frame="0 = North, clockwise",
            description="Track angle",
            default_tolerance=0.03,
        ),
        SignalSpec(
            name="roll_rad",
            symbol="phi",
            unit="rad",
            frame="3-2-1 Euler",
            description="Roll (bank) attitude",
            default_tolerance=0.03,
        ),
        SignalSpec(
            name="pitch_rad",
            symbol="theta",
            unit="rad",
            frame="3-2-1 Euler",
            description="Pitch attitude",
            default_tolerance=0.03,
        ),
        SignalSpec(
            name="yaw_rad",
            symbol="psi",
            unit="rad",
            frame="3-2-1 Euler",
            description="Heading",
            default_tolerance=0.03,
        ),
        SignalSpec(
            name="alpha_rad",
            symbol="alpha",
            unit="rad",
            frame="body/wind",
            description="Angle of attack (optional)",
            default_tolerance=0.01,
        ),
        SignalSpec(
            name="nx_g",
            symbol="n_x",
            unit="g",
            frame="body, specific-force based",
            description="Longitudinal load factor",
            default_tolerance=0.08,
        ),
        SignalSpec(
            name="nz_g",
            symbol="n_z",
            unit="g",
            frame="body, specific-force based",
            description="Normal load factor",
            default_tolerance=0.08,
        ),
        SignalSpec(
            name="throttle",
            symbol="delta_t",
            unit="-",
            frame="control, 0-1",
            description="Throttle command",
            default_tolerance=None,
            plausible_range=(0.0, 1.0),
        ),
        SignalSpec(
            name="bank_cmd_rad",
            symbol="phi_cmd",
            unit="rad",
            frame="control",
            description="Commanded bank",
            default_tolerance=None,
        ),
        SignalSpec(
            name="alpha_cmd_rad",
            symbol="alpha_cmd",
            unit="rad",
            frame="control",
            description="Commanded angle of attack",
            default_tolerance=None,
        ),
        SignalSpec(
            name="mass_kg",
            symbol="m",
            unit="kg",
            frame="-",
            description="Vehicle mass (optional)",
            default_tolerance=None,
            plausible_range=(1.0, 2000.0),
        ),
        SignalSpec(
            name="wind_n_mps",
            symbol="W_n",
            unit="m/s",
            frame="inertial, optional",
            description="Wind, north component",
            default_tolerance=None,
        ),
        SignalSpec(
            name="wind_e_mps",
            symbol="W_e",
            unit="m/s",
            frame="inertial, optional",
            description="Wind, east component",
            default_tolerance=None,
        ),
        SignalSpec(
            name="wind_d_mps",
            symbol="W_d",
            unit="m/s",
            frame="inertial, optional",
            description="Wind, down component",
            default_tolerance=None,
        ),
        SignalSpec(
            name="mach",
            symbol="M",
            unit="-",
            frame="derived",
            description="Mach number",
            default_tolerance=0.005,
            plausible_range=(0.0, 3.0),
        ),
        SignalSpec(
            name="q_pa",
            symbol="q",
            unit="Pa",
            frame="derived",
            description="Dynamic pressure",
            default_tolerance=30.0,
            plausible_range=(0.0, 1.0e6),
        ),
    )
}


def get_signal(name: str) -> SignalSpec:
    """
    Look up a signal's metadata by canonical name.

    Parameters
    ----------
    name : str
        Canonical signal column name, e.g. `"alt_m"`.

    Returns
    -------
    SignalSpec
        The signal's registry entry.

    Raises
    ------
    SignalRegistryError
        If `name` is not a known canonical signal.
    """
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY))
        raise SignalRegistryError(f"Unknown signal {name!r}. Known signals: {known}") from exc


def known_signals() -> list[str]:
    """Return all canonical signal names in the registry, sorted."""
    return sorted(_REGISTRY)
