"""Exception hierarchy for AeroValidate. Every raised error is a subclass of AeroValidError
with an actionable message; no bare `except: pass` anywhere in the codebase (CLAUDE.md §2, §7).
"""

from __future__ import annotations


class AeroValidError(Exception):
    """Base class for every AeroValidate exception."""


class ConfigError(AeroValidError):
    """Config file failed to parse, resolve, or validate. Message names the offending field."""


class SchemaError(AeroValidError):
    """A DataFrame failed canonical-schema validation. Message lists all offending columns."""


class SignalRegistryError(AeroValidError):
    """An unknown signal name was requested from the signal registry."""


class ManifestError(AeroValidError):
    """The run manifest could not be read, written, or updated consistently."""


class SimulationError(AeroValidError):
    """A simulator run failed to produce a valid trajectory (e.g. integration blow-up)."""


class DataQualityError(AeroValidError):
    """Input data failed quality checks beyond what quarantine rules tolerate."""


class ConvergenceError(AeroValidError):
    """An iterative numerical procedure (fitting, calibration) failed to converge."""
