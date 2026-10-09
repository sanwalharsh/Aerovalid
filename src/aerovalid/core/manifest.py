"""`run_manifest.json` writer (FR-COR-002). One manifest per run, updated incrementally
as each CLI step completes, recording enough to reproduce or audit the run: config hash,
git SHA, package versions, seeds, input/output artifact hashes, and step timings.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from pydantic import BaseModel, Field

from aerovalid.core.errors import ManifestError

_TRACKED_PACKAGES = (
    "aerovalid",
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "statsmodels",
    "pydantic",
    "pyyaml",
    "typer",
    "pandera",
    "joblib",
)


class StepRecord(BaseModel):
    """Timing and artifact record for one completed (or failed) CLI step."""

    status: str
    started_at: str
    completed_at: str | None = None
    duration_s: float | None = None
    input_hashes: dict[str, str] = Field(default_factory=dict)
    output_hashes: dict[str, str] = Field(default_factory=dict)
    extra: dict[str, object] = Field(default_factory=dict)


class RunManifest(BaseModel):
    """Full run manifest, serialised to `runs/<run_id>/run_manifest.json`."""

    run_id: str
    created_at: str
    git_sha: str | None
    config_hash: str
    python_version: str
    package_versions: dict[str, str]
    master_seed: int
    steps: dict[str, StepRecord] = Field(default_factory=dict)


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return result.stdout.strip() or None


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def hash_config(resolved_config_yaml: str) -> str:
    """
    SHA-256 of a resolved config's canonical YAML text, prefixed `sha256:`.

    Parameters
    ----------
    resolved_config_yaml : str
        The fully resolved config serialised as YAML (after includes/overrides).

    Returns
    -------
    str
        `"sha256:<hexdigest>"`.
    """
    digest = hashlib.sha256(resolved_config_yaml.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def hash_file(path: Path) -> str:
    """SHA-256 of a file's bytes, prefixed `sha256:`."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"sha256:{digest}"


def create_manifest(run_id: str, config_hash: str, master_seed: int) -> RunManifest:
    """
    Create a fresh manifest for a new run (not yet written to disk).

    Parameters
    ----------
    run_id : str
        Unique identifier for this run.
    config_hash : str
        Output of `hash_config` for the resolved config.
    master_seed : int
        The study's master seed.

    Returns
    -------
    RunManifest
    """
    return RunManifest(
        run_id=run_id,
        created_at=datetime.now(UTC).isoformat(),
        git_sha=_git_sha(),
        config_hash=config_hash,
        python_version=sys.version.split()[0],
        package_versions=_package_versions(),
        master_seed=master_seed,
    )


def manifest_path(run_dir: Path) -> Path:
    """Canonical path of a run's manifest file."""
    return Path(run_dir) / "run_manifest.json"


def write_manifest(manifest: RunManifest, run_dir: Path) -> Path:
    """
    Write `manifest` to `<run_dir>/run_manifest.json`, creating `run_dir` if needed.

    Returns
    -------
    Path
        The path written to.
    """
    out_path = manifest_path(run_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(manifest.model_dump_json(indent=2))
    return out_path


def read_manifest(run_dir: Path) -> RunManifest:
    """
    Read an existing manifest from `<run_dir>/run_manifest.json`.

    Raises
    ------
    ManifestError
        If the file is missing or not valid JSON matching `RunManifest`.
    """
    path = manifest_path(run_dir)
    if not path.exists():
        raise ManifestError(f"No manifest found at {path}")
    try:
        return RunManifest.model_validate_json(path.read_text())
    except (json.JSONDecodeError, ValueError) as exc:
        raise ManifestError(f"Manifest at {path} is corrupt: {exc}") from exc


def update_step(
    manifest: RunManifest,
    step_name: str,
    status: str,
    *,
    started_at: str,
    completed_at: str | None = None,
    duration_s: float | None = None,
    input_hashes: dict[str, str] | None = None,
    output_hashes: dict[str, str] | None = None,
    extra: dict[str, object] | None = None,
) -> RunManifest:
    """
    Return a new `RunManifest` with `step_name`'s record set.

    Parameters
    ----------
    manifest : RunManifest
        The manifest to update (not mutated; a copy is returned).
    step_name : str
        CLI step name, e.g. `"generate"`.
    status : str
        One of `"running"`, `"completed"`, `"failed"`, `"skipped"`.

    Returns
    -------
    RunManifest
        A new manifest with `steps[step_name]` set.
    """
    record = StepRecord(
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        duration_s=duration_s,
        input_hashes=input_hashes or {},
        output_hashes=output_hashes or {},
        extra=extra or {},
    )
    new_steps = {**manifest.steps, step_name: record}
    return manifest.model_copy(update={"steps": new_steps})
