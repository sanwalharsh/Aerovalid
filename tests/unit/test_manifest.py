from __future__ import annotations

from pathlib import Path

import pytest

from aerovalid.core.errors import ManifestError
from aerovalid.core.manifest import (
    create_manifest,
    hash_config,
    read_manifest,
    update_step,
    write_manifest,
)


@pytest.mark.req("FR-COR-002")
def test_hash_config_is_deterministic_and_prefixed() -> None:
    h1 = hash_config("study:\n  seed: 1\n")
    h2 = hash_config("study:\n  seed: 1\n")
    h3 = hash_config("study:\n  seed: 2\n")
    assert h1 == h2
    assert h1 != h3
    assert h1.startswith("sha256:")


@pytest.mark.req("FR-COR-002")
def test_create_manifest_has_required_fields() -> None:
    manifest = create_manifest(run_id="r1", config_hash=hash_config("x"), master_seed=42)
    assert manifest.run_id == "r1"
    assert manifest.master_seed == 42
    assert manifest.config_hash.startswith("sha256:")
    assert "aerovalid" in manifest.package_versions
    assert manifest.steps == {}


@pytest.mark.req("FR-COR-002")
def test_write_then_read_manifest_round_trips(tmp_path: Path) -> None:
    manifest = create_manifest(run_id="r2", config_hash=hash_config("x"), master_seed=1)
    run_dir = tmp_path / "runs" / "r2"
    write_manifest(manifest, run_dir)

    reloaded = read_manifest(run_dir)
    assert reloaded == manifest
    assert (run_dir / "run_manifest.json").exists()


@pytest.mark.req("FR-COR-002")
def test_update_step_is_immutable_and_adds_record() -> None:
    manifest = create_manifest(run_id="r3", config_hash=hash_config("x"), master_seed=1)
    updated = update_step(
        manifest,
        "generate",
        "completed",
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:00:47Z",
        duration_s=47.2,
        output_hashes={"raw/telemetry/F0001.parquet": "sha256:abc"},
    )
    assert manifest.steps == {}  # original untouched
    assert updated.steps["generate"].status == "completed"
    assert updated.steps["generate"].duration_s == 47.2
    assert updated.steps["generate"].output_hashes["raw/telemetry/F0001.parquet"] == "sha256:abc"


@pytest.mark.req("FR-COR-002")
def test_read_missing_manifest_raises_manifest_error(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="No manifest found"):
        read_manifest(tmp_path / "nonexistent_run")


@pytest.mark.req("FR-COR-002")
def test_read_corrupt_manifest_raises_manifest_error(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text("not json")
    with pytest.raises(ManifestError, match="corrupt"):
        read_manifest(run_dir)
