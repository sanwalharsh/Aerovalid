"""Shared pytest fixtures. Phase 0 only needs tmp-path config fixtures; later phases add
tiny synthetic-flight fixtures (<=60 s, fixed seeds) here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def default_config_path() -> Path:
    """Path to the repo's real `configs/default.yaml` (read-only)."""
    return Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def minimal_config_path(tmp_path: Path) -> Path:
    """Write a minimal config (relies on defaults for everything else) and return its path."""
    path = tmp_path / "minimal.yaml"
    path.write_text(yaml.safe_dump({"study": {"name": "unit-test", "seed": 7}}))
    return path
