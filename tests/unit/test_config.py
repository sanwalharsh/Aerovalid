from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aerovalid.core.config import StudyConfig, load_config
from aerovalid.core.errors import ConfigError


@pytest.mark.req("FR-COR-001")
def test_default_config_loads(default_config_path: Path) -> None:
    cfg = load_config(default_config_path)
    assert isinstance(cfg, StudyConfig)
    assert cfg.study.name == "demo"
    assert cfg.generate.n_flights == 12
    assert cfg.variants == ["V0", "V1", "V2", "V3", "V4", "V5"]


@pytest.mark.req("FR-COR-001")
def test_minimal_config_uses_defaults(minimal_config_path: Path) -> None:
    cfg = load_config(minimal_config_path)
    assert cfg.study.name == "unit-test"
    assert cfg.study.seed == 7
    assert cfg.generate.n_flights == 12  # untouched default


@pytest.mark.req("FR-COR-001")
def test_round_trip_load_dump_equal(default_config_path: Path) -> None:
    cfg1 = load_config(default_config_path)
    dumped = yaml.safe_dump(cfg1.model_dump(mode="json", by_alias=True))
    reloaded = StudyConfig.model_validate(yaml.safe_load(dumped))
    assert cfg1 == reloaded


@pytest.mark.req("FR-COR-001")
def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does_not_exist.yaml")


@pytest.mark.req("FR-COR-001")
def test_unknown_field_raises_config_error_naming_field(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"study": {"seeed": 1}}))  # typo
    with pytest.raises(ConfigError, match="seeed"):
        load_config(path)


@pytest.mark.req("FR-COR-001")
def test_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("study: {name: [unclosed\n")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(path)


@pytest.mark.req("FR-COR-001")
def test_set_override_dotted_path(minimal_config_path: Path) -> None:
    cfg = load_config(minimal_config_path, overrides=["validate.bootstrap.n=500"])
    assert cfg.validation.bootstrap.n == 500
    assert cfg.validation.bootstrap.block_s == 10.0  # untouched sibling default


@pytest.mark.req("FR-COR-001")
def test_set_override_typed_values(minimal_config_path: Path) -> None:
    cfg = load_config(
        minimal_config_path,
        overrides=["generate.n_flights=3", "predict.seq=true", "variants=[V0, V5]"],
    )
    assert cfg.generate.n_flights == 3
    assert cfg.predict.seq is True
    assert cfg.variants == ["V0", "V5"]


@pytest.mark.req("FR-COR-001")
def test_malformed_override_raises(minimal_config_path: Path) -> None:
    with pytest.raises(ConfigError, match="Malformed"):
        load_config(minimal_config_path, overrides=["no_equals_sign_here"])


@pytest.mark.req("FR-COR-001")
def test_include_directive_merges(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"study": {"name": "base-study", "seed": 1}}))
    child = tmp_path / "child.yaml"
    child.write_text(yaml.safe_dump({"include": ["base.yaml"], "study": {"seed": 42}}))

    cfg = load_config(child)
    assert cfg.study.name == "base-study"  # inherited from include
    assert cfg.study.seed == 42  # overridden by child


@pytest.mark.req("FR-COR-001")
def test_circular_include_raises(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(yaml.safe_dump({"include": ["b.yaml"], "study": {"name": "a"}}))
    b.write_text(yaml.safe_dump({"include": ["a.yaml"], "study": {"name": "b"}}))

    with pytest.raises(ConfigError, match="Circular"):
        load_config(a)
