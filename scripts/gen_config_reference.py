"""Regenerate `docs/CONFIG_REFERENCE.md` from the `StudyConfig` pydantic model tree.

Run via `make config-ref`. Never hand-edit `docs/CONFIG_REFERENCE.md`; it is derived.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import get_args

from pydantic import BaseModel
from pydantic.fields import FieldInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aerovalid.core.config import StudyConfig  # noqa: E402

_OUT_PATH = Path(__file__).resolve().parents[1] / "docs" / "CONFIG_REFERENCE.md"


def _is_model(annotation: object) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _type_name(annotation: object) -> str:
    return getattr(annotation, "__name__", None) or str(annotation)


def _render_field(name: str, field: FieldInfo) -> str:
    yaml_key = field.alias or name  # aliased fields show their YAML key, e.g. validation->validate
    default = (
        "required" if field.is_required() else repr(field.get_default(call_default_factory=True))
    )
    description = field.description or ""
    return f"| `{yaml_key}` | `{_type_name(field.annotation)}` | {default} | {description} |"


def _render_model(
    model: type[BaseModel], heading: str, lines: list[str], seen: set[type[BaseModel]]
) -> None:
    if model in seen:
        return
    seen.add(model)

    lines.append(f"\n### {heading}\n")
    lines.append("| Field | Type | Default | Description |")
    lines.append("|---|---|---|---|")
    nested: list[tuple[str, type[BaseModel]]] = []

    for name, field in model.model_fields.items():
        lines.append(_render_field(name, field))
        annotation = field.annotation
        if _is_model(annotation):
            nested.append((f"{heading}.{name}", annotation))  # type: ignore[arg-type]
        else:
            for arg in get_args(annotation):
                if _is_model(arg):
                    nested.append((f"{heading}.{name}", arg))

    for sub_heading, sub_model in nested:
        _render_model(sub_model, sub_heading, lines, seen)


def main() -> None:
    lines: list[str] = [
        "# Config Reference",
        "",
        "Auto-generated from `src/aerovalid/core/config.py` pydantic models by",
        "`scripts/gen_config_reference.py`. Do not hand-edit; run `make config-ref`.",
    ]
    _render_model(StudyConfig, "StudyConfig", lines, seen=set())
    _OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {_OUT_PATH}")


if __name__ == "__main__":
    main()
