"""Table export (FR-RPT-003): every report table as both CSV and Markdown, alongside
the machine-readable `summary.json` (`report/summary.py`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_table(df: pd.DataFrame, output_dir: Path, name: str) -> tuple[Path, Path]:
    """
    Write `df` as `<output_dir>/<name>.csv` and `<output_dir>/<name>.md`.

    Parameters
    ----------
    df : pandas.DataFrame
    output_dir : Path
        Created if missing.
    name : str
        File stem (no extension).

    Returns
    -------
    tuple[Path, Path]
        `(csv_path, markdown_path)`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}.csv"
    md_path = output_dir / f"{name}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False))
    return csv_path, md_path
