from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aerovalid.report.tables import write_table


@pytest.mark.req("FR-RPT-003")
def test_write_table_creates_csv_and_markdown(tmp_path: Path) -> None:
    df = pd.DataFrame({"variant": ["V0", "V1"], "rmse": [1.234, 2.345]})
    csv_path, md_path = write_table(df, tmp_path, "metrics")

    assert csv_path == tmp_path / "metrics.csv"
    assert md_path == tmp_path / "metrics.md"
    assert csv_path.exists()
    assert md_path.exists()

    reloaded = pd.read_csv(csv_path)
    pd.testing.assert_frame_equal(reloaded, df)

    md_text = md_path.read_text()
    assert "V0" in md_text
    assert "rmse" in md_text


@pytest.mark.req("FR-RPT-003")
def test_write_table_creates_output_dir(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b"
    df = pd.DataFrame({"x": [1]})
    write_table(df, nested, "t")
    assert (nested / "t.csv").exists()
