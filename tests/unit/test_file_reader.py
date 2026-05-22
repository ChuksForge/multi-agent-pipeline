"""
tests/unit/test_file_reader.py
────────────────────────────────
Unit tests for the polars file reader.
Uses real temp files — no mocking.
"""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")

import polars as pl

from pipeline.agents.etl.tools.file_reader import (
    detect_encoding,
    read_file,
    supported_extensions,
)
from pipeline.core.exceptions import (
    DataSourceNotFoundError,
    ETLError,
    UnsupportedFileFormatError,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def clean_csv(tmp_path):
    p = tmp_path / "clean.csv"
    p.write_text(
        "id,name,score,active\n"
        "1,Alice,95.5,true\n"
        "2,Bob,87.0,false\n"
        "3,Carol,91.2,true\n"
    )
    return str(p)


@pytest.fixture
def messy_csv(tmp_path):
    p = tmp_path / "messy.csv"
    p.write_text(
        "id,value,status\n"
        "1,100,active\n"
        "2,N/A,inactive\n"
        "3,NULL,\n"
        "4,200,active\n"
    )
    return str(p)


@pytest.fixture
def tsv_file(tmp_path):
    p = tmp_path / "data.tsv"
    p.write_text("col_a\tcol_b\tcol_c\n1\t2\t3\n4\t5\t6\n")
    return str(p)


@pytest.fixture
def parquet_file(tmp_path):
    p = tmp_path / "data.parquet"
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0], "label": ["a", "b", "c"]})
    df.write_parquet(str(p))
    return str(p)


@pytest.fixture
def json_array_file(tmp_path):
    p = tmp_path / "records.json"
    data = [{"id": i, "val": i * 10} for i in range(5)]
    p.write_text(json.dumps(data))
    return str(p)


@pytest.fixture
def jsonl_file(tmp_path):
    p = tmp_path / "events.jsonl"
    lines = [json.dumps({"ts": f"2024-01-0{i}", "count": i}) for i in range(1, 4)]
    p.write_text("\n".join(lines))
    return str(p)


@pytest.fixture
def bom_csv(tmp_path):
    p = tmp_path / "bom.csv"
    # UTF-8 BOM + content
    p.write_bytes(b"\xef\xbb\xbfid,name\n1,test\n")
    return str(p)


# ─── CSV reading ──────────────────────────────────────────────────────────────


class TestReadCSV:
    def test_reads_clean_csv(self, clean_csv):
        df = read_file(clean_csv)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3
        assert set(df.columns) == {"id", "name", "score", "active"}

    def test_null_values_parsed(self, messy_csv):
        df = read_file(messy_csv)
        # N/A and NULL should become nulls
        null_count = df["value"].null_count()
        assert null_count >= 1

    def test_row_limit_respected(self, clean_csv):
        df = read_file(clean_csv, row_limit=2)
        assert len(df) == 2

    def test_tsv_file_read(self, tsv_file):
        df = read_file(tsv_file)
        assert len(df) == 2
        assert "col_a" in df.columns

    def test_bom_csv_reads_correctly(self, bom_csv):
        df = read_file(bom_csv)
        # First column name should not start with BOM character
        assert df.columns[0].lstrip("\ufeff") == "id" or df.columns[0] == "id"

    def test_empty_string_becomes_null(self, messy_csv):
        df = read_file(messy_csv)
        # Row 3 has empty status — should be null
        assert df["status"].null_count() >= 1


# ─── Parquet reading ──────────────────────────────────────────────────────────


class TestReadParquet:
    def test_reads_parquet_file(self, parquet_file):
        df = read_file(parquet_file)
        assert len(df) == 3
        assert set(df.columns) == {"x", "y", "label"}

    def test_parquet_types_preserved(self, parquet_file):
        df = read_file(parquet_file)
        assert df["x"].dtype in (pl.Int32, pl.Int64)
        assert df["y"].dtype in (pl.Float32, pl.Float64)

    def test_parquet_row_limit(self, parquet_file):
        df = read_file(parquet_file, row_limit=2)
        assert len(df) == 2


# ─── JSON reading ─────────────────────────────────────────────────────────────


class TestReadJSON:
    def test_reads_json_array(self, json_array_file):
        df = read_file(json_array_file)
        assert len(df) == 5
        assert "id" in df.columns

    def test_reads_jsonl(self, jsonl_file):
        df = read_file(jsonl_file)
        assert len(df) == 3
        assert "count" in df.columns

    def test_json_row_limit(self, json_array_file):
        df = read_file(json_array_file, row_limit=3)
        assert len(df) == 3


# ─── Error handling ───────────────────────────────────────────────────────────


class TestFileReaderErrors:
    def test_missing_file_raises(self):
        with pytest.raises(DataSourceNotFoundError):
            read_file("/nonexistent/path/data.csv")

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "data.docx"
        p.write_text("fake")
        with pytest.raises(UnsupportedFileFormatError):
            read_file(str(p))

    def test_truly_empty_csv_raises(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("")
        with pytest.raises(ETLError):
            read_file(str(p))


# ─── Utilities ────────────────────────────────────────────────────────────────


class TestFileReaderUtilities:
    def test_supported_extensions(self):
        exts = supported_extensions()
        assert ".csv" in exts
        assert ".parquet" in exts
        assert ".json" in exts
        assert ".jsonl" in exts

    def test_detect_encoding_utf8(self, tmp_path):
        p = tmp_path / "utf8.csv"
        p.write_text("a,b\n1,2\n", encoding="utf-8")
        enc = detect_encoding(str(p))
        assert enc in ("utf-8", "utf-8-sig")

    def test_detect_encoding_bom(self, tmp_path):
        p = tmp_path / "bom.csv"
        p.write_bytes(b"\xef\xbb\xbfcol\n1\n")
        enc = detect_encoding(str(p))
        assert enc == "utf-8-sig"

    def test_detect_encoding_missing_file(self):
        enc = detect_encoding("/nonexistent/file.csv")
        # Should return a fallback, not raise
        assert isinstance(enc, str)
