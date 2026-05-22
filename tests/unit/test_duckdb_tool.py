"""
tests/unit/test_duckdb_tool.py
───────────────────────────────
Unit tests for DuckDBTool.

Uses real DuckDB in-memory connections against temp CSV/Parquet files —
no mocking needed since DuckDB is an in-process engine with no network calls.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")

import polars as pl

from pipeline.agents.etl.tools.duckdb_tool import DuckDBTool
from pipeline.core.exceptions import (
    DataSourceNotFoundError,
    ETLError,
    UnsupportedFileFormatError,
)
from pipeline.core.schemas import DataSource, DataSourceType


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def csv_file(tmp_path):
    p = tmp_path / "sales.csv"
    p.write_text(
        "date,revenue,units,region\n"
        "2024-01-01,1000.0,10,north\n"
        "2024-01-02,1200.5,12,south\n"
        "2024-01-03,950.0,9,north\n"
        "2024-01-04,1500.0,15,east\n"
        "2024-01-05,800.0,8,west\n"
    )
    return str(p)


@pytest.fixture
def parquet_file(tmp_path):
    p = tmp_path / "metrics.parquet"
    df = pl.DataFrame({
        "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "cpu_pct": [45.2, 92.1, 38.7],
        "mem_gb": [8.1, 15.9, 7.2],
    })
    df.write_parquet(str(p))
    return str(p)


@pytest.fixture
def json_file(tmp_path):
    p = tmp_path / "events.json"
    p.write_text(
        '[{"id": 1, "event": "click", "value": 10},'
        '{"id": 2, "event": "view", "value": 5},'
        '{"id": 3, "event": "click", "value": 20}]'
    )
    return str(p)


@pytest.fixture
def db():
    tool = DuckDBTool(row_limit=10_000)
    yield tool
    tool.close()


# ─── DuckDBTool: basic construction ───────────────────────────────────────────


class TestDuckDBToolInit:
    def test_creates_in_memory_connection(self):
        db = DuckDBTool()
        assert db._conn is not None
        db.close()

    def test_default_row_limit(self):
        db = DuckDBTool()
        assert db.row_limit == 500_000
        db.close()

    def test_custom_row_limit(self):
        db = DuckDBTool(row_limit=1000)
        assert db.row_limit == 1000
        db.close()

    def test_context_manager(self):
        with DuckDBTool() as db:
            result = db.execute("SELECT 42 AS answer")
            assert result["answer"][0] == 42

    def test_registered_views_empty_on_init(self):
        db = DuckDBTool()
        assert db.list_views() == []
        db.close()


# ─── CSV registration and querying ────────────────────────────────────────────


class TestCSVSource:
    def test_register_csv_source(self, db, csv_file):
        source = DataSource(uri=csv_file, source_type=DataSourceType.CSV)
        view_name = db.register_source(source)
        assert view_name in db.list_views()

    def test_query_csv_returns_dataframe(self, db, csv_file):
        source = DataSource(uri=csv_file, source_type=DataSourceType.CSV)
        view_name = db.register_source(source)
        df = db.execute(f"SELECT * FROM {view_name}")
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 5

    def test_csv_column_names_correct(self, db, csv_file):
        source = DataSource(uri=csv_file, source_type=DataSourceType.CSV)
        view_name = db.register_source(source)
        df = db.execute(f"SELECT * FROM {view_name}")
        assert set(df.columns) == {"date", "revenue", "units", "region"}

    def test_csv_custom_table_name(self, db, csv_file):
        source = DataSource(uri=csv_file, table_name="my_sales", source_type=DataSourceType.CSV)
        view_name = db.register_source(source)
        assert view_name == "my_sales"

    def test_csv_auto_detects_type(self, db, csv_file):
        source = DataSource(uri=csv_file)  # No source_type specified
        view_name = db.register_source(source)
        df = db.execute(f"SELECT COUNT(*) AS n FROM {view_name}")
        assert df["n"][0] == 5


class TestParquetSource:
    def test_register_parquet_source(self, db, parquet_file):
        source = DataSource(uri=parquet_file, source_type=DataSourceType.PARQUET)
        view_name = db.register_source(source)
        df = db.execute(f"SELECT * FROM {view_name}")
        assert len(df) == 3
        assert "cpu_pct" in df.columns

    def test_parquet_numeric_types_preserved(self, db, parquet_file):
        source = DataSource(uri=parquet_file, source_type=DataSourceType.PARQUET)
        view_name = db.register_source(source)
        df = db.execute(f"SELECT * FROM {view_name}")
        assert df["cpu_pct"].dtype in (pl.Float32, pl.Float64)


class TestJSONSource:
    def test_register_json_source(self, db, json_file):
        source = DataSource(uri=json_file, source_type=DataSourceType.JSON)
        view_name = db.register_source(source)
        df = db.execute(f"SELECT * FROM {view_name}")
        assert len(df) == 3
        assert "event" in df.columns


# ─── Row limit enforcement ────────────────────────────────────────────────────


class TestRowLimit:
    def test_row_limit_applied_to_query(self, tmp_path):
        # Write 20 rows
        p = tmp_path / "big.csv"
        rows = "\n".join([f"2024-01-{i:02d},{i*100},{i}" for i in range(1, 21)])
        p.write_text("date,revenue,units\n" + rows)

        with DuckDBTool(row_limit=5) as db:
            source = DataSource(uri=str(p), source_type=DataSourceType.CSV)
            view_name = db.register_source(source)
            df = db.execute(f"SELECT * FROM {view_name}")
            assert len(df) == 5

    def test_explicit_limit_in_query_respected(self, tmp_path):
        p = tmp_path / "data.csv"
        rows = "\n".join([f"{i},{i*10}" for i in range(1, 11)])
        p.write_text("id,value\n" + rows)

        with DuckDBTool(row_limit=100) as db:
            source = DataSource(uri=str(p))
            view_name = db.register_source(source)
            # Explicit LIMIT in query should not be double-wrapped
            df = db.execute(f"SELECT * FROM {view_name} LIMIT 3")
            assert len(df) == 3


# ─── SQL queries ──────────────────────────────────────────────────────────────


class TestSQLQueries:
    def test_aggregation_query(self, db, csv_file):
        source = DataSource(uri=csv_file)
        view_name = db.register_source(source)
        df = db.execute(f"SELECT SUM(revenue) AS total FROM {view_name}")
        assert df["total"][0] == pytest.approx(5450.5)

    def test_filter_query(self, db, csv_file):
        source = DataSource(uri=csv_file)
        view_name = db.register_source(source)
        df = db.execute(f"SELECT * FROM {view_name} WHERE region = 'north'")
        assert len(df) == 2

    def test_row_count_helper(self, db, csv_file):
        source = DataSource(uri=csv_file)
        view_name = db.register_source(source)
        count = db.row_count(view_name)
        assert count == 5

    def test_sample_helper(self, db, csv_file):
        source = DataSource(uri=csv_file)
        view_name = db.register_source(source)
        sample = db.sample(view_name, n=3)
        assert len(sample) == 3

    def test_column_names_helper(self, db, csv_file):
        source = DataSource(uri=csv_file)
        view_name = db.register_source(source)
        cols = db.column_names(view_name)
        assert set(cols) == {"date", "revenue", "units", "region"}

    def test_execute_raw_returns_tuples(self, db, csv_file):
        source = DataSource(uri=csv_file)
        view_name = db.register_source(source)
        rows = db.execute_raw(f"SELECT COUNT(*) FROM {view_name}")
        assert rows[0][0] == 5


# ─── Error handling ───────────────────────────────────────────────────────────


class TestDuckDBToolErrors:
    def test_missing_file_raises(self, db):
        source = DataSource(uri="/nonexistent/path/data.csv", source_type=DataSourceType.CSV)
        with pytest.raises(DataSourceNotFoundError):
            db.register_source(source)

    def test_unsupported_extension_raises(self, db, tmp_path):
        p = tmp_path / "data.xlsx"
        p.write_text("fake content")
        source = DataSource(uri=str(p), source_type=DataSourceType.UNKNOWN)
        with pytest.raises(UnsupportedFileFormatError):
            db.register_source(source)

    def test_invalid_sql_raises(self, db, csv_file):
        source = DataSource(uri=csv_file)
        db.register_source(source)
        with pytest.raises(ETLError):
            db.execute("SELECT * FROM nonexistent_view_xyz")

    def test_multiple_sources_registered(self, db, csv_file, parquet_file):
        s1 = DataSource(uri=csv_file, table_name="sales")
        s2 = DataSource(uri=parquet_file, table_name="metrics")
        db.register_source(s1)
        db.register_source(s2)
        assert set(db.list_views()) == {"sales", "metrics"}
