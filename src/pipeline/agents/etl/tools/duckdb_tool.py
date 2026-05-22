"""
agents/etl/tools/duckdb_tool.py
────────────────────────────────
Universal query tool using DuckDB.

DuckDB reads CSV, Parquet, JSON, S3, and Postgres through the same SQL
interface. This single tool replaces separate reader classes for each format.

Design decisions:
  - One persistent in-memory DuckDB connection per tool instance
  - All sources registered as views — agents query by view name, not path
  - Row limit enforced at query time via LIMIT clause — never reads full file
  - Returns polars DataFrame for zero-copy interop with the stats engine
  - Thread-safe via a per-instance connection (not shared across agents)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from pipeline.core.exceptions import (
    DataSourceNotFoundError,
    ETLError,
    UnsupportedFileFormatError,
)
from pipeline.core.schemas import DataSource, DataSourceType
from pipeline.middleware.logger import get_logger

logger = get_logger(__name__)

# Supported extensions → DuckDB reader function
_READER_MAP: dict[str, str] = {
    ".csv": "read_csv_auto",
    ".tsv": "read_csv_auto",
    ".parquet": "read_parquet",
    ".json": "read_json_auto",
    ".jsonl": "read_json_auto",
    ".ndjson": "read_json_auto",
}


class DuckDBTool:
    """
    Wraps a DuckDB in-memory connection.
    Registers each DataSource as a named view, then exposes
    execute() for arbitrary SQL queries against those views.
    """

    def __init__(self, row_limit: int = 500_000) -> None:
        self.row_limit = row_limit
        self._conn: duckdb.DuckDBPyConnection = duckdb.connect(":memory:")
        self._registered_views: dict[str, str] = {}  # view_name → uri
        self._install_extensions()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _install_extensions(self) -> None:
        """Install DuckDB extensions needed for remote sources."""
        try:
            # httpfs enables S3 / HTTP reads
            self._conn.execute("INSTALL httpfs; LOAD httpfs;")
        except Exception:
            # Non-fatal — local file reading still works without httpfs
            logger.warning("duckdb_httpfs_unavailable", note="S3/HTTP sources will not work")

    # ── Source Registration ───────────────────────────────────────────────────

    def register_source(self, source: DataSource) -> str:
        """
        Register a DataSource as a DuckDB view.
        Returns the view name to use in subsequent queries.

        For file sources: creates a view over read_csv_auto / read_parquet / etc.
        For Postgres: uses postgres_scan (requires pg extension).
        """
        view_name = source.table_name or f"source_{source.source_id}"
        uri = source.uri

        if source.source_type == DataSourceType.POSTGRES:
            view_name = self._register_postgres(uri, view_name)
        elif source.source_type in (
            DataSourceType.CSV, DataSourceType.PARQUET,
            DataSourceType.JSON, DataSourceType.DUCKDB,
        ):
            view_name = self._register_file(uri, view_name, source.source_type)
        else:
            # Try to infer from extension
            ext = Path(uri).suffix.lower()
            if ext not in _READER_MAP:
                raise UnsupportedFileFormatError(ext)
            view_name = self._register_file(uri, view_name, source.source_type)

        self._registered_views[view_name] = uri
        logger.debug("duckdb_view_registered", view=view_name, uri=uri)
        return view_name

    def _register_file(
        self, uri: str, view_name: str, source_type: DataSourceType
    ) -> str:
        """Register a local or remote file as a DuckDB view."""
        # Check local file existence (skip for S3/HTTP URIs)
        if not uri.startswith(("s3://", "http://", "https://")):
            if not Path(uri).exists():
                raise DataSourceNotFoundError(uri)

        ext = Path(uri).suffix.lower()
        reader_fn = _READER_MAP.get(ext)

        if reader_fn is None:
            # Parquet registered without extension (e.g. S3 key without .parquet)
            if source_type == DataSourceType.PARQUET:
                reader_fn = "read_parquet"
            elif source_type == DataSourceType.CSV:
                reader_fn = "read_csv_auto"
            else:
                raise UnsupportedFileFormatError(ext or "<no extension>")

        sql = f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM {reader_fn}('{uri}')"
        try:
            self._conn.execute(sql)
        except duckdb.IOException as e:
            raise DataSourceNotFoundError(uri) from e
        except Exception as e:
            raise ETLError(f"Failed to register '{uri}' as DuckDB view: {e}") from e

        return view_name

    def _register_postgres(self, uri: str, view_name: str) -> str:
        """Register a Postgres table via DuckDB's postgres_scan."""
        try:
            self._conn.execute("INSTALL postgres; LOAD postgres;")
            # uri format: "postgresql://user:pass@host/db::schema.table"
            if "::" in uri:
                conn_str, table_ref = uri.split("::", 1)
                schema, table = table_ref.split(".") if "." in table_ref else ("public", table_ref)
            else:
                raise ETLError(f"Postgres URI must include '::schema.table': {uri}")

            sql = (
                f"CREATE OR REPLACE VIEW {view_name} AS "
                f"SELECT * FROM postgres_scan('{conn_str}', '{schema}', '{table}')"
            )
            self._conn.execute(sql)
        except ETLError:
            raise
        except Exception as e:
            raise ETLError(f"Postgres registration failed: {e}") from e

        return view_name

    # ── Query Execution ───────────────────────────────────────────────────────

    def execute(self, sql: str) -> pl.DataFrame:
        """
        Execute arbitrary SQL and return a polars DataFrame.
        The row_limit is applied automatically via a wrapper LIMIT clause
        unless the query already contains LIMIT.
        """
        effective_sql = self._apply_row_limit(sql)
        start = time.monotonic()
        try:
            result = self._conn.execute(effective_sql).pl()
        except duckdb.CatalogException as e:
            raise ETLError(f"DuckDB query error (unknown table/column): {e}") from e
        except duckdb.ParserException as e:
            raise ETLError(f"DuckDB SQL parse error: {e}") from e
        except Exception as e:
            raise ETLError(f"DuckDB execution error: {e}") from e

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            "duckdb_query_executed",
            rows=len(result),
            cols=len(result.columns),
            elapsed_ms=round(elapsed_ms, 1),
        )
        return result

    def execute_raw(self, sql: str) -> list[tuple[Any, ...]]:
        """Execute SQL and return raw tuples (for schema inspection queries)."""
        try:
            return self._conn.execute(sql).fetchall()
        except Exception as e:
            raise ETLError(f"DuckDB raw query error: {e}") from e

    def sample(self, view_name: str, n: int = 5) -> pl.DataFrame:
        """Return n sample rows from a registered view."""
        return self.execute(f"SELECT * FROM {view_name} LIMIT {n}")

    def row_count(self, view_name: str) -> int:
        """Fast row count without loading all data."""
        rows = self.execute_raw(f"SELECT COUNT(*) FROM {view_name}")
        return int(rows[0][0]) if rows else 0

    def column_names(self, view_name: str) -> list[str]:
        """Return column names for a registered view."""
        rows = self.execute_raw(
            f"SELECT column_name FROM information_schema.columns "
            f"WHERE table_name = '{view_name}' ORDER BY ordinal_position"
        )
        if rows:
            return [r[0] for r in rows]
        # Fallback: read one row
        sample = self.execute(f"SELECT * FROM {view_name} LIMIT 1")
        return sample.columns

    def list_views(self) -> list[str]:
        """Return all registered view names."""
        return list(self._registered_views.keys())

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _apply_row_limit(self, sql: str) -> str:
        """Wrap query in a subquery with LIMIT if none already present."""
        sql_upper = sql.upper().strip()
        if "LIMIT" in sql_upper:
            return sql
        return f"SELECT * FROM ({sql}) AS _limited LIMIT {self.row_limit}"

    def close(self) -> None:
        """Release the DuckDB connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> "DuckDBTool":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
