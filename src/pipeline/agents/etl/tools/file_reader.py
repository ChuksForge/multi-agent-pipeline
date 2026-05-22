"""
agents/etl/tools/file_reader.py
────────────────────────────────
Reads CSV, Parquet, and JSON files into polars DataFrames.

Used as a pre-step before DuckDB registration when:
  - The file needs preprocessing (encoding fix, header detection)
  - We want polars-native type inference instead of DuckDB's
  - The file is small enough to load fully in memory for validation

For large files, the DuckDBTool is preferred (lazy evaluation, no full load).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from pipeline.core.exceptions import (
    DataSourceNotFoundError,
    ETLError,
    UnsupportedFileFormatError,
)
from pipeline.middleware.logger import get_logger

logger = get_logger(__name__)

# Bytes threshold above which we warn about in-memory load
_LARGE_FILE_BYTES = 100 * 1024 * 1024  # 100 MB


def read_file(
    path: str,
    *,
    row_limit: int | None = None,
    infer_schema_length: int = 1000,
    null_values: list[str] | None = None,
    encoding: str = "utf-8",
) -> pl.DataFrame:
    """
    Read a local file into a polars DataFrame.
    Dispatches to the correct reader based on file extension.

    Args:
        path: Absolute or relative file path.
        row_limit: Cap number of rows read (None = all).
        infer_schema_length: Rows used for schema inference (CSV).
        null_values: Extra strings to treat as null (e.g. ["N/A", "na", "-"]).
        encoding: File encoding for CSV (default UTF-8).

    Returns:
        polars DataFrame with inferred schema.

    Raises:
        DataSourceNotFoundError: File does not exist.
        UnsupportedFileFormatError: Extension not supported.
        ETLError: Parse or read failure.
    """
    p = Path(path)
    if not p.exists():
        raise DataSourceNotFoundError(path)

    size = p.stat().st_size
    if size > _LARGE_FILE_BYTES:
        logger.warning(
            "large_file_detected",
            path=path,
            size_mb=round(size / 1024 / 1024, 1),
            note="Consider using DuckDBTool for lazy evaluation",
        )

    ext = p.suffix.lower()
    null_vals = null_values or ["", "NULL", "null", "N/A", "n/a", "NA", "na", "None", "none", "-"]

    try:
        if ext == ".csv" or ext == ".tsv":
            sep = "\t" if ext == ".tsv" else ","
            df = _read_csv(p, sep, row_limit, infer_schema_length, null_vals, encoding)
        elif ext == ".parquet":
            df = _read_parquet(p, row_limit)
        elif ext in (".json", ".jsonl", ".ndjson"):
            df = _read_json(p, row_limit)
        else:
            raise UnsupportedFileFormatError(ext)
    except (DataSourceNotFoundError, UnsupportedFileFormatError, ETLError):
        raise
    except Exception as e:
        raise ETLError(f"Failed to read '{path}': {e}") from e

    logger.debug(
        "file_read",
        path=path,
        rows=len(df),
        cols=len(df.columns),
        ext=ext,
    )
    return df


def _read_csv(
    path: Path,
    sep: str,
    row_limit: int | None,
    infer_schema_length: int,
    null_values: list[str],
    encoding: str,
) -> pl.DataFrame:
    """Read CSV with polars — handles messy headers, mixed types, BOM."""
    kwargs: dict[str, Any] = {
        "separator": sep,
        "infer_schema_length": infer_schema_length,
        "null_values": null_values,
        "encoding": encoding,
        "ignore_errors": True,          # Don't crash on bad rows
        "truncate_ragged_lines": True,  # Handle uneven column counts
        "try_parse_dates": True,        # Auto-detect date columns
    }
    if row_limit is not None:
        kwargs["n_rows"] = row_limit

    try:
        return pl.read_csv(path, **kwargs)
    except pl.exceptions.NoDataError:
        raise ETLError(f"CSV file is empty: {path}")
    except Exception as e:
        # Second attempt: let polars guess the separator
        try:
            kwargs.pop("separator", None)
            return pl.read_csv(path, **kwargs)
        except Exception:
            raise ETLError(f"CSV parse failure on '{path}': {e}") from e


def _read_parquet(path: Path, row_limit: int | None) -> pl.DataFrame:
    """Read Parquet — polars handles row group filtering natively."""
    try:
        df = pl.read_parquet(path, n_rows=row_limit)
        return df
    except Exception as e:
        raise ETLError(f"Parquet read failure on '{path}': {e}") from e


def _read_json(path: Path, row_limit: int | None) -> pl.DataFrame:
    """
    Read JSON / JSONL into a flat DataFrame.
    Handles both newline-delimited JSON (one object per line)
    and standard JSON arrays.
    """
    try:
        # Try JSONL first (more common in data pipelines)
        df = pl.read_ndjson(path)
    except Exception:
        try:
            # Fall back to standard JSON array
            df = pl.read_json(path)
        except Exception as e:
            # Last resort: parse manually and create DataFrame
            try:
                raw = json.loads(path.read_text())
                if isinstance(raw, list):
                    df = pl.DataFrame(raw)
                elif isinstance(raw, dict):
                    df = pl.DataFrame([raw])
                else:
                    raise ETLError(f"JSON root must be array or object: {path}")
            except json.JSONDecodeError as je:
                raise ETLError(f"JSON parse failure on '{path}': {je}") from je

    if row_limit is not None:
        df = df.head(row_limit)

    return df


# ── Utilities ─────────────────────────────────────────────────────────────────

def detect_encoding(path: str) -> str:
    """
    Heuristic encoding detection.
    Reads first 4KB and checks for BOM markers.
    Falls back to UTF-8 if inconclusive.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read(4096)
        if raw.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        if raw.startswith(b"\xff\xfe"):
            return "utf-16-le"
        if raw.startswith(b"\xfe\xff"):
            return "utf-16-be"
        # Try decoding as UTF-8
        raw.decode("utf-8")
        return "utf-8"
    except (UnicodeDecodeError, OSError):
        return "latin-1"  # Safe fallback for Western European files


def supported_extensions() -> list[str]:
    """Return list of extensions this reader handles."""
    return [".csv", ".tsv", ".parquet", ".json", ".jsonl", ".ndjson"]
