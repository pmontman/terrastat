"""Paths and secrets. Nothing else in the package hardcodes a path or reads the environment."""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()  # a .env in the current directory wins if present
_DATA_ROOT: ContextVar[Path | None] = ContextVar("terrastat_data_root", default=None)
_CACHE_ROOT: ContextVar[Path | None] = ContextVar("terrastat_cache_root", default=None)


def data_dir() -> Path:
    """Root of the data tree. Override with TERRASTAT_DATA_DIR."""
    return _DATA_ROOT.get() or Path(os.getenv("TERRASTAT_DATA_DIR", PROJECT_ROOT / "data")).resolve()


def reference_cache_dir() -> Path:
    return _CACHE_ROOT.get() or data_dir() / "cache"


@contextmanager
def data_workspace(root: Path, *, cache: Path | None = None):
    """Isolate adapter writes without changing process-wide environment variables."""
    data_token = _DATA_ROOT.set(Path(root).resolve())
    cache_token = _CACHE_ROOT.set(Path(cache).resolve() if cache is not None else None)
    try:
        yield
    finally:
        _CACHE_ROOT.reset(cache_token)
        _DATA_ROOT.reset(data_token)


def fred_api_key() -> str:
    key = os.getenv("FRED_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Put it in terrastat/.env (see .env.example); "
            "get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return key


USER_AGENT = "terrastat/0.1 (academic research data gathering)"
