"""Paths and secrets. Nothing else in the package hardcodes a path or reads the environment."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()  # a .env in the current directory wins if present


def data_dir() -> Path:
    """Root of the data tree. Override with TERRASTAT_DATA_DIR."""
    return Path(os.getenv("TERRASTAT_DATA_DIR", PROJECT_ROOT / "data")).resolve()


def fred_api_key() -> str:
    key = os.getenv("FRED_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Put it in terrastat/.env (see .env.example); "
            "get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return key


USER_AGENT = "terrastat/0.1 (academic research data gathering)"
