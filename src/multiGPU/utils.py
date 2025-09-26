"""Small utility helpers used across multi-GPU modules."""

import os


def safe_makedirs(path: str):
    """Create directories if they do not already exist."""
    os.makedirs(path, exist_ok=True)


def getenv_int(name: str, default: int) -> int:
    """Fetch an environment variable as int with a safe fallback."""
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default
