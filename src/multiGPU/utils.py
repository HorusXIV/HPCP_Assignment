"""Utility helpers for multiGPU package."""
import os


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def getenv_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default
