"""HPCP Assignment package."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hpcp-assignment")  # your distribution name in pyproject
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__: list[str] = []
