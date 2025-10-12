# tests/test_dask_main_callables.py
"""Tests for Dask main module callables."""

from __future__ import annotations

import pytest
from src.dask.main import parse_size, parse_idx


def test_parse_size_square():
    """Test parsing square size specification."""
    h, w = parse_size("256")
    assert h == 256 and w == 256, f"Expected (256, 256), got ({h}, {w})"


def test_parse_size_rectangular():
    """Test parsing rectangular size specification."""
    h, w = parse_size("128x256")
    assert h == 128 and w == 256, f"Expected (128, 256), got ({h}, {w})"


def test_parse_size_with_spaces():
    """Test parsing with whitespace."""
    h, w = parse_size(" 512 x 1024 ")
    assert h == 512 and w == 1024, f"Expected (512, 1024), got ({h}, {w})"


def test_parse_size_invalid():
    """Test that invalid input raises ValueError."""
    with pytest.raises(ValueError):
        parse_size("invalid")


def test_parse_idx_all():
    """Test parsing 'all' returns None."""
    assert parse_idx("all") is None
    assert parse_idx("-1") is None


def test_parse_idx_integer():
    """Test parsing integer index."""
    assert parse_idx("0") == 0
    assert parse_idx("5") == 5


def test_parse_idx_invalid():
    """Test that invalid input raises ValueError."""
    with pytest.raises(ValueError):
        parse_idx("invalid")