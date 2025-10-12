# tests/test_tiles.py
"""Tests for Dask tiling (suite.py)."""

from __future__ import annotations

import numpy as np
import dask.array as da
import pytest

from src.dask.suite import build_graph


def test_build_graph_shape():
    """
    Test that build_graph returns correctly shaped lazy arrays.
    Does NOT compute - just checks graph structure.
    """
    # Create a small fake frame: (H=128, W=128, C=6), chunked (64, 64, 6)
    frame_np = np.random.rand(128, 128, 6).astype(np.float32)
    frame_da = da.from_array(frame_np, chunks=(64, 64, 6))
    
    # Build graph with small parameters
    nt = 10
    dem_lazy, edem_lazy, chisq_lazy = build_graph(
        frame_da,
        nmu=5,
        nt=nt,
    )
    
    # Check shapes (lazy - no computation)
    assert dem_lazy.shape == (128, 128, nt), f"DEM shape mismatch: {dem_lazy.shape}"
    assert edem_lazy.shape == (128, 128, nt), f"eDEM shape mismatch: {edem_lazy.shape}"
    assert chisq_lazy.shape == (128, 128), f"chisq shape mismatch: {chisq_lazy.shape}"
    
    # Check chunking
    assert dem_lazy.chunks[0] == (64, 64), f"DEM H chunks: {dem_lazy.chunks[0]}"
    assert dem_lazy.chunks[1] == (64, 64), f"DEM W chunks: {dem_lazy.chunks[1]}"
    assert dem_lazy.chunks[2] == (nt,), f"DEM T chunks: {dem_lazy.chunks[2]}"


def test_build_graph_wrong_input_shape():
    """Test that build_graph raises on invalid input shape."""
    # Wrong: missing channel dimension
    bad_frame = da.from_array(np.random.rand(64, 64), chunks=(32, 32))
    
    with pytest.raises(ValueError, match="Expected .* frame"):
        build_graph(bad_frame, nmu=5, nt=10)


def test_build_graph_wrong_channel_count():
    """Test that build_graph raises on wrong channel count."""
    # Wrong: 3 channels instead of 6
    bad_frame = da.from_array(
        np.random.rand(64, 64, 3).astype(np.float32),
        chunks=(32, 32, 3)
    )
    
    with pytest.raises(ValueError, match="Expected .* frame"):
        build_graph(bad_frame, nmu=5, nt=10)