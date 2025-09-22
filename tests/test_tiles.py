"""Unit tests for `src.dask.tiles` helpers.

Covers:
- `parse_hw` accepting None/int/str/sequence forms.
- `parse_tile` accepting None/int/str forms.
- `gen_tiles` producing full coverage of an HxW domain with edge-aligned tiles.
"""

from __future__ import annotations

from src.dask.tiles import parse_hw, parse_tile, gen_tiles


def test_parse_hw_variants() -> None:
    # Defaults and scalar/string/sequence variants
    assert parse_hw(None) == (2048, 2048)
    assert parse_hw(512) == (512, 512)
    assert parse_hw("256") == (256, 256)
    assert parse_hw("128x64") == (128, 64)
    assert parse_hw(["32", "48"]) == (32, 48)


def test_parse_tile_variants() -> None:
    # Defaults and scalar/string variants
    assert parse_tile(None) == (256, 256)
    assert parse_tile(64) == (64, 64)
    assert parse_tile("32") == (32, 32)
    assert parse_tile("96x32") == (96, 32)


def test_gen_tiles_covers_domain_and_edges() -> None:
    H, W, Th, Tw = 100, 70, 32, 16
    tiles = gen_tiles(H, W, Th, Tw)

    # Coverage and bounds
    for (y0, y1, x0, x1) in tiles:
        assert 0 <= y0 < y1 <= H
        assert 0 <= x0 < x1 <= W
        assert (y1 - y0) <= Th
        assert (x1 - x0) <= Tw

    # Edges: last tiles must reach the full domain extents
    assert max(y1 for (_, y1, _, _) in tiles) == H
    assert max(x1 for (_, _, _, x1) in tiles) == W
