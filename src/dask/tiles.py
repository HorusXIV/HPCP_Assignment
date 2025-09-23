# src/dask/tiles.py
from __future__ import annotations
from typing import Tuple, List, Sequence, Union


def parse_hw(
    sizes: Union[int, str, Sequence[int], None],
    default: Tuple[int, int] = (2048, 2048),
) -> Tuple[int, int]:
    """
    Normalize a crop-size specification into a (H, W) pair.

    Accepted forms
    --------------
    - None                → default
    - int                 → (v, v)
    - str  "HxW" or "H,W" → (H, W)
    - str  "H"            → (H, H)
    - sequence[int]       → (H, W) or (H,) → (H, H)

    Parameters
    ----------
    sizes : int | str | Sequence[int] | None
        Size specifier to parse.
    default : (int, int)
        Fallback if `sizes` is None or unparsable.

    Returns
    -------
    (H, W) : tuple[int, int]
    """
    if sizes is None:
        return default

    if isinstance(sizes, int):
        return sizes, sizes

    if isinstance(sizes, str):
        s = sizes.lower().replace(" ", "").replace("x", ",")
        parts = [p for p in s.split(",") if p]
        if not parts:
            return default
        if len(parts) == 1:
            v = int(parts[0])
            return v, v
        return int(parts[0]), int(parts[1])

    try:
        seq = list(sizes)  # type: ignore[arg-type]
        if not seq:
            return default
        if len(seq) == 1:
            v = int(seq[0])
            return v, v
        return int(seq[0]), int(seq[1])
    except Exception:
        return default


def parse_tile(
    tile: Union[int, str, Sequence[int], None],
    default: Tuple[int, int] = (256, 256),
) -> Tuple[int, int]:
    """
    Normalize a tile-size specification into a (Th, Tw) pair.

    Accepted forms mirror `parse_hw`.

    Parameters
    ----------
    tile : int | str | Sequence[int] | None
        Tile specifier to parse.
    default : (int, int)
        Fallback if `tile` is None or unparsable.

    Returns
    -------
    (Th, Tw) : tuple[int, int]
    """
    if tile is None:
        return default

    if isinstance(tile, int):
        return tile, tile

    if isinstance(tile, str):
        s = tile.lower().replace(" ", "").replace("x", ",")
        parts = [p for p in s.split(",") if p]
        if not parts:
            return default
        if len(parts) == 1:
            v = int(parts[0])
            return v, v
        return int(parts[0]), int(parts[1])

    try:
        seq = list(tile)  # type: ignore[arg-type]
        if not seq:
            return default
        if len(seq) == 1:
            v = int(seq[0])
            return v, v
        return int(seq[0]), int(seq[1])
    except Exception:
        return default


def gen_tiles(H: int, W: int, Th: int, Tw: int) -> List[tuple[int, int, int, int]]:
    """
    Enumerate row-major tiles that cover the rectangle [0:H) × [0:W).

    The final tiles along the bottom/right edges may be smaller than (Th, Tw)
    if H or W is not an exact multiple of the tile size.

    Parameters
    ----------
    H, W : int
        Full frame height and width.
    Th, Tw : int
        Tile (block) height and width.

    Returns
    -------
    list[tuple[int, int, int, int]]
        A list of slices (y0, y1, x0, x1), inclusive-exclusive, suitable for
        NumPy/Dask indexing: `arr[y0:y1, x0:x1, ...]`.
    """
    tiles: List[tuple[int, int, int, int]] = []
    for y0 in range(0, H, Th):
        y1 = min(y0 + Th, H)
        for x0 in range(0, W, Tw):
            x1 = min(x0 + Tw, W)
            tiles.append((y0, y1, x0, x1))
    return tiles
