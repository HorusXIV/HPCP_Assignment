from pathlib import Path
from typing import Optional, Sequence, Union, Iterable, Tuple, overload
import numpy as np

# Resolve relative to this file, not CWD
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "np32"

def default_files(
    ext: Union[str, Iterable[str]] = "*.npz",
    directory: Optional[Union[str, Path]] = None,
) -> list[Path]:
    """Return sorted list of files under directory (default: DEFAULT_DATA_DIR)."""
    base = Path(directory) if directory is not None else DEFAULT_DATA_DIR
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {base}")
    patterns = (ext,) if isinstance(ext, str) else tuple(ext)
    files: list[Path] = []
    for pat in patterns:
        files.extend(base.glob(pat))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No files matching {patterns} in {base}")
    return files

IndexLike = Union[int, slice, Sequence[int], None]

def _ensure_6hw(a: np.ndarray, *, src: Union[str, Path]) -> np.ndarray:
    if a.ndim != 3 or a.shape[0] != 6:
        raise ValueError(f"{src} expected (6,H,W), got {a.shape}")
    return a

def _cast(a: np.ndarray, *, dtype: Optional[np.dtype], contiguous: bool) -> np.ndarray:
    if dtype is not None and a.dtype != dtype:
        a = a.astype(dtype, copy=False)
    if contiguous:
        a = np.ascontiguousarray(a)
    return a

def _select_indices(n: int, idx: IndexLike) -> list[int]:
    if idx is None or idx == -1:  # keep your sentinel working
        return list(range(n))
    if isinstance(idx, int):
        if idx < 0:
            idx += n
        if not (0 <= idx < n):
            raise IndexError(f"Index {idx} out of range for n={n}")
        return [idx]
    if isinstance(idx, slice):
        return list(range(n))[idx]
    # sequence of ints
    out = []
    for i in idx:
        j = i + n if i < 0 else i
        if not (0 <= j < n):
            raise IndexError(f"Index {i} out of range for n={n}")
        out.append(j)
    return out

def load_np_stack(
    file_list: Sequence[Union[str, Path]],
    idx: IndexLike = -1,
    *,
    channels_last: bool = False,   # True -> return (..., H, W, 6)
    dtype: Optional[np.dtype] = None,
    contiguous: bool = True,
    return_paths: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, list[Path]]]:
    """
    Load one or multiple .npz stacks with 'bands' -> (6, H, W).

    Returns
    -------
    arr : np.ndarray
        Shape (N, 6, H, W)  or (N, H, W, 6) if channels_last=True
        where N = number of selected files.
    paths : list[Path]  (only if return_paths=True)
        Paths in the same order as stacked.
    """
    files = sorted(Path(f) for f in file_list)
    if not files:
        raise FileNotFoundError("No .npz files provided.")

    select = _select_indices(len(files), idx)
    arrays = []
    ref_shape = None

    for j in select:
        p = files[j]
        with np.load(p, allow_pickle=False) as z:
            if "bands" not in z:
                raise KeyError(f"{p.name} missing 'bands' array")
            a = _ensure_6hw(z["bands"], src=p)  # (6,H,W)
        if ref_shape is None:
            ref_shape = a.shape
        elif a.shape != ref_shape:
            raise ValueError(f"Shape mismatch: {p.name} {a.shape} != {ref_shape}")
        arrays.append(a)

    arr = np.stack(arrays, axis=0)            # (N,6,H,W)
    if channels_last:
        arr = np.moveaxis(arr, 1, -1)         # (N,H,W,6)
    arr = _cast(arr, dtype=dtype, contiguous=contiguous)

    return (arr, [files[j] for j in select]) if return_paths else arr

def frame_for_solver(stack_any: np.ndarray, i: int = 0) -> np.ndarray:
    """Return (H, W, 6) from either (N,6,H,W) or (N,H,W,6)."""
    f = stack_any[i]
    if f.ndim != 3:
        raise ValueError(f"Expected 3D frame, got {f.shape}")
    if f.shape[0] == 6:           # (6,H,W) -> (H,W,6)
        f = np.moveaxis(f, 0, -1)
    elif f.shape[-1] != 6:
        raise ValueError(f"Cannot infer channels axis for shape {f.shape}")
    return np.ascontiguousarray(f)
