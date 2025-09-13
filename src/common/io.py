from pathlib import Path
from typing import Sequence, Union
import numpy as np

PROJECT_ROOT = Path("../..").resolve()
NP_DIR = PROJECT_ROOT / "data" / "np32"
FILE_LIST = list(NP_DIR.glob("*.npz"))

from pathlib import Path
from typing import Sequence, Union, Optional
import numpy as np

def load_np_stack(
    file_list: Sequence[Union[str, Path]],
    idx: int = -1,
    *,
    channels_last: bool = False,   # True -> return (N, H, W, 6)
    dtype: Optional[np.dtype] = None,
    contiguous: bool = True
) -> np.ndarray:
    """
    Load one or all .npz stacks with 'bands' -> (6, H, W).

    Returns
    -------
    arr : np.ndarray
        If idx == -1:
            (N, 6, H, W)  or (N, H, W, 6) if channels_last=True
        If idx in [0..len(files)-1]:
            (1, 6, H, W)  or (1, H, W, 6) if channels_last=True
    """
    files = sorted(Path(f) for f in file_list)
    if not files:
        raise FileNotFoundError("No .npz files provided.")
    n_files = len(files)

    def _cast(x: np.ndarray) -> np.ndarray:
        if dtype is not None and x.dtype != dtype:
            x = x.astype(dtype, copy=False)
        if contiguous:
            x = np.ascontiguousarray(x)
        return x

    if idx == -1:
        arrays = []
        shape0 = None
        for p in files:
            with np.load(p, allow_pickle=False) as z:
                if "bands" not in z:
                    raise KeyError(f"{p.name} missing 'bands' array")
                a = z["bands"]               # (6, H, W)
            if shape0 is None:
                shape0 = a.shape
            elif a.shape != shape0:
                raise ValueError(f"Shape mismatch: {p.name} {a.shape} != {shape0}")
            arrays.append(a)

        arr = np.stack(arrays, axis=0)       # (N, 6, H, W)
        if channels_last:
            arr = np.moveaxis(arr, 1, -1)    # (N, H, W, 6)
        return _cast(arr)

    if 0 <= idx < n_files:
        with np.load(files[idx], allow_pickle=False) as z:
            if "bands" not in z:
                raise KeyError(f"{files[idx].name} missing 'bands' array")
            a = z["bands"]                   # (6, H, W)
        a = a[np.newaxis, ...]               # (1, 6, H, W)
        if channels_last:
            a = np.moveaxis(a, 1, -1)        # (1, H, W, 6)
        return _cast(a)

    raise IndexError(f"Index {idx} out of range for file_list of length {n_files}")


def frame_for_solver(stack_any: np.ndarray, i: int = 0) -> np.ndarray:
    """Return (H, W, 6) view/copy from either (N,6,H,W) or (N,H,W,6)."""
    f = stack_any[i]
    if f.ndim != 3:
        raise ValueError(f"Expected 3D frame, got {f.shape}")
    if f.shape[0] == 6:           # (6, H, W) -> (H, W, 6)
        f = np.moveaxis(f, 0, -1)
    elif f.shape[-1] != 6:
        raise ValueError(f"Cannot infer channels axis for shape {f.shape}")
    return np.ascontiguousarray(f)
