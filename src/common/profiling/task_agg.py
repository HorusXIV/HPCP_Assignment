# src/common/profiling/task_agg.py
from __future__ import annotations
"""
Aggregation and CSV utilities for Dask task-stream events.

This module provides:
  • `aggregate_task_stream`: reduce raw task events into simple per-key stats
  • `write_task_csv`: persist raw task events to CSV
  • `write_agg_csv`: persist aggregated stats to CSV

Event Schema (best-effort)
--------------------------
Dask task-stream events can vary across versions and contexts. This code tries
to be tolerant and looks for durations in this order:

1) `event["duration"]`
2) `event["stop"] - event["start"]`
3) If a non-standard `startstops` dict is present, it will attempt to compute
   `stop["t"] - start["t"]`.

Task "key" is taken from `event["key"]` if present, otherwise `event["task"]`,
and finally the literal `"unknown"` as a last resort.
"""

from collections import defaultdict
from typing import Any, Dict, Iterable, List
from pathlib import Path
import csv


def aggregate_task_stream(
    events: Iterable[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate Dask task-stream events into simple per-key statistics.

    Parameters
    ----------
    events : Iterable[dict]
        Raw task-stream events (as produced by `get_task_stream` or similar).

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping: key → {
            "count": number of events,
            "sum":   total duration,
            "mean":  average duration,
            "p50":   median duration (approx by rank),
            "p95":   95th percentile (approx by rank),
            "max":   maximum duration
        }

    Notes
    -----
    - Durations are interpreted in seconds (typical for Dask events), but no
      unit conversion is applied here.
    - Percentiles use a simple nearest-rank approach over the sorted values.
    """
    buckets: Dict[str, List[float]] = defaultdict(list)
    for ev in events:
        key = str(ev.get("key", ev.get("task", "unknown")))

        # Prefer 'duration' if present; otherwise compute from timestamps.
        dur = ev.get("duration")
        if dur is None:
            # Fallback: generic stop-start
            try:
                dur = float(ev.get("stop", 0)) - float(ev.get("start", 0))
            except Exception:
                dur = None

        # Rare/non-standard shape: startstops = {"start": {...}, "stop": {...}}
        if dur is None:
            startstops = ev.get("startstops")
            if isinstance(startstops, dict):
                try:
                    start = startstops.get("start", {})
                    stop = startstops.get("stop", {})
                    dur = float(stop.get("t", 0)) - float(start.get("t", 0))
                except Exception:
                    dur = None

        try:
            if dur is not None:
                buckets[key].append(float(dur))
        except Exception:
            # Ignore events with non-numeric durations
            continue

    out: Dict[str, Dict[str, float]] = {}
    for k, vals in buckets.items():
        if not vals:
            continue
        vs = sorted(vals)
        n = len(vs)

        def pct(p: float) -> float:
            # nearest-rank index in [0, n-1]
            i = min(n - 1, max(0, int(round(p * (n - 1)))))
            return float(vs[i])

        total = float(sum(vs))
        out[k] = {
            "count": float(n),
            "sum": total,
            "mean": total / n,
            "p50": pct(0.5),
            "p95": pct(0.95),
            "max": float(vs[-1]),
        }
    return out


def write_task_csv(events: Iterable[Dict[str, Any]], path: Path) -> None:
    """
    Write raw task-stream events to CSV.

    Parameters
    ----------
    events : Iterable[dict]
        Raw events. The union of all keys across rows becomes the CSV header.
    path : pathlib.Path
        Destination CSV path. Parent directories are created as needed.
    """
    rows = list(events)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_agg_csv(agg: Dict[str, Dict[str, float]], path: Path) -> None:
    """
    Write aggregated per-key statistics to CSV.

    Parameters
    ----------
    agg : dict[str, dict[str, float]]
        Output of `aggregate_task_stream`.
    path : pathlib.Path
        Destination CSV path. Parent directories are created as needed.
    """
    if not agg:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["key", "count", "sum", "mean", "p50", "p95", "max"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for k, m in sorted(agg.items()):
            row = {"key": k, **m}
            w.writerow(row)
