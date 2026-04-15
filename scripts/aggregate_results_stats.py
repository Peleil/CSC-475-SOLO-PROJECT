#!/usr/bin/env python3
"""
Aggregate GTZAN-by-genre statistics from CSV files under results/.

- Tempo: existing files among tempo_{dsp,madmom,autocorr,1dss}.csv
  (dataset==gtzan_genre and status==ok only)
- Beat: beat_{madmom,1dss}.csv
  (status==ok, has_ref_beats=true, numeric beat_fmeasure)

Genre parsing example: track_id `blues.00042` -> blues
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


def _genre_from_track_id(tid: str) -> str:
    if "." not in tid:
        return "unknown"
    return tid.split(".", 1)[0]


def _parse_bool(s: str) -> Optional[bool]:
    t = str(s).strip().lower()
    if t == "true":
        return True
    if t == "false":
        return False
    return None


def _parse_float(s: str) -> Optional[float]:
    s = str(s).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def aggregate_tempo_csv(path: Path) -> dict[str, dict[str, Any]]:
    by_genre: dict[str, dict[str, list]] = defaultdict(
        lambda: {"error_pct": [], "mae_bpm": [], "acc1": [], "acc2": []}
    )
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("dataset") != "gtzan_genre":
                continue
            if row.get("status") != "ok":
                continue
            g = _genre_from_track_id(str(row.get("track_id", "")))
            pe = _parse_float(row.get("error_pct", ""))
            mb = _parse_float(row.get("mae_bpm", ""))
            a1 = _parse_bool(row.get("acc1", ""))
            a2 = _parse_bool(row.get("acc2", ""))
            if pe is not None:
                by_genre[g]["error_pct"].append(pe)
            if mb is not None:
                by_genre[g]["mae_bpm"].append(mb)
            if a1 is not None:
                by_genre[g]["acc1"].append(1.0 if a1 else 0.0)
            if a2 is not None:
                by_genre[g]["acc2"].append(1.0 if a2 else 0.0)

    out: dict[str, dict[str, Any]] = {}
    for g, buckets in sorted(by_genre.items()):
        n = len(buckets["error_pct"])
        out[g] = {
            "n": n,
            "mean_error_pct": round(sum(buckets["error_pct"]) / n, 6) if n else None,
            "mean_mae_bpm": round(sum(buckets["mae_bpm"]) / len(buckets["mae_bpm"]), 6)
            if buckets["mae_bpm"]
            else None,
            "acc1_rate": round(sum(buckets["acc1"]) / len(buckets["acc1"]), 6)
            if buckets["acc1"]
            else None,
            "acc2_rate": round(sum(buckets["acc2"]) / len(buckets["acc2"]), 6)
            if buckets["acc2"]
            else None,
        }
    return out


def aggregate_beat_csv(path: Path) -> dict[str, dict[str, Any]]:
    by_genre: dict[str, list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("status") != "ok":
                continue
            if str(row.get("has_ref_beats", "")).lower() != "true":
                continue
            bf = _parse_float(row.get("beat_fmeasure", ""))
            if bf is None:
                continue
            g = _genre_from_track_id(str(row.get("track_id", "")))
            by_genre[g].append(bf)

    out: dict[str, dict[str, Any]] = {}
    for g, fs in sorted(by_genre.items()):
        n = len(fs)
        if not n:
            continue
        s = sorted(fs)
        mid = (s[n // 2] + s[(n - 1) // 2]) / 2.0
        out[g] = {
            "n": n,
            "mean_beat_fmeasure": round(sum(fs) / n, 6),
            "median_beat_fmeasure": round(mid, 6),
        }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate GTZAN-by-genre stats from results CSVs")
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing tempo_*.csv and beat_*.csv",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/aggregated_gtzan_by_genre.json"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rd = args.results_dir.resolve()
    tempo_methods = ("dsp", "madmom", "autocorr", "1dss")
    beat_methods = ("madmom", "1dss")

    payload: dict[str, Any] = {
        "source_dir": str(rd),
        "gtzan_by_genre": {"tempo": {}, "beat": {}},
    }

    for m in tempo_methods:
        p = rd / f"tempo_{m}.csv"
        if p.is_file():
            payload["gtzan_by_genre"]["tempo"][m] = aggregate_tempo_csv(p)
        else:
            payload["gtzan_by_genre"]["tempo"][m] = {}

    for m in beat_methods:
        p = rd / f"beat_{m}.csv"
        if p.is_file():
            payload["gtzan_by_genre"]["beat"][m] = aggregate_beat_csv(p)
        else:
            payload["gtzan_by_genre"]["beat"][m] = {}

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[aggregate-results] wrote {args.output_json}")


if __name__ == "__main__":
    main()
