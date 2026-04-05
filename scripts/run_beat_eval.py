#!/usr/bin/env python3
"""
비트 평가: madmom · 1D-StateSpace — CSV 2개만 (beat_madmom.csv, beat_1dss.csv).

GTZAN-Genre만 대상 (참 비트: gtzan_tempo_beat-main/beats/*.beats, 첫 열 = 초).
GiantSteps 공개 템포 세트에는 비트 GT가 없어 비트 평가에서 제외한다.

출력 (기본 results/):
  beat_madmom.csv, beat_1dss.csv

환경: requirements-beat.txt
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_common import beat_fmeasure, estimate_1dss_beats_and_tempo, estimate_madmom_beats
from eval_dataset import iter_gtzan_tasks, load_excluded_track_ids, load_gtzan_beat_times

BEAT_FIELDNAMES = [
    "dataset",
    "track_id",
    "file_or_stem",
    "audio_path",
    "ref_beats_path",
    "has_ref_beats",
    "num_ref_beats",
    "num_est_beats",
    "beat_fmeasure",
    "status",
    "note",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Beat eval (GTZAN only): madmom, 1dss — 2 CSV files"
    )
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--data-home",
        type=Path,
        default=Path("dataset/mirdata_gtzan_genre"),
    )
    p.add_argument(
        "--exclude-invalid-json",
        type=Path,
        default=Path("results/invalid_tracks_gtzan_genre.json"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
    )
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def _row_base(
    track_id: str,
    audio_path: Optional[Path],
    ref_path: Optional[Path],
    ref_beats: Optional[np.ndarray],
) -> dict[str, Any]:
    has_ref = ref_beats is not None and ref_beats.size > 0
    return {
        "dataset": "gtzan_genre",
        "track_id": track_id,
        "file_or_stem": track_id,
        "audio_path": str(audio_path) if audio_path else "",
        "ref_beats_path": str(ref_path) if ref_path else "",
        "has_ref_beats": has_ref,
        "num_ref_beats": int(ref_beats.size) if has_ref else 0,
        "num_est_beats": "",
        "beat_fmeasure": "",
        "status": "ok",
        "note": "",
    }


def _fill_est(
    row: dict[str, Any],
    est: np.ndarray,
    ref: Optional[np.ndarray],
) -> None:
    row["num_est_beats"] = int(est.size)
    if ref is None or ref.size == 0:
        row["note"] = "no_reference_beats"
        return
    fd = beat_fmeasure(ref, est)
    row["beat_fmeasure"] = "" if fd is None else round(fd, 6)


def main() -> None:
    args = parse_args()
    excluded = load_excluded_track_ids(args.exclude_invalid_json)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    rows_mm: list[dict] = []
    rows_1d: list[dict] = []

    tasks = list(
        iter_gtzan_tasks(args.data_home, excluded, args.sample_size, args.seed)
    )
    it = tqdm(tasks, desc="beat[gtzan_genre]", disable=args.no_progress)
    for tid, _bpm, audio_path, beats_path in it:
        ref = load_gtzan_beat_times(beats_path) if beats_path else None
        rm = _row_base(tid, audio_path, beats_path, ref)
        r1 = rm.copy()

        if audio_path is None or not audio_path.is_file():
            rm["status"] = r1["status"] = "missing_audio"
            rm["note"] = r1["note"] = "오디오 없음"
            rows_mm.append(rm)
            rows_1d.append(r1)
            continue

        mm_b, mm_st = estimate_madmom_beats(audio_path)
        if mm_st != "ok":
            rm["status"] = "madmom_fail"
            rm["note"] = mm_st
        else:
            _fill_est(rm, mm_b, ref)

        ss_b, _t, ss_st = estimate_1dss_beats_and_tempo(audio_path)
        if ss_st != "ok":
            r1["status"] = "1dss_fail"
            r1["note"] = ss_st
        else:
            _fill_est(r1, ss_b, ref)

        rows_mm.append(rm)
        rows_1d.append(r1)

    for name, rows in (
        ("beat_madmom.csv", rows_mm),
        ("beat_1dss.csv", rows_1d),
    ):
        p = out / name
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=BEAT_FIELDNAMES,
                extrasaction="ignore",
            )
            w.writeheader()
            for r in rows:
                line = dict(r)
                line["has_ref_beats"] = (
                    "true" if line.get("has_ref_beats") else "false"
                )
                w.writerow(line)

    print(f"[beat-eval] wrote {out / 'beat_madmom.csv'}, {out / 'beat_1dss.csv'}")


if __name__ == "__main__":
    main()
