#!/usr/bin/env python3
"""
템포 평가: DSP · madmom · 1D-StateSpace 각각 결과 CSV 1개씩 (총 3파일).
GiantSteps + GTZAN-Genre, mirdata 불필요.

출력 (기본 results/):
  tempo_dsp.csv, tempo_madmom.csv, tempo_1dss.csv

환경: conda env + requirements-tempo.txt (1dss는 jump-reward 필요 — 동일 env 권장)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_common import (
    _fmt_exc,
    acc1_hit,
    acc2_hit,
    estimate_1dss_beats_and_tempo,
    estimate_dsp_tempo,
    estimate_madmom_beats,
    mae_bpm,
    pct_error,
    tempo_from_beat_times,
)
from eval_dataset import (
    iter_giantsteps_tasks,
    iter_gtzan_tasks,
    load_excluded_track_ids,
    load_gt_bpm_file,
)

TEMPO_FIELDNAMES = [
    "dataset",
    "track_id",
    "file_or_stem",
    "annotation_path",
    "audio_path",
    "gt_bpm",
    "pred_tempo_bpm",
    "error_pct",
    "mae_bpm",
    "acc1",
    "acc2",
    "status",
    "note",
]


def _base_row(
    dataset: str,
    track_id: str,
    file_or_stem: str,
    bpm_path: Path,
    audio_path: Optional[Path],
    gt_bpm: Optional[float],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "track_id": track_id,
        "file_or_stem": file_or_stem,
        "annotation_path": str(bpm_path),
        "audio_path": str(audio_path) if audio_path else "",
        "gt_bpm": "" if gt_bpm is None else gt_bpm,
        "pred_tempo_bpm": "",
        "error_pct": "",
        "mae_bpm": "",
        "acc1": "",
        "acc2": "",
        "status": "ok",
        "note": "",
    }


def _fill_tempo_metrics(
    row: dict[str, Any], pred: Optional[float], gt: Optional[float]
) -> None:
    if gt is None or gt <= 0:
        row["status"] = "skip_bad_gt"
        return
    row["pred_tempo_bpm"] = "" if pred is None else round(float(pred), 4)
    pe = pct_error(pred, gt)
    row["error_pct"] = "" if pe is None else round(pe, 4)
    mb = mae_bpm(pred, gt)
    row["mae_bpm"] = "" if mb is None else round(mb, 4)
    a1 = acc1_hit(pred, gt)
    a2 = acc2_hit(pred, gt)
    row["acc1"] = "" if a1 is None else a1
    row["acc2"] = "" if a2 is None else a2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tempo eval: DSP, madmom, 1dss — 3 CSV files")
    p.add_argument(
        "--dataset",
        choices=("giantsteps", "gtzan_genre", "both"),
        default="both",
    )
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--giantsteps-root",
        type=Path,
        default=Path("dataset/giantsteps-tempo-dataset-master"),
    )
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
    p.add_argument("--sample-rate-dsp", type=int, default=22050)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="tempo_dsp.csv 등 저장",
    )
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    excluded = load_excluded_track_ids(args.exclude_invalid_json)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    rows_dsp: list[dict] = []
    rows_mm: list[dict] = []
    rows_1d: list[dict] = []

    if args.dataset in ("giantsteps", "both"):
        ann = args.giantsteps_root / "annotations_v2" / "tempo"
        audio_root = args.giantsteps_root / "audio"
        if not ann.is_dir():
            raise FileNotFoundError(f"GiantSteps annotation 없음: {ann}")
        tasks = list(
            iter_giantsteps_tasks(ann, audio_root, excluded, args.sample_size, args.seed)
        )
        it = tqdm(tasks, desc="tempo[giantsteps]", disable=args.no_progress)
        for stem, bpm_path, audio_path in it:
            tid = stem.split(".")[0]
            gt = load_gt_bpm_file(bpm_path)
            rd = _base_row("giantsteps", tid, stem, bpm_path, audio_path, gt)
            rm = rd.copy()
            r1 = rd.copy()

            if gt is None:
                for r in (rd, rm, r1):
                    r["status"] = "skip_bad_gt"
                    r["note"] = "GT BPM 없음"
                rows_dsp.append(rd)
                rows_mm.append(rm)
                rows_1d.append(r1)
                continue
            if audio_path is None or not audio_path.is_file():
                for r in (rd, rm, r1):
                    r["status"] = "missing_audio"
                    r["note"] = "오디오 없음"
                rows_dsp.append(rd)
                rows_mm.append(rm)
                rows_1d.append(r1)
                continue

            try:
                dsp_t = estimate_dsp_tempo(audio_path, args.sample_rate_dsp)
            except Exception as exc:
                dsp_t = None
                rd["note"] = f"dsp:{_fmt_exc(exc)}"
                rd["status"] = "dsp_fail"
            _fill_tempo_metrics(rd, dsp_t, gt)
            if rd["status"] == "ok" and dsp_t is None:
                rd["status"] = "dsp_fail"

            mm_b, mm_st = estimate_madmom_beats(audio_path)
            mm_t = tempo_from_beat_times(mm_b) if mm_st == "ok" else None
            if mm_st != "ok":
                rm["status"] = "madmom_fail"
                rm["note"] = mm_st
            else:
                _fill_tempo_metrics(rm, mm_t, gt)

            ss_b, ss_t, ss_st = estimate_1dss_beats_and_tempo(audio_path)
            if ss_st != "ok":
                r1["status"] = "1dss_fail"
                r1["note"] = ss_st
            else:
                _fill_tempo_metrics(r1, ss_t, gt)

            rows_dsp.append(rd)
            rows_mm.append(rm)
            rows_1d.append(r1)

    if args.dataset in ("gtzan_genre", "both"):
        seed_g = args.seed + 17 if args.dataset == "both" else args.seed
        tasks = list(
            iter_gtzan_tasks(args.data_home, excluded, args.sample_size, seed_g)
        )
        it = tqdm(tasks, desc="tempo[gtzan_genre]", disable=args.no_progress)
        for tid, bpm_path, audio_path, _beats in it:
            gt = load_gt_bpm_file(bpm_path)
            rd = _base_row("gtzan_genre", tid, tid, bpm_path, audio_path, gt)
            rm = rd.copy()
            r1 = rd.copy()

            if gt is None:
                for r in (rd, rm, r1):
                    r["status"] = "skip_bad_gt"
                    r["note"] = "GT BPM 없음"
                rows_dsp.append(rd)
                rows_mm.append(rm)
                rows_1d.append(r1)
                continue
            if audio_path is None or not audio_path.is_file():
                for r in (rd, rm, r1):
                    r["status"] = "missing_audio"
                    r["note"] = "오디오 없음"
                rows_dsp.append(rd)
                rows_mm.append(rm)
                rows_1d.append(r1)
                continue

            try:
                dsp_t = estimate_dsp_tempo(audio_path, args.sample_rate_dsp)
            except Exception as exc:
                dsp_t = None
                rd["note"] = f"dsp:{_fmt_exc(exc)}"
                rd["status"] = "dsp_fail"
            _fill_tempo_metrics(rd, dsp_t, gt)
            if rd["status"] == "ok" and dsp_t is None:
                rd["status"] = "dsp_fail"

            mm_b, mm_st = estimate_madmom_beats(audio_path)
            mm_t = tempo_from_beat_times(mm_b) if mm_st == "ok" else None
            if mm_st != "ok":
                rm["status"] = "madmom_fail"
                rm["note"] = mm_st
            else:
                _fill_tempo_metrics(rm, mm_t, gt)

            ss_b, ss_t, ss_st = estimate_1dss_beats_and_tempo(audio_path)
            if ss_st != "ok":
                r1["status"] = "1dss_fail"
                r1["note"] = ss_st
            else:
                _fill_tempo_metrics(r1, ss_t, gt)

            rows_dsp.append(rd)
            rows_mm.append(rm)
            rows_1d.append(r1)

    for name, rows in (
        ("tempo_dsp.csv", rows_dsp),
        ("tempo_madmom.csv", rows_mm),
        ("tempo_1dss.csv", rows_1d),
    ):
        p = out / name
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=TEMPO_FIELDNAMES, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    print(
        f"[tempo-eval] wrote {out / 'tempo_dsp.csv'}, "
        f"{out / 'tempo_madmom.csv'}, {out / 'tempo_1dss.csv'}"
    )
    print(
        f"[tempo-eval] status_counts dsp={dict(Counter(str(r['status']) for r in rows_dsp))}"
    )


if __name__ == "__main__":
    main()
