#!/usr/bin/env python3
"""
GTZAN-Genre(mirdata)와 GiantSteps에 대해 동일한 전처리·동일 지표로 tempo(및 가능 시 beat) 평가.

지표 (tempo): pct_error, MAE(BPM), ACC1(4%), ACC2(octave 허용 4%)
지표 (beat, GTZAN만 GT 비트 있음): madmom 추정 비트 vs 참조 비트 — mir_eval F-measure

예:
  python scripts/run_unified_eval.py --dataset giantsteps --sample-size 50
  python scripts/run_unified_eval.py --dataset gtzan_genre --data-home dataset/mirdata_gtzan_genre --sample-size 100
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from tqdm import tqdm

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_common import (
    AUDIO_EXTENSIONS_DEFAULT,
    _fmt_exc,
    acc1_hit,
    acc2_hit,
    beat_fmeasure,
    estimate_dsp_tempo,
    estimate_madmom_beats,
    mae_bpm,
    pct_error,
    tempo_from_beat_times,
)


def load_gt_bpm_file(bpm_path: Path) -> Optional[float]:
    try:
        text = bpm_path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        v = float(text.split()[0])
        if v <= 0:
            return None
        return v
    except Exception:
        return None


def load_excluded_track_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        ids = payload.get("invalid_track_ids", [])
        if isinstance(ids, list):
            return {str(x) for x in ids}
    except Exception:
        return set()
    return set()


def candidate_track_ids_from_bpm_stem(stem: str) -> set[str]:
    candidates = {stem, stem.split(".")[0]}
    if stem.startswith("gtzan_"):
        parts = stem.split("_")
        if len(parts) >= 3:
            candidates.add(f"{parts[1]}.{parts[2]}")
    return candidates


def find_audio_giantsteps(bpm_stem: str, audio_root: Path) -> Optional[Path]:
    for ext in AUDIO_EXTENSIONS_DEFAULT:
        direct = audio_root / f"{bpm_stem}{ext}"
        if direct.is_file():
            return direct
    for ext in AUDIO_EXTENSIONS_DEFAULT:
        matches = list(audio_root.rglob(f"{bpm_stem}{ext}"))
        if matches:
            return matches[0]
    return None


def iter_giantsteps_tasks(
    annotation_dir: Path,
    audio_root: Path,
    excluded: set[str],
    limit: Optional[int],
    seed: int,
) -> Iterator[tuple[str, Path, Optional[Path]]]:
    bpm_files = sorted(annotation_dir.rglob("*.bpm"))
    rng = random.Random(seed)
    rng.shuffle(bpm_files)
    n = 0
    for bpm_file in bpm_files:
        stem = bpm_file.stem
        cands = candidate_track_ids_from_bpm_stem(stem)
        if excluded.intersection(cands):
            continue
        audio = find_audio_giantsteps(stem, audio_root)
        yield stem, bpm_file, audio
        n += 1
        if limit is not None and n >= limit:
            break


def iter_gtzan_mirdata_tasks(
    data_home: Path,
    version: str,
    excluded: set[str],
    limit: Optional[int],
    seed: int,
) -> Iterator[tuple[str, object, Path]]:
    from mirdata.datasets.gtzan_genre import Dataset

    ds = Dataset(data_home=str(data_home), version=version)
    ids = list(ds.track_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = 0
    for tid in ids:
        if tid in excluded:
            continue
        track = ds.track(tid)
        ap = Path(track.audio_path) if track.audio_path else None
        if ap is None or not ap.is_file():
            continue
        yield tid, track, ap
        n += 1
        if limit is not None and n >= limit:
            break


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified tempo/beat eval (GTZAN + GiantSteps)")
    p.add_argument(
        "--dataset",
        choices=("giantsteps", "gtzan_genre"),
        required=True,
    )
    p.add_argument("--sample-size", type=int, default=None, help="처리할 트랙 수 (기본: 전부)")
    p.add_argument("--seed", type=int, default=0, help="셔플 시드")
    p.add_argument("--sample-rate", type=int, default=22050)
    p.add_argument(
        "--giantsteps-root",
        type=Path,
        default=Path("dataset/giantsteps-tempo-dataset-master"),
    )
    p.add_argument("--data-home", type=Path, default=Path("dataset/mirdata_gtzan_genre"))
    p.add_argument("--gtzan-version", type=str, default="default")
    p.add_argument(
        "--exclude-invalid-json",
        type=Path,
        default=Path("results/invalid_tracks_gtzan_genre.json"),
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/unified_eval.csv"),
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=Path("results/unified_eval_summary.json"),
        help="집계(평균 ACC1 등) JSON",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="tqdm 진행 표시 끄기 (로그만 필요할 때)",
    )
    return p.parse_args()


def summarize(rows: list[dict], methods: list[str]) -> dict:
    """
    집계에 포함하는 status:
    - ok: DSP·madmom 모두 성공
    - dsp_ok_madmom_fail: madmom만 실패 (DSP 지표 집계)
    - madmom_ok_dsp_fail: DSP만 실패 (madmom 지표 집계)
    """
    def mean(xs: list[float]) -> Optional[float]:
        return float(np.mean(xs)) if xs else None

    def row_ok_for_dsp(r: dict) -> bool:
        st = r.get("status")
        return st in ("ok", "dsp_ok_madmom_fail")

    def row_ok_for_madmom(r: dict) -> bool:
        return r.get("status") in ("ok", "madmom_ok_dsp_fail")

    out: dict = {}
    for m in methods:
        prefix = m
        acc1_vals: list[float] = []
        acc2_vals: list[float] = []
        mae_vals: list[float] = []
        pct_vals: list[float] = []
        f_vals: list[float] = []
        for r in rows:
            if m == "dsp" and not row_ok_for_dsp(r):
                continue
            if m == "madmom" and not row_ok_for_madmom(r):
                continue
            a1 = r.get(f"{prefix}_acc1")
            if a1 is not None and a1 != "":
                acc1_vals.append(1.0 if a1 is True else 0.0)
            a2 = r.get(f"{prefix}_acc2")
            if a2 is not None and a2 != "":
                acc2_vals.append(1.0 if a2 is True else 0.0)
            mb = r.get(f"{prefix}_mae_bpm")
            if mb is not None and mb != "":
                mae_vals.append(float(mb))
            pe = r.get(f"{prefix}_error_pct")
            if pe is not None and pe != "":
                pct_vals.append(float(pe))
            fv = r.get(f"{prefix}_beat_fmeasure")
            if fv is not None and fv != "":
                f_vals.append(float(fv))
        out[m] = {
            "n_acc": len(acc1_vals),
            "mean_acc1": mean(acc1_vals),
            "mean_acc2": mean(acc2_vals),
            "mean_mae_bpm": mean(mae_vals),
            "mean_error_pct": mean(pct_vals),
            "n_beat_f": len(f_vals),
            "mean_beat_fmeasure": mean(f_vals),
        }
    return out


def main() -> None:
    args = parse_args()
    excluded = load_excluded_track_ids(args.exclude_invalid_json)
    rows: list[dict[str, object]] = []

    limit = args.sample_size

    if args.dataset == "giantsteps":
        ann = args.giantsteps_root / "annotations_v2" / "tempo"
        audio_root = args.giantsteps_root / "audio"
        if not ann.is_dir():
            raise FileNotFoundError(f"annotation 없음: {ann}")
        tasks = list(
            iter_giantsteps_tasks(ann, audio_root, excluded, limit, args.seed)
        )
        dataset_label = "giantsteps"
    else:
        tasks = list(
            iter_gtzan_mirdata_tasks(
                args.data_home, args.gtzan_version, excluded, limit, args.seed
            )
        )
        dataset_label = "gtzan_genre"

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    n_tasks = len(tasks)
    print(
        f"[unified-eval] {n_tasks} tracks — madmom(RNN+DBN)는 CPU에서 곡당 수십 초~수분 걸릴 수 있음. "
        "멈춘 것처럼 보여도 계산 중일 수 있음 (Ctrl+C = 중단).",
        flush=True,
    )
    task_iter = tqdm(
        tasks,
        total=n_tasks,
        desc=f"unified-eval[{dataset_label}]",
        unit="track",
        disable=args.no_progress or n_tasks == 0,
    )

    for item in task_iter:
        if args.dataset == "giantsteps":
            stem, bpm_path, audio_path = item
            track_id = stem.split(".")[0]
            file_or_stem = stem
            gt_bpm = load_gt_bpm_file(bpm_path)
            ref_beats = None
        else:
            tid, track, audio_path = item
            track_id = tid
            file_or_stem = tid
            bpm_path = Path(track.tempo_path) if track.tempo_path else None
            gt_bpm = float(track.tempo) if track.tempo is not None else None
            ref_beats = None
            if track.beats is not None and track.beats.times is not None:
                ref_beats = np.asarray(track.beats.times, dtype=float).ravel()

        row: dict[str, object] = {
            "dataset": dataset_label,
            "track_id": track_id,
            "file_or_stem": file_or_stem,
            "annotation_path": str(bpm_path) if bpm_path else "",
            "audio_path": str(audio_path) if audio_path else "",
            "gt_bpm": gt_bpm if gt_bpm is not None else "",
            "dsp_tempo_bpm": "",
            "madmom_tempo_bpm": "",
            "num_beats_madmom": "",
            "dsp_error_pct": "",
            "madmom_error_pct": "",
            "dsp_mae_bpm": "",
            "madmom_mae_bpm": "",
            "dsp_acc1": "",
            "madmom_acc1": "",
            "dsp_acc2": "",
            "madmom_acc2": "",
            "dsp_beat_fmeasure": "",
            "madmom_beat_fmeasure": "",
            "status": "ok",
            "note": "",
        }

        if gt_bpm is None:
            row["status"] = "skip_bad_gt"
            row["note"] = "GT BPM 없음"
            rows.append(row)
            continue

        if audio_path is None or not audio_path.is_file():
            row["status"] = "missing_audio"
            row["note"] = "오디오 없음"
            rows.append(row)
            continue

        try:
            dsp_tempo = estimate_dsp_tempo(audio_path, args.sample_rate)
        except Exception as exc:
            dsp_tempo = None
            row["note"] = f"dsp:{_fmt_exc(exc)}"

        madmom_beats, mm_st = estimate_madmom_beats(audio_path)
        mm_tempo = tempo_from_beat_times(madmom_beats) if mm_st == "ok" else None
        n_b = int(madmom_beats.size)
        if mm_st != "ok":
            extra = f"madmom:{mm_st}"
            row["note"] = f"{row['note']};{extra}" if row["note"] else extra
            row["status"] = "dsp_ok_madmom_fail" if dsp_tempo is not None else "eval_fail"
        elif dsp_tempo is None:
            row["status"] = "madmom_ok_dsp_fail"

        for prefix, pred in ("dsp", dsp_tempo), ("madmom", mm_tempo):
            row[f"{prefix}_tempo_bpm"] = "" if pred is None else round(float(pred), 4)
            pe = pct_error(pred, gt_bpm)
            row[f"{prefix}_error_pct"] = "" if pe is None else round(pe, 4)
            mb = mae_bpm(pred, gt_bpm)
            row[f"{prefix}_mae_bpm"] = "" if mb is None else round(mb, 4)
            a1 = acc1_hit(pred, gt_bpm)
            a2 = acc2_hit(pred, gt_bpm)
            row[f"{prefix}_acc1"] = "" if a1 is None else a1
            row[f"{prefix}_acc2"] = "" if a2 is None else a2

        row["num_beats_madmom"] = n_b

        if ref_beats is not None and ref_beats.size > 0 and madmom_beats.size > 0:
            fd = beat_fmeasure(ref_beats, madmom_beats)
            row["madmom_beat_fmeasure"] = "" if fd is None else round(fd, 6)
            if dsp_tempo is not None:
                row["dsp_beat_fmeasure"] = ""
        else:
            row["dsp_beat_fmeasure"] = ""
            row["madmom_beat_fmeasure"] = ""

        rows.append(row)

    fieldnames = [
        "dataset",
        "track_id",
        "file_or_stem",
        "annotation_path",
        "audio_path",
        "gt_bpm",
        "dsp_tempo_bpm",
        "madmom_tempo_bpm",
        "num_beats_madmom",
        "dsp_error_pct",
        "madmom_error_pct",
        "dsp_mae_bpm",
        "madmom_mae_bpm",
        "dsp_acc1",
        "madmom_acc1",
        "dsp_acc2",
        "madmom_acc2",
        "dsp_beat_fmeasure",
        "madmom_beat_fmeasure",
        "status",
        "note",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    methods = ["dsp", "madmom"]
    summary = {
        "dataset": dataset_label,
        "n_rows": len(rows),
        "metrics": summarize(rows, methods),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    ok = sum(1 for r in rows if r["status"] == "ok")
    partial = sum(1 for r in rows if r["status"] == "dsp_ok_madmom_fail")
    status_counts = dict(Counter(str(r["status"]) for r in rows))
    print(f"[unified-eval] wrote: {args.output_csv}")
    print(f"[unified-eval] summary: {args.summary_json}")
    print(
        f"[unified-eval] rows={len(rows)}, ok(both)={ok}, "
        f"dsp_only={partial}, dataset={dataset_label}"
    )
    print(f"[unified-eval] status_counts: {status_counts}")


if __name__ == "__main__":
    main()
