#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_common import (
    DEFAULT_UNIFIED_SAMPLE_RATE,
    _fmt_exc,
    acc1_hit,
    acc2_hit,
    estimate_1dss_beats_and_tempo,
    estimate_autocorr_tempo,
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


def _base_row(dataset: str, track_id: str, file_or_stem: str, bpm_path: Path, audio_path: Optional[Path], gt_bpm: Optional[float]) -> dict[str, Any]:
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


def _fill_tempo_metrics(row: dict[str, Any], pred: Optional[float], gt: Optional[float]) -> None:
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


def _float_cell(row: dict[str, Any], key: str) -> Optional[float]:
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _bool_cell(row: dict[str, Any], key: str) -> Optional[bool]:
    v = row.get(key)
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    return None


def summarize_tempo_rows(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = dict(Counter(str(r.get("status", "")) for r in rows))
    by_ds: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_ds.setdefault(str(r.get("dataset", "")), []).append(r)

    def _agg(sub: list[dict[str, Any]]) -> dict[str, Any]:
        mrows = []
        for r in sub:
            if str(r.get("status")) != "ok":
                continue
            if r.get("pred_tempo_bpm") in ("", None):
                continue
            if _float_cell(r, "error_pct") is None:
                continue
            mrows.append(r)
        if not mrows:
            return {
                "n_with_metrics": 0,
                "mean_error_pct": None,
                "mean_mae_bpm": None,
                "acc1_rate": None,
                "acc2_rate": None,
            }
        pe = [_float_cell(x, "error_pct") for x in mrows]
        mb = [_float_cell(x, "mae_bpm") for x in mrows]
        a1_ok = [b for x in mrows if (b := _bool_cell(x, "acc1")) is not None]
        a2_ok = [b for x in mrows if (b := _bool_cell(x, "acc2")) is not None]
        return {
            "n_with_metrics": len(mrows),
            "mean_error_pct": round(sum(p for p in pe if p is not None) / len(mrows), 6),
            "mean_mae_bpm": round(sum(m for m in mb if m is not None) / len(mrows), 6),
            "acc1_rate": round(sum(a1_ok) / len(a1_ok), 6) if a1_ok else None,
            "acc2_rate": round(sum(a2_ok) / len(a2_ok), 6) if a2_ok else None,
        }

    return {
        "method": method,
        "n_rows": len(rows),
        "status_counts": status_counts,
        "overall": _agg(rows),
        "by_dataset": {ds: {"n_rows": len(rs), "status_counts": dict(Counter(str(r.get("status", "")) for r in rs)), **_agg(rs)} for ds, rs in sorted(by_ds.items())},
    }


def _parse_methods(s: str) -> tuple[list[str], frozenset[str]]:
    allowed = ("dsp", "madmom", "1dss", "autocorr")
    seen: list[str] = []
    for part in s.split(","):
        p = part.strip().lower()
        if not p:
            continue
        if p not in allowed:
            raise SystemExit(f"[tempo-eval] unknown method {part!r}; allowed: {', '.join(allowed)}")
        if p not in seen:
            seen.append(p)
    if not seen:
        raise SystemExit("[tempo-eval] --methods needs at least one of: dsp, madmom, 1dss, autocorr")
    return seen, frozenset(seen)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tempo eval: DSP, madmom, 1dss, autocorr — CSV per method")
    p.add_argument("--dataset", choices=("giantsteps", "gtzan_genre", "both"), default="both")
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--giantsteps-root", type=Path, default=Path("dataset/giantsteps-tempo-dataset-master"))
    p.add_argument("--data-home", type=Path, default=Path("dataset/mirdata_gtzan_genre"))
    p.add_argument("--exclude-invalid-json", type=Path, default=Path("results/invalid_tracks_gtzan_genre.json"))
    p.add_argument("--sample-rate", type=int, default=DEFAULT_UNIFIED_SAMPLE_RATE)
    p.add_argument("--sample-rate-dsp", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--methods", type=str, default="dsp,madmom")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def _predict_rows_for_track(dataset: str, tid: str, stem_or_tid: str, bpm_path: Path, audio_path: Optional[Path], gt: Optional[float], eval_sr: int, methods_set: frozenset[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for tag in methods_set:
        out[tag] = _base_row(dataset, tid, stem_or_tid, bpm_path, audio_path, gt)
    if gt is None:
        for tag in methods_set:
            out[tag]["status"] = "skip_bad_gt"
            out[tag]["note"] = "GT BPM 없음"
        return out
    if audio_path is None or not audio_path.is_file():
        for tag in methods_set:
            out[tag]["status"] = "missing_audio"
            out[tag]["note"] = "오디오 없음"
        return out

    if "dsp" in methods_set:
        r = out["dsp"]
        try:
            t = estimate_dsp_tempo(audio_path, eval_sr)
        except Exception as exc:
            t = None
            r["note"] = f"dsp:{_fmt_exc(exc)}"
            r["status"] = "dsp_fail"
        _fill_tempo_metrics(r, t, gt)
        if r["status"] == "ok" and t is None:
            r["status"] = "dsp_fail"

    if "madmom" in methods_set:
        r = out["madmom"]
        b, st = estimate_madmom_beats(audio_path, sample_rate=eval_sr)
        t = tempo_from_beat_times(b) if st == "ok" else None
        if st != "ok":
            r["status"] = "madmom_fail"
            r["note"] = st
        else:
            _fill_tempo_metrics(r, t, gt)

    if "1dss" in methods_set:
        r = out["1dss"]
        b, t, st = estimate_1dss_beats_and_tempo(audio_path, sample_rate=eval_sr)
        if st != "ok":
            r["status"] = "1dss_fail"
            r["note"] = st
        else:
            # 기존 동작: local tempo 중앙값 우선, 없으면 IBI 기반
            _fill_tempo_metrics(r, t, gt)

    if "autocorr" in methods_set:
        r = out["autocorr"]
        try:
            t = estimate_autocorr_tempo(audio_path, eval_sr)
        except Exception as exc:
            t = None
            r["note"] = f"autocorr:{_fmt_exc(exc)}"
            r["status"] = "autocorr_fail"
        _fill_tempo_metrics(r, t, gt)
        if r["status"] == "ok" and t is None:
            r["status"] = "autocorr_fail"

    return out


def main() -> None:
    args = parse_args()
    eval_sr = args.sample_rate_dsp if args.sample_rate_dsp is not None else args.sample_rate
    methods, methods_set = _parse_methods(args.methods)
    excluded = load_excluded_track_ids(args.exclude_invalid_json)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}

    if args.dataset in ("giantsteps", "both"):
        ann = args.giantsteps_root / "annotations_v2" / "tempo"
        audio_root = args.giantsteps_root / "audio"
        if not ann.is_dir():
            raise FileNotFoundError(f"GiantSteps annotation 없음: {ann}")
        tasks = list(iter_giantsteps_tasks(ann, audio_root, excluded, args.sample_size, args.seed))
        for stem, bpm_path, audio_path in tqdm(tasks, desc="tempo[giantsteps]", disable=args.no_progress):
            tid = stem.split(".")[0]
            gt = load_gt_bpm_file(bpm_path)
            pred = _predict_rows_for_track("giantsteps", tid, stem, bpm_path, audio_path, gt, eval_sr, methods_set)
            for tag in methods:
                rows_by[tag].append(pred[tag])

    if args.dataset in ("gtzan_genre", "both"):
        seed_g = args.seed + 17 if args.dataset == "both" else args.seed
        tasks = list(iter_gtzan_tasks(args.data_home, excluded, args.sample_size, seed_g))
        for tid, bpm_path, audio_path, _beats in tqdm(tasks, desc="tempo[gtzan_genre]", disable=args.no_progress):
            gt = load_gt_bpm_file(bpm_path)
            pred = _predict_rows_for_track("gtzan_genre", tid, tid, bpm_path, audio_path, gt, eval_sr, methods_set)
            for tag in methods:
                rows_by[tag].append(pred[tag])

    out_names = {"dsp": "tempo_dsp.csv", "madmom": "tempo_madmom.csv", "1dss": "tempo_1dss.csv", "autocorr": "tempo_autocorr.csv"}
    written: list[str] = []
    for tag in methods:
        p = out_dir / out_names[tag]
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=TEMPO_FIELDNAMES, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows_by[tag])
        written.append(str(p))

    for tag in methods:
        summ = summarize_tempo_rows(tag, rows_by[tag])
        sp = out_dir / f"tempo_summary_{tag}.json"
        sp.write_text(json.dumps(summ, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(str(sp))

    print(f"[tempo-eval] methods={','.join(methods)}")
    print(f"[tempo-eval] unified_sample_rate_hz={eval_sr}")
    print("[tempo-eval] wrote:\n  " + "\n  ".join(written))


if __name__ == "__main__":
    main()
