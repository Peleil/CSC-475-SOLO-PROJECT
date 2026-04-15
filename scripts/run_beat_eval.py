#!/usr/bin/env python3
"""
Beat evaluation: dsp, madmom, and 1D-StateSpace (up to 3 CSV files)
(beat_dsp.csv, beat_madmom.csv, beat_1dss.csv).

GTZAN-Genre only (reference beats: gtzan_tempo_beat-main/beats/*.beats, first
column in seconds). GiantSteps has no public beat ground truth in this setup.

Paper-style reproduction notes (Heydari et al., arXiv:2111.00704v2, Table 1):
  - Metrics: mir_eval.beat.f_measure (default window 0.07 s),
    mir_eval.beat.cemgil (default sigma 0.04).
  - 1DSS input: with --one-dss-original-audio, BeatNet loads source files
    directly at 22050 Hz.
  - Full dataset: omit --sample-size. Summary JSON: --write-summary-json.

Output directory defaults to results/ and follows --methods.

Environments:
  - csc475-dsp-madmom + requirements-dsp-madmom.txt → --methods dsp,madmom
  - csc475-1dss + requirements-1dss.txt → --methods 1dss
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_common import (
    DEFAULT_UNIFIED_SAMPLE_RATE,
    beat_cemgil,
    beat_fmeasure,
    estimate_1dss_beats_and_tempo,
    estimate_dsp_beats,
    estimate_madmom_beats_via_tempo_dbn,
)
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
    "beat_cemgil",
    "beat_cemgil_best",
    "status",
    "note",
]


def _parse_beat_methods(s: str) -> tuple[list[str], frozenset[str]]:
    allowed = ("dsp", "madmom", "1dss")
    seen: list[str] = []
    for part in s.split(","):
        p = part.strip().lower()
        if not p:
            continue
        if p not in allowed:
            raise SystemExit(
                f"[beat-eval] unknown method {part!r}; allowed: {', '.join(allowed)}"
            )
        if p not in seen:
            seen.append(p)
    if not seen:
        raise SystemExit("[beat-eval] --methods needs at least one of: dsp, madmom, 1dss")
    return seen, frozenset(seen)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Beat eval (GTZAN only): dsp, madmom, 1dss — CSV per method"
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
        "--sample-rate",
        type=int,
        default=DEFAULT_UNIFIED_SAMPLE_RATE,
        help="Shared madmom/1dss mono resample rate in Hz. Default: 44100.",
    )
    p.add_argument(
        "--sample-rate-dsp",
        type=int,
        default=None,
        help="Compatibility alias: if set, overrides --sample-rate.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
    )
    p.add_argument(
        "--methods",
        type=str,
        default="madmom",
        help="Comma-separated: dsp, madmom, 1dss (dsp-madmom env: dsp,madmom — 1dss env: 1dss)",
    )
    p.add_argument(
        "--f-measure-threshold-s",
        type=float,
        default=0.07,
        help="mir_eval beat F-measure matching window (seconds), default 0.07.",
    )
    p.add_argument(
        "--cemgil-sigma",
        type=float,
        default=0.04,
        help="mir_eval.beat.cemgil sigma in seconds. Default: 0.04.",
    )
    p.add_argument(
        "--one-dss-original-audio",
        action="store_true",
        help="1DSS only: pass original audio path to joint_inference.process.",
    )
    p.add_argument(
        "--write-summary-json",
        action="store_true",
        help="Write beat_summary_<method>.json with aggregate metrics.",
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
        "beat_cemgil": "",
        "beat_cemgil_best": "",
        "status": "ok",
        "note": "",
    }


def _fill_est(
    row: dict[str, Any],
    est: np.ndarray,
    ref: Optional[np.ndarray],
    f_window_s: float,
    cemgil_sigma: float,
) -> None:
    row["num_est_beats"] = int(est.size)
    if ref is None or ref.size == 0:
        row["note"] = "no_reference_beats"
        return
    fd = beat_fmeasure(ref, est, window=f_window_s)
    row["beat_fmeasure"] = "" if fd is None else round(fd, 6)
    cg, cg_best = beat_cemgil(ref, est, sigma=cemgil_sigma)
    row["beat_cemgil"] = "" if cg is None else round(cg, 6)
    row["beat_cemgil_best"] = "" if cg_best is None else round(cg_best, 6)


def _beat_summary_for_method(
    tag: str,
    rows: list[dict[str, Any]],
    f_window_s: float,
    cemgil_sigma: float,
    one_dss_original: bool,
    eval_sr: int,
) -> dict[str, Any]:
    """Aggregation suitable for direct comparison with paper-style tables."""
    fs: list[float] = []
    cgs: list[float] = []
    cg_bests: list[float] = []
    n_ref_ok = 0
    status_counts: dict[str, int] = {}
    for r in rows:
        st = str(r.get("status", ""))
        status_counts[st] = status_counts.get(st, 0) + 1
        if st != "ok":
            continue
        if not r.get("has_ref_beats"):
            continue
        n_ref_ok += 1
        v = r.get("beat_fmeasure")
        if v != "" and v is not None:
            try:
                fs.append(float(v))
            except (TypeError, ValueError):
                pass
        cg = r.get("beat_cemgil")
        if cg != "" and cg is not None:
            try:
                cgs.append(float(cg))
            except (TypeError, ValueError):
                pass
        cgb = r.get("beat_cemgil_best")
        if cgb != "" and cgb is not None:
            try:
                cg_bests.append(float(cgb))
            except (TypeError, ValueError):
                pass
    arr = np.asarray(fs, dtype=float) if fs else np.array([], dtype=float)
    arr_cg = np.asarray(cgs, dtype=float) if cgs else np.array([], dtype=float)
    arr_cgb = np.asarray(cg_bests, dtype=float) if cg_bests else np.array([], dtype=float)
    paper = {
        "citation": "Heydari et al., arXiv:2111.00704v2 (ICASSP 2022), Table 1, GTZAN",
        "reported_beat_fmeasure_1d_state_space": 76.48,
        "reported_beat_fmeasure_beatnet_pf": 75.44,
    }
    out = {
        "method": tag,
        "paper_reference": paper,
        "metric": "mir_eval.beat.f_measure",
        "f_measure_threshold_s": f_window_s,
        "cemgil_metric": "mir_eval.beat.cemgil",
        "cemgil_sigma_s": cemgil_sigma,
        "eval_sample_rate_hz_madmom_and_resampled_1dss": eval_sr,
        "n_rows": len(rows),
        "status_counts": status_counts,
        "n_status_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "n_with_reference_beats": n_ref_ok,
        "n_with_beat_fmeasure": int(arr.size),
        "mean_beat_fmeasure": round(float(np.mean(arr)), 6) if arr.size else None,
        "median_beat_fmeasure": round(float(np.median(arr)), 6) if arr.size else None,
        "std_beat_fmeasure": round(float(np.std(arr)), 6) if arr.size else None,
        "n_with_beat_cemgil": int(arr_cg.size),
        "mean_beat_cemgil": round(float(np.mean(arr_cg)), 6) if arr_cg.size else None,
        "median_beat_cemgil": round(float(np.median(arr_cg)), 6) if arr_cg.size else None,
        "std_beat_cemgil": round(float(np.std(arr_cg)), 6) if arr_cg.size else None,
        "n_with_beat_cemgil_best": int(arr_cgb.size),
        "mean_beat_cemgil_best": round(float(np.mean(arr_cgb)), 6) if arr_cgb.size else None,
        "median_beat_cemgil_best": round(float(np.median(arr_cgb)), 6) if arr_cgb.size else None,
        "std_beat_cemgil_best": round(float(np.std(arr_cgb)), 6) if arr_cgb.size else None,
        "note": "mean_* values are arithmetic means over per-track F scores.",
    }
    if tag == "1dss":
        out["use_backend_native_audio"] = one_dss_original
    return out


def main() -> None:
    args = parse_args()
    eval_sr = (
        args.sample_rate_dsp
        if args.sample_rate_dsp is not None
        else args.sample_rate
    )
    methods, methods_set = _parse_beat_methods(args.methods)
    excluded = load_excluded_track_ids(args.exclude_invalid_json)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    rows_by: dict[str, list[dict]] = {m: [] for m in methods}

    tasks = list(
        iter_gtzan_tasks(args.data_home, excluded, args.sample_size, args.seed)
    )
    it = tqdm(tasks, desc="beat[gtzan_genre]", disable=args.no_progress)
    for tid, _bpm, audio_path, beats_path in it:
        ref = load_gtzan_beat_times(beats_path) if beats_path else None

        if audio_path is None or not audio_path.is_file():
            for tag in methods:
                r = _row_base(tid, audio_path, beats_path, ref)
                r["status"] = "missing_audio"
                r["note"] = "missing_audio"
                rows_by[tag].append(r)
            continue

        if "dsp" in methods_set:
            rd = _row_base(tid, audio_path, beats_path, ref)
            dsp_b, _dsp_t, dsp_st = estimate_dsp_beats(audio_path, sample_rate=eval_sr)
            if dsp_st != "ok":
                rd["status"] = "dsp_fail"
                rd["note"] = dsp_st
            else:
                _fill_est(
                    rd,
                    dsp_b,
                    ref,
                    args.f_measure_threshold_s,
                    args.cemgil_sigma,
                )
            rows_by["dsp"].append(rd)

        if "madmom" in methods_set:
            rm = _row_base(tid, audio_path, beats_path, ref)
            mm_b, mm_st = estimate_madmom_beats_via_tempo_dbn(
                audio_path, sample_rate=eval_sr
            )
            if mm_st != "ok":
                rm["status"] = "madmom_fail"
                rm["note"] = mm_st
            else:
                _fill_est(
                    rm,
                    mm_b,
                    ref,
                    args.f_measure_threshold_s,
                    args.cemgil_sigma,
                )
            rows_by["madmom"].append(rm)

        if "1dss" in methods_set:
            r1 = _row_base(tid, audio_path, beats_path, ref)
            ss_b, _t, ss_st = estimate_1dss_beats_and_tempo(
                audio_path,
                sample_rate=eval_sr,
                use_backend_native_audio=args.one_dss_original_audio,
            )
            if ss_st != "ok":
                r1["status"] = "1dss_fail"
                r1["note"] = ss_st
            else:
                _fill_est(
                    r1,
                    ss_b,
                    ref,
                    args.f_measure_threshold_s,
                    args.cemgil_sigma,
                )
            rows_by["1dss"].append(r1)

    out_names = {
        "dsp": "beat_dsp.csv",
        "madmom": "beat_madmom.csv",
        "1dss": "beat_1dss.csv",
    }
    written: list[str] = []
    for tag in methods:
        p = out / out_names[tag]
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=BEAT_FIELDNAMES,
                extrasaction="ignore",
            )
            w.writeheader()
            for r in rows_by[tag]:
                line = dict(r)
                line["has_ref_beats"] = (
                    "true" if line.get("has_ref_beats") else "false"
                )
                w.writerow(line)
        written.append(str(p))

    print(f"[beat-eval] methods={','.join(methods)}")
    print(f"[beat-eval] unified_sample_rate_hz={eval_sr}")
    print(f"[beat-eval] f_measure_threshold_s={args.f_measure_threshold_s}")
    print(f"[beat-eval] cemgil_sigma_s={args.cemgil_sigma}")
    if args.one_dss_original_audio and "1dss" in methods_set:
        print("[beat-eval] 1dss: BeatNet native load via original file path")
    print("[beat-eval] wrote:\n  " + "\n  ".join(written))

    if args.write_summary_json:
        for tag in methods:
            summ = _beat_summary_for_method(
                tag,
                rows_by[tag],
                args.f_measure_threshold_s,
                args.cemgil_sigma,
                args.one_dss_original_audio,
                eval_sr,
            )
            sp = out / f"beat_summary_{tag}.json"
            sp.write_text(
                json.dumps(summ, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"[beat-eval] wrote {sp}")


if __name__ == "__main__":
    main()
