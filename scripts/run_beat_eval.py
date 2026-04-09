#!/usr/bin/env python3
"""
비트 평가: madmom · 1D-StateSpace — CSV 최대 2개 (beat_madmom.csv, beat_1dss.csv).

GTZAN-Genre만 대상 (참 비트: gtzan_tempo_beat-main/beats/*.beats, 첫 열 = 초).
GiantSteps 공개 템포 세트에는 비트 GT가 없어 비트 평가에서 제외한다.

논문 재현 (Heydari et al., arXiv:2111.00704v2, Table 1 — GTZAN 비트 F-measure):
  - 지표: mir_eval.beat.f_measure, 기본 창 0.07 s (MIREX 계열).
  - 1DSS 입력: --one-dss-original-audio 를 켜면 BeatNet이 파일을 22050 Hz로 직접 로드 (공식 예제와 동일).
  - 전체 1000곡: --sample-size 생략. 요약 JSON: --write-summary-json

출력 (기본 results/): --methods 로 선택

환경:
  - csc475-dsp-madmom + requirements-dsp-madmom.txt → --methods madmom
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
    beat_fmeasure,
    estimate_1dss_beats_and_tempo,
    estimate_madmom_beats,
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
    "status",
    "note",
]


def _parse_beat_methods(s: str) -> tuple[list[str], frozenset[str]]:
    allowed = ("madmom", "1dss")
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
        raise SystemExit("[beat-eval] --methods needs at least one of: madmom, 1dss")
    return seen, frozenset(seen)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Beat eval (GTZAN only): madmom, 1dss — CSV per method"
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
        help="madmom/1dss 공통: mono 리샘플 Hz. 기본 44100.",
    )
    p.add_argument(
        "--sample-rate-dsp",
        type=int,
        default=None,
        help="호환: 지정 시 --sample-rate 대신 이 Hz로 통일.",
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
        help="Comma-separated: madmom, 1dss (dsp-madmom env: madmom — 1dss env: 1dss)",
    )
    p.add_argument(
        "--f-measure-threshold-s",
        type=float,
        default=0.07,
        help="mir_eval beat F-measure 매칭 창(초). 논문/MIREX 계열 기본 0.07.",
    )
    p.add_argument(
        "--one-dss-original-audio",
        action="store_true",
        help="1DSS만: 리샘플·임시 WAV 없이 원본 경로로 joint_inference.process (BeatNet 22050 로드).",
    )
    p.add_argument(
        "--write-summary-json",
        action="store_true",
        help="methods별 beat_summary_<method>.json (평균·중앙값 F 등) 기록.",
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
    f_window_s: float,
) -> None:
    row["num_est_beats"] = int(est.size)
    if ref is None or ref.size == 0:
        row["note"] = "no_reference_beats"
        return
    fd = beat_fmeasure(ref, est, window=f_window_s)
    row["beat_fmeasure"] = "" if fd is None else round(fd, 6)


def _beat_summary_for_method(
    tag: str,
    rows: list[dict[str, Any]],
    f_window_s: float,
    one_dss_original: bool,
    eval_sr: int,
) -> dict[str, Any]:
    """논문 Table 1 과 직접 비교 가능한 집계."""
    fs: list[float] = []
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
    arr = np.asarray(fs, dtype=float) if fs else np.array([], dtype=float)
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
        "eval_sample_rate_hz_madmom_and_resampled_1dss": eval_sr,
        "n_rows": len(rows),
        "status_counts": status_counts,
        "n_status_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "n_with_reference_beats": n_ref_ok,
        "n_with_beat_fmeasure": int(arr.size),
        "mean_beat_fmeasure": round(float(np.mean(arr)), 6) if arr.size else None,
        "median_beat_fmeasure": round(float(np.median(arr)), 6) if arr.size else None,
        "std_beat_fmeasure": round(float(np.std(arr)), 6) if arr.size else None,
        "note": "mean_* 은 트랙별 F의 산술평균; 논문 Table 1은 전체 GTZAN에 대한 단일 F일 수 있음(보고 방식 확인).",
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
                r["note"] = "오디오 없음"
                rows_by[tag].append(r)
            continue

        if "madmom" in methods_set:
            rm = _row_base(tid, audio_path, beats_path, ref)
            mm_b, mm_st = estimate_madmom_beats(audio_path, sample_rate=eval_sr)
            if mm_st != "ok":
                rm["status"] = "madmom_fail"
                rm["note"] = mm_st
            else:
                _fill_est(rm, mm_b, ref, args.f_measure_threshold_s)
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
                _fill_est(r1, ss_b, ref, args.f_measure_threshold_s)
            rows_by["1dss"].append(r1)

    out_names = {"madmom": "beat_madmom.csv", "1dss": "beat_1dss.csv"}
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
    if args.one_dss_original_audio and "1dss" in methods_set:
        print("[beat-eval] 1dss: BeatNet native load via original file path")
    print("[beat-eval] wrote:\n  " + "\n  ".join(written))

    if args.write_summary_json:
        for tag in methods:
            summ = _beat_summary_for_method(
                tag,
                rows_by[tag],
                args.f_measure_threshold_s,
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
