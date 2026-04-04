#!/usr/bin/env python3
"""
GiantSteps만 사용할 때: annotations_v2/tempo/*.bpm 과 오디오(.mp3 등)를 매칭해
DSP tempo + madmom(beat 기반 tempo) quick check CSV를 만든다.

GTZAN은 쓰지 않는다. 오디오는 데이터셋 README의 audio_dl.sh 등으로 받아
기본값인 dataset/giantsteps-tempo-dataset-master/audio/ 에 두면 된다.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_common import estimate_dsp_tempo, estimate_madmom_beats_and_tempo

AUDIO_EXTENSIONS = [".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aif", ".aiff"]


def load_gt_bpm(bpm_path: Path) -> Optional[float]:
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


def find_audio_for_bpm(bpm_path: Path, audio_root: Path) -> Optional[Path]:
    """bpm 파일명 stem(예: 1030011.LOFI)과 동일한 이름의 오디오를 찾는다."""
    stem = bpm_path.stem
    for ext in AUDIO_EXTENSIONS:
        direct = audio_root / f"{stem}{ext}"
        if direct.is_file():
            return direct
    for ext in AUDIO_EXTENSIONS:
        matches = list(audio_root.rglob(f"{stem}{ext}"))
        if matches:
            return matches[0]
    return None


def pct_error(pred: Optional[float], gt: Optional[float]) -> Optional[float]:
    if pred is None or gt is None or gt <= 0:
        return None
    return abs(pred - gt) / gt * 100.0


def acc1_hit(pred: Optional[float], gt: Optional[float]) -> Optional[bool]:
    """Gouyon-style: 상대 오차 4% 이내면 True."""
    if pred is None or gt is None or gt <= 0:
        return None
    return abs(pred - gt) / gt <= 0.04


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GiantSteps tempo quick check (DSP + madmom)")
    p.add_argument(
        "--giantsteps-root",
        type=Path,
        default=Path("dataset/giantsteps-tempo-dataset-master"),
        help="GiantSteps 데이터셋 루트 (annotations_v2, audio 등)",
    )
    p.add_argument(
        "--annotation-dir",
        type=Path,
        default=None,
        help="기본: <giantsteps-root>/annotations_v2/tempo",
    )
    p.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="기본: <giantsteps-root>/audio",
    )
    p.add_argument("--sample-size", type=int, default=5, help="처음 N개 .bpm만 처리")
    p.add_argument("--sample-rate", type=int, default=22050)
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/giantsteps_quick_check.csv"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gs = args.giantsteps_root
    ann = args.annotation_dir or (gs / "annotations_v2" / "tempo")
    audio_root = args.audio_root or (gs / "audio")

    if not ann.is_dir():
        raise FileNotFoundError(f"annotation 폴더가 없습니다: {ann}")
    bpm_files = sorted(ann.rglob("*.bpm"))
    if not bpm_files:
        raise FileNotFoundError(f".bpm 파일이 없습니다: {ann}")

    sample_files = bpm_files[: args.sample_size]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for bpm_file in sample_files:
        file_id = bpm_file.stem
        gt_bpm = load_gt_bpm(bpm_file)
        audio_path = find_audio_for_bpm(bpm_file, audio_root)

        row: dict[str, object] = {
            "file_id": file_id,
            "annotation_path": str(bpm_file),
            "audio_path": str(audio_path) if audio_path else "",
            "gt_bpm": gt_bpm if gt_bpm is not None else "",
            "dsp_tempo_bpm": "",
            "madmom_tempo_bpm": "",
            "num_beats_madmom": "",
            "dsp_error_pct": "",
            "madmom_error_pct": "",
            "dsp_acc1": "",
            "madmom_acc1": "",
            "status": "ok",
            "note": "",
        }

        if gt_bpm is None:
            row["status"] = "skip_bad_gt"
            row["note"] = "BPM이 0이거나 파싱 실패"
            rows.append(row)
            continue

        if audio_path is None:
            row["status"] = "missing_audio"
            row["note"] = f"audio_root에 {file_id}.mp3(등) 없음 — {audio_root}"
            rows.append(row)
            continue

        try:
            dsp_tempo = estimate_dsp_tempo(audio_path, args.sample_rate)
        except Exception as exc:
            dsp_tempo = None
            row["note"] = f"dsp_error:{type(exc).__name__}"

        n_beats, madmom_tempo, mm_st = estimate_madmom_beats_and_tempo(audio_path)
        if mm_st != "ok":
            row["status"] = mm_st

        row["dsp_tempo_bpm"] = "" if dsp_tempo is None else round(float(dsp_tempo), 4)
        row["madmom_tempo_bpm"] = "" if madmom_tempo is None else round(float(madmom_tempo), 4)
        row["num_beats_madmom"] = "" if n_beats is None else n_beats

        de = pct_error(dsp_tempo, gt_bpm)
        me = pct_error(madmom_tempo, gt_bpm)
        row["dsp_error_pct"] = "" if de is None else round(de, 4)
        row["madmom_error_pct"] = "" if me is None else round(me, 4)

        a1 = acc1_hit(dsp_tempo, gt_bpm)
        a2 = acc1_hit(madmom_tempo, gt_bpm)
        row["dsp_acc1"] = "" if a1 is None else a1
        row["madmom_acc1"] = "" if a2 is None else a2

        rows.append(row)

    fieldnames = [
        "file_id",
        "annotation_path",
        "audio_path",
        "gt_bpm",
        "dsp_tempo_bpm",
        "madmom_tempo_bpm",
        "num_beats_madmom",
        "dsp_error_pct",
        "madmom_error_pct",
        "dsp_acc1",
        "madmom_acc1",
        "status",
        "note",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"[giantsteps] wrote: {args.output_csv}")
    print(f"[giantsteps] processed: {len(rows)}, ok: {ok}, non-ok: {len(rows) - ok}")
    print(f"[giantsteps] audio_root: {audio_root}")


if __name__ == "__main__":
    main()
