#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_common import estimate_dsp_tempo, estimate_madmom_beats_and_tempo


AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aif", ".aiff"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="샘플 N곡으로 tempo/beat quick check CSV 생성"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="데이터셋 루트 경로",
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=Path("dataset/giantsteps-tempo-dataset-master/annotations_v2/tempo"),
        help=".bpm annotation 폴더",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("dataset"),
        help="오디오 파일 검색 루트 경로",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="검증할 샘플 곡 수",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="librosa 로딩 sample rate",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/quick_check.csv"),
        help="출력 CSV 경로",
    )
    parser.add_argument(
        "--exclude-invalid-json",
        type=Path,
        default=Path("results/invalid_tracks_gtzan_genre.json"),
        help="검증에서 제외할 track id 목록 JSON 경로 (없으면 무시)",
    )
    return parser.parse_args()


def load_gt_bpm(bpm_path: Path) -> Optional[float]:
    try:
        text = bpm_path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        return float(text.split()[0])
    except Exception:
        return None


def find_audio_for_track(track_id: str, audio_root: Path) -> Optional[Path]:
    for ext in AUDIO_EXTENSIONS:
        matches = list(audio_root.rglob(f"{track_id}*{ext}"))
        if matches:
            return matches[0]
    return None


def pct_error(pred: Optional[float], gt: Optional[float]) -> Optional[float]:
    if pred is None or gt is None or gt <= 0:
        return None
    return abs(pred - gt) / gt * 100.0


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


def candidate_track_ids_from_bpm_file(bpm_file: Path) -> set[str]:
    """
    서로 다른 네이밍 규칙을 모두 대응하기 위한 후보 집합.
    예:
    - GiantSteps: 1030011.LOFI.bpm -> {1030011.LOFI, 1030011}
    - GTZAN tempo_beat: gtzan_jazz_00054.bpm -> {gtzan_jazz_00054, jazz.00054}
    """
    stem = bpm_file.stem
    candidates = {stem, stem.split(".")[0]}
    if stem.startswith("gtzan_"):
        parts = stem.split("_")
        if len(parts) >= 3:
            genre = parts[1]
            idx = parts[2]
            candidates.add(f"{genre}.{idx}")
    return candidates


def main() -> None:
    args = parse_args()
    excluded_ids = load_excluded_track_ids(args.exclude_invalid_json)
    annotation_dir = args.annotation_dir
    if not annotation_dir.exists():
        raise FileNotFoundError(f"annotation 폴더가 없습니다: {annotation_dir}")

    bpm_files = sorted(annotation_dir.rglob("*.bpm"))
    if not bpm_files:
        raise FileNotFoundError(f".bpm 파일을 찾지 못했습니다: {annotation_dir}")

    sample_files = bpm_files[: args.sample_size]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for bpm_file in sample_files:
        candidate_ids = candidate_track_ids_from_bpm_file(bpm_file)
        if excluded_ids.intersection(candidate_ids):
            skip_id = sorted(excluded_ids.intersection(candidate_ids))[0]
            rows.append(
                {
                    "track_id": bpm_file.stem.split(".")[0],
                    "annotation_path": str(bpm_file),
                    "audio_path": "",
                    "gt_bpm": "",
                    "dsp_tempo_bpm": "",
                    "madmom_tempo_bpm": "",
                    "num_beats_madmom": "",
                    "dsp_error_pct": "",
                    "madmom_error_pct": "",
                    "status": "skipped_invalid_track",
                    "note": f"invalid track 목록에 포함되어 스킵: {skip_id}",
                }
            )
            continue

        track_id = bpm_file.stem.split(".")[0]
        gt_bpm = load_gt_bpm(bpm_file)
        audio_path = find_audio_for_track(track_id, args.audio_root)

        row = {
            "track_id": track_id,
            "annotation_path": str(bpm_file),
            "audio_path": str(audio_path) if audio_path else "",
            "gt_bpm": gt_bpm if gt_bpm is not None else "",
            "dsp_tempo_bpm": "",
            "madmom_tempo_bpm": "",
            "num_beats_madmom": "",
            "dsp_error_pct": "",
            "madmom_error_pct": "",
            "status": "ok",
            "note": "",
        }

        if audio_path is None:
            row["status"] = "missing_audio"
            row["note"] = "audio_root에서 track_id와 매칭되는 오디오를 찾지 못함"
            rows.append(row)
            continue

        try:
            dsp_tempo = estimate_dsp_tempo(audio_path, args.sample_rate)
        except Exception as exc:
            dsp_tempo = None
            row["note"] = f"dsp_error:{type(exc).__name__}"

        n_beats, madmom_tempo, madmom_status = estimate_madmom_beats_and_tempo(audio_path)
        if madmom_status != "ok":
            row["status"] = madmom_status

        row["dsp_tempo_bpm"] = "" if dsp_tempo is None else round(float(dsp_tempo), 4)
        row["madmom_tempo_bpm"] = "" if madmom_tempo is None else round(float(madmom_tempo), 4)
        row["num_beats_madmom"] = "" if n_beats is None else n_beats

        dsp_err = pct_error(dsp_tempo, gt_bpm)
        mm_err = pct_error(madmom_tempo, gt_bpm)
        row["dsp_error_pct"] = "" if dsp_err is None else round(dsp_err, 4)
        row["madmom_error_pct"] = "" if mm_err is None else round(mm_err, 4)

        rows.append(row)

    fieldnames = [
        "track_id",
        "annotation_path",
        "audio_path",
        "gt_bpm",
        "dsp_tempo_bpm",
        "madmom_tempo_bpm",
        "num_beats_madmom",
        "dsp_error_pct",
        "madmom_error_pct",
        "status",
        "note",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_count = sum(1 for r in rows if r["status"] == "ok")
    print(f"[quick-check] wrote: {args.output_csv}")
    print(f"[quick-check] processed: {len(rows)}, ok: {ok_count}, non-ok: {len(rows) - ok_count}")


if __name__ == "__main__":
    main()
