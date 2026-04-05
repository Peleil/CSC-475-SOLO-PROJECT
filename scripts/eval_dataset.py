"""
GiantSteps / GTZAN-Genre 공통 데이터 로딩 (mirdata 불필요).
GTZAN은 mirdata data_home 폴더 레이아웃을 따른다.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

# eval_common 와 동일 (순환/무거운 import 방지)
AUDIO_EXTENSIONS_DEFAULT = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aif", ".aiff"]

__all__ = [
    "load_gt_bpm_file",
    "load_excluded_track_ids",
    "load_gtzan_beat_times",
    "find_audio_giantsteps",
    "find_gtzan_audio",
    "iter_giantsteps_tasks",
    "iter_gtzan_tasks",
    "tempo_bpm_stem_to_track_id",
    "gtzan_track_id_to_stem",
]


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


def tempo_bpm_stem_to_track_id(stem: str) -> str:
    """gtzan_classical_00000 -> classical.00000"""
    if not stem.startswith("gtzan_"):
        return stem
    rest = stem[len("gtzan_") :]
    i = rest.rfind("_")
    if i <= 0:
        return stem
    return f"{rest[:i]}.{rest[i + 1 :]}"


def gtzan_track_id_to_stem(tid: str) -> str:
    """classical.00000 -> gtzan_classical_00000"""
    if "." not in tid:
        return tid
    genre, idx = tid.split(".", 1)
    return f"gtzan_{genre}_{idx}"


def find_gtzan_audio(data_home: Path, track_id: str) -> Optional[Path]:
    genre, _num = track_id.split(".", 1)
    base = data_home / "gtzan_genre" / "genres" / genre / track_id
    for ext in AUDIO_EXTENSIONS_DEFAULT:
        p = base.with_suffix(ext)
        if p.is_file():
            return p
    return None


def load_gtzan_beat_times(beats_path: Path) -> Optional[np.ndarray]:
    """GTZAN .beats 파일: 첫 열 = 시간(초)."""
    try:
        data = np.loadtxt(beats_path, ndmin=2)
        if data.size == 0:
            return None
        t = np.asarray(data[:, 0], dtype=float).ravel()
        t = t[np.isfinite(t)]
        return t if t.size > 0 else None
    except Exception:
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


def iter_gtzan_tasks(
    data_home: Path,
    excluded: set[str],
    limit: Optional[int],
    seed: int,
) -> Iterator[tuple[str, Path, Optional[Path], Optional[Path]]]:
    """
    (track_id, tempo_bpm_path, audio_path, beats_path or None)
    beats_path는 파일이 있으면 설정.
    """
    tempo_dir = data_home / "gtzan_tempo_beat-main" / "tempo"
    beats_dir = data_home / "gtzan_tempo_beat-main" / "beats"
    if not tempo_dir.is_dir():
        raise FileNotFoundError(f"GTZAN tempo 디렉터리 없음: {tempo_dir}")
    bpm_files = sorted(tempo_dir.glob("*.bpm"))
    rng = random.Random(seed)
    rng.shuffle(bpm_files)
    n = 0
    for bpm_path in bpm_files:
        stem = bpm_path.stem
        tid = tempo_bpm_stem_to_track_id(stem)
        if tid in excluded:
            continue
        audio = find_gtzan_audio(data_home, tid)
        ann_stem = gtzan_track_id_to_stem(tid)
        bp = beats_dir / f"{ann_stem}.beats"
        beats_path = bp if bp.is_file() else None
        yield tid, bpm_path, audio, beats_path
        n += 1
        if limit is not None and n >= limit:
            break
