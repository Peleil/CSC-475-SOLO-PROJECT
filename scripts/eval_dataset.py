"""
GiantSteps / GTZAN-Genre 공통 데이터 로딩 (mirdata 불필요).

GTZAN 레이아웃 (data_home = 보통 dataset/mirdata_gtzan_genre):
  - 오디오(기본): data_home/gtzan_genre/genres/{장르}/{track_id}.wav|.au|…
    예: mirdata_gtzan_genre/gtzan_genre/genres/blues/blues.00042.wav
    보조: gtzan_genre/{장르}/{track_id} (genres 없는 배치)도 탐색.
  - 템포·비트 라벨: gtzan_tempo_beat-main/tempo/*.bpm, beats/*.beats (iter_gtzan_tasks 가 여기서만
    BPM·beats 경로를 잡음. 오디오는 gtzan_tempo_beat-main 을 쓰지 않음.)
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

# eval_common 와 동일 (순환/무거운 import 방지). GTZAN 원본(Marsyas)은 .au 가 흔함.
AUDIO_EXTENSIONS_DEFAULT = [
    ".wav",
    ".au",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aif",
    ".aiff",
]

__all__ = [
    "load_gt_bpm_file",
    "load_excluded_track_ids",
    "load_gtzan_beat_times",
    "find_audio_giantsteps",
    "find_gtzan_audio",
    "iter_giantsteps_tasks",
    "iter_gtzan_tasks",
    "first_giantsteps_task_with_audio",
    "first_gtzan_task_with_audio",
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
    """mirdata 오디오 루트(gtzan_genre) 아래에서 track_id 에 맞는 파일을 찾는다."""
    if "." not in track_id:
        return None
    genre, _num = track_id.split(".", 1)
    # track_id 가 classical.00071 처럼 점이 두 개 이상이면 Path(.../classical.00071).with_suffix(".wav")
    # 가 classical.wav 로 깨지므로, 반드시 f"{track_id}{ext}" 로 이어 붙인다.
    dirs = (
        data_home / "gtzan_genre" / "genres" / genre,
        data_home / "gtzan_genre" / genre,
    )
    for d in dirs:
        for ext in AUDIO_EXTENSIONS_DEFAULT:
            p = d / f"{track_id}{ext}"
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


def first_giantsteps_task_with_audio(
    annotation_dir: Path,
    audio_root: Path,
    excluded: set[str],
) -> Optional[tuple[str, Path, Path]]:
    """
    `iter_giantsteps_tasks` 와 동일 규칙이나 **셔플 없이** 정렬된 *.bpm 순으로,
    오디오 파일이 실제로 있는 **첫** 트랙.
    반환: (stem, bpm_path, audio_path)
    """
    for bpm_file in sorted(annotation_dir.rglob("*.bpm")):
        stem = bpm_file.stem
        cands = candidate_track_ids_from_bpm_stem(stem)
        if excluded.intersection(cands):
            continue
        audio = find_audio_giantsteps(stem, audio_root)
        if audio is not None and audio.is_file():
            return stem, bpm_file, audio
    return None


def first_gtzan_task_with_audio(
    data_home: Path,
    excluded: set[str],
) -> Optional[tuple[str, Path, Path, Optional[Path]]]:
    """
    `iter_gtzan_tasks` 와 동일 규칙, 셔플 없이 tempo/*.bpm 정렬 순의 첫 오디오.
    반환: (track_id, tempo_bpm_path, audio_path, beats_path or None)
    """
    tempo_dir = data_home / "gtzan_tempo_beat-main" / "tempo"
    beats_dir = data_home / "gtzan_tempo_beat-main" / "beats"
    if not tempo_dir.is_dir():
        return None
    for bpm_path in sorted(tempo_dir.glob("*.bpm")):
        stem = bpm_path.stem
        tid = tempo_bpm_stem_to_track_id(stem)
        if tid in excluded:
            continue
        audio = find_gtzan_audio(data_home, tid)
        if audio is not None and audio.is_file():
            ann_stem = gtzan_track_id_to_stem(tid)
            bp = beats_dir / f"{ann_stem}.beats"
            beats_path = bp if bp.is_file() else None
            return tid, bpm_path, audio, beats_path
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
    tempo/bpm·beats 는 gtzan_tempo_beat-main 만 참조하고,
    audio_path 는 find_gtzan_audio(data_home) → gtzan_genre/ 아래에서만 찾는다.
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
