#!/usr/bin/env python3
"""
GTZAN-Genre 데이터셋을 mirdata로 내려받는다.

- default + --source mirdata: mirdata 기본 URL (UVic opihi) — 방화벽/서버 다운 시 타임아웃 날 수 있음.
- default + --source huggingface: Hugging Face `marsyas/gtzan`의 `data/genres.tar.gz`로 오디오 확보 후,
  mirdata로 tempo/beat 어노테이션만 추가 다운로드.
- mini / test: mirdata 경로만 사용 (--source huggingface 는 default 전용).

사용 예:
  python scripts/download_gtzan_genre_mirdata.py --version default --source huggingface
  python scripts/download_gtzan_genre_mirdata.py --version mini
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path
from urllib.error import URLError

from mirdata.datasets.gtzan_genre import Dataset

# mirdata REMOTES["all"] 와 동일 (genres.tar.gz 무결성)
GTZAN_GENRES_MD5 = "5b3d6dddb579ab49814ab86dba69e7c7"
HF_REPO = "marsyas/gtzan"
HF_GENRES_RELPATH = "data/genres.tar.gz"


def _has_validation_errors(missing_files: dict, invalid_checksums: dict) -> bool:
    for container in (missing_files, invalid_checksums):
        for section in container.values():
            if isinstance(section, dict) and len(section) > 0:
                return True
    return False


def _extract_invalid_track_ids(invalid_checksums: dict) -> list[str]:
    tracks = invalid_checksums.get("tracks", {})
    if isinstance(tracks, dict):
        return sorted(tracks.keys())
    return []


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_and_extract_genres_from_huggingface(
    data_home: Path,
    *,
    force: bool,
    skip_md5: bool,
) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub 가 필요합니다. conda/pip: pip install huggingface_hub"
        ) from exc

    dest_dir = data_home / "gtzan_genre"
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / "genres.tar.gz"
    genres_dir = dest_dir / "genres"

    need_tar = force or not tar_path.is_file()
    if not need_tar and not skip_md5:
        got = _md5_file(tar_path)
        if got != GTZAN_GENRES_MD5:
            print(
                f"[hf] 기존 genres.tar.gz MD5 불일치 ({got} != {GTZAN_GENRES_MD5}). "
                "다시 받으려면 --force 또는 --skip-md5-check 로 검증 생략.",
                file=sys.stderr,
            )
            raise SystemExit(1)

    if need_tar:
        print(f"[hf] 다운로드: {HF_REPO} / {HF_GENRES_RELPATH} (용량 ~1.2GB, 시간이 걸릴 수 있음)")
        cached = hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_GENRES_RELPATH,
            repo_type="dataset",
        )
        shutil.copy2(cached, tar_path)

    if not skip_md5:
        got = _md5_file(tar_path)
        if got != GTZAN_GENRES_MD5:
            print(
                f"[hf] 경고: MD5가 mirdata 기대값과 다름 ({got} vs {GTZAN_GENRES_MD5}). "
                "다른 미러 파일일 수 있습니다. 계속 압축 해제합니다.",
                file=sys.stderr,
            )

    need_extract = force or not genres_dir.is_dir() or not any(genres_dir.iterdir())
    if need_extract:
        print(f"[hf] 압축 해제: {tar_path} -> {dest_dir}")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(path=str(dest_dir))


def main() -> int:
    p = argparse.ArgumentParser(description="Download gtzan_genre via mirdata (or HF mirror for audio)")
    p.add_argument(
        "--data-home",
        type=Path,
        default=Path("dataset/mirdata_gtzan_genre"),
        help="mirdata가 데이터를 풀어둘 디렉터리",
    )
    p.add_argument(
        "--version",
        choices=("default", "mini", "test"),
        default="default",
        help="default=전체; mini=소규모; test=샘플 인덱스(개발용)",
    )
    p.add_argument(
        "--source",
        choices=("mirdata", "huggingface"),
        default="mirdata",
        help="default 전체 오디오: mirdata(UVic URL) 또는 huggingface(HF에 호스팅된 genres.tar.gz)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="이미 있어도 덮어쓰기",
    )
    p.add_argument(
        "--skip-md5-check",
        action="store_true",
        help="genres.tar.gz MD5 검증 생략 (HF 파일이 기대값과 다를 때만 임시로)",
    )
    p.add_argument(
        "--strict-validate",
        action="store_true",
        help="검증 경고(누락/체크섬 불일치)가 있으면 실패(exit 1). 기본은 경고만 출력 후 계속 진행.",
    )
    p.add_argument(
        "--invalid-list-path",
        type=Path,
        default=Path("results/invalid_tracks_gtzan_genre.json"),
        help="검증에서 불일치로 나온 track id 목록 저장 경로",
    )
    args = p.parse_args()

    args.data_home.mkdir(parents=True, exist_ok=True)
    root = args.data_home.resolve()

    if args.source == "huggingface" and args.version != "default":
        print(
            "[error] --source huggingface 는 --version default 전체 GTZAN과 함께만 사용합니다. "
            "mini는 mirdata mini URL을 쓰세요.",
            file=sys.stderr,
        )
        return 2

    print(f"[mirdata] dataset=gtzan_genre version={args.version} source={args.source}")
    print(f"[mirdata] data_home={root}")

    if args.source == "huggingface":
        _download_and_extract_genres_from_huggingface(
            args.data_home,
            force=args.force,
            skip_md5=args.skip_md5_check,
        )
        ds = Dataset(data_home=str(root), version=args.version)
        print("[mirdata] tempo/beat 어노테이션 + 인덱스만 mirdata로 다운로드")
        ds.download(
            partial_download=["tempo_beat_annotations"],
            force_overwrite=args.force,
        )
    else:
        ds = Dataset(data_home=str(root), version=args.version)
        try:
            ds.download(force_overwrite=args.force)
        except (OSError, URLError, TimeoutError) as exc:
            err = str(exc)
            if "10060" in err or "timed out" in err.lower() or "failed to respond" in err.lower():
                print(
                    "\n[hint] UVic 서버(opihi) 연결이 실패했습니다. Hugging Face 미러로 받으세요:\n"
                    "  python scripts/download_gtzan_genre_mirdata.py "
                    "--version default --source huggingface --data-home "
                    f'"{args.data_home}"\n',
                    file=sys.stderr,
                )
            raise

    print("[mirdata] download 단계 완료.")

    ds = Dataset(data_home=str(root), version=args.version)
    missing_files, invalid_checksums = ds.validate(verbose=True)
    invalid_track_ids = _extract_invalid_track_ids(invalid_checksums)
    args.invalid_list_path.parent.mkdir(parents=True, exist_ok=True)
    args.invalid_list_path.write_text(
        json.dumps(
            {
                "dataset": "gtzan_genre",
                "version": args.version,
                "invalid_track_ids": invalid_track_ids,
                "missing": missing_files,
                "invalid": invalid_checksums,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    has_errors = _has_validation_errors(missing_files, invalid_checksums)
    if has_errors:
        print("[mirdata] validate: 일부 파일 누락 또는 체크섬 불일치", file=sys.stderr)
        print(f"  missing: {missing_files}", file=sys.stderr)
        print(f"  invalid: {invalid_checksums}", file=sys.stderr)
        print(
            f"[mirdata] invalid track 목록 저장: {args.invalid_list_path}",
            file=sys.stderr,
        )
        if args.strict_validate:
            return 1
        print(
            "[mirdata] strict 모드가 아니므로 계속 진행합니다. "
            "평가/학습 시 invalid track을 제외하세요.",
            file=sys.stderr,
        )
    else:
        print(f"[mirdata] invalid track 목록 저장: {args.invalid_list_path}")

    print("[mirdata] validate: OK")
    try:
        n = len(ds.track_ids)
        print(f"[mirdata] track_ids: {n}개")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
