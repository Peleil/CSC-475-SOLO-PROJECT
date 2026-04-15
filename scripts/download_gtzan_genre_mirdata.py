#!/usr/bin/env python3
"""
Download the GTZAN-Genre dataset via mirdata.

- default + --source mirdata: mirdata default URL (UVic opihi).
- default + --source huggingface: download `data/genres.tar.gz` from
  Hugging Face `marsyas/gtzan`, then download tempo/beat annotations via mirdata.
- mini / test: mirdata path only (--source huggingface is default-only).

Examples:
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
            "huggingface_hub is required. Install with: pip install huggingface_hub"
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
                f"[hf] Existing genres.tar.gz MD5 mismatch ({got} != {GTZAN_GENRES_MD5}). "
                "Use --force to redownload or --skip-md5-check to skip verification.",
                file=sys.stderr,
            )
            raise SystemExit(1)

    if need_tar:
        print(f"[hf] Downloading: {HF_REPO} / {HF_GENRES_RELPATH} (~1.2GB, may take time)")
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
                f"[hf] Warning: MD5 differs from mirdata expectation ({got} vs {GTZAN_GENRES_MD5}). "
                "Proceeding with extraction.",
                file=sys.stderr,
            )

    need_extract = force or not genres_dir.is_dir() or not any(genres_dir.iterdir())
    if need_extract:
        print(f"[hf] Extracting: {tar_path} -> {dest_dir}")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(path=str(dest_dir))


def main() -> int:
    p = argparse.ArgumentParser(description="Download gtzan_genre via mirdata (or HF mirror for audio)")
    p.add_argument(
        "--data-home",
        type=Path,
        default=Path("dataset/mirdata_gtzan_genre"),
        help="Directory where mirdata stores dataset files",
    )
    p.add_argument(
        "--version",
        choices=("default", "mini", "test"),
        default="default",
        help="default=full; mini=small subset; test=sample index",
    )
    p.add_argument(
        "--source",
        choices=("mirdata", "huggingface"),
        default="mirdata",
        help="Source for full audio in default mode: mirdata or huggingface",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    p.add_argument(
        "--skip-md5-check",
        action="store_true",
        help="Skip genres.tar.gz MD5 check",
    )
    p.add_argument(
        "--strict-validate",
        action="store_true",
        help="Fail with exit 1 on validation warnings (missing/checksum mismatch).",
    )
    p.add_argument(
        "--invalid-list-path",
        type=Path,
        default=Path("results/invalid_tracks_gtzan_genre.json"),
        help="Path to save track IDs flagged by validation",
    )
    args = p.parse_args()

    args.data_home.mkdir(parents=True, exist_ok=True)
    root = args.data_home.resolve()

    if args.source == "huggingface" and args.version != "default":
        print(
            "[error] --source huggingface is only supported with --version default. "
            "Use mirdata source for mini/test.",
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
        print("[mirdata] Downloading tempo/beat annotations and index via mirdata")
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
                    "\n[hint] Failed to reach UVic opihi server. Try Hugging Face mirror:\n"
                    "  python scripts/download_gtzan_genre_mirdata.py "
                    "--version default --source huggingface --data-home "
                    f'"{args.data_home}"\n',
                    file=sys.stderr,
                )
            raise

    print("[mirdata] download stage completed.")

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
        print("[mirdata] validate: some files are missing or have checksum mismatch", file=sys.stderr)
        print(f"  missing: {missing_files}", file=sys.stderr)
        print(f"  invalid: {invalid_checksums}", file=sys.stderr)
        print(
            f"[mirdata] invalid track list saved: {args.invalid_list_path}",
            file=sys.stderr,
        )
        if args.strict_validate:
            return 1
        print(
            "[mirdata] Continuing because strict mode is off. "
            "Exclude invalid tracks during evaluation/training.",
            file=sys.stderr,
        )
    else:
        print(f"[mirdata] invalid track list saved: {args.invalid_list_path}")

    print("[mirdata] validate: OK")
    try:
        n = len(ds.track_ids)
        print(f"[mirdata] track_ids: {n}")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
