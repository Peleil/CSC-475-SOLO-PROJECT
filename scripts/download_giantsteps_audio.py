#!/usr/bin/env python3
"""
Download GiantSteps audio (mp3) with the same URL/MD5 flow as audio_dl.sh.

1) https://www.cp.jku.at/datasets/giantsteps/backup/<file>.mp3
2) On failure or MD5 mismatch: http://geo-samples.beatport.com/lofi/<file>.mp3

Works from any working directory if --giantsteps-root is set correctly.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PRIMARY_BASE = "https://www.cp.jku.at/datasets/giantsteps/backup/"
BACKUP_BASE = "http://geo-samples.beatport.com/lofi/"
UA = "Mozilla/5.0 (compatible; CSC475-giantsteps-dl/1.0)"


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_to(url: str, dest: Path, timeout: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        with dest.open("wb") as out:
            while True:
                chunk = resp.read(1024 * 512)
                if not chunk:
                    break
                out.write(chunk)


def load_expected_md5(md5_path: Path) -> str:
    return md5_path.read_text(encoding="utf-8").strip().split()[0].lower()


def process_one(
    md5_path: Path,
    audio_dir: Path,
    timeout: int,
    force: bool,
) -> tuple[str, str]:
    """
    Returns (mp3_name, status) where status in ok_skip, ok_primary, ok_backup, error
    """
    mp3_name = md5_path.stem + ".mp3"
    expected = load_expected_md5(md5_path)
    dest = audio_dir / mp3_name

    if dest.is_file() and not force:
        try:
            if md5_file(dest).lower() == expected:
                return mp3_name, "ok_skip"
        except OSError:
            pass

    last_err: str | None = None
    for label, base in (("primary", PRIMARY_BASE), ("backup", BACKUP_BASE)):
        url = base + mp3_name
        try:
            if dest.is_file():
                dest.unlink()
            download_to(url, dest, timeout=timeout)
            got = md5_file(dest).lower()
            if got == expected:
                return mp3_name, "ok_primary" if label == "primary" else "ok_backup"
            last_err = f"{label}_md5_mismatch"
            if dest.is_file():
                dest.unlink()
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_err = f"{label}:{type(exc).__name__}"
            if dest.is_file():
                try:
                    dest.unlink()
                except OSError:
                    pass
            continue

    return mp3_name, f"error:{last_err}"


def main() -> int:
    p = argparse.ArgumentParser(description="Download GiantSteps mp3 audio (mirrors audio_dl.sh)")
    p.add_argument(
        "--giantsteps-root",
        type=Path,
        default=Path("dataset/giantsteps-tempo-dataset-master"),
        help="Path to giantsteps-tempo-dataset-master",
    )
    p.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds")
    p.add_argument("--workers", type=int, default=4, help="Number of parallel download workers")
    p.add_argument("--force", action="store_true", help="Redownload and revalidate MD5 even if file exists")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Test mode: process first N .md5 files only (0 = all)",
    )
    args = p.parse_args()

    root = args.giantsteps_root.resolve()
    md5_dir = root / "md5"
    audio_dir = root / "audio"
    if not md5_dir.is_dir():
        print(f"[error] md5 folder not found: {md5_dir}", file=sys.stderr)
        return 1

    md5_files = sorted(md5_dir.glob("*.md5"))
    if not md5_files:
        print(f"[error] no .md5 files found: {md5_dir}", file=sys.stderr)
        return 1
    if args.limit and args.limit > 0:
        md5_files = md5_files[: args.limit]

    audio_dir.mkdir(parents=True, exist_ok=True)
    print(f"[giantsteps] root={root}", flush=True)
    print(f"[giantsteps] files={len(md5_files)} workers={args.workers}", flush=True)

    stats = {"ok_skip": 0, "ok_primary": 0, "ok_backup": 0, "error": 0}
    errors: list[str] = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {
            ex.submit(process_one, mp, audio_dir, args.timeout, args.force): mp
            for mp in md5_files
        }
        done = 0
        for fut in as_completed(futs):
            done += 1
            name, status = fut.result()
            if status == "ok_skip":
                stats["ok_skip"] += 1
            elif status == "ok_primary":
                stats["ok_primary"] += 1
            elif status == "ok_backup":
                stats["ok_backup"] += 1
            else:
                stats["error"] += 1
                errors.append(f"{name} {status}")
            if done == 1 or done % 10 == 0 or done == len(md5_files):
                elapsed = time.perf_counter() - t0
                print(
                    f"[giantsteps] {done}/{len(md5_files)} "
                    f"skip={stats['ok_skip']} pri={stats['ok_primary']} "
                    f"bak={stats['ok_backup']} err={stats['error']} "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )

    elapsed = time.perf_counter() - t0
    print(f"[giantsteps] done {elapsed:.1f}s : {stats}", flush=True)
    if errors:
        log = root / "audio_download_errors.txt"
        log.write_text("\n".join(errors[:200]) + ("\n..." if len(errors) > 200 else ""), encoding="utf-8")
        print(f"[giantsteps] error list (up to 200): {log}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
