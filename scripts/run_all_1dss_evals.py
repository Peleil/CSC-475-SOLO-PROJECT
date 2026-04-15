#!/usr/bin/env python3
"""
Run 1DSS evaluations sequentially in one conda env.

Steps:
  1) tempo 1dss
  2) beat 1dss
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 1DSS evals in order")
    p.add_argument(
        "--dataset",
        choices=("giantsteps", "gtzan_genre", "both"),
        default="both",
        help="Dataset option passed to run_tempo_eval.py",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Optional: passed to run_tempo_eval and run_beat_eval (mono resample Hz).",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional: passed to both scripts (omit for full corpora).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Optional: passed to both scripts.",
    )
    p.add_argument(
        "--write-summary-json",
        action="store_true",
        help="Pass to run_beat_eval.py (beat_summary_<method>.json).",
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _run(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print("[run-all-1dss] " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    tempo = root / "scripts" / "run_tempo_eval.py"
    beat = root / "scripts" / "run_beat_eval.py"

    tempo_prefix = [sys.executable, str(tempo)]
    beat_prefix = [sys.executable, str(beat)]
    extra_tempo: list[str] = []
    extra_beat: list[str] = []
    if args.sample_rate is not None:
        extra_tempo.extend(["--sample-rate", str(args.sample_rate)])
        extra_beat.extend(["--sample-rate", str(args.sample_rate)])
    if args.sample_size is not None:
        extra_tempo.extend(["--sample-size", str(args.sample_size)])
        extra_beat.extend(["--sample-size", str(args.sample_size)])
    if args.seed != 0:
        extra_tempo.extend(["--seed", str(args.seed)])
        extra_beat.extend(["--seed", str(args.seed)])
    if args.write_summary_json:
        extra_beat.append("--write-summary-json")
    jobs = [
        [*tempo_prefix, "--dataset", args.dataset, "--methods", "1dss", *extra_tempo],
        [*beat_prefix, "--methods", "1dss", *extra_beat],
    ]
    for i, cmd in enumerate(jobs, start=1):
        print(f"[run-all-1dss] step {i}/{len(jobs)}")
        _run(cmd, root, args.dry_run)
    print("[run-all-1dss] done")


if __name__ == "__main__":
    main()

