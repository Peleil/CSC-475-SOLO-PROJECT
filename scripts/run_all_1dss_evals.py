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
    jobs = [
        [*tempo_prefix, "--dataset", args.dataset, "--methods", "1dss"],
        [*beat_prefix, "--methods", "1dss"],
    ]
    for i, cmd in enumerate(jobs, start=1):
        print(f"[run-all-1dss] step {i}/{len(jobs)}")
        _run(cmd, root, args.dry_run)
    print("[run-all-1dss] done")


if __name__ == "__main__":
    main()

