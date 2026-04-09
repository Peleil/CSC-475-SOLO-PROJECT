#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_common import (
    DEFAULT_UNIFIED_SAMPLE_RATE,
    _write_mono_wav_temp,
    estimate_1dss_beats_and_tempo,
    estimate_dsp_beats,
    estimate_madmom_beats,
    load_mono_resampled,
)
from eval_dataset import (
    first_giantsteps_task_with_audio,
    first_gtzan_task_with_audio,
    load_excluded_track_ids,
)


def _prepare_input_path(audio: Path, sample_rate: int, max_sec: float | None) -> tuple[Path, str | None, np.ndarray]:
    y = load_mono_resampled(audio, sample_rate)
    if max_sec is not None and max_sec > 0:
        n = int(max_sec * sample_rate)
        y = y[:n]
        tmp = _write_mono_wav_temp(y, sample_rate)
        return Path(tmp), tmp, y
    return audio, None, y


def collect_beat_series(work_path: Path, sample_rate: int, duration: float, autocorr_grid: bool) -> list[tuple[str, np.ndarray, str]]:
    series: list[tuple[str, np.ndarray, str]] = []

    b, _t, st = estimate_dsp_beats(work_path, sample_rate)
    if st == "ok" and b.size > 0:
        series.append(("librosa DSP", b, "C0"))
    else:
        label = "librosa DSP (no beats)" if st == "ok" else f"librosa DSP ({st})"
        series.append((label, np.array([]), "C0"))

    mb, mst = estimate_madmom_beats(work_path, sample_rate)
    if mst == "ok" and mb.size > 0:
        series.append(("madmom", mb, "C1"))
    else:
        series.append((f"madmom ({mst})", np.array([]), "C1"))

    sb, _stempo, sst = estimate_1dss_beats_and_tempo(work_path, sample_rate)
    if sst == "ok" and sb.size > 0:
        series.append(("1DSS", np.sort(np.unique(np.asarray(sb, dtype=float).ravel())), "C2"))
    else:
        series.append((f"1DSS ({sst})", np.array([]), "C2"))

    if autocorr_grid:
        try:
            from eval_common import estimate_autocorr_tempo

            bpm = estimate_autocorr_tempo(work_path, sample_rate)
            if bpm is not None and bpm > 0:
                period = 60.0 / float(bpm)
                grid = np.arange(0.0, duration + 1e-9, period, dtype=float)
                series.append((f"autocorr grid (BPM≈{bpm:.1f})", grid, "C3"))
            else:
                series.append(("autocorr (BPM 없음)", np.array([]), "C3"))
        except Exception as exc:
            series.append((f"autocorr ({exc})", np.array([]), "C3"))

    return series


def save_figure(y: np.ndarray, sample_rate: int, series: list[tuple[str, np.ndarray, str]], output: Path, title: str, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    duration = float(len(y)) / float(sample_rate)
    n = len(series)
    fig_h = 1.2 + 0.55 * n + 1.2
    fig, axes = plt.subplots(
        n + 1,
        1,
        sharex=True,
        figsize=(11, fig_h),
        gridspec_kw={"height_ratios": [1.2] + [0.55] * n},
    )
    t_ax = np.linspace(0.0, duration, num=min(len(y), 8000), endpoint=False)
    idx = (np.linspace(0, len(y) - 1, num=len(t_ax))).astype(int)
    axes[0].plot(t_ax, y[idx], color="0.35", linewidth=0.5)
    axes[0].set_ylabel("waveform")
    axes[0].set_yticks([])

    for i, (name, times, color) in enumerate(series):
        ax = axes[i + 1]
        if times.size > 0:
            ax.eventplot([times], lineoffsets=[0], linelengths=[0.9], colors=[color], linewidths=[1.2])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
        ax.set_yticks([])
        ax.set_xlim(0.0, duration)
        ax.grid(True, axis="x", alpha=0.25)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def run_one_track(audio_path: Path, output: Path, sample_rate: int, max_seconds: float | None, autocorr_grid: bool, dpi: int, title: str | None = None) -> None:
    if not audio_path.is_file():
        raise SystemExit(f"오디오 없음: {audio_path}")

    work_path, tmp_path, y = _prepare_input_path(audio_path, sample_rate, max_seconds)
    duration = float(len(y)) / float(sample_rate)
    series = collect_beat_series(work_path, sample_rate, duration, autocorr_grid)
    ttl = title or f"Beat times — {audio_path.name}  (sr={sample_rate} Hz)"
    save_figure(y, sample_rate, series, output, ttl, dpi)

    if tmp_path is not None:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    print(f"[compare-beats] wrote {output}")
    for name, times, _ in series:
        print(f"  {name}: n={int(times.size)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare beat times: single --audio or --from-datasets-first")
    p.add_argument("--audio", type=Path, default=None)
    p.add_argument("--from-datasets-first", action="store_true")
    p.add_argument("--output", type=Path, default=Path("figures/beats_compare.png"))
    p.add_argument("--output-giantsteps", type=Path, default=Path("figures/beats_first_giantsteps.png"))
    p.add_argument("--output-gtzan", type=Path, default=Path("figures/beats_first_gtzan_genre.png"))
    p.add_argument("--giantsteps-root", type=Path, default=Path("dataset/giantsteps-tempo-dataset-master"))
    p.add_argument("--data-home", type=Path, default=Path("dataset/mirdata_gtzan_genre"))
    p.add_argument("--exclude-invalid-json", type=Path, default=Path("results/invalid_tracks_gtzan_genre.json"))
    p.add_argument("--sample-rate", type=int, default=DEFAULT_UNIFIED_SAMPLE_RATE)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--autocorr-grid", action="store_true")
    p.add_argument("--dpi", type=int, default=120)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise SystemExit("matplotlib 이 필요합니다. 예: pip install matplotlib") from None

    if args.from_datasets_first:
        excluded = load_excluded_track_ids(args.exclude_invalid_json)
        ann = args.giantsteps_root / "annotations_v2" / "tempo"
        audio_root = args.giantsteps_root / "audio"
        gs = first_giantsteps_task_with_audio(ann, audio_root, excluded)
        if gs is not None:
            stem, _bpm, ap = gs
            run_one_track(
                ap,
                args.output_giantsteps,
                args.sample_rate,
                args.max_seconds,
                args.autocorr_grid,
                args.dpi,
                title=f"GiantSteps (first sorted) — {stem} — {ap.name}  (sr={args.sample_rate} Hz)",
            )
        gz = first_gtzan_task_with_audio(args.data_home, excluded)
        if gz is not None:
            tid, _bpm, ap, _beats = gz
            run_one_track(
                ap,
                args.output_gtzan,
                args.sample_rate,
                args.max_seconds,
                args.autocorr_grid,
                args.dpi,
                title=f"GTZAN-Genre (first sorted) — {tid} — {ap.name}  (sr={args.sample_rate} Hz)",
            )
        return

    if args.audio is None:
        raise SystemExit("--audio 가 필요합니다 (--from-datasets-first 가 아니면).")
    run_one_track(args.audio, args.output, args.sample_rate, args.max_seconds, args.autocorr_grid, args.dpi)


if __name__ == "__main__":
    main()
