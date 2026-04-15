## Benchmarking Beat Tracking and Tempo Estimation

This repository contains the CSC 475 final project benchmarking four MIR methods for:

- **Tempo estimation** on **GTZAN + GiantSteps**
- **Beat tracking** on **GTZAN** (GiantSteps has no public beat annotations in this setup)

Compared methods:

1. `librosa` baseline (`librosa.beat.beat_track`)
2. Autocorrelation pipeline (`OSS + generalized ACF + pulse scoring`)
3. `madmom` (`RNN + DBN`)
4. `1D-StateSpace` (`jump-reward-inference` / BeatNet-based backend)

## Repository Layout

- `scripts/`: dataset download, evaluation, plotting, and diagnostics
- `results/`: per-track CSV outputs and aggregate JSON summaries
- `requirements-dsp-madmom.txt`: dependencies for DSP/madmom/autocorr environment
- `requirements-1dss.txt`: dependencies for 1DSS environment

## Environment Setup

This project is designed with two separate Python environments to avoid dependency conflicts.

### 1) DSP + madmom + autocorr environment

```bash
python -m venv .venv-dsp-madmom
.venv-dsp-madmom\Scripts\activate
pip install -r requirements-dsp-madmom.txt
```

### 2) 1DSS environment

```bash
python -m venv .venv-1dss
.venv-1dss\Scripts\activate
pip install -r requirements-1dss.txt
```

## Dataset Preparation

### GTZAN via mirdata (supports HF mirror for full audio)

```bash
python scripts/download_gtzan_genre_mirdata.py --version default --source mirdata
```

If the default mirror fails:

```bash
python scripts/download_gtzan_genre_mirdata.py --version default --source huggingface
```

This script also validates data and writes invalid track IDs to:

- `results/invalid_tracks_gtzan_genre.json`

### GiantSteps audio

Place the GiantSteps dataset root at:

- `dataset/giantsteps-tempo-dataset-master`

Then download/validate audio:

```bash
python scripts/download_giantsteps_audio.py --giantsteps-root dataset/giantsteps-tempo-dataset-master
```

## Main Evaluation Scripts

### Tempo evaluation (`scripts/run_tempo_eval.py`)

Generates per-method CSV and summary JSON:

- `results/tempo_dsp.csv`, `results/tempo_summary_dsp.json`
- `results/tempo_madmom.csv`, `results/tempo_summary_madmom.json`
- `results/tempo_autocorr.csv`, `results/tempo_summary_autocorr.json`
- `results/tempo_1dss.csv`, `results/tempo_summary_1dss.json`

Examples:

```bash
python scripts/run_tempo_eval.py --dataset both --methods dsp,madmom,autocorr
python scripts/run_tempo_eval.py --dataset both --methods 1dss
```

### Beat evaluation (`scripts/run_beat_eval.py`)

GTZAN-only beat evaluation (F-measure + Cemgil), outputs:

- `results/beat_dsp.csv`
- `results/beat_madmom.csv`
- `results/beat_1dss.csv`
- optional summaries: `results/beat_summary_<method>.json`

Examples:

```bash
python scripts/run_beat_eval.py --methods dsp,madmom --write-summary-json
python scripts/run_beat_eval.py --methods 1dss --write-summary-json
```

## Convenience Runners

- `scripts/run_all_dsp_madmom_evals.py`
  - Runs tempo for `dsp`, `madmom`, `autocorr`, then beat for `madmom`
- `scripts/run_all_1dss_evals.py`
  - Runs tempo then beat for `1dss`

Examples:

```bash
python scripts/run_all_dsp_madmom_evals.py --dataset both
python scripts/run_all_1dss_evals.py --dataset both --write-summary-json
```

## Metrics

### Tempo

- `ACC1`: strict relative tempo accuracy (default threshold 4%)
- `ACC2`: octave-tolerant tempo accuracy (checks `pred`, `pred/2`, `pred*2`)
- `MAE (BPM)`: absolute tempo error in BPM

### Beat

- `mir_eval.beat.f_measure` (default 70 ms tolerance)
- `mir_eval.beat.cemgil` (default `sigma=0.04s`)

## Typical Reproduction Workflow

1. Create and activate one of the environments
2. Download/validate GTZAN and GiantSteps
3. Run tempo/beat evaluations for that environment's methods
4. Review generated CSV and JSON outputs under `results/`

## Notes

- In this codebase, the autocorr implementation follows the OSS/ACF/pulse pipeline but does **not** include the SVM octave-correction stage.
- `1DSS` is integrated through `jump-reward-inference`; runtime is efficient, but accuracy is sensitive to preprocessing and backend alignment.
- Some generated files under `results/` and `figures/` may be large or machine-specific and can be regenerated using the scripts above.

## References

[1] F. Gouyon, A. Klapuri, S. Dixon, M. Alonso, G. Tzanetakis, C. Uhle, and P. Cano, "An experimental comparison of audio tempo induction algorithms," IEEE Transactions on Audio, Speech, and Language Processing, vol. 14, no. 5, pp. 1832-1844, 2006.

[2] S. Bock and M. Schedl, "Enhanced beat tracking with context-aware neural networks," in Proc. Int. Conf. Digital Audio Effects (DAFx), 2011.

[3] M. Heydari, M. McCallum, A. Ehmann, and Z. Duan, "A Novel 1D State Space for Efficient Music Rhythmic Analysis," in Proc. IEEE Int. Conf. Acoust. Speech Signal Process. (ICASSP), 2022.

[4] G. Percival and G. Tzanetakis, "Streamlined Tempo Estimation Based on Autocorrelation and Cross-correlation With Pulses," IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 22, no. 12, pp. 1765-1776, Dec. 2014.

[5] M. J. Hydri, "1D-StateSpace repository." [Online]. Available: https://github.com/mjhydri/1D-StateSpace

[6] S. Bock, F. Krebs, and G. Widmer, "madmom: a new Python audio and music signal processing library," in Proc. ACM Multimedia, 2016.
