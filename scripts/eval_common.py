"""
Shared tempo/beat estimators and metrics (DSP librosa, madmom DBN, optional
1D-StateSpace / jump-reward-inference), reused across evaluation scripts.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import librosa
import numpy as np

try:
    from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
    from madmom.features.tempo import TempoEstimationProcessor
except Exception:
    DBNBeatTrackingProcessor = None
    RNNBeatProcessor = None
    TempoEstimationProcessor = None

AUDIO_EXTENSIONS_DEFAULT = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aif", ".aiff"]


def _patch_numpy_typing_ndarray() -> None:
    """
    Compatibility shim for old NumPy (e.g., 1.20.x) + newer BeatNet imports.
    Some BeatNet stacks reference numpy.typing.NDArray, which may be missing.
    """
    try:
        import numpy.typing as npt

        if not hasattr(npt, "NDArray"):
            class _CompatNDArray:
                def __class_getitem__(cls, _item):
                    return np.ndarray

            npt.NDArray = _CompatNDArray
    except Exception:
        return

def _patch_scipy_signal_windows() -> None:
    """
    Patch legacy scipy signal window symbols expected by some librosa stacks.
    """
    try:
        import scipy.signal as s

        if not hasattr(s, "hann") and hasattr(s, "windows"):
            s.hann = s.windows.hann
    except Exception:
        return


def _audio_fspath(audio_path: Path) -> str:
    """Return an absolute path string for stable Windows path handling."""
    p = audio_path.expanduser()
    try:
        return str(p.resolve(strict=False))
    except TypeError:
        return str(p.resolve())


def _fmt_exc(exc: BaseException, max_len: int = 160) -> str:
    s = f"{type(exc).__name__}:{exc}"
    return s.replace(";", ",").replace("\n", " ")[:max_len]


MADMOM_SAMPLE_RATE = 44100
DEFAULT_UNIFIED_SAMPLE_RATE = MADMOM_SAMPLE_RATE


def load_mono_resampled(audio_path: Path, sample_rate: int) -> np.ndarray:
    """
    Load audio as mono float32 at a fixed sample rate for all evaluators.
    """
    _patch_scipy_signal_windows()
    y, _ = librosa.load(_audio_fspath(audio_path), sr=sample_rate, mono=True)
    return np.asarray(y, dtype=np.float32)


def _write_mono_wav_temp(y: np.ndarray, sample_rate: int) -> str:
    """Write a temporary float WAV for path-only APIs (caller deletes it)."""
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float32).ravel())
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        import soundfile as sf

        sf.write(path, y, sample_rate, subtype="FLOAT")
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path


def estimate_dsp_tempo(audio_path: Path, sample_rate: int) -> Optional[float]:
    """
    Normalize librosa beat_track tempo outputs across scalar/array variants.
    """
    y = load_mono_resampled(audio_path, sample_rate)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sample_rate)
    arr = np.asarray(tempo, dtype=float).ravel()
    if arr.size == 0 or not np.isfinite(arr[0]):
        return None
    v = float(arr[0])
    return v if v > 0 else None


def estimate_dsp_beats(
    audio_path: Path, sample_rate: int = DEFAULT_UNIFIED_SAMPLE_RATE
) -> tuple[np.ndarray, Optional[float], str]:
    """librosa beat_track to beat times (s) and optional tempo estimate."""
    try:
        y = load_mono_resampled(audio_path, sample_rate)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sample_rate)
        frames = np.asarray(beat_frames).ravel()
        if frames.size == 0:
            return np.array([]), None, "dsp_no_beats"
        beat_times = librosa.frames_to_time(frames, sr=sample_rate)
        bt = np.asarray(beat_times, dtype=float).ravel()
        arr = np.asarray(tempo, dtype=float).ravel()
        t0 = (
            float(arr[0])
            if arr.size and np.isfinite(arr[0]) and float(arr[0]) > 0
            else None
        )
        return bt, t0, "ok"
    except Exception as exc:
        return np.array([]), None, f"dsp_runtime_error:{_fmt_exc(exc)}"


def estimate_madmom_beats(
    audio_path: Path, sample_rate: int = DEFAULT_UNIFIED_SAMPLE_RATE
) -> tuple[np.ndarray, str]:
    """
    RNN + DBN beat times (seconds).

    Instead of path-only ffmpeg decoding, decode with librosa/soundfile and pass
    float32 audio directly for more stable cross-platform behavior.
    """
    if DBNBeatTrackingProcessor is None or RNNBeatProcessor is None:
        return np.array([]), "madmom_import_error"
    try:
        y = load_mono_resampled(audio_path, sample_rate)
        act = RNNBeatProcessor()(y)
        beats = DBNBeatTrackingProcessor(fps=100)(act)
    except Exception as exc:
        return np.array([]), f"madmom_runtime_error:{_fmt_exc(exc)}"
    return np.asarray(beats, dtype=float).ravel(), "ok"


def estimate_madmom_tempo_dbn(
    audio_path: Path, sample_rate: int = DEFAULT_UNIFIED_SAMPLE_RATE
) -> tuple[Optional[float], str]:
    """
    Global BPM estimate using madmom TempoEstimationProcessor(method='dbn').
    Returns (tempo_bpm or None, status).
    """
    if TempoEstimationProcessor is None or RNNBeatProcessor is None:
        return None, "madmom_import_error"
    try:
        y = load_mono_resampled(audio_path, sample_rate)
        act = RNNBeatProcessor()(y)
        tempi = TempoEstimationProcessor(method="dbn", fps=100)(act)
    except Exception as exc:
        return None, f"madmom_runtime_error:{_fmt_exc(exc)}"

    arr = np.asarray(tempi, dtype=float)
    if arr.size == 0:
        return None, "madmom_no_tempo"
    if arr.ndim == 1:
        if arr.size < 1 or not np.isfinite(arr[0]) or float(arr[0]) <= 0:
            return None, "madmom_no_tempo"
        return float(arr[0]), "ok"
    if arr.shape[1] < 1:
        return None, "madmom_no_tempo"
    bpm = float(arr[0, 0])
    if not np.isfinite(bpm) or bpm <= 0:
        return None, "madmom_no_tempo"
    return bpm, "ok"


def estimate_madmom_beats_via_tempo_dbn(
    audio_path: Path, sample_rate: int = DEFAULT_UNIFIED_SAMPLE_RATE
) -> tuple[np.ndarray, str]:
    """
    Estimate tempo via madmom DBN and synthesize an evenly spaced beat grid
    anchored at the strongest activation phase.
    """
    if TempoEstimationProcessor is None or RNNBeatProcessor is None:
        return np.array([]), "madmom_import_error"
    try:
        y = load_mono_resampled(audio_path, sample_rate)
        act = np.asarray(RNNBeatProcessor()(y), dtype=float).ravel()
        tempi = TempoEstimationProcessor(method="dbn", fps=100)(act)
    except Exception as exc:
        return np.array([]), f"madmom_runtime_error:{_fmt_exc(exc)}"

    arr = np.asarray(tempi, dtype=float)
    if arr.size == 0:
        return np.array([]), "madmom_no_tempo"
    bpm = float(arr[0] if arr.ndim == 1 else arr[0, 0])
    if not np.isfinite(bpm) or bpm <= 0:
        return np.array([]), "madmom_no_tempo"
    if act.size == 0:
        return np.array([]), "madmom_no_activation"

    period = 60.0 / bpm
    duration = float(len(y)) / float(sample_rate) if sample_rate > 0 else 0.0
    if not np.isfinite(period) or period <= 0 or duration <= 0:
        return np.array([]), "madmom_no_beats"

    anchor = float(np.argmax(act)) / 100.0
    start = anchor - np.floor(anchor / period) * period
    beats = np.arange(start, duration + period, period, dtype=float)
    beats = beats[np.isfinite(beats) & (beats >= 0.0) & (beats <= duration)]
    if beats.size == 0:
        return np.array([]), "madmom_no_beats"
    return beats, "ok"


def tempo_from_beat_times(beats_sec: np.ndarray) -> Optional[float]:
    if beats_sec.size < 2:
        return None
    ibi = np.diff(beats_sec)
    med_ibi = float(np.median(ibi))
    if med_ibi <= 0:
        return None
    return 60.0 / med_ibi


def estimate_madmom_beats_and_tempo(
    audio_path: Path, sample_rate: int = DEFAULT_UNIFIED_SAMPLE_RATE
) -> tuple[Optional[int], Optional[float], str]:
    beats, st = estimate_madmom_beats(audio_path, sample_rate=sample_rate)
    n = int(beats.size)
    if st != "ok":
        return n, None, st
    t = tempo_from_beat_times(beats)
    return n, t, "ok"


_1DSS_BEATNET_MODEL_ID = 1
_1DSS_ESTIMATOR: Any = None
_1DSS_IMPORT_ERROR: Optional[str] = None


def get_1dss_joint_estimator():
    """Singleton estimator; BeatNet weights are loaded only once."""
    global _1DSS_ESTIMATOR, _1DSS_IMPORT_ERROR
    if _1DSS_IMPORT_ERROR is not None:
        return None
    if _1DSS_ESTIMATOR is not None:
        return _1DSS_ESTIMATOR
    try:
        _patch_numpy_typing_ndarray()
        from jump_reward_inference.joint_tracker import joint_inference

        _1DSS_ESTIMATOR = joint_inference(_1DSS_BEATNET_MODEL_ID, plot=False)
    except Exception as exc:
        _1DSS_IMPORT_ERROR = _fmt_exc(exc)
        return None
    return _1DSS_ESTIMATOR


def estimate_1dss_beats_and_tempo(
    audio_path: Path,
    sample_rate: int = DEFAULT_UNIFIED_SAMPLE_RATE,
    *,
    use_backend_native_audio: bool = False,
) -> tuple[np.ndarray, Optional[float], str]:
    """
    ICASSP 2022-style 1D state-space + jump-reward inference.
    Output rows contain: (time[s], event_type, local_tempo, meter); beat times
    are taken from column 0.
    Global tempo uses median positive local tempo first, then IBI fallback.

    Default(False): load mono audio at fixed SR and process via temp WAV.
    use_backend_native_audio=True: pass original path and let BeatNet decode.
    """
    est = get_1dss_joint_estimator()
    if est is None:
        return (
            np.array([]),
            None,
            f"1dss_import_error:{_1DSS_IMPORT_ERROR or 'unknown'}",
        )
    tmp: Optional[str] = None
    try:
        if use_backend_native_audio:
            raw = est.process(_audio_fspath(audio_path))
        else:
            y = load_mono_resampled(audio_path, sample_rate)
            tmp = _write_mono_wav_temp(y, sample_rate)
            raw = est.process(tmp)
    except Exception as exc:
        return np.array([]), None, f"1dss_runtime_error:{_fmt_exc(exc)}"
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    out = np.asarray(raw, dtype=float)
    if out.size == 0:
        return np.array([]), None, "1dss_empty_output"
    if out.ndim != 2 or out.shape[1] < 3:
        return np.array([]), None, "1dss_bad_shape"
    times = np.asarray(out[:, 0], dtype=float).ravel()
    beats = times[np.isfinite(times) & (times >= 0)]
    local_tp = out[:, 2] if out.shape[1] > 2 else np.array([])
    t_ibi = tempo_from_beat_times(beats)
    tempo_global: Optional[float] = None
    t_loc: Optional[float] = None
    if local_tp.size > 0:
        pos = local_tp[np.isfinite(local_tp) & (local_tp > 0)]
        if pos.size > 0:
            t_loc = float(np.median(pos))
    if t_loc is not None and t_loc > 0:
        tempo_global = t_loc
    elif t_ibi is not None and t_ibi > 0:
        tempo_global = t_ibi
    else:
        tempo_global = t_loc if t_loc else t_ibi
    return beats, tempo_global, "ok"


def pct_error(pred: Optional[float], gt: Optional[float]) -> Optional[float]:
    if pred is None or gt is None or gt <= 0:
        return None
    return abs(pred - gt) / gt * 100.0


def mae_bpm(pred: Optional[float], gt: Optional[float]) -> Optional[float]:
    if pred is None or gt is None:
        return None
    return abs(float(pred) - float(gt))


def acc1_hit(pred: Optional[float], gt: Optional[float], thr: float = 0.04) -> Optional[bool]:
    """Gouyon-style ACC1: relative error within thr."""
    if pred is None or gt is None or gt <= 0:
        return None
    return abs(pred - gt) / gt <= thr


def acc2_hit(pred: Optional[float], gt: Optional[float], thr: float = 0.04) -> Optional[bool]:
    """
    Octave-tolerant tempo hit: pred, pred/2, or 2*pred within thr of gt.
    """
    if pred is None or gt is None or gt <= 0:
        return None
    candidates = [pred, pred * 0.5, pred * 2.0]
    return any(abs(c - gt) / gt <= thr for c in candidates)


def beat_fmeasure(
    ref_times: np.ndarray,
    est_times: np.ndarray,
    window: float = 0.07,
) -> Optional[float]:
    """mir_eval beat F-measure; returns None if ref/est is empty."""
    try:
        import mir_eval
    except ImportError:
        return None
    ref = np.asarray(ref_times, dtype=float).ravel()
    est = np.asarray(est_times, dtype=float).ravel()
    if ref.size == 0 or est.size == 0:
        return None
    return float(
        mir_eval.beat.f_measure(ref, est, f_measure_threshold=window)
    )


def beat_cemgil(
    ref_times: np.ndarray,
    est_times: np.ndarray,
    *,
    sigma: float = 0.04,
) -> tuple[Optional[float], Optional[float]]:
    """
    mir_eval beat Cemgil score + best metric level.
    If ref/est is empty, returns (None, None).
    """
    try:
        import mir_eval
    except ImportError:
        return None, None
    ref = np.asarray(ref_times, dtype=float).ravel()
    est = np.asarray(est_times, dtype=float).ravel()
    if ref.size == 0 or est.size == 0:
        return None, None
    score, best = mir_eval.beat.cemgil(ref, est, cemgil_sigma=sigma)
    return float(score), float(best)


def estimate_autocorr_tempo(
    audio_path: Path, sample_rate: int = DEFAULT_UNIFIED_SAMPLE_RATE
) -> Optional[float]:
    """Single BPM via OSS + generalized ACF + pulse + Gaussian accumulation."""
    from autocorr_tempo import estimate_autocorr_tempo as _est

    return _est(audio_path, sample_rate)
