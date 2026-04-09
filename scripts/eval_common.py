"""
공통 tempo/beat 추정과 지표 (DSP librosa, madmom DBN, 선택적 1D-StateSpace / jump-reward-inference).
통합 eval 및 quick check 스크립트에서 재사용한다.
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
except Exception:  # pragma: no cover
    DBNBeatTrackingProcessor = None
    RNNBeatProcessor = None

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
                # Support annotations like NDArray[np.float32]
                def __class_getitem__(cls, _item):
                    return np.ndarray

            npt.NDArray = _CompatNDArray  # type: ignore[attr-defined]
    except Exception:
        return

def _patch_scipy_signal_windows() -> None:
    """
    librosa 내부에서 `scipy.signal.hann` 같은 예전 심볼을 참조할 때,
    scipy 버전에 따라 제거되어 있으면 windows.hann로 보정한다.
    """
    try:
        import scipy.signal as s

        if not hasattr(s, "hann") and hasattr(s, "windows"):
            # SciPy 1.9+ 에서 hann이 windows로 이동하면서 signal.hann이 사라진 케이스 대응
            s.hann = s.windows.hann  # type: ignore[attr-defined]
    except Exception:
        # 패치가 실패해도 기본 동작은 그대로 두되, 에러가 더 명확히 드러나게 둔다.
        return


def _audio_fspath(audio_path: Path) -> str:
    """Windows에서 상대/혼합 경로 이슈 줄이기 위해 절대 경로 문자열."""
    p = audio_path.expanduser()
    try:
        return str(p.resolve(strict=False))
    except TypeError:
        return str(p.resolve())


def _fmt_exc(exc: BaseException, max_len: int = 160) -> str:
    s = f"{type(exc).__name__}:{exc}"
    return s.replace(";", ",").replace("\n", " ")[:max_len]


# madmom RNNBeatProcessor 기본 학습 조건과 맞추기 쉬운 기본값.
MADMOM_SAMPLE_RATE = 44100
DEFAULT_UNIFIED_SAMPLE_RATE = MADMOM_SAMPLE_RATE


def load_mono_resampled(audio_path: Path, sample_rate: int) -> np.ndarray:
    """
    모든 평가 경로에서 동일한 입력을 쓰기 위해: mono, float32, 지정 SR로 librosa.load.
    """
    _patch_scipy_signal_windows()
    y, _ = librosa.load(_audio_fspath(audio_path), sr=sample_rate, mono=True)
    return np.asarray(y, dtype=np.float32)


def _write_mono_wav_temp(y: np.ndarray, sample_rate: int) -> str:
    """FLOAT WAV 임시 파일 경로 (1DSS 등 경로-only API용). 호출측에서 삭제."""
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
    librosa 0.11+ / NumPy 2: beat_track 가 반환하는 tempo 가 float, np.float64,
    0차원 ndarray 등 여러 형태라서 ravel 로 통일한다 (구버전 np.isscalar + len 분기는 깨짐).
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
    """librosa beat_track → 비트 시각(초), 단일 템포 추정값(있으면)."""
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
    RNN + DBN beat times (초).

    파일 경로만 넘기면 madmom 이 ffmpeg 로 디코드한다(Windows 에서 PATH 없으면 실패).
    librosa/soundfile 로 디코드한 뒤 float32 로 넣어 ffmpeg 없이 동일 파이프라인을 쓴다.
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


# mjhydri/1D-StateSpace — PyPI: jump-reward-inference (BeatNet + jump-reward inference)
_1DSS_BEATNET_MODEL_ID = 1
_1DSS_ESTIMATOR: Any = None
_1DSS_IMPORT_ERROR: Optional[str] = None


def get_1dss_joint_estimator():
    """싱글톤: BeatNet 가중치 로드는 첫 호출에만 (무겁다)."""
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
    ICASSP 2022 1D state-space + jump-reward (BeatNet 활성도 → 추론).
    출력 행: (시간[s], 이벤트종류, local tempo, meter) — 비트 시각은 col 0.
    전역 템포: local tempo 열의 양수 중앙값 우선, 없으면 IBI(비트 간격) 중앙값.

    기본(False): DSP/madmom과 동일하게 mono·지정 SR로 librosa 로드 후 임시 WAV로 process.
    use_backend_native_audio=True: 원본 파일 경로만 넘김 → BeatNet이 librosa.load(..., sr=22050)로
    직접 디코드 (jump-reward-inference / 논문 예제와 동일한 입력 경로).
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
    """Gouyon-style ACC1: 상대 오차 thr 이내."""
    if pred is None or gt is None or gt <= 0:
        return None
    return abs(pred - gt) / gt <= thr


def acc2_hit(pred: Optional[float], gt: Optional[float], thr: float = 0.04) -> Optional[bool]:
    """
    Octave-tolerant tempo hit: pred, pred/2, 2*pred 중 하나가 gt에 대해 thr 이내.
    (단일 템포 추정에서 흔한 octave 혼동 완화)
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
    """mir_eval beat F-measure; ref/est가 비어 있으면 None."""
    try:
        import mir_eval
    except ImportError:
        return None
    ref = np.asarray(ref_times, dtype=float).ravel()
    est = np.asarray(est_times, dtype=float).ravel()
    if ref.size == 0 or est.size == 0:
        return None
    # mir_eval 0.7: f_measure(reference_beats, estimated_beats, f_measure_threshold=...)
    return float(
        mir_eval.beat.f_measure(ref, est, f_measure_threshold=window)
    )


def estimate_autocorr_tempo(
    audio_path: Path, sample_rate: int = DEFAULT_UNIFIED_SAMPLE_RATE
) -> Optional[float]:
    """OSS·광의 자기상관·펄스·가우시안 누적 단일 BPM (텍스트 알고리즘, stem base)."""
    from autocorr_tempo import estimate_autocorr_tempo as _est

    return _est(audio_path, sample_rate)
