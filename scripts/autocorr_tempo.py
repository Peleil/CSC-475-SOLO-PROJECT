"""
Single-track global BPM: onset strength (OSS) -> generalized autocorrelation ->
harmonic enhancement -> pulse-train scoring -> Gaussian accumulation.

Reference text: `papers/text version.txt`, Section II.
Final BPM follows the stem-based formula T = 60 * F_sO / L
(where L is the peak lag in the accumulator).
SVM octave-correction weights are omitted because they are not provided
in that text source.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal

from eval_common import load_mono_resampled

_FRAME_AUDIO = 1024
_HOP_AUDIO = 128
_FRAME_OSS = 2048
_HOP_OSS = 128
_MIN_BPM = 50
_MAX_BPM = 210
_AC_COMPRESS_C = 0.5
_GAUSSIAN_SIGMA_SAMPLES = 10.0
_ACCUM_SIZE = 512


def _fs_o(sample_rate: int, hop_audio: int) -> float:
    return float(sample_rate) / float(hop_audio)


def _oss_spectrogram(
    y: np.ndarray, sr: int, frame: int, hop: int
) -> tuple[np.ndarray, np.ndarray]:
    """L_P, |X| with Hamming window and DFT magnitude."""
    y = np.asarray(y, dtype=np.float64)
    win = signal.windows.hamming(frame, sym=False)
    n = len(y)
    if n < frame:
        return (
            np.zeros((0, 0), dtype=np.float64),
            np.zeros((0, 0), dtype=np.float64),
        )
    n_frames = 1 + (n - frame) // hop
    k_bins = frame // 2 + 1
    mag = np.zeros((k_bins, n_frames), dtype=np.float64)
    for t in range(n_frames):
        start = t * hop
        seg = y[start : start + frame] * win
        spec = np.fft.rfft(seg, n=frame)
        mag[:, t] = np.abs(spec)
    lp = np.log(1.0 + 1000.0 * mag)
    return mag, lp


def _oss_flux(mag: np.ndarray, lp: np.ndarray) -> np.ndarray:
    """k=1..N-1 excluding DC; only positive |X| increments."""
    k_bins, n_frames = mag.shape
    if n_frames < 2:
        return np.array([], dtype=np.float64)
    flux = np.zeros(n_frames, dtype=np.float64)
    for n in range(1, n_frames):
        for k in range(1, k_bins):
            if mag[k, n] - mag[k, n - 1] <= 0:
                continue
            flux[n] += lp[k, n] - lp[k, n - 1]
    return flux


def _lowpass_oss(oss: np.ndarray, fs_o: float) -> np.ndarray:
    """14th-order FIR low-pass at 7 Hz (15 taps)."""
    if oss.size == 0:
        return oss
    taps = signal.firwin(
        15, 7.0, fs=fs_o, window="hamming"
    )
    return signal.lfilter(taps, np.array([1.0]), oss)


def _generalized_acf(oss_frame: np.ndarray, c: float) -> np.ndarray:
    """2x zero-pad -> DFT -> |.|^c -> IDFT; use real lag axis."""
    # 4.1 formula: A(tau) = Re{ F^{-1}(|Y|^c) }(tau)
    L = len(oss_frame)
    nfft = 2 * L
    pad = np.zeros(nfft, dtype=np.float64)
    pad[:L] = oss_frame
    Y = np.fft.fft(pad)
    Z = np.abs(Y) ** c
    ac = np.fft.ifft(Z).real
    return ac


def _enhance_acf(ac: np.ndarray) -> np.ndarray:
    """EAC(t)=A(t)+A(2t)+A(4t), sampling from original A."""
    # 4.1 formula: E(tau) = A(tau) + A(2tau) + A(4tau)
    a = np.asarray(ac, dtype=np.float64)
    n = len(a)
    eac = np.zeros(n, dtype=np.float64)
    for t in range(n):
        s = a[t]
        if 2 * t < n:
            s += a[2 * t]
        if 4 * t < n:
            s += a[4 * t]
        eac[t] = s
    return eac


def _top_peaks_eac(
    eac: np.ndarray, min_lag: int, max_lag: int, max_peaks: int = 10
) -> np.ndarray:
    max_lag = min(max_lag, len(eac) - 1)
    min_lag = max(1, min_lag)
    if min_lag >= max_lag:
        return np.array([], dtype=np.float64)
    seg = eac[min_lag : max_lag + 1]
    cand: list[tuple[int, float]] = []
    for i in range(1, len(seg) - 1):
        if seg[i] > seg[i - 1] and seg[i] > seg[i + 1]:
            cand.append((min_lag + i, float(seg[i])))
    if not cand:
        return np.array([], dtype=np.float64)
    cand.sort(key=lambda u: -u[1])
    return np.array([float(u[0]) for u in cand[:max_peaks]], dtype=np.float64)


def _pulse_correlation_scores(oss_win: np.ndarray, period: int) -> tuple[float, float]:
    """phi=0..P-1, v in {1,1.5,2}, weights 1, 0.5, 0.5."""
    if period <= 0:
        return 0.0, 0.0
    P = period
    o = oss_win
    rho = np.zeros(P, dtype=np.float64)
    for phi in range(P):
        acc = 0.0
        for b in range(4):
            for v, w in ((1.0, 1.0), (1.5, 0.5), (2.0, 0.5)):
                j = int(round(phi + v * b * P))
                if 0 <= j < len(o):
                    acc += w * o[j]
        rho[phi] = acc
    sc_x = float(np.max(rho))
    sc_v = float(np.var(rho))
    return sc_x, sc_v


def _best_lag_for_frame(
    oss_win: np.ndarray, peak_candidates: np.ndarray
) -> int:
    """Per-frame best lag from pulse-score combination."""
    if peak_candidates.size == 0:
        return -1
    scx: list[float] = []
    scv: list[float] = []
    for p in peak_candidates:
        P = int(round(float(p)))
        if P <= 0:
            scx.append(0.0)
            scv.append(0.0)
            continue
        x, v = _pulse_correlation_scores(oss_win, P)
        scx.append(x)
        scv.append(v)
    ts = np.array(scx, dtype=np.float64)
    vs = np.array(scv, dtype=np.float64)
    st = float(np.sum(ts))
    so = float(np.sum(vs))
    if st <= 0.0 or so <= 0.0:
        return -1
    combo = ts / st + vs / so
    best = int(np.argmax(combo))
    return int(round(float(peak_candidates[best])))


def _gaussian_kernel(mu: float, sigma: float, size: int) -> np.ndarray:
    """Gaussian kernel on discrete samples x=0..size-1."""
    x = np.arange(size, dtype=np.float64)
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
        -((x - mu) ** 2) / (2.0 * sigma**2)
    )


def estimate_autocorr_tempo(
    audio_path: Path,
    sample_rate: int,
    *,
    min_bpm: int = _MIN_BPM,
    max_bpm: int = _MAX_BPM,
) -> float | None:
    """Stem-based BPM via Gaussian-lag accumulation and lag-to-BPM conversion."""
    y = load_mono_resampled(audio_path, sample_rate)
    fs_o = _fs_o(sample_rate, _HOP_AUDIO)

    mag, lp = _oss_spectrogram(y, sample_rate, _FRAME_AUDIO, _HOP_AUDIO)
    if mag.size == 0:
        return None
    flux = _oss_flux(mag, lp)
    if flux.size < _FRAME_OSS:
        return None

    oss = _lowpass_oss(flux.astype(np.float64), fs_o)
    if np.any(~np.isfinite(oss)):
        return None

    min_lag = int(np.floor(60.0 * fs_o / float(max_bpm)))
    max_lag = int(np.floor(60.0 * fs_o / float(min_bpm))) + 1
    min_lag = max(1, min_lag)
    max_lag = min(max_lag, _ACCUM_SIZE - 1)
    if min_lag >= max_lag:
        return None

    accum = np.zeros(_ACCUM_SIZE, dtype=np.float64)
    sigma = _GAUSSIAN_SIGMA_SAMPLES

    for start in range(0, len(oss) - _FRAME_OSS + 1, _HOP_OSS):
        w = oss[start : start + _FRAME_OSS].copy()
        ac = _generalized_acf(w, _AC_COMPRESS_C)
        eac = _enhance_acf(ac)
        peaks = _top_peaks_eac(eac, min_lag, max_lag, max_peaks=10)
        L_m = _best_lag_for_frame(w, peaks)
        if L_m <= 0 or L_m >= _ACCUM_SIZE:
            continue
        g = _gaussian_kernel(float(L_m), sigma, _ACCUM_SIZE)
        accum += g

    L_star = int(np.argmax(accum))
    if L_star <= 0 or accum[L_star] <= 0.0:
        return None
    # 4.1 formula: BPM = 60 * F_OSS / L*
    bpm = 60.0 * fs_o / float(L_star)
    return float(bpm) if bpm > 0 else None
