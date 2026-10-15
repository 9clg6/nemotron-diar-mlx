"""Log-mel features identical to NeMo's `FilterbankFeatures` (inference mode).

`log_mel` processes a whole signal; `IncrementalLogMel` produces exactly the
same frames from audio pushed in arbitrary blocks (streaming).

The Hann window and the mel filterbank are taken from the `.nemo` checkpoint
when available; otherwise `hann_window` / `mel_filterbank` rebuild them
(identical to `torch.hann_window(periodic=False)` and
`librosa.filters.mel(norm="slaney")`, which NeMo uses).
"""

from __future__ import annotations

from typing import Tuple

import mlx.core as mx
import numpy as np

from .config import PreprocessorConfig


def hann_window(length: int) -> np.ndarray:
    """Symmetric Hann window, as `torch.hann_window(length, periodic=False)`."""
    n = np.arange(length, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / (length - 1))).astype(np.float32)


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    # Slaney mel scale (librosa, htk=False): linear below 1 kHz, log above.
    f = np.asarray(f, dtype=np.float64)
    mels = f / (200.0 / 3)
    log_region = f >= 1000.0
    mels[log_region] = 15.0 + np.log(f[log_region] / 1000.0) / (np.log(6.4) / 27.0)
    return mels


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    freqs = (200.0 / 3) * m
    log_region = m >= 15.0
    freqs[log_region] = 1000.0 * np.exp((np.log(6.4) / 27.0) * (m[log_region] - 15.0))
    return freqs


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float = None) -> np.ndarray:
    """`librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False, norm="slaney")`."""
    fmax = sample_rate / 2.0 if fmax is None else fmax
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, 1 + n_fft // 2)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(np.array([fmin]))[0], _hz_to_mel(np.array([fmax]))[0], n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, None]
    return weights.astype(np.float32)


def num_feature_frames(num_samples: int, cfg: PreprocessorConfig) -> int:
    """Same as `FilterbankFeatures.get_seq_len` (centered STFT)."""
    pad = cfg.n_fft // 2 * 2
    return (num_samples + pad - cfg.n_fft) // cfg.hop_length


def _padded_window(window: mx.array, n_fft: int) -> mx.array:
    # torch.stft centers a window shorter than n_fft.
    left = (n_fft - window.shape[0]) // 2
    return mx.pad(window.astype(mx.float32), [(left, n_fft - window.shape[0] - left)])


def _frames_log_mel(
    x: mx.array, n_frames: int, window: mx.array, mel_fb: mx.array, cfg: PreprocessorConfig
) -> mx.array:
    """Log-mel of the first `n_frames` STFT frames of `x` (pre-emphasized and padded)."""
    if n_frames <= 0:
        return mx.zeros((0, mel_fb.shape[0]))
    idx = mx.arange(n_frames)[:, None] * cfg.hop_length + mx.arange(cfg.n_fft)[None, :]
    spec = mx.fft.rfft(x[idx] * _padded_window(window, cfg.n_fft), n=cfg.n_fft, axis=-1)
    power = mx.real(spec) ** 2 + mx.imag(spec) ** 2  # |X|^2 (mag_power = 2)
    mel = power @ mel_fb.astype(mx.float32).T
    return mx.log(mel + cfg.log_zero_guard_value)


def log_mel(
    waveform: mx.array,
    window: mx.array,
    mel_fb: mx.array,
    cfg: PreprocessorConfig,
) -> Tuple[mx.array, int]:
    """Features of a mono signal.

    Args:
        waveform: float32 samples at `cfg.sample_rate`, shape `(num_samples,)`.
        window: analysis window `(win_length,)`.
        mel_fb: filterbank `(n_mels, n_fft // 2 + 1)`.

    Returns:
        Features `(num_frames, n_mels)` and the number of valid frames.
    """
    x = waveform.astype(mx.float32)
    valid = num_feature_frames(x.shape[0], cfg)
    # Pre-emphasis (dither is only applied during training).
    x = mx.concatenate([x[:1], x[1:] - cfg.preemph * x[:-1]])
    # torch.stft(center=True, pad_mode="constant").
    half = cfg.n_fft // 2
    x = mx.pad(x, [(half, half)])
    # NeMo zeroes frames past seq_len and the model truncates them: only valid
    # frames are computed.
    return _frames_log_mel(x, valid, window, mel_fb, cfg), valid


class IncrementalLogMel:
    """Log-mel features computed on the fly, identical to `log_mel`.

    Frame `f` is centered on sample `f * hop`: it is emitted as soon as the
    following `n_fft / 2` samples are available. `flush` appends the end
    padding of the centered STFT.
    """

    def __init__(self, window: mx.array, mel_fb: mx.array, cfg: PreprocessorConfig):
        self.window = window
        self.mel_fb = mel_fb
        self.cfg = cfg
        self.half = cfg.n_fft // 2
        self.num_samples = 0
        self.next_frame = 0
        self._prev = np.float32(0.0)
        # Pre-emphasized signal preceded by the left padding; `_start` is the
        # absolute sample index of `_buf[0]` (negative inside the padding).
        self._buf = np.zeros(self.half, dtype=np.float32)
        self._start = -self.half

    def push(self, audio: np.ndarray) -> mx.array:
        """Append samples; return the new frames `(n, n_mels)`."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size:
            shifted = np.concatenate([[self._prev], audio[:-1]]).astype(np.float32)
            self._prev = audio[-1]
            self._buf = np.concatenate([self._buf, audio - np.float32(self.cfg.preemph) * shifted])
            self.num_samples += audio.size
        ready = 0
        if self.num_samples >= self.half:
            ready = (self.num_samples - self.half) // self.cfg.hop_length + 1
        return self._emit(ready)

    def flush(self) -> mx.array:
        """End the signal; return the last frames."""
        self._buf = np.concatenate([self._buf, np.zeros(self.half, dtype=np.float32)])
        return self._emit(num_feature_frames(self.num_samples, self.cfg))

    def _emit(self, until: int) -> mx.array:
        n = until - self.next_frame
        if n <= 0:
            return mx.zeros((0, self.mel_fb.shape[0]))
        offset = self.next_frame * self.cfg.hop_length - self.half - self._start
        frames = _frames_log_mel(mx.array(self._buf[offset:]), n, self.window, self.mel_fb, self.cfg)
        self.next_frame = until
        # Keep only the samples needed by the next frames.
        keep_from = self.next_frame * self.cfg.hop_length - self.half
        self._buf = self._buf[keep_from - self._start :]
        self._start = keep_from
        return frames
