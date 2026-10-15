import mlx.core as mx
import numpy as np
import pytest

from nemotron_diar_mlx.config import PreprocessorConfig
from nemotron_diar_mlx.features import IncrementalLogMel, hann_window, log_mel, mel_filterbank, num_feature_frames

from .conftest import voice_like_audio

CFG = PreprocessorConfig()


def _buffers():
    return mx.array(hann_window(CFG.win_length)), mx.array(mel_filterbank(CFG.sample_rate, CFG.n_fft, CFG.features))


def test_frame_count_matches_nemo_seq_len():
    for n in (0, 159, 160, 161, 16000, 16123):
        assert num_feature_frames(n, CFG) == n // 160


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_incremental_features_equal_one_shot(seed):
    window, fb = _buffers()
    audio = voice_like_audio(3.3, seed=seed)
    whole, n = log_mel(mx.array(audio), window, fb, CFG)
    inc = IncrementalLogMel(window, fb, CFG)
    rng = np.random.default_rng(seed)
    parts, pos = [], 0
    while pos < len(audio):
        step = int(rng.integers(1, 5000))
        parts.append(np.array(inc.push(audio[pos : pos + step])))
        pos += step
    parts.append(np.array(inc.flush()))
    got = np.concatenate(parts)
    assert got.shape == (n, CFG.features)
    if mx.default_device() == mx.cpu:
        np.testing.assert_array_equal(np.array(whole), got)
    else:
        # GPU kernels (Metal FFT / matmul) round differently with the batch size.
        np.testing.assert_allclose(np.array(whole), got, rtol=1e-5, atol=1e-4)


def test_hann_window_is_symmetric_hann():
    w = hann_window(400)
    assert w[0] == 0.0 and w[-1] == 0.0
    np.testing.assert_allclose(w, w[::-1], atol=1e-7)


def test_hann_window_matches_torch():
    torch = pytest.importorskip("torch")
    np.testing.assert_allclose(hann_window(400), torch.hann_window(400, periodic=False).numpy(), atol=1e-6)


def test_mel_filterbank_matches_librosa():
    librosa = pytest.importorskip("librosa")
    ref = librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0, fmax=None, norm="slaney")
    np.testing.assert_allclose(mel_filterbank(16000, 512, 128), ref, atol=1e-7)


def test_mel_filterbank_shape_and_coverage():
    fb = mel_filterbank(16000, 512, 128)
    assert fb.shape == (128, 257)
    assert (fb >= 0).all()
    assert (fb.sum(axis=1) > 0).all()  # every band has support
