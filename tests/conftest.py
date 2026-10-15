import dataclasses

import mlx.core as mx
import numpy as np
import pytest

from nemotron_diar_mlx.config import EncoderConfig, ModelConfig, ModulesConfig
from nemotron_diar_mlx.features import hann_window, mel_filterbank
from nemotron_diar_mlx.model import Model


def tiny_config(**modules) -> ModelConfig:
    """Same architecture as Nemotron 3, much smaller, with short chunks so tests exercise the cache."""
    stream = dict(spkcache_len=16, fifo_len=4, chunk_len=6, chunk_right_context=2, spkcache_update_period=5)
    stream.update(modules)
    return ModelConfig(
        encoder=EncoderConfig(d_model=64, n_heads=4, n_layers=2),
        modules=ModulesConfig(num_spks=4, fc_d_model=64, tf_d_model=32, **stream),
    )


def random_model(config: ModelConfig, seed: int = 0) -> Model:
    mx.random.seed(seed)
    model = Model(config)
    pre = config.preprocessor
    model.preprocessor_window = mx.array(hann_window(pre.win_length))
    model.preprocessor_fb = mx.array(mel_filterbank(pre.sample_rate, pre.n_fft, pre.features))
    if config.modules.use_learnable_sil_emb:
        model.sortformer.learnable_sil_emb = mx.random.normal((config.modules.fc_d_model,))
    # Larger output weights so speakers are clearly on/off and the cache logic is exercised.
    head = model.sortformer.single_hidden_to_spks
    head.weight = head.weight * 8
    mx.eval(model.parameters())
    model.eval()
    return model


@pytest.fixture
def tiny_model() -> Model:
    return random_model(tiny_config())


def voice_like_audio(seconds: float, seed: int = 0, sr: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    audio = 0.01 * rng.standard_normal(t.shape)
    for i, f0 in enumerate((110.0, 180.0, 240.0)):
        gate = (np.sin(2 * np.pi * (0.3 + 0.11 * i) * t + i) > 0.2).astype(np.float64)
        audio += 0.2 * gate * sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 5))
    return audio.astype(np.float32)


def replace_modules(config: ModelConfig, **changes) -> ModelConfig:
    return dataclasses.replace(config, modules=dataclasses.replace(config.modules, **changes))
