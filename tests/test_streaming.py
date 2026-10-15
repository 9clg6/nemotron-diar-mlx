import numpy as np
import pytest

from nemotron_diar_mlx.config import ModulesConfig
from nemotron_diar_mlx.streaming import StreamingDiarizer, StreamingState, compress_spkcache, streaming_update

from .conftest import random_model, tiny_config, voice_like_audio


def test_feed_block_size_does_not_change_output(tiny_model):
    audio = voice_like_audio(9.0)
    reference = tiny_model(audio)["preds"]
    assert reference.shape == (len(audio) // 160, 4)
    rng = np.random.default_rng(0)
    diarizer = StreamingDiarizer(tiny_model)
    outs, pos = [], 0
    while pos < len(audio):
        step = int(rng.integers(1, 4000))
        outs.append(diarizer.feed(audio[pos : pos + step]))
        pos += step
    outs.append(diarizer.flush())
    assert [o.start_frame for o in outs] == list(np.cumsum([0] + [len(o.preds) for o in outs[:-1]]))
    np.testing.assert_allclose(np.concatenate([o.preds for o in outs]), reference, atol=1e-5)


def test_cache_is_used_and_bounded(tiny_model):
    sizes = []
    diarizer = StreamingDiarizer(tiny_model)
    audio = voice_like_audio(9.0)
    for i in range(0, len(audio), 1600):
        diarizer.feed(audio[i : i + 1600])
        sizes.append(diarizer.state.spkcache.shape[0])
    diarizer.flush()
    cfg = tiny_model.config.modules
    assert max(sizes) <= cfg.spkcache_len
    assert diarizer.state.spkcache_compressed, "the test audio should overflow the cache"


def test_single_chunk_streaming_equals_offline():
    # Audio shorter than one chunk: no cache, no FIFO -> same computation as offline.
    model = random_model(tiny_config(chunk_len=40, chunk_right_context=0, spkcache_len=48, fifo_len=0))
    audio = voice_like_audio(2.0)
    np.testing.assert_allclose(model.forward_streaming(audio)["preds"], model.forward_offline(audio)["preds"], atol=1e-5)


def test_short_and_empty_audio(tiny_model):
    assert tiny_model(np.zeros(0, np.float32))["preds"].shape == (0, 4)
    assert tiny_model(np.zeros(100, np.float32))["preds"].shape == (0, 4)
    assert tiny_model(np.zeros(1700, np.float32))["preds"].shape == (10, 4)


def test_flush_twice_raises(tiny_model):
    diarizer = StreamingDiarizer(tiny_model)
    diarizer.flush()
    with pytest.raises(RuntimeError):
        diarizer.flush()
    with pytest.raises(RuntimeError):
        diarizer.feed(np.zeros(10, np.float32))


def _random_cache_inputs(n, cfg, seed):
    rng = np.random.default_rng(seed)
    logits = rng.normal(-2.0, 2.5, size=(n, cfg.num_spks)).astype(np.float32)
    logits[np.arange(n), rng.integers(0, cfg.num_spks, size=n)] += 6.0
    return rng.standard_normal((n, cfg.fc_d_model)).astype(np.float32), 1 / (1 + np.exp(-logits))


def test_compress_spkcache_layout():
    cfg = ModulesConfig(num_spks=4, fc_d_model=8, spkcache_len=20, spkcache_sil_frames_per_spk=1)
    emb, preds = _random_cache_inputs(37, cfg, seed=0)
    sil = np.full(8, 7.0, np.float32)
    new_emb, new_preds = compress_spkcache(emb, preds, sil, cfg)
    assert new_emb.shape == (20, 8) and new_preds.shape == (20, 4)
    is_sil = (new_emb == 7.0).all(axis=1)
    assert (new_preds[is_sil] == 0).all()
    # Kept frames come from the input, with their predictions.
    kept = new_emb[~is_sil]
    assert len(kept) > 0
    for e, p in zip(kept, new_preds[~is_sil]):
        row = int(np.flatnonzero((emb == e).all(axis=1))[0])
        np.testing.assert_array_equal(preds[row], p)


def test_compress_spkcache_ties_are_deterministic():
    cfg = ModulesConfig(num_spks=4, fc_d_model=8, spkcache_len=20, spkcache_sil_frames_per_spk=1)
    emb, preds = _random_cache_inputs(30, cfg, seed=1)
    emb = np.repeat(emb[:10], 3, axis=0)
    preds = np.repeat(preds[:10], 3, axis=0)  # every score appears three times
    a = compress_spkcache(emb, preds, np.zeros(8, np.float32), cfg)
    b = compress_spkcache(emb, preds, np.zeros(8, np.float32), cfg)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_streaming_update_moves_frames_from_fifo_to_cache():
    cfg = ModulesConfig(num_spks=4, fc_d_model=8, spkcache_len=20, fifo_len=4, chunk_len=6, spkcache_update_period=5)
    state = StreamingState.empty(8, 4)
    rng = np.random.default_rng(0)
    for _ in range(6):
        n = state.spkcache.shape[0] + state.fifo.shape[0] + 6
        out = streaming_update(state, rng.standard_normal((6, 8)).astype(np.float32),
                               rng.uniform(size=(n, 4)).astype(np.float32), cfg, np.zeros(8, np.float32))
        assert out.shape == (6, 4)
        assert state.fifo.shape[0] <= cfg.fifo_len
        assert state.spkcache.shape[0] <= cfg.spkcache_len
        assert state.spkcache_preds.shape[0] == state.spkcache.shape[0]
