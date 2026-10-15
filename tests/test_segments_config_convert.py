import json

import mlx.core as mx
import numpy as np
import pytest

from nemotron_diar_mlx import STREAMING_PRESETS, Model, ModelConfig, speaker_segments
from nemotron_diar_mlx.convert import convert, resolve
from nemotron_diar_mlx.segments import binarize, filter_segments

from .conftest import tiny_config, voice_like_audio


# --------------------------------------------------------------------------- segments


def test_binarize_matches_nemo_semantics():
    seq = np.array([0, 0, 0.9, 0.9, 0.2, 0, 0.8, 0.8, 0.8])
    # NeMo closes a final open segment at the *last* frame index.
    np.testing.assert_allclose(binarize(seq, 0.01), [[0.02, 0.04], [0.06, 0.08]])


def test_hysteresis_and_padding():
    seq = np.array([0, 0.8, 0.45, 0.45, 0.2, 0])
    np.testing.assert_allclose(binarize(seq, 0.1, onset=0.7, offset=0.3), [[0.1, 0.4]])
    np.testing.assert_allclose(binarize(seq, 0.1, onset=0.7, offset=0.3, pad_onset=0.5, pad_offset=0.1), [[0.0, 0.5]])


def test_filtering_short_speech_and_gaps():
    segs = np.array([[0.0, 1.0], [1.1, 2.0], [3.0, 3.05], [5.0, 6.0]])
    np.testing.assert_allclose(filter_segments(segs, min_duration_on=0.1), [[0.0, 1.0], [1.1, 2.0], [5.0, 6.0]])
    np.testing.assert_allclose(filter_segments(segs, min_duration_on=0.1, min_duration_off=0.5),
                               [[0.0, 2.0], [5.0, 6.0]])


def test_speaker_segments_sorted_with_speakers():
    preds = np.zeros((100, 3))
    preds[10:40, 1] = 0.9
    preds[30:60, 0] = 0.9
    segs = speaker_segments(preds, 0.01)
    assert [(round(s.start, 2), round(s.end, 2), s.speaker) for s in segs] == [(0.1, 0.4, 1), (0.3, 0.6, 0)]
    assert segs[0].to_dict() == {"start": 0.1, "end": 0.4, "speaker": 1}


# --------------------------------------------------------------------------- config


def test_presets_apply_and_validate():
    cfg = ModelConfig()
    for name, values in STREAMING_PRESETS.items():
        modules = cfg.with_streaming(name).modules
        assert {k: getattr(modules, k) for k in values} == values
    with pytest.raises(ValueError):
        cfg.with_streaming("nope")
    with pytest.raises(ValueError):
        cfg.with_streaming(chunk_len=0)
    with pytest.raises(ValueError):
        cfg.with_streaming(spkcache_len=8)  # needs >= (1 + sil frames) * 8 speakers


def test_rejects_fastconformer_sortformer():
    with pytest.raises(ValueError, match="Unsupported encoder"):
        ModelConfig.from_nemo_config({"encoder": {"_target_": "nemo.collections.asr.modules.ConformerEncoder"}})


def test_config_round_trip():
    cfg = tiny_config()
    assert ModelConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg


# --------------------------------------------------------------------------- Transformers-format conversion


def _write_tiny_hf_checkpoint(path, cfg: ModelConfig, seed=0):
    """A tiny checkpoint in the layout of the Transformers export of Nemotron 3 Diarization."""
    rng = np.random.default_rng(seed)
    e, m = cfg.encoder, cfg.modules

    def w(*shape):
        return mx.array(rng.standard_normal(shape).astype(np.float32) / np.sqrt(shape[-1]))

    t = {"model.audio_tower.embedder.projection.weight": w(e.d_model, e.feat_in * e.subsampling_factor)}
    for name in ("input_layer_norm", "layer_norm"):
        t[f"model.audio_tower.{name}.weight"] = mx.ones((e.d_model,))
        t[f"model.audio_tower.{name}.bias"] = mx.zeros((e.d_model,))
    for i in range(e.n_layers):
        p = f"model.audio_tower.layers.{i}."
        for n in ("layer_norm1", "layer_norm2"):
            t[p + n + ".weight"], t[p + n + ".bias"] = mx.ones((e.d_model,)), mx.zeros((e.d_model,))
        for n in ("q", "k", "v"):
            t[p + f"self_attn.{n}_proj.weight"] = w(e.d_model, e.d_model)
        t[p + "self_attn.o_proj.weight"], t[p + "self_attn.o_proj.bias"] = w(e.d_model, e.d_model), w(e.d_model)
        t[p + "mlp.fc1.weight"], t[p + "mlp.fc1.bias"] = w(e.ff_hidden, e.d_model), w(e.ff_hidden)
        t[p + "mlp.fc2.weight"], t[p + "mlp.fc2.bias"] = w(e.d_model, e.ff_hidden), w(e.d_model)
    u = e.subsampling_factor
    t.update({
        "model.proj.weight": w(m.tf_d_model, m.fc_d_model), "model.proj.bias": w(m.tf_d_model),
        "model.upsampler.conv.weight": w(m.tf_d_model * u, m.tf_d_model, 3),
        "model.upsampler.conv.bias": w(m.tf_d_model * u),
        "classifier.dense.weight": w(m.tf_d_model, m.tf_d_model), "classifier.dense.bias": w(m.tf_d_model),
        "classifier.out_proj.weight": w(m.num_spks, m.tf_d_model), "classifier.out_proj.bias": w(m.num_spks),
        "silence_embeds": w(m.fc_d_model),
    })
    path.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path / "model.safetensors"), t)
    config = {
        "model_type": "nemotron3_diarization",
        "audio_config": {
            "hidden_size": e.d_model, "intermediate_size": e.ff_hidden, "num_attention_heads": e.n_heads,
            "num_hidden_layers": e.n_layers, "num_mel_bins": e.feat_in, "subsampling_factor": u,
            "hidden_act": "gelu", "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 1.0},
        },
        "head_config": {"audio_hidden_size": m.fc_d_model, "hidden_size": m.tf_d_model, "num_speakers": m.num_spks},
        "chunk_length": m.chunk_len, "chunk_right_context": m.chunk_right_context, "fifo_length": m.fifo_len,
        "speaker_cache_update_period": m.spkcache_update_period,
        "streaming_config": {"speaker_cache_length": m.spkcache_len, "speaker_cache_silence_frames_per_speaker": 1},
    }
    (path / "config.json").write_text(json.dumps(config))
    return t


def test_convert_transformers_checkpoint(tmp_path):
    cfg = tiny_config()
    tensors = _write_tiny_hf_checkpoint(tmp_path / "hf", cfg)
    out = convert(tmp_path / "hf", tmp_path / "mlx")
    model = Model.from_pretrained(out, streaming=None)
    # q/k/v are fused as [q | k | v].
    qkv = model.encoder.layers[1].attn.w_qkv.weight
    d = cfg.encoder.d_model
    for i, n in enumerate("qkv"):
        assert mx.array_equal(qkv[i * d : (i + 1) * d], tensors[f"model.audio_tower.layers.1.self_attn.{n}_proj.weight"])
    assert model.config.modules.chunk_len == cfg.modules.chunk_len
    assert not model.config.modules.use_activity_head
    out_preds = model(voice_like_audio(4.0))["preds"]
    assert out_preds.shape == (400, cfg.modules.num_spks) and np.isfinite(out_preds).all()


def test_resolve_caches_conversion(tmp_path):
    _write_tiny_hf_checkpoint(tmp_path / "hf", tiny_config())
    first = resolve(tmp_path / "hf", cache_dir=tmp_path / "cache")
    mtime = (first / "model.safetensors").stat().st_mtime_ns
    assert resolve(tmp_path / "hf", cache_dir=tmp_path / "cache") == first
    assert (first / "model.safetensors").stat().st_mtime_ns == mtime
    assert resolve(first) == first  # an MLX directory resolves to itself
