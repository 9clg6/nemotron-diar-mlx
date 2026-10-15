"""Configuration of the MLX port of NVIDIA Nemotron 3 Diarization.

Defaults match `nvidia/Nemotron-3-Diarization`. A config can be built from
either published format:

- `ModelConfig.from_nemo_config`: the `model_config.yaml` inside the `.nemo`
  checkpoint (NeMo);
- `ModelConfig.from_hf_config`: `config.json` + `processor_config.json` of the
  Hugging Face Transformers format.

Unsupported architectures (e.g. Sortformer v2.x with a FastConformer encoder)
are rejected explicitly.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Streaming configurations recommended on the model card, in 80 ms encoder
# frames. Input buffer latency = (chunk_len + chunk_right_context) x 80 ms.
STREAMING_PRESETS: Dict[str, Dict[str, int]] = {
    # 30.4 s buffer: best accuracy, for recorded audio.
    "offline": dict(spkcache_len=264, fifo_len=40, chunk_len=340, chunk_right_context=40, spkcache_update_period=300),
    "low_latency": dict(spkcache_len=264, fifo_len=264, chunk_len=9, chunk_right_context=4, spkcache_update_period=222),
    "very_low_latency": dict(
        spkcache_len=264, fifo_len=264, chunk_len=6, chunk_right_context=2, spkcache_update_period=222
    ),
    "ultra_low_latency": dict(
        spkcache_len=264, fifo_len=264, chunk_len=3, chunk_right_context=1, spkcache_update_period=222
    ),
}
DEFAULT_PRESET = "offline"


@dataclass
class PreprocessorConfig:
    sample_rate: int = 16000
    window_size: float = 0.025
    window_stride: float = 0.01
    n_fft: int = 512
    features: int = 128
    preemph: float = 0.97
    log_zero_guard_value: float = 2**-24

    @property
    def win_length(self) -> int:
        return int(round(self.window_size * self.sample_rate))

    @property
    def hop_length(self) -> int:
        return int(round(self.window_stride * self.sample_rate))


@dataclass
class EncoderConfig:
    feat_in: int = 128
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 31
    subsampling_factor: int = 8
    ff_expansion: float = 4.0
    qkv_bias: bool = False
    qk_norm: bool = False
    pre_block_norm: bool = True
    rope_base: float = 10000.0
    xscaling: bool = False

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ff_hidden(self) -> int:
        return int(self.ff_expansion * self.d_model)


@dataclass
class ModulesConfig:
    """Sortformer heads and streaming (speaker cache) parameters."""

    num_spks: int = 8
    fc_d_model: int = 512
    tf_d_model: int = 192
    use_activity_head: bool = True
    use_learnable_sil_emb: bool = True
    spkcache_len: int = 264
    fifo_len: int = 40
    chunk_len: int = 340
    spkcache_update_period: int = 300
    chunk_left_context: int = 0
    chunk_right_context: int = 40
    spkcache_sil_frames_per_spk: int = 1
    pred_score_threshold: float = 0.25
    max_index: int = 99999
    scores_boost_latest: float = 0.05
    sil_threshold: float = 0.2
    strong_boost_rate: float = 0.75
    weak_boost_rate: float = 1.5
    min_pos_scores_rate: float = 0.5


@dataclass
class ModelConfig:
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    modules: ModulesConfig = field(default_factory=ModulesConfig)
    high_resolution: bool = True
    output_subsampling_factor: int = 1
    streaming_mode: bool = True
    eps: float = 1e-3

    @property
    def upsample_factor(self) -> int:
        return self.encoder.subsampling_factor if self.high_resolution else 1

    def with_streaming(self, preset: Optional[str] = None, **params: int) -> "ModelConfig":
        """Copy with a streaming preset (see `STREAMING_PRESETS`) and/or explicit parameters."""
        values: Dict[str, int] = {}
        if preset is not None:
            if preset not in STREAMING_PRESETS:
                raise ValueError(f"Unknown streaming preset {preset!r}; choose from {sorted(STREAMING_PRESETS)}")
            values.update(STREAMING_PRESETS[preset])
        values.update(params)
        unknown = set(values) - {f.name for f in dataclasses.fields(ModulesConfig)}
        if unknown:
            raise ValueError(f"Unknown streaming parameters: {sorted(unknown)}")
        modules = dataclasses.replace(self.modules, **values)
        _check_streaming(modules)
        return dataclasses.replace(self, modules=modules)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        return cls(
            preprocessor=_build(PreprocessorConfig, d.get("preprocessor", {})),
            encoder=_build(EncoderConfig, d.get("encoder", {})),
            modules=_build(ModulesConfig, d.get("modules", {})),
            **{k: v for k, v in d.items() if k not in ("preprocessor", "encoder", "modules")},
        )

    @classmethod
    def from_hf_config(cls, config: Dict[str, Any], processor: Optional[Dict[str, Any]] = None) -> "ModelConfig":
        """Build from the Transformers `config.json` (+ `processor_config.json`)."""
        if config.get("model_type") != "nemotron3_diarization":
            raise ValueError(f"Unsupported model_type {config.get('model_type')!r} (expected 'nemotron3_diarization')")
        audio, head = config["audio_config"], config["head_config"]
        stream = config.get("streaming_config", {})
        rope = audio.get("rope_parameters", {})
        if float(rope.get("partial_rotary_factor", audio.get("partial_rotary_factor", 1.0))) != 1.0:
            raise ValueError("partial_rotary_factor != 1.0 is not supported")
        if audio.get("hidden_act", "gelu") != "gelu":
            raise ValueError(f"hidden_act={audio.get('hidden_act')!r} is not supported")
        fe = (processor or {}).get("feature_extractor", {})
        sr = int(fe.get("sampling_rate", 16000))
        subsampling = int(audio.get("subsampling_factor", 8))
        return cls(
            preprocessor=PreprocessorConfig(
                sample_rate=sr,
                window_size=int(fe.get("win_length", 400)) / sr,
                window_stride=int(fe.get("hop_length", 160)) / sr,
                n_fft=int(fe.get("n_fft", 512)),
                features=int(audio.get("num_mel_bins", fe.get("feature_size", 128))),
                preemph=float(fe.get("preemphasis", 0.97)),
            ),
            encoder=EncoderConfig(
                feat_in=int(audio.get("num_mel_bins", 128)),
                d_model=int(audio["hidden_size"]),
                n_heads=int(audio["num_attention_heads"]),
                n_layers=int(audio["num_hidden_layers"]),
                subsampling_factor=subsampling,
                ff_expansion=int(audio["intermediate_size"]) / int(audio["hidden_size"]),
                rope_base=float(rope.get("rope_theta", audio.get("rope_theta", 10000.0))),
            ),
            modules=ModulesConfig(
                num_spks=int(head["num_speakers"]),
                fc_d_model=int(head.get("audio_hidden_size", audio["hidden_size"])),
                tf_d_model=int(head["hidden_size"]),
                use_activity_head=False,  # not part of the Transformers export
                use_learnable_sil_emb=True,
                spkcache_len=int(stream.get("speaker_cache_length", 264)),
                fifo_len=int(config.get("fifo_length", 40)),
                chunk_len=int(config.get("chunk_length", 340)),
                spkcache_update_period=int(config.get("speaker_cache_update_period", 300)),
                chunk_left_context=0,
                chunk_right_context=int(config.get("chunk_right_context", 40)),
                spkcache_sil_frames_per_spk=int(stream.get("speaker_cache_silence_frames_per_speaker", 1)),
                pred_score_threshold=float(stream.get("prediction_score_threshold", 0.25)),
                scores_boost_latest=float(stream.get("latest_frames_score_boost", 0.05)),
                strong_boost_rate=float(stream.get("strong_boost_rate", 0.75)),
                weak_boost_rate=float(stream.get("weak_boost_rate", 1.5)),
                min_pos_scores_rate=float(stream.get("min_positive_scores_rate", 0.5)),
            ),
            high_resolution=True,
            output_subsampling_factor=1,
            streaming_mode=True,
        )

    @classmethod
    def from_nemo_config(cls, cfg: Dict[str, Any]) -> "ModelConfig":
        """Build from the (resolved) `model_config.yaml` of a `.nemo` checkpoint."""
        enc = cfg["encoder"]
        target = enc.get("_target_", "")
        if not target.endswith(".TransformerEncoder") or "transformer_encoders" in target:
            raise ValueError(
                f"Unsupported encoder {target!r}: this port targets Nemotron 3 Diarization "
                "(nemo.collections.asr.modules.TransformerEncoder)."
            )
        if cfg.get("transformer_encoder") is not None:
            raise ValueError("A separate `transformer_encoder` is not supported.")
        _expect(enc, "subsampling", "feature_stacking")
        _expect(enc, "self_attention_model", "rope")
        _expect(enc, "attn_mode", "full")
        if float(enc.get("rotary_fraction", 1.0)) != 1.0:
            raise ValueError("rotary_fraction != 1.0 is not supported.")
        if enc.get("feat_out", -1) not in (-1, None, enc["d_model"]):
            raise ValueError("feat_out (encoder output projection) is not supported.")

        pre = cfg["preprocessor"]
        _expect(pre, "normalize", "NA")
        _expect(pre, "frame_splicing", 1)
        _expect(pre, "window", "hann")
        for key, expected in (("log", True), ("mag_power", 2.0), ("exact_pad", False)):
            if pre.get(key, expected) != expected:
                raise ValueError(f"preprocessor.{key}={pre.get(key)!r} is not supported (expected {expected!r}).")

        # Fallbacks are the `SortformerModules` constructor defaults, which NeMo
        # uses when the config omits a parameter.
        mods = cfg["sortformer_modules"]
        num_spks = int(cfg.get("max_num_of_spks", mods.get("num_spks", 4)))
        subsampling = int(enc["subsampling_factor"])
        out_factor = cfg.get("output_subsampling_factor", subsampling)
        if isinstance(out_factor, str):  # unresolved `${...}` interpolation
            out_factor = subsampling

        return cls(
            preprocessor=PreprocessorConfig(
                sample_rate=int(pre.get("sample_rate", 16000)),
                window_size=float(pre.get("window_size", 0.025)),
                window_stride=float(pre.get("window_stride", 0.01)),
                n_fft=int(pre.get("n_fft", 512)),
                features=int(pre.get("features", 128)),
                preemph=float(pre.get("preemph", 0.97)),
                log_zero_guard_value=float(pre.get("log_zero_guard_value", 2**-24)),
            ),
            encoder=EncoderConfig(
                feat_in=int(enc.get("feat_in", pre.get("features", 128))),
                d_model=int(enc["d_model"]),
                n_heads=int(enc["n_heads"]),
                n_layers=int(enc["n_layers"]),
                subsampling_factor=subsampling,
                ff_expansion=float(enc.get("ff_expansion", 4.0)),
                qkv_bias=bool(enc.get("qkv_bias", False)),
                qk_norm=bool(enc.get("qk_norm", False)),
                pre_block_norm=bool(enc.get("pre_block_norm", True)),
                rope_base=float(enc.get("rope_base", 10000.0)),
                xscaling=bool(enc.get("xscaling", False)),
            ),
            modules=ModulesConfig(
                num_spks=num_spks,
                fc_d_model=int(mods["fc_d_model"]),
                tf_d_model=int(mods["tf_d_model"]),
                use_activity_head=float(cfg.get("activity_weight", 0.0)) > 0.0,
                use_learnable_sil_emb=bool(mods.get("use_learnable_sil_emb", False)),
                spkcache_len=int(mods.get("spkcache_len", 188)),
                fifo_len=int(mods.get("fifo_len", 0)),
                chunk_len=int(mods.get("chunk_len", 188)),
                spkcache_update_period=int(mods.get("spkcache_update_period", 188)),
                chunk_left_context=int(mods.get("chunk_left_context", 1)),
                chunk_right_context=int(mods.get("chunk_right_context", 1)),
                spkcache_sil_frames_per_spk=int(mods.get("spkcache_sil_frames_per_spk", 3)),
                pred_score_threshold=float(mods.get("pred_score_threshold", 0.25)),
                max_index=int(mods.get("max_index", 99999)),
                scores_boost_latest=float(mods.get("scores_boost_latest", 0.05)),
                sil_threshold=float(mods.get("sil_threshold", 0.2)),
                strong_boost_rate=float(mods.get("strong_boost_rate", 0.75)),
                weak_boost_rate=float(mods.get("weak_boost_rate", 1.5)),
                min_pos_scores_rate=float(mods.get("min_pos_scores_rate", 0.5)),
            ),
            high_resolution=bool(cfg.get("high_resolution", False)),
            output_subsampling_factor=int(out_factor),
            streaming_mode=bool(cfg.get("streaming_mode", False)),
            eps=float(cfg.get("eps", 1e-3)),
        )


def _check_streaming(m: ModulesConfig) -> None:
    """Mirror of `SortformerModules._check_streaming_parameters` (hard constraints)."""
    minimum = {
        "spkcache_len": (1 + m.spkcache_sil_frames_per_spk) * m.num_spks,
        "fifo_len": 0,
        "chunk_len": 1,
        "spkcache_update_period": 1,
        "chunk_left_context": 0,
        "chunk_right_context": 0,
        "spkcache_sil_frames_per_spk": 0,
    }
    for name, low in minimum.items():
        value = getattr(m, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < low:
            raise ValueError(f"{name} must be an integer >= {low}, got {value!r}")


def _build(klass, values: Dict[str, Any]):
    names = {f.name for f in dataclasses.fields(klass)}
    return klass(**{k: v for k, v in values.items() if k in names})


def _expect(section: Dict[str, Any], key: str, expected: Any) -> None:
    value = section.get(key, expected)
    if value != expected:
        raise ValueError(f"{key}={value!r} is not supported (expected {expected!r}).")
