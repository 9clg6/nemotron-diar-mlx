"""Nemotron 3 Diarization in MLX: Transformer encoder + Sortformer heads.

Mapping to NeMo:
- `TransformerEncoder` <- `nemo.collections.asr.modules.transformer_encoder.TransformerEncoder`
  (feature stacking x8, LayerNorm, N pre-norm RoPE/GELU blocks, final LayerNorm)
- `SortformerHeads`    <- `nemo.collections.asr.modules.sortformer_modules.SortformerModules`
  (encoder_proj, sub-pixel upsampling, speaker and activity heads)
- `Model.forward_offline` <- `SortformerEncLabelModel.frontend_encoder` + `forward_infer`

Streaming inference (speaker cache) lives in `streaming.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import DEFAULT_PRESET, EncoderConfig, ModelConfig, ModulesConfig
from .features import log_mel

if TYPE_CHECKING:
    from .segments import Segment


class FeatureStacking(nn.Module):
    """Stack `factor` consecutive frames, then project to `d_model`."""

    def __init__(self, factor: int, feat_in: int, d_model: int):
        super().__init__()
        self.factor = factor
        self.proj = nn.Linear(factor * feat_in, d_model, bias=False)

    def __call__(self, x: mx.array, lengths: mx.array) -> Tuple[mx.array, mx.array]:
        b, t, c = x.shape
        pad = (self.factor - t % self.factor) % self.factor
        if pad:
            x = mx.pad(x, [(0, 0), (0, pad), (0, 0)])
        x = x.reshape(b, (t + pad) // self.factor, c * self.factor)
        return self.proj(x), (lengths + self.factor - 1) // self.factor


class Attention(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_base = cfg.rope_base
        self.w_qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.qkv_bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        if cfg.qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        self.qk_norm = cfg.qk_norm

    def __call__(self, x: mx.array, mask: Optional[mx.array]) -> mx.array:
        b, t, d = x.shape
        qkv = self.w_qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # "rotate_half" (non-traditional) RoPE, as NeMo's RotaryPositionalEncoding.
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=0)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=0)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(b, t, d))


class FeedForward(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.ff_hidden)
        self.fc2 = nn.Linear(cfg.ff_hidden, cfg.d_model)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def __call__(self, x: mx.array, mask: Optional[mx.array]) -> mx.array:
        x = x + self.attn(self.norm1(x), mask)
        return x + self.ffn(self.norm2(x))


class TransformerEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.pre_encode = FeatureStacking(cfg.subsampling_factor, cfg.feat_in, cfg.d_model)
        self.embed_norm = nn.LayerNorm(cfg.d_model) if cfg.pre_block_norm else None
        self.layers = [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        self.final_norm = nn.LayerNorm(cfg.d_model)

    def __call__(
        self, features: mx.array, lengths: mx.array, bypass_pre_encode: bool = False
    ) -> Tuple[mx.array, mx.array]:
        """`features`: `(B, T, feat_in)`, or `(B, T', d_model)` with `bypass_pre_encode`."""
        if bypass_pre_encode:
            x = features
        else:
            x, lengths = self.pre_encode(features, lengths)
        if self.cfg.xscaling:
            x = x * (self.cfg.d_model**0.5)
        if self.embed_norm is not None:
            x = self.embed_norm(x)
        mask = _key_padding_mask(lengths, x.shape[1])
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_norm(x), lengths


class SortformerHeads(nn.Module):
    def __init__(self, cfg: ModulesConfig, upsample_factor: int):
        super().__init__()
        self.upsample_factor = upsample_factor
        h = cfg.tf_d_model
        self.encoder_proj = nn.Linear(cfg.fc_d_model, h)
        if upsample_factor > 1:
            self.subpixel_upsample = nn.Conv1d(h, h * upsample_factor, kernel_size=3, padding=1)
        self.first_hidden_to_hidden = nn.Linear(h, h)
        self.single_hidden_to_spks = nn.Linear(h, cfg.num_spks)
        if cfg.use_activity_head:
            self.activity_norm = nn.LayerNorm(h)
            self.activity_proj = nn.Linear(h, 3)
        if cfg.use_learnable_sil_emb:
            # Used by the streaming speaker cache.
            self.learnable_sil_emb = mx.zeros((cfg.fc_d_model,))
        self.use_activity_head = cfg.use_activity_head

    def upsample(self, x: mx.array) -> mx.array:
        if self.upsample_factor == 1:
            return x
        b, t, h = x.shape
        y = self.subpixel_upsample(x)  # (B, T, H * u), channel = u_idx * H + h
        return y.reshape(b, t * self.upsample_factor, h)

    def speaker_logits(self, x: mx.array) -> mx.array:
        x = self.first_hidden_to_hidden(nn.relu(x))
        return self.single_hidden_to_spks(nn.relu(x))

    def activity_logits(self, x: mx.array) -> Optional[mx.array]:
        if not self.use_activity_head:
            return None
        return self.activity_proj(self.activity_norm(x))


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        pre = config.preprocessor
        self.encoder = TransformerEncoder(config.encoder)
        self.sortformer = SortformerHeads(config.modules, config.upsample_factor)
        # Preprocessor buffers (analysis window, mel filterbank).
        self.preprocessor_window = mx.zeros((pre.win_length,))
        self.preprocessor_fb = mx.zeros((pre.features, pre.n_fft // 2 + 1))

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_pretrained(
        cls,
        source: Union[str, Path] = "nvidia/Nemotron-3-Diarization",
        dtype: mx.Dtype = mx.float32,
        streaming: Optional[str] = DEFAULT_PRESET,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> "Model":
        """Load a model.

        Args:
            source: a directory converted by `nemotron_diar_mlx.convert`, a
                Hugging Face repo id or local copy (Transformers format), or a
                `.nemo` file. Non-MLX sources are converted once and cached.
            dtype: compute dtype (`mx.float32`, `mx.float16` or `mx.bfloat16`).
            streaming: streaming preset (see `STREAMING_PRESETS`); `None`
                keeps the parameters stored with the weights.
            cache_dir: where converted weights are cached.
        """
        from .convert import resolve

        path = resolve(source, cache_dir=cache_dir)
        config = ModelConfig.from_dict(json.loads((path / "config.json").read_text()))
        if streaming is not None:
            config = config.with_streaming(streaming)
        model = cls(config)
        weights = mx.load(str(path / "model.safetensors"))
        model.load_weights(list(weights.items()), strict=True)
        if dtype != mx.float32:
            # Preprocessor buffers stay in float32.
            model.encoder.set_dtype(dtype)
            model.sortformer.set_dtype(dtype)
        mx.eval(model.parameters())
        model.eval()
        return model

    def set_streaming(self, preset: Optional[str] = None, **params: int) -> "Model":
        """Change the streaming parameters (preset and/or explicit values)."""
        self.config = self.config.with_streaming(preset, **params)
        return self

    # ------------------------------------------------------------------ inference

    def features(self, waveform: mx.array) -> Tuple[mx.array, int]:
        """Features `(T, n_mels)` of a mono signal, normalized like NeMo's `process_signal`."""
        if not self.config.streaming_mode:
            waveform = waveform / (mx.max(waveform) + self.config.eps)
        return log_mel(waveform, self.preprocessor_window, self.preprocessor_fb, self.config.preprocessor)

    def encode(self, features: mx.array, lengths: mx.array) -> Tuple[mx.array, mx.array]:
        """`frontend_encoder`: features `(B, T, n_mels)` -> embeddings `(B, T/8, tf_d_model)`."""
        emb, emb_lengths = self.encoder(features.astype(self._dtype), lengths)
        return self.sortformer.encoder_proj(emb), emb_lengths

    def forward_infer(
        self, emb: mx.array, emb_lengths: mx.array
    ) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
        """Probabilities `(B, T*u, num_spks)`, logits and activity logits."""
        hidden = self.sortformer.upsample(emb)
        mask = _length_mask(emb_lengths * self.config.upsample_factor, hidden.shape[1])
        logits = self.sortformer.speaker_logits(hidden)
        preds = mx.sigmoid(logits) * mask[..., None]
        return preds, logits, self.sortformer.activity_logits(hidden)

    def __call__(self, waveform) -> Dict[str, np.ndarray]:
        """Diarize a mono 16 kHz float32 signal.

        Like `SortformerEncLabelModel.forward`: chunked streaming when the
        config says so (Nemotron 3), otherwise full attention over the signal.
        Returns `preds` `(T_out, num_spks)`, the activity probability of each
        speaker every `frame_duration` seconds (speakers are numbered by order
        of first appearance), and `activity` `(T_out, 3)` (silence / one
        speaker / overlap) when the model has that head.
        """
        if self.config.streaming_mode:
            return self.forward_streaming(waveform)
        return self.forward_offline(waveform)

    def diarize(self, waveform, **postprocessing) -> List["Segment"]:
        """Speaker segments of a mono 16 kHz signal (see `segments.speaker_segments`)."""
        from .segments import speaker_segments

        return speaker_segments(self(waveform)["preds"], self.frame_duration, **postprocessing)

    def forward_streaming(self, waveform) -> Dict[str, np.ndarray]:
        """Chunked streaming with the speaker cache: constant memory."""
        from .streaming import StreamingDiarizer

        diarizer = StreamingDiarizer(self)
        first = diarizer.feed(np.asarray(waveform, dtype=np.float32))
        last = diarizer.flush()
        out = {"preds": np.concatenate([first.preds, last.preds])}
        if first.activity is not None:
            out["activity"] = np.concatenate([first.activity, last.activity])
        return out

    def forward_offline(self, waveform) -> Dict[str, np.ndarray]:
        """Full attention over the whole signal: quadratic cost, short clips only."""
        feats, n_valid = self.features(mx.array(np.asarray(waveform, dtype=np.float32)))
        lengths = mx.array([n_valid])
        emb, emb_lengths = self.encode(feats[None], lengths)
        preds, _, activity = self.forward_infer(emb, emb_lengths)

        frame_factor = 1 if self.config.high_resolution else self.config.encoder.subsampling_factor
        if self.config.output_subsampling_factor // frame_factor > 1:
            raise NotImplementedError("output_subsampling_factor above the native resolution is not supported.")
        n_out = min(preds.shape[1], -(-n_valid // frame_factor))
        out = {"preds": np.array(preds[0, :n_out].astype(mx.float32))}
        if activity is not None:
            out["activity"] = np.array(mx.softmax(activity[0, :n_out].astype(mx.float32), axis=-1))
        return out

    @property
    def frame_duration(self) -> float:
        """Duration (s) of one output frame."""
        return self.config.output_subsampling_factor * self.config.preprocessor.window_stride

    @property
    def _dtype(self) -> mx.Dtype:
        return self.sortformer.encoder_proj.weight.dtype


def _length_mask(lengths: mx.array, max_len: int) -> mx.array:
    return (mx.arange(max_len)[None, :] < lengths[:, None]).astype(mx.float32)


def _key_padding_mask(lengths: mx.array, max_len: int) -> Optional[mx.array]:
    """Boolean key mask `(B, 1, 1, T)`; `None` when nothing is padded."""
    if lengths.shape[0] == 1 and int(lengths[0]) == max_len:
        return None
    return (mx.arange(max_len)[None, :] < lengths[:, None])[:, None, None, :]
