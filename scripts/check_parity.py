"""Compare the MLX port with NeMo (PyTorch) on the same `.nemo` checkpoint.

    python scripts/check_parity.py model.nemo [audio_16k.wav] [--preset offline]

Requires `nemo_toolkit[asr]`. The checkpoint is converted to a temporary
directory, then the script compares:

- the offline path step by step: mel features, encoder output, speaker
  logits/probabilities, activity logits;
- speaker-cache compression and the full streaming state (cache, FIFO,
  predictions) chunk after chunk, on random inputs (half with duplicated
  frames, i.e. exactly tied scores);
- streaming against `forward_streaming`, in *guided* mode (every chunk starts
  from NeMo's exact state: strict tolerance) and *free* mode (audio pushed in
  irregular blocks, like a microphone; an exact score tie in the cache
  selection may be broken differently there by float rounding);
- incremental vs one-shot features;
- final speaker segments against `SortformerEncLabelModel.diarize`.

Without an audio file a synthetic signal (simulated voices + noise) is used.
Exits non-zero when a check fails.
"""

import argparse
import contextlib
import copy
import sys
import tempfile

import mlx.core as mx
import numpy as np
import torch
from nemo.collections.asr.models import SortformerEncLabelModel

from nemotron_diar_mlx.config import STREAMING_PRESETS
from nemotron_diar_mlx.convert import convert
from nemotron_diar_mlx.features import IncrementalLogMel, log_mel
from nemotron_diar_mlx.model import Model
from nemotron_diar_mlx.segments import speaker_segments
from nemotron_diar_mlx.streaming import StreamingDiarizer, StreamingState, compress_spkcache, streaming_update


@contextlib.contextmanager
def stable_topk():
    """Make `torch.topk` break ties by increasing index, like the port.

    `torch.topk` does not specify which of several equal values it keeps; the
    cache compression meets such ties (identical frames, saturated
    predictions). Both sides use the same rule so everything else can be
    compared.
    """
    original = torch.topk

    def topk(x, k, dim=-1, largest=True, sorted=True, **kwargs):
        if not largest:
            return original(x, k, dim=dim, largest=largest, sorted=sorted, **kwargs)
        values, indices = torch.sort(x, dim=dim, descending=True, stable=True)
        return torch.return_types.topk((values.narrow(dim, 0, k), indices.narrow(dim, 0, k)))

    torch.topk = topk
    try:
        yield
    finally:
        torch.topk = original


def synthetic_audio(seconds: float, sr: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * sr)) / sr
    audio = 0.01 * rng.standard_normal(t.shape)
    # A few "speakers": harmonics at different f0, alternating and sometimes overlapping.
    for i, f0 in enumerate((110.0, 180.0, 240.0)):
        gate = (np.sin(2 * np.pi * (0.2 + 0.07 * i) * t + i) > 0.2).astype(np.float64)
        voice = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
        audio += 0.2 * gate * voice * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    return audio.astype(np.float32)


def load_audio(path: str, sr: int) -> np.ndarray:
    import soundfile as sf

    audio, file_sr = sf.read(path, dtype="float32", always_2d=True)
    if file_sr != sr:
        raise SystemExit(f"{path}: {file_sr} Hz, resample to {sr} Hz first")
    return audio.mean(axis=1)


def report(name: str, ref: np.ndarray, got: np.ndarray, atol: float) -> bool:
    if ref.shape != got.shape:
        print(f"[FAIL] {name}: shapes {ref.shape} (NeMo) != {got.shape} (MLX)")
        return False
    diff = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    scale = np.abs(ref).max() + 1e-12
    ok = diff.max() <= atol
    print(
        f"[{'OK' if ok else 'FAIL'}] {name:<20} shape={ref.shape} "
        f"max err={diff.max():.3e} mean={diff.mean():.3e} (scale {scale:.3e}, tol {atol:.0e})"
    )
    return ok


def apply_preset(ref_model, mlx_model, preset):
    if preset is None:
        return
    for key, value in STREAMING_PRESETS[preset].items():
        setattr(ref_model.sortformer_modules, key, value)
    ref_model._check_streaming_parameters()
    mlx_model.set_streaming(preset)


def check_offline(ref_model, mlx_model, audio: np.ndarray) -> bool:
    """Offline path: `frontend_encoder` + `forward_infer` over the whole signal."""
    print("== Offline (full attention)")
    ok = True
    with torch.no_grad():
        signal = torch.from_numpy(audio)[None]
        length = torch.tensor([signal.shape[1]])
        ref_feats, ref_len = ref_model.process_signal(audio_signal=signal, audio_signal_length=length)
        ref_feats = ref_feats[:, :, : ref_len.max()]
        ref_emb, ref_emb_len = ref_model.frontend_encoder(ref_feats, ref_len)
        ref_preds, ref_logits, ref_activity = ref_model.forward_infer(ref_emb, ref_emb_len, return_logits=True)

    feats, _ = mlx_model.features(mx.array(audio))
    ok &= report("mel features", ref_feats[0].T.numpy(), np.array(feats), atol=1e-3)
    # Feed NeMo's features to isolate the encoder's own error.
    emb, emb_len = mlx_model.encode(mx.array(ref_feats.transpose(1, 2).numpy()), mx.array(ref_len.numpy()))
    ok &= report("encoder output", ref_emb.numpy(), np.array(emb), atol=1e-3)
    preds, logits, activity = mlx_model.forward_infer(emb, emb_len)
    ok &= report("speaker logits", ref_logits.numpy(), np.array(logits), atol=1e-3)
    ok &= report("speaker probs", ref_preds.numpy(), np.array(preds), atol=1e-4)
    if ref_activity is not None:
        ok &= report("activity logits", ref_activity.numpy(), np.array(activity), atol=1e-3)
    full = mlx_model.forward_offline(audio)["preds"]
    ok &= report("end to end", ref_preds[0, : full.shape[0]].numpy(), full, atol=1e-4)
    return ok


def check_compression(ref_model, mlx_model, trials: int = 20) -> bool:
    """`compress_spkcache` vs `SortformerModules._compress_spkcache` on random inputs."""
    print("== Speaker-cache compression")
    mods = ref_model.sortformer_modules
    cfg = mlx_model.config.modules
    sil = np.array(mlx_model.sortformer.learnable_sil_emb) if cfg.use_learnable_sil_emb else None
    rng = np.random.default_rng(1)
    ok = True
    for trial in range(trials):
        n = cfg.spkcache_len + int(rng.integers(1, cfg.chunk_len + cfg.fifo_len + 1))
        # Mix of silence, single speakers and overlaps.
        logits = rng.normal(-2.0, 2.5, size=(n, cfg.num_spks)).astype(np.float32)
        logits[np.arange(n), rng.integers(0, cfg.num_spks, size=n)] += rng.uniform(0, 6, size=n).astype(np.float32)
        preds = 1 / (1 + np.exp(-logits))
        emb = rng.standard_normal((n, cfg.fc_d_model)).astype(np.float32)
        if trial % 2:
            # Duplicated frames (e.g. digital silence): exactly tied scores in both implementations.
            src = rng.integers(0, n, size=n // 3)
            dst = rng.choice(n, size=n // 3, replace=False)
            preds[dst], emb[dst] = preds[src], emb[src]
        mean_sil = rng.standard_normal(cfg.fc_d_model).astype(np.float32)
        with torch.no_grad(), stable_topk():
            ref_emb, ref_preds, _ = mods._compress_spkcache(
                torch.from_numpy(emb)[None], torch.from_numpy(preds)[None], torch.from_numpy(mean_sil)[None]
            )
        got_emb, got_preds = compress_spkcache(emb, preds, sil if sil is not None else mean_sil, cfg)
        same = np.array_equal(ref_emb[0].numpy(), got_emb) and np.array_equal(ref_preds[0].numpy(), got_preds)
        if not same:
            print(f"[FAIL] trial {trial}: compressed cache differs (n={n})")
        ok &= same
    if ok:
        print(f"[OK] {trials} random trials ({trials // 2} with ties): bit-identical cache")
    return ok


def check_streaming_update(ref_model, mlx_model, steps: int = 6) -> bool:
    """Full state (cache, FIFO, predictions) after each `streaming_update`, vs NeMo."""
    print("== Streaming state update")
    mods = ref_model.sortformer_modules
    cfg = mlx_model.config.modules
    sil = np.array(mlx_model.sortformer.learnable_sil_emb) if cfg.use_learnable_sil_emb else None
    rng = np.random.default_rng(3)
    ref_state = mods.init_streaming_state(batch_size=1, device="cpu")
    state = StreamingState.empty(cfg.fc_d_model, cfg.num_spks)
    ok = True
    for step in range(steps):
        n_chunk = cfg.chunk_len if step < steps - 1 else int(rng.integers(1, cfg.chunk_len + 1))
        total = state.spkcache.shape[0] + state.fifo.shape[0] + n_chunk
        logits = rng.normal(-2.0, 2.5, size=(total, cfg.num_spks)).astype(np.float32)
        logits[np.arange(total), rng.integers(0, cfg.num_spks, size=total)] += 5.0
        preds = 1 / (1 + np.exp(-logits))
        chunk = rng.standard_normal((n_chunk, cfg.fc_d_model)).astype(np.float32)
        with torch.no_grad(), stable_topk():
            ref_state, ref_chunk_preds = mods.streaming_update(
                ref_state, torch.from_numpy(chunk)[None], torch.from_numpy(preds)[None]
            )
        chunk_preds = streaming_update(state, chunk, preds, cfg, sil)
        pairs = {
            "chunk_preds": (ref_chunk_preds, chunk_preds),
            "spkcache": (ref_state.spkcache, state.spkcache),
            "spkcache_preds": (ref_state.spkcache_preds, state.spkcache_preds),
            "fifo": (ref_state.fifo, state.fifo),
            "fifo_preds": (ref_state.fifo_preds, state.fifo_preds),
            "mean_sil_emb": (ref_state.mean_sil_emb, state.mean_sil_emb[None]),
        }
        for name, (ref, got) in pairs.items():
            ref = None if ref is None else ref[0].numpy() if name != "mean_sil_emb" else ref.numpy()
            if ref is None or got is None:
                same = ref is None and got is None
            else:
                same = ref.shape == got.shape and np.allclose(ref, got, atol=1e-6)
            if not same:
                print(f"[FAIL] chunk {step}: {name} differs")
            ok &= same
    if ok:
        print(f"[OK] {steps} chunks: cache, FIFO and predictions match")
    return ok


def _to_mlx_state(st) -> StreamingState:
    def arr(t):
        return None if t is None else t[0].detach().numpy().astype(np.float32).copy()

    fifo_preds = arr(st.fifo_preds)
    return StreamingState(
        spkcache=arr(st.spkcache),
        spkcache_preds=arr(st.spkcache_preds),
        fifo=arr(st.fifo),
        fifo_preds=fifo_preds if fifo_preds is not None else np.zeros((0, st.mean_sil_emb.shape[1]), np.float32),
        mean_sil_emb=arr(st.mean_sil_emb),
        n_sil_frames=int(st.n_sil_frames[0]),
        spkcache_compressed=bool(st.spkcache_compressed),
    )


class _GuidedDiarizer(StreamingDiarizer):
    """Restart every chunk from NeMo's state: isolates the error of each step."""

    def __init__(self, model, states):
        super().__init__(model)
        self._states = list(states)

    def _step(self, end, right):
        self.state = _to_mlx_state(self._states.pop(0))
        return super()._step(end, right)


def check_streaming(ref_model, mlx_model, audio: np.ndarray) -> bool:
    """`StreamingDiarizer` vs NeMo's `forward_streaming`."""
    mods = mlx_model.config.modules
    print(
        f"== Streaming ({len(audio) / 16000:.0f} s audio; chunk {mods.chunk_len}, fifo {mods.fifo_len}, "
        f"right context {mods.chunk_right_context}, cache {mods.spkcache_len})"
    )
    ok = True
    # NeMo reference, recording the state before each chunk and each chunk's output range.
    states, bounds = [], []
    step_fn = ref_model.forward_streaming_step

    def spy_step(*args, **kwargs):
        states.append(copy.deepcopy(kwargs["streaming_state"]))
        before = kwargs["total_preds"].shape[1]
        out = step_fn(*args, **kwargs)
        bounds.append((before, out[1].shape[1]))
        return out

    with torch.no_grad():
        signal = torch.from_numpy(audio)[None]
        length = torch.tensor([signal.shape[1]])
        ref_feats, ref_len = ref_model.preprocessor(input_signal=signal, length=length)
        ref_feats = ref_feats[:, :, : ref_len.max()]
        ref_model.forward_streaming_step = spy_step
        try:
            with stable_topk():
                ref_preds, _, ref_activity = ref_model.forward_streaming(ref_feats, ref_len, return_logits=True)
        finally:
            del ref_model.forward_streaming_step
    ref_preds = ref_preds[0].numpy()
    ref_act = torch.softmax(ref_activity[0], dim=-1).numpy() if ref_activity is not None else None

    # 1) Guided: every chunk starts from NeMo's exact state -> strict tolerance.
    guided = _GuidedDiarizer(mlx_model, states)
    g_out = [guided.feed(audio), guided.flush()]
    ok &= report(f"guided ({len(states)} chunks)", ref_preds, np.concatenate([o.preds for o in g_out]), atol=1e-4)
    if ref_act is not None:
        ok &= report("guided activity", ref_act, np.concatenate([o.activity for o in g_out]), atol=1e-4)

    # 2) Free, audio pushed in irregular blocks like a microphone.
    rng = np.random.default_rng(2)
    diarizer = StreamingDiarizer(mlx_model)
    pos, outs = 0, []
    while pos < len(audio):
        step = int(rng.integers(100, 24000))
        outs.append(diarizer.feed(audio[pos : pos + step]))
        pos += step
    outs.append(diarizer.flush())
    preds = np.concatenate([o.preds for o in outs])
    contiguous = all(o.start_frame == sum(len(p.preds) for p in outs[:i]) for i, o in enumerate(outs))
    print(f"[{'OK' if contiguous else 'FAIL'}] {len(outs)} feed/flush calls, contiguous outputs")
    ok &= contiguous
    if preds.shape != ref_preds.shape:
        print(f"[FAIL] free: shapes {ref_preds.shape} != {preds.shape}")
        return False
    per_chunk = [np.abs(ref_preds[a:b] - preds[a:b]).max() for a, b in bounds]
    first = next((i for i, d in enumerate(per_chunk) if d > 1e-4), None)
    diff = np.abs(ref_preds - preds)
    note = "none" if first is None else f"from chunk {first}"
    print(f"[INFO] free                 max err={diff.max():.3e} mean={diff.mean():.3e}; divergence > 1e-4: {note}")
    if first is not None:
        print("       (guided mode being exact, it comes from a cache-selection score tie broken by float rounding)")
    ok &= report("free (tolerance)", ref_preds, preds, atol=5e-2)
    flips = int(((ref_preds > 0.5) != (preds > 0.5)).sum())
    print(f"[INFO] free: {flips} of {preds.size} speaker/frame decisions (p > 0.5) differ")

    # Incremental features must equal the one-shot ones.
    inc = IncrementalLogMel(mlx_model.preprocessor_window, mlx_model.preprocessor_fb, mlx_model.config.preprocessor)
    pos, parts = 0, []
    while pos < len(audio):
        step = int(rng.integers(1, 9000))
        parts.append(np.array(inc.push(audio[pos : pos + step])))
        pos += step
    parts.append(np.array(inc.flush()))
    whole, _ = log_mel(
        mx.array(audio), mlx_model.preprocessor_window, mlx_model.preprocessor_fb, mlx_model.config.preprocessor
    )
    ok &= report("incremental features", np.array(whole), np.concatenate(parts), atol=1e-4)
    return ok


def check_segments(ref_model, mlx_model, audio: np.ndarray) -> bool:
    """Final segments vs `SortformerEncLabelModel.diarize` (default post-processing)."""
    print("== Segments (diarize)")
    with torch.no_grad(), stable_topk():
        ref_lines = ref_model.diarize(audio=[audio], batch_size=1, sample_rate=16000, verbose=False)[0]
    ref = sorted((float(a), float(b), int(s.split("_")[-1])) for a, b, s in (line.split() for line in ref_lines))
    got = sorted((s.start, s.end, s.speaker) for s in speaker_segments(mlx_model(audio)["preds"], mlx_model.frame_duration))
    print(f"[INFO] NeMo: {len(ref)} segments, {len({s for *_, s in ref})} speakers; "
          f"MLX: {len(got)} segments, {len({s for *_, s in got})} speakers")
    same_count = len(ref) == len(got)
    max_shift = max((max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(ref, got)), default=0.0)
    same_speakers = all(a[2] == b[2] for a, b in zip(ref, got))
    ok = same_count and same_speakers and max_shift <= 0.02 + 1e-9
    print(f"[{'OK' if ok else 'FAIL'}] same segments and speakers, max boundary shift {max_shift:.3f} s (tol 0.02 s)")
    for line in ref_lines[:5]:
        print(f"       NeMo: {line}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("nemo")
    parser.add_argument("audio", nargs="?")
    parser.add_argument("--seconds", type=float, default=20.0, help="offline check duration")
    parser.add_argument("--streaming-seconds", type=float, default=95.0, help="streaming check duration (synthetic)")
    parser.add_argument("--preset", choices=sorted(STREAMING_PRESETS), help="streaming preset (default: checkpoint's)")
    args = parser.parse_args()

    ref_model = SortformerEncLabelModel.restore_from(args.nemo, map_location="cpu")
    ref_model.eval()
    sr = ref_model.cfg.sample_rate
    if args.audio:
        long_audio = load_audio(args.audio, sr)
        short_audio = long_audio[: int(args.seconds * sr)]
    else:
        short_audio = synthetic_audio(args.seconds, sr)
        long_audio = synthetic_audio(args.streaming_seconds, sr)

    with tempfile.TemporaryDirectory() as tmp:
        convert(args.nemo, tmp)
        mlx_model = Model.from_pretrained(tmp, streaming=None)
    apply_preset(ref_model, mlx_model, args.preset)

    ok = check_offline(ref_model, mlx_model, short_audio)
    ok &= check_compression(ref_model, mlx_model)
    ok &= check_streaming_update(ref_model, mlx_model)
    ok &= check_streaming(ref_model, mlx_model, long_audio)
    if mlx_model.config.streaming_mode:
        ok &= check_segments(ref_model, mlx_model, long_audio)
    print(f"Output: {mlx_model.frame_duration * 1000:.0f} ms frames, {mlx_model.config.modules.num_spks} speakers")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
