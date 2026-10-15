# nemotron-diar-mlx

[NVIDIA Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) on Apple Silicon with
[MLX](https://github.com/ml-explore/mlx): **who spoke when**, up to 8 speakers, in streaming or on recordings of any
length — no PyTorch, no CUDA.

- Faithful port of NeMo's inference: features, RoPE Transformer encoder, Sortformer heads, streaming speaker cache
  (AOSC + FIFO) and segment post-processing — validated against NeMo on the real checkpoint (see
  [Validation](#validation)).
- Loads the official weights straight from Hugging Face (Transformers export) and converts them once, without PyTorch.
- Streaming API that takes audio in blocks of any size, with the latency presets of the model card.
- Speaker attribution for any ASR with word timestamps (e.g. [parakeet-mlx](https://github.com/senstella/parakeet-mlx)).

## Install

```bash
pip install git+https://github.com/9clg6/nemotron-diar-mlx
```

Python 3.10–3.14 on macOS with Apple Silicon (MLX also runs on Linux CPU with `mlx[cpu]`, slowly).

## Quick start

```bash
python -m nemotron_diar_mlx meeting_16k.wav            # start end speaker_N
python -m nemotron_diar_mlx meeting_16k.wav --json     # or --rttm
```

```python
import mlx.core as mx
from nemotron_diar_mlx import Model

model = Model.from_pretrained("nvidia/Nemotron-3-Diarization", dtype=mx.bfloat16)  # downloads + converts once
for seg in model.diarize(audio):   # mono float32 numpy array at 16 kHz
    print(f"{seg.start:.2f}-{seg.end:.2f} speaker_{seg.speaker}")

out = model(audio)
out["preds"]   # (frames, 8) speaker activity probabilities, one frame per 10 ms
```

Speakers are numbered by order of first appearance. The first call downloads `model.safetensors` (~400 MB) from
Hugging Face and caches MLX weights in `~/.cache/nemotron-diar-mlx` (`cache_dir=` or `NEMOTRON_DIAR_MLX_CACHE` to
change it).

## Streaming

Audio is processed in chunks; each chunk is encoded with a cache of the most representative frames of every speaker
heard so far, so memory is constant and speaker numbers stay consistent over hours of audio. The model card's
configurations are available as presets:

| Preset | Input buffer latency | `chunk_len` | `chunk_right_context` | `fifo_len` | `spkcache_update_period` |
|---|---|---|---|---|---|
| `offline` (default) | 30.4 s | 340 | 40 | 40 | 300 |
| `low_latency` | 1.04 s | 9 | 4 | 264 | 222 |
| `very_low_latency` | 0.64 s | 6 | 2 | 264 | 222 |
| `ultra_low_latency` | 0.32 s | 3 | 1 | 264 | 222 |

All values are in 80 ms encoder frames; latency = (chunk + right context) × 80 ms, excluding compute.

```python
from nemotron_diar_mlx import Model, StreamingDiarizer

model = Model.from_pretrained(streaming="low_latency")
diar = StreamingDiarizer(model)
for block in microphone_blocks():        # np.float32 at 16 kHz, any size
    out = diar.feed(block)               # frames finalized since the previous call
    show(out.start_frame, out.preds)
show_last(diar.flush())
```

The concatenated outputs are identical to a single call on the whole signal.

## Who said what

`words_to_turns` gives each time-stamped word to the most active speaker over its span and groups words into turns:

```python
import mlx.core as mx
from parakeet_mlx import from_pretrained
from nemotron_diar_mlx import Model, tokens_to_words, words_to_turns

model = Model.from_pretrained(dtype=mx.bfloat16)
preds = model(audio)["preds"]
asr = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3").transcribe("meeting_16k.wav")
words = tokens_to_words(tok for sentence in asr.sentences for tok in sentence.tokens)
for turn in words_to_turns(words, preds, model.frame_duration):
    print(f"[speaker {turn.speaker}] {turn.text}")
```

## Converting weights explicitly

```bash
python -m nemotron_diar_mlx.convert nvidia/Nemotron-3-Diarization ./nemotron3-diar-mlx --dtype bfloat16
pip install "nemotron-diar-mlx[nemo] @ git+https://github.com/9clg6/nemotron-diar-mlx"
python -m nemotron_diar_mlx.convert Nemotron-3-Diarization.nemo ./nemotron3-diar-mlx   # from the NeMo checkpoint
```

The Transformers export has no auxiliary activity head (silence / one speaker / overlap); converting the `.nemo`
checkpoint keeps it (`out["activity"]`). The `.nemo` file stores everything in bfloat16, including the mel filterbank
and window; the Hugging Face path rebuilds those exactly, as in training.

## Validation

`scripts/check_parity.py` compares this port with NeMo (PyTorch, CPU) on the same `.nemo` checkpoint: the offline path
step by step, speaker-cache compression and the full streaming state chunk after chunk, streaming end to end, and the
final segments of `SortformerEncLabelModel.diarize`.

Real checkpoint, 98 s of real multi-speaker audio, `offline` preset, float32:

| Check | Result |
|---|---|
| Speaker probabilities, offline path | max error 4e-6 |
| Streaming, every chunk started from NeMo's state | max error 9e-6 |
| Streaming end to end, audio fed in irregular blocks | max error 8e-6; 0 of 78,080 speaker/frame decisions differ |
| Speaker-cache compression (incl. exactly tied scores) | bit-identical |
| Segments vs `diarize()` | identical: 29 segments, 6 speakers, 0.000 s boundary shift |
| Transformers weights vs `.nemo` weights | 355/355 tensors bit-identical |

In bfloat16, segment boundaries move by at most 30 ms on the same audio, with the same speakers.

**Score ties.** Cache compression keeps the best-scoring frames with `torch.topk`, whose order among exactly equal
values is unspecified. This port breaks ties deterministically (oldest frame first) and the parity script imposes the
same rule on NeMo. Two implementations can still round a tie differently, which slightly changes later chunks (seen on
synthetic audio with the training-time configuration: 3e-4 on one chunk); NeMo itself is not guaranteed to agree with
itself across CPU and GPU there.

Unit tests (`pytest`) run without NeMo or the weights, on a small random model.

## License

Code: Apache-2.0 (see `LICENSE` and `NOTICE`); the inference logic is a reimplementation of NVIDIA NeMo (Apache-2.0).
Model weights: not included; downloaded from Hugging Face under NVIDIA's
[OpenMDW-1.1](https://huggingface.co/nvidia/Nemotron-3-Diarization) license.
