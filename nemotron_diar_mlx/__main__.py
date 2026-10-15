"""Command line: `python -m nemotron_diar_mlx audio.wav [--json] [--rttm]`.

Audio must be 16 kHz; stereo is averaged to mono. Prints one segment per line
(`start end speaker_N`), RTTM, or JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .config import DEFAULT_PRESET, STREAMING_PRESETS
from .convert import DEFAULT_REPO
from .model import Model


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a PCM WAV file (8/16/24/32-bit int or 32-bit float) as mono float32."""
    try:
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        return audio.mean(axis=1), sr
    except ImportError:
        import wave

        with wave.open(path, "rb") as w:
            sr, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if width == 2:
            audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 4:
            audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise SystemExit(f"{path}: {8 * width}-bit WAV needs `soundfile` (pip install soundfile)")
        return audio.reshape(-1, channels).mean(axis=1), sr


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m nemotron_diar_mlx", description=__doc__)
    parser.add_argument("audio", help="16 kHz WAV file")
    parser.add_argument("--model", default=DEFAULT_REPO, help="MLX dir, Transformers dir, .nemo file or HF repo id")
    parser.add_argument("--preset", default=DEFAULT_PRESET, choices=sorted(STREAMING_PRESETS))
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--cache-dir", help="where converted weights are cached")
    parser.add_argument("--min-duration-on", type=float, default=0.0, help="drop shorter segments (s)")
    parser.add_argument("--min-duration-off", type=float, default=0.0, help="fill shorter gaps (s)")
    out = parser.add_mutually_exclusive_group()
    out.add_argument("--json", action="store_true", help="print JSON")
    out.add_argument("--rttm", action="store_true", help="print RTTM")
    args = parser.parse_args()

    audio, sr = read_wav(args.audio)
    if sr != 16000:
        raise SystemExit(f"{args.audio}: {sr} Hz, resample to 16 kHz first (e.g. ffmpeg -ar 16000 -ac 1)")

    t0 = time.time()
    model = Model.from_pretrained(args.model, dtype=getattr(mx, args.dtype), streaming=args.preset,
                                  cache_dir=args.cache_dir)
    t1 = time.time()
    segments = model.diarize(audio, min_duration_on=args.min_duration_on, min_duration_off=args.min_duration_off)
    t2 = time.time()
    print(f"loaded in {t1 - t0:.1f} s, diarized {len(audio) / sr:.1f} s of audio in {t2 - t1:.1f} s", file=sys.stderr)

    if args.json:
        speakers = sorted({s.speaker for s in segments})
        print(json.dumps({"num_speakers": len(speakers), "segments": [s.to_dict() for s in segments]}))
    elif args.rttm:
        name = Path(args.audio).stem
        for s in segments:
            print(f"SPEAKER {name} 1 {s.start:.3f} {s.end - s.start:.3f} <NA> <NA> speaker_{s.speaker} <NA> <NA>")
    else:
        for s in segments:
            print(f"{s.start:.2f} {s.end:.2f} speaker_{s.speaker}")


if __name__ == "__main__":
    main()
