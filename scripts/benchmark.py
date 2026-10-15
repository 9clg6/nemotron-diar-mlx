"""Speed and memory benchmark on real speech.

    python scripts/benchmark.py [--minutes 10] [--presets offline low_latency] [--dtypes bfloat16 float32]

Downloads the official weights and a short multi-speaker clip (8 voices) from
Hugging Face, loops the clip to the requested duration and reports, for each
preset and dtype, the real-time factor and peak memory. Prints a Markdown
table (also appended to `$GITHUB_STEP_SUMMARY` when set).
"""

from __future__ import annotations

import argparse
import os
import platform
import resource
import subprocess
import time

import mlx.core as mx
import numpy as np

from nemotron_diar_mlx import Model, StreamingDiarizer

CLIP = ("hf-internal-testing/dummy-audio-samples", "diarization_example.mp3")


def load_clip() -> np.ndarray:
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(CLIP[0], CLIP[1], repo_type="dataset")
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly

        g = np.gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
    return audio


def chip() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or platform.machine()


def peak_memory_gb() -> float:
    try:
        return mx.get_peak_memory() / 1e9
    except AttributeError:
        return mx.metal.get_peak_memory() / 1e9


def reset_peak_memory() -> None:
    try:
        mx.reset_peak_memory()
    except AttributeError:
        mx.metal.reset_peak_memory()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--presets", nargs="+", default=["offline", "low_latency"])
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16", "float32"])
    args = parser.parse_args()

    clip = load_clip()
    reps = int(np.ceil(args.minutes * 60 * 16000 / len(clip)))
    audio = np.tile(clip, reps)[: int(args.minutes * 60 * 16000)]
    seconds = len(audio) / 16000

    rows = []
    for dtype in args.dtypes:
        for preset in args.presets:
            t0 = time.perf_counter()
            model = Model.from_pretrained(dtype=getattr(mx, dtype), streaming=preset)
            load = time.perf_counter() - t0
            model(audio[: 16000 * 5])  # warm-up (kernel compilation)
            reset_peak_memory()
            diar = StreamingDiarizer(model)
            block = 16000 // 2  # 0.5 s, like a microphone callback
            worst = 0.0
            t0 = time.perf_counter()
            for i in range(0, len(audio), block):
                t = time.perf_counter()
                diar.feed(audio[i : i + block])
                worst = max(worst, time.perf_counter() - t)
            diar.flush()
            elapsed = time.perf_counter() - t0
            rows.append((preset, dtype, seconds / elapsed, elapsed, worst, peak_memory_gb(), load))
            print(f"{preset:>18} {dtype:>9}: {seconds / elapsed:6.1f}x real time "
                  f"({elapsed:.1f} s for {seconds / 60:g} min), worst feed {worst * 1000:.0f} ms, "
                  f"peak {peak_memory_gb():.2f} GB", flush=True)

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 30 if platform.system() == "Darwin" else 1 << 20)
    lines = [
        f"### nemotron-diar-mlx benchmark — {chip()}, MLX {mx.__version__}, device {mx.default_device()}",
        "",
        f"{seconds / 60:g} min of real multi-speaker speech, fed in 0.5 s blocks.",
        "",
        "| Preset | dtype | Speed | Time | Slowest `feed` | Peak MLX memory | Load (first incl. download) |",
        "|---|---|---|---|---|---|---|",
    ]
    for preset, dtype, rtf, elapsed, worst, peak, load in rows:
        lines.append(f"| `{preset}` | {dtype} | **{rtf:.1f}× real time** | {elapsed:.1f} s | {worst * 1000:.0f} ms "
                     f"| {peak:.2f} GB | {load:.1f} s |")
    lines += ["", f"Process peak RSS: {rss:.2f} GB"]
    report = "\n".join(lines)
    print("\n" + report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
