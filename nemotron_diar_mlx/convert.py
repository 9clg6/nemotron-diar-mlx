"""Convert Nemotron 3 Diarization weights to MLX.

Two sources are supported:

- the Hugging Face Transformers export (`config.json`, `processor_config.json`,
  `model.safetensors`), as a repo id such as `nvidia/Nemotron-3-Diarization`
  or a local copy. No PyTorch needed. The window and mel filterbank are
  rebuilt exactly; the auxiliary activity head is not part of this export.
- the NeMo `.nemo` checkpoint. Needs `torch` and `pyyaml`. Keeps the activity
  head and the checkpoint's own (bfloat16-rounded) preprocessor buffers, which
  makes outputs bit-comparable with NeMo.

Usage:
    python -m nemotron_diar_mlx.convert nvidia/Nemotron-3-Diarization out_dir [--dtype bfloat16]
    python -m nemotron_diar_mlx.convert Nemotron-3-Diarization.nemo out_dir

The output directory holds `config.json` + `model.safetensors`, loadable with
`Model.from_pretrained(out_dir)`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

from .config import ModelConfig
from .features import hann_window, mel_filterbank

DEFAULT_REPO = "nvidia/Nemotron-3-Diarization"
HF_FILES = ("config.json", "processor_config.json", "model.safetensors")
_MLX_FORMAT = "nemotron-diar-mlx"

# --------------------------------------------------------------------------- NeMo checkpoint

_NEMO_IGNORED = (re.compile(r"^sortformer_modules\.hidden_to_spks\."),)  # legacy head, frozen and unused
_NEMO_RENAMES: Tuple[Tuple[re.Pattern, str], ...] = tuple(
    (re.compile(p), r)
    for p, r in (
        (r"^preprocessor\.featurizer\.window$", "preprocessor_window"),
        (r"^preprocessor\.featurizer\.fb$", "preprocessor_fb"),
        (r"^encoder\.layers\.(\d+)\.ffn\.net\.0\.", r"encoder.layers.\1.ffn.fc1."),
        (r"^encoder\.layers\.(\d+)\.ffn\.net\.3\.", r"encoder.layers.\1.ffn.fc2."),
        (r"^encoder\.", "encoder."),
        (r"^sortformer_modules\.activity_head\.0\.", "sortformer.activity_norm."),
        (r"^sortformer_modules\.activity_head\.1\.", "sortformer.activity_proj."),
        (r"^sortformer_modules\.", "sortformer."),
    )
)


def read_nemo(path: Union[str, Path]) -> Tuple[dict, Dict[str, np.ndarray]]:
    """Read `model_config.yaml` and `model_weights.ckpt` from a `.nemo` archive."""
    import torch  # conversion-only dependency
    import yaml

    with tarfile.open(path, "r:*") as tar:
        members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
        for required in ("model_config.yaml", "model_weights.ckpt"):
            if required not in members:
                raise ValueError(f"{path}: {required} not found in the .nemo archive")
        config = yaml.safe_load(tar.extractfile(members["model_config.yaml"]).read())
        state = torch.load(tar.extractfile(members["model_weights.ckpt"]), map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    return config, {k: v.detach().to(torch.float32).numpy() for k, v in state.items()}


def map_nemo_weights(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Rename / reorder NeMo tensors for the MLX module."""
    out: Dict[str, np.ndarray] = {}
    unknown = []
    for name, value in state.items():
        if any(p.search(name) for p in _NEMO_IGNORED):
            continue
        new = next((p.sub(r, name, count=1) for p, r in _NEMO_RENAMES if p.search(name)), None)
        if new is None:
            unknown.append(name)
            continue
        if new == "preprocessor_fb":
            value = value.reshape(value.shape[-2], value.shape[-1])  # (1, n_mels, F) -> (n_mels, F)
        elif new == "sortformer.subpixel_upsample.weight":
            value = value.transpose(0, 2, 1)  # PyTorch Conv1d (O, I, K) -> MLX (O, K, I)
        out[new] = value
    if unknown:
        raise ValueError("Unrecognized NeMo tensors (unexpected architecture?):\n  " + "\n  ".join(unknown))
    return out


# --------------------------------------------------------------------------- Transformers export

_HF_RENAMES: Tuple[Tuple[re.Pattern, str], ...] = tuple(
    (re.compile(p), r)
    for p, r in (
        (r"^model\.audio_tower\.embedder\.projection\.", "encoder.pre_encode.proj."),
        (r"^model\.audio_tower\.input_layer_norm\.", "encoder.embed_norm."),
        (r"^model\.audio_tower\.layer_norm\.", "encoder.final_norm."),
        (r"^model\.audio_tower\.layers\.(\d+)\.layer_norm1\.", r"encoder.layers.\1.norm1."),
        (r"^model\.audio_tower\.layers\.(\d+)\.layer_norm2\.", r"encoder.layers.\1.norm2."),
        (r"^model\.audio_tower\.layers\.(\d+)\.mlp\.", r"encoder.layers.\1.ffn."),
        (r"^model\.audio_tower\.layers\.(\d+)\.self_attn\.o_proj\.", r"encoder.layers.\1.attn.out_proj."),
        (r"^model\.proj\.", "sortformer.encoder_proj."),
        (r"^model\.upsampler\.conv\.", "sortformer.subpixel_upsample."),
        (r"^classifier\.dense\.", "sortformer.first_hidden_to_hidden."),
        (r"^classifier\.out_proj\.", "sortformer.single_hidden_to_spks."),
        (r"^silence_embeds$", "sortformer.learnable_sil_emb"),
    )
)
_HF_QKV = re.compile(r"^model\.audio_tower\.layers\.(\d+)\.self_attn\.([qkv])_proj\.(weight|bias)$")


def map_hf_weights(state: Dict[str, np.ndarray], config: ModelConfig) -> Dict[str, np.ndarray]:
    """Rename Transformers tensors; fuse q/k/v; add the preprocessor buffers."""
    out: Dict[str, np.ndarray] = {}
    qkv: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    unknown = []
    for name, value in state.items():
        m = _HF_QKV.match(name)
        if m:
            qkv.setdefault((m.group(1), m.group(3)), {})[m.group(2)] = value
            continue
        new = next((p.sub(r, name, count=1) for p, r in _HF_RENAMES if p.search(name)), None)
        if new is None:
            unknown.append(name)
            continue
        if new == "sortformer.subpixel_upsample.weight":
            value = value.transpose(0, 2, 1)
        out[new] = value
    if unknown:
        raise ValueError("Unrecognized Transformers tensors (unexpected architecture?):\n  " + "\n  ".join(unknown))
    for (layer, kind), parts in qkv.items():
        if set(parts) != {"q", "k", "v"}:
            raise ValueError(f"Layer {layer}: incomplete q/k/v {kind} ({sorted(parts)})")
        # NeMo's fused projection is laid out as [q | k | v].
        out[f"encoder.layers.{layer}.attn.w_qkv.{kind}"] = np.concatenate([parts["q"], parts["k"], parts["v"]])
    pre = config.preprocessor
    out["preprocessor_window"] = hann_window(pre.win_length)
    out["preprocessor_fb"] = mel_filterbank(pre.sample_rate, pre.n_fft, pre.features)
    return out


def read_hf(path: Union[str, Path]) -> Tuple[ModelConfig, Dict[str, np.ndarray]]:
    path = Path(path)
    hf_config = json.loads((path / "config.json").read_text())
    processor_file = path / "processor_config.json"
    processor = json.loads(processor_file.read_text()) if processor_file.exists() else None
    config = ModelConfig.from_hf_config(hf_config, processor)
    state = {k: np.array(v.astype(mx.float32)) for k, v in mx.load(str(path / "model.safetensors")).items()}
    return config, map_hf_weights(state, config)


# --------------------------------------------------------------------------- conversion / resolution


def convert(source: Union[str, Path], out_dir: Union[str, Path], dtype: str = "float32") -> Path:
    """Convert a `.nemo` file or a Transformers directory to an MLX directory."""
    source = Path(source)
    if source.suffix == ".nemo":
        config_dict, state = read_nemo(source)
        config = ModelConfig.from_nemo_config(config_dict)
        weights = map_nemo_weights(state)
    else:
        config, weights = read_hf(source)
    _check_against_model(config, weights)

    target = getattr(mx, dtype)
    arrays = {
        k: mx.array(v) if k.startswith("preprocessor_") else mx.array(v).astype(target) for k, v in weights.items()
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "model.partial.safetensors"  # mx.save_safetensors keeps this suffix
    mx.save_safetensors(str(tmp), arrays, metadata={"format": _MLX_FORMAT})
    (out_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    os.replace(tmp, out_dir / "model.safetensors")  # atomic: a complete file marks a usable cache
    return out_dir


def resolve(source: Union[str, Path] = DEFAULT_REPO, cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """Return a directory of MLX weights for `source`, converting (and downloading) once.

    `source` may be an MLX directory, a `.nemo` file, a local Transformers
    directory or a Hugging Face repo id (optionally `repo@revision`).
    """
    cache = Path(cache_dir or os.environ.get("NEMOTRON_DIAR_MLX_CACHE", Path.home() / ".cache" / "nemotron-diar-mlx"))
    path = Path(source)
    if path.is_dir() and _is_mlx_dir(path):
        return path
    if path.is_file() and path.suffix == ".nemo":
        stat = path.stat()
        key = hashlib.sha1(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:12]
        out = cache / f"{path.stem}-{key}"
        return out if _is_mlx_dir(out) else convert(path, out)
    if path.is_dir():
        key = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]
        out = cache / f"{path.resolve().name}-{key}"
        return out if _is_mlx_dir(out) else convert(path, out)

    repo, _, revision = str(source).partition("@")
    out = cache / (repo.replace("/", "--") + (f"@{revision}" if revision else ""))
    if _is_mlx_dir(out):
        return out
    from huggingface_hub import snapshot_download

    hf_dir = snapshot_download(repo, revision=revision or None, allow_patterns=list(HF_FILES))
    return convert(hf_dir, out)


def _is_mlx_dir(path: Path) -> bool:
    weights, config = path / "model.safetensors", path / "config.json"
    if not (weights.exists() and config.exists()):
        return False
    try:
        return "preprocessor" in json.loads(config.read_text())
    except (OSError, ValueError):
        return False


def _check_against_model(config: ModelConfig, weights: Dict[str, np.ndarray]) -> None:
    """Names and shapes must match the MLX module exactly."""
    from mlx.utils import tree_flatten

    from .model import Model

    expected = dict(tree_flatten(Model(config).parameters()))
    missing = sorted(set(expected) - set(weights))
    extra = sorted(set(weights) - set(expected))
    bad_shapes = [
        f"{k}: checkpoint {weights[k].shape} != model {tuple(expected[k].shape)}"
        for k in sorted(set(expected) & set(weights))
        if tuple(weights[k].shape) != tuple(expected[k].shape)
    ]
    problems = _section("missing", missing) + _section("unexpected", extra) + _section("shapes", bad_shapes)
    if problems:
        raise ValueError("Weights do not match the MLX model:" + problems)


def _section(title: str, items: Iterable[str]) -> str:
    items = list(items)
    return f"\n {title}:\n  " + "\n  ".join(items) if items else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", default=DEFAULT_REPO, help=".nemo file, Transformers dir or HF repo id")
    parser.add_argument("out_dir", help="MLX output directory")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    args = parser.parse_args()
    source = args.source
    if not Path(source).exists():
        from huggingface_hub import snapshot_download

        repo, _, revision = source.partition("@")
        source = snapshot_download(repo, revision=revision or None, allow_patterns=list(HF_FILES))
    print(f"MLX weights written to {convert(source, args.out_dir, args.dtype)}")


if __name__ == "__main__":
    main()
