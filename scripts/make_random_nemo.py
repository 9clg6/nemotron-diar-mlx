"""Create a random-weight `.nemo` with the Nemotron 3 Diarization architecture.

Lets `check_parity.py` exercise every code path without the real weights:
    python scripts/make_random_nemo.py <NeMo>/examples/speaker_tasks/diarization/conf/neural_diarizer/sortformer_streaming_8spk.yaml random.nemo
"""

import sys

import torch
from nemo.collections.asr.models import SortformerEncLabelModel
from omegaconf import OmegaConf, open_dict


def main() -> None:
    cfg_path, out = sys.argv[1], sys.argv[2]
    cfg = OmegaConf.load(cfg_path)
    with open_dict(cfg):
        for split in ("train_ds", "validation_ds", "test_ds"):
            cfg.model[split].manifest_filepath = None
    torch.manual_seed(0)
    model = SortformerEncLabelModel(cfg=cfg.model)
    # Randomize everything, including what NeMo initializes to zero/identity
    # (upsampler bias, silence embedding), so the comparison exercises every weight.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() > 1:
                p.copy_(torch.randn_like(p) / p[0].numel() ** 0.5)
            elif "norm" in name and name.endswith("weight"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(0.1 * torch.randn_like(p))
    model.eval()
    model.save_to(out)
    print(f"{out}: {sum(p.numel() for p in model.parameters()):,} parameters")


if __name__ == "__main__":
    main()
