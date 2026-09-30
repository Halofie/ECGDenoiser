"""Generate fixed (input, expected_output) test vectors for board validation.

Run this on the laptop (plain host Python env, not the Vitis AI Docker --
just needs torch + this repo's dataset code). Produces a set of real,
deterministic ECG windows and the trained float model's output on each,
saved as .npy files.

Why this exists: deploy_zcu104.py's --test-input mode generates a NEW random
array on the board itself, which cannot be reproduced on the laptop for
comparison. These golden vectors are fixed and known ahead of time, so the
same input can be run through the DPU on the board and diffed against a
known-correct expected output -- the only way to actually validate hardware
correctness rather than just "did VART load the xmodel."

The expected output here is the FLOAT model's output, not the quantized
model's. INT8 quantization introduces real, expected error, so don't expect
an exact match -- compare within a tolerance (deploy_zcu104.py's
--golden-dir mode reports MSE/max-abs-diff per window; use those numbers to
judge whether the DPU's output is "close enough", not exact equality).

Usage:
    python generate_golden_vectors.py \
        --checkpoint ../checkpoints/ecg_resunet1d.pt \
        --clean-dir ../data/clean \
        --noise-dir ../mit-bih-noise-stress-test-database-1.0.0 \
        --output-dir ../artifacts/golden_vectors \
        --num-vectors 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset import ECG512Dataset, default_train_val_split
from model import ECGResUNet1D, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate golden test vectors for board validation.")
    parser.add_argument("--checkpoint", default="../checkpoints/ecg_resunet1d.pt", type=Path)
    parser.add_argument("--clean-dir", default="../data/clean")
    parser.add_argument("--noise-dir", default="../mit-bih-noise-stress-test-database-1.0.0")
    parser.add_argument("--output-dir", default="../artifacts/golden_vectors", type=Path)
    parser.add_argument("--num-vectors", type=int, default=5)
    args = parser.parse_args()

    model = ECGResUNet1D()
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval()

    # Use held-out validation records (same split used during training) so
    # these vectors are drawn from patients the model never trained on --
    # a meaningful correctness check, not a memorized-training-window check.
    _, val_records = default_train_val_split()
    dataset = ECG512Dataset(
        clean_dir=args.clean_dir,
        noise_dir=args.noise_dir,
        records=val_records,
        deterministic=True,
        seed=1000,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(dataset) - 1, args.num_vectors, dtype=int)

    with torch.no_grad():
        for vector_id, idx in enumerate(indices):
            noisy, clean = dataset[int(idx)]  # each (1, 512)
            model_input = noisy.unsqueeze(0).unsqueeze(2)  # (1, 1, 1, 512), matches DPU input contract
            expected = model(model_input).squeeze().numpy().astype(np.float32)  # (512,)

            input_flat = noisy.squeeze().numpy().astype(np.float32)  # (512,)
            clean_flat = clean.squeeze().numpy().astype(np.float32)

            np.save(args.output_dir / f"input_{vector_id}.npy", input_flat)
            np.save(args.output_dir / f"expected_{vector_id}.npy", expected)
            np.save(args.output_dir / f"clean_{vector_id}.npy", clean_flat)

    print(f"Wrote {len(indices)} golden vectors to {args.output_dir}")
    print("Each vector: input_i.npy (DPU input, float32[512]), "
          "expected_i.npy (float model output, float32[512]), "
          "clean_i.npy (true clean signal, float32[512], for reference).")


if __name__ == "__main__":
    main()
