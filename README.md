# ECG Denoiser FPGA Project

This repository is being developed from the PRD in [PRD-ECG-Denoising-Autoencoder-FPGA.md](PRD-ECG-Denoising-Autoencoder-FPGA.md).

Current milestone: Phase 1 — project structure and environment setup.

## Scope of Phase 1
- Establish a reproducible Python project structure.
- Define the package layout for future dataset, preprocessing, model, training, and FPGA export work.
- Provide a baseline configuration file and project smoke tests.
- Do not yet implement the ECG denoising pipeline itself.

## Repository layout
- `src/` — implementation code for each project stage.
- `scripts/` — runnable project entry points.
- `tests/` — project tests.
- `configs/` — YAML configuration files.
- `docs/` — design and specification notes.

## Status
This repository currently includes the project specification and the initial Phase 1 engineering scaffold only.

## Waveform visualizer
Run the local model-backed dashboard from the project root:

```bash
python scripts/visualizer.py
```

Open http://127.0.0.1:8000 in a browser. The dashboard loads deterministic windows from the test split, runs the checkpoint in `checkpoints/best_model.pt`, and shows the noisy input, denoised output, clean reference, residual noise, and per-window metrics. Use `--config`, `--checkpoint`, or `--port` to point it at another experiment.

Training progress can be written while an experiment runs:

```powershell
python scripts/train_model.py --epochs 60 --checkpoint checkpoints/model_60_epochs.pt --progress runs/model_60_epochs_progress.json
Get-Content runs/model_60_epochs_progress.json -Wait
```

The progress file is updated after every completed epoch with `completed_epochs`, `total_epochs`, the latest losses, and the best validation PRD.
