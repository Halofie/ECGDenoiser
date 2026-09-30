# ECG Denoiser — Vitis AI 3.5 / DPU Deployment Path (ZCU104)

This is a separate deployment path from `hardware/vivado/` and `scripts/export_hls.py`.
That path targets a custom HLS accelerator IP wired via AXI DMA + UART,
built with hls4ml, at a 256-sample window length. **This** path targets the
Zynq UltraScale+ **DPU** (DPUCZDX8G) IP via Vitis AI 3.5 + VART, at a
512-sample window length, using a different model architecture
(`ECGResUNet1D`). The two pipelines do not share a bitstream, a runtime, or
a model checkpoint — pick one per deployment, don't mix artifacts between
them.

None of the Vitis AI-specific steps (PTQ/QAT quantization, `vai_c_xir`
compilation, board deployment) have been run against a real Vitis AI 3.5
install or a ZCU104 board in this workspace -- `pytorch_nndct` is not a
public pip package (Xilinx ships it as a patched wheel bound to a specific
torch 1.13 build inside their Docker image, incompatible with the torch 2.5
in this dev environment). Treat version-specific API details (quantizer call
signatures, arch.json paths, DPU fingerprint) as needing verification
against your actual installed tool, not as confirmed facts -- each script
below flags the specific spots that need that check.

**Training itself has been run for real**, on GPU, against the full
26-record default MIT-BIH set (21 train / 5 held-out validation records, by
record so there is no window-level leakage) mixed with real NSTDB em/bw
noise at SNR -5..15 dB. Current best checkpoint
(`checkpoints/ecg_resunet1d.pt`, epoch 3 of an early-stopped run):

| Metric | Value |
| --- | --- |
| Validation PRD | 36.4% |
| Validation output SNR | 9.87 dB (input SNR range was -5..15 dB) |
| Validation MSE | 0.0329 |

By the common clinical PRD banding (<9% very good, 9-19% good, 19-30%
acceptable, >30% poor), this checkpoint is in the **poor** band -- it is a
real, trained, board-ready-shaped model, not a validated-good denoiser yet.
Two rounds of trying to fix this (AdamW + weight decay, ReduceLROnPlateau)
did not move the number; a shared-scale normalization fix (see `dataset.py`
-- noisy and clean windows were being min-max normalized *independently*,
which forces the network to learn a noise-dependent rescaling on top of
denoising) only produced a small improvement. The SNR range itself
(down to -5 dB) is intrinsically hard and likely dominates the averaged PRD;
architecture capacity is the next thing to revisit if training quality needs
to improve further. Proceeding to quantization/compilation with this
checkpoint regardless, per project decision -- treat PRD improvement as a
parallel, not blocking, follow-up.

## Why the model is built from Conv2d, not Conv1d

The DPUCZDX8G IP is a 2D image core. `nn.Conv1d` is not a dependable op
through the Vitis AI PyTorch quantizer/compiler path. `model.py` implements
every layer with `nn.Conv2d` / `nn.ConvTranspose2d` using kernel shape
`(1, k)`, treating the ECG window as a `(N, 1, 1, 512)` "image". Keep this
convention if you extend the architecture — dropping in a real `Conv1d`
layer will likely get compiled off the DPU (or fail `vai_c_xir` outright).

## Dependencies

Host-side (training, quantization, compilation) — run inside the Vitis AI
3.5 Docker image, not your regular Python environment:

```bash
# Xilinx-provided image, pick cpu or gpu variant matching your host
docker pull xilinx/vitis-ai-pytorch-cpu:3.5.0.001-...   # or -gpu-...
docker run -it --rm -v $(pwd):/workspace xilinx/vitis-ai-pytorch-cpu:latest
```

Inside the container, `torch` (pinned ~1.13.1 for Vitis AI 3.5),
`pytorch_nndct`, and `vai_c_xir` are already installed — do not pip-install
a different `torch` version over them or the quantizer will break.

Board-side (`deploy_zcu104.py`) — the ZCU104's PetaLinux/Vitis AI board
image ships `vart`, `xir`, and `numpy` in its default Python 3 environment
already. Do not attempt to `pip install vart` off PyPI; it is not a
general-purpose package and only exists inside that board image / the Vitis
AI runtime install.

## Pipeline

### 1. Float training (already done — checkpoint exists)

```bash
cd vitis_ai_deployment
python train.py --mode float --epochs 60 --patience 15 \
    --device cuda --batch-size 64 \
    --clean-dir ../data/clean \
    --noise-dir ../mit-bih-noise-stress-test-database-1.0.0 \
    --checkpoint ../checkpoints/ecg_resunet1d.pt \
    --progress ../runs/ecg_resunet1d_progress.json
```

`--clean-dir`/`--noise-dir` point at real MIT-BIH/NSTDB data (auto-downloaded
via `wfdb` if missing); omitting them falls back to `PlaceholderECGDataset`
(random data, not meaningful). Records are split by record via
`dataset.default_train_val_split()` (last 20% of the record list held out),
so validation windows never come from a patient the model trained on. Best
checkpoint is selected by validation PRD, with early stopping and a
`ReduceLROnPlateau` scheduler. `--device cuda` needs a CUDA-capable GPU and
torch built with CUDA support; drop to `--device cpu` otherwise (~7x
slower per the benchmarks run in this workspace: 0.9ms/window on an RTX 3050
vs 6.0ms/window on CPU).

### 2a. Quantization-aware training (recommended — "quantized training")

```bash
python train.py --mode qat --epochs 5 \
    --checkpoint ../checkpoints/ecg_resunet1d.pt \
    --qat-checkpoint ../checkpoints/ecg_resunet1d_qat.pt \
    --qat-output-dir ../artifacts/qat_xmodel \
    --clean-dir ../data/clean \
    --noise-dir ../mit-bih-noise-stress-test-database-1.0.0
```

Fine-tunes the converged float model (`checkpoints/ecg_resunet1d.pt`, the
real trained checkpoint described above) through
`pytorch_nndct.QatProcessor`, which simulates INT8 DPU arithmetic during
training. This directly exports an xmodel at the end
(`artifacts/qat_xmodel/`) — you can skip step 2b if you use this path. QAT
typically preserves more accuracy than PTQ alone for a skip-connection-heavy
UNet, at the cost of a few epochs of retraining. Given the current
checkpoint's PRD, QAT fine-tuning is also a reasonable place to look for a
quality improvement, not just a quantization step.

### 2b. Post-training quantization (alternative / PTQ sanity check)

```bash
python quantize.py --mode calib --checkpoint ../checkpoints/ecg_resunet1d.pt \
    --clean-dir ../data/clean --noise-dir ../mit-bih-noise-stress-test-database-1.0.0
python quantize.py --mode test  --checkpoint ../checkpoints/ecg_resunet1d.pt \
    --clean-dir ../data/clean --noise-dir ../mit-bih-noise-stress-test-database-1.0.0
```

`calib` collects activation statistics over a calibration set (again, pass
`--clean-dir`/`--noise-dir` for real representative ECG windows — this
matters more here than in training, since PTQ has no further chance to
correct for a bad calibration distribution). `test` exports
`artifacts/quantize_result/ECGResUNet1D_int.xmodel`.

### 3. Compile for the ZCU104 DPU

```bash
bash compile.sh artifacts/quantize_result/ECGResUNet1D_int.xmodel \
    /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
    ./compiled_model ecg_denoiser
```

Still inside the Vitis AI Docker image. See the comment block at the top of
`compile.sh` for why the arch.json path differs from the Vitis-AI-1.x-style
path (`/arch/dpuv2/ZCU104/arch.json`) — and why you must use the arch.json
matching your board's actual DPU build if it's not the stock ZCU104
reference design.

Output: `compiled_model/ecg_denoiser.xmodel`.

### 4. Generate golden test vectors (on the laptop, not the board)

```bash
cd vitis_ai_deployment
python generate_golden_vectors.py \
    --checkpoint ../checkpoints/ecg_resunet1d.pt \
    --clean-dir ../data/clean \
    --noise-dir ../mit-bih-noise-stress-test-database-1.0.0 \
    --output-dir ../artifacts/golden_vectors \
    --num-vectors 5
```

Produces fixed, known (input, expected_float_output) pairs from real
held-out validation windows -- needed because `deploy_zcu104.py
--test-input` generates a *new* random array on the board itself, which
can't be reproduced on the laptop for comparison. These golden vectors are
the only way to check DPU *correctness*, not just "did it run."

### 5. Copy the xmodel, deploy script, and golden inputs to the board

Only the `input_*.npy` files need to go to the board -- `expected_*.npy`
stays on the laptop, since comparison happens there (step 7), not on the
board:

```bash
scp compiled_model/ecg_denoiser.xmodel root@<zcu104-ip>:/home/root/
scp deploy_zcu104.py root@<zcu104-ip>:/home/root/
scp -r ../artifacts/golden_vectors root@<zcu104-ip>:/home/root/   # input_*.npy AND expected_*.npy come along, but only input_*.npy gets used on-board
```

If your board's SSH server doesn't support SFTP (`scp` fails with
`/usr/libexec/sftp-server: No such file or directory`) or the legacy
protocol (`scp -O ...` resets the connection), fall back to serving the
files over plain HTTP from the laptop (`python -m http.server 8000` in this
directory) and `wget`/`curl -O`-ing them from the board instead.

### 6. Run on the ZCU104

First, confirm VART loads the model and the DPU subgraph executes at all:

```bash
python3 deploy_zcu104.py --xmodel ecg_denoiser.xmodel --test-input --loop 5
```

Then run the golden inputs through the DPU and save each raw output --
the board does NOT compare against expected values itself, it just runs
inference and writes `output_i.npy` per vector:

```bash
python3 deploy_zcu104.py --xmodel ecg_denoiser.xmodel \
    --golden-dir golden_vectors \
    --golden-output-dir golden_outputs
```

### 7. Copy the DPU outputs back to the laptop and compare there

```bash
scp -r root@<zcu104-ip>:/home/root/golden_outputs ../artifacts/golden_outputs_from_board
```

(or the HTTP fallback in reverse: `python -m http.server 8000` on the
board, `curl`/download each `output_i.npy` from the laptop)

```bash
python compare_golden_results.py \
    --golden-dir ../artifacts/golden_vectors \
    --dpu-output-dir ../artifacts/golden_outputs_from_board \
    --report-dir ../reports/dpu_validation
```

This prints per-vector MSE/max-abs-diff between the DPU's INT8 output and
the float model's expected output, plus a summary, AND saves a timestamped
JSON report under `reports/dpu_validation/` -- a persistent record of every
validation run, not just terminal output that scrolls away. Don't expect an
exact match between DPU and float outputs: INT8 quantization introduces
real error versus the float model. Small, consistent per-window error
(the script's default warn threshold is MSE > 0.05) is expected; huge
error, NaNs, or a shape mismatch indicates an actual bug (quantization,
compilation, or fix-point handling in `deploy_zcu104.py`), not normal
quantization noise -- the script flags these explicitly as `HIGH_ERROR`,
`NON_FINITE_OUTPUT`, or `SHAPE_MISMATCH` in the report.

Once both of those look right, implement `read_axi_stream_window()` in
`deploy_zcu104.py` against your DSP pre-filter IP's actual interface (UIO
mmap, PYNQ DMA buffer, custom char device — whichever your PL design uses),
then drop `--test-input`/`--golden-dir` for the real signal path.

## Verification checklist before trusting board output

- [ ] `deploy_zcu104.py` prints exactly one DPU subgraph. More than one
      means part of the compiled graph fell back to CPU — check the
      `vai_c_xir` compile log for unsupported ops.
- [ ] Input/output tensor `dims` printed at startup match `(1, 1, 1, 512)`.
- [ ] `xdputil query` on the board reports a DPU fingerprint matching the
      arch.json used to compile.
- [ ] `read_axi_stream_window()` is implemented against the real PL
      interface, not left as the `NotImplementedError` stub.
- [ ] DPU output compared against the float/QAT PyTorch model's output on
      the same input, within an agreed tolerance, before treating the
      board's denoised waveform as trustworthy (`generate_golden_vectors.py`
      + `deploy_zcu104.py --golden-dir` + `compare_golden_results.py`;
      reports saved under `reports/dpu_validation/`).
- [ ] Quantized (PTQ or QAT) model's validation PRD/SNR/MSE compared against
      the float baseline above (PRD 36.4%, SNR 9.87 dB, MSE 0.0329) using
      `train.py`'s `evaluate()` -- INT8 quantization should not silently
      make an already-mediocre model meaningfully worse. If it does, that's
      a real regression to flag, not something to attribute to "the model
      just isn't great yet."
