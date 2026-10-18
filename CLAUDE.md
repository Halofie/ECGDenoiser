# ECG Denoiser on ZCU104 — Project Status & Next Steps

Last reviewed: 2026-10-18 (board-side code complete; waiting for board access)

## Goal and definition of done

Denoise ECG signals using a trained neural network running on the Xilinx
**ZCU104** (Zynq UltraScale+ XCZU7EV).

The project is **complete** when all of these are true:

1. A host PC streams **noisy ECG windows over UART** to the ZCU104.
2. The board denoises each window on the FPGA fabric (DPU) and **sends the denoised
   window back over UART** to the host.
3. The returned output is **checked against the PC reference model** (golden vectors)
   and scored (PRD / SNR / MSE against the clean signal).
4. **Latency** is measured and reported: DPU-only, on-board end-to-end, and host
   round trip.
5. **Power consumption** of the board/chip is measured: idle compared with active,
   plus energy per window.
6. Results are written up in `reports/hardware_results.md`.

All code for 1–6 exists and has been rehearsed end to end on the laptop.
**What remains needs the Vitis AI Docker image (re-quantize and compile) or the
board itself.** The full procedure is in
**[vitis_ai_deployment/BOARD_DAY_RUNBOOK.md](vitis_ai_deployment/BOARD_DAY_RUNBOOK.md)**.

---

## Current status

### Two deployment paths exist. Use Path B.

| | Path A: custom HLS IP (parked) | **Path B: Vitis AI DPU (active)** |
|---|---|---|
| Code | `src/fpga/`, `scripts/export_hls.py`, `hardware/vivado/` | `vitis_ai_deployment/` |
| Model | 6-layer Conv1d, 256 samples | `ECGResUNet1D` (Conv2d `(1,k)`), 512 samples |
| Checkpoint | none | **`checkpoints/ecg_resunet1d_noisy_minmax.pt`** (deployable) |
| HW artifact | none | xmodels exist but were built from the **old** checkpoint, so they must be recompiled |
| Runtime | mock-tested only | `vitis_ai_deployment/board/` (rehearsed) |

Path A's open issues are in `TO_BE_CONTINUED_HARDWARE_IMPLEMENTATION.txt`. Don't
mix artifacts between the two paths.

### Done

- **Data pipeline.**
  - 26 MIT-BIH records are in `data/clean/`.
  - NSTDB `em` and `bw` noise are in `mit-bih-noise-stress-test-database-1.0.0/`.
  - Noise is mixed 0.7 em : 0.3 bw at SNR −5…15 dB.
  - Train/val is split by record; the held-out records are 222, 230, 231, 233
    and 234.
- **Normalization problem found and fixed (2026-10-18).**
  - The original checkpoint (`ecg_resunet1d.pt`, PRD 36.4%) scaled the noisy input
    with the *clean* window's min/max (`clean_oracle`). The board can't do that.
    Using normalization the board can compute, it scores PRD 46.7%, the same as
    the raw noisy input.
  - Retrained with `--normalization noisy_minmax`:
    **`ecg_resunet1d_noisy_minmax.pt` reaches PRD 38.56% / SNR 9.80 dB, compared
    with 46.61% / 9.02 dB for the noisy input**, on all 12,690 held-out windows
    scored exactly as the board runs. See `reports/normalization_study.json`.
  - Checkpoints now store their normalization, and `quantize.py`,
    `generate_golden_vectors.py` and QAT refuse a mismatch.
- **Board runtime: `vitis_ai_deployment/board/`.** This is a numpy-only package that
  runs on Python 3.7+ (a test enforces the syntax level). Its pieces:
  - `ecg_board/protocol.py`: UART protocol v2. Frames are 512-sample int16,
    1038 B per request and 1058 B per response, with CRC32, resync after
    garbage, PING/ERROR/BYE messages, and board timing in each response.
  - `ecg_board/transport.py`: pyserial, a raw-termios fallback (for a board
    image without pyserial), and TCP.
  - `ecg_board/backends.py`: `DpuBackend` (VART, NHWC-aware, fix-point
    scaling), plus `TorchBackend` and `IdentityBackend` for laptop rehearsal.
  - `ecg_board/power.py`: generic hwmon power discovery (INA226/PMBus), an
    optional rail map, and a phase-tagged CSV logger.
  - Entry points: `uart_server.py`, `board_session.py`, `power_monitor.py` and
    `uart_setup.sh`.
- **Host side.**
  - `scripts/uart_host_client.py`: windows mode and continuous real-time mode,
    with board-vs-laptop agreement, PRD, and latency breakdown.
  - `scripts/make_board_bundle.py`: builds the board bundle with an md5
    manifest.
  - `scripts/build_hardware_report.py`: builds the final report.
  - `scripts/rehearse_board_day.py`: full dry run.
- **Docker side.**
  - `vitis_ai_deployment/requantize_and_compile.sh`: runs calibration, the INT8
    check and compilation.
  - `quantize.py --evaluate`: scores float and INT8 through the board's exact
    pre/post-processing and writes `quant_eval.json`.
- **Rehearsal passed (2026-10-18)** with the torch backend over TCP and fake power
  sensors:
  - golden 5/5 OK
  - 300 windows with 0 CRC errors and 0 retries
  - report generated
- **Tests.** `python -m pytest -q` gives 97 passed and 1 skipped. The skipped test
  is a POSIX pty test that runs on Linux.
- **DSP baseline comparison.** See `reports/results_and_analysis.md`. That report
  uses clean-oracle numbers, and §6.3 now explains the deployable result.

### Not done: needs Docker or the board

- The xmodel has not yet been recompiled from `ecg_resunet1d_noisy_minmax.pt`.
  The existing `compiled_model*/` xmodels come from the old, non-deployable
  checkpoint.
- No INT8 accuracy number yet. `quant_eval.json` is produced in Docker.
- No board run yet: golden vectors on the DPU, UART link, latency and power.
- The work above is on branch **`dpu-board-deployment`** (pushed 2026-10-18) and is
  not merged into `main` yet.
- **Checkpoints are not in git** (`*.pt` is gitignored). Copy
  `checkpoints/ecg_resunet1d_noisy_minmax.pt` (md5 `e4404e44f84547523dc1d5722d0884b9`)
  to the Docker machine by hand.

### Known risks for board day

1. **The Vitis AI version must match the board image.** A 3.5-compiled xmodel won't
   load on a 3.0 image. Compile for both if the SD image is unknown (runbook §A).
2. **UART choice.** PS UART0 (`/dev/ttyPS0`) is usually the Linux console. Use
   `ttyPS1` if it exists, or free the console with `uart_setup.sh free ttyPS0`
   over SSH. TCP over Ethernet is a fallback only.
3. **The power sensors depend on the board image.** If there are no hwmon power
   channels, use an external meter on the 12 V input (runbook C5).
4. **Untestable without hardware.** The VART API details were exercised only
   against a fake: tensor dims, `fix_point`, and `execute_async`/`wait`.
   `board_session.py` prints all of them at startup so a mismatch shows up
   immediately.

---

## Next steps

### Before board day

1. Review and merge `dpu-board-deployment` into `main`. Copy the new checkpoint
   to the Docker machine, since it is not in git.
2. **Vitis AI Docker** (the lab machine that produced `compiled_model_v3.0`):
   ```bash
   cd vitis_ai_deployment
   bash requantize_and_compile.sh ../checkpoints/ecg_resunet1d_noisy_minmax.pt
   ```
   Exit criteria:
   - INT8 PRD within about 2 points of float (38.6%)
   - exactly 1 DPU subgraph

   If the board's Vitis AI version is unknown, repeat with the other Docker
   version and a different `TAG`.
3. `cd vitis_ai_deployment && python generate_golden_vectors.py` regenerates the
   golden vectors from the new checkpoint (now the default).
4. `python scripts/make_board_bundle.py --xmodel vitis_ai_deployment/compiled_model_noisy_minmax/ecg_denoiser.xmodel`
5. `python scripts/rehearse_board_day.py` re-confirms the flow. Set the FTDI
   latency timer to 1 ms on Windows.

### On board day (about 1 hour; details in the runbook)

1. Boot, SSH in, run `xdputil query`, and pick the bundle that matches.
2. Run `sh run_all.sh` on the board. It covers golden vectors, latency,
   throughput and power. Then run `compare_golden_results.py` on the laptop.
3. Start `uart_server.py --serial /dev/ttyPS1 --power` on the board. On the
   laptop, run `uart_host_client.py --port COMx`: once in windows mode with
   `--no-bye`, then once in continuous mode with `--realtime`.
4. Run `build_hardware_report.py` → `reports/hardware_results.md`, then commit it.

### Parallel and not blocking: model quality

PRD is 38.6% against a 5% target. Most of the error is in the low-SNR bins
(57.5% PRD at −5..0 dB compared with 24.4% at 10..15 dB). Ideas:
- QAT (`train.py --mode qat --normalization noisy_minmax`)
- more capacity
- a loss weighted toward morphology

Any retrain means re-running quantize, compile, golden vectors and the bundle.

---

## Working notes for this repo

- Windows host with PowerShell or Git Bash. Python tests: `python -m pytest -q`.
- **Board code (`vitis_ai_deployment/board/`) must stay numpy-only and Python 3.7
  compatible.** `tests/test_board_protocol.py` parses the code with
  `feature_version=(3,7)`. Never import torch, scipy or `src/` there.
- **Pre- and post-processing live in one place:** `board/ecg_board/preprocess.py`.
  Training (`dataset.py`), evaluation (`evaluation.py`), the host client and the
  board all import it. Don't duplicate it.
- **Quantization and compilation run only inside the Vitis AI Docker image.** It
  ships its own torch, so don't pip-install over it.
- **`vart` and `xir` exist only on the board image.**
- The DPU model must stay Conv2d with `(1, k)` kernels. `nn.Conv1d` falls off the
  DPU.
- Line endings: most files are LF, but `.gitignore`, `requirements.txt` and some
  tests are CRLF. Preserve whatever a file already uses. On Windows, Python
  `write_text` produces CRLF, so write bytes instead.
- `results/` and `dist/` are gitignored scratch output. Commit only
  `reports/hardware_results.md`, `reports/figures/hw_*` and
  `reports/dpu_validation/*.json`.
- Key docs:
  - `vitis_ai_deployment/BOARD_DAY_RUNBOOK.md`
  - `vitis_ai_deployment/README.md` (see its 2026-10-18 update section)
  - `reports/results_and_analysis.md`
  - `TO_BE_CONTINUED_HARDWARE_IMPLEMENTATION.txt` (Path A)
