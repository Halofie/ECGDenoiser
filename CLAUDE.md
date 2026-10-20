# ECG Denoiser on ZCU104 — Project Status & Next Steps

Last reviewed: 2026-10-20 (first board day: on-board latency/throughput/power run
completed, but the DPU golden-vector accuracy check **failed** — see "Board day
2026-10-20 findings" below. This blocks calling the project done; UART round trip
and the final report are also still outstanding.)

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

All code for 1–6 exists and has been rehearsed end to end on the laptop. The
deployable model is quantized and compiled for both Vitis AI 3.0 and 3.5, and
the board bundles are built. **Everything that remains needs the board**, which
will be connected to this laptop. The full procedure is in
**[vitis_ai_deployment/BOARD_DAY_RUNBOOK.md](vitis_ai_deployment/BOARD_DAY_RUNBOOK.md)**.

---

## Current status

### Two deployment paths exist. Use Path B.

| | Path A: custom HLS IP (parked) | **Path B: Vitis AI DPU (active)** |
|---|---|---|
| Code | `src/fpga/`, `scripts/export_hls.py`, `hardware/vivado/` | `vitis_ai_deployment/` |
| Model | 6-layer Conv1d, 256 samples | `ECGResUNet1D` (Conv2d `(1,k)`), 512 samples |
| Checkpoint | none | **`checkpoints/ecg_resunet1d_noisy_minmax.pt`** (deployable) |
| HW artifact | none | **`compiled_model_noisy_minmax_v35/` and `_v30/`** (deploy these; the older `compiled_model/` and `compiled_model_v3.0/` come from the non-deployable checkpoint) |
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
- **Quantized and compiled on this laptop (2026-10-19)**, using
  `bash scripts/compile_in_docker.sh 3.5|3.0`. Both Docker images are installed
  locally, and Docker Desktop must be running.
  - INT8 PRD 41.10% vs float 39.40% vs noisy input 47.38% on 500 held-out
    windows (+1.70 points for INT8). Results are in
    `reports/quant_eval_noisy_minmax_v3{0,5}.json`.
  - 1 DPU subgraph in each, target `DPUCZDX8G_ISA1_B4096`. xmodel md5s:
    `26c740ea45b2d5179012590cdec6c03b` (v3.5) and
    `6fdaf1c3a751aa3e42be2a239f5ec4c2` (v3.0).
  - Real I/O tensors, checked with `vitis_ai_deployment/inspect_xmodel.py`:
    dims `[1,1,512,1]`, xint8, input fix_point 6 and output fix_point 7. This
    matches what `DpuBackend` assumes. Output fix_point 7 clips 0.27% of
    samples, costing about 0.6 PRD points (already in the INT8 number).
- **Golden vectors regenerated** from the new checkpoint (`artifacts/golden_vectors/`).
  **Bundles built**: `dist/zcu104_bundle_v35.tar.gz` and
  `dist/zcu104_bundle_v30.tar.gz`. These are gitignored; rebuild them with
  `make_board_bundle.py` if needed.
- **Rehearsal passed (2026-10-18)** with the torch backend over TCP and fake power
  sensors:
  - golden 5/5 OK
  - 300 windows with 0 CRC errors and 0 retries
  - report generated
- **Tests.** `python -m pytest -q` gives 97 passed and 1 skipped. The skipped test
  is a POSIX pty test that runs on Linux.
- **DSP baseline comparison.** See `reports/results_and_analysis.md`. That report
  uses clean-oracle numbers, and §6.3 now explains the deployable result.

### Board day 2026-10-20 findings — accuracy check FAILED on real hardware

First time on the actual board (`xilinx-zcu104-20222`, direct Ethernet via a
USB-C dongle, static IP `192.168.1.10`). `xdputil query` confirmed
**VAI Version 3.0.0**, `DPU Arch DPUCZDX8G_ISA1_B4096` — so the **v3.0 bundle**
(`dist/zcu104_bundle_v30.tar.gz`) is the right one for this board/image.

**Latency, throughput and power all ran successfully** (`sh run_all.sh`,
session `session_20261005_181229`, pulled back to
`dist/results/session_20261005_181229/` — gitignored, laptop-only):
- DPU-only inference: mean 0.734 ms, p99 0.773 ms.
- Full pipeline (pre+infer+post): mean 1.884 ms, p99 2.229 ms.
- Throughput: 1,322 windows/s (1 thread) → 2,627 (2 threads) → 2,894 (4 threads).
- Power: idle 14.83 W mean; active (4 threads, saturated) 19.47 W mean; energy
  per window as low as 6.73 mJ at 4 threads.
- These numbers are solid and don't need to be re-captured.

**But the golden-vector accuracy check failed badly.** Running
`compare_golden_results.py` against this session's `golden_outputs/` gives
0/5 OK, mean MSE 0.097 (report saved at
`reports/dpu_validation/dpu_validation_20261020T060225Z.json`). Scored properly
against the **clean** signal (not just the float model's own prediction), the
DPU's actual PRD on these 5 vectors averages **~83%** — worse than doing
nothing (46.6% for raw noisy input), and far worse than the 41.10% predicted by
the software INT8 simulation in `quant_eval_noisy_minmax_v30.json`.

Ruled out as the cause (all checked and confirmed correct):
- Bundle/board version match (confirmed via `xdputil query`).
- xmodel identity — md5 `6fdaf1c3a751aa3e42be2a239f5ec4c2` matches everywhere:
  the compile output, the bundle `MANIFEST.json`, and the board's own
  self-reported `xmodel_md5` in `session.json`.
- Golden input files — bit-identical md5s between `artifacts/golden_vectors/`
  and what shipped to the board.
- `DpuBackend`'s quantize/dequantize math in `backends.py` (scale = 2^fix_point,
  fix_point 6 in / 7 out) — correct and matches `inspect_xmodel.py`.
- Not a simple scale/bias miscalibration — fitting the best possible affine
  correction (`expected ≈ a·dpu + b`) barely helps; correlation between DPU
  output and the expected output stays weak-to-near-zero for 3 of 5 vectors
  even after that fit.
- Not a dead/passthrough subgraph — DPU output correlates moderately
  (0.2–0.67) with the raw input for 4 of 5 vectors, so the DPU is computing
  *something*, just not reproducing the trained transform well.

**Working hypothesis:** this is the risk that used to be listed as "untestable
without hardware" (`execute_async`/`wait` were only ever exercised against a
fake runner) — it has now materialized. Most likely cause: the real DPU's
power-of-2 shift-based requantization between layers is coarser than the
software NNDCT fake-quant simulation, and for this U-Net's depth and skip
connections the error compounds far more on real silicon than predicted. A
compile/op-support bug specific to this DPU ISA/compiler version is the other
candidate; not yet distinguished from the hardware-requant theory.

**Next steps to isolate the cause** (not yet done):
1. Try the **v3.5 bundle** on this same board (may be rejected outright by the
   VAI 3.0 runtime — worth a quick check) — if the same magnitude of error
   shows up, it points away from a v3.0-compiler-specific bug.
2. Re-run `bash scripts/compile_in_docker.sh 3.0` fresh in case the existing
   compiled artifact came from a stale/bad calibration run, then regenerate
   golden vectors and retest.
3. Inspect per-layer quantization scales in
   `vitis_ai_deployment/artifacts/quantize_result_noisy_minmax/` for anything
   unusually tight/saturated, especially around skip-connection concatenations.
4. QAT retrain (`train.py --mode qat --normalization noisy_minmax`) was already
   on the roadmap for model quality (see "Parallel and not blocking" below) —
   this result makes it look necessary, not just nice-to-have, since plain PTQ
   is clearly inadequate for this architecture on real hardware.

### Not done: needs the board

- **UART round trip** (runbook §B3: `uart_server.py` on the board,
  `uart_host_client.py` on the laptop) — not attempted yet this session.
- **Final report** (`scripts/build_hardware_report.py` → `reports/hardware_results.md`)
  — not generated yet.
- **The accuracy bug above must be root-caused and fixed** before the project
  can be called done — latency/throughput/power alone aren't sufficient per the
  definition of done (item 3 requires the output to check out against the PC
  reference, scored by PRD/SNR/MSE).
- The work is on branch **`dpu-board-deployment`** and is not merged into `main` yet.
- Checkpoints are not in git (`*.pt` is gitignored).
  `checkpoints/ecg_resunet1d_noisy_minmax.pt` (md5 `e4404e44f84547523dc1d5722d0884b9`)
  exists only on this laptop, so back it up.

### Known risks for board day

1. **The Vitis AI version must match the board image.** A 3.5-compiled xmodel won't
   load on a 3.0 image. Both bundles are built; `xdputil query` on the board
   decides which one to use. (Confirmed 2026-10-20: this board is VAI 3.0 — use
   the v3.0 bundle.)
2. **UART choice.** PS UART0 (`/dev/ttyPS0`) is usually the Linux console. Use
   `ttyPS1` if it exists, or free the console with `uart_setup.sh free ttyPS0`
   over SSH. TCP over Ethernet is a fallback only. (Not yet attempted on this
   board — `ttyPS1` exists per `run_all.sh`'s serial-device listing, so that's
   the one to try first.)
3. **The power sensors depend on the board image.** If there are no hwmon power
   channels, use an external meter on the 12 V input (runbook C5). (Not needed
   on this board — one `ina226` hwmon channel was found and worked.)
4. **Materialized 2026-10-20: the DPU produces wrong output on real hardware.**
   The runner calls (`execute_async`/`wait`) were only ever exercised against a
   fake before board day. They now run correctly mechanically (no crashes, right
   tensor shapes/dtypes), but the actual numbers are badly wrong — see "Board
   day 2026-10-20 findings" above. This is the top-priority open item.
5. **The laptop's Ethernet connection to the board (via a USB-C-to-Ethernet
   dongle) dropped repeatedly** during board day — mid-`scp`, mid-`run_all.sh`.
   Likely Windows USB selective suspend / NIC power management kicking in
   during low-traffic phases; not yet confirmed fixed. Workarounds that worked:
   - This board's minimal image has **no `tmux`/`screen` and no `sftp-server`**.
     Use `scp -O` (legacy protocol) or `ssh root@<ip> "cat > file" < localfile`
     for transfers, and `nohup sh run_all.sh > run_all.log 2>&1 & disown` to
     keep long jobs alive across SSH drops (the board process survives;
     reconnect and check `run_all.log` / `ps aux`).
   - The board's static IP on `eth0` reverted to link-local-only (no `inet`
     line) at least once mid-session for no clear reason — re-run
     `ifconfig eth0 192.168.1.10 netmask 255.255.255.0 up` if `ssh`/`ping`
     suddenly stop reaching the board.
   - If `run_all.sh` hangs on "waiting for process to release the resource:
     DPU_0" after a dropped connection, an orphaned process from the previous
     attempt still holds the DPU — `ps aux | grep python`, `kill -9 <pid>`, retry.
   - Before fixing the dongle, try: Device Manager → the dongle's USB
     controller and NIC entries → Power Management tab → uncheck "Allow the
     computer to turn off this device to save power"; also disable USB
     selective suspend in Windows power settings.
   - The board's root password was blank initially; `passwd root` was used to
     set one so SSH password auth would work.

---

## Next steps

### Before board day (done 2026-10-19; redo only after a retrain)

For a retrain, the sequence is:
1. `bash scripts/compile_in_docker.sh 3.5` and `bash scripts/compile_in_docker.sh 3.0`
2. `cd vitis_ai_deployment && python generate_golden_vectors.py`
3. `make_board_bundle.py --xmodel ... --out dist/zcu104_bundle_v35` (and `_v30`)
4. `rehearse_board_day.py`

Still to do by hand:
- Set the FTDI latency timer to 1 ms on the board's COM ports once the board is
  plugged in (runbook §A).
- Merge `dpu-board-deployment` into `main`.

### On board day (about 1 hour; details in the runbook)

1. ✅ Boot, SSH in, run `xdputil query`, and pick `dist/zcu104_bundle_v35` or
   `_v30` to match. (Done 2026-10-20 — this board is v3.0.)
2. ✅ Run `sh run_all.sh` on the board. It covers golden vectors, latency,
   throughput and power. Then run `compare_golden_results.py` on the laptop.
   (Done 2026-10-20 — latency/throughput/power are good, but the golden-vector
   accuracy check **failed**; see "Board day 2026-10-20 findings" above. Fix
   that before re-running this step, or at minimum re-validate after any fix.)
3. ⬜ Start `uart_server.py --serial /dev/ttyPS1 --power` on the board. On the
   laptop, run `uart_host_client.py --port COMx`: once in windows mode with
   `--no-bye`, then once in continuous mode with `--realtime`. (Not attempted
   yet — next thing to do when back at the board.)
4. ⬜ Run `build_hardware_report.py` → `reports/hardware_results.md`, then
   commit it. (Blocked on step 3, and arguably on fixing the accuracy bug
   first — the report should reflect a working model, not a broken one.)

### Immediate next session priorities

1. **Root-cause the accuracy failure** (see "Board day 2026-10-20 findings").
   Start with re-running `compile_in_docker.sh 3.0` fresh and trying the v3.5
   bundle on the same board, since those are the cheapest checks.
2. If the board is accessible again, **finish UART round trip** (step 3 above)
   regardless — it doesn't depend on the accuracy bug being fixed yet, and
   it's one less thing to redo later. Use `nohup`/`ifconfig` workarounds above
   if the USB-C dongle keeps dropping the connection.
3. Don't build the final report or merge `dpu-board-deployment` into `main`
   until the accuracy check passes — the current numbers would misrepresent
   the project as done when it isn't (definition-of-done item 3 fails).

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
  ships its own torch, so don't pip-install over it. `scripts/compile_in_docker.sh`
  handles it:
  - it maps `3.5` to the `xilinx/vitis-ai-pytorch-cpu:latest` image and `3.0` to
    `:ubuntu2004-3.0.0.106`
  - it installs `wfdb` from `dist/wheels/` with `--no-deps`, since the images
    lack it and their pip network access is unreliable
  - Docker Desktop lives at `%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe`
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
