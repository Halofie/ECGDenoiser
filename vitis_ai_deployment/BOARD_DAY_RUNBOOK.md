# ZCU104 Board-Day Runbook

Goal: in one sitting with the board, get every result needed to close the project.

- **Correctness.** DPU output matches the laptop model.
- **Denoising quality.** Measured on the board, over UART.
- **Latency.** DPU-only, on-board pipeline, and UART round trip.
- **Power.** Idle, active and energy per window.

Everything below was rehearsed on the laptop with `scripts/rehearse_board_day.py`.
The only untested parts are the ones that need the board itself: VART/DPU, the
serial port, and the hwmon power sensors.

---

## A. Before board day (laptop and Docker)

Do these in order. Each one has a check.

| # | Step | Command | Check |
|---|---|---|---|
| A1 | Deployable checkpoint (done) | `vitis_ai_deployment/train.py --normalization noisy_minmax ...` | `checkpoints/ecg_resunet1d_noisy_minmax.pt` exists |
| A2 | Score it as the board will run it | `python scripts/eval_deployable_normalization.py --checkpoint deploy=checkpoints/ecg_resunet1d_noisy_minmax.pt --methods noisy_minmax` | PRD below the noisy input's PRD |
| A3 | Quantize, check INT8, compile (**in the Vitis AI Docker**) | `cd vitis_ai_deployment && bash requantize_and_compile.sh ../checkpoints/ecg_resunet1d_noisy_minmax.pt` | INT8 within about 2 PRD points of float; compile log shows **1 DPU subgraph** |
| A4 | Golden vectors for that checkpoint | `cd vitis_ai_deployment && python generate_golden_vectors.py` | `artifacts/golden_vectors/golden_meta.json` says `noisy_minmax` |
| A5 | Bundle for the board | `python scripts/make_board_bundle.py --xmodel vitis_ai_deployment/compiled_model_noisy_minmax/ecg_denoiser.xmodel` | `dist/zcu104_bundle.tar.gz` |
| A6 | Full dry run | `python scripts/rehearse_board_day.py` | ends with "Rehearsal complete" |
| A7 | Host serial support | `pip install pyserial` | `python -c "import serial"` |

**Vitis AI version must match the board image.** An xmodel compiled with the
3.5 tools will not load on a 3.0 board image, and the reverse is also true.
`compiled_model_v3.0/` exists because this has already happened once. If you
don't know which SD image the board has, run A3 in **both** the 3.0 and the 3.5
Docker images. Set `TAG=noisy_minmax_v30` or `TAG=noisy_minmax_v35` so the two
outputs don't overwrite each other, and build two bundles with
`--out dist/zcu104_bundle_v30` / `--out dist/zcu104_bundle_v35`.

**Windows FTDI latency.** The ZCU104 USB-UART is an FTDI part. Its Windows
driver holds received bytes for up to 16 ms by default, which inflates
round-trip times. To fix it: Device Manager → Ports → *the board's COM port* →
Port Settings → Advanced → **Latency Timer = 1 ms**. Do this on every COM port
of the board.

---

## B. On board day

### B1. Boot and reach the board (≈10 min)

1. Connect micro-USB UART + Ethernet + power. Boot the Vitis AI SD image.
2. Find the console COM port. The board shows up as several COM ports; open each
   at 115200 in PuTTY or Tera Term and press Enter until you get a login prompt.
   Note which one is the console.
3. Bring up networking:
   - `ip addr` on the board
   - on the laptop, `ssh root@<board-ip>`
   - if there's no DHCP: `ifconfig eth0 192.168.1.10` on the board, and set the
     laptop NIC to 192.168.1.x
4. Identify the DPU and runtime:
   ```sh
   xdputil query        # DPU arch must say DPUCZDX8G_ISA1_B4096; note the Vitis AI version
   python3 -c "import vart, xir, numpy; print(numpy.__version__)"
   ```
   Pick the bundle that matches that Vitis AI version.

### B2. Copy the bundle and run the offline session (≈5 min)

```sh
# laptop
scp dist/zcu104_bundle.tar.gz root@<board-ip>:~/
# board
tar xzf zcu104_bundle.tar.gz && cd zcu104_bundle
sh run_all.sh
```

If `scp` fails with `sftp-server: not found`, try `scp -O ...`. If that also
fails, run `python -m http.server 8000` in `dist/` on the laptop and
`wget http://<laptop-ip>:8000/zcu104_bundle.tar.gz` on the board.

`run_all.sh` runs these steps:

1. `uart_setup.sh` lists the serial devices, the console, and the UART nodes in
   the device tree.
2. `power_monitor.py list` shows which power sensors exist and writes
   `rails_template.json`.
3. `board_session.py` runs the golden vectors, 60 s of idle power, 2000-window
   latency, and throughput at 1, 2 and 4 threads, each for 30 s with power logged.
   It writes `results/session_<ts>/` and `results/session_<ts>.tar.gz`.

What to look for in its output:

| Line | Good | Bad → see section C |
|---|---|---|
| `backend: {... 'dpu_subgraphs': 1 ...}` | 1 | >1 → C2 |
| `input dims` | `[1, 1, 512, 1]` (or any 512-only shape) | error → C3 |
| `golden N: ... range [...]` | finite and in the expected range | NaN or all-zero → C4 |
| `DPU-only mean X ms` | a few ms or less | — |
| `power: N hwmon channels` | N ≥ 1 | WARNING → C5 |

**Power rails (optional, 5 min).** If `power_monitor.py list` showed channels,
copy `rails_template.json` to `rails.json` and rename the rails. Keep in
`total` only rails that don't feed each other. Then rerun
`python3 board_session.py --xmodel ecg_denoiser.xmodel --golden-dir golden_vectors --rail-map rails.json`.
Without a rail map, all channels are summed. That is fine for idle-vs-active
deltas but can double count absolute watts.

Copy the session back and check correctness right away:

```sh
# laptop
scp root@<board-ip>:~/zcu104_bundle/results/session_*.tar.gz results/
tar xzf results/session_<ts>.tar.gz -C results/
python vitis_ai_deployment/compare_golden_results.py --golden-dir artifacts/golden_vectors \
    --dpu-output-dir results/session_<ts>/golden_outputs --report-dir reports/dpu_validation
```

You want `num_ok == 5`. INT8 is expected to differ slightly from float.
`HIGH_ERROR` means something is wrong with quantization, the fix-point
scaling, or the xmodel/board mismatch. Stop and debug that before going on.

### B3. UART link (≈20 min)

**Pick the data UART** (on the board: `sh uart_setup.sh`):

- **Option 1: a second PS UART** (`/dev/ttyPS1`) if it exists and is not the
  console. On the PC it appears as another of the board's COM ports. Find out
  which one by elimination, or by sending a few bytes with
  `echo hi > /dev/ttyPS1` while watching the COM ports in a terminal.
- **Option 2: reuse the console UART.** Run `sh uart_setup.sh free ttyPS0` over
  **SSH**. This stops the login getty and kernel console messages. The laptop
  then uses the same COM port that showed the console. Close PuTTY first.
- **Option 3: a USB-serial dongle** in the board's USB port (`/dev/ttyUSB0`),
  cross-wired to a second dongle on the laptop.
- **Fallback: Ethernet.** Use `--tcp-port 5555` on the board and
  `--port tcp://<board-ip>:5555` on the laptop. Same protocol, same scripts.
  This counts as a fallback only, because the requirement is UART.

**Run:**

```sh
# board (over SSH), from zcu104_bundle/
python3 uart_server.py --xmodel ecg_denoiser.xmodel --serial /dev/ttyPS1 --baud 115200 --power
```
Wait for `READY on serial:/dev/ttyPS1@115200`.

```powershell
# laptop -- 1. held-out windows: quality + board-vs-laptop agreement + latency (~3.5 min for 1000)
python scripts/uart_host_client.py --port COM7 --baud 115200 --num-windows 1000 --no-bye
# laptop -- 2. continuous real-time stream (60 s of ECG, paced at 360 Hz)
python scripts/uart_host_client.py --port COM7 --baud 115200 --mode continuous --record 231 --seconds 60 --realtime
```

The second command sends BYE at the end, and the board server then writes
`results/uart_<ts>/server_summary.json` (with power during the session).

**Optional: a faster link.** Restart the server with `--baud 921600` and rerun
command 1 with `--baud 921600 --num-windows 300`. That shows how much of the
round trip is wire time. If 921600 gives CRC errors, try 460800.

What to look for on the laptop:

| Line | Good |
|---|---|
| `Connected via serial:COM7@115200; ping X ms` | ping succeeds |
| `board PRD` vs `noisy PRD` | board is lower |
| `board vs laptop: RMSE ... corr` | corr > 0.99 (INT8 vs float) |
| `round trip mean` | ≈ 182 ms of wire time at 115200 + board time + a few ms |
| `client stats` | `timeouts` and `crc_errors` at or near 0 |

### B4. Collect and build the report (≈5 min)

```sh
# laptop
scp -r root@<board-ip>:~/zcu104_bundle/results/uart_* results/
python scripts/build_hardware_report.py \
    --session results/session_<ts> --server results/uart_<ts> \
    --host results/host_windows_<ts> --host-continuous results/host_continuous_<ts> \
    --golden-report reports/dpu_validation/dpu_validation_<ts>.json \
    --quant-eval vitis_ai_deployment/artifacts/quantize_result_noisy_minmax/quant_eval.json
```

This writes `reports/hardware_results.md` and `reports/figures/hw_*.png`.
Commit those, and the golden validation JSON, as the project's final results.

---

## C. Troubleshooting

| # | Symptom | Likely cause | Action |
|---|---|---|---|
| C1 | `xir.Graph.deserialize` / runner creation fails | xmodel compiled with a different Vitis AI version, or a DPU fingerprint mismatch | Compare `xdputil query` with the compile target and use the bundle for the board's VAI version. `meta.json` next to the compiled xmodel holds its target (see `xdputil xmodel --help` for inspecting it on the board). |
| C2 | More than 1 DPU subgraph | an op fell back to the CPU at compile time | Check the `vai_c_xir` log. The model must stay Conv2d `(1,k)` / BN / ReLU / add / ConvTranspose2d. |
| C3 | `Unexpected DPU input tensor dims` | a model with different I/O | Print `backend.info()`. The code only accepts 512 samples × 1 channel. |
| C4 | Golden `HIGH_ERROR` or NaN | wrong fix-point scaling, a bad calibration, or the wrong xmodel | Check `fix_point` in `env.json`, rerun A3, and confirm the A4 golden vectors came from the same checkpoint. |
| C5 | No hwmon power channels | the image lacks INA226/PMBus drivers, or the sensors sit behind the system controller | Check `ls /sys/class/hwmon/*/name` and `dmesg \| grep -i -e ina -e pmbus`. Fallback: a USB/wall power meter on the 12 V input. Read idle vs `run_all.sh` throughput phases manually and note the numbers in the report. |
| C6 | Host: timeouts, nothing received | wrong COM port or tty, baud mismatch, or the console getty still attached | Check `fuser /dev/ttyPS*` on the board and run `uart_setup.sh free <tty>`. Try the other COM ports. |
| C7 | Host: steady `crc_errors` / board `discarded_bytes` | kernel messages on the console UART, or the baud rate is too high | Run `dmesg -n 1` (done by `uart_setup.sh free`) and lower the baud. |
| C8 | `pyserial` missing on the board | — | No action needed: the server falls back to raw termios automatically (`(termios)` in the READY line). |
| C9 | `Permission denied` on `/dev/ttyPS*` | not root | Run as root. |
| C10 | Board ping OK but round trip ≫ wire time | the FTDI latency timer | Set it to 1 ms (section A). |
