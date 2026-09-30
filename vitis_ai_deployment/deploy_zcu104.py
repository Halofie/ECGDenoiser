"""Board-side VART host application for the compiled ecg_denoiser.xmodel.

Runs on the ZCU104's ARM Cortex-A53 (PetaLinux, in the Python 3 environment
that ships with the Vitis AI 3.5 board image -- vart/xir are preinstalled
there, do not `pip install` them separately).

Pipeline:
    AXI-Stream window from the DSP pre-filter IP (512 float samples)
      -> quantize to INT8 using the DPU's expected input scale
      -> DPU inference via VART
      -> dequantize back to float
      -> clean ECG waveform (512 samples)

The AXI-Stream read-out (`read_axi_stream_window`) is stubbed: how you pull
a window off that IP (UIO mmap, an AXI DMA + /dev/xdma char device, a PYNQ
overlay, etc.) is specific to your PL design and is not something this
script can know without that interface definition. Replace the stub with
your actual read path; everything from `run_inference` onward is generic
VART and does not need to change.

This script has not been run against a real ZCU104 / compiled xmodel in
this workspace -- no board or Vitis AI install is available here. Verify the
input/output tensor names, shapes, and fix-point scales printed by
`get_input_tensors` / `get_output_tensors` against your actual compiled
model before trusting the quantize/dequantize math below.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import vart
import xir

INPUT_LENGTH = 512


def get_child_subgraph_dpu(graph: "xir.Graph") -> list:
    root = graph.get_root_subgraph()
    children = root.toposort_child_subgraph()
    return [
        s
        for s in children
        if s.has_attr("device") and s.get_attr("device").upper() == "DPU"
    ]


def load_dpu_runner(xmodel_path: str) -> vart.Runner:
    graph = xir.Graph.deserialize(xmodel_path)
    dpu_subgraphs = get_child_subgraph_dpu(graph)
    if not dpu_subgraphs:
        raise RuntimeError(f"No DPU subgraph found in {xmodel_path}")
    if len(dpu_subgraphs) > 1:
        print(
            f"WARNING: {len(dpu_subgraphs)} DPU subgraphs found; using the "
            "first. A fully DPU-offloaded model should have exactly one -- "
            "extra subgraphs usually mean some ops fell back to CPU during "
            "compilation. Check the vai_c_xir compile log.",
            file=sys.stderr,
        )
    return vart.Runner.create_runner(dpu_subgraphs[0], "run")


def get_fixpos_scale(tensor) -> float:
    """DPU tensors are quantized as fixed-point; fix_point gives the number
    of fractional bits. scale = 2**fix_point converts float <-> int8.
    """
    fix_point = tensor.get_attr("fix_point")
    return float(2**fix_point)


def read_axi_stream_window() -> np.ndarray:
    """Stub: read one 512-sample pre-filtered ECG window from the DSP IP's
    AXI-Stream output.

    Replace this with the real read path for your PL design, e.g.:
      - mmap a UIO device exposing an AXI-Stream FIFO / DMA S2MM buffer
      - read from a PYNQ-allocated DMA buffer if the DSP IP hands off via
        pynq.allocate + axi_dma, mirroring the pattern already used in
        src/fpga/runtime.py for the hls4ml deployment path
      - read a raw byte stream device node exposed by a custom driver

    Must return a float32 numpy array of shape (512,).
    """
    raise NotImplementedError(
        "Wire this up to your DSP IP's AXI-Stream output. "
        "Returning random data only for interface testing:"
    )


def quantize_input(window: np.ndarray, input_scale: float) -> np.ndarray:
    scaled = np.round(window * input_scale)
    return np.clip(scaled, -128, 127).astype(np.int8)


def dequantize_output(raw: np.ndarray, output_scale: float) -> np.ndarray:
    return raw.astype(np.float32) / output_scale


def run_inference(dpu_runner: vart.Runner, window: np.ndarray) -> np.ndarray:
    input_tensors = dpu_runner.get_input_tensors()
    output_tensors = dpu_runner.get_output_tensors()

    input_shape = tuple(input_tensors[0].dims)   # e.g. (1, 1, 1, 512)
    output_shape = tuple(output_tensors[0].dims)

    input_scale = get_fixpos_scale(input_tensors[0])
    output_scale = get_fixpos_scale(output_tensors[0])

    quantized = quantize_input(window, input_scale).reshape(input_shape)

    input_data = [np.empty(input_shape, dtype=np.int8, order="C")]
    output_data = [np.empty(output_shape, dtype=np.int8, order="C")]
    input_data[0][:] = quantized

    job_id = dpu_runner.execute_async(input_data, output_data)
    dpu_runner.wait(job_id)

    clean_int8 = output_data[0].reshape(-1)
    return dequantize_output(clean_int8, output_scale)


def run_golden_vectors(dpu_runner: vart.Runner, golden_dir: str, output_dir: str) -> None:
    """Run fixed golden inputs through the DPU and save each raw output.

    Only needs input_*.npy from generate_golden_vectors.py -- the matching
    expected_*.npy files (which require torch + the checkpoint) don't need
    to be transferred to the board at all. Comparison against those happens
    back on the laptop (compare_golden_results.py), which also keeps a
    persistent, timestamped record of each test run -- the board itself is
    treated as stateless/disposable here.
    """
    golden_path = Path(golden_dir)
    input_files = sorted(golden_path.glob("input_*.npy"))
    if not input_files:
        raise FileNotFoundError(f"No input_*.npy files found under {golden_dir}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for input_file in input_files:
        vector_id = input_file.stem.split("_", 1)[1]
        window = np.load(input_file).astype(np.float32)

        start = time.perf_counter()
        clean = run_inference(dpu_runner, window)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        np.save(out_path / f"output_{vector_id}.npy", clean)
        print(f"[{vector_id}] inference {elapsed_ms:.2f} ms, "
              f"dpu_range=[{clean.min():.4f}, {clean.max():.4f}], "
              f"saved to {out_path / f'output_{vector_id}.npy'}")

    print(f"\nWrote {len(input_files)} DPU outputs to {out_path}. "
          f"Copy this directory back to the laptop and run compare_golden_results.py.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ECG denoiser DPU model on ZCU104.")
    parser.add_argument("--xmodel", default="./compiled_model/ecg_denoiser.xmodel")
    parser.add_argument(
        "--test-input",
        action="store_true",
        help="Use synthetic input instead of read_axi_stream_window(), for interface testing.",
    )
    parser.add_argument(
        "--golden-dir",
        default=None,
        help="Directory of input_i.npy files from generate_golden_vectors.py. Runs each "
        "through the DPU and saves output_i.npy for each into --golden-output-dir.",
    )
    parser.add_argument(
        "--golden-output-dir",
        default="./golden_outputs",
        help="Where to save DPU outputs when using --golden-dir. Copy this back to the "
        "laptop and run compare_golden_results.py.",
    )
    parser.add_argument("--loop", type=int, default=1, help="Number of windows to process (ignored with --golden-dir).")
    args = parser.parse_args()

    dpu_runner = load_dpu_runner(args.xmodel)
    input_tensors = dpu_runner.get_input_tensors()
    output_tensors = dpu_runner.get_output_tensors()
    print(f"Input tensor:  name={input_tensors[0].name} dims={list(input_tensors[0].dims)}")
    print(f"Output tensor: name={output_tensors[0].name} dims={list(output_tensors[0].dims)}")

    if args.golden_dir:
        run_golden_vectors(dpu_runner, args.golden_dir, args.golden_output_dir)
        return

    for i in range(args.loop):
        if args.test_input:
            window = np.random.randn(INPUT_LENGTH).astype(np.float32) * 0.1
        else:
            window = read_axi_stream_window()

        start = time.perf_counter()
        clean = run_inference(dpu_runner, window)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        print(f"[{i}] inference {elapsed_ms:.2f} ms, output range [{clean.min():.4f}, {clean.max():.4f}]")


if __name__ == "__main__":
    main()
