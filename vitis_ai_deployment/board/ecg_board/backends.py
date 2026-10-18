"""Inference backends: one interface, swappable between board and laptop.

    backend.infer(window_float32[512]) -> float32[512]   (normalized domain)
    backend.last_infer_s                                 time of the last call
    backend.info()                                       JSON-able description

DpuBackend   -- the real thing: VART runner on the ZCU104 DPU (board only).
TorchBackend -- float PyTorch model on the laptop, so the full UART/TCP flow
                can be rehearsed before the board is available.
IdentityBackend -- returns its input; for link/protocol tests without a model.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time

import numpy as np

WINDOW_LENGTH = 512


def file_md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Backend(object):
    name = "backend"

    def __init__(self):
        self.last_infer_s = 0.0

    def infer(self, window: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def info(self) -> dict:
        return {"backend": self.name}

    def close(self) -> None:
        pass


class IdentityBackend(Backend):
    name = "identity"

    def infer(self, window: np.ndarray) -> np.ndarray:
        start = time.perf_counter()
        out = np.asarray(window, dtype=np.float32).copy()
        self.last_infer_s = time.perf_counter() - start
        return out


def _dpu_subgraphs(graph) -> list:
    children = graph.get_root_subgraph().toposort_child_subgraph()
    return [s for s in children if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]


_GRAPHS = {}  # keep xir graphs alive for as long as runners built from them exist


def load_dpu_subgraph(xmodel_path: str):
    """Return (graph, dpu_subgraph, num_dpu_subgraphs)."""
    import xir

    if xmodel_path not in _GRAPHS:
        _GRAPHS[xmodel_path] = xir.Graph.deserialize(xmodel_path)
    graph = _GRAPHS[xmodel_path]
    subgraphs = _dpu_subgraphs(graph)
    if not subgraphs:
        raise RuntimeError("No DPU subgraph found in %s" % xmodel_path)
    return graph, subgraphs[0], len(subgraphs)


class DpuBackend(Backend):
    """VART runner for the compiled ECGResUNet1D xmodel.

    Layout note: xmodel tensors are NHWC. The PyTorch input (N, C=1, H=1,
    W=512) therefore shows up as dims [1, 1, 512, 1] on the board, not
    [1, 1, 1, 512]. Because C == 1 (and H == 1) the 512 values are contiguous
    in either layout, so a flat copy is exact. This is checked at load time;
    a model with C > 1 at the I/O would need a real transpose.
    """

    name = "dpu"

    def __init__(self, xmodel_path: str):
        super().__init__()
        import vart

        self.xmodel_path = xmodel_path
        _graph, subgraph, self.num_dpu_subgraphs = load_dpu_subgraph(xmodel_path)
        if self.num_dpu_subgraphs != 1:
            sys.stderr.write(
                "WARNING: %d DPU subgraphs in %s; using the first. A fully offloaded model has "
                "exactly one -- extra subgraphs mean ops fell back to the CPU at compile time.\n"
                % (self.num_dpu_subgraphs, xmodel_path)
            )
        self.subgraph_name = subgraph.get_name() if hasattr(subgraph, "get_name") else "?"
        self.runner = vart.Runner.create_runner(subgraph, "run")
        in_tensor = self.runner.get_input_tensors()[0]
        out_tensor = self.runner.get_output_tensors()[0]
        self.in_dims = tuple(int(d) for d in in_tensor.dims)
        self.out_dims = tuple(int(d) for d in out_tensor.dims)
        self.in_name, self.out_name = in_tensor.name, out_tensor.name
        for label, dims in (("input", self.in_dims), ("output", self.out_dims)):
            if [d for d in dims[1:] if d != 1] != [WINDOW_LENGTH]:
                raise RuntimeError(
                    "Unexpected DPU %s tensor dims %s: expected batch x 512 samples with every other "
                    "dim 1 (NHWC [N,1,512,1])." % (label, list(dims))
                )
        self.batch = self.in_dims[0]
        self.in_fix_point = int(in_tensor.get_attr("fix_point"))
        self.out_fix_point = int(out_tensor.get_attr("fix_point"))
        self.in_scale = float(2 ** self.in_fix_point)
        self.out_scale = float(2 ** self.out_fix_point)
        self.in_dtype = str(getattr(in_tensor, "dtype", "xint8"))
        self.out_dtype = str(getattr(out_tensor, "dtype", "xint8"))
        self._inputs = [np.zeros(self.in_dims, dtype=np.int8, order="C")]
        self._outputs = [np.zeros(self.out_dims, dtype=np.int8, order="C")]
        self._in_flat = self._inputs[0].reshape(self.batch, WINDOW_LENGTH)
        self._out_flat = self._outputs[0].reshape(self.batch, WINDOW_LENGTH)

    def quantize(self, window: np.ndarray) -> np.ndarray:
        scaled = np.round(np.asarray(window, dtype=np.float32) * self.in_scale)
        return np.clip(scaled, -128, 127).astype(np.int8)

    def dequantize(self, raw: np.ndarray) -> np.ndarray:
        return raw.astype(np.float32) / self.out_scale

    def run_int8(self, quantized: np.ndarray) -> np.ndarray:
        """DPU only: int8[512] in -> int8[512] out. Times execute_async + wait."""
        self._in_flat[0, :] = quantized
        start = time.perf_counter()
        job = self.runner.execute_async(self._inputs, self._outputs)
        self.runner.wait(job)
        self.last_infer_s = time.perf_counter() - start
        return self._out_flat[0].copy()

    def infer(self, window: np.ndarray) -> np.ndarray:
        return self.dequantize(self.run_int8(self.quantize(window)))

    def info(self) -> dict:
        return {
            "backend": self.name,
            "xmodel": os.path.basename(self.xmodel_path),
            "xmodel_md5": file_md5(self.xmodel_path),
            "dpu_subgraphs": self.num_dpu_subgraphs,
            "subgraph": self.subgraph_name,
            "input": {"name": self.in_name, "dims": list(self.in_dims), "fix_point": self.in_fix_point,
                      "dtype": self.in_dtype},
            "output": {"name": self.out_name, "dims": list(self.out_dims), "fix_point": self.out_fix_point,
                       "dtype": self.out_dtype},
        }


class TorchBackend(Backend):
    """Float ECGResUNet1D on the laptop (needs torch + vitis_ai_deployment/model.py).

    ``io_fix_points=(in_fp, out_fp)`` additionally rounds the input and output
    to int8 with those fix points, mimicking the DPU's I/O quantization (not
    its internal INT8 arithmetic).
    """

    name = "torch"

    def __init__(self, checkpoint: str, device: str = "cpu", io_fix_points=None):
        super().__init__()
        import torch

        model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from model import ECGResUNet1D, checkpoint_normalization, load_checkpoint

        self._torch = torch
        self.checkpoint = checkpoint
        self.device = device
        self.normalization = checkpoint_normalization(checkpoint)
        self.model = load_checkpoint(ECGResUNet1D(), checkpoint, map_location=device).to(device)
        self.io_fix_points = tuple(io_fix_points) if io_fix_points else None

    def _fake_quant(self, values: np.ndarray, fix_point: int) -> np.ndarray:
        scale = float(2 ** fix_point)
        return np.clip(np.round(values * scale), -128, 127).astype(np.float32) / scale

    def infer(self, window: np.ndarray) -> np.ndarray:
        values = np.asarray(window, dtype=np.float32)
        if self.io_fix_points:
            values = self._fake_quant(values, self.io_fix_points[0])
        tensor = self._torch.from_numpy(values.reshape(1, 1, 1, WINDOW_LENGTH)).to(self.device)
        start = time.perf_counter()
        with self._torch.no_grad():
            out = self.model(tensor)
            if self.device != "cpu":
                self._torch.cuda.synchronize()
        self.last_infer_s = time.perf_counter() - start
        result = out.reshape(-1).cpu().numpy().astype(np.float32)
        if self.io_fix_points:
            result = self._fake_quant(result, self.io_fix_points[1])
        return result

    def info(self) -> dict:
        return {
            "backend": self.name,
            "checkpoint": os.path.basename(self.checkpoint),
            "checkpoint_md5": file_md5(self.checkpoint),
            "checkpoint_normalization": self.normalization,
            "device": self.device,
            "io_fix_points": list(self.io_fix_points) if self.io_fix_points else None,
        }


def create_backend(kind: str, xmodel: str = None, checkpoint: str = None, device: str = "cpu",
                   io_fix_points=None) -> Backend:
    if kind == "dpu":
        if not xmodel:
            raise ValueError("--xmodel is required for the dpu backend")
        return DpuBackend(xmodel)
    if kind == "torch":
        if not checkpoint:
            raise ValueError("--checkpoint is required for the torch backend")
        return TorchBackend(checkpoint, device=device, io_fix_points=io_fix_points)
    if kind == "identity":
        return IdentityBackend()
    raise ValueError("Unknown backend %r" % kind)
