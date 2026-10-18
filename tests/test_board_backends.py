"""DpuBackend against a fake vart/xir that mimics the Vitis AI Python API shape."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vitis_ai_deployment" / "board"))

from ecg_board import backends  # noqa: E402
from ecg_board.pipeline import DenoisePipeline  # noqa: E402


class FakeTensor:
    def __init__(self, name, dims, fix_point):
        self.name = name
        self.dims = dims
        self.dtype = "xint8"
        self._fix_point = fix_point

    def get_attr(self, key):
        assert key == "fix_point"
        return self._fix_point


class FakeSubgraph:
    def __init__(self, name, device):
        self._name = name
        self._device = device

    def get_name(self):
        return self._name

    def has_attr(self, key):
        return key == "device"

    def get_attr(self, key):
        return self._device


class FakeRunner:
    """'DPU' that negates its int8 input, so data flow and scaling are observable."""

    def __init__(self, in_dims, out_dims, in_fp, out_fp):
        self.inputs = [FakeTensor("in", in_dims, in_fp)]
        self.outputs = [FakeTensor("out", out_dims, out_fp)]
        self.calls = 0

    def get_input_tensors(self):
        return self.inputs

    def get_output_tensors(self):
        return self.outputs

    def execute_async(self, inputs, outputs):
        assert inputs[0].dtype == np.int8 and inputs[0].flags["C_CONTIGUOUS"]
        assert tuple(inputs[0].shape) == tuple(self.inputs[0].dims)
        outputs[0][...] = np.clip(-inputs[0].astype(np.int16), -128, 127).reshape(outputs[0].shape)
        self.calls += 1
        return (self.calls, 0)

    def wait(self, job):
        assert job[1] == 0


@pytest.fixture
def fake_vitis(monkeypatch):
    config = {"in_dims": [1, 1, 512, 1], "out_dims": [1, 1, 512, 1], "in_fp": 6, "out_fp": 5,
              "devices": ["CPU", "DPU", "CPU"]}
    created = []

    class Graph:
        @staticmethod
        def deserialize(path):
            graph = types.SimpleNamespace()
            children = [FakeSubgraph("sg%d" % i, d) for i, d in enumerate(config["devices"])]
            root = types.SimpleNamespace(toposort_child_subgraph=lambda: children)
            graph.get_root_subgraph = lambda: root
            return graph

    def create_runner(subgraph, mode):
        assert mode == "run" and subgraph._device == "DPU"
        runner = FakeRunner(config["in_dims"], config["out_dims"], config["in_fp"], config["out_fp"])
        created.append(runner)
        return runner

    monkeypatch.setitem(sys.modules, "xir", types.SimpleNamespace(Graph=Graph))
    monkeypatch.setitem(sys.modules, "vart",
                        types.SimpleNamespace(Runner=types.SimpleNamespace(create_runner=create_runner)))
    monkeypatch.setattr(backends, "_GRAPHS", {})
    return config, created


@pytest.fixture
def xmodel(tmp_path):
    path = tmp_path / "ecg_denoiser.xmodel"
    path.write_bytes(b"fake")
    return str(path)


def test_dpu_backend_quantizes_runs_and_dequantizes(fake_vitis, xmodel):
    backend = backends.DpuBackend(xmodel)
    window = np.linspace(-1.0, 1.0, 512).astype(np.float32)
    out = backend.infer(window)
    # in: round(x * 2^6) clipped to int8; fake DPU negates; out: / 2^5
    quantized = np.clip(np.round(window * 64), -128, 127)
    expected = np.clip(-quantized, -128, 127) / 32.0
    np.testing.assert_allclose(out, expected)
    assert backend.last_infer_s >= 0.0
    info = backend.info()
    assert info["input"]["dims"] == [1, 1, 512, 1]
    assert info["input"]["fix_point"] == 6 and info["output"]["fix_point"] == 5
    assert info["dpu_subgraphs"] == 1


def test_dpu_backend_accepts_nchw_style_dims(fake_vitis, xmodel):
    config, _ = fake_vitis
    config["in_dims"] = config["out_dims"] = [1, 1, 1, 512]
    backends.DpuBackend(xmodel).infer(np.zeros(512, np.float32))


def test_dpu_backend_rejects_multichannel_io(fake_vitis, xmodel):
    config, _ = fake_vitis
    config["in_dims"] = [1, 1, 256, 2]
    with pytest.raises(RuntimeError, match="Unexpected DPU input tensor dims"):
        backends.DpuBackend(xmodel)


def test_dpu_backend_warns_on_cpu_fallback_subgraphs(fake_vitis, xmodel, capsys):
    config, _ = fake_vitis
    config["devices"] = ["DPU", "CPU", "DPU"]
    backends.DpuBackend(xmodel)
    assert "2 DPU subgraphs" in capsys.readouterr().err


def test_pipeline_restores_physical_units_with_identity_backend():
    pipeline = DenoisePipeline(backends.IdentityBackend())
    samples = (np.sin(np.linspace(0, 12, 512)) * 2500 + 300).astype(np.int16)
    result = pipeline.process(samples)
    assert result.samples.dtype == np.int16
    assert np.max(np.abs(result.samples.astype(int) - samples)) <= 1
    assert result.clipped == 0


def test_pipeline_rejects_wrong_length():
    with pytest.raises(ValueError):
        DenoisePipeline(backends.IdentityBackend()).process(np.zeros(256, np.int16))
