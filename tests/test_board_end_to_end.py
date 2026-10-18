"""Server <-> client over a real socket, power logging on a fake sysfs, and the offline session."""

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

BOARD_DIR = Path(__file__).resolve().parents[1] / "vitis_ai_deployment" / "board"
sys.path.insert(0, str(BOARD_DIR))

from ecg_board import power, protocol  # noqa: E402
from ecg_board.backends import IdentityBackend  # noqa: E402
from ecg_board.client import BoardClient, BoardError  # noqa: E402
from ecg_board.pipeline import DenoisePipeline  # noqa: E402
from ecg_board.server import DenoiseServer  # noqa: E402
from ecg_board.transport import SocketTransport  # noqa: E402


class FlakyBackend(IdentityBackend):
    """Fails the first ``failures`` calls, then behaves like identity."""

    def __init__(self, failures):
        super().__init__()
        self.failures = failures

    def infer(self, window):
        if self.failures:
            self.failures -= 1
            raise RuntimeError("simulated DPU fault")
        return super().infer(window)


def _start_server(backend, tmp_path):
    board_sock, host_sock = socket.socketpair()
    server = DenoiseServer(SocketTransport(board_sock, 0.2, "board"), DenoisePipeline(backend),
                           {"backend": backend.info()}, timing_csv=str(tmp_path / "timings.csv"), log=lambda *_: None)
    result = {}
    thread = threading.Thread(target=lambda: result.update(server.serve(idle_timeout=10)))
    thread.start()
    return server, thread, result, SocketTransport(host_sock, 2.0, "host")


def test_round_trip_ping_requests_and_bye(tmp_path):
    server, thread, stats, host = _start_server(IdentityBackend(), tmp_path)
    client = BoardClient(host, retries=1, log=lambda *_: None)
    info, _rtt = client.ping()
    assert info["backend"]["backend"] == "identity"
    for seed in range(5):
        samples = (np.random.default_rng(seed).normal(size=512) * 2000).astype(np.int16)
        reply = client.denoise(samples)
        assert np.max(np.abs(reply.samples.astype(int) - samples)) <= 1
        assert reply.timing.board_us >= reply.timing.infer_us
        assert reply.attempts == 1
    client.bye()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert stats["requests"] == 5
    rows = (tmp_path / "timings.csv").read_text().strip().splitlines()
    assert len(rows) == 6  # header + 5 windows
    assert len(server.latencies["infer_s"]) == 5


def test_line_noise_before_a_request_is_skipped(tmp_path):
    _server, thread, stats, host = _start_server(IdentityBackend(), tmp_path)
    host.write(b"\r\nkernel: random console chatter EC\x02\x00garbage")
    client = BoardClient(host, retries=2, log=lambda *_: None)
    reply = client.denoise(np.arange(512, dtype=np.int16))
    assert reply.samples.shape == (512,)
    client.bye()
    thread.join(timeout=5)
    assert stats["discarded_bytes"] > 0


def test_board_error_triggers_retry_then_succeeds(tmp_path):
    _server, thread, stats, host = _start_server(FlakyBackend(failures=1), tmp_path)
    client = BoardClient(host, retries=2, log=lambda *_: None)
    reply = client.denoise(np.zeros(512, np.int16))
    assert reply.attempts == 2
    assert client.stats["board_errors"] == 1
    client.bye()
    thread.join(timeout=5)
    assert stats["errors_sent"] == 1


def test_client_gives_up_after_retries(tmp_path):
    _server, thread, _stats, host = _start_server(FlakyBackend(failures=10), tmp_path)
    client = BoardClient(host, retries=1, log=lambda *_: None)
    with pytest.raises(BoardError, match="after 2 attempts"):
        client.denoise(np.zeros(512, np.int16))
    client.bye()
    thread.join(timeout=5)


def _fake_hwmon(root: Path) -> Path:
    ina = root / "hwmon0"
    ina.mkdir(parents=True)
    (ina / "name").write_text("ina226\n")
    (ina / "power1_input").write_text("1500000\n")  # 1.5 W
    pmbus = root / "hwmon1"
    pmbus.mkdir()
    (pmbus / "name").write_text("irps5401\n")
    (pmbus / "power1_input").write_text("2000000\n")
    (pmbus / "power1_label").write_text("pout1\n")
    (pmbus / "power2_input").write_text("250000\n")
    (pmbus / "temp1_input").write_text("41000\n")
    return root


def test_power_discovery_logging_and_rail_map(tmp_path):
    root = _fake_hwmon(tmp_path / "hwmon")
    channels = power.discover_channels(str(root))
    assert [c.key for c in channels] == ["ina226@hwmon0/power1", "irps5401@hwmon1/power1", "irps5401@hwmon1/power2"]
    assert channels[1].label == "pout1"

    rail_map = tmp_path / "rails.json"
    rail_map.write_text(json.dumps({"names": {"ina226@hwmon0/power1": "VCCINT"},
                                    "total": ["ina226@hwmon0/power1", "irps5401@hwmon1/power1"]}))
    logger = power.PowerLogger(str(tmp_path / "p.csv"), str(root), interval_s=0.01, rail_map_path=str(rail_map))
    logger.start("idle")
    time.sleep(0.08)
    (root / "hwmon0" / "power1_input").write_text("2500000\n")
    logger.set_phase("busy")
    time.sleep(0.08)
    logger.stop()
    summary = logger.summary()
    assert summary["phases"]["idle"]["VCCINT"]["mean_w"] == pytest.approx(1.5)
    assert summary["phases"]["idle"]["total"]["mean_w"] == pytest.approx(3.5)
    assert summary["phases"]["busy"]["total"]["mean_w"] == pytest.approx(4.5)
    header = (tmp_path / "p.csv").read_text().splitlines()[0]
    assert header.startswith("t_wall,phase,VCCINT,")


def test_offline_board_session_with_identity_backend(tmp_path):
    import board_session

    golden = tmp_path / "golden"
    golden.mkdir()
    for i in range(2):
        np.save(golden / f"input_{i}.npy", np.linspace(-1, 1, 512, dtype=np.float32) * (i + 1) / 2)
    out_dir = tmp_path / "session"
    session = board_session.main([
        "--backend", "identity", "--golden-dir", str(golden), "--out-dir", str(out_dir),
        "--hwmon-root", str(_fake_hwmon(tmp_path / "hwmon")), "--power-interval", "0.01",
        "--idle-seconds", "0.1", "--settle-seconds", "0.02", "--latency-iterations", "20",
        "--throughput-seconds", "0.1", "--threads", "1,2",
    ])
    assert session["golden"]["count"] == 2
    np.testing.assert_allclose(np.load(out_dir / "golden_outputs" / "output_1.npy"),
                               np.load(golden / "input_1.npy"))
    assert session["latency"]["infer"]["count"] == 20
    assert [t["threads"] for t in session["throughput"]] == [1, 2]
    assert all(t["windows_per_s"] > 0 and not t["errors"] for t in session["throughput"])
    assert session["energy"]["available"]
    assert session["energy"]["per_threads"][0]["energy_per_window_total_mj"] > 0
    assert (out_dir / "session.json").is_file()
    assert Path(str(out_dir) + ".tar.gz").is_file()


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="needs a POSIX pseudo-terminal")
def test_termios_serial_transport_over_pty():
    from ecg_board.transport import TermiosSerialTransport

    master, slave = os.openpty()
    port = TermiosSerialTransport(os.ttyname(slave), 115200, timeout=0.5)
    frame = protocol.encode_request(1, np.arange(512, dtype=np.int16))
    os.write(master, frame)
    packet = protocol.PacketReader(port.read).read_packet()
    assert packet.sequence == 1
    port.write(b"hello")
    assert os.read(master, 5) == b"hello"
    port.close()


def test_pyserial_transport_loopback():
    pytest.importorskip("serial")
    from ecg_board.transport import open_serial

    port = open_serial("loop://", 115200, timeout=0.2)
    assert "pyserial" in port.description
    port.write(protocol.encode_request(3, np.arange(512, dtype=np.int16)))
    packet = protocol.PacketReader(port.read).read_packet()
    assert packet.sequence == 3
    assert port.read(10) == b""  # timeout -> empty, as the reader expects
    port.close()
