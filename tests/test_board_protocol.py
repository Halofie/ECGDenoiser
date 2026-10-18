import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vitis_ai_deployment" / "board"))

from ecg_board import preprocess, protocol  # noqa: E402
from ecg_board.protocol import MsgType, PacketReader, ProtocolError, ReadTimeout, ResponseTiming  # noqa: E402


def _reader(data: bytes, chunk: int = 4096) -> PacketReader:
    stream = bytearray(data)

    def read(size: int) -> bytes:
        out = bytes(stream[: min(size, chunk)])
        del stream[: len(out)]
        return out

    return PacketReader(read)


def _window(seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).normal(size=512) * 3000).astype(np.int16)


def test_frame_sizes_match_documented_wire_budget():
    assert protocol.REQUEST_SIZE == 1038
    assert protocol.RESPONSE_SIZE == 1058


def test_request_round_trip_through_byte_at_a_time_reader():
    samples = _window()
    packet = _reader(protocol.encode_request(7, samples), chunk=1).read_packet()
    assert packet.msg_type == MsgType.REQUEST
    assert packet.sequence == 7
    np.testing.assert_array_equal(protocol.samples_from_bytes(packet.body), samples)


def test_response_carries_timing_and_samples():
    samples = _window(1)
    timing = ResponseTiming(10, 2000, 30, 2100, 3)
    packet = _reader(protocol.encode_response(0xFFFFFFFF, samples, timing)).read_packet()
    decoded_timing, decoded = protocol.decode_response(packet.body)
    assert packet.sequence == 0xFFFFFFFF
    assert decoded_timing == timing
    np.testing.assert_array_equal(decoded, samples)


def test_reader_resyncs_after_garbage_and_false_magic():
    garbage = b"\x00login: root\r\nEC\x09junk" + b"E"
    frames = protocol.encode_request(1, _window(1)) + protocol.encode(MsgType.PING, 2)
    reader = _reader(garbage + frames, chunk=5)
    assert reader.read_packet().sequence == 1
    assert reader.read_packet().msg_type == MsgType.PING
    assert reader.discarded_bytes >= len(garbage)


def test_crc_error_reports_sequence_then_recovers():
    bad = bytearray(protocol.encode_request(5, _window()))
    bad[100] ^= 0xFF
    reader = _reader(bytes(bad) + protocol.encode_request(6, _window(2)))
    with pytest.raises(ProtocolError) as error:
        reader.read_packet()
    assert error.value.sequence == 5
    assert error.value.code == protocol.ErrorCode.BAD_CRC
    assert reader.read_packet().sequence == 6


def test_timeout_keeps_partial_frame_buffered():
    frame = protocol.encode_request(9, _window())
    stream = bytearray(frame[:300])

    def read(size):
        out = bytes(stream[:size])
        del stream[: len(out)]
        return out

    reader = PacketReader(read)
    with pytest.raises(ReadTimeout):
        reader.read_packet()
    stream.extend(frame[300:])
    assert reader.read_packet().sequence == 9


def test_error_and_json_messages():
    code, message = protocol.decode_error(_reader(protocol.encode_error(3, 4, "boom")).read_packet().body)
    assert (code, message) == (4, "boom")
    payload = protocol.decode_json(_reader(protocol.encode_json(MsgType.PONG, 1, {"a": 1})).read_packet().body)
    assert payload == {"a": 1}


def test_crc_covers_header():
    frame = protocol.encode(MsgType.PING, 1)
    assert zlib.crc32(frame[:-4]) & 0xFFFFFFFF == int.from_bytes(frame[-4:], "little")


def test_normalization_round_trip_and_degenerate_window():
    window = np.linspace(-2.0, 3.0, 512).astype(np.float32)
    for method in preprocess.METHODS:
        normalized, affine = preprocess.normalize(window, method)
        assert normalized.min() >= -1.0 and normalized.max() <= 1.0
        if method == "noisy_minmax":
            np.testing.assert_allclose(preprocess.invert_affine(normalized, affine), window, atol=1e-5)
    flat, _ = preprocess.normalize(np.full(512, 7.0, np.float32))
    assert np.all(flat == 0.0)


def test_robust_normalization_clips_spikes():
    window = np.sin(np.linspace(0, 20, 512)).astype(np.float32)
    window[100] = 50.0
    normalized, affine = preprocess.normalize(window, "noisy_robust", percentile=1.0)
    assert affine.half_span < 5.0  # the spike did not set the scale
    assert normalized[100] == 1.0


def test_to_int16_saturates_and_counts():
    values, clipped = preprocess.to_int16(np.array([40000.0, -40000.0, 1.4, -1.6]))
    assert values.tolist() == [32767, -32768, 1, -2]
    assert clipped == 2


def test_board_sources_parse_as_python37():
    """The board image may run an old Python; keep the board package 3.7-compatible."""
    import ast

    board_dir = Path(__file__).resolve().parents[1] / "vitis_ai_deployment" / "board"
    files = sorted(board_dir.rglob("*.py"))
    assert files
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 7))
