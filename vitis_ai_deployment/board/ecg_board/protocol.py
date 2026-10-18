"""Framed binary protocol between the host PC and the ZCU104 (UART or TCP).

Version 2 of the packet format in src/fpga/uart_protocol.py, generalized to
512-sample windows and extended with message types so the board can report
its own timing and errors. Every frame is:

    magic     2 bytes   b"EC"
    version   u8        2
    msg_type  u8        see MsgType
    sequence  u32 LE    echoed back in the reply
    body_len  u16 LE    number of body bytes that follow
    body      body_len bytes
    crc32     u32 LE    zlib.crc32 over header + body

Bodies:
    REQUEST   host->board  N x int16 LE noisy samples (N = 512)
    RESPONSE  board->host  ResponseTiming (20 bytes) + N x int16 LE denoised samples
    ERROR     board->host  u16 error code + UTF-8 message
    PING      host->board  empty
    PONG      board->host  UTF-8 JSON board/model info
    BYE       either way   empty; the board replies BYE and stops serving

A REQUEST frame is 1038 bytes and a RESPONSE frame 1058 bytes, so one round
trip moves 20,960 bits: ~182 ms of pure wire time at 115200 baud, ~23 ms at
921600 baud.
"""

from __future__ import annotations

import json
import struct
import zlib
from typing import Callable, NamedTuple, Optional

import numpy as np

MAGIC = b"EC"
VERSION = 2
WINDOW_LENGTH = 512
MAX_BODY = 4096

_HEADER = struct.Struct("<2sBBIH")
_CRC = struct.Struct("<I")
_ERROR = struct.Struct("<H")
HEADER_SIZE = _HEADER.size
CRC_SIZE = _CRC.size


class MsgType:
    REQUEST = 0x01
    RESPONSE = 0x02
    ERROR = 0x03
    PING = 0x04
    PONG = 0x05
    BYE = 0x06

    ALL = (REQUEST, RESPONSE, ERROR, PING, PONG, BYE)


class ErrorCode:
    BAD_CRC = 1
    BAD_HEADER = 2
    BAD_LENGTH = 3
    INFERENCE_FAILED = 4
    UNSUPPORTED_TYPE = 5


class ResponseTiming(NamedTuple):
    """Board-side timing for one window, in microseconds."""

    pre_us: int  # int16 -> float, normalize, quantize
    infer_us: int  # DPU execute_async + wait (or host backend forward)
    post_us: int  # dequantize, de-normalize, -> int16
    board_us: int  # request decoded -> response encoded (everything on the board but the wire)
    clipped: int  # output samples saturated to int16
    reserved: int = 0


_TIMING = struct.Struct("<IIIIHH")


class Packet(NamedTuple):
    msg_type: int
    sequence: int
    body: bytes


class ProtocolError(ValueError):
    """A frame was received but is invalid. ``sequence`` is set when the header parsed."""

    def __init__(self, message: str, code: int, sequence: Optional[int] = None):
        super().__init__(message)
        self.code = code
        self.sequence = sequence


class ReadTimeout(TimeoutError):
    """No complete frame arrived before the transport timed out."""


def frame_size(body_len: int) -> int:
    return HEADER_SIZE + body_len + CRC_SIZE


REQUEST_SIZE = frame_size(WINDOW_LENGTH * 2)
RESPONSE_SIZE = frame_size(_TIMING.size + WINDOW_LENGTH * 2)


def encode(msg_type: int, sequence: int, body: bytes = b"") -> bytes:
    if msg_type not in MsgType.ALL:
        raise ValueError("Unknown message type %r" % msg_type)
    if not 0 <= sequence <= 0xFFFFFFFF:
        raise ValueError("Sequence must fit in uint32")
    if len(body) > MAX_BODY:
        raise ValueError("Body too large: %d > %d" % (len(body), MAX_BODY))
    header = _HEADER.pack(MAGIC, VERSION, msg_type, sequence, len(body))
    return header + body + _CRC.pack(zlib.crc32(header + body) & 0xFFFFFFFF)


def _samples_to_bytes(samples: np.ndarray) -> bytes:
    values = np.asarray(samples)
    if values.ndim != 1 or values.dtype != np.int16:
        raise ValueError("Samples must be a 1D int16 array, got %s %s" % (values.dtype, values.shape))
    return values.astype("<i2", copy=False).tobytes()


def samples_from_bytes(body: bytes) -> np.ndarray:
    if len(body) % 2:
        raise ProtocolError("Odd sample payload length %d" % len(body), ErrorCode.BAD_LENGTH)
    return np.frombuffer(body, dtype="<i2").astype(np.int16)


def encode_request(sequence: int, samples: np.ndarray) -> bytes:
    return encode(MsgType.REQUEST, sequence, _samples_to_bytes(samples))


def encode_response(sequence: int, samples: np.ndarray, timing: ResponseTiming) -> bytes:
    clamp = [max(0, min(int(v), 0xFFFFFFFF)) for v in timing[:4]]
    small = [max(0, min(int(v), 0xFFFF)) for v in timing[4:]]
    return encode(MsgType.RESPONSE, sequence, _TIMING.pack(*(clamp + small)) + _samples_to_bytes(samples))


def decode_response(body: bytes):
    """Return (ResponseTiming, int16 samples)."""
    if len(body) < _TIMING.size:
        raise ProtocolError("Response body too short", ErrorCode.BAD_LENGTH)
    timing = ResponseTiming(*_TIMING.unpack(body[: _TIMING.size]))
    return timing, samples_from_bytes(body[_TIMING.size :])


def encode_error(sequence: int, code: int, message: str) -> bytes:
    text = message.encode("utf-8", "replace")[: MAX_BODY - _ERROR.size]
    return encode(MsgType.ERROR, sequence, _ERROR.pack(code) + text)


def decode_error(body: bytes):
    """Return (code, message)."""
    if len(body) < _ERROR.size:
        return 0, ""
    return _ERROR.unpack(body[: _ERROR.size])[0], body[_ERROR.size :].decode("utf-8", "replace")


def encode_json(msg_type: int, sequence: int, payload: dict) -> bytes:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(body) > MAX_BODY:
        body = json.dumps({"truncated": True}).encode("utf-8")
    return encode(msg_type, sequence, body)


def decode_json(body: bytes) -> dict:
    return json.loads(body.decode("utf-8")) if body else {}


class PacketReader:
    """Incremental frame parser over a ``read(n) -> bytes`` callable.

    ``read`` may return fewer than ``n`` bytes; it returns b"" on timeout.
    Garbage between frames (console noise, a half-sent frame from an aborted
    run) is skipped by scanning for the next valid magic + header + CRC, so
    the link recovers by itself instead of staying misaligned forever.
    """

    def __init__(self, read: Callable[[int], bytes], max_body: int = MAX_BODY):
        self._read = read
        self._max_body = max_body
        self._buffer = bytearray()
        self.discarded_bytes = 0
        self.crc_errors = 0

    def _fill(self, size: int) -> None:
        while len(self._buffer) < size:
            chunk = self._read(size - len(self._buffer))
            if not chunk:
                raise ReadTimeout("Timed out with %d buffered bytes" % len(self._buffer))
            self._buffer.extend(chunk)

    def _discard(self, count: int) -> None:
        del self._buffer[:count]
        self.discarded_bytes += count

    def read_packet(self) -> Packet:
        while True:
            self._fill(2)
            start = self._buffer.find(MAGIC)
            if start < 0:
                # Keep a trailing b"E" in case it is the first half of the magic.
                keep = 1 if self._buffer.endswith(MAGIC[:1]) else 0
                self._discard(len(self._buffer) - keep)
                continue
            if start:
                self._discard(start)
            self._fill(HEADER_SIZE)
            magic, version, msg_type, sequence, body_len = _HEADER.unpack(bytes(self._buffer[:HEADER_SIZE]))
            if version != VERSION or msg_type not in MsgType.ALL or body_len > self._max_body:
                self._discard(1)  # false magic; rescan from the next byte
                continue
            total = frame_size(body_len)
            self._fill(total)
            frame = bytes(self._buffer[:total])
            expected = _CRC.unpack(frame[-CRC_SIZE:])[0]
            if zlib.crc32(frame[:-CRC_SIZE]) & 0xFFFFFFFF != expected:
                self.crc_errors += 1
                self._discard(1)
                raise ProtocolError("CRC32 mismatch", ErrorCode.BAD_CRC, sequence)
            del self._buffer[:total]
            return Packet(msg_type, sequence, frame[HEADER_SIZE:-CRC_SIZE])
