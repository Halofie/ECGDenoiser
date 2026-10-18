"""Host-side client: send a noisy window, get the denoised window + timing back."""

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np

from . import protocol
from .protocol import MsgType, ProtocolError, ReadTimeout, ResponseTiming


class BoardError(RuntimeError):
    pass


class DenoiseReply(NamedTuple):
    samples: np.ndarray
    timing: ResponseTiming
    round_trip_s: float  # last attempt: first byte written -> full reply parsed
    attempts: int


class BoardClient(object):
    def __init__(self, transport, retries: int = 3, log=print):
        self.transport = transport
        self.retries = retries
        self.log = log
        self.reader = protocol.PacketReader(transport.read)
        self.sequence = 0
        self.stats = {"timeouts": 0, "board_errors": 0, "crc_errors": 0, "stale_replies": 0, "retries": 0}

    def _next_sequence(self) -> int:
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        return self.sequence

    def _await(self, sequence: int, expected_type: int) -> protocol.Packet:
        """Read until the reply for ``sequence`` arrives; raises ReadTimeout or BoardError."""
        while True:
            try:
                packet = self.reader.read_packet()
            except ProtocolError:
                self.stats["crc_errors"] += 1
                continue
            if packet.sequence != sequence:
                self.stats["stale_replies"] += 1  # reply to an earlier, retried request
                continue
            if packet.msg_type == MsgType.ERROR:
                code, message = protocol.decode_error(packet.body)
                self.stats["board_errors"] += 1
                raise BoardError("Board error %d: %s" % (code, message))
            if packet.msg_type != expected_type:
                raise BoardError("Expected message type %d, got %d" % (expected_type, packet.msg_type))
            return packet

    def _transact(self, msg_type: int, body: bytes, expected_type: int):
        sequence = self._next_sequence()
        frame = protocol.encode(msg_type, sequence, body)
        last_error = None
        for attempt in range(1, self.retries + 2):
            start = time.perf_counter()
            self.transport.write(frame)
            try:
                packet = self._await(sequence, expected_type)
                return packet, time.perf_counter() - start, attempt
            except ReadTimeout as error:
                self.stats["timeouts"] += 1
                last_error = error
            except BoardError as error:
                last_error = error
            if attempt <= self.retries:
                self.stats["retries"] += 1
        raise BoardError("No valid reply for sequence %d after %d attempts: %s"
                         % (sequence, self.retries + 1, last_error))

    def ping(self):
        """Return (board info dict, round trip seconds)."""
        packet, rtt, _ = self._transact(MsgType.PING, b"", MsgType.PONG)
        return protocol.decode_json(packet.body), rtt

    def denoise(self, samples: np.ndarray) -> DenoiseReply:
        body = np.asarray(samples, dtype=np.int16).astype("<i2").tobytes()
        packet, rtt, attempts = self._transact(MsgType.REQUEST, body, MsgType.RESPONSE)
        timing, output = protocol.decode_response(packet.body)
        return DenoiseReply(output, timing, rtt, attempts)

    def bye(self) -> None:
        try:
            self._transact(MsgType.BYE, b"", MsgType.BYE)
        except BoardError as error:
            self.log("BYE not acknowledged: %s" % error)
