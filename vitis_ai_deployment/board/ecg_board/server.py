"""Board-side request loop: read frame -> denoise -> reply, logging timing per window."""

from __future__ import annotations

import csv
import time

from . import protocol
from .protocol import ErrorCode, MsgType, ProtocolError, ReadTimeout, ResponseTiming
from .transport import TransportClosed

TIMING_FIELDS = ("sequence", "t_wall", "pre_us", "infer_us", "post_us", "board_us", "clipped")


def _us(seconds: float) -> int:
    return int(round(seconds * 1e6))


class DenoiseServer(object):
    def __init__(self, transport, pipeline, info: dict, timing_csv: str = None, log=print):
        self.transport = transport
        self.pipeline = pipeline
        self.info = info
        self.log = log
        self.reader = protocol.PacketReader(transport.read, max_body=protocol.MAX_BODY)
        self.stats = {"requests": 0, "errors_sent": 0, "crc_errors": 0, "pings": 0, "started": time.time()}
        self.latencies = {"pre_s": [], "infer_s": [], "post_s": [], "board_s": []}
        self._csv_handle = open(timing_csv, "w", newline="") if timing_csv else None
        self._csv = csv.writer(self._csv_handle) if self._csv_handle else None
        if self._csv:
            self._csv.writerow(TIMING_FIELDS)

    def _send(self, frame: bytes) -> None:
        self.transport.write(frame)

    def _send_error(self, sequence: int, code: int, message: str) -> None:
        self.stats["errors_sent"] += 1
        self._send(protocol.encode_error(sequence or 0, code, message))

    def handle(self, packet: protocol.Packet) -> bool:
        """Process one frame. Returns False when the server should stop."""
        if packet.msg_type == MsgType.REQUEST:
            t_start = time.perf_counter()
            try:
                samples = protocol.samples_from_bytes(packet.body)
                result = self.pipeline.process(samples)
            except ProtocolError as error:
                self._send_error(packet.sequence, error.code, str(error))
                return True
            except ValueError as error:
                self._send_error(packet.sequence, ErrorCode.BAD_LENGTH, str(error))
                return True
            except Exception as error:  # keep serving; report the failure to the host
                self._send_error(packet.sequence, ErrorCode.INFERENCE_FAILED, repr(error))
                return True
            board_s = time.perf_counter() - t_start
            timing = ResponseTiming(_us(result.pre_s), _us(result.infer_s), _us(result.post_s),
                                    _us(board_s), result.clipped)
            self._send(protocol.encode_response(packet.sequence, result.samples, timing))
            self.stats["requests"] += 1
            for key, value in zip(("pre_s", "infer_s", "post_s", "board_s"),
                                  (result.pre_s, result.infer_s, result.post_s, board_s)):
                self.latencies[key].append(value)
            if self._csv:
                self._csv.writerow((packet.sequence, "%.6f" % time.time()) + tuple(timing[:5]))
            return True
        if packet.msg_type == MsgType.PING:
            self.stats["pings"] += 1
            payload = dict(self.info)
            payload["stats"] = dict(self.stats, crc_errors=self.reader.crc_errors,
                                    discarded_bytes=self.reader.discarded_bytes)
            self._send(protocol.encode_json(MsgType.PONG, packet.sequence, payload))
            return True
        if packet.msg_type == MsgType.BYE:
            self._send(protocol.encode(MsgType.BYE, packet.sequence))
            return False
        self._send_error(packet.sequence, ErrorCode.UNSUPPORTED_TYPE, "Unsupported message type %d" % packet.msg_type)
        return True

    def serve(self, max_requests: int = None, idle_timeout: float = None) -> dict:
        """Serve until BYE, peer disconnect, ``max_requests`` or ``idle_timeout`` seconds of silence."""
        last_activity = time.monotonic()
        try:
            while max_requests is None or self.stats["requests"] < max_requests:
                try:
                    packet = self.reader.read_packet()
                except ReadTimeout:
                    if idle_timeout is not None and time.monotonic() - last_activity > idle_timeout:
                        self.log("Idle for %.0f s, stopping." % idle_timeout)
                        break
                    continue
                except ProtocolError as error:
                    self._send_error(error.sequence, error.code, str(error))
                    continue
                except TransportClosed:
                    self.log("Peer disconnected.")
                    break
                last_activity = time.monotonic()
                if not self.handle(packet):
                    self.log("BYE received, stopping.")
                    break
                if self._csv_handle and self.stats["requests"] % 100 == 0:
                    self._csv_handle.flush()
        finally:
            self.stats["crc_errors"] = self.reader.crc_errors
            self.stats["discarded_bytes"] = self.reader.discarded_bytes
            self.stats["stopped"] = time.time()
            if self._csv_handle:
                self._csv_handle.close()
        return self.stats
