from __future__ import annotations

import struct
import zlib

import numpy as np

MAGIC = b"EC"
VERSION = 1
SAMPLE_COUNT = 256
_HEADER = struct.Struct("<2sBBIH")
_CRC = struct.Struct("<I")
PACKET_SIZE = _HEADER.size + SAMPLE_COUNT * np.dtype("<i2").itemsize + _CRC.size


class PacketError(ValueError):
    """Raised when a UART packet violates the deployment protocol."""


def encode_window(sequence: int, samples: np.ndarray) -> bytes:
    """Encode one 256-sample signed-int16 ECG window with CRC32."""
    values = np.asarray(samples)
    if values.shape != (SAMPLE_COUNT,) or values.dtype != np.int16:
        raise PacketError(f"Expected int16 samples with shape ({SAMPLE_COUNT},)")
    if not 0 <= sequence <= 0xFFFFFFFF:
        raise PacketError("Sequence must fit in uint32")
    header = _HEADER.pack(MAGIC, VERSION, 0, sequence, SAMPLE_COUNT)
    payload = values.astype("<i2", copy=False).tobytes()
    return header + payload + _CRC.pack(zlib.crc32(header + payload) & 0xFFFFFFFF)


def decode_window(packet: bytes) -> tuple[int, np.ndarray]:
    """Validate and decode one complete UART ECG window packet."""
    if len(packet) != PACKET_SIZE:
        raise PacketError(f"Expected packet size {PACKET_SIZE}, got {len(packet)}")
    header = packet[: _HEADER.size]
    magic, version, flags, sequence, sample_count = _HEADER.unpack(header)
    if magic != MAGIC or version != VERSION or flags != 0:
        raise PacketError("Invalid packet header")
    if sample_count != SAMPLE_COUNT:
        raise PacketError(f"Expected {SAMPLE_COUNT} samples, got {sample_count}")
    payload_end = _HEADER.size + SAMPLE_COUNT * 2
    expected_crc = _CRC.unpack(packet[payload_end:])[0]
    actual_crc = zlib.crc32(packet[:payload_end]) & 0xFFFFFFFF
    if expected_crc != actual_crc:
        raise PacketError("CRC32 validation failed")
    samples = np.frombuffer(packet[_HEADER.size:payload_end], dtype="<i2").copy()
    return sequence, samples
