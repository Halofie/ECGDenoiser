from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.fpga.runtime import RuntimeErrorBase
from src.fpga.uart_protocol import PACKET_SIZE, decode_window, encode_window


def read_exact(read: Callable[[int], bytes], size: int) -> bytes:
    """Read exactly ``size`` bytes or fail on an early serial EOF."""
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = read(remaining)
        if not chunk:
            raise RuntimeErrorBase(
                f"Serial stream ended before packet completed ({size - remaining}/{size})"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def infer_window(
    read: Callable[[int], bytes],
    write: Callable[[bytes], None],
    sequence: int,
    samples: np.ndarray,
) -> np.ndarray:
    """Send one window to the board and validate its response."""
    write(encode_window(sequence, samples))
    response = read_exact(read, PACKET_SIZE)
    response_sequence, output = decode_window(response)
    if response_sequence != sequence:
        raise RuntimeErrorBase(
            f"Unexpected response sequence {response_sequence}; expected {sequence}"
        )
    return output
