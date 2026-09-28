from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from src.fpga.uart_protocol import PACKET_SIZE, decode_window, encode_window


class DmaChannel(Protocol):
    def transfer(self, buffer: np.ndarray) -> None: ...

    def wait(self) -> None: ...


class DmaEngine(Protocol):
    sendchannel: DmaChannel
    recvchannel: DmaChannel


class RuntimeErrorBase(RuntimeError):
    """Base error for board runtime failures."""


def process_window(dma: DmaEngine, samples: np.ndarray) -> np.ndarray:
    """Transfer one int16 window through an AXI DMA-backed accelerator.

    The buffers are deliberately allocated by the caller's NumPy runtime in
    this abstraction. On PYNQ, replace them with ``pynq.allocate`` buffers.
    """
    values = np.asarray(samples)
    if values.shape != (256,) or values.dtype != np.int16:
        raise RuntimeErrorBase("Runtime requires an int16 array with 256 samples")
    input_buffer = np.array(values, dtype=np.int16, copy=True)
    output_buffer = np.empty(256, dtype=np.int16)
    try:
        dma.recvchannel.transfer(output_buffer)
        dma.sendchannel.transfer(input_buffer)
        dma.sendchannel.wait()
        dma.recvchannel.wait()
    except Exception as error:
        raise RuntimeErrorBase("AXI DMA inference failed") from error
    return output_buffer.copy()


def process_uart_packet(
    dma: DmaEngine,
    packet: bytes,
    *,
    expected_sequence: int | None = None,
) -> bytes:
    """Decode, infer, and encode one UART packet."""
    sequence, samples = decode_window(packet)
    if expected_sequence is not None and sequence != expected_sequence:
        raise RuntimeErrorBase(
            f"Unexpected sequence {sequence}; expected {expected_sequence}"
        )
    output = process_window(dma, samples)
    return encode_window(sequence, output)


def serve_uart(
    dma: DmaEngine,
    read_exact: Callable[[int], bytes],
    write: Callable[[bytes], None],
) -> None:
    """Run a blocking packet server over a UART-like byte transport."""
    expected_sequence = 0
    while True:
        packet = read_exact(PACKET_SIZE)
        if len(packet) != PACKET_SIZE:
            raise RuntimeErrorBase(
                f"UART closed with incomplete packet ({len(packet)} bytes)"
            )
        response = process_uart_packet(
            dma, packet, expected_sequence=expected_sequence
        )
        write(response)
        expected_sequence = (expected_sequence + 1) & 0xFFFFFFFF
