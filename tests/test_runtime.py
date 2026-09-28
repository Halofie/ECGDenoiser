import numpy as np
import pytest

from src.fpga.runtime import RuntimeErrorBase, process_uart_packet, process_window
from src.fpga.uart_protocol import decode_window, encode_window


class FakeChannel:
    def __init__(self, transform=None):
        self.buffer = None
        self.transform = transform

    def transfer(self, buffer):
        self.buffer = buffer
        if self.transform is not None:
            self.transform(buffer)

    def wait(self):
        return None


class FakeDma:
    def __init__(self):
        self.sendchannel = FakeChannel()

        def fill_output(buffer):
            buffer[:] = 0

        self.recvchannel = FakeChannel(transform=fill_output)


def test_process_window_returns_accelerator_output():
    output = process_window(FakeDma(), np.arange(256, dtype=np.int16))
    assert output.dtype == np.int16
    assert output.shape == (256,)
    assert np.all(output == 0)


def test_process_uart_packet_preserves_sequence():
    packet = encode_window(4, np.arange(256, dtype=np.int16))
    response = process_uart_packet(FakeDma(), packet, expected_sequence=4)
    sequence, samples = decode_window(response)
    assert sequence == 4
    assert np.all(samples == 0)


def test_process_uart_packet_rejects_unexpected_sequence():
    packet = encode_window(4, np.zeros(256, dtype=np.int16))
    with pytest.raises(RuntimeErrorBase, match="Unexpected sequence"):
        process_uart_packet(FakeDma(), packet, expected_sequence=3)
