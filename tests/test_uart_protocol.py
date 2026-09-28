import numpy as np
import pytest

from src.fpga.uart_protocol import PACKET_SIZE, PacketError, decode_window, encode_window


def test_uart_packet_round_trip():
    samples = np.arange(256, dtype=np.int16)
    packet = encode_window(17, samples)
    sequence, decoded = decode_window(packet)
    assert len(packet) == PACKET_SIZE
    assert sequence == 17
    assert np.array_equal(decoded, samples)


def test_uart_packet_rejects_crc_corruption():
    packet = bytearray(encode_window(1, np.zeros(256, dtype=np.int16)))
    packet[20] ^= 0x01
    with pytest.raises(PacketError, match="CRC32"):
        decode_window(bytes(packet))


def test_uart_packet_rejects_wrong_sample_shape_or_dtype():
    with pytest.raises(PacketError, match="Expected int16"):
        encode_window(0, np.zeros(255, dtype=np.int16))
    with pytest.raises(PacketError, match="Expected int16"):
        encode_window(0, np.zeros(256, dtype=np.int32))
