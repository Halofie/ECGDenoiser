import numpy as np
import pytest

from src.fpga.runtime import RuntimeErrorBase
from src.fpga.serial_client import infer_window, read_exact
from src.fpga.uart_protocol import encode_window


def test_read_exact_handles_short_reads():
    chunks = iter([b"ab", b"c", b"def"])
    assert read_exact(lambda size: next(chunks), 6) == b"abcdef"


def test_read_exact_rejects_early_eof():
    with pytest.raises(RuntimeErrorBase, match="before packet completed"):
        read_exact(lambda size: b"", 3)


def test_infer_window_sends_and_validates_response():
    requests = []
    output = np.full(256, 7, dtype=np.int16)

    def write(packet):
        requests.append(packet)

    response = encode_window(2, output)
    result = infer_window(lambda size: response, write, 2, np.zeros(256, dtype=np.int16))
    assert len(requests) == 1
    assert np.array_equal(result, output)
