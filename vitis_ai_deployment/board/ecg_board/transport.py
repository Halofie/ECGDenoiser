"""Byte transports with one contract: ``read(n)`` returns 1..n bytes, or b"" on timeout.

* Serial ports use pyserial when it is installed. The ZCU104 board image may
  not ship pyserial, so on Linux there is a dependency-free termios fallback.
* TCP is used to rehearse the full host<->board flow on one laptop (and is a
  fallback link over Ethernet if the UART cannot be freed on board day).

Specs accepted by ``open_transport``:
    /dev/ttyPS1, /dev/ttyUSB0, COM5     serial port
    tcp://HOST:PORT                     TCP client
"""

from __future__ import annotations

import os
import select
import socket
import sys
import time


class TransportClosed(EOFError):
    """The peer closed the connection."""


class Transport(object):
    description = "transport"

    def read(self, size: int) -> bytes:
        raise NotImplementedError

    def write(self, data: bytes) -> None:
        raise NotImplementedError

    def drain(self) -> None:
        """Block until written bytes have left the local buffer (no-op if unsupported)."""

    def reset_input(self) -> None:
        """Drop any bytes already received but not yet read."""

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class PySerialTransport(Transport):
    def __init__(self, device: str, baud: int, timeout: float):
        import serial  # noqa: WPS433 (optional dependency)

        # serial_for_url also accepts plain device names (COM7, /dev/ttyPS1) and loop:// for tests.
        self._port = serial.serial_for_url(device, baudrate=baud, timeout=timeout, write_timeout=max(timeout, 5.0))
        self.description = "serial:%s@%d (pyserial)" % (device, baud)

    def read(self, size: int) -> bytes:
        return self._port.read(size)

    def write(self, data: bytes) -> None:
        self._port.write(data)

    def drain(self) -> None:
        self._port.flush()

    def reset_input(self) -> None:
        self._port.reset_input_buffer()

    def close(self) -> None:
        self._port.close()


class TermiosSerialTransport(Transport):
    """Raw 8N1 serial on Linux without pyserial."""

    def __init__(self, device: str, baud: int, timeout: float):
        import termios

        self._termios = termios
        speed = getattr(termios, "B%d" % baud, None)
        if speed is None:
            raise ValueError("Baud rate %d is not supported by termios on this system" % baud)
        self._timeout = timeout
        self._fd = os.open(device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            attrs = termios.tcgetattr(self._fd)
            attrs[0] &= ~(termios.IGNBRK | termios.BRKINT | termios.PARMRK | termios.ISTRIP
                          | termios.INLCR | termios.IGNCR | termios.ICRNL | termios.IXON
                          | termios.IXOFF | termios.IXANY)
            attrs[1] &= ~termios.OPOST
            attrs[2] &= ~(termios.CSIZE | termios.PARENB | termios.CSTOPB | getattr(termios, "CRTSCTS", 0))
            attrs[2] |= termios.CS8 | termios.CREAD | termios.CLOCAL
            attrs[3] &= ~(termios.ECHO | termios.ECHONL | termios.ICANON | termios.ISIG | termios.IEXTEN)
            attrs[4] = speed
            attrs[5] = speed
            attrs[6][termios.VMIN] = 0
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(self._fd, termios.TCSANOW, attrs)
            termios.tcflush(self._fd, termios.TCIOFLUSH)
        except Exception:
            os.close(self._fd)
            raise
        self.description = "serial:%s@%d (termios)" % (device, baud)

    def read(self, size: int) -> bytes:
        deadline = time.monotonic() + self._timeout
        data = b""
        while len(data) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready, _, _ = select.select([self._fd], [], [], remaining)
            if not ready:
                break
            try:
                chunk = os.read(self._fd, size - len(data))
            except BlockingIOError:
                continue
            if chunk:
                data += chunk
        return data

    def write(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            select.select([], [self._fd], [], 5.0)
            try:
                written = os.write(self._fd, view)
            except BlockingIOError:
                continue
            view = view[written:]

    def drain(self) -> None:
        self._termios.tcdrain(self._fd)

    def reset_input(self) -> None:
        self._termios.tcflush(self._fd, self._termios.TCIFLUSH)

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


class SocketTransport(Transport):
    def __init__(self, sock: socket.socket, timeout: float, description: str):
        sock.settimeout(timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock
        self.description = description

    def read(self, size: int) -> bytes:
        try:
            chunk = self._sock.recv(size)
        except socket.timeout:
            return b""
        if not chunk:
            raise TransportClosed("Peer closed the TCP connection")
        return chunk

    def write(self, data: bytes) -> None:
        self._sock.sendall(data)

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


def open_serial(device: str, baud: int = 115200, timeout: float = 1.0) -> Transport:
    try:
        import serial  # noqa: F401
    except ImportError:
        if sys.platform.startswith("win"):
            raise RuntimeError("pyserial is required for serial ports on Windows: pip install pyserial")
        return TermiosSerialTransport(device, baud, timeout)
    return PySerialTransport(device, baud, timeout)


def connect_tcp(host: str, port: int, timeout: float = 1.0, connect_timeout: float = 10.0) -> Transport:
    sock = socket.create_connection((host, port), timeout=connect_timeout)
    return SocketTransport(sock, timeout, "tcp:%s:%d" % (host, port))


def listen_tcp(port: int, host: str = "0.0.0.0", timeout: float = 1.0, ready=None) -> Transport:
    """Block until one client connects. ``ready`` (callable) is invoked once listening."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((host, port))
        server.listen(1)
        if ready is not None:
            ready(server.getsockname()[1])
        conn, addr = server.accept()
    finally:
        server.close()
    return SocketTransport(conn, timeout, "tcp:%s:%d<-%s:%d" % (host, port, addr[0], addr[1]))


def open_transport(spec: str, baud: int = 115200, timeout: float = 1.0) -> Transport:
    if spec.startswith("tcp://"):
        host, _, port = spec[len("tcp://"):].rpartition(":")
        return connect_tcp(host or "127.0.0.1", int(port), timeout)
    return open_serial(spec, baud, timeout)
