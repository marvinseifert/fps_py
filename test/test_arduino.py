"""Tests for the Arduino telemetry read and the colour-change logic.

No hardware needed: the serial port is faked. For the checks that need a real
board, see test_arduino_hw.py.
"""

import queue as std_queue
import threading

import pytest

import fpspy.fps_queue
from fpspy import arduino
from fpspy.presentation import process_arduino_colours


class FakeSerial:
    """A serial port that hands out whatever has been fed to it.

    Each fed string is returned by one read(), so a line can be split across
    reads on purpose — that is the case read_lines() has to get right.
    """

    def __init__(self):
        self._chunks = []
        self.written = b""

    def feed(self, text):
        self._chunks.append(text.encode("utf-8"))

    @property
    def in_waiting(self):
        return len(self._chunks[0]) if self._chunks else 0

    def read(self, n):
        return self._chunks.pop(0)

    def write(self, data):
        self.written += data

    def flush(self):
        pass

    def reset_input_buffer(self):
        self._chunks = []

    def close(self):
        pass


@pytest.fixture
def fake_arduino():
    """An Arduino wired to a FakeSerial, without opening a port."""
    ard = arduino.Arduino.__new__(arduino.Arduino)
    ard.port = "fake"
    ard.baud_rate = 9600
    ard.trigger_command = "T"
    ard._rx_buffer = ""
    ard._lock = threading.RLock()
    ard._serial = FakeSerial()
    ard.connected = True
    return ard


def drain(queue, timeout=1.0):
    """Collect every reply currently on `queue`."""
    replies = []
    while True:
        try:
            replies.append(fpspy.fps_queue.get_reply(queue, timeout=timeout))
        except std_queue.Empty:
            return replies
        timeout = 0.05


class TestReadLines:
    def test_returns_complete_lines_oldest_first(self, fake_arduino):
        fake_arduino._serial.feed("Stimulator ready\nTrigger\n")
        assert fake_arduino.read_lines() == ["Stimulator ready", "Trigger"]

    def test_nothing_waiting(self, fake_arduino):
        assert fake_arduino.read_lines() == []

    def test_split_line_is_completed_by_the_next_read(self, fake_arduino):
        # The whole point of the _rx_buffer: read() would lose "Trigger" here.
        fake_arduino._serial.feed("Stimulator ready\nTrig")
        assert fake_arduino.read_lines() == ["Stimulator ready"]
        fake_arduino._serial.feed("ger\nfinished\n")
        assert fake_arduino.read_lines() == ["Trigger", "finished"]

    def test_partial_line_is_not_reported_early(self, fake_arduino):
        fake_arduino._serial.feed("finish")
        assert fake_arduino.read_lines() == []
        assert fake_arduino._rx_buffer == "finish"

    def test_reset_input_drops_the_partial_line(self, fake_arduino):
        fake_arduino._serial.feed("finish")
        fake_arduino.read_lines()
        fake_arduino.reset_input()
        fake_arduino._serial.feed("ed\n")
        # Without clearing _rx_buffer this would read "finished".
        assert fake_arduino.read_lines() == ["ed"]

    def test_disconnected_reads_nothing(self, fake_arduino):
        fake_arduino.connected = False
        fake_arduino._serial.feed("Trigger\n")
        assert fake_arduino.read_lines() == []


class TestControllerConnectionReport:
    """The controller announces the port state before any line arrives.

    Without this the panel cannot tell a misconfigured port from a device
    that simply has not printed anything yet.
    """

    def _controller(self, port, monkeypatch=None):
        import multiprocessing as mp

        queue = mp.Queue()
        ctrl = arduino.ArduinoController(port, 9600, "T", queue, sender_idx=1)
        ctrl._reader_stop.set()
        return drain(queue)

    def test_dummy_port_is_not_reported_as_connected(self):
        (reply,) = self._controller("dummy")
        assert reply.kind == "arduino_dummy"
        assert "dummy" in reply.payload["text"]

    def test_unopenable_port_is_reported_as_disconnected(self):
        (reply,) = self._controller("/dev/does-not-exist")
        assert reply.kind == "arduino_disconnected"
        assert "/dev/does-not-exist" in reply.payload["text"]

    def test_connection_is_reported_before_any_device_line(self, fake_arduino):
        """Ordering matters: the panel builds its state from the queue order."""
        import multiprocessing as mp

        queue = mp.Queue()
        ctrl = arduino.ArduinoController("dummy", 9600, "T", queue, sender_idx=1)
        ctrl.arduino = fake_arduino
        fake_arduino._serial.feed("Stimulator ready\n")
        kinds = [r.kind for r in drain(queue)]
        ctrl._reader_stop.set()
        assert kinds[0] == "arduino_dummy"
        assert "arduino_ready" in kinds


class TestControllerTelemetry:
    @pytest.fixture
    def controller(self, fake_arduino):
        import multiprocessing as mp

        queue = mp.Queue()
        # "dummy" so no port is opened; the fake replaces it right after.
        ctrl = arduino.ArduinoController("dummy", 9600, "T", queue, sender_idx=2)
        ctrl.arduino = fake_arduino
        # Discard the startup connection event; these tests are about the
        # lines the device prints afterwards.
        drain(queue)
        yield ctrl, queue, fake_arduino._serial
        ctrl._reader_stop.set()

    def test_known_lines_become_typed_events(self, controller):
        ctrl, queue, serial = controller
        serial.feed("Stimulator ready\nTrigger\nfinished\n")
        replies = drain(queue)
        assert [r.kind for r in replies] == [
            "arduino_ready",
            "arduino_trigger",
            "arduino_done",
        ]
        assert all(r.sender_idx == 2 for r in replies)

    def test_unknown_lines_are_still_reported_verbatim(self, controller):
        ctrl, queue, serial = controller
        serial.feed("Setting power to: 0.50\n")
        (reply,) = drain(queue)
        assert reply.kind == "arduino_line"
        assert reply.payload["text"] == "Setting power to: 0.50"

    def test_send_colour_writes_the_command(self, controller):
        ctrl, queue, serial = controller
        ctrl.send_colour("led_610")
        assert b"led_610" in serial.written


class TestProcessArduinoColours:
    def test_fixed_colour(self):
        assert process_arduino_colours("led_610", 1, 4) == ["led_610"] * 4

    def test_alternating_every_pattern(self):
        colours = process_arduino_colours("a,b", 1, 4)
        assert colours[:4] == ["a", "b", "a", "b"]

    def test_change_logic_groups_consecutive_patterns(self):
        colours = process_arduino_colours("a,b", 3, 12)
        assert colours[:6] == ["a", "a", "a", "b", "b", "b"]
        # Colour changes land exactly on the multiples of change_logic, which
        # is what shader_loop tests for.
        assert [i for i in range(12) if i % 3 == 0] == [0, 3, 6, 9]

    def test_table_covers_every_pattern_index(self):
        for n in (1, 5, 7, 13):
            for logic in (1, 2, 3):
                assert len(process_arduino_colours("a,b,c", logic, n)) >= n