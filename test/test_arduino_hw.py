"""Hardware-in-the-loop checks against a connected ESP32 stimulator.

Skipped unless FPSPY_ARDUINO_PORT names a real port, so a normal test run is
unaffected:

    FPSPY_ARDUINO_PORT=/dev/ttyUSB0 python -m pytest test/test_arduino_hw.py -v

What these can and cannot check
-------------------------------
The firmware has no status query, so device state is never readable directly —
only the lines it chooses to print. That limits assertions to commands with
an observable echo:

    trigger_test      -> "Trigger_test" then "Trigger"
    ledpower <n> <p>  -> "Changing LED: n" then "Setting power to: p"

LED commands (led_610, white, O) print nothing, so their effect can only be
seen with an eye or a photodiode. The tests below send them and check the send
path does not raise or stall; the light itself is left to the operator.

Latency: while trigger mode is off, the firmware's loop() ends in delay(1000),
so a command can wait up to a second to be handled. Timeouts here are sized
for that, which is also why these tests are slow.
"""

import multiprocessing as mp
import os
import queue as std_queue
import time

import pytest

import fpspy.fps_queue
from fpspy import arduino
from fpspy.presentation import process_arduino_colours

PORT = "/dev/ttyUSB0"#os.environ.get("FPSPY_ARDUINO_PORT")
BAUD = int(os.environ.get("FPSPY_ARDUINO_BAUD", "9600"))

pytestmark = pytest.mark.skipif(
    not PORT, reason="Set FPSPY_ARDUINO_PORT to run the hardware tests."
)

# Generous: the device polls for commands once a second when idle.
REPLY_TIMEOUT = 4.0


@pytest.fixture
def controller():
    """A live controller, with the telemetry reader already running."""
    queue = mp.Queue()
    ctrl = arduino.ArduinoController(PORT, BAUD, "T", queue, sender_idx=1)
    assert ctrl.arduino.connected, f"Could not open {PORT}"
    # Opening the port resets the board on most ESP32 boards; give it time to
    # boot and say so, then start each test from a quiet queue.
    time.sleep(2.0)
    drain(queue, timeout=0.2)
    yield ctrl, queue
    ctrl.on_stop()
    ctrl.close()


def drain(queue, timeout=0.2):
    """Collect every reply currently waiting."""
    replies = []
    while True:
        try:
            replies.append(fpspy.fps_queue.get_reply(queue, timeout=timeout))
        except std_queue.Empty:
            return replies


def wait_for(queue, kind, timeout=REPLY_TIMEOUT):
    """Wait for one reply of `kind`, returning it and the ones skipped past."""
    deadline = time.monotonic() + timeout
    seen = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            pytest.fail(f"No {kind!r} within {timeout}s. Saw: {[r.kind for r in seen]}")
        try:
            reply = fpspy.fps_queue.get_reply(queue, timeout=remaining)
        except std_queue.Empty:
            continue
        if reply.kind == kind:
            return reply, seen
        seen.append(reply)


def test_board_announces_itself_on_connect():
    """A fresh connection resets the board, which then prints its banner.

    This is the connection-status check the instrument panel needs: the banner
    is the only positive proof the thing on the other end is the stimulator and
    not just an open tty.
    """
    queue = mp.Queue()
    ctrl = arduino.ArduinoController(PORT, BAUD, "T", queue, sender_idx=1)
    try:
        reply, _ = wait_for(queue, "arduino_ready", timeout=6.0)
        assert reply.payload["text"] == "Stimulator ready"
    finally:
        ctrl.close()


def test_trigger_test_round_trip(controller):
    """The one command with a deterministic, LED-free echo.

    Proves the whole path end to end: send on the render-loop thread, device
    executes, reader thread picks the lines up, both arrive as typed events on
    the queue in order.
    """
    ctrl, queue = controller
    ctrl.handle_command("trigger_test")
    _, before = wait_for(queue, "arduino_trigger_test")
    assert not [r for r in before if r.kind == "arduino_trigger"]
    wait_for(queue, "arduino_trigger")


def test_fast_triggers_produce_no_telemetry(controller):
    """The per-frame trigger must stay silent.

    "T" maps to trigger_signal_fast(), which deliberately does not print — the
    print costs 1-2 ms and would wreck noise stimuli. If a firmware change ever
    made "T" print, the reader thread would flood the queue at frame rate and
    contend with the render loop for the serial lock. This is that alarm.
    """
    ctrl, queue = controller
    ctrl.on_play_start()  # t_s_on
    time.sleep(1.5)  # let trigger mode actually engage
    drain(queue)
    for _ in range(60):
        ctrl.on_trigger()
        time.sleep(1 / 60)
    time.sleep(0.5)
    assert drain(queue) == []


def test_trigger_send_is_cheap_enough_for_the_render_loop(controller):
    """A trigger send must cost far less than a frame.

    on_trigger() runs synchronously on the render loop right after the buffer
    swap, so its cost eats into the frame budget (8.3 ms at 120 Hz). The write
    is kernel-buffered, so this measures the lock plus the syscall.
    """
    ctrl, queue = controller
    ctrl.on_play_start()
    time.sleep(1.5)
    worst = 0.0
    for _ in range(200):
        t0 = time.perf_counter()
        ctrl.on_trigger()
        worst = max(worst, time.perf_counter() - t0)
        time.sleep(1 / 120)
    assert worst < 1e-3, f"Worst trigger send took {worst * 1e3:.2f} ms"


def test_reader_thread_does_not_starve_the_sender(controller):
    """Telemetry reading and trigger sending share one lock; check the cost.

    Same measurement as above but while the device is chattering, so the
    reader thread is actually holding the lock between sends.
    """
    ctrl, queue = controller
    ctrl.handle_command("trigger_test")
    worst = 0.0
    for _ in range(200):
        t0 = time.perf_counter()
        ctrl.on_trigger()
        worst = max(worst, time.perf_counter() - t0)
        time.sleep(1 / 120)
    assert worst < 2e-3, f"Worst trigger send under load took {worst * 1e3:.2f} ms"


def test_command_with_arguments_is_echoed(controller):
    """ledpower echoes its parsed arguments, so it checks argument framing.

    Nothing else the presenter sends takes arguments, but this is the only way
    to confirm the device received what was meant, byte for byte.
    """
    ctrl, queue = controller
    ctrl.handle_command("ledpower 0 50")
    reply, _ = wait_for(queue, "arduino_line")
    lines = [reply.payload["text"]]
    time.sleep(1.5)
    lines += [r.payload["text"] for r in drain(queue)]
    assert "Changing LED: 0" in lines
    assert any(line.startswith("Setting power to: 0.50") for line in lines)


@pytest.mark.parametrize("colour", ["led_610", "led_560", "white", "O"])
def test_colour_commands_are_accepted(controller, colour):
    """Send each colour and confirm the device does not complain.

    The firmware prints nothing for these, so the assertion is negative: no
    "Unknown command", no exception, no stall. Whether the right LED lit is
    for the operator to see.
    """
    ctrl, queue = controller
    ctrl.send_colour(colour)
    time.sleep(1.5)
    assert "arduino_unknown_command" not in [r.kind for r in drain(queue)]


def test_colour_schedule_walks_the_leds(controller):
    """Replay what shader_loop does, at a pace an operator can watch.

    Drives the real colour table through the real send path with change_logic
    > 1, so the LEDs should visibly step 610 -> 560 -> 610 -> ... , changing
    every third pattern. Automated only as far as "nothing errored"; run it
    with the rig in view.
    """
    ctrl, queue = controller
    n_patterns = 12
    change_logic = 3
    colours = process_arduino_colours("led_610,led_560", change_logic, n_patterns)
    sent = []
    for pattern_index in range(n_patterns):
        if pattern_index % change_logic == 0 and pattern_index < len(colours):
            ctrl.send_colour(colours[pattern_index])
            sent.append(colours[pattern_index])
            time.sleep(0.5)
    ctrl.send_colour("O")
    assert sent == ["led_610", "led_560", "led_610", "led_560"]
    assert "arduino_unknown_command" not in [r.kind for r in drain(queue)]