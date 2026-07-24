"""Connect and communicate with Arduino devices over serial."""

import serial
import threading
import time

import fpspy.fps_queue


def connect_to_arduino(port="COM3", baud_rate=9600):
    """Establish a connection to the Arduino."""
    try:
        arduino = serial.Serial(port, baud_rate)
        return arduino
    except Exception as e:
        print(f"Error connecting to Arduino: {e}")
        return None


class Arduino:
    """
    Communicate with an Arduino device.

    This class wraps serial communication in thread-safe methods.

    The class is _not_ process-safe: don't try to open the same Arduino from multiple
    processes.
    """

    def __init__(
        self,
        port="COM3",
        baud_rate=9600,
        trigger_command="T",
    ):
        self.port = port
        self.baud_rate = baud_rate
        self.trigger_command = trigger_command
        # Holds the tail of a line that arrived split across two reads, so
        # read_lines() can complete it on the next call. Only read_lines()
        # uses it; read() throws the buffer away by design.
        self._rx_buffer = ""
        self._serial = None
        self.connected = False
        self.connect()
        self._lock = threading.RLock()

    def connect(self):
        self._serial = connect_to_arduino(self.port, self.baud_rate)
        if self._serial is not None:
            self.connected = True
            print("Arduino connected")
        else:
            self.connected = False
            print("Arduino not connected")

    def send(self, message):
        """Send a message to the Arduino."""
        with self._lock:
            if self.connected:
                # Convert the colour string to bytes
                txt = f"\n{message}\n".encode("utf-8")
                self._serial.write(txt)
                # self._serial.flush()
            else:
                self.connect()
                if self.connected:
                    txt = f"\n{message}\n".encode("utf-8")
                    self._serial.write(txt)
                    # self._serial.flush()
                else:
                    print("Could not connect to Arduino or send message")

    def send_trigger(self):
        """Send a trigger signal to the Arduino."""
        with self._lock:
            self.send(self.trigger_command)

    def read_lines(self):
        """Read every complete line currently waiting, oldest first.

        Unlike read(), nothing is dropped: this is the telemetry read, where
        each line is an event in its own right ("Trigger", "finished", ...)
        and losing one loses the event. A trailing partial line stays in
        _rx_buffer and is completed by the next call.

        Returns [] when nothing is waiting, so it is cheap to poll.
        """
        with self._lock:
            if not self.connected:
                return []
            available = getattr(self._serial, "in_waiting", 0)
            if not available:
                return []
            data = self._serial.read(available)
            self._rx_buffer += data.decode("utf-8", errors="ignore")
            *complete, self._rx_buffer = self._rx_buffer.split("\n")
            return [line.strip() for line in complete if line.strip()]

    def read(self):
        """Read a line from the Arduino."""
        with self._lock:
            if not self.connected:
                return None
            last_line = None
            try:
                available = getattr(self._serial, "in_waiting", 0)
                if available and available > 0:
                    data = self._serial.read(available)
                    decoded = data.decode("utf-8", errors="ignore")
                    lines = [ln.strip() for ln in decoded.splitlines() if ln.strip()]
                    if lines:
                        last_line = lines[-1]
            finally:
                try:
                    self._serial.reset_input_buffer()
                except Exception:
                    pass
            return last_line

    def reset_input(self):
        with self._lock:
            self._serial.reset_input_buffer()
            # The partial line is part of the input too; keeping it would
            # splice pre-reset bytes onto the first line read after it.
            self._rx_buffer = ""

    def flush(self):
        """Flush the serial output buffer."""
        with self._lock:
            self._serial.flush()

    def disconnect(self):
        """Disconnect from the Arduino."""
        with self._lock:
            # None when the port never opened. Reached via __del__ at garbage
            # collection, where an AttributeError surfaces only as an
            # unraisable-exception warning, so guard rather than let it throw.
            if self._serial is not None:
                self._serial.close()
            self.connected = False
            print("Arduino disconnected")

    def __del__(self):
        self.disconnect()


class DummyArduino:
    def __init__(self, port="COM3", baud_rate=9600, trigger_command="T"):
        self.port = port
        self.baud_rate = baud_rate
        self.trigger_command = trigger_command
        self.arduino = None
        self.connected = True  # pretend it's always connected

    def connect(self):
        # no real connection, just mark as connected
        self.connected = True

    def send(self, message):
        # mimic the same interface, but only log
        txt = f"\n{message}\n".encode("utf-8")

    def send_trigger(self):
        """Send a trigger signal to the Arduino."""
        self.send(self.trigger_command)

    def read(self):
        # return None or some test data
        return None

    def read_lines(self):
        # A dummy device says nothing, so the telemetry stream stays empty.
        return []

    def flush(self):
        pass

    def reset_input(self):
        pass

    def disconnect(self):
        self.connected = False


def create_arduino(port, baud_rate=9600, trigger_command="T"):
    """Create an Arduino, or a DummyArduino if port is "dummy"."""
    if port == "dummy":
        return DummyArduino(
            port=port, baud_rate=baud_rate, trigger_command=trigger_command
        )
    return Arduino(port=port, baud_rate=baud_rate, trigger_command=trigger_command)


class ArduinoController:
    """Presentation-level Arduino semantics, decoupled from the Presenter.

    Owns the serial connection (which must live in exactly one process; see
    Arduino's docstring) inside the lead presenter process. present_live()
    registers this controller's methods as Presenter callbacks, so the
    Presenter itself has no Arduino knowledge. The command codes ("t_s_on",
    "t_s_off", "b", "O") mirror the original play.py implementation.

    Threading: on_play_start/on_trigger/on_stop run synchronously on the
    presenter's render loop thread (serial writes are kernel-buffered and
    cheap). A reader thread runs for the controller's whole life and turns
    every line the device prints into an event on the status queue; Arduino's
    methods are RLock-protected, so both threads can share the connection.
    """

    # Lines the firmware prints, mapped to the event kind they are reported
    # as. Anything not listed here is still reported, as "arduino_line" — the
    # firmware prints values as well as fixed words (LED channel, power,
    # micros(), protocol names), and those are worth showing verbatim.
    #
    # "Unknown command" is not an error report. The firmware prints it only
    # for "b", the interrupt byte, and only when "b" reached the command
    # parser instead of being consumed mid-protocol by loop_interrupt() —
    # i.e. when nothing was running to interrupt. on_stop sends "b" on every
    # stop, so an idle stop prints it every time. A mistyped command, by
    # contrast, produces no output at all.
    LINE_EVENTS = {
        "Stimulator ready": "arduino_ready",
        "finished": "arduino_done",
        "Trigger": "arduino_trigger",
        "Trigger_test": "arduino_trigger_test",
        "Unknown command": "arduino_unknown_command",
    }

    # The device is idle-polled at 1 Hz by its own loop(), so there is nothing
    # to gain from reading faster than this, and the lock is shared with the
    # render thread's send_trigger().
    READ_INTERVAL = 0.02

    def __init__(self, port, baud_rate, trigger_command, status_queue, sender_idx=1):
        self.arduino = create_arduino(port, baud_rate, trigger_command)
        # A queue of its own: these are asynchronous device events, not answers
        # to presenter commands, so they must not share the presenter's reply
        # queue.
        self.status_queue = status_queue
        self.sender_idx = sender_idx
        # Whether the port opened is the only thing about the device that is
        # known without waiting for it to print something, and a panel that
        # says nothing until the first line arrives is indistinguishable from
        # one talking to a dead port. So report it immediately.
        self._report_connection(port)
        self._reader_stop = threading.Event()
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _report_connection(self, port):
        """Announce whether there is a real device on the other end.

        A DummyArduino is called out separately rather than reported as
        connected: it accepts every command and answers nothing, so treating
        it as a working device would make a misconfigured port look like
        silent hardware.
        """
        if isinstance(self.arduino, DummyArduino):
            kind, text = "arduino_dummy", f"No device: port is {port!r} (dummy)"
        elif self.arduino.connected:
            kind, text = "arduino_connected", f"Connected on {port}"
        else:
            kind, text = "arduino_disconnected", f"Could not open {port}"
        fpspy.fps_queue.put_reply(
            self.status_queue, kind, self.sender_idx, text=text
        )

    def on_play_start(self):
        """Enable per-frame trigger mode. Fired just before the render loop."""
        self.arduino.send("t_s_on")
        self.arduino.flush()

    def on_trigger(self):
        """Send one trigger pulse. Fired after each triggered frame's swap."""
        self.arduino.send_trigger()

    def send_colour(self, colour: str):
        """Switch the LEDs. Fired from the render loop when the colour changes.

        `colour` is a firmware command such as "led_610" or "white"; it is sent
        as-is, so an unknown one is silently ignored by the device.
        """
        self.arduino.send(colour)

    def on_stop(self):
        """Disable trigger mode and reset LEDs. Fired on stop and play end."""
        self.arduino.send("t_s_off")
        self.arduino.send("b")
        self.arduino.send("O")

    def handle_command(self, command: str, monitor: bool = False):
        """Forward a command from the GUI/main process to the Arduino.

        `monitor` is accepted for callers that still pass it, but ignored: the
        reader thread reports "finished" (as "arduino_done") whenever it
        arrives, so there is no longer an on-demand monitor to start.
        """
        self.arduino.send(command)

    def _read_loop(self):
        """Report every line the device prints, as it arrives.

        Reading continuously rather than on demand means events are not missed
        between commands, and the panel can show device state at any moment
        instead of only while a command is outstanding.
        """
        while not self._reader_stop.is_set():
            try:
                lines = self.arduino.read_lines()
            except Exception:
                # A disconnected or closing port must not kill the thread and
                # take the rest of the telemetry with it.
                lines = []
            for line in lines:
                fpspy.fps_queue.put_reply(
                    self.status_queue,
                    self.LINE_EVENTS.get(line, "arduino_line"),
                    self.sender_idx,
                    text=line,
                )
            self._reader_stop.wait(self.READ_INTERVAL)

    def close(self):
        self._reader_stop.set()
        self._reader_thread.join(timeout=2)
        self.arduino.disconnect()


class Arduino2:
    """
    Communicate with an Arduino device, including encoded text (v2).

    This class wraps serial communication in thread-safe methods.

    The class is _not_ process-safe: don't try to open the same Arduino from multiple
    processes.

    Sending text
    ------------
    The firmware's message command: "M<text>\n" puts <text> on the trigger wire as
    a framed, CRC-checked message, then replies MESSAGE_ACK. See the wire format in
    the firmware repo (code/arduino/), which also holds the offline decoder.
    """

    TEXT_CMD = "M"
    TEXT_ACK = "MOK"
    TEXT_ERR = "MERR"
    # The firmware's payload limit, in bytes of UTF-8 (not characters).
    MAX_TEXT_BYTES = 255

    def __init__(
        self,
        port="COM3",
        baud_rate=9600,
        trigger_command="T",
    ):
        self.port = port
        self.baud_rate = baud_rate
        self.trigger_command = trigger_command
        self._serial = None
        self.connected = False
        self.connect()
        self._lock = threading.RLock()

    def connect(self):
        self._serial = connect_to_arduino(self.port, self.baud_rate)
        if self._serial is not None:
            self.connected = True
            print("Arduino connected")
        else:
            self.connected = False
            print("Arduino not connected")

    def send(self, message):
        """Send a message to the Arduino."""
        with self._lock:
            if self.connected:
                # Convert the colour string to bytes
                txt = f"\n{message}\n".encode("utf-8")
                self._serial.write(txt)
                # self._serial.flush()
            else:
                self.connect()
                if self.connected:
                    txt = f"\n{message}\n".encode("utf-8")
                    self._serial.write(txt)
                    # self._serial.flush()
                else:
                    print("Could not connect to Arduino or send message")

    def send_trigger(self):
        """Send a trigger signal to the Arduino."""
        with self._lock:
            self.send(self.trigger_command)

    def send_text(self, text, timeout=10.0):
        """Put `text` on the trigger wire as a message, and wait for the ack.

        Blocks until the Arduino reports it has finished transmitting. That wait
        is not optional: the Arduino cannot service triggers while transmitting,
        so returning early would let triggers queue up in its serial buffer and
        fire late, in a burst. Only send messages while the wire is idle.

        Returns True on success, False if not connected or the ack times out.
        """
        payload = text.encode("utf-8")
        if "\n" in text:
            raise ValueError("A message cannot contain a newline; it ends the message.")
        if len(payload) > self.MAX_TEXT_BYTES:
            raise ValueError(
                f"Text is {len(payload)} bytes of UTF-8, max is {self.MAX_TEXT_BYTES}."
            )
        with self._lock:
            if not self.connected:
                self.connect()
                if not self.connected:
                    print("Could not connect to Arduino or send message")
                    return False
            self.reset_input()
            self.send(f"{self.TEXT_CMD}{text}")
            self.flush()
            # timeout is only a safety net; the ack arrives when the wire is idle.
            deadline = time.monotonic() + timeout
            prev_timeout = self._serial.timeout
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        print(f"Timed out waiting for Arduino to send: {text!r}")
                        return False
                    self._serial.timeout = remaining
                    line = (
                        self._serial.readline().decode("utf-8", errors="ignore").strip()
                    )
                    if line == self.TEXT_ACK:
                        return True
                    if line == self.TEXT_ERR:
                        raise RuntimeError(
                            f"Arduino rejected message (payload overflow): {text!r}"
                        )
            finally:
                self._serial.timeout = prev_timeout

    def read(self):
        """Read a line from the Arduino."""
        with self._lock:
            if not self.connected:
                return None
            last_line = None
            try:
                available = getattr(self._serial, "in_waiting", 0)
                if available and available > 0:
                    data = self._serial.read(available)
                    decoded = data.decode("utf-8", errors="ignore")
                    lines = [ln.strip() for ln in decoded.splitlines() if ln.strip()]
                    if lines:
                        last_line = lines[-1]
            finally:
                try:
                    self._serial.reset_input_buffer()
                except Exception:
                    pass
            return last_line

    def reset_input(self):
        with self._lock:
            self._serial.reset_input_buffer()

    def flush(self):
        """Flush the serial output buffer."""
        with self._lock:
            self._serial.flush()

    def disconnect(self):
        """Disconnect from the Arduino."""
        with self._lock:
            if self._serial is not None:
                self._serial.close()
            self.connected = False
            print("Arduino disconnected")

    def __del__(self):
        self.disconnect()