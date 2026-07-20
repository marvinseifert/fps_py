"""Connect and communicate with Arduino devices over serial."""

import serial
import threading
import time


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
    cheap). Status monitoring after a flash command runs on a background
    thread (mirroring play.py's receive_arduino_status); Arduino's methods
    are RLock-protected, so both threads can share the connection.
    """

    def __init__(self, port, baud_rate, trigger_command, status_queue):
        self.arduino = create_arduino(port, baud_rate, trigger_command)
        self.status_queue = status_queue
        self._monitor_thread = None
        self._monitor_stop = threading.Event()

    def on_play_start(self):
        """Enable per-frame trigger mode. Fired just before the render loop."""
        self.arduino.send("t_s_on")
        self.arduino.flush()

    def on_trigger(self):
        """Send one trigger pulse. Fired after each triggered frame's swap."""
        self.arduino.send_trigger()

    def on_stop(self):
        """Disable trigger mode and reset LEDs. Fired on stop and play end."""
        self._stop_monitor()
        self.arduino.send("t_s_off")
        self.arduino.send("b")
        self.arduino.send("O")

    def handle_command(self, command: str, monitor: bool = False):
        """Forward a command from the GUI/main process to the Arduino.

        With monitor=True, a background thread reads the Arduino until it
        reports "finished" and then puts "done" on the status queue. Used for
        stimuli that run on the Arduino itself (e.g. LED flashes) while the
        presenter shows a static screen.
        """
        self.arduino.send(command)
        if monitor:
            self._start_monitor()

    def _start_monitor(self):
        self._stop_monitor()
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_status, daemon=True
        )
        self._monitor_thread.start()

    def _stop_monitor(self):
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._monitor_stop.set()
            self._monitor_thread.join(timeout=2)
        self._monitor_thread = None

    def _monitor_status(self):
        """Wait for the Arduino to report that its stimulus has finished.

        A "Trigger" line marks the actual start; "finished" lines seen before
        it are stale output from a previous run (same buffering logic as
        play.py's receive_arduino_status).
        """
        buffer = True
        self.arduino.reset_input()
        while not self._monitor_stop.is_set():
            status = self.arduino.read()
            if status == "Trigger":
                buffer = False
            if status == "finished" and not buffer:
                self.status_queue.put("done")
                break
            time.sleep(0.001)

    def close(self):
        self._stop_monitor()
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