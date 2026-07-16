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
            if self._serial is not None:
                self._serial.close()
            self.connected = False
            print("Arduino disconnected")

    def __del__(self):
        self.disconnect()


class DummyArduino:
    def __init__(self, port="COM3", baud_rate=9600):
        self.port = port
        self.baud_rate = baud_rate
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
        self.send("T")

    def read(self):
        # return None or some test data
        return None

    def disconnect(self):
        self.connected = False


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
