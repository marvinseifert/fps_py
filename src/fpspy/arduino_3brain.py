import serial
import threading


def connect_to_arduino(port="COM3", baud_rate=9600):
    """Establish a connection to the Arduino."""
    try:
        arduino = serial.Serial(port, baud_rate)
        return arduino
    except Exception as e:
        print(f"Error connecting to Arduino: {e}")
        return None


class Arduino3Brain:
    """
    Communicate with an Arduino device.

    This class wraps serial communication in thread-safe methods.

    The class is _not_ process-safe: don't try to open the same Arduino from multiple
    processes.
    """

    def __init__(
        self,
        port="COM3",
        baud_rate=230400,
    ):
        self.port = port
        self.baud_rate = baud_rate
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
            self._serial.write(b"1")
            self._serial.write(b"0")

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
