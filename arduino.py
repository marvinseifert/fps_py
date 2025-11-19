import serial


def connect_to_arduino(port="COM3", baud_rate=9600):
    """Establish a connection to the Arduino."""
    try:
        arduino = serial.Serial(port, baud_rate)
        return arduino
    except Exception as e:
        print(f"Error connecting to Arduino: {e}")
        return None


class Arduino:
    def __init__(self, port="COM3", baud_rate=9600, queue=None, queue_lock=None):
        self.port = port
        self.baud_rate = baud_rate
        self.arduino = None
        self.queue = queue
        self.queue_lock = queue_lock
        self.connected = False
        self.connect()

    def connect(self):
        self.arduino = connect_to_arduino(self.port, self.baud_rate)
        if self.arduino is not None:
            self.connected = True
            print("Arduino connected")
        else:
            self.connected = False
            print("Arduino not connected")

    def send(self, message):
        if self.connected:
            txt = f"\n{message}\n".encode("utf-8")  # Convert the colour string to bytes
            self.arduino.write(txt)
            self.arduino.flush()
        else:
            self.connect()
            if self.connected:
                txt = f"\n{message}\n".encode("utf-8")
                self.arduino.write(txt)
                self.arduino.flush()
            else:
                print("Could not connect to Arduino or send message")

    def read(self):
        if not self.connected:
            return None
        last_line = None
        try:
            available = getattr(self.arduino, "in_waiting", 0)
            if available and available > 0:
                data = self.arduino.read(available)
                decoded = data.decode("utf-8", errors="ignore")
                lines = [ln.strip() for ln in decoded.splitlines() if ln.strip()]
                if lines:
                    last_line = lines[-1]
        finally:
            try:
                self.arduino.reset_input_buffer()
            except Exception:
                pass
        return last_line

    def disconnect(self):
        self.arduino.close()
        self.connected = False
        print("Arduino disconnected")


class DummyArduino:
    def __init__(self, port="COM3", baud_rate=9600, queue=None, queue_lock=None):
        self.port = port
        self.baud_rate = baud_rate
        self.arduino = None
        self.queue = queue
        self.queue_lock = queue_lock
        self.connected = True  # pretend it's always connected

    def connect(self):
        # no real connection, just mark as connected
        self.connected = True


    def send(self, message):
        # mimic the same interface, but only log
        txt = f"\n{message}\n".encode("utf-8")


    def read(self):
        # return None or some test data

        return None

    def disconnect(self):
        self.connected = False
