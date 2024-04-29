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
        else:
            self.connect()
            if self.connected:
                txt = f"\n{message}\n".encode("utf-8")
                self.arduino.write(txt)
            else:
                print("Could not connect to Arduino or send message")

    def disconnect(self):
        self.arduino.close()
        self.connected = False
        print("Arduino disconnected")
