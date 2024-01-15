import serial
def connect_to_arduino(port='COM2', baud_rate=9600):
    """Establish a connection to the Arduino."""
    try:
        arduino = serial.Serial(port, baud_rate)
        return arduino
    except Exception as e:
        print(f"Error connecting to Arduino: {e}")
        return None