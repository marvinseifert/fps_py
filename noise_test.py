import pygame
import serial
import random

# Initialize pygame
pygame.init()

# Set the dimensions of the window
width, height = 800, 600
checker_size = 50

# Create a window
window = pygame.display.set_mode((width, height))

# Set the title of the window
pygame.display.set_caption("Checkerboard Stimuli Display")

# Connect to the serial port (change 'COM3' to your port)
#ser = serial.Serial('COM3', 9600)

def send_trigger():
    # Send a trigger signal via the serial port
    #ser.write(b'T')
    print("Trigger sent")

def draw_checkerboard():
    for x in range(0, width, checker_size):
        for y in range(0, height, checker_size):
            color = (255, 255, 255) if random.randint(0, 1) else (0, 0, 0)
            pygame.draw.rect(window, color, (x, y, checker_size, checker_size))

frame_count = 0
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            ser.close()
            exit()

    # If it's time to update the checkerboard pattern (every 3 frames)
    if frame_count % 3 == 0:
        draw_checkerboard()
        send_trigger()
        pygame.display.flip()

    frame_count += 1
