import os
import pygame
import serial
import random
import numpy as np

# Set starting coordinates for the secondary screen
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (2560, 0)  # Start from the end of the primary screen

# Initialize pygame
pygame.init()

# Create a window in full-screen mode
window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

# Set the title of the window
pygame.display.set_caption("Checkerboard Stimuli Display")

# Connect to the serial port (change 'COM3' to your port)
#ser = serial.Serial('COM3', 9600)

# Adjust width and height to match screen resolution
info_object = pygame.display.Info()
#width, height = info_object.current_w, info_object.current_h
width = 500
height = 500
checker_size = 1

def send_trigger():
    return
    #ser.write(b'T')

# ... [other code remains unchanged]

def generate_checkerboard_pattern(width, height, checker_size):
    pattern_shape = (width // checker_size, height // checker_size)
    pattern = np.random.randint(0, 2, pattern_shape, dtype=np.uint8)  # 0 for black, 1 for white
    return pattern


def draw_checkerboard(pattern, window, checker_size):


    # Create a pattern surface
    pattern_surface = pygame.Surface((pattern.shape[1] * checker_size, pattern.shape[0] * checker_size))

    # Iterate through the pattern and blit the corresponding square to the pattern surface
    for i in range(pattern.shape[0]):
        for j in range(pattern.shape[1]):
            square = white_square if pattern[i, j] else black_square
            pattern_surface.blit(square, (j * checker_size, i * checker_size))

    # Blit the pattern surface to the main window
    window.blit(pattern_surface, (0, 0))

# ... [rest of the code remains unchanged]
white_square = pygame.Surface((checker_size, checker_size))
white_square.fill((255, 255, 255))
black_square = pygame.Surface((checker_size, checker_size))
black_square.fill((0, 0, 0))

# Pre-generate 1000 checkerboard patterns
patterns = [generate_checkerboard_pattern(width, height, checker_size) for _ in range(36000)]

frame_count = 0
clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            pygame.quit()
            #ser.close()
            exit()

    # If it's time to update the checkerboard pattern (every 3 frames)
    if frame_count % 3 == 0:
        pattern_index = frame_count // 3
        if pattern_index < len(patterns):
            draw_checkerboard(patterns[pattern_index], window, checker_size)
            send_trigger()
            pygame.display.flip()

    frame_count += 1
    time_elapsed = clock.tick(60)
    if time_elapsed > 1000/60:
        print(f"WARNING: Frame rate dropped below 60 Hz!, was {1000/time_elapsed}")
