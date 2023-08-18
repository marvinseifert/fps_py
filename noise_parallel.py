import multiprocessing
import h5py
import numpy as np
import pygame


# Load data function
def load_data(shared_buffers, events, hdf5_path, chunk_size, buffer_index, nr_boxes):
    with h5py.File(hdf5_path, 'r') as f:
        dataset = f['/Red_Noise']
        while True:
            # Wait for signal to load data into buffer
            events[buffer_index][1].wait()

            start_idx = (buffer_index * chunk_size) % len(dataset)
            end_idx = start_idx + chunk_size
            shared_buffers[buffer_index][:chunk_size] = dataset[start_idx:end_idx, :nr_boxes]



            # Signal that data is loaded into buffer
            events[buffer_index][0].set()
            events[buffer_index][1].clear()
            buffer_index = 1 - buffer_index  # Toggle between 0 and 1


# Display function
def display_noise(shared_buffers, events, frames_between_loads, width, height, checker_size):
    # Initialize pygame and set up display
    pygame.init()
    window = pygame.display.set_mode((width, height), pygame.FULLSCREEN)

    def draw_checkerboard(pattern):
        for i, x in enumerate(range(0, width, checker_size)):
            for j, y in enumerate(range(0, height, checker_size)):
                idx = i * (height // checker_size) + j
                color = (255, 255, 255) if pattern[idx] else (0, 0, 0)
                pygame.draw.rect(window, color, (x, y, checker_size, checker_size))

    buffer_index = 0
    frame_count = 0
    while True:
        # Process events like keyboard input (e.g., to exit the program)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                # Signal the loaders to terminate (optional but clean)
                events[0][1].set()
                events[1][1].set()
                return

        # Draw the checkerboard from the current buffer
        draw_checkerboard(shared_buffers[buffer_index])

        # Update the display
        pygame.display.flip()

        frame_count += 1
        if frame_count == frames_between_loads:
            # Signal that we're ready for the next chunk
            events[buffer_index][1].set()

            # Switch to the other buffer
            buffer_index = 1 - buffer_index

            # Wait for the other buffer to be ready
            events[buffer_index][0].wait()
            events[buffer_index][0].clear()

            frame_count = 0


# Main function
if __name__ == "__main__":
    chunk_size = 100
    buffers = [multiprocessing.Array('f', chunk_size) for _ in range(2)]
    events = [(multiprocessing.Event(), multiprocessing.Event()) for _ in range(2)]

    # Start the initial load on both loaders
    events[0][1].set()
    events[1][1].set()

    # Create processes
    loader1 = multiprocessing.Process(target=load_data, args=(buffers, events, r'C:\Users\Marvin\Chicken_analysis_script\Noise.h5', chunk_size, 0, 1600))
    loader2 = multiprocessing.Process(target=load_data, args=(buffers, events, r'C:\Users\Marvin\Chicken_analysis_script\Noise.h5', chunk_size, 1, 1600))
    displayer = multiprocessing.Process(target=display_noise, args=(buffers, events, 50))

    # Start processes
    loader1.start()
    loader2.start()
    displayer.start()

    # Join processes
    loader1.join()
    loader2.join()
    displayer.join()
