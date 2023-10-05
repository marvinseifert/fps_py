import cv2
import numpy as np


def read_video_to_array(video_filename):
    # Open the video file
    cap = cv2.VideoCapture(video_filename)

    # Get video properties: total number of frames, width, and height
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize an array to hold the video data
    video_data = np.zeros((total_frames, height, width, 3), dtype=np.uint8)

    # Read each frame and store it in the array
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Couldn't read frame {} from video.".format(i))
        video_data[i] = frame

    # Release the video capture object
    cap.release()

    return video_data

# %%


import create_noise

create_noise.generate_and_store_video(frames=100, checkerboard_size=100, width_in_pixels=1920, height_in_pixels=1080,
                                      fps=30, name="test.mp4")

#%%

video_data = read_video_to_array("test_video.mp4")