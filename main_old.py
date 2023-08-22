import tkinter as tk
from tkinter import ttk
import create_noise
from pathlib import Path
from play_noise import Presenter

def on_generate_noise():
    generate_noise_button.config(text="Generating...")
    root.update()
    # Extract values and convert to appropriate data types
    checkerboard_size = int(checkerboard_var.get())
    window_size = tuple(map(int, window_size_var.get().split(",")))
    noise_frequency = int(noise_frequency_var.get())
    noise_duration = float(noise_duration_var.get())
    file_name = str(noise_name_var.get())

    # Validate the noise name's suffix
    file_name_path = Path(file_name)
    if file_name_path.suffix and file_name_path.suffix != ".h5":
        print(f"Invalid suffix '{file_name_path.suffix}' provided. Using '.h5' instead.")
        file_name = file_name_path.stem  # Get the filename without any suffix
    file_name += ".h5"

    complete_file_name = Path("stimuli", file_name)

    # Print the values (for now)
    print("Checkerboard Size:", checkerboard_size)
    print("Window Size:", window_size)
    print("Noise Frequency:", noise_frequency)
    print("Noise Duration:", noise_duration)
    print("File Name:", file_name)

    # Calculate frames needed
    frames = int(noise_duration*60*noise_frequency)
    width = window_size[0]
    height = window_size[1]

    # Call the function to generate the noise
    create_noise.generate_and_store_3d_array(frames, checkerboard_size, width, height, name=complete_file_name)
    refresh_file_list()
    generate_noise_button.config(text="Generate Noise")

def on_play_noise():
    # Check selected file in the file_listbox
    selected_indices = file_listbox.curselection()  # Returns a tuple of selected indices
    if selected_indices:
        selected_file = file_listbox.get(selected_indices[0])
        print(f"Playing noise from file: {selected_file}")
        # Add your logic to play the noise from the selected file
    else:
        print("No file selected!")

def refresh_file_list():
    """Refresh the list of .h5py files in the stimuli directory."""
    stimuli_dir = Path("stimuli")

    if stimuli_dir.exists() and stimuli_dir.is_dir():
        files = [f.name for f in stimuli_dir.iterdir() if f.suffix == '.h5']
        file_listbox.delete(0, tk.END)  # Clear the listbox
        for file in files:
            file_listbox.insert(tk.END, file)
    else:
        print(f"'{stimuli_dir}' directory does not exist.")


def compute_size(*args):  # *args to make it compatible with trace callback
    noise_frequency = int(noise_frequency_var.get())
    noise_duration = float(noise_duration_var.get())
    frames = int(noise_duration*60*noise_frequency)
    width, height = map(int, window_size_var.get().split(","))

    size_in_bytes = frames * width * height  # uint8: 1 byte per element

    if size_in_bytes < 1024:
        size_str = f"{size_in_bytes} bytes"
    elif size_in_bytes < 1024 ** 2:
        size_in_kb = size_in_bytes / 1024
        size_str = f"{size_in_kb:.2f} KB"
    elif size_in_bytes < 1024 ** 3:
        size_in_mb = size_in_bytes / (1024 ** 2)
        size_str = f"{size_in_mb:.2f} MB"
    else:
        size_in_gb = size_in_bytes / (1024 ** 3)
        size_str = f"{size_in_gb:.2f} GB"

    size_label.config(text=f"Estimated size: {size_str}")


# Create the main window
root = tk.Tk()
root.title("Noise Generator GUI")

# Create frames for left and right sections
left_frame = ttk.Frame(root, padding="10")
left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))


right_frame = ttk.Frame(root, padding="10")
right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

# Standard values and data types for the input fields
standard_values = ["5", "1000,1000", "20", "5.0"]

# Variables to store the values of the input fields
checkerboard_var = tk.StringVar(value=standard_values[0])
window_size_var = tk.StringVar(value=standard_values[1])
noise_frequency_var = tk.StringVar(value=standard_values[2])
noise_duration_var = tk.StringVar(value=standard_values[3])



noise_name_var = tk.StringVar()
noise_name_entry = ttk.Entry(right_frame, textvariable=noise_name_var)
noise_name_entry.grid(row=0, column=1, padx=10, pady=5)


variables = [checkerboard_var, window_size_var, noise_frequency_var, noise_duration_var]

labels = ["checkerboard size", "window size", "noise frequency", "noise duration"]
for i, (label, var) in enumerate(zip(labels, variables)):
    ttk.Label(left_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=5)
    ttk.Entry(left_frame, textvariable=var).grid(row=i, column=1, pady=5)
root.grid_rowconfigure(0, weight=1)
left_frame.grid_rowconfigure(len(labels), weight=1)

#Listbox
file_listbox = tk.Listbox(left_frame, height=5, width=30)
file_listbox.grid(row=len(labels), column=0, columnspan=2, pady=5, sticky=tk.N+tk.S+tk.E+tk.W)
refresh_button = ttk.Button(left_frame, text="Refresh", command=refresh_file_list)
refresh_button.grid(row=len(labels), column=2, pady=5, sticky=tk.N+tk.E)
refresh_file_list()


generate_noise_button = ttk.Button(right_frame, text="generate noise", command=on_generate_noise)
generate_noise_button.grid(row=0, column=0, pady=5)

ttk.Button(right_frame, text="play noise", command=on_play_noise).grid(row=1, column=0, pady=5)

checkerboard_var.trace_add("write", compute_size)
window_size_var.trace_add("write", compute_size)
noise_frequency_var.trace_add("write", compute_size)
noise_duration_var.trace_add("write", compute_size)


size_label = ttk.Label(right_frame, text="")
size_label.grid(row=0, column=2, padx=10, pady=5)


# Call compute_size initially to set the label text
compute_size()


# # Create buttons on the right
# buttons = [("generate noise", on_generate_noise), ("play noise", on_play_noise)]
# for i, (text, handler) in enumerate(buttons):
#     ttk.Button(right_frame, text=text, command=handler).grid(row=i, column=0, pady=5)

root.mainloop()
