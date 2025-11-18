"""
This file contains code for the GUI application which runs in a separate process.
It allows the user to generate noise patterns, play them, and send commands to an
Arduino device. The GUI is built using tkinter and uses multiprocessing for
inter-process communication.

@ Marvin Seifert, 2024
"""

import contextlib
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from multiprocessing import Queue
import fpspy.config
import fpspy.help_dialog
import h5py
import time
import numpy as np
import fpspy.create_noise
import fpspy.shuffle_noise


class NoiseGeneratorApp:
    """Class for the Noise Generator GUI."""

    def __init__(
        self,
        root: tk.Tk,
        config: dict,
        queue1: Queue,
        lock: threading.Lock,
        ard_queue: Queue,
        ard_lock: threading.Lock,
        status_queue: Queue,
        status_lock: threading.Lock,
        nr_processes: int = 1,
    ):
        """
        Parameters
        ----------
        root : tkinter.Tk
            Root window of the GUI.
        queue : multiprocessing.Queue
            Queue for communication with the main process (gui).

        """

        self.config = config
        self.lock = lock
        self.queue1 = queue1
        self.ard_queue = ard_queue
        self.ard_lock = ard_lock
        self.status_queue = status_queue
        self.status_lock = status_lock
        self.root = root
        self.nr_processes = nr_processes
        self.root.title("Noise Generator GUI")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.geometry("600x500")  # application window size

        # Create frames for left and right sections
        self.left_frame = ttk.Frame(self.root, padding="10")
        self.left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.right_frame = ttk.Frame(self.root, padding="10")
        self.right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Variables to store the values of the input fields
        standard_values = ["5", "1000,1000", "20", "5.0"]
        # Variables to store the values of the input fields
        self.checkerboard_var = tk.StringVar(
            value=standard_values[0]
        )  # variable for checkerboard size
        self.window_size_var = tk.StringVar(
            value=standard_values[1]
        )  # variable for noise size
        self.noise_frequency_var = tk.StringVar(
            value=standard_values[2]
        )  # variable for noise frequency
        self.noise_duration_var = tk.StringVar(
            value=standard_values[3]
        )  # variable for noise duration
        self.noise_name_var = tk.StringVar()  # variable for noise name
        self.selected_file_info_var = tk.StringVar(
            value=""
        )  # variable for selected file info
        self.shuffle = tk.IntVar()  # variable for shuffle checkbox
        self.style = ttk.Style()  # style for ttk widgets
        self.loop = tk.StringVar(value="")  # variable for loop checkbox
        self.colours = tk.StringVar(value="")  # variable for colours checkbox
        self.colour_change = tk.StringVar(
            value=""
        )  # variable for colour change checkbox
        self.arduino_cmd_var = tk.StringVar(value="")  # variable for arduino command
        self.arduino_running = False
        self._initialize_ui()  # initialize the UI

    def _initialize_ui(self):
        """Initialize the UI."""

        # 1. Widgets for the left frame
        # Noise name entry
        self.noise_name_entry = ttk.Entry(
            self.left_frame, textvariable=self.noise_name_var
        )
        self.noise_name_entry.insert(0, "Noise_name")
        self.noise_name_entry.grid(row=4, column=0, padx=10, pady=5)

        self.arduino_command = ttk.Entry(
            self.left_frame, textvariable=self.arduino_cmd_var
        )
        self.arduino_command.insert(0, "Arduino Command")
        self.arduino_command.grid(row=8, column=0, padx=10, pady=5)

        # Insert a traffic light indicator for the arduino
        self.arduino_light = tk.Label(
            self.left_frame, text="no stim", bg="red", width=10
        )
        self.arduino_light.grid(row=9, column=0, padx=10, pady=5)

        # Variables and labels for the left frame entries
        variables = [
            self.checkerboard_var,
            self.window_size_var,
            self.noise_frequency_var,
            self.noise_duration_var,
        ]
        labels = [
            "checkerboard size",
            "window size",
            "noise frequency",
            "noise duration",
        ]

        for i, (label, var) in enumerate(zip(labels, variables)):
            ttk.Label(self.left_frame, text=label).grid(
                row=i, column=0, sticky=tk.W, pady=5
            )
            ttk.Entry(self.left_frame, textvariable=var).grid(row=i, column=1, pady=5)

        # Generate noise button
        self.generate_noise_button = ttk.Button(
            self.left_frame, text="generate noise", command=self.on_generate_noise
        )
        self.generate_noise_button.grid(row=4, column=1, pady=5, padx=0)

        self.send_arduino_cmd = ttk.Button(
            self.left_frame, text="send to arduino", command=self.on_send_arduino_cmd
        )
        self.send_arduino_cmd.grid(row=8, column=1, pady=5, padx=0)

        self.stop_arduino_cmd = ttk.Button(
            self.left_frame, text="stop arduino", command=self.stop_arduino
        )
        self.stop_arduino_cmd.grid(row=9, column=1, pady=5, padx=0)
        # Size label
        self.size_label = ttk.Label(self.left_frame, text="")
        self.size_label.grid(row=5, column=1, padx=10, pady=5)

        # Shuffle checkbox
        self.shuffle_box = ttk.Checkbutton(
            self.left_frame,
            text="shuffle",
            variable=self.shuffle,
            onvalue=1,
            offvalue=0,
        )
        self.shuffle_box.grid(row=5, column=0, pady=0, padx=0)

        # Help button at bottom left
        self.help_button = ttk.Button(
            self.left_frame, text="help", command=self.show_help
        )
        self.help_button.grid(row=10, column=0, pady=10, padx=0, sticky=tk.W)

        # 2. Widgets for the right frame
        self.root.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(
            0, weight=1
        )  # This makes the listbox expand horizontally
        self.right_frame.grid_columnconfigure(1, weight=0)

        # Scrollbar for right frame
        self.scrollbar = ttk.Scrollbar(self.right_frame, orient="vertical")
        self.scrollbar.grid(row=3, column=1, sticky="ns", padx=0)

        # File listbox
        self.file_listbox = tk.Listbox(
            self.right_frame,
            height=25,
            width=30,
            selectmode=tk.SINGLE,
            yscrollcommand=self.scrollbar.set,
        )
        self.file_listbox.grid(
            row=4, column=0, columnspan=1, pady=5, sticky=tk.N + tk.S + tk.E + tk.W
        )
        self.scrollbar.config(command=self.file_listbox.yview)

        # Loop Frame
        loop_frame = ttk.Frame(self.right_frame)
        loop_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=tk.W)
        # Loop label
        self.loop_label = ttk.Label(loop_frame, text="Loops", width=7)
        self.loop_label.pack(side=tk.LEFT, padx=2)
        # Loop entry
        self.loop_entry = ttk.Entry(loop_frame, textvariable=self.loop, width=5)
        self.loop_entry.pack(side=tk.LEFT, padx=5)
        self.loop_entry.insert(0, "1")

        # Colour Frame
        self.colour_label = ttk.Label(loop_frame, text="colour_logic", width=12)
        self.colour_label.pack(side=tk.LEFT, padx=2)

        # Colour Frame
        self.colour_entry = ttk.Entry(loop_frame, textvariable=self.colours, width=10)
        self.colour_entry.pack(side=tk.LEFT, padx=5)
        self.colour_entry.insert(0, "white")

        colour_frame = ttk.Frame(self.right_frame)
        colour_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky=tk.W)

        self.colour_change_label = ttk.Label(
            colour_frame, text="change_every:", width=13
        )
        self.colour_change_label.pack(side=tk.LEFT, padx=2)

        self.colour_change = ttk.Entry(
            colour_frame, textvariable=self.colour_change, width=10
        )
        self.colour_change.pack(side=tk.LEFT, padx=5)
        self.colour_change.insert(0, "1")

        self.colour_change_frames = ttk.Label(colour_frame, text="Frames.", width=12)
        self.colour_change_frames.pack(side=tk.LEFT, padx=2)

        # Button frame for right frame
        button_frame = ttk.Frame(self.right_frame)
        button_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=tk.W)

        self.play_noise = ttk.Button(
            button_frame,
            text="play noise",
            command=self.on_play_noise,
            style="Play.TButton",
        )
        self.play_noise.pack(side=tk.LEFT, padx=5)

        self.stop_noise = ttk.Button(
            button_frame, text="stop noise", command=self.on_stop_noise
        )
        self.stop_noise.pack(side=tk.LEFT, padx=5)

        self.refresh_button = ttk.Button(
            button_frame, text="refresh", command=self.refresh_file_list
        )
        self.refresh_button.pack(side=tk.LEFT, padx=5)

        self.change_data_dir_button = ttk.Button(
            button_frame, text="change data dir", command=self.change_data_directory
        )
        self.change_data_dir_button.pack(side=tk.LEFT, padx=5)

        # Selected file info label
        self.selected_file_info_label = ttk.Label(
            self.right_frame, textvariable=self.selected_file_info_var, width=40
        )
        self.selected_file_info_label.grid(row=3, column=0, pady=5, sticky=tk.E)

        # 3. Function calls
        self.refresh_file_list()
        self.checkerboard_var.trace_add("write", self.compute_size)
        self.window_size_var.trace_add("write", self.compute_size)
        self.noise_frequency_var.trace_add("write", self.compute_size)
        self.noise_duration_var.trace_add("write", self.compute_size)
        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)
        self.compute_size()  # Call compute_size initially to set the label text

    def on_generate_noise(self):
        """Generate noise and save it to a file."""

        self.generate_noise_button.config(text="Generating...")  # change button text
        self.root.update()  # update the UI

        # Extract values and convert to appropriate data types
        checkerboard_size = int(self.checkerboard_var.get())
        window_size = tuple(map(int, self.window_size_var.get().split(",")))
        noise_frequency = int(self.noise_frequency_var.get())
        noise_duration = float(self.noise_duration_var.get())
        file_name = str(self.noise_name_var.get())

        # Validate the noise name's suffix
        file_name_path = Path(file_name)
        if file_name_path.suffix and file_name_path.suffix != ".h5":
            print(
                f"Invalid suffix '{file_name_path.suffix}' provided. Using '.h5' instead."
            )
            file_name = file_name_path.stem  # Get the filename without any suffix
        file_name += ".h5"
        complete_file_name = fpspy.config.user_stimuli_dir(self.config) / file_name

        # Calculate number of frames needed
        frames = int(noise_duration * 60 * noise_frequency)
        width = window_size[0]
        height = window_size[1]

        # Call the function to generate the noise
        if self.shuffle.get() == 0:
            fpspy.create_noise.generate_and_store_3d_array(
                frames,
                checkerboard_size,
                width,
                height,
                noise_frequency,
                name=complete_file_name,
            )
        else:
            fpspy.shuffle_noise.generate_and_store_3d_array(
                frames,
                checkerboard_size,
                width,
                height,
                noise_frequency,
                name=complete_file_name,
            )
        # Update the list of files
        self.refresh_file_list()
        # Reset the button text
        self.generate_noise_button.config(text="Generate Noise")

    def on_play_noise(self):
        """Play the selected noise file."""

        # Check selected file in the file_listbox
        index = self.file_listbox.curselection()
        noise_name = self.file_listbox.get(index[0])
        noise_path = fpspy.config.user_stimuli_dir(self.config) / noise_name
        _, _, frames, frame_rate = load_noise_info(noise_path)
        s_frames = schedule_frames(frames, frame_rate)

        queue_data = {
            "file": noise_name,
            "loops": int(self.loop_entry.get()),
            "colours": self.colours.get(),
            "change_logic": int(self.colour_change.get()),
            "s_frames": s_frames,
        }
        with self.lock:
            for _ in range(self.nr_processes):
                self.queue1.put(
                    queue_data
                )  # Put the noise name in the queue for each window thread to read
        self.arduino_running = True
        # self.arduino_light.config(bg="green")
        # # change text of the button
        # self.arduino_light.config(text="Stim running")


    def on_stop_noise(self):
        """Stop the noise playback."""
        with self.lock:
            for _ in range(self.nr_processes):
                self.queue1.put(
                    "stop"
                )  # Put "stop" in the queue for the pyglet thread to read
        self.arduino_done_callback()

    def refresh_file_list(self):
        """Refresh the list of .h5py files in the stimuli directory."""

        stimuli_dir = fpspy.config.user_stimuli_dir(self.config)

        if stimuli_dir.is_dir():
            files = [f.name for f in stimuli_dir.iterdir() if f.suffix == ".h5"]
            self.file_listbox.delete(0, tk.END)  # Clear the listbox
            for file in files:
                self.file_listbox.insert(tk.END, file)
        else:
            print(f"'{stimuli_dir}' directory does not exist.")

    # Your refresh_file_list method code here, use self where needed.

    def on_file_select(self, event):
        """Update the selected file info label when a file is selected.
        Parameters
        ----------
        event : tkinter.Event
            The event object (unused).

        """

        if index := self.file_listbox.curselection():
            try:
                file_name = self.file_listbox.get(index[0])

                # Get some info about the selected file and display it:
                with h5py.File(fpspy.config.user_stimuli_dir(self.config) / file_name, "r") as f:
                    noise_size = f["Noise"].shape
                    fps = f["Frame_Rate"][()]
                    checkerboard_size = f["Checkerboard_Size"][()]
                    shuffle = f["Shuffle"][()]
                duration = noise_size[0] / fps / 60
                self.selected_file_info_var.set(
                    f"size: {checkerboard_size}, fps: {fps}, time: {duration:.2f} min, "
                    f"shuffle: {shuffle}"
                )
                self.style.configure(
                    "Play.TButton", background="green"
                )  # Change the button color to green

            except KeyError:
                # This could happen if the file is not a valid noise file
                self.style.configure(
                    "Play.TButton", background="SystemButtonFace"
                )  # Change the button color back to default
                self.selected_file_info_var.set("")
        else:
            self.style.configure(
                "Play.TButton", background="SystemButtonFace"
            )  # Change the button color back to default
            self.selected_file_info_var.set("")

    def compute_size(self, *args):
        """Compute the estimated size of the noise file and update the label text."""
        with contextlib.suppress(ValueError):
            noise_frequency = int(self.noise_frequency_var.get())
            noise_duration = float(self.noise_duration_var.get())
            frames = int(noise_duration * 60 * noise_frequency)
            width, height = map(int, self.window_size_var.get().split(","))

            size_in_bytes = frames * width * height  # uint8: 1 byte per element

            # Do some unit conversions to make the size more readable:
            if size_in_bytes < 1024:
                size_str = f"{size_in_bytes} bytes"
            elif size_in_bytes < 1024**2:
                size_in_kb = size_in_bytes / 1024
                size_str = f"{size_in_kb:.2f} KB"
            elif size_in_bytes < 1024**3:
                size_in_mb = size_in_bytes / (1024**2)
                size_str = f"{size_in_mb:.2f} MB"
            else:
                size_in_gb = size_in_bytes / (1024**3)
                size_str = f"{size_in_gb:.2f} GB"

            self.size_label.config(
                text=f"Estimated size: {size_str}"
            )  # Update the label text

    def on_send_arduino_cmd(self, *args):
        """Send the arduino command to the arduino."""
        self.arduino_running = True
        self.arduino_light.config(bg="green")
        # change text of the button
        self.arduino_light.config(text="Stim running")

        with self.lock:
            for _ in range(self.nr_processes):
                self.queue1.put("white_screen")
        with self.ard_lock:
            self.ard_queue.put(self.arduino_cmd_var.get())
            arduino_thread = threading.Thread(target=self.arduino_done_callback)
            self.root.after(100, arduino_thread.start)
        return

    def arduino_done_callback(self):
        while self.arduino_running:
            with self.status_lock:
                if not self.status_queue.empty():
                    status = self.status_queue.get()
                    if status == "done":
                        self.arduino_light.config(bg="red")
                        self.arduino_light.config(text="no stim")
                        self.arduino_running = False
                        break
                else:
                    time.sleep(0.1)  # Avoid busy-waiting

        # Clear any remaining messages in the queue
        with self.status_lock:
            while not self.status_queue.empty():
                self.status_queue.get()

    def stop_arduino(self, *args):
        """Stop the arduino."""
        # self.arduino_spinner.stop()
        with self.lock:
            # Drain any pending items in the queue so "stop" is processed next
            try:
                while True:
                    self.queue1.get_nowait()
            except Exception:
                pass
            for _ in range(self.nr_processes):
                self.queue1.put("stop")
        # self.arduino_running = False
        # with self.status_lock:
        #     self.status_queue.get()
        # self.arduino_light.config(bg="red")

    def show_help(self):
        """Show help dialog with directory information."""
        fpspy.help_dialog.show_directories_help(self.root, self.config)

    def change_data_directory(self):
        """Open dialog to change the data directory and save to config."""
        current_data_dir = Path(self.config["paths"]["data_dir"])

        # Create custom dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Data Directory")
        dialog.geometry("500x150")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")

        # Current directory display
        current_var = tk.StringVar(value=str(current_data_dir))
        ttk.Label(dialog, text="Current data directory:").pack(pady=5)
        current_entry = ttk.Entry(dialog, textvariable=current_var, width=60, state="readonly")
        current_entry.pack(pady=5)

        # Button frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        result = [None]  # Use list to allow modification in nested functions

        def select_directory():
            new_dir = filedialog.askdirectory(
                title="Select Data Directory",
                initialdir=current_var.get(),
                parent=dialog
            )
            if new_dir:
                current_var.set(new_dir)

        def use_default():
            default_dir = fpspy.config.default_data_dir()
            current_var.set(str(default_dir))

        def apply_changes():
            new_path = Path(current_var.get())
            result[0] = new_path
            dialog.destroy()

        def cancel():
            dialog.destroy()

        # Buttons
        ttk.Button(button_frame, text="Browse...", command=select_directory).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Use Default", command=use_default).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Apply", command=apply_changes).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=5)

        # Wait for dialog to close
        self.root.wait_window(dialog)

        # Process result
        if result[0] and result[0] != current_data_dir:
            # Update the config
            self.config["paths"]["data_dir"] = str(result[0])

            # Save to user config file
            fpspy.config.save_user_config(self.config)

            # Refresh the file list to show files from new directory
            self.refresh_file_list()

            messagebox.showinfo(
                "Data Directory Changed",
                f"Data directory changed to:\n{result[0]}\n\nStimuli will now be saved to:\n{result[0] / 'stimuli'}"
            )

    def on_close(self):
        """Called when the window is closed."""
        # Can add cleanup here if needed
        # Disconnect Arduino
        with self.ard_lock:
            self.ard_queue.put("destroy")
        with self.lock:
            for _ in range(self.nr_processes):
                self.queue1.put(
                    "stop"
                )  # Put "stop" in the queue for the pyglet thread to read
                self.queue1.put(
                    "destroy"
                )  # Put "destroy" in the queue for the pyglet thread.

        # Will be read by the pyglet thread to close the window.
        self.root.destroy()


def tkinter_app(
    config, queue1, lock, ard_queue, ard_lock, status_queue, status_lock, nr_processes
):
    """Create the tkinter GUI and run the mainloop. Used to run the GUI in a separate
    process.

    Parameters
    ----------
    queue : multiprocessing.Queue
        The queue used to communicate with the pyglet thread.
    """

    root = tk.Tk()  # Create the root window
    app = NoiseGeneratorApp(
        root, config, queue1, lock, ard_queue, ard_lock, status_queue, status_lock, nr_processes
    )  # Create the NoiseGeneratorApp instance
    root.protocol(
        "WM_DELETE_WINDOW", app.on_close
    )  # Set the on_close method as the callback for the close button
    root.mainloop()  # Run the mainloop


# Your compute_size method code here, use self where needed.

# For debugging purposes:
# if __name__ == "__main__":
#     tkinter_app(Queue())


def load_noise_info(path: str):
    """Load the noise and metadata (width, height, frames, frame rate) from .h5 file.

    Parameters
    ----------
    file : str | Path
        Path to the noise file.
    Returns
    -------
    noise : np.ndarray
        Noise data.
    width : int
        Width of the noise.
    height : int
        Height of the noise.
    frames : int
        Number of frames in the noise.
    frame_rate : int
        Frame rate of the noise.

    """
    with h5py.File(path, "r") as f:
        size = f["Noise"][:].shape
        frame_rate = f["Frame_Rate"][()]

    width = size[2]
    height = size[1]
    frames = size[0]

    return width, height, frames, frame_rate


def schedule_frames(frames, frame_rate):
    current_time = time.perf_counter()

    fps = frame_rate
    frame_duration = 1 / fps
    buffer = 5
    s_frames = np.linspace(
        current_time, current_time + frames * frame_duration, frames + 1
    )
    return s_frames
