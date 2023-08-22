import tkinter as tk
from tkinter import ttk
from pathlib import Path
import create_noise
import multiprocessing
from multiprocessing import Process, Queue
import h5py
import shuffle_noise



class NoiseGeneratorApp:
    def __init__(self, root, queue):
        self.queue = queue
        self.root = root
        self.root.title("Noise Generator GUI")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.geometry("600x500")

        # Create frames for left and right sections
        self.left_frame = ttk.Frame(self.root, padding="10")
        self.left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.right_frame = ttk.Frame(self.root, padding="10")
        self.right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Variables to store the values of the input fields
        standard_values = ["5", "1000,1000", "20", "5.0"]
        self.checkerboard_var = tk.StringVar(value=standard_values[0])
        self.window_size_var = tk.StringVar(value=standard_values[1])
        self.noise_frequency_var = tk.StringVar(value=standard_values[2])
        self.noise_duration_var = tk.StringVar(value=standard_values[3])
        self.noise_name_var = tk.StringVar()
        self.selected_file_info_var = tk.StringVar(value="")
        self.shuffle = tk.IntVar()
        self.style = ttk.Style()



        self._initialize_ui()



    def _initialize_ui(self):
        # UI initialization code
        self.noise_name_entry = ttk.Entry(self.left_frame, textvariable=self.noise_name_var)
        self.noise_name_entry.insert(0, "Noise_name")
        self.noise_name_entry.grid(row=4, column=0, padx=10, pady=5)

        variables = [self.checkerboard_var, self.window_size_var, self.noise_frequency_var, self.noise_duration_var]
        labels = ["checkerboard size", "window size", "noise frequency", "noise duration"]

        for i, (label, var) in enumerate(zip(labels, variables)):
            ttk.Label(self.left_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=5)
            ttk.Entry(self.left_frame, textvariable=var).grid(row=i, column=1, pady=5)

        self.root.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)  # This makes the listbox expand horizontally
        self.right_frame.grid_columnconfigure(1, weight=0)

        self.scrollbar = ttk.Scrollbar(self.right_frame, orient="vertical")
        self.scrollbar.grid(row=2, column=1, sticky='ns', padx=0)

        # Listbox
        self.file_listbox = tk.Listbox(self.right_frame, height=25, width=30, selectmode=tk.SINGLE, yscrollcommand=self.scrollbar.set)
        self.file_listbox.grid(row=2, column=0, columnspan=1, pady=5, sticky=tk.N + tk.S + tk.E + tk.W)

        self.scrollbar.config(command=self.file_listbox.yview)



        self.generate_noise_button = ttk.Button(self.left_frame, text="generate noise", command=self.on_generate_noise)
        self.generate_noise_button.grid(row=4, column=1, pady=5, padx=0)

        button_frame = ttk.Frame(self.right_frame)
        button_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=tk.W)

        self.play_noise = ttk.Button(button_frame, text="play noise", command=self.on_play_noise, style='Play.TButton')

        self.play_noise.pack(side=tk.LEFT, padx=5)

        self.stop_noise = ttk.Button(button_frame, text="stop noise", command=self.on_stop_noise)
        self.stop_noise.pack(side=tk.LEFT, padx=5)

        self.refresh_button = ttk.Button(button_frame, text="Refresh", command=self.refresh_file_list)
        self.refresh_button.pack(side=tk.LEFT, padx=5)
        self.refresh_file_list()


        self.checkerboard_var.trace_add("write", self.compute_size)
        self.window_size_var.trace_add("write", self.compute_size)
        self.noise_frequency_var.trace_add("write", self.compute_size)
        self.noise_duration_var.trace_add("write", self.compute_size)

        self.size_label = ttk.Label(self.left_frame, text="")
        self.size_label.grid(row=5, column=1, padx=10, pady=5)

        self.selected_file_info_label = ttk.Label(self.right_frame, textvariable=self.selected_file_info_var, width=40)
        self.shuffle_box = ttk.Checkbutton(self.left_frame, text='shuffle',
                                          variable=self.shuffle, onvalue=1, offvalue=0)
        self.shuffle_box.grid(row=5, column=0, pady=0, padx=0)

        self.selected_file_info_label.grid(row=1, column=0, pady=5, sticky=tk.E)
        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)

        # Call compute_size initially to set the label text
        self.compute_size()

    def on_generate_noise(self):
        self.generate_noise_button.config(text="Generating...")
        self.root.update()
        # Extract values and convert to appropriate data types
        checkerboard_size = int(self.checkerboard_var.get())
        window_size = tuple(map(int, self.window_size_var.get().split(",")))
        noise_frequency = int(self.noise_frequency_var.get())
        noise_duration = float(self.noise_duration_var.get())
        file_name = str(self.noise_name_var.get())

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
        frames = int(noise_duration * 60 * noise_frequency)
        width = window_size[0]
        height = window_size[1]

        # Call the function to generate the noise
        if self.shuffle.get() == 0:
            create_noise.generate_and_store_3d_array(frames, checkerboard_size, width, height,noise_frequency, name=complete_file_name)
        else:
            shuffle_noise.generate_and_store_3d_array(frames, checkerboard_size, width, height,noise_frequency, name=complete_file_name)
        self.refresh_file_list()
        self.generate_noise_button.config(text="Generate Noise")

    # Your on_generate_noise method code here, use self where needed.

    def on_play_noise(self):
        # Check selected file in the file_listbox
        index = self.file_listbox.curselection()
        noise_name  = self.file_listbox.get(index[0])
        self.queue.put(noise_name)

    def on_stop_noise(self):
        self.queue.put("stop")

    # Your on_play_noise method code here, use self where needed.

    def refresh_file_list(self):
        """Refresh the list of .h5py files in the stimuli directory."""
        stimuli_dir = Path("stimuli")

        if stimuli_dir.exists() and stimuli_dir.is_dir():
            files = [f.name for f in stimuli_dir.iterdir() if f.suffix == '.h5']
            self.file_listbox.delete(0, tk.END)  # Clear the listbox
            for file in files:
                self.file_listbox.insert(tk.END, file)
        else:
            print(f"'{stimuli_dir}' directory does not exist.")

    # Your refresh_file_list method code here, use self where needed.

    def on_file_select(self, event):
        index = self.file_listbox.curselection()
        if index:
            try:
                file_name = self.file_listbox.get(index[0])
                # This is just an example. You can provide any information you want.
                with h5py.File(Path("stimuli", file_name), "r") as f:
                    noise_size = f["Noise"].shape
                    fps = f["Frame_Rate"][()]
                    checkerboard_size = f["Checkerboard_Size"][()]
                    shuffle = f["Shuffle"][()]
                duration = noise_size[0] / fps / 60
                self.selected_file_info_var.set(f"size: {checkerboard_size}, fps: {fps}, time: {duration:.2f} min, shuffle: {shuffle}")
                self.style.configure('Play.TButton', background='green')
            except KeyError:
                self.style.configure('Play.TButton', background='SystemButtonFace')
                self.selected_file_info_var.set("")
        else:
            self.style.configure('Play.TButton', background='SystemButtonFace')
            self.selected_file_info_var.set("")
    def compute_size(self, *args):
        noise_frequency = int(self.noise_frequency_var.get())
        noise_duration = float(self.noise_duration_var.get())
        frames = int(noise_duration * 60 * noise_frequency)
        width, height = map(int, self.window_size_var.get().split(","))

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

        self.size_label.config(text=f"Estimated size: {size_str}")

    def on_close(self):
        # Do any cleanup here
        self.root.destroy()


def tkinter_app(queue):
    root = tk.Tk()
    app = NoiseGeneratorApp(root, queue)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

# Your compute_size method code here, use self where needed.


if __name__ == "__main__":
    tkinter_app(Queue())
