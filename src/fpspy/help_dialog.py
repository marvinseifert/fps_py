"""
Help dialog functionality for fpspy GUI.

This module contains the help dialog that shows directory locations
and other useful information to users.
"""

import tkinter as tk
from tkinter import ttk
from pathlib import Path
import fpspy.config


def show_directories_help(parent_window, config):
    """Show a dialog with all important directories.

    Parameters
    ----------
    parent_window : tk.Tk
        The parent window for the dialog
    config : dict
        The application configuration dictionary
    """
    # Get all the directories
    data_dir = Path(config["paths"]["data_dir"])
    stimuli_dir = fpspy.config.user_stimuli_dir(config)
    logs_dir = fpspy.config.user_log_dir()
    config_dir = fpspy.config.user_config_dir()
    config_file = fpspy.config.user_config_file_path()

    # Create help dialog
    dialog = tk.Toplevel(parent_window)
    dialog.title("Directory Locations")
    dialog.geometry("600x300")
    dialog.resizable(True, True)
    dialog.transient(parent_window)
    dialog.grab_set()

    # Center the dialog
    dialog.update_idletasks()
    x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
    y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
    dialog.geometry(f"+{x}+{y}")

    # Create scrollable text widget
    frame = ttk.Frame(dialog)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    text_widget = tk.Text(frame, wrap=tk.WORD, padx=10, pady=10)
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text_widget.yview)
    text_widget.configure(yscrollcommand=scrollbar.set)

    # Directory information
    config_details = fpspy.config.config_to_str(config)

    info_text = f"""fpspy Directory Locations

DATA DIRECTORY (configurable):
{data_dir}

STIMULI DIRECTORY:
{stimuli_dir}
- Generated stimulus files (.h5) are saved here
- Files shown in the main GUI file list

LOG DIRECTORY (system location):
{logs_dir}
- Presentation logs and timing information
- CSV files with stimulus details and performance data

CONFIG DIRECTORY (system location):
{config_dir}

USER CONFIG FILE:
{config_file}
- Custom settings override defaults
- Edit this file to change window positions, ports, etc.

Note: You can change the data directory using the "Change Data Dir" button.
The stimuli directory will automatically be created inside your chosen data directory.
Log and config directories are fixed system locations.

==================================================
CURRENT CONFIGURATION:
==================================================

{config_details}"""

    text_widget.insert(tk.END, info_text)
    text_widget.config(state=tk.DISABLED)

    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Close button
    button_frame = ttk.Frame(dialog)
    button_frame.pack(pady=10)
    ttk.Button(button_frame, text="Close", command=dialog.destroy).pack()