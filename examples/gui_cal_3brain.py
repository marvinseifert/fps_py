"""
3Brain Calibration GUI (Qt Version).

A GUI for stepping through stimulus frames and capturing images for calibration.
Supports manual next/previous stepping and automated capture-remaining functionality.

This replaces the tkinter version which doesn't work well on Wayland.
"""

import logging
import multiprocessing as mp
import signal
import time
from pathlib import Path
from typing import Optional

import requests
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QLineEdit,
    QProgressBar,
    QStatusBar,
    QSizePolicy,
)

import fpspy.queue
import fpspy.config
import fpspy._logging as _logging
import fpspy.play_3brain
import typer

_logger = logging.getLogger(__name__)

app = typer.Typer(help="3Brain Calibration GUI.")


class CaptureWorker(QThread):
    """Worker thread for capture operations."""

    progress = pyqtSignal(int, str)  # progress percentage, status message
    finished = pyqtSignal(bool, str)  # success, message
    frame_updated = pyqtSignal(int)  # current frame index
    error_occurred = pyqtSignal(str)  # error message (for red/orange display)
    stats_updated = pyqtSignal(float, float)  # captures_per_sec, eta_seconds

    def __init__(
        self,
        capture_dir: Path,
        server_url: str,
        frame_idx: int,
        exposure_ms: Optional[int] = None,
        single_capture: bool = True,
    ):
        super().__init__()
        self.capture_dir = capture_dir
        self.server_url = server_url
        self.frame_idx = frame_idx
        self.exposure_ms = exposure_ms
        self.single_capture = single_capture
        self.cancelled = False

        # These will be set for capture-remaining mode
        self.total_frames = 0
        self.current_frame = -1
        self.start_frame = 0  # Frame to start capturing from
        self.cmd_queues: list[mp.Queue] = []
        self.status_queue: Optional[mp.Queue] = None
        self.n_windows = 0

        # For rate calculation
        self.capture_times: list[float] = []
        self.last_frame_time = None

    def cancel(self):
        self.cancelled = True

    def wait_for_exposure(self):
        if self.last_frame_time is None:
            return
        assert self.exposure_ms is not None
        wait_until = self.last_frame_time + (self.exposure_ms / 1_000_000)
        now = time.perf_counter()
        wait_dur = wait_until - now
        if wait_dur > 0:
            _logger.info(f"Waiting {round(wait_dur*1000)} ms for exposure")
            time.sleep(wait_dur)

    def capture_image(self, frame_idx: int, retry: bool = True) -> bool:
        """Capture an image from the server and save it.

        On first failure, retries once if retry=True.
        """
        timeout = 15  # Reduced from 30

        for attempt in range(2 if retry else 1):
            try:
                self.wait_for_exposure()
                response = requests.post(
                    f"{self.server_url}/capture",
                    params={"format": "dng"},
                    timeout=timeout,
                )
                response.raise_for_status()
                self.last_frame_time = time.perf_counter()

                filename = f"capture_{frame_idx:04d}.dng"
                filepath = self.capture_dir / filename
                filepath.write_bytes(response.content)
                return True

            except requests.RequestException as e:
                if attempt == 0 and retry:
                    _logger.warning(f"Capture failed (attempt 1), retrying: {e}")
                    self.error_occurred.emit(f"Retry: {e}")
                    time.sleep(0.5)  # Brief pause before retry
                    continue
                _logger.error(f"Capture failed: {e}")
                self.error_occurred.emit(f"Capture failed: {e}")
                return False
        return False

    def send_to_all(self, cmd_type: str, **kwargs):
        """Send a command to all presenter windows."""
        for queue in self.cmd_queues:
            fpspy.queue.put_onto(queue, cmd_type, **kwargs)

    def wait_for_responses(self, timeout: float = 5.0):
        """Wait for responses from all presenter windows."""
        first_response = None
        for _ in range(self.n_windows):
            response = self.status_queue.get(timeout=timeout)
            if first_response is None:
                first_response = response
        return first_response

    def run(self):
        if self.single_capture:
            self._run_single_capture()
        else:
            self._run_capture_remaining()

    def _run_single_capture(self):
        self.progress.emit(50, f"Capturing frame {self.frame_idx}...")
        success = self.capture_image(self.frame_idx)
        if success:
            self.finished.emit(True, f"Captured frame {self.frame_idx}")
        else:
            self.finished.emit(False, "Capture failed")

    def _run_capture_remaining(self):
        """Capture from current frame to end."""
        # Determine start point
        if self.current_frame < 0:
            # Need to step to frame 0 first
            self.send_to_all("step_next")
            try:
                response = self.wait_for_responses(timeout=5)
                if isinstance(response, dict) and "stepped" in response:
                    self.current_frame = response["stepped"]
                    self.frame_updated.emit(self.current_frame)
                else:
                    self.error_occurred.emit(
                        f"Failed to step to first frame: {response}"
                    )
                    self.finished.emit(False, "Failed to step to first frame")
                    return
            except Exception as e:
                self.error_occurred.emit(f"Step error: {e}")
                self.finished.emit(False, f"Step error: {e}")
                return

        self.start_frame = self.current_frame
        frames_to_capture = self.total_frames - self.start_frame

        if frames_to_capture <= 0:
            self.finished.emit(True, "No frames remaining to capture")
            return

        # Capture current frame first (already displayed)
        capture_start_time = time.perf_counter()
        frames_captured = 0

        for i in range(frames_to_capture):
            if self.cancelled:
                self.finished.emit(False, f"Cancelled at frame {self.current_frame}")
                return

            frame_to_capture = self.start_frame + i
            # Progress is the frame number (GUI sets max to total_frames)
            self.progress.emit(
                frame_to_capture,
                f"Capturing frame {frame_to_capture}/{self.total_frames - 1}",
            )

            # Step to next frame (except for first iteration if already on a frame)
            if i > 0:
                self.send_to_all("step_next")
                try:
                    response = self.wait_for_responses(timeout=5)
                    if isinstance(response, dict) and "stepped" in response:
                        self.current_frame = response["stepped"]
                        self.frame_updated.emit(self.current_frame)
                    else:
                        self.error_occurred.emit(
                            f"Step failed at frame {frame_to_capture}"
                        )
                        self.finished.emit(
                            False, f"Step failed at frame {frame_to_capture}"
                        )
                        return
                except Exception as e:
                    self.error_occurred.emit(
                        f"Step error at frame {frame_to_capture}: {e}"
                    )
                    self.finished.emit(
                        False, f"Step error at frame {frame_to_capture}: {e}"
                    )
                    return

            # Capture
            capture_time_start = time.perf_counter()
            if not self.capture_image(self.current_frame):
                self.finished.emit(
                    False, f"Capture failed at frame {self.current_frame}"
                )
                return

            # Track timing for rate calculation
            capture_time = time.perf_counter() - capture_time_start
            self.capture_times.append(capture_time)
            frames_captured += 1

            # Calculate and emit stats
            elapsed = time.perf_counter() - capture_start_time
            if elapsed > 0:
                captures_per_sec = frames_captured / elapsed
                remaining_frames = frames_to_capture - frames_captured
                eta = remaining_frames / captures_per_sec if captures_per_sec > 0 else 0
                self.stats_updated.emit(captures_per_sec, eta)

        self.progress.emit(self.total_frames, "Complete")
        self.finished.emit(True, f"Captured {frames_captured} frames")


class CalibrationGui(QMainWindow):
    """Qt GUI for stepping through stimulus frames and capturing images."""

    def __init__(
        self,
        out_dir: Optional[Path],
        config: dict,
        cmd_queues: list[mp.Queue],
        status_queue: mp.Queue,
    ):
        super().__init__()
        self.out_dir = out_dir
        self.config = config
        self.cmd_queues = cmd_queues
        self.status_queue = status_queue
        self.n_windows = len(cmd_queues)

        # State
        self.current_frame = -1
        self.total_frames = 0
        self.stimulus_loaded = False
        self.capture_dir: Optional[Path] = out_dir
        self.worker: Optional[CaptureWorker] = None
        self._exposure_ms = None
        self._last_frame_time = None

        # Server settings
        self.server_url = "http://139.184.162.187:5000"

        self._init_ui()

        if self.capture_dir:
            self.capture_dir.mkdir(parents=True, exist_ok=True)

    def _init_ui(self):
        """Initialize the UI components."""
        self.setWindowTitle("3Brain Calibration")
        self.setMinimumSize(600, 520)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # --- Stimulus Section ---
        stim_group = QGroupBox("Stimulus")
        stim_layout = QHBoxLayout(stim_group)
        stim_layout.addWidget(QLabel("File:"))
        self.stim_path_label = QLabel("Loading...")
        self.stim_path_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        stim_layout.addWidget(self.stim_path_label)
        layout.addWidget(stim_group)

        # --- Frame Navigation Section ---
        nav_group = QGroupBox("Frame Navigation")
        nav_layout = QVBoxLayout(nav_group)

        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        nav_layout.addWidget(self.frame_label)

        btn_row = QHBoxLayout()
        self.prev_btn = QPushButton("< Previous")
        self.prev_btn.clicked.connect(self.on_previous)
        self.prev_btn.setEnabled(False)
        btn_row.addWidget(self.prev_btn)

        btn_row.addStretch()

        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self.on_next)
        self.next_btn.setEnabled(False)
        btn_row.addWidget(self.next_btn)

        nav_layout.addLayout(btn_row)
        layout.addWidget(nav_group)

        # --- Camera Section ---
        camera_group = QGroupBox("Camera Server")
        camera_layout = QVBoxLayout(camera_group)

        url_row = QHBoxLayout()
        url_row.addWidget(QLabel("URL:"))
        self.server_url_edit = QLineEdit(self.server_url)
        self.server_url_edit.setMinimumWidth(200)
        url_row.addWidget(self.server_url_edit)

        self.refresh_btn = QPushButton("Refresh Settings")
        self.refresh_btn.clicked.connect(self._refresh_settings)
        url_row.addWidget(self.refresh_btn)

        self.save_config_btn = QPushButton("Save Config")
        self.save_config_btn.clicked.connect(self.on_save_config)
        url_row.addWidget(self.save_config_btn)

        url_row.addStretch()

        camera_layout.addLayout(url_row)

        self.camera_settings_label = QLabel("Settings: Not fetched")
        camera_layout.addWidget(self.camera_settings_label)

        layout.addWidget(camera_group)

        # --- Capture Section ---
        capture_group = QGroupBox("Capture")
        capture_layout = QVBoxLayout(capture_group)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Output:"))
        self.capture_dir_label = QLabel(
            str(self.capture_dir.resolve()) if self.capture_dir else "Not set"
        )
        self.capture_dir_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        dir_row.addWidget(self.capture_dir_label)
        capture_layout.addLayout(dir_row)

        cap_btn_row = QHBoxLayout()
        self.capture_btn = QPushButton("Capture")
        self.capture_btn.clicked.connect(self.on_capture)
        self.capture_btn.setEnabled(False)
        cap_btn_row.addWidget(self.capture_btn)

        self.capture_remaining_btn = QPushButton("Capture Remaining")
        self.capture_remaining_btn.clicked.connect(self.on_capture_remaining)
        self.capture_remaining_btn.setEnabled(False)
        cap_btn_row.addWidget(self.capture_remaining_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.on_cancel)
        self.cancel_btn.setEnabled(False)
        cap_btn_row.addWidget(self.cancel_btn)

        cap_btn_row.addStretch()
        capture_layout.addLayout(cap_btn_row)

        layout.addWidget(capture_group)

        # --- Progress Section ---
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p% - Frame %v")
        progress_layout.addWidget(self.progress_bar)

        # Stats row (captures/sec and ETA)
        self.stats_label = QLabel("")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.stats_label)

        layout.addWidget(progress_group)

        # --- Busy/Error Indicator ---
        self.busy_label = QLabel("")
        self.busy_label.setStyleSheet("color: #0066cc; font-weight: bold;")
        self.busy_label.setWordWrap(True)
        layout.addWidget(self.busy_label)

        layout.addStretch()

        # --- Status Bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _capture_enabled(self) -> bool:
        """Determine if capture button should be enabled."""
        has_dir = self.capture_dir is not None
        frame_displayed = self.current_frame >= 0
        return has_dir and frame_displayed

    def update_status(self, message: str):
        """Update the status bar."""
        self.status_bar.showMessage(message)

    def update_frame_label(self):
        """Update the frame counter label."""
        last_frame = self.total_frames - 1
        if self.current_frame < 0:
            self.frame_label.setText(f"Frame: - [0–{last_frame}]")
        else:
            self.frame_label.setText(f"Frame: {self.current_frame} [0–{last_frame}]")

    def update_button_states(self):
        """Update button enabled/disabled states based on current state."""
        working = self.worker is not None and self.worker.isRunning()

        if working:
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.capture_btn.setEnabled(False)
            self.capture_remaining_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
        elif self.stimulus_loaded:
            self.prev_btn.setEnabled(self.current_frame >= 0)
            self.next_btn.setEnabled(self.current_frame < self.total_frames - 1)
            has_dir = self.capture_dir is not None
            frame_displayed = self.current_frame >= 0
            self.capture_btn.setEnabled(has_dir and frame_displayed)
            # Enable capture remaining if there are frames left
            has_remaining = (
                self.current_frame < self.total_frames - 1 or self.current_frame < 0
            )
            self.capture_remaining_btn.setEnabled(has_dir and has_remaining)
            self.cancel_btn.setEnabled(False)
        else:
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.capture_btn.setEnabled(False)
            self.capture_remaining_btn.setEnabled(False)
            self.cancel_btn.setEnabled(False)

    def send_to_all(self, cmd_type: str, **kwargs):
        """Send a command to all presenter windows."""
        for queue in self.cmd_queues:
            fpspy.queue.put_onto(queue, cmd_type, **kwargs)

    def wait_for_responses(self, timeout: float = 5.0):
        """Wait for responses from all presenter windows."""
        first_response = None
        for _ in range(self.n_windows):
            response = self.status_queue.get(timeout=timeout)
            if first_response is None:
                first_response = response
        return first_response

    def load_stimulus(self, stim_path: Path):
        """Load a stimulus file."""
        self.update_status(f"Loading {stim_path.name}...")

        self.send_to_all("load", stim_path=stim_path, stim_config=None, loops=1)

        try:
            response = self.wait_for_responses(timeout=10)
            if isinstance(response, dict) and "total_frames" in response:
                self.total_frames = response["total_frames"]
                self.current_frame = -1
                self.stimulus_loaded = True
                self.stim_path_label.setText(str(stim_path.resolve()))
                self.update_frame_label()
                self.update_button_states()
                self.update_status(
                    f"Loaded: {self.total_frames} frames. Press Next to show frame 0."
                )
            else:
                self.update_status(f"Load failed: {response}")
        except Exception as e:
            self.update_status(f"Load error: {e}")

    def on_next(self):
        """Step to the next frame."""
        if self.current_frame >= self.total_frames - 1:
            return
        self.step_next()

    def on_previous(self):
        """Step to the previous frame (or to cleared state at -1)."""
        if self.current_frame < 0:
            return
        self.step_prev()

    def step_next(self) -> bool:
        """Send step_next command to all presenters and update state."""
        self.send_to_all("step_next")

        try:
            response = self.wait_for_responses(timeout=5)
            if isinstance(response, dict) and "stepped" in response:
                self.current_frame = response["stepped"]
                self.update_frame_label()
                self.update_button_states()
                return True
        except Exception as e:
            self.update_status(f"Step error: {e}")
        return False

    def step_prev(self) -> bool:
        """Send step_prev command to all presenters and update state."""
        self.send_to_all("step_prev")

        try:
            response = self.wait_for_responses(timeout=5)
            if isinstance(response, dict) and "stepped" in response:
                self.current_frame = response["stepped"]
                self.update_frame_label()
                self.update_button_states()
                return True
        except Exception as e:
            self.update_status(f"Step error: {e}")
        return False

    def _refresh_settings(self):
        """Fetch and display camera settings from the server."""
        server_url = self.server_url_edit.text()
        self.update_status(f"Fetching settings from {server_url}...")

        try:
            response = requests.get(f"{server_url}/settings", timeout=5)
            response.raise_for_status()
            settings = response.json()

            exposure = settings.get("exposure_time", "N/A")
            gain = settings.get("analogue_gain", "N/A")
            settings_text = f"Exposure: {exposure} µs, Gain: {gain}"
            if not isinstance(exposure, int):
                raise ValueError(f"Non-integer exposure time received: {exposure=}")
            if not isinstance(gain, (int, float)):
                raise ValueError(f"Non-numeric gain received: {gain=}")
            self._exposure_ms = exposure

            self.camera_settings_label.setText(settings_text)
            self.update_status("Settings fetched successfully")

        except requests.RequestException as e:
            _logger.error(f"Failed to fetch settings: {e}")
            self.camera_settings_label.setText(f"Error: {e}")
            self.update_status(f"Failed to fetch settings: {e}")

    def exposure_ms(self) -> int:
        if self._exposure_ms is None:
            self._refresh_settings()
        assert self._exposure_ms is not None
        return self._exposure_ms

    def on_save_config(self):
        """Fetch camera config from server and save to output directory."""
        if self.capture_dir is None:
            self._show_error("No output directory set")
            return

        server_url = self.server_url_edit.text()
        self.update_status(f"Fetching config from {server_url}...")

        try:
            response = requests.get(f"{server_url}/config", timeout=5)
            response.raise_for_status()
            config_text = response.text

            file_path = self.capture_dir / "camera_config.txt"
            file_path.write_text(config_text)
            self.update_status(f"Config saved to {file_path}")

        except requests.RequestException as e:
            _logger.error(f"Failed to fetch config: {e}")
            self._show_error(f"Failed to fetch config: {e}")

    def on_capture(self):
        """Capture the current frame in a background thread."""
        if self.capture_dir is None:
            return

        self._clear_error()
        self.busy_label.setText("Capturing...")
        self.busy_label.setStyleSheet("color: #0066cc; font-weight: bold;")
        self.stats_label.setText("")

        self.worker = CaptureWorker(
            self.capture_dir,
            self.server_url_edit.text(),
            self.current_frame,
            self.exposure_ms(),
            single_capture=True,
        )
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error_occurred.connect(self._show_error)
        self.worker.start()
        self.update_button_states()

    def on_capture_remaining(self):
        """Start capturing remaining frames in a background thread."""
        if not self.stimulus_loaded or self.capture_dir is None:
            return

        self._clear_error()
        # Progress bar shows overall progress (0 to total_frames)
        self.progress_bar.setMaximum(self.total_frames)
        # Start from current position (or 0 if not yet started)
        start_frame = max(0, self.current_frame)
        self.progress_bar.setValue(start_frame)
        self.stats_label.setText("")

        self.worker = CaptureWorker(
            self.capture_dir,
            self.server_url_edit.text(),
            0,
            self.exposure_ms(),
            single_capture=False,
        )
        self.worker.total_frames = self.total_frames
        self.worker.current_frame = self.current_frame
        self.worker.cmd_queues = self.cmd_queues
        self.worker.status_queue = self.status_queue
        self.worker.n_windows = self.n_windows

        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.frame_updated.connect(self._on_frame_updated)
        self.worker.error_occurred.connect(self._show_error)
        self.worker.stats_updated.connect(self._on_stats_updated)
        self.worker.start()
        self.update_button_states()

        # Also print to console for tqdm-style output
        remaining = self.total_frames - start_frame
        print(
            f"\n[Capture] Starting capture of {remaining} frames from frame {start_frame}..."
        )

    def _on_worker_progress(self, progress: int, message: str):
        """Handle progress updates from worker."""
        self.progress_bar.setValue(progress)
        self.update_status(message)

    def _on_worker_finished(self, success: bool, message: str):
        """Handle worker completion."""
        self.busy_label.setText("")
        if success:
            self.progress_bar.setValue(self.progress_bar.maximum())
            self.busy_label.setStyleSheet("color: #0066cc; font-weight: bold;")
        else:
            # Don't change progress on failure, keep current position
            self._show_error(message)
        self.update_status(message)
        self.worker = None
        self.update_button_states()

        # Console output
        if success:
            print(f"[Capture] Complete: {message}")
        else:
            print(f"[Capture] FAILED: {message}")

    def _on_frame_updated(self, frame: int):
        """Handle frame update from capture worker."""
        self.current_frame = frame
        self.update_frame_label()

    def _on_stats_updated(self, captures_per_sec: float, eta_seconds: float):
        """Handle stats update from capture worker."""
        if eta_seconds < 60:
            eta_str = f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            eta_str = f"{eta_seconds / 60:.1f}m"
        else:
            eta_str = f"{eta_seconds / 3600:.1f}h"

        self.stats_label.setText(
            f"{captures_per_sec:.2f} captures/sec | ETA: {eta_str}"
        )

        # Also print to console
        print(
            f"\r[Capture] {captures_per_sec:.2f} cap/s | ETA: {eta_str} | Frame {self.current_frame}/{self.total_frames-1}",
            end="",
            flush=True,
        )

    def _show_error(self, message: str):
        """Show error message in orange/red."""
        self.busy_label.setText(message)
        self.busy_label.setStyleSheet("color: #cc4400; font-weight: bold;")

    def _clear_error(self):
        """Clear error display."""
        self.busy_label.setText("")
        self.busy_label.setStyleSheet("color: #0066cc; font-weight: bold;")

    def on_cancel(self):
        """Cancel the current operation."""
        if self.worker:
            self.worker.cancel()
            self.update_status("Cancelling...")

    def closeEvent(self, event):
        """Handle window close."""
        if self.worker:
            self.worker.cancel()
            self.worker.wait()
        self.send_to_all("stop")
        self.send_to_all("destroy")
        event.accept()


def qt_app(
    config: dict,
    capture_dir: Optional[Path],
    cmd_queues: list[mp.Queue],
    status_queue: mp.Queue,
    stim_path: Path,
):
    """
    Create the Qt GUI and run the event loop.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    capture_dir : Path, optional
        Directory to save captured images.
    cmd_queues : list[multiprocessing.Queue]
        Queues for sending commands to presenter processes (one per window).
    status_queue : multiprocessing.Queue
        Queue for receiving status updates from presenter processes.
    stim_path : Path
        Path to stimulus file to load.
    """
    app = QApplication([])

    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda *args: app.quit())
    # Timer to allow Python to process signals (Qt blocks the main loop)
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    window = CalibrationGui(capture_dir, config, cmd_queues, status_queue)
    window.show()

    # Load stimulus on startup (slight delay to let window show)
    QTimer.singleShot(100, lambda: window.load_stimulus(stim_path))

    app.exec()


@app.command()
def cal_gui(
    stim_path: Path = typer.Argument(
        ...,
        help="Path to stimulus file (.h5, .json, .py)",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to the TOML configuration file. If omitted, try loading user"
        f"config from {fpspy.config.user_config_dir()}. If that fails, an "
        "bundled default is used.",
    ),
    out_dir: Optional[Path] = typer.Option(
        None,
        "--out-dir",
        "-o",
        help=(
            "Directory to save logs and output data. If omitted, use "
            f"{fpspy.config.default_log_dir()}/<timestamp>/."
        ),
    ),
    enable_triggers: bool = typer.Option(
        False,
        "--triggers/--no-triggers",
        help="Enable Arduino triggers during presentation.",
    ),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    ),
):
    """Calibration GUI for stepping through stimulus frames and capturing images."""
    log_level = "WARNING" if verbose == 0 else "INFO" if verbose == 1 else "DEBUG"
    _logging.setup_main_logging(log_level)

    # Validate stimulus file exists
    if not stim_path.exists():
        _logger.error(f"Error: stimulus file not found: {stim_path}")
        raise typer.Exit(1)

    config = fpspy.config.load_config(config_path)
    if out_dir is None:
        out_dir = fpspy.config.create_outdir(config)
    out_dir.mkdir(parents=False, exist_ok=True)

    # We add subdirectory based on stimulus filename
    stim_stem = stim_path.stem
    sub_dir = out_dir / f"{stim_stem}"
    sub_dir.mkdir(parents=False, exist_ok=True)

    # Start presenter process (single window for calibration).
    presenter_processes, cmd_queues, status_queue = (
        fpspy.play_3brain.start_presenter_processes(
            config,
            sub_dir,
            delay=0,
            enable_triggers=enable_triggers,
            log_level=log_level,
        )
    )

    # Start calibration GUI (in main process for simplicity).
    capture_dir = sub_dir / "captures"
    qt_app(config, capture_dir, cmd_queues, status_queue, stim_path)

    # Cleanup
    for p in presenter_processes:
        p.join(timeout=2)
        if p.is_alive():
            p.terminate()


if __name__ == "__main__":
    app()
