"""PyQt6 instrument panel for fpspy.

An instrument panel first: what the rig is doing right now, visible at a
glance, with the controls next to the readout they affect. The generation and
log tabs described in GUI_manifesto.md are not built yet; the tab bar is here
so they can be added without moving anything.

Threading and blocking
----------------------
The Qt event loop never waits on a queue. Every command is sent and forgotten;
replies and device events are collected by a timer (`_poll`) and folded into
the displayed state when they arrive. That is also why the panel shows a
"pending" state rather than an instant confirmation: an idle ESP32 polls its
serial line about once a second, so a command can legitimately take that long
to have any visible effect.

Last known state, not live truth
--------------------------------
The firmware has no status query. Everything shown about the device was
inferred from unsolicited lines it printed since the panel connected, and the
panel says so rather than implying it is reading the device now.
"""

import json
import logging
import signal
import time
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHeaderView,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import numpy as np

import fpspy.arduino_commands as arduino_commands
import fpspy.config
import fpspy.generators
import fpspy.stim
from fpspy.gui_client import PresenterClient

_logger = logging.getLogger(__name__)

# Suffixes create_program() can actually open (fpspy.stim.create_program).
STIM_SUFFIXES = (".h5", ".hdf5", ".py") + fpspy.stim.MOVIE_SUFFIXES

# How often replies and device events are collected. Fast enough that a button
# press feels answered, slow enough to be free.
POLL_INTERVAL_MS = 50

# Device lines kept in the event view. Old ones are dropped; the file log has
# the full history.
MAX_EVENT_LINES = 200

def _format_duration(seconds: float) -> str:
    """Seconds as something readable at a glance, keeping the raw value."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    clock = (
        f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
    )
    return f"{clock} ({seconds:,.0f} s)"


def _format_value(value) -> str:
    """One metadata value as text, without numpy's array/scalar decoration."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _format_value(value.item())
        return ", ".join(_format_value(v) for v in value.tolist())
    if isinstance(value, np.generic):
        return _format_value(value.item())
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def describe_stimulus(info: dict, path: Optional[Path] = None) -> list[tuple[str, str]]:
    """Turn a preview_hdf5() dict into ordered (label, value) rows.

    Kept separate from the widget so the formatting can be tested directly,
    and because the two HDF5 format versions report different keys: v0 has no
    channels/zoom/triggers, and v1 puts everything beyond the array shape into
    a free-form `metadata` dict. Anything not recognised here is still shown,
    from that dict, rather than being silently dropped.
    """
    rows: list[tuple[str, str]] = []

    n_frames = info.get("n_frames")
    total = info.get("n_total_frames")
    repeats = info.get("n_mask_repeats", 1)
    if n_frames is not None:
        if total is not None and repeats and repeats > 1:
            rows.append(
                ("Frames", f"{n_frames:,} × {repeats} mask repeats = {total:,}")
            )
        else:
            rows.append(("Frames", f"{n_frames:,}"))

    fps = info.get("fps")
    # v1 stores per-frame durations, in which case there is no single rate.
    rows.append(
        ("Frame rate", f"{fps:g} Hz" if fps else "variable (per-frame durations)")
    )

    if (duration := info.get("duration")) is not None:
        rows.append(("Duration", _format_duration(duration)))

    width, height = info.get("width"), info.get("height")
    zoom = info.get("zoom", 1) or 1
    if width is not None and height is not None:
        size = f"{width} × {height} px"
        if zoom > 1:
            size += f"  →  {width * zoom} × {height * zoom} px on screen"
        rows.append(("Frame size", size))
    if zoom > 1:
        # For a checkerboard stimulus this is the box size: one stored pixel
        # becomes a zoom × zoom block.
        rows.append(("Zoom / box size", f"{zoom} px"))

    if (channels := info.get("channels")) is not None:
        rows.append(("Channels", str(channels)))
    if (mask_shape := info.get("channel_mask.shape")) is not None:
        rows.append(("Channel mask", " × ".join(str(d) for d in mask_shape)))
    if (n_triggers := info.get("n_triggers")) is not None:
        rows.append(("Triggers", f"{n_triggers:,}"))

    for key, value in (info.get("metadata") or {}).items():
        rows.append((str(key).replace("_", " ").capitalize(), _format_value(value)))

    if (version := info.get("version")) is not None:
        rows.append(("HDF5 format", f"v{version}"))
    if path is not None:
        rows.append(("File size", f"{path.stat().st_size / 1e6:,.1f} MB"))
    return rows


_OK = "color: #1b7f3b; font-weight: bold;"
_PENDING = "color: #0066cc; font-weight: bold;"
_WARN = "color: #cc4400; font-weight: bold;"
# palette(mid) was wrong here: "mid" is a 3D-frame shading role, not a text
# role, so it carries no contrast guarantee against the window background and
# reads as dark-on-dark. placeholder-text is the role meant for de-emphasised
# text and follows the active theme.
_MUTED = "color: palette(placeholder-text);"


class InstrumentPanel(QMainWindow):
    """The main window: device state, stimulus transport, window state."""

    def __init__(self, client: PresenterClient, config: dict,
                 config_path: Optional[Path] = None):
        super().__init__()
        self.client = client
        self.config = config
        # The file whose values were merged over the bundled defaults. Shown
        # in the Settings tab and named when a setting looks wrong, because
        # "which config is actually in force" is the usual confusion.
        self.config_path = fpspy.config.active_config_path(config_path)
        self.stim_dir = fpspy.config.user_data_dir(config)

        # --- displayed state -------------------------------------------
        # Everything below is inferred from replies; nothing is read back.
        self.window_states = {i: "idle" for i in range(1, client.n_windows + 1)}
        self.playing = False
        self.total_frames = 0
        self.current_frame = -1
        self.trigger_count = 0
        self.device_last_line: Optional[str] = None
        self.device_last_time: Optional[float] = None
        # None until the presenter reports opening the port; False for a
        # dummy or unopenable port.
        self.device_connected: Optional[bool] = None
        self.pending: Optional[str] = None
        # False for a selected file that could not be previewed, which keeps
        # Play and Load disabled rather than letting it kill a presenter.
        self.selection_is_playable = False

        self._build_ui()
        self._refresh_file_list()
        self._update_controls()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(POLL_INTERVAL_MS)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.setWindowTitle("fpspy — instrument panel")

        tabs = QTabWidget()
        tabs.addTab(self._build_instrument_tab(), "Instrument")
        tabs.addTab(self._build_generation_tab(), "Generation")
        tabs.addTab(self._build_settings_tab(), "Settings")
        self.setCentralWidget(tabs)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Ask the finished layout what it needs rather than hardcoding a size
        # that silently stops being right when a widget is added: below this,
        # Qt starts overlapping widgets instead of shrinking them.
        self.setMinimumSize(self.minimumSizeHint())
        self.resize(self.minimumSizeHint().expandedTo(QSize(940, 900)))

    def _build_instrument_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        layout.addWidget(self._build_device_group())
        layout.addWidget(self._build_stimulus_group(), stretch=1)
        layout.addWidget(self._build_transport_group())
        layout.addWidget(self._build_windows_group())
        return tab

    def _build_generation_tab(self) -> QWidget:
        """One form per registered generator (fpspy.generators.REGISTRY).

        Adding a new stimulus template means registering it there; this tab
        does not change.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        intro = QLabel(
            f"Writes into the stimulus folder shown in the Instrument tab: "
            f"{self.stim_dir}"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(_MUTED)
        layout.addWidget(intro)

        for generator in fpspy.generators.REGISTRY.values():
            layout.addWidget(self._build_generator_group(generator))
        layout.addStretch()
        return tab

    def _build_generator_group(self, generator: fpspy.generators.Generator) -> QGroupBox:
        group = QGroupBox(generator.name)
        form = QGridLayout(group)

        row = 0
        form.addWidget(QLabel("File name:"), row, 0)
        name_edit = QLineEdit(generator.key)
        form.addWidget(name_edit, row, 1, 1, 3)
        row += 1

        getters: dict[str, Callable[[], object]] = {}
        widgets: dict[str, object] = {}
        for param in generator.params:
            form.addWidget(QLabel(param.label + ":"), row, 0)
            if param.kind == "bool":
                widget = QCheckBox()
                widget.setChecked(bool(param.default))
                getters[param.name] = widget.isChecked
            elif param.kind == "int":
                widget = QSpinBox()
                widget.setRange(int(param.minimum), int(param.maximum))
                widget.setValue(int(param.default))
                getters[param.name] = widget.value
            elif param.kind == "float":
                widget = QDoubleSpinBox()
                widget.setDecimals(3)
                widget.setRange(float(param.minimum), float(param.maximum))
                widget.setValue(float(param.default))
                getters[param.name] = widget.value
            else:  # "choice": valid values depend on another field's value.
                widget = QComboBox()
                dep_widget = widgets[param.depends_on]
                dep_getter = getters[param.depends_on]

                def refresh_choices(
                    widget=widget,
                    param=param,
                    dep_getter=dep_getter,
                ):
                    # Keep the current selection if it's still valid, so
                    # editing an unrelated field doesn't reset this one.
                    previous = widget.currentData()
                    choices = param.choices_fn(dep_getter())
                    widget.blockSignals(True)
                    widget.clear()
                    for choice in choices:
                        widget.addItem(str(choice), choice)
                    widget.setEnabled(bool(choices))
                    idx = widget.findData(previous)
                    widget.setCurrentIndex(idx if idx >= 0 else 0)
                    widget.blockSignals(False)

                refresh_choices()
                if isinstance(dep_widget, QCheckBox):
                    dep_widget.stateChanged.connect(lambda *_: refresh_choices())
                else:
                    dep_widget.valueChanged.connect(lambda *_: refresh_choices())
                # currentData() is None for an empty (no valid choice) combo.
                getters[param.name] = widget.currentData
            if param.tooltip:
                widget.setToolTip(param.tooltip)
            widgets[param.name] = widget
            form.addWidget(widget, row, 1, 1, 3)
            row += 1

        status_label = QLabel("")
        status_label.setWordWrap(True)
        form.addWidget(status_label, row, 0, 1, 4)
        row += 1

        generate_btn = QPushButton("Generate")
        form.addWidget(generate_btn, row, 3)

        def on_generate():
            generate_btn.setEnabled(False)
            generate_btn.setText("Generating…")
            QApplication.processEvents()
            try:
                kwargs = {name: getter() for name, getter in getters.items()}
                path = generator.generate(self.stim_dir, name_edit.text(), **kwargs)
            except Exception as e:
                _logger.exception(f"Generator {generator.key!r} failed")
                status_label.setText(f"Failed: {e}")
                status_label.setStyleSheet(_WARN)
            else:
                status_label.setText(f"Wrote {path.name}")
                status_label.setStyleSheet(_OK)
                self._refresh_file_list()
            finally:
                generate_btn.setEnabled(True)
                generate_btn.setText("Generate")

        generate_btn.clicked.connect(on_generate)
        return group

    def _build_settings_tab(self) -> QWidget:
        """Show the settings actually in force, and where they came from.

        The effective config is what load_config() produced after merging the
        bundled defaults, any user or -c file, and the [windows.default]
        template. Showing the merged result rather than one file's text is the
        point: which file a value came from is the usual source of confusion,
        and the merge is not visible in any single file.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Merged over the bundled defaults from:"))
        path_label = QLabel(str(self.config_path))
        path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        path_label.setStyleSheet("font-weight: bold;")
        source_row.addWidget(path_label, stretch=1)
        layout.addLayout(source_row)

        summary = QLabel(
            f"Arduino port: {fpspy.config.get_arduino_port(self.config)}   ·   "
            f"Windows: {self.client.n_windows}   ·   "
            f"Presentation delay: "
            f"{fpspy.config.get_presentation_delay(self.config):g} s"
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        note = QLabel(
            "Read-only. Edit the file above and restart to change these; the "
            "panel reads the config once, at startup."
        )
        note.setWordWrap(True)
        note.setStyleSheet(_MUTED)
        layout.addWidget(note)

        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setPlainText(fpspy.config.config_to_str(self.config))
        # A config is read line by line and column alignment matters.
        view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(view, stretch=1)
        return tab

    def _build_device_group(self) -> QGroupBox:
        group = QGroupBox("Arduino")
        layout = QVBoxLayout(group)

        # Two separate readouts. The connection line is what the panel knows
        # about the port and changes rarely; the activity line is the last
        # thing the device said. Merging them would let a routine info line
        # overwrite the fact that there is no device at all.
        conn_row = QHBoxLayout()
        self.device_conn_label = QLabel("Waiting for the presenter to open the port…")
        self.device_conn_label.setStyleSheet(_MUTED)
        conn_row.addWidget(self.device_conn_label, stretch=1)
        self.trigger_label = QLabel("Triggers: 0")
        conn_row.addWidget(self.trigger_label)
        layout.addLayout(conn_row)

        self.device_state_label = QLabel("No device output yet")
        self.device_state_label.setStyleSheet(_MUTED)
        layout.addWidget(self.device_state_label)

        caption = QLabel(
            "Last known state, inferred from what the device has printed. "
            "The firmware cannot be queried, and answers an idle command in "
            "up to a second."
        )
        caption.setWordWrap(True)
        caption.setStyleSheet(_MUTED)
        layout.addWidget(caption)

        cmd_row = QHBoxLayout()
        cmd_row.addWidget(QLabel("Command:"))
        self.command_box = QComboBox()
        for cmd, description in arduino_commands.COMMANDS:
            self.command_box.addItem(cmd)
            self.command_box.setItemData(
                self.command_box.count() - 1, description, Qt.ItemDataRole.ToolTipRole
            )
        self.command_box.currentIndexChanged.connect(self._on_command_changed)
        cmd_row.addWidget(self.command_box)
        self.send_cmd_btn = QPushButton("Send")
        self.send_cmd_btn.clicked.connect(self._on_send_command)
        cmd_row.addWidget(self.send_cmd_btn)
        self.stop_device_btn = QPushButton("Stop device")
        self.stop_device_btn.setToolTip(
            "Interrupt the protocol the device is running (a chirp, a flash "
            'series) by sending "b".\n'
            "This is read mid-protocol rather than through the command "
            "parser, so unlike every other command it takes effect at once.\n"
            'If nothing is running, the device answers "Unknown command".'
        )
        self.stop_device_btn.clicked.connect(self._on_stop_device)
        cmd_row.addWidget(self.stop_device_btn)
        cmd_row.addStretch()
        layout.addLayout(cmd_row)

        self.command_help = QLabel("")
        self.command_help.setWordWrap(True)
        self.command_help.setStyleSheet(_MUTED)
        layout.addWidget(self.command_help)
        self._on_command_changed()

        self.event_view = QPlainTextEdit()
        self.event_view.setReadOnly(True)
        self.event_view.setMaximumBlockCount(MAX_EVENT_LINES)
        # Fixed, not merely minimum: otherwise the log expands into the
        # space the stimulus metadata table should get.
        self.event_view.setFixedHeight(90)
        self.event_view.setPlaceholderText("Device output appears here")
        layout.addWidget(self.event_view)

        return group

    def _build_stimulus_group(self) -> QGroupBox:
        group = QGroupBox("Stimulus")
        layout = QVBoxLayout(group)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Folder:"))
        self.dir_label = QLabel(str(self.stim_dir))
        self.dir_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        dir_row.addWidget(self.dir_label, stretch=1)
        change_dir_btn = QPushButton("Change…")
        change_dir_btn.clicked.connect(self._on_change_dir)
        dir_row.addWidget(change_dir_btn)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_file_list)
        dir_row.addWidget(refresh_btn)
        layout.addLayout(dir_row)

        # File list and its metadata side by side, so reading the properties
        # never means losing sight of which file they belong to.
        browse_row = QHBoxLayout()
        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self._on_file_selected)
        browse_row.addWidget(self.file_list, stretch=3)

        self.metadata_table = QTableWidget(0, 2)
        self.metadata_table.horizontalHeader().setVisible(False)
        self.metadata_table.verticalHeader().setVisible(False)
        self.metadata_table.setAlternatingRowColors(True)
        self.metadata_table.setShowGrid(False)
        # Read-only, but selectable so a value can be copied out.
        self.metadata_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.metadata_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.metadata_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        # A floor only: enough to be usable, low enough that the group's
        # minimum still fits inside the window's. Extra height goes here
        # because this is the only row in the tab with a stretch factor, and
        # longer metadata scrolls inside the table.
        self.metadata_table.setMinimumHeight(150)
        self.file_list.setMinimumHeight(150)
        browse_row.addWidget(self.metadata_table, stretch=4)
        layout.addLayout(browse_row, stretch=1)

        # Carries the "cannot read this file" message; the table shows nothing
        # in that case, so the reason needs somewhere of its own to appear.
        self.file_info_label = QLabel("")
        self.file_info_label.setStyleSheet(_MUTED)
        self.file_info_label.setWordWrap(True)
        layout.addWidget(self.file_info_label)

        options = QGridLayout()
        options.addWidget(QLabel("Loops:"), 0, 0)
        self.loops_spin = QSpinBox()
        self.loops_spin.setRange(1, 10_000)
        options.addWidget(self.loops_spin, 0, 1)

        options.addWidget(QLabel("LED colours:"), 0, 2)
        self.colours_edit = QLineEdit()
        self.colours_edit.setPlaceholderText("e.g. white or led_610,led_560")
        self.colours_edit.setToolTip(
            "Comma-separated firmware LED commands, sent during the stimulus.\n"
            "Known LED commands: " + ", ".join(arduino_commands.COLOUR_COMMANDS)
        )
        self.colours_edit.textChanged.connect(self._on_colours_changed)
        options.addWidget(self.colours_edit, 0, 3)

        options.addWidget(QLabel("Change every:"), 0, 4)
        self.change_logic_spin = QSpinBox()
        self.change_logic_spin.setRange(1, 100_000)
        self.change_logic_spin.setSuffix(" pattern(s)")
        self.change_logic_spin.setToolTip(
            "How many consecutive stimulus patterns share one colour. "
            "1 sends a single fixed colour before the first frame."
        )
        options.addWidget(self.change_logic_spin, 0, 5)
        options.setColumnStretch(3, 1)
        layout.addLayout(options)

        self.colours_warning = QLabel("")
        self.colours_warning.setWordWrap(True)
        self.colours_warning.setStyleSheet(_WARN)
        layout.addWidget(self.colours_warning)

        return group

    def _build_transport_group(self) -> QGroupBox:
        group = QGroupBox("Transport")
        layout = QVBoxLayout(group)

        btn_row = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._on_play)
        btn_row.addWidget(self.play_btn)
        self.lazy_textures_check = QCheckBox("Lazy textures")
        self.lazy_textures_check.setToolTip(
            "Upload each frame's texture when it is first needed, instead of "
            "building them all before the first frame.\n"
            "Faster to start and much lighter on GPU memory for a long "
            "stimulus, but the uploads then happen inside the render loop, "
            "where they can cost dropped frames.\n"
            "Only applies to HDF5 (TextureSequence) stimuli. Off matches the "
            "`fpspy play` default."
        )
        btn_row.addWidget(self.lazy_textures_check)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self.stop_btn)
        btn_row.addSpacing(20)
        self.white_btn = QPushButton("White screen")
        self.white_btn.setToolTip("Show white on every window until the next Stop.")
        self.white_btn.clicked.connect(self._on_white_screen)
        btn_row.addWidget(self.white_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        step_row = QHBoxLayout()
        self.load_btn = QPushButton("Load for stepping")
        self.load_btn.clicked.connect(self._on_load)
        step_row.addWidget(self.load_btn)
        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.clicked.connect(self._on_prev)
        step_row.addWidget(self.prev_btn)
        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self._on_next)
        step_row.addWidget(self.next_btn)
        self.frame_label = QLabel("No stimulus loaded")
        step_row.addWidget(self.frame_label)
        step_row.addStretch()
        layout.addLayout(step_row)

        return group

    def _build_windows_group(self) -> QGroupBox:
        """One readout per presenter process, so a stuck window is visible.

        Each window is a separate process with its own reply queue, and they
        can disagree: one can still be loading while another is already
        playing. A single combined status would hide that, and hide a window
        that never answered at all.
        """
        group = QGroupBox("Presenter windows")
        layout = QHBoxLayout(group)
        self.window_labels = {}
        for idx in sorted(self.window_states):
            label = QLabel()
            label.setStyleSheet(_MUTED)
            label.setToolTip(
                f"Presenter process for window {idx}.\n"
                "idle: waiting for a command · loaded: a stimulus is ready to "
                "step · playing: rendering · finished: play ended, window open"
            )
            self.window_labels[idx] = label
            layout.addWidget(label)
        layout.addStretch()
        self._update_window_labels()
        return group

    # ------------------------------------------------------------------
    # Polling: the only place displayed state changes because of the backend
    # ------------------------------------------------------------------

    def _poll(self):
        for reply in self.client.drain_arduino():
            self._handle_device_event(reply)
        for reply in self.client.drain_replies():
            self._handle_presenter_reply(reply)
        dead = self.client.dead_processes()
        if dead:
            for p in dead:
                _logger.error(f"Presenter process {p.pid} exited ({p.exitcode}).")
            self.timer.stop()
            self._set_status(
                f"{len(dead)} presenter process(es) exited. Restart fpspy.", _WARN
            )

    def _handle_device_event(self, reply):
        text = reply.payload.get("text", "")
        self.device_last_line = text
        self.device_last_time = time.monotonic()

        if reply.kind == "arduino_trigger":
            # Triggers arrive at frame rate; counting them keeps the event view
            # readable and still shows the wire is alive.
            self.trigger_count += 1
            self.trigger_label.setText(f"Triggers: {self.trigger_count}")
            return

        connection = {
            "arduino_connected": (True, _OK),
            # A dummy port accepts every command and answers nothing, so the
            # controls are left enabled but the panel must not claim a device.
            "arduino_dummy": (False, _WARN),
            "arduino_disconnected": (False, _WARN),
        }
        if reply.kind in connection:
            self.device_connected, style = connection[reply.kind]
            self.device_conn_label.setText(text)
            self.device_conn_label.setStyleSheet(style)
            self._append_event(text)
            if not self.device_connected:
                self._set_status(
                    f"{text}. Commands will be accepted and go nowhere. "
                    f"Set [arduino] port in {self.config_path}.",
                    _WARN,
                )
            return

        if reply.kind == "arduino_unknown_command":
            # Printed only for "b", the interrupt byte. It reaches the command
            # parser (and so gets reported as unrecognised) exactly when no
            # protocol was running to consume it mid-loop. Not a fault: a stop
            # with the device already idle produces this every time.
            self._append_event(f"{text} — nothing was running to interrupt")
        else:
            self._append_event(text)

        states = {
            "arduino_ready": ("Device ready", _OK),
            "arduino_done": ("Device finished its protocol", _OK),
            "arduino_trigger_test": ("Test trigger sent", _OK),
        }
        label, style = states.get(reply.kind, (f"Last line: {text}", _MUTED))
        self.device_state_label.setText(label)
        self.device_state_label.setStyleSheet(style)

    def _handle_presenter_reply(self, reply):
        idx = reply.sender_idx
        match reply.kind:
            case "total_frames":
                self.total_frames = reply.payload.get("total", 0)
                self.current_frame = -1
                self.window_states[idx] = "loaded"
                self.pending = None
                self._set_status(
                    f"Loaded {self.total_frames} frames. Press Next to show frame 0."
                )
            case "stepped":
                self.current_frame = reply.payload.get("frame", -1)
                self.pending = None
            case "no_stimulus":
                self.window_states[idx] = "idle"
                self.pending = None
                self._set_status(f"Window {idx} has no stimulus loaded.", _WARN)
            case "play_finished":
                self.window_states[idx] = "finished"
                if all(s != "playing" for s in self.window_states.values()):
                    self.playing = False
                    self.pending = None
                    self._set_status("Playback finished.", _OK)
            case "done":
                # Sent on stop and when a presentation ends. The presenter
                # clears the screen and returns its step position to -1, but
                # keeps the stimulus loaded, so stepping stays available and
                # starts again from frame 0. Mirror both halves of that here,
                # or Prev/Next would offer moves the presenter disagrees with.
                self.window_states[idx] = "idle"
                self.current_frame = -1
                self.pending = None
                if all(s != "playing" for s in self.window_states.values()):
                    self.playing = False
        self._update_frame_label()
        self._update_window_labels()
        self._update_controls()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_command_changed(self):
        command = self.command_box.currentText()
        self.command_help.setText(arduino_commands.DESCRIPTIONS.get(command, ""))

    def _on_send_command(self):
        command = self.command_box.currentText()
        self.client.send_device_command(command)
        self._append_event(f"→ {command}")
        self._set_status(f"Sent {command!r}; the device answers when it next polls.",
                         _PENDING)

    def _on_stop_device(self):
        """Interrupt whatever protocol the device is running.

        Sent on its own rather than via the presenter's "stop", so it works
        when nothing is playing — which is exactly when a protocol started
        from the command picker is running.
        """
        self.client.send_device_command(arduino_commands.INTERRUPT)
        self._append_event(f"→ {arduino_commands.INTERRUPT}")
        self._set_status("Interrupting the running protocol…", _PENDING)

    def _on_change_dir(self):
        chosen = QFileDialog.getExistingDirectory(
            self, "Stimulus folder", str(self.stim_dir)
        )
        if not chosen:
            return
        self.stim_dir = Path(chosen)
        self.dir_label.setText(str(self.stim_dir))
        self._refresh_file_list()

    def _refresh_file_list(self):
        self.file_list.clear()
        if not self.stim_dir.is_dir():
            self._set_status(f"Folder does not exist: {self.stim_dir}", _WARN)
            return
        names = sorted(
            f.name for f in self.stim_dir.iterdir() if f.suffix.lower() in STIM_SUFFIXES
        )
        self.file_list.addItems(names)
        self._update_controls()

    def _on_file_selected(self):
        """Show everything known about the selected file.

        Reading the metadata is a sub-millisecond header read (preview_hdf5
        touches no array data), so it is done inline on selection rather than
        behind a button.
        """
        path = self.selected_path()
        self.selection_is_playable = path is not None
        self.file_info_label.setStyleSheet(_MUTED)

        if path is None:
            self.file_info_label.setText("")
            self._show_metadata([])
        elif path.suffix.lower() in (".h5", ".hdf5"):
            try:
                info = fpspy.stim.StimArray.preview_hdf5(path)
            except Exception as e:
                # A file that cannot be previewed cannot be played either.
                # Saying so now, and refusing to play it, beats finding out
                # when the windows go black and a presenter process dies.
                _logger.warning(f"Could not preview {path}: {e}")
                self.selection_is_playable = False
                self.file_info_label.setText(f"Cannot read this file: {e}")
                self.file_info_label.setStyleSheet(_WARN)
                self._show_metadata([])
            else:
                self.file_info_label.setText("")
                self._show_metadata(describe_stimulus(info, path))
        else:
            # Only HDF5 carries readable metadata; a shader or movie stimulus
            # is taken on trust and fails at load time if bad.
            self.file_info_label.setText(
                f"{path.suffix} stimulus — properties are only readable for HDF5."
            )
            self._show_metadata([])
        self._update_controls()

    def _show_metadata(self, rows: list[tuple[str, str]]):
        """Fill the properties table, replacing whatever was there."""
        self.metadata_table.clearContents()
        self.metadata_table.setRowCount(len(rows))
        for row, (label, value) in enumerate(rows):
            # Both columns keep the default text colour. These are readings,
            # not decoration, so the property name has to be as legible as the
            # value it labels.
            self.metadata_table.setItem(row, 0, QTableWidgetItem(label))
            self.metadata_table.setItem(row, 1, QTableWidgetItem(value))
        self.metadata_table.resizeRowsToContents()

    def _on_colours_changed(self):
        unknown = arduino_commands.unknown_colours(self.colours_edit.text())
        if unknown:
            self.colours_warning.setText(
                "Not firmware commands, and would be ignored silently: "
                + ", ".join(unknown)
            )
        else:
            self.colours_warning.setText("")
        self._update_controls()

    def _stim_config_for(self, path: Path) -> Optional[str]:
        """The per-stimulus config string create_program() expects.

        Only TextureSequence (HDF5) takes an option, so everything else gets
        None. Mirrors the choice made in the CLI's play command.
        """
        if path.suffix.lower() in (".h5", ".hdf5"):
            return json.dumps(
                {"lazy_textures": self.lazy_textures_check.isChecked()}
            )
        return None

    def _on_play(self):
        path = self.selected_path()
        if path is None:
            return
        colours = self.colours_edit.text().strip() or None
        self.client.send_all(
            "play",
            stim_path=path,
            stim_config=self._stim_config_for(path),
            loops=self.loops_spin.value(),
            t0=time.perf_counter(),
            close_after=False,
            arduino_colours=colours,
            change_logic=self.change_logic_spin.value(),
        )
        self.playing = True
        self.total_frames = 0
        self.current_frame = -1
        for idx in self.window_states:
            self.window_states[idx] = "playing"
        self.pending = "play"
        delay = fpspy.config.get_presentation_delay(self.config)
        self._set_status(f"Playing {path.name}; first frame in ~{delay:g} s.", _PENDING)
        self._update_window_labels()
        self._update_frame_label()
        self._update_controls()

    def _on_stop(self):
        self.client.send_all("stop")
        self.pending = "stop"
        self._set_status("Stopping…", _PENDING)
        self._update_controls()

    def _on_white_screen(self):
        self.client.send_all("white_screen")
        self._set_status("White screen until the next Stop.", _PENDING)

    def _on_load(self):
        path = self.selected_path()
        if path is None:
            return
        self.client.send_all(
            "load",
            stim_path=path,
            stim_config=self._stim_config_for(path),
            loops=self.loops_spin.value(),
        )
        self.total_frames = 0
        self.current_frame = -1
        self.pending = "load"
        self._set_status(f"Loading {path.name}…", _PENDING)
        self._update_controls()

    def _on_next(self):
        if self.current_frame >= self.total_frames - 1:
            return
        self.client.send_all("step_next")
        self.pending = "step"
        self._update_controls()

    def _on_prev(self):
        if self.current_frame < 0:
            return
        self.client.send_all("step_prev")
        self.pending = "step"
        self._update_controls()

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def selected_path(self) -> Optional[Path]:
        item = self.file_list.currentItem()
        if item is None:
            return None
        return self.stim_dir / item.text()

    def _append_event(self, text: str):
        stamp = time.strftime("%H:%M:%S")
        self.event_view.appendPlainText(f"{stamp}  {text}")

    def _set_status(self, message: str, style: str = ""):
        self.status_bar.showMessage(message)
        self.status_bar.setStyleSheet(style)

    def _update_frame_label(self):
        if self.total_frames == 0:
            self.frame_label.setText("No stimulus loaded")
        elif self.current_frame < 0:
            self.frame_label.setText(f"Frame: — [0–{self.total_frames - 1}]")
        else:
            self.frame_label.setText(
                f"Frame: {self.current_frame} [0–{self.total_frames - 1}]"
            )

    def _update_window_labels(self):
        for idx, label in self.window_labels.items():
            state = self.window_states[idx]
            label.setText(f"Window {idx}: {state}")
            label.setStyleSheet(_OK if state == "playing" else _MUTED)

    def _update_controls(self):
        """Disable whatever cannot be done right now, rather than failing later."""
        busy = self.playing or self.pending is not None
        can_open = self.selected_path() is not None and self.selection_is_playable
        stepping = self.total_frames > 0 and not self.playing

        self.play_btn.setEnabled(can_open and not busy)
        self.load_btn.setEnabled(can_open and not busy)
        # The option only reaches TextureSequence, so it is meaningless for a
        # shader or movie stimulus. Greying it out says that without a note.
        selected = self.selected_path()
        self.lazy_textures_check.setEnabled(
            selected is not None and selected.suffix.lower() in (".h5", ".hdf5")
        )
        # Stop is the emergency exit and stays available: it also clears a
        # white screen and resets the LEDs, neither of which implies that
        # something is playing.
        self.stop_btn.setEnabled(True)
        self.white_btn.setEnabled(not self.playing)
        self.prev_btn.setEnabled(stepping and self.current_frame >= 0)
        self.next_btn.setEnabled(stepping and self.current_frame < self.total_frames - 1)

    def closeEvent(self, event):
        self.timer.stop()
        self.client.shutdown()
        event.accept()


def qt_app(client: PresenterClient, config: dict,
           config_path: Optional[Path] = None):
    """Run the panel's event loop until the window closes."""
    app = QApplication([])
    # Qt owns the main loop, so Python only sees SIGINT when it gets a slice.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    heartbeat = QTimer()
    heartbeat.timeout.connect(lambda: None)
    heartbeat.start(100)

    panel = InstrumentPanel(client, config, config_path)
    panel.show()
    app.exec()
