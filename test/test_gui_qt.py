"""Tests for the instrument panel's state handling.

The panel is built for real (Qt offscreen), but no presenter processes exist:
the client's queues are plain queue.Queue, so replies can be posted by hand
and the panel's reaction to them checked. The poll timer is not relied on —
_poll() is called directly, so the tests do not depend on timing.
"""

import json
import os
import queue as std_queue

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import fpspy.arduino_commands as arduino_commands
import fpspy.config
import fpspy.fps_queue as fps_queue
import fpspy.stim
from fpspy.gui_client import PresenterClient

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from fpspy.gui_qt import (  # noqa: E402
    InstrumentPanel,
    _format_duration,
    _format_value,
    describe_stimulus,
)


class FakeProcess:
    pid = 1
    exitcode = None

    def is_alive(self):
        return True

    def join(self, timeout=None):
        pass

    def terminate(self):
        pass


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def client():
    return PresenterClient(
        [FakeProcess(), FakeProcess()],
        [std_queue.Queue(), std_queue.Queue()],
        [std_queue.Queue(), std_queue.Queue()],
        std_queue.Queue(),
    )


@pytest.fixture
def panel(qapp, client, tmp_path, monkeypatch):
    # The panel lists the config's data dir on startup; point it at an empty
    # temporary one so the test does not depend on the machine's stimuli.
    monkeypatch.setattr(fpspy.config, "user_data_dir", lambda config: tmp_path)
    config = fpspy.config.load_config()
    panel = InstrumentPanel(client, config)
    panel.timer.stop()  # Tests drive _poll() themselves.
    yield panel
    panel.deleteLater()


def post_reply(panel, kind, sender_idx=1, **payload):
    fps_queue.put_reply(panel.client.reply_queues[sender_idx - 1], kind, sender_idx,
                        **payload)
    panel._poll()


def post_device(panel, kind, text=""):
    fps_queue.put_reply(panel.client.arduino_queue, kind, 1, text=text)
    panel._poll()


def drain_commands(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def write_stimulus(path):
    """A real, minimal HDF5 stimulus, so the panel's preview succeeds.

    Empty files will not do: the panel refuses to play a file it cannot
    preview, which is the behaviour several of these tests depend on.
    """
    frames = np.zeros((2, 1, 1, 1), dtype=np.uint8)
    fpspy.stim.StimArray(frames, frame_durations=0.1, zoom=1).write_hdf5(path)
    return path


# --- device telemetry ---------------------------------------------------


def test_triggers_are_counted_not_listed(panel):
    """At frame rate, one line per trigger would bury everything else."""
    for _ in range(3):
        post_device(panel, "arduino_trigger", "Trigger")
    assert panel.trigger_label.text() == "Triggers: 3"
    assert panel.event_view.toPlainText() == ""


def test_ready_line_updates_the_state_readout(panel):
    post_device(panel, "arduino_ready", "Stimulator ready")
    assert "ready" in panel.device_state_label.text().lower()
    assert "Stimulator ready" in panel.event_view.toPlainText()


def test_unknown_command_is_explained_not_flagged_as_a_fault(panel):
    """Printed only for "b" arriving with no protocol running to consume it."""
    post_device(panel, "arduino_unknown_command", "Unknown command")
    assert "nothing was running to interrupt" in panel.event_view.toPlainText()


def test_unrecognised_line_is_still_shown_verbatim(panel):
    post_device(panel, "arduino_line", "Changing LED: 3")
    assert "Changing LED: 3" in panel.event_view.toPlainText()


# --- connection state ---------------------------------------------------


def test_connection_is_unknown_before_the_presenter_reports(panel):
    assert panel.device_connected is None
    assert "waiting" in panel.device_conn_label.text().lower()


def test_connected_port_is_shown(panel):
    post_device(panel, "arduino_connected", "Connected on /dev/ttyUSB0")
    assert panel.device_connected is True
    assert "/dev/ttyUSB0" in panel.device_conn_label.text()


def test_dummy_port_is_flagged_and_names_the_config_file(panel):
    """The commonest cause of "nothing happens": port = "dummy" in the config."""
    post_device(panel, "arduino_dummy", "No device: port is 'dummy' (dummy)")
    assert panel.device_connected is False
    assert "dummy" in panel.device_conn_label.text()
    assert str(panel.config_path) in panel.status_bar.currentMessage()


def test_unopenable_port_is_flagged(panel):
    post_device(panel, "arduino_disconnected", "Could not open /dev/ttyUSB0")
    assert panel.device_connected is False
    assert "Could not open" in panel.device_conn_label.text()


def test_a_later_info_line_does_not_erase_the_connection_readout(panel):
    """Two readouts, so routine chatter cannot hide "there is no device"."""
    post_device(panel, "arduino_dummy", "No device: port is 'dummy' (dummy)")
    post_device(panel, "arduino_line", "Changing LED: 3")
    assert "dummy" in panel.device_conn_label.text()


# --- device stop --------------------------------------------------------


def test_stop_device_sends_the_interrupt_byte(panel):
    """"b", not "O": only the interrupt byte aborts a running protocol."""
    panel._on_stop_device()
    (cmd,) = drain_commands(panel.client.cmd_queues[0])
    assert cmd.type == "device_cmd"
    assert cmd.args == ["b"]


def test_stop_device_works_when_nothing_is_playing(panel):
    """A protocol started from the command picker runs with playing=False."""
    assert panel.playing is False
    assert panel.stop_device_btn.isEnabled() is True


def test_stop_stays_available_as_the_emergency_exit(panel):
    """Also clears a white screen, which does not imply anything is playing."""
    assert panel.playing is False
    assert panel.stop_btn.isEnabled() is True
    panel._on_white_screen()
    assert panel.stop_btn.isEnabled() is True


# --- settings -----------------------------------------------------------


def test_settings_tab_names_the_file_actually_in_force(panel):
    assert panel.config_path == fpspy.config.active_config_path(None)
    assert panel.config_path.exists()


def test_active_config_path_is_the_bundled_default_without_a_user_config(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        fpspy.config, "user_config_file_path", lambda: tmp_path / "settings.toml"
    )
    assert fpspy.config.active_config_path(None).name == "default_settings.toml"


def test_active_config_path_prefers_an_explicit_path(tmp_path):
    explicit = tmp_path / "mine.toml"
    assert fpspy.config.active_config_path(explicit) == explicit


# --- presenter replies --------------------------------------------------


def test_total_frames_enables_stepping(panel, tmp_path):
    write_stimulus(tmp_path / "stim.h5")
    panel._refresh_file_list()
    panel.file_list.setCurrentRow(0)
    panel._on_load()
    assert panel.next_btn.isEnabled() is False  # pending: nothing loaded yet

    post_reply(panel, "total_frames", total=10)
    assert panel.total_frames == 10
    assert panel.current_frame == -1
    assert panel.next_btn.isEnabled() is True
    assert panel.prev_btn.isEnabled() is False  # nothing shown to step back from


def test_stepped_moves_the_frame_readout(panel, tmp_path):
    post_reply(panel, "total_frames", total=3)
    post_reply(panel, "stepped", frame=0)
    assert panel.current_frame == 0
    assert "0" in panel.frame_label.text()
    assert panel.prev_btn.isEnabled() is True


def test_next_is_disabled_on_the_last_frame(panel):
    post_reply(panel, "total_frames", total=2)
    post_reply(panel, "stepped", frame=1)
    assert panel.next_btn.isEnabled() is False


def test_play_finished_only_clears_once_every_window_reports(panel, tmp_path):
    write_stimulus(tmp_path / "stim.h5")
    panel._refresh_file_list()
    panel.file_list.setCurrentRow(0)
    panel._on_play()
    assert panel.playing is True

    post_reply(panel, "play_finished", sender_idx=1)
    assert panel.playing is True, "window 2 has not finished yet"
    post_reply(panel, "play_finished", sender_idx=2)
    assert panel.playing is False
    assert panel.play_btn.isEnabled() is True


def test_stop_leaves_a_loaded_stimulus_steppable(panel):
    """A stop resets the device, but the loaded stimulus stays in place."""
    post_reply(panel, "total_frames", total=5)
    post_reply(panel, "stepped", frame=2)
    panel._on_stop()
    post_reply(panel, "done", sender_idx=1)
    post_reply(panel, "done", sender_idx=2)
    assert panel.total_frames == 5
    assert panel.next_btn.isEnabled() is True


def test_a_dead_presenter_stops_polling_and_says_so(panel, monkeypatch):
    monkeypatch.setattr(panel.client, "dead_processes",
                        lambda: [panel.client.processes[0]])
    panel._poll()
    assert panel.timer.isActive() is False
    assert "exited" in panel.status_bar.currentMessage()


# --- commands sent ------------------------------------------------------


def test_play_sends_the_colour_settings(panel, tmp_path):
    write_stimulus(tmp_path / "stim.h5")
    panel._refresh_file_list()
    panel.file_list.setCurrentRow(0)
    panel.colours_edit.setText("led_610,led_560")
    panel.change_logic_spin.setValue(4)
    panel.loops_spin.setValue(3)
    panel._on_play()

    for q in panel.client.cmd_queues:
        (cmd,) = drain_commands(q)
        assert cmd.type == "play"
        assert cmd.kwargs["arduino_colours"] == "led_610,led_560"
        assert cmd.kwargs["change_logic"] == 4
        assert cmd.kwargs["loops"] == 3
        assert cmd.kwargs["stim_path"] == tmp_path / "stim.h5"


def test_empty_colour_field_sends_none(panel, tmp_path):
    """No colours means the LEDs are left exactly as they were."""
    write_stimulus(tmp_path / "stim.h5")
    panel._refresh_file_list()
    panel.file_list.setCurrentRow(0)
    panel._on_play()
    (cmd,) = drain_commands(panel.client.cmd_queues[0])
    assert cmd.kwargs["arduino_colours"] is None


def test_lazy_textures_checkbox_reaches_the_stim_config(panel, tmp_path):
    write_stimulus(tmp_path / "stim.h5")
    panel._refresh_file_list()
    panel.file_list.setCurrentRow(0)

    panel.lazy_textures_check.setChecked(True)
    panel._on_play()
    (cmd,) = drain_commands(panel.client.cmd_queues[0])
    assert json.loads(cmd.kwargs["stim_config"]) == {"lazy_textures": True}

    panel.lazy_textures_check.setChecked(False)
    panel._on_play()
    (cmd,) = drain_commands(panel.client.cmd_queues[0])
    assert json.loads(cmd.kwargs["stim_config"]) == {"lazy_textures": False}


def test_lazy_textures_defaults_off_like_the_cli(panel):
    """`fpspy play` defaults to eager; the panel must not differ silently."""
    assert panel.lazy_textures_check.isChecked() is False


def test_lazy_textures_is_disabled_for_non_hdf5_stimuli(panel, tmp_path):
    """The option only reaches TextureSequence, so it is meaningless there."""
    write_stimulus(tmp_path / "a.h5")
    (tmp_path / "b.py").touch()
    panel._refresh_file_list()
    rows = {panel.file_list.item(i).text(): i for i in range(panel.file_list.count())}

    panel.file_list.setCurrentRow(rows["a.h5"])
    assert panel.lazy_textures_check.isEnabled() is True
    panel.file_list.setCurrentRow(rows["b.py"])
    assert panel.lazy_textures_check.isEnabled() is False


def test_non_hdf5_stimulus_gets_no_stim_config(panel, tmp_path):
    (tmp_path / "bar.py").touch()
    panel._refresh_file_list()
    panel.file_list.setCurrentRow(0)
    panel._on_play()
    (cmd,) = drain_commands(panel.client.cmd_queues[0])
    assert cmd.kwargs["stim_config"] is None


def test_device_command_is_sent_to_window_one_only(panel):
    panel.command_box.setCurrentText("chirp")
    panel._on_send_command()
    (cmd,) = drain_commands(panel.client.cmd_queues[0])
    assert cmd.type == "device_cmd"
    assert cmd.args == ["chirp"]
    assert drain_commands(panel.client.cmd_queues[1]) == []


# --- error prevention ---------------------------------------------------


def test_command_picker_offers_only_real_firmware_commands(panel):
    offered = [panel.command_box.itemText(i) for i in range(panel.command_box.count())]
    assert offered == [cmd for cmd, _ in arduino_commands.COMMANDS]


def test_unknown_colour_is_warned_about_before_playing(panel):
    """A bad colour is dropped silently by the firmware, so warn beforehand."""
    panel.colours_edit.setText("led_610,purple")
    assert "purple" in panel.colours_warning.text()
    panel.colours_edit.setText("led_610,white")
    assert panel.colours_warning.text() == ""


def test_play_needs_a_selected_file(panel, tmp_path):
    assert panel.play_btn.isEnabled() is False
    write_stimulus(tmp_path / "stim.h5")
    panel._refresh_file_list()
    panel.file_list.setCurrentRow(0)
    assert panel.play_btn.isEnabled() is True


def test_file_list_shows_only_playable_suffixes(panel, tmp_path):
    write_stimulus(tmp_path / "a.h5")
    for name in ("b.py", "c.mp4", "notes.txt", "data.npz"):
        (tmp_path / name).touch()
    panel._refresh_file_list()
    listed = {panel.file_list.item(i).text() for i in range(panel.file_list.count())}
    assert listed == {"a.h5", "b.py", "c.mp4"}


def test_unreadable_h5_cannot_be_played(panel, tmp_path):
    """Playing it would kill a presenter process; refuse instead."""
    (tmp_path / "broken.h5").write_text("not an hdf5 file")
    panel._refresh_file_list()
    panel.file_list.setCurrentRow(0)
    assert "cannot read" in panel.file_info_label.text().lower()
    assert panel.play_btn.isEnabled() is False
    assert panel.load_btn.isEnabled() is False


# --- stimulus metadata --------------------------------------------------


def rows_as_dict(rows):
    return dict(rows)


class TestDescribeStimulus:
    """describe_stimulus() turns a preview_hdf5() dict into display rows.

    The two HDF5 versions report different keys, so the main risk is a field
    being silently dropped for one of them.
    """

    V1 = {
        "version": 1, "n_frames": 24000, "n_mask_repeats": 1,
        "n_total_frames": 24000, "n_triggers": 24000,
        "channel_mask.shape": None, "fps": 10.0, "duration": 2400.0,
        "height": 96, "width": 96, "channels": 3, "zoom": 8, "metadata": {},
    }
    V0 = {
        "version": 0, "n_frames": 300, "height": 500, "width": 500,
        "fps": 5, "duration": 60.0,
        "metadata": {"checkerboard_size": 50, "shuffle": False},
    }

    def test_v1_reports_the_headline_numbers(self):
        d = rows_as_dict(describe_stimulus(self.V1))
        assert d["Frames"] == "24,000"
        assert d["Frame rate"] == "10 Hz"
        assert d["Channels"] == "3"
        assert d["Triggers"] == "24,000"
        assert d["HDF5 format"] == "v1"

    def test_zoom_is_reported_as_the_on_screen_size(self):
        """96 px stored at zoom 8 is 768 px displayed — the number that matters."""
        d = rows_as_dict(describe_stimulus(self.V1))
        assert "96 × 96 px" in d["Frame size"]
        assert "768 × 768 px" in d["Frame size"]
        assert d["Zoom / box size"] == "8 px"

    def test_v0_metadata_is_shown_including_box_size(self):
        d = rows_as_dict(describe_stimulus(self.V0))
        assert d["Checkerboard size"] == "50"
        assert d["Shuffle"] == "no"
        assert d["Frames"] == "300"
        assert d["HDF5 format"] == "v0"

    def test_v0_omits_fields_it_does_not_have(self):
        """Rather than showing "Channels: None"."""
        d = rows_as_dict(describe_stimulus(self.V0))
        assert "Channels" not in d
        assert "Triggers" not in d
        assert "Zoom / box size" not in d

    def test_unrecognised_metadata_keys_are_still_shown(self):
        info = dict(self.V1, metadata={"stimulus_seed": 12345, "notes": "run 3"})
        d = rows_as_dict(describe_stimulus(info))
        assert d["Stimulus seed"] == "12345"
        assert d["Notes"] == "run 3"

    def test_mask_repeats_are_spelled_out(self):
        info = dict(self.V1, n_mask_repeats=4, n_total_frames=96000)
        d = rows_as_dict(describe_stimulus(info))
        assert d["Frames"] == "24,000 × 4 mask repeats = 96,000"

    def test_missing_fps_is_named_not_blank(self):
        """v1 allows per-frame durations, so there is no single rate."""
        d = rows_as_dict(describe_stimulus(dict(self.V1, fps=None)))
        assert "variable" in d["Frame rate"]

    def test_file_size_is_included_when_a_path_is_given(self, tmp_path):
        path = tmp_path / "s.h5"
        path.write_bytes(b"x" * 2_000_000)
        d = rows_as_dict(describe_stimulus(self.V1, path))
        assert d["File size"] == "2.0 MB"


class TestFormatting:
    @pytest.mark.parametrize(
        "seconds, expected",
        [(12.34, "12.3 s"), (60.0, "1:00 (60 s)"), (2400.0, "40:00 (2,400 s)"),
         (3661.0, "1:01:01 (3,661 s)")],
    )
    def test_duration(self, seconds, expected):
        assert _format_duration(seconds) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [(np.bool_(False), "no"), (np.int64(50), "50"), (np.float64(2.5), "2.5"),
         (b"noise", "noise"), (np.array([1, 2, 3]), "1, 2, 3"),
         (np.array(7), "7"), (True, "yes")],
    )
    def test_values_lose_their_numpy_decoration(self, value, expected):
        assert _format_value(value) == expected


class TestMetadataTable:
    def test_selecting_a_file_fills_the_table(self, panel, tmp_path):
        write_stimulus(tmp_path / "stim.h5")
        panel._refresh_file_list()
        panel.file_list.setCurrentRow(0)

        shown = {
            panel.metadata_table.item(r, 0).text():
                panel.metadata_table.item(r, 1).text()
            for r in range(panel.metadata_table.rowCount())
        }
        assert shown["Frames"] == "2"
        assert shown["Frame rate"] == "10 Hz"
        assert "File size" in shown

    def test_table_is_cleared_for_an_unreadable_file(self, panel, tmp_path):
        (tmp_path / "broken.h5").write_text("not an hdf5 file")
        panel._refresh_file_list()
        panel.file_list.setCurrentRow(0)
        assert panel.metadata_table.rowCount() == 0
        assert "cannot read" in panel.file_info_label.text().lower()

    def test_non_hdf5_says_why_there_is_nothing_to_show(self, panel, tmp_path):
        (tmp_path / "bar.py").touch()
        panel._refresh_file_list()
        panel.file_list.setCurrentRow(0)
        assert panel.metadata_table.rowCount() == 0
        assert "only readable for HDF5" in panel.file_info_label.text()

    def test_switching_files_replaces_rather_than_appends(self, panel, tmp_path):
        write_stimulus(tmp_path / "a.h5")
        write_stimulus(tmp_path / "b.h5")
        panel._refresh_file_list()
        panel.file_list.setCurrentRow(0)
        first = panel.metadata_table.rowCount()
        panel.file_list.setCurrentRow(1)
        assert panel.metadata_table.rowCount() == first
