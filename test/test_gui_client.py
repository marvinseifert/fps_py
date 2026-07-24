"""Tests for the shared presenter queue client.

No presenter processes are started: the queues are plain queue.Queue, which
has the same get_nowait/get interface the client uses.
"""

import queue as std_queue

import pytest

import fpspy.fps_queue as fps_queue
from fpspy.gui_client import PresenterClient


class FakeProcess:
    """Stands in for mp.Process, so a "dead presenter" can be arranged."""

    def __init__(self, alive=True, pid=1, exitcode=None):
        self._alive = alive
        self.pid = pid
        self.exitcode = exitcode
        self.terminated = False
        self.joined = False

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        self.joined = True

    def terminate(self):
        self.terminated = True
        self._alive = False


@pytest.fixture
def client():
    processes = [FakeProcess(), FakeProcess(pid=2)]
    cmd_queues = [std_queue.Queue(), std_queue.Queue()]
    reply_queues = [std_queue.Queue(), std_queue.Queue()]
    return PresenterClient(processes, cmd_queues, reply_queues, std_queue.Queue())


def drain_commands(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_send_all_reaches_every_presenter(client):
    client.send_all("play", stim_path="a.h5", loops=2)
    for q in client.cmd_queues:
        (cmd,) = drain_commands(q)
        assert cmd.type == "play"
        assert cmd.kwargs == {"stim_path": "a.h5", "loops": 2}


def test_device_command_goes_only_to_window_one(client):
    """Window 1 owns the serial port; sending to the others would do nothing."""
    client.send_device_command("led_610")
    (cmd,) = drain_commands(client.cmd_queues[0])
    assert cmd.type == "device_cmd"
    assert cmd.args == ["led_610"]
    assert drain_commands(client.cmd_queues[1]) == []


def test_drain_replies_is_empty_when_nothing_waiting(client):
    assert client.drain_replies() == []
    assert client.drain_arduino() == []


def test_drain_replies_collects_from_all_queues(client):
    fps_queue.put_reply(client.reply_queues[0], "total_frames", 1, total=10)
    fps_queue.put_reply(client.reply_queues[1], "total_frames", 2, total=10)
    kinds = [(r.kind, r.sender_idx) for r in client.drain_replies()]
    assert sorted(kinds) == [("total_frames", 1), ("total_frames", 2)]
    # Draining consumes: a second call must not repeat the same replies.
    assert client.drain_replies() == []


def test_drain_skips_non_reply_items(client):
    """A stray bare value must not stop the real replies being delivered."""
    client.reply_queues[0].put("not a Reply")
    fps_queue.put_reply(client.reply_queues[0], "done", 1)
    (reply,) = client.drain_replies()
    assert reply.kind == "done"


def test_arduino_events_do_not_appear_among_replies(client):
    """Device events are unsolicited, so they must stay off the reply path."""
    fps_queue.put_reply(client.arduino_queue, "arduino_trigger", 1, text="Trigger")
    assert client.drain_replies() == []
    (event,) = client.drain_arduino()
    assert event.kind == "arduino_trigger"


def test_wait_for_all_ignores_other_kinds(client):
    for idx, q in enumerate(client.reply_queues, start=1):
        fps_queue.put_reply(q, "done", idx)
        fps_queue.put_reply(q, "play_finished", idx)
    replies = client.wait_for_all("play_finished", "playing", timeout=0.1)
    assert [r.sender_idx for r in replies] == [1, 2]


def test_wait_for_all_raises_when_a_presenter_died(client):
    client.processes[0]._alive = False
    client.processes[0].exitcode = 1
    with pytest.raises(RuntimeError, match="Presenter process died"):
        client.wait_for_all("play_finished", "playing", timeout=0.01)


def test_dead_processes_lists_only_the_exited_ones(client):
    client.processes[1]._alive = False
    assert client.dead_processes() == [client.processes[1]]


def test_shutdown_sends_stop_then_destroy_and_reaps(client):
    client.shutdown(timeout=0)
    for q in client.cmd_queues:
        assert [c.type for c in drain_commands(q)] == ["stop", "destroy"]
    assert all(p.joined for p in client.processes)


def test_shutdown_terminates_a_presenter_that_will_not_exit(client):
    client.shutdown(timeout=0)
    # FakeProcess.join() does not exit the process, so both stay alive and
    # must be terminated rather than leaked.
    assert all(p.terminated for p in client.processes)
