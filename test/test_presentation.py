"""Tests for Presenter logic that does not need a GL window.

Presenter.__init__ opens a moderngl window, so these drive the unbound
methods against a stub carrying just the attributes each one touches. That
keeps the real code under test — communicate() here is the same function the
presenter process runs — without needing a display.
"""

import queue as std_queue

import pytest

import fpspy.fps_queue as fps_queue
from fpspy.presentation import PlayState, Presenter


class StubPresenter:
    """The attributes Presenter.communicate() touches on a stop, and no more."""

    communicate = Presenter.communicate
    _is_step_play = Presenter._is_step_play

    def __init__(self, play_state=None):
        self.queue = std_queue.Queue()
        self.play_state = play_state
        self.clear_rgba = (0.664, 0.664, 0.664, 1.0)
        self.active_clear_rgba = (1.0, 1.0, 1.0, 1.0)  # as if white_screen ran
        self.replies = []
        self.stopped = False

    def notify_stop(self):
        self.stopped = True

    def reply(self, kind, **payload):
        self.replies.append((kind, payload))

    def close_window(self):
        pass


def loaded_state(current_frame):
    """A PlayState as load() + step_next() would leave it."""
    return PlayState(
        prog=None, frame_idxs=[0, 1, 2], triggers=[0, 0, 0],
        current_frame=current_frame,
    )


def test_is_step_play_is_false_when_nothing_is_displayed():
    """run_empty() clears the window only while this is False."""
    assert StubPresenter(loaded_state(-1))._is_step_play() is False
    assert StubPresenter(loaded_state(0))._is_step_play() is True
    assert StubPresenter(None)._is_step_play() is False


def test_stop_during_step_play_releases_the_screen():
    """The regression: a stepped frame used to stay on screen after a stop.

    run_empty() skips its clear/swap while _is_step_play() is true, so leaving
    current_frame at the stepped frame meant the front buffer was never
    redrawn and the stimulus stayed visible.
    """
    presenter = StubPresenter(loaded_state(current_frame=2))
    fps_queue.put_onto(presenter.queue, "stop")

    assert presenter.communicate() is True
    assert presenter._is_step_play() is False, "window would never be cleared"


def test_stop_keeps_the_stimulus_loaded():
    """Stepping stays available after a stop, restarting from frame 0."""
    presenter = StubPresenter(loaded_state(current_frame=2))
    fps_queue.put_onto(presenter.queue, "stop")
    presenter.communicate()

    assert presenter.play_state is not None
    assert presenter.play_state.current_frame == -1


def test_stop_restores_the_configured_clear_colour():
    """A white_screen is undone by the next stop."""
    presenter = StubPresenter()
    fps_queue.put_onto(presenter.queue, "stop")
    presenter.communicate()

    assert presenter.active_clear_rgba == presenter.clear_rgba


def test_stop_notifies_and_answers():
    presenter = StubPresenter()
    fps_queue.put_onto(presenter.queue, "stop")
    presenter.communicate()

    assert presenter.stopped is True
    assert [kind for kind, _ in presenter.replies] == ["done"]


def test_stop_without_a_loaded_stimulus_is_harmless():
    """Stop is the emergency exit and is reachable with nothing loaded."""
    presenter = StubPresenter(play_state=None)
    fps_queue.put_onto(presenter.queue, "stop")

    assert presenter.communicate() is True
    assert presenter.play_state is None


def test_an_empty_queue_does_nothing():
    presenter = StubPresenter(loaded_state(current_frame=1))
    assert presenter.communicate() is None
    assert presenter.stopped is False
    assert presenter.play_state.current_frame == 1
