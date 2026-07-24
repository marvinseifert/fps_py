"""One owner for the presenter queues.

Everything a front end needs to talk to the presenter processes lives here:
the command queues (one per window), the reply queues (one per window), and
the Arduino telemetry queue (one, written only by the lead presenter).

Two ways to read, because there are two kinds of caller:

* `drain_replies` / `drain_arduino` never block. A Qt event loop cannot afford
  to wait on a queue, so the GUI polls these from a timer and reacts to
  whatever has arrived.
* `wait_for_all` blocks until every presenter has answered, for scripted
  callers that run a sequence step by step.

The Arduino queue is kept separate from the reply queues throughout: its
messages are unsolicited device events, not answers, so mixing them into a
reply wait would desynchronise the count.
"""

import logging
import multiprocessing as mp
import queue as std_queue

import fpspy.fps_queue
import fpspy.presentation

_logger = logging.getLogger(__name__)


class PresenterClient:
    """The main process's handle on the running presenter processes."""

    def __init__(self, processes, cmd_queues, reply_queues, arduino_queue):
        self.processes = processes
        self.cmd_queues = cmd_queues
        self.reply_queues = reply_queues
        self.arduino_queue = arduino_queue

    @classmethod
    def start(cls, config, out_dir, delay, enable_triggers, log_level="WARNING"):
        """Start the presenter processes and return a client for them."""
        processes, cmd_queues, reply_queues, arduino_queue = (
            fpspy.presentation.start_presenter_processes(
                config, out_dir, delay, enable_triggers, log_level
            )
        )
        return cls(processes, cmd_queues, reply_queues, arduino_queue)

    @property
    def n_windows(self) -> int:
        return len(self.cmd_queues)

    # --- sending -------------------------------------------------------

    def send_all(self, cmd_type: str, *args, **kwargs):
        """Send one command to every presenter."""
        for q in self.cmd_queues:
            fpspy.fps_queue.put_onto(q, cmd_type, *args, **kwargs)

    def send_to(self, window_idx: int, cmd_type: str, *args, **kwargs):
        """Send one command to a single presenter. `window_idx` is 0-based."""
        fpspy.fps_queue.put_onto(
            self.cmd_queues[window_idx], cmd_type, *args, **kwargs
        )

    def send_device_command(self, command: str):
        """Forward a firmware command to the Arduino.

        Only window 1 owns the serial port, so this goes to that presenter
        alone. It is fire-and-forget: the firmware has no acknowledgement, and
        the effect (if any) turns up on the telemetry queue.
        """
        self.send_to(0, "device_cmd", command)

    # --- receiving -----------------------------------------------------

    def drain_replies(self) -> list:
        """Take every reply waiting on any presenter queue. Never blocks."""
        return [r for q in self.reply_queues for r in _drain(q)]

    def drain_arduino(self) -> list:
        """Take every device event waiting on the Arduino queue. Never blocks."""
        return _drain(self.arduino_queue)

    def wait_for_all(self, kind: str, what: str, timeout: float = 1.0) -> list:
        """Block until every presenter has sent a reply of `kind`.

        Replies of other kinds arriving in the meantime are discarded, so this
        is for scripted callers only; a GUI should drain instead. Raises
        RuntimeError if a presenter dies while waiting, which is the only way
        out other than success.
        """
        collected = []
        for q in self.reply_queues:
            while True:
                try:
                    reply = fpspy.fps_queue.get_reply(q, timeout=timeout)
                except std_queue.Empty:
                    dead = [p for p in self.processes if not p.is_alive()]
                    if dead:
                        raise RuntimeError(
                            f"Presenter process died (exit code {dead[0].exitcode}) "
                            f"while {what}."
                        )
                    continue
                if reply.kind == kind:
                    collected.append(reply)
                    break
        return collected

    # --- lifecycle -----------------------------------------------------

    def dead_processes(self) -> list:
        """The presenter processes that have exited."""
        return [p for p in self.processes if not p.is_alive()]

    def shutdown(self, timeout: float = 2.0):
        """Stop the presentation, close the windows, and reap the processes."""
        try:
            self.send_all("stop")
            self.send_all("destroy")
        except (ValueError, OSError):
            # Queues already closed, e.g. because the processes are gone.
            _logger.debug("Could not send shutdown commands; queues are closed.")
        for p in self.processes:
            p.join(timeout=timeout)
            if p.is_alive():
                _logger.warning(f"Presenter {p.pid} did not exit; terminating.")
                p.terminate()


def _drain(q: mp.Queue) -> list:
    """Take everything currently on `q`, oldest first."""
    items = []
    while True:
        try:
            item = q.get_nowait()
        except std_queue.Empty:
            break
        if not isinstance(item, fpspy.fps_queue.Reply):
            _logger.warning(f"Discarding non-Reply item from queue: {item!r}")
            continue
        items.append(item)
    return items
