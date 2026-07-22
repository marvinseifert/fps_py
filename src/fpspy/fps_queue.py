from typing import Literal, List, Any
import dataclasses

# Could make Command.type typed, but not worth it.
# RenderCmd = Literal["play", "white_screen", "stop", "destroy"]
# ArduinoCmd = Literal[""] 


@dataclasses.dataclass
class Command:
    type: str
    args: List[Any]
    kwargs: dict


@dataclasses.dataclass
class Reply:
    """A presenter's answer to a Command, or an async device event.

    Every reply carries its sender so a log line or a merged reader can say
    which window (or device) it came from, even though replies travel on
    per-sender queues and identity is already implied by the queue.
    """

    kind: str
    sender_idx: int
    payload: dict


def put_onto(queue, cmd_type: str, *args, **kwargs):
    command = Command(type=cmd_type, args=list(args), kwargs=kwargs)
    queue.put(command)


def get_from(queue):
    cmd = queue.get()
    if not isinstance(cmd, Command):
        raise ValueError(f"Received invalid command from queue. Got: {cmd}")
    return cmd


def put_reply(queue, kind: str, sender_idx: int, **payload):
    queue.put(Reply(kind=kind, sender_idx=sender_idx, payload=payload))


def get_reply(queue, timeout=None):
    """Read one Reply. Raises queue.Empty on timeout, like Queue.get."""
    reply = queue.get(timeout=timeout)
    if not isinstance(reply, Reply):
        raise ValueError(f"Received invalid reply from queue. Got: {reply}")
    return reply




