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


def put(queue, cmd_type: str, *args, **kwargs):
    command = Command(type=cmd_type, args=list(args), kwargs=kwargs)
    queue.put(command)


def get(queue):
    cmd = queue.get()
    if not isinstance(cmd, Command):
        raise ValueError(f"Received invalid command from queue. Got: {cmd}")
    return cmd




