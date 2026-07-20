import time
from typing import Optional


def _wait_or_skip(target_time, next_frame_time: Optional[float], fps, logger):
    """Semi-busy-wait for the frame time, or possibly skip to next frame."""
    max_busy_wait_sec = 0.002
    now = time.perf_counter()
    remaining = target_time - now
    if remaining <= 0:
        logger.warning(f"{target_time=} sec already passed, {now=} sec.")
        has_next_frame = next_frame_time is not None
        if not has_next_frame:
            return False
        half_period = 0.5 / fps
        # If displaying the current frame would cause us to miss the next frame by more
        # than half a period, then skip to the next frame.
        if now + half_period >= next_frame_time:
            logger.warning("Skipping to next frame.")
            return True
        return False
    if remaining > max_busy_wait_sec:
        time.sleep(remaining - max_busy_wait_sec)
    while time.perf_counter() < target_time:
        pass
    return False