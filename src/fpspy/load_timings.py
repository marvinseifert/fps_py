import numpy as np

d = np.load(r"/home/mawa/.local/state/fpspy/log/20260714T132647Z/frame_timings_win1.npz")
t, s = d["timings"], d["s_frames"]
budget_left = s[1:] - t[
    :, 3]  # slack before each next deadline
render_time = t[:, 2] - t[
    :, 1]  # lazy-path decode+upload cost
refresh_period = np.nanmedian(np.diff(t[:, 3]))  # true display refresh, from vsync quantization


