import numpy as np
import matplotlib.pyplot as plt
# %%
d = np.load(r"/home/mawa/.local/state/fpspy/log/20260720T145139Z/frame_timings_win1.npz")
t, s = d["timings"], d["s_frames"]
budget_left = s[1:] - t[
    :, 3]  # slack before each next deadline
render_time = t[:, 2] - t[
    :, 1]  # lazy-path decode+upload cost
refresh_period = np.nanmedian(np.diff(t[:, 3]))  # true display refresh, from vsync quantization


# %%
fig, ax = plt.subplots(figsize=(20, 10))
ax.hist(np.diff(t[:, 3]), bins=np.arange(-1, 1, 0.05))
fig.show()