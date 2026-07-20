def export(prog: stim.StimProgram, config):
    n_windows = len(config["windows"])
    win_outs = []
    for idx in range(n_windows):
        _logger.info(f"Exporting for window {idx + 1}/{n_windows}")
        # Use ArrayRenderer for offscreen GPU rendering
        renderer = ArrayRenderer(process_idx=1 + idx, config=config)
        frames, frame_times, triggers = renderer.render(prog)
        win_outs.append((frames, frame_times, triggers))

    # Get channel mapping, from array to windows.
    src_to_out = {}
    for w_idx, w in enumerate(config["windows"]):
        for c_idx, ch in enumerate(w["channels"]):
            src_to_out[ch] = [w_idx, c_idx]
    max_ch = max(src_to_out.keys())
    stim_shapes = [r[0].shape[0:3] for r in win_outs]
    assert all(
        s == stim_shapes[0] for s in stim_shapes
    ), "All windows must have the same (f, h, w) stimulus shape."
    f, h, w = stim_shapes[0]
    # Create the output stimulus array.
    frames = np.zeros((f, h, w, max_ch + 1), dtype=np.uint8)
    for ch in range(max_ch + 1):
        if ch in src_to_out:
            w_idx, c_idx = src_to_out[ch]
            frames[:, :, :, ch] = win_outs[w_idx][0][:, :, :, c_idx]
    # We need triggers and frame times from only the first window.
    frame_times = win_outs[0][1]
    frame_durs = np.diff(frame_times)
    assert len(frame_durs) == f, f"{len(frame_durs)=}, {f=}"
    triggers = win_outs[0][2]
    out_stim = fpspy.stim.StimArray(
        frames,
        frame_durations=frame_durs,
        zoom=1,
        triggers=triggers,
        label="exported_stimulus",
    )
    return out_stim