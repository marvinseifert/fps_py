# Call flow: `_play()` → end of stimulus presentation

Chronological function-call trace for `examples/main_3brain.py:_play()`, as traced
for an `.h5` stimulus (the `TextureSequence` path). Alternative branches for `.py`
and movie stimuli are noted inline. Line numbers refer to the state of the code at
the time of writing (2026-07-14).

```
MAIN PROCESS                                          PRESENTER PROCESS (one per window)
════════════════════════════════════════════         ═══════════════════════════════════════════════

_play()                        main_3brain.py:40
│
├─ _logging.setup_main_logging()   _logging.py:41
├─ stim_path.exists()                    [check]
├─ fpspy.config.load_config()      config.py:195
│    ├─ _load_default_config()
│    ├─ _load_user_config() / _load_config_from_path()
│    ├─ _deep_merge()
│    ├─ _apply_window_defaults()
│    └─ _resolve_paths()
├─ config.get_presentation_delay() config.py:272   (only if delay is None)
├─ config.create_outdir()          config.py:102   (only if out_dir is None)
├─ _preview_h5_stim()          main_3brain.py:29   (.h5 only)
│    └─ StimArray.preview_hdf5()    stim.py:830
├─ json.dumps({"lazy_textures": ...})
├─ t0 = time.perf_counter()
│
├─ presentation.start_presenter_processes()
│                              presentation.py:98
│    └─ mp.Process(target=present_live).start() ──▶  present_live()          presentation.py:32
│       (× n_windows)                                │
│                                                    ├─ Presenter.__init__()  presentation.py:200
│                                                    │    ├─ setup_logging()             :349
│                                                    │    │    └─ _logging.enable_file_logging()
│                                                    │    └─ setup_window()              :315
│                                                    │         ├─ moderngl_window.create_window_from_settings()
│                                                    │         ├─ window.init_mgl_context()
│                                                    │         └─ window.set_default_viewport()
│                                                    ├─ arduino.Arduino()      arduino.py:17
│                                                    │    (window 1 only, if enable_triggers)
│                                                    ├─ presenter.register_on_trigger(arduino.send_trigger)
│                                                    │
│                                                    └─ presenter.run_empty() presentation.py:383
│                                                         loop: ctx.clear / swap_buffers
│                                                               communicate()          :398
├─ fps_queue.put_onto(q, "play", ...)                           │
│                               fps_queue.py:16  ─────────────▶ │  (queue delivers Command)
│                                                               │
├─ p.join()  ◀── blocks ──┐                          communicate() matches "play"
│                         │                          └─ Presenter.play()      presentation.py:535
│                         │                               │
│                         │                               ├─ _load()                    :443
│                         │                               │    ├─ stim.create_program()  stim.py:1317
│                         │                               │    │    ├─ StimArray.read_hdf5()   stim.py:791  (lazy, v1)
│                         │                               │    │    └─ TextureSequence(stim_arr, lazy_textures)
│                         │                               │    │      [.py file → from_script() → to_program();
│                         │                               │    │       movie → MoviePlayer]
│                         │                               │    ├─ prog.setup(ctx, w, h, channels,
│                         │                               │    │            mirror, rotation, win_id)
│                         │                               │    │             TextureSequence.setup, stim.py:1348
│                         │                               │    │    ├─ stim_arr.with_channels()
│                         │                               │    │    ├─ eager: masked_frames() → ctx.texture()/frame
│                         │                               │    │    ├─ _compile_program()   (GLSL vert+frag)
│                         │                               │    │    ├─ create_centered_quad()  stim.py:138
│                         │                               │    │    └─ returns (frame_times, triggers)
│                         │                               │    ├─ s_frames = s_frames*speed + t0
│                         │                               │    ├─ stim.loop()               stim.py:60
│                         │                               │    └─ stim.decompress_triggers() stim.py:102
│                         │                               │
│                         │                               ├─ stim.delay(s_frames, delay)  stim.py:122
│                         │                               │    (raises if start time already passed)
│                         │                               │
│                         │                               ├─ shader_loop()               presentation.py:563
│                         │                               │    ├─ probe_default_fbo_srgb()  moderngl_helpers.py:4
│                         │                               │    └─ FOR EACH frame i:  ◀──────────┐
│                         │                               │         ├─ communicate()  (early    │
│                         │                               │         │   exit if "stop" arrives) │
│                         │                               │         ├─ _wait_or_skip()          │
│                         │                               │         │     frame_handling.py:5   │
│                         │                               │         ├─ ctx.clear(0,0,0)         │
│                         │                               │         ├─ prog.render(ctx,         │
│                         │                               │         │    frame_idxs[i], i)      │
│                         │                               │         │    TextureSequence.render │
│                         │                               │         │    stim.py:1420           │
│                         │                               │         ├─ window.swap_buffers()    │
│                         │                               │         └─ if triggers[i]:          │
│                         │                               │              notify_trigger() ──────┤
│                         │                               │              └─ arduino.send_trigger()
│                         │                               │                 (window 1 only) ────┘
│                         │                               │
│                         │                               ├─ record_dropped_frames()  presentation.py:601
│                         │                               ├─ prog.cleanup()   TextureSequence.cleanup,
│                         │                               │                   stim.py:1448
│                         │                               │    └─ stim_arr.close()  (releases HDF5 file)
│                         │                               └─ return close_after=True
│                         │                                        │
│                         │                          back in communicate():  do_stop=True
│                         │                               ├─ notify_stop()
│                         │                               └─ status_queue.put("done")
│                         │                                        │
│                         │                          back in run_empty(): resumes idle loop
│                         └────────────────────────  (until window.is_closing → close_window())
│
└─ _logger.info("Stimulus playback completed.")
```

## Notes

1. **The presentation itself never runs in the main process.** `_play` ends at
   `p.join()`; everything from `Presenter.__init__` down happens in the
   `mp.Process` children. Debugger breakpoints in that code require attaching to
   subprocesses (e.g. PyCharm's "attach to subprocess automatically" option).
2. **End-of-presentation ≠ process exit.** After `play()` returns
   `close_after=True`, `communicate()` only sends `"done"` on the status queue —
   nothing in the `"play"` path calls `close_window()` (only a `"destroy"`
   command does). The presenter returns to the `run_empty()` idle loop, so
   `p.join()` in `_play` stays blocked until the presenter windows are closed by
   other means.