# GUI Workplan

Working plan for the Qt6 GUI rewrite and the backend work it depends on.
Companion to `GUI_manifesto.md` (what the GUI should do) and
`gui_design_principles.txt` (how it should look and behave).

Status as of commit `d75a1fd`.

---

## 1. Decisions taken

| # | Decision | Rationale |
|---|---|---|
| 1 | GUI is an **instrument panel** first, with a generation tab and a log tab | `GUI_manifesto.md` |
| 2 | **PyQt6**, replacing tkinter | Already a dependency (`pyproject.toml:33`); `examples/gui_cal_3brain.py` is already PyQt6 and states tkinter "doesn't work well on Wayland" |
| 3 | Arduino stays **owned by presenter process 1**; state reaches the GUI via a **telemetry queue** (option 1 of 3) | Serial port must have exactly one owner; avoids perturbing trigger timing, which runs on the render thread right after buffer swap |
| 4 | Per-presenter **reply queues** replace the shared status queue | Removes cross-presenter message mix-up; done in `d75a1fd` |
| 5 | `change_logic == 1` means **one fixed colour**, sent once before the loop | Matches old `play.py` behaviour |
| 6 | Colour schedule is indexed **by pattern index**, not playback position | Same pattern always gets the same LED; needs a bounds guard the original lacked |
| 7 | Colour-logic GUI control is just a **string input**, wired up later | User decision — backend first |
| 8 | The new GUI **does not absorb** `gui_cal_3brain.py` — it absorbs some of its *functions* | The calibration GUI stays a standalone tool; frame-stepping and the queue-client pattern are reused, not the application |
| 9 | `ArduinoController` stays on the **`Arduino`** class (via `create_arduino`) | User decision; `Arduino2` is not the target |
| 10 | **No acks, no CRC, no length validation.** Bare `Arduino.send()` only | The serial line is reliable until it is crowded; verification traffic adds crowding without adding reliability. The bare `Arduino` implementation is known to work in practice |
| 11 | **Drop the label/marker feature.** What was actually needed is already built (§3c) | The ESP32 has no command that accepts text; the real requirement is sending existing `commands.cpp` commands and arming the trigger cycle, both of which already work |

### Still open

- Nothing blocking. Remaining work is §3a (telemetry) and §3b (colour logic).

---

## 2. Done — committed in `d75a1fd`

Typed, per-sender queue transport.

- `fps_queue.py` — added `Reply(kind, sender_idx, payload)` mirroring `Command`, with
  `put_reply` / `get_reply`. `get_reply` validates type, so a stray bare value raises
  instead of being silently mishandled.
- `presentation.py` — `start_presenter_processes` now returns
  `(processes, cmd_queues, reply_queues, arduino_queue)`: one reply queue **per
  presenter**, plus one Arduino queue. `Presenter.reply()` stamps `process_idx` on
  every message. Reply kinds: `total_frames`, `stepped`, `no_stimulus`,
  `play_finished`, `done`.
- `arduino.py` — `ArduinoController` takes `sender_idx`, emits `arduino_done` on its
  own queue instead of `"done"` on the shared one.
- `batch_3brain.py` — `_wait_for_status` → `_wait_for_reply`: waits for one reply of a
  given kind *from each queue*, instead of counting N messages on a shared queue.
- `examples/main_3brain.py` — updated both call sites; dropped the dummy `arduino_queue`.

Untouched by design: `gui.py`, `gui_cal_3brain.py`, `play_3brain.py`
(`play_3brain.start_presenter_processes` is used only by the calibration GUI).

---

## 3. Backend work remaining

### 3a. Arduino telemetry stream  → unblocks the instrument panel

The panel cannot see the Arduino directly. Presenter 1 must forward device events.

- [ ] Add `Arduino.read_lines()` (and a `DummyArduino` stub). **Required**: today's
      `read()` (`arduino.py:75-94`) returns only the *last* buffered line and then
      discards the input buffer, so events are lost when two arrive together.
      Needs an `_rx_buffer` for trailing partial lines.
- [ ] Continuous reader thread in `ArduinoController`, emitting one event per line
      onto the Arduino queue. Subsumes the current on-demand `_monitor_status`.
- [ ] Map firmware lines to event kinds (see §5).
- [ ] Update `on_stop` / `close()`, which currently call `_stop_monitor()`.

**Watch:** `read()` and `send()` share an `RLock`, so a continuous reader contends
with `send_trigger()` on the render thread. Poll at a modest interval (~20 ms), keep
the lock hold short, and measure before trusting it.

### 3b. Colour change logic  → restores a feature that currently has no backend

The old tkinter GUI passed `arduino_colours` and `change_logic` into `play`
(`gui.py:357-363`), but no current presenter accepts them —
`presentation.py:604` is `play(self, stim_path, stim_config, loops, t0, speed=None,
close_after=False)`. The parameters go nowhere. Original implementation is in the
deleted `play.py` (`git show 5628889^:src/fpspy/play.py`, lines 229-251 and 419-437).

- [ ] Port `process_arduino_colours(colours, change_logic, n_frames)`:
      `split(",")` → `np.repeat(colours, change_logic)` → tile to cover `n_frames`.
- [ ] `register_on_colour` / `notify_colour` on `Presenter`, mirroring
      `register_on_trigger` (`presentation.py:398`).
- [ ] `ArduinoController.send_colour`; register it in `present_live` next to
      `register_on_trigger` (`presentation.py:104`). Only window 1 has an Arduino, so
      the callback is `None` elsewhere — no special casing needed.
- [ ] Fire from `shader_loop` beside `notify_trigger()` (`presentation.py:679`):
      - `change_logic == 1` → send `colours[0]` once **before** the loop (decision 5)
      - `change_logic > 1` → `if pattern_index % change_logic == 0: send(colours[pattern_index])`,
        **with a bounds guard** (decision 6)
- [ ] Extend `play()` to accept `arduino_colours=None, change_logic=1`.

### 3c. Remove the label/marker path — fixes a silent hang

**Resolution (decisions 10, 11): the feature is deleted, not implemented.** What the
rig actually needs already exists:

- **Send any `commands.cpp` command** — the `device_cmd` type routes to
  `ArduinoController.handle_command` (`presentation.py:107`), which is a bare
  `arduino.send(command)`. The GUI just sends `device_cmd` with a string, with
  `monitor=False` so nothing blocks.
- **Arm the fast trigger cycle before a stimulus** — `play()` calls
  `notify_play_start()` (`presentation.py:627`) *before* `shader_loop`, firing
  `on_play_start` → `send("t_s_on")` + `flush()` (`arduino.py:179-182`).
  With `presentation_delay = 4` (`resources/default_settings.toml:28`) and the
  firmware's ~1 s idle poll, `t_s_on` lands ~3 s before the first frame.
  **Watch:** a `presentation_delay` below ~1-2 s could let the first trigger fire
  before `t_s_on` is processed.

Why the text feature cannot work as written: `esp32_stimulator` registers 46 commands
(`src/commands.cpp:13-61`) and **none accepts a payload**. `Arduino2`'s `send_text`
targets different firmware (its docstring points at `code/arduino/`). A bare
`send("M<label>")` to the ESP32 hits `unrecognized()` and is dropped silently.

**Work:**

- [ ] Remove `_send_marker`, `_check_label`, `START_PREFIX`/`END_PREFIX`,
      `start_msg`/`end_msg`, and the two call sites at `batch_3brain.py:148` and
      `:164`. Drop `label` from the playlist item if nothing else uses it.
- [ ] Consider a `case _` fallback in `communicate()` that logs an unhandled command
      type, so a future missing case surfaces as a warning instead of a hang.

**The hang being removed.** A playlist run with `enable_triggers=True` and any
labelled item hangs indefinitely — windows open and idle, no error output, Ctrl-C the
only exit. `_send_marker` puts a `"message"` command on `cmd_queues[0]` and blocks in
`_wait_for_reply`; `communicate()` has eight cases (lines 461-501) and **no `case _`
fallback**, so `"message"` matches nothing and no reply is ever sent.
`_wait_for_reply`'s only escape is a dead presenter, and the presenters are healthy.
Pre-existing: introduced when `batch_3brain.py:140` was pointed at `presentation.py`
(cf. the commented-out `#import fpspy.play_3brain` at line 51).

**Symptom.** A playlist run with `enable_triggers=True` and any item carrying a
`label` hangs indefinitely, windows open and idle, with no error output. Ctrl-C is
the only exit.

**Cause.** `batch_3brain.py:148` calls `_send_marker`, which puts a `"message"`
command on `cmd_queues[0]` and then blocks in `_wait_for_reply` for a `message_sent`
reply. But `presentation.py`'s `communicate()` has eight cases
(`white_screen`, `device_cmd`, `load`, `step_next`, `step_prev`, `play`, `stop`,
`destroy`, lines 461-501) and **no `case _` fallback** — `"message"` matches nothing,
so the command is silently dropped and no reply is ever sent. `_wait_for_reply`'s
only escape is a dead presenter, and the presenters are perfectly healthy.

Pre-existing: introduced when `batch_3brain.py:140` was pointed at `presentation.py`
(cf. the commented-out `#import fpspy.play_3brain` at line 51). `play_3brain.py` has
the `case "message"` at line 648 that `presentation.py` never received.

---

## 4. GUI build

Only start once §3a and §3b land — the panel has nothing to display before that.

`gui_cal_3brain.py` remains a **standalone tool** (decision 8). The new GUI reuses
some of its functions — frame-stepping, the worker/threading pattern, the queue
client — but does not replace the application.

- [ ] Shared queue-client module: one place that owns `cmd_queues`, `reply_queues`,
      and the Arduino queue. `gui_cal_3brain.py` currently duplicates
      `send_to_all` / `wait_for_responses` twice (lines 131/145 and 468/473). Extract
      it so both the calibration tool and the new GUI use one implementation, rather
      than adding a third copy.
- [ ] **Instrument tab** — Arduino connection + last-known state, command entry
      (build a picker from the `stimuli.md` table rather than free text), stimulus
      selection, play/stop, folder change, per-window presenter state.
- [ ] Stimulus preview + step-through. Transport already exists: `load` →
      `total_frames`, `step_next`/`step_prev` → `stepped`.
- [ ] Colour-logic string input (decision 7).
- [ ] **Generation tab** — noise stimuli, moving bar; entry points in
      `create_noise.py` (`generate_and_store_3d_array`,
      `generate_and_store_3d_array_multicolour`, `multicolor_checkerboard`, …).
      Needs a registry so new stimulus templates can be added without touching the GUI.
- [ ] **Log tab** — reads `dropped_frames_win{idx}.csv` (`presentation.py:724-735`)
      and `frame_timings_win{idx}.npz` with `timings` + `s_frames`
      (`presentation.py:694`), one pair per window per run.

---

## 5. ESP32 facts the panel must respect

Firmware at `/home/mawa/CLionProjects/esp32_stimulator`. Command table: `stimuli.md`.

Everything the firmware ever sends back:

| Line | Source | Event kind |
|---|---|---|
| `Stimulator ready` | `main.cpp:12` | `ready` — boot / liveness |
| `finished` | `main.cpp:23` | `finished` |
| `Trigger` | `trigger.cpp:11` | `trigger` |
| `Trigger_test` | `trigger.cpp:29` | `trigger_test` |
| `Unknown command` | `commands.cpp:67` | **not an error** — means `b` found nothing to interrupt; see below |
| `Invalid LED channel` | `stimuli.cpp:121` | `error` |
| `Changing LED: <n>`, `Setting power to: <f>` | `utils.cpp:55-65` | `info` |
| `micros()`, protocol names, contrast factor | `stimuli.cpp` | `info` |

Four hard constraints:

1. **No status-query command.** The firmware only emits unsolicited lines — the panel
   cannot poll. It can only show *last known state* inferred since connection, and the
   UI should say so rather than imply live truth.
2. **~1 Hz command latency when idle.** `main.cpp:19-26`: while `trigger_state == false`,
   `loop()` ends in `delay(1000)`. Commands sent from an idle panel may take up to a
   second. Design principle 2 (visibility & feedback) therefore needs a pending state,
   not instant confirmation.

3. **Bad commands vanish silently.** `unrecognized()` (`commands.cpp:65-70`) prints
   `Unknown command` **only** when the command is literally `"b"`; every other
   unrecognized command is dropped with no output at all. So there is **no** general
   signal for a mistyped command, and an error display cannot be built on this.
   Validate command strings against the `stimuli.md` table on the GUI side instead
   (design principle 7, error *prevention* rather than recovery).

4. **`b` is an interrupt byte, not a command — and it is the only way to stop a running
   protocol.** It is absent from `stimuli.md` because it is never registered with
   `sCmd.addCommand`. Instead `loop_interrupt()` (`utils.cpp:37-42`) reads it with a raw
   `Serial.read()` from *inside* a protocol's wait loop (`wait_break`, `utils.cpp:25-35`)
   and sets `interrupt_flag`, which the protocol loops check and bail out on
   (`stimuli.cpp:237, 253, 271, 305, 329, 350`).

   Two consequences the panel depends on:

   - **`b` is the one command exempt from constraint 2.** Bypassing the parser is
     exactly why it takes effect at once, while everything else waits for the current
     protocol to end. A device stop button must send `b`. `O` (`commands.cpp:56` →
     `all_off`) only switches LEDs off, and only gets parsed *between* protocols, so it
     cannot abort anything.
   - **`Unknown command` means "`b` arrived with nothing running to interrupt".** It
     falls through to the parser only when no protocol was in a wait loop to consume it.
     `on_stop` (`arduino.py`) sends `b` on every stop, so an idle stop prints it every
     time. Report it as explained-and-harmless, not as a fault.

   Superseded: earlier revisions of this document described `b` as "not a registered
   command" whose only effect was the `Unknown command` line, and stated that a running
   protocol could not be interrupted. Both were wrong.

Also: `_monitor_status` currently keeps only `Trigger` and `finished` and discards the
rest, so `Invalid LED channel` is thrown away today. §3a recovers it.