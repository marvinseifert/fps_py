"""The commands the ESP32 stimulator firmware understands.

Transcribed from `stimuli.md` in the firmware repo (esp32_stimulator), which
is the command table generated from `src/commands.cpp`. It is kept here, in
this repo, because the GUI needs it on machines that do not have the firmware
sources checked out.

Why a table at all: a mistyped command produces no output whatsoever. The
firmware's `unrecognized()` prints "Unknown command" only for the literal
command "b", so there is no general signal for a bad command and nothing to
build an error display on. The only workable approach is to stop bad commands
being sent, by offering this list instead of a free-text field.

Keep in sync by hand when the firmware's command table changes.
"""

# Not a registered command, and so absent from stimuli.md. The firmware reads
# this byte with a raw Serial.read() inside loop_interrupt() (utils.cpp:37-42),
# which runs *during* a protocol's wait loop and sets interrupt_flag; the
# protocol loops in stimuli.cpp check that flag and bail out. Bypassing the
# command parser is exactly why it is the only command that takes effect
# immediately — everything else waits for the protocol to finish.
#
# Sent while nothing is running, it falls through to the parser instead and
# comes back as "Unknown command".
INTERRUPT = "b"

# (command, what the firmware does with it)
COMMANDS: list[tuple[str, str]] = [
    ("f", "Play a white full field flash for a configured number of cycles."),
    ("ledpower", "Set the LED power (contrast) based on user input for channel and percentage."),
    ("chirp_660", "Play a frequency chirp on LED 660 (Red)."),
    ("chirp_610", "Play a frequency chirp on LED 610 (Orange)."),
    ("chirp_560", "Play a frequency chirp on LED 560 (Green-Yellow)."),
    ("chirp_530", "Play a frequency chirp on LED 535 (Green)."),
    ("chirp_500", "Play a frequency chirp on LED 500 (Cyan)."),
    ("chirp_460", "Play a frequency chirp on LED 460 (Blue)."),
    ("chirp_413", "Play a frequency chirp on LED 413 (Purple)."),
    ("chirp_365", "Play a frequency chirp on LED 365 (UV)."),
    ("chirp", "Play a frequency chirp on all active LEDs."),
    ("noise", "Play color noise at 30Hz on configured noise channels."),
    ("scf", "Play single chromatic full-field flashes for all LEDs sequentially."),
    ("wscf", "Play single chromatic full-field flashes for all LEDs sequentially with white."),
    ("scf_non_uv", "Play single chromatic full-field flashes for all LEDs except UV."),
    ("pcf", "Play pulse chromatic full-field flashes for all LEDs sequentially."),
    ("a", "Switch on all LEDs at max power for a specified interval."),
    ("off", "Switch off all LEDs."),
    ("forever", "Switch on custom LEDs continuously based on multiple serial arguments."),
    ("nflashes", "Play a flash at a given wavelength followed by one at the next lower wavelength."),
    ("csteps", "Play contrast steps (decreasing intensity by 10%) for all active LEDs."),
    ("csteps_660", "Play contrast steps for LED 660."),
    ("csteps_610", "Play contrast steps for LED 610."),
    ("csteps_560", "Play contrast steps for LED 560."),
    ("csteps_535", "Play contrast steps for LED 535."),
    ("csteps_500", "Play contrast steps for LED 500."),
    ("csteps_460", "Play contrast steps for LED 460."),
    ("csteps_413", "Play contrast steps for LED 413."),
    ("csteps_365", "Play contrast steps for LED 365."),
    ("contrast_luminance_on", "Play contrast vs luminance protocol (ON steps)."),
    ("contrast_luminance_off", "Play contrast vs luminance protocol (OFF steps)."),
    ("led_660", "Switch on LED 660."),
    ("led_610", "Switch on LED 610."),
    ("led_560", "Switch on LED 560."),
    ("led_535", "Switch on LED 535."),
    ("led_500", "Switch on LED 500."),
    ("led_460", "Switch on LED 460."),
    ("led_413", "Switch on LED 413 (420)."),
    ("led_365", "Switch on LED 365."),
    ("led_365_a_416", "Switch on LED 365 and LED 413 together."),
    ("led_460_a_560", "Switch on LED 460 and LED 560 together."),
    ("white", "Switch on all active LEDs."),
    ("O", "Switch off all LEDs."),
    ("T", "Send a fast trigger signal without console delay."),
    ("trigger_test", "Send a trigger signal with console output for testing."),
    ("t_s_on", "Enable the trigger loop state."),
    ("t_s_off", "Disable the trigger loop state."),
    (
        INTERRUPT,
        "Interrupt the running protocol. Takes effect immediately, unlike "
        "every other command.",
    ),
]

DESCRIPTIONS: dict[str, str] = dict(COMMANDS)

# The subset that only switches LEDs on or off. These are the ones that make
# sense in a colour schedule, where a command is sent between stimulus frames:
# anything that plays a protocol of its own would run against the stimulus.
COLOUR_COMMANDS: list[str] = [
    cmd for cmd, _ in COMMANDS if cmd.startswith("led_")
] + ["white", "O"]


def is_known(command: str) -> bool:
    """Whether the firmware has a handler for `command`."""
    return command in DESCRIPTIONS


def unknown_colours(colour_spec: str) -> list[str]:
    """The entries of a comma-separated colour spec the firmware would drop.

    Used to warn before a stimulus starts, since a bad colour is silently
    ignored mid-presentation and the LEDs simply never change.
    """
    entries = [c.strip() for c in colour_spec.split(",") if c.strip()]
    return [c for c in entries if not is_known(c)]
