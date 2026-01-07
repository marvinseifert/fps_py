import numpy as np
import pytest

from fpspy import stim


@pytest.fixture
def frametimes_and_triggers(np_rng):
    """Fixture providing sample s_frames and triggers."""
    n_trials = 100
    min_frames = 5
    max_frames = int(1e4)
    min_dur = 1 / 120
    max_dur = 5.0

    def get_triggers(n_frames):
        type = np_rng.choice(["one", "every", "two", "random"])
        res = None
        if type == "one":
            res = np.array([0])
        elif type == "every":
            res = np.arange(0, n_frames)
        elif type == "two":
            res = np.linspace(0, n_frames - 1, 2, dtype=int)
        elif type == "random":
            n_trigs = np_rng.integers(1, n_frames)
            res = np.sort(
                np_rng.choice(np.arange(n_frames), size=n_trigs, replace=False)
            )
        assert res is not None
        return res

    def generate_trial():
        for _ in range(n_trials):
            n_frames = np_rng.integers(min_frames, max_frames)
            start_time = np_rng.uniform(0.0, 10.0)
            frame_dur = np.cumsum(np_rng.uniform(min_dur, max_dur, n_frames))
            s_frames = np.concatenate(([start_time], start_time + frame_dur))
            triggers = get_triggers(n_frames)
            yield s_frames, triggers

    return generate_trial()


class TestLoop:
    """Tests for stim.loop() function."""

    def test_trigger_loop(self):
        """Insure a loop with full triggers forms an arithmetic sequence."""
        s_frames = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        triggers = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
        n_loops = 4
        idxs, s_out, triggers_out = stim.loop(s_frames, triggers, n_loops=n_loops)
        assert len(triggers_out) == len(triggers) * n_loops
        expected_triggers = np.arange(0, len(triggers) * n_loops)
        np.testing.assert_array_equal(triggers_out, expected_triggers)


    def test_single_loop(self, frametimes_and_triggers):
        """With n_loops=1, output should match input (minus end time)."""
        for s_frames, triggers in frametimes_and_triggers:
            expected_n_frames = len(s_frames) - 1  # exclude end time
            idxs, s_out, triggers_out = stim.loop(s_frames, triggers, n_loops=1)
            assert len(idxs) == expected_n_frames == len(s_out) - 1
            np.testing.assert_array_equal(idxs, np.arange(expected_n_frames))
            np.testing.assert_array_almost_equal(s_out, s_frames)
            np.testing.assert_array_equal(triggers_out, triggers)

    def test_multiple_loops(self, frametimes_and_triggers, np_rng):
        """Test loop with n_loops > 1."""
        for s_frames, triggers in frametimes_and_triggers:
            n_frames = len(s_frames) - 1
            n_triggers = len(triggers)
            period = s_frames[-1] - s_frames[0]
            n_loops = np_rng.integers(2, 10)

            idxs, s_out, triggers_out = stim.loop(s_frames, triggers, n_loops=n_loops)

            # Output lengths scale correctly
            assert len(idxs) == n_frames * n_loops
            assert len(s_out) == n_frames * n_loops + 1
            assert len(triggers_out) == n_triggers * n_loops

            # Frame indices tile correctly
            expected_idxs = np.tile(np.arange(n_frames), n_loops)
            np.testing.assert_array_equal(idxs, expected_idxs)

            # Schedule is strictly monotonically increasing
            assert np.all(np.diff(s_out) > 0)

            # Triggers are integers
            assert np.issubdtype(triggers_out.dtype, np.integer)

            # Each loop's schedule is offset by period
            for i in range(n_loops):
                loop_times = s_out[i * n_frames : (i + 1) * n_frames]
                expected_times = s_frames[:-1] + i * period
                np.testing.assert_array_almost_equal(loop_times, expected_times)

            # Triggers are offset by n_frames (including end time) per loop
            for i in range(n_loops):
                loop_triggers = triggers_out[i * n_triggers : (i + 1) * n_triggers]
                expected_triggers = triggers + i * n_frames
                np.testing.assert_array_equal(loop_triggers, expected_triggers)

    def test_errors(self):
        s_frames = np.array([0.0, 0.1, 0.2, 0.3])

        # Don't allow None s_frames.
        with pytest.raises(ValueError):
            stim.loop(s_frames, None, n_loops=2)

        # Don't allow empty triggers.
        with pytest.raises(ValueError):
            stim.loop(s_frames, np.array([]), n_loops=2)

        # Don't allow negative loops.
        with pytest.raises(ValueError):
            stim.loop(s_frames, np.array([]), n_loops=-1)

        # Can't be more triggers than frames.
        with pytest.raises(ValueError):
            stim.loop(s_frames, np.array([0, 1, 2, 3]), n_loops=2)


def test_spf_to_frame_times():
    """Main bug concern is not adding the end time."""
    spfs = np.arange(120*5) / 120
    n_frames = 1024
    for spf in spfs:
        frame_times = stim.spf_to_frame_times(spf, n_frames)
        assert len(frame_times) == n_frames + 1
