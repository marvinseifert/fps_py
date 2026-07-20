import numpy as np
import pytest

from fpspy import stim
from fpspy.stim import StimArray, TextureSequence


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


def _make_stim_array(n_channels, channel_mask=None, dtype=np.uint8, value=None):
    """Build a small StimArray for with_channels() tests.

    F=3, H=W=2. If `value` is given, every real channel is filled with it
    (useful for asserting exactly which channel ended up where); otherwise
    filled with a deterministic RNG so channels are distinguishable.
    """
    F, H, W = 3, 2, 2
    if value is not None:
        frames = np.full((F, H, W, n_channels), value, dtype=dtype)
    else:
        rng = np.random.default_rng(0)
        if dtype == np.uint8:
            frames = rng.integers(0, 255, size=(F, H, W, n_channels), dtype=np.uint8)
        else:
            frames = rng.random((F, H, W, n_channels)).astype(dtype)
    return StimArray(
        frames=frames, frame_durations=1 / 60, zoom=1, channel_mask=channel_mask
    )


class TestWithChannels:
    """Tests for StimArray.with_channels(): the physical-channel lookup table."""

    def test_mono_no_mask_broadcast_stays_mono(self):
        """A mono stimulus with no mask stays a real 1-channel array.

        Broadcasting to multiple physical outputs happens later (at render/shader
        time), not by materializing duplicate channels here.
        """
        s = _make_stim_array(n_channels=1)
        result = s.with_channels([0, 0, 0])
        assert result.n_channels == 1
        np.testing.assert_array_equal(result._frames, s._frames)

    def test_mono_no_mask_invalid_channel_raises(self):
        """A mono stimulus only has index 0; requesting others is a real config error."""
        s = _make_stim_array(n_channels=1)
        with pytest.raises(ValueError, match=r"1, 2"):
            s.with_channels([0, 1, 2])

    def test_multichannel_selects_requested_channels(self):
        s = _make_stim_array(n_channels=6)
        result = s.with_channels([3, 4, 5])
        assert result.n_channels == 3
        np.testing.assert_array_equal(result._frames, s._frames[..., [3, 4, 5]])

    def test_multichannel_out_of_range_raises(self):
        s = _make_stim_array(n_channels=6)
        with pytest.raises(ValueError, match=r"6, 7, 8"):
            s.with_channels([6, 7, 8])

    def test_multichannel_partial_out_of_range_raises_only_invalid(self):
        """Only the genuinely invalid indices are named, not the valid ones too."""
        s = _make_stim_array(n_channels=3)
        with pytest.raises(ValueError) as exc_info:
            s.with_channels([0, 1, 5])
        msg = str(exc_info.value)
        assert "5" in msg
        assert "0" not in msg.split("exist")[0].split("[")[-1].replace("5", "")

    def test_multichannel_allows_repeated_index(self):
        """Requesting the same real channel for multiple physical outputs is fine."""
        s = _make_stim_array(n_channels=3)
        result = s.with_channels([2, 2, 2])
        assert result.n_channels == 3
        for i in range(3):
            np.testing.assert_array_equal(result._frames[..., i], s._frames[..., 2])

    def test_mono_with_mask_expands_effective_channel_count(self):
        """A channel_mask on a mono stimulus is a broadcasting mask, not a filter:

        it expands how many (virtual) channels are valid to request, without
        changing the real, still-mono, underlying frame data.
        """
        mask = np.array([0, 0, 1], dtype=bool)
        s = _make_stim_array(n_channels=1, channel_mask=mask)
        assert s.n_channels == 3

        result = s.with_channels([0, 1, 2])
        assert result.n_channels == 3
        # Underlying real frame data is untouched -- still genuinely mono.
        np.testing.assert_array_equal(result._frames, s._frames)
        np.testing.assert_array_equal(result._channel_mask, mask[[0, 1, 2]])

    def test_mono_with_mask_out_of_range_raises(self):
        mask = np.array([0, 0, 1], dtype=bool)
        s = _make_stim_array(n_channels=1, channel_mask=mask)
        with pytest.raises(ValueError, match=r"\[3\]"):
            s.with_channels([0, 1, 2, 3])

    def test_mono_with_mask_requesting_masked_off_channel_is_allowed(self):
        """Requesting a masked-off (always-zero) channel is valid, not an error.

        The mask zeroes data, it doesn't remove the channel -- selecting it just
        means that physical output faithfully shows black/zero, as the mask says.
        """
        mask = np.array([0, 0, 1], dtype=bool)
        s = _make_stim_array(n_channels=1, channel_mask=mask)
        result = s.with_channels([0, 0, 0])
        assert result.n_channels == 3
        np.testing.assert_array_equal(
            result._channel_mask, np.array([0, 0, 0], dtype=bool)
        )

    def test_multichannel_with_matching_mask(self):
        """A mask the same size as a genuine multi-channel array gates those channels."""
        mask = np.array([1, 0, 1], dtype=bool)
        s = _make_stim_array(n_channels=3, channel_mask=mask)
        result = s.with_channels([0, 1, 2])
        assert result.n_channels == 3
        np.testing.assert_array_equal(result._frames, s._frames)
        np.testing.assert_array_equal(result._channel_mask, mask)


class TestTextureSequenceChannels:
    """Integration tests: TextureSequence renders the channel mapping correctly.

    Uses a standalone (headless) moderngl context -- no window/display needed.
    """

    @pytest.fixture
    def gl_ctx(self):
        import moderngl

        ctx = moderngl.create_context(standalone=True)
        yield ctx
        ctx.release()

    def _setup_and_render(self, ctx, stim_arr, channels, size=(4, 4)):
        prog = TextureSequence(stim_arr, lazy_textures=False)
        color_tex = ctx.texture(size, 4)
        fbo = ctx.framebuffer(color_attachments=[color_tex])
        fbo.use()
        ctx.viewport = (0, 0, *size)
        prog.setup(ctx, size[0], size[1], channels=channels, mirror=False, rotation=0)
        prog.render(ctx, 0, 0)
        data = np.frombuffer(color_tex.read(), dtype=np.uint8).reshape(
            size[1], size[0], 4
        )
        pixel = data[size[1] // 2, size[0] // 2]
        prog.cleanup()
        return pixel

    def test_mono_broadcast_renders_gray_not_red(self, gl_ctx):
        """A genuinely mono stimulus must use the mono-broadcast shader.

        Regression test: fragment_shader_colour.glsl only populates the red
        component when sampling an unswizzled single-channel texture, which
        would render red-only instead of neutral gray.
        """
        value = 200
        stim_arr = _make_stim_array(n_channels=1, dtype=np.uint8, value=value)
        pixel = self._setup_and_render(gl_ctx, stim_arr, channels=[0, 0, 0])
        assert pixel[0] == pixel[1] == pixel[2] == value, f"{pixel=}"

    def test_multichannel_renders_each_channel_correctly(self, gl_ctx):
        F, H, W = 1, 1, 1
        frames = np.zeros((F, H, W, 3), dtype=np.uint8)
        frames[0, 0, 0] = [10, 20, 30]
        stim_arr = StimArray(frames=frames, frame_durations=1 / 60, zoom=1)
        pixel = self._setup_and_render(gl_ctx, stim_arr, channels=[0, 1, 2])
        assert list(pixel[:3]) == [10, 20, 30], f"{pixel=}"

    def test_setup_raises_on_invalid_channels(self, gl_ctx):
        """The original bug: a mono stimulus with a multi-channel window config."""
        stim_arr = _make_stim_array(n_channels=1, dtype=np.uint8, value=100)
        prog = TextureSequence(stim_arr, lazy_textures=False)
        with pytest.raises(ValueError, match=r"1, 2"):
            prog.setup(gl_ctx, 4, 4, channels=[0, 1, 2], mirror=False, rotation=0)
