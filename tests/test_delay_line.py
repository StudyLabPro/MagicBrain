"""Regression tests for the axonal delay line.

``TextBrain.forward`` rotates six delay buffers every step. Rebinding the list
entries (instead of copying values) aliased the arrays: after four steps all of
``buffers[1..5]`` referenced one array and the 1-5 step axonal delays collapsed
into a single accumulator.
"""
import numpy as np

from magicbrain import TextBrain

GENOME = "30121033102301230112332100123"


def test_delay_buffers_stay_distinct():
    brain = TextBrain(GENOME, 16)
    for _ in range(12):
        brain.forward(0)
    ids = {id(buf) for buf in brain.buffers}
    assert len(ids) == len(brain.buffers), "delay buffers alias each other"


def test_delay_groups_are_independent():
    """Each delay slot must own its memory."""
    brain = TextBrain(GENOME, 16)
    for _ in range(8):
        brain.forward(1)
    for buf in brain.buffers:
        buf.fill(0.0)
    brain.buffers[3][0] = 1.0
    others = [float(brain.buffers[d][0]) for d in (1, 2, 4, 5)]
    assert others == [0.0, 0.0, 0.0, 0.0]


def test_spike_arrives_after_its_delay():
    """A value parked in delay slot d must reach slot 1 exactly d-1 steps later."""
    brain = TextBrain(GENOME, 16)
    brain.noise_std = 0.0
    for _ in range(8):  # let the rotation run long enough to alias, if it can
        brain.forward(0)
    brain.w_slow[:] = 0.0  # silence recurrent traffic: only the marker moves
    brain.w_fast[:] = 0.0
    for buf in brain.buffers:
        buf.fill(0.0)
    brain.buffers[3][7] = 1.0

    seen = [float(brain.buffers[1][7])]
    for _ in range(3):
        brain.forward(0)
        seen.append(float(brain.buffers[1][7]))

    assert seen[0] == 0.0
    assert seen[1] == 0.0
    assert seen[2] > 0.0, f"marker never arrived in slot 1: {seen}"
