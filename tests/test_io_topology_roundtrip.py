"""Regression tests for model persistence.

Structural plasticity rewires ``src``/``dst``/``delay`` during training, so a
model rebuilt from the genome alone is not the model that was saved: the weight
vector ends up attached to a different set of synapses.
"""
import numpy as np
import pytest

from magicbrain import TextBrain, load_model, save_model
from magicbrain.io import FORMAT_VERSION
from magicbrain.tasks.text_task import build_vocab, train_loop_with_history

GENOME = "30121033102301230112332100123"
TEXT = "the quick brown fox jumps over the lazy dog. " * 60


@pytest.fixture(scope="module")
def trained():
    stoi, itos = build_vocab(TEXT)
    brain = TextBrain(GENOME, len(stoi))
    # Long enough for at least one prune/rewire cycle.
    train_loop_with_history(brain, TEXT, stoi, steps=int(brain.p["prune_every"]) * 2 + 50)
    assert brain._prune_count >= 1, "test needs at least one rewire to be meaningful"
    return brain, stoi, itos


def test_topology_survives_roundtrip(trained, tmp_path):
    brain, stoi, itos = trained
    path = str(tmp_path / "m.npz")
    save_model(brain, stoi, itos, path)
    restored, _, _, meta = load_model(path)

    assert meta["format_version"] == FORMAT_VERSION
    assert meta["topology_restored"] is True
    assert np.array_equal(restored.src, brain.src)
    assert np.array_equal(restored.dst, brain.dst)
    assert np.array_equal(restored.delay, brain.delay)
    assert np.array_equal(restored.is_inhib, brain.is_inhib)
    for d in range(1, 6):
        assert np.array_equal(restored.idx_by_delay[d], brain.idx_by_delay[d])


def test_reloaded_model_predicts_identically(trained, tmp_path):
    brain, stoi, itos = trained
    path = str(tmp_path / "m.npz")
    save_model(brain, stoi, itos, path)
    restored, r_stoi, _, _ = load_model(path)

    ids = [stoi[c] for c in TEXT[:200]]
    brain.reset_state()
    restored.reset_state()
    for token in ids:
        expected = brain.forward(token)
        actual = restored.forward(token)
        assert np.allclose(expected, actual, atol=1e-7)


def test_rng_state_survives_roundtrip(trained, tmp_path):
    brain, stoi, itos = trained
    path = str(tmp_path / "m.npz")
    save_model(brain, stoi, itos, path)
    restored, _, _, _ = load_model(path)
    assert restored.rng.bit_generator.state == brain.rng.bit_generator.state


def test_legacy_file_without_topology_still_loads(trained, tmp_path):
    """Files written before FORMAT_VERSION 2 must keep loading."""
    import json

    brain, stoi, itos = trained
    path = str(tmp_path / "legacy.npz")
    np.savez_compressed(
        path,
        genome_str=str(brain.genome_str),
        vocab=json.dumps({"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}}),
        w_slow=brain.w_slow,
        w_fast=brain.w_fast,
        R=brain.R,
        b=brain.b,
        theta=brain.theta,
        meta=json.dumps({"step": brain.step, "N": brain.N, "K": brain.K}),
    )
    restored, _, _, meta = load_model(path)
    assert meta["format_version"] == 1
    assert meta["topology_restored"] is False
    assert restored.N == brain.N
