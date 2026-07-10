"""Tests for the energy → learning seam (Hopfield energy wired into TextBrain).

See docs/REFLEXIVE_VERTICAL_SEAMS.md (шов 3) and METAMIND_ALIGNMENT_PLAN §1-2:
the neurogenesis energy function is folded into the online learning loop as an
observable (energy, delta_energy) and a Survival/Debt control signal.
"""
import numpy as np

from magicbrain import TextBrain

GENOME = "30121033102301230112332100123"
TEXT = "the quick brown fox jumps over the lazy dog. " * 40
_CHARS = sorted(set(TEXT))
_STOI = {c: i for i, c in enumerate(_CHARS)}
VOCAB = len(_CHARS)
IDS = [_STOI[c] for c in TEXT]


def _train(track_energy, steps, seed=7, energy_window=200, debt_max=0.0):
    brain = TextBrain(
        GENOME, vocab_size=VOCAB, seed_override=seed,
        track_energy=track_energy, energy_window=energy_window, debt_max=debt_max,
    )
    losses, deltas = [], []
    for t in range(steps):
        tok, nxt = IDS[t % (len(IDS) - 1)], IDS[(t + 1) % len(IDS)]
        probs = brain.forward(tok)
        losses.append(brain.learn(nxt, probs))
        if track_energy:
            deltas.append(brain.delta_energy)
    return brain, np.array(losses), np.array(deltas)


def test_track_energy_off_is_default_and_noninterfering():
    """Below the debt window the hook cannot fire, so track_energy must not
    change a single number vs the default path."""
    _, loss_off, _ = _train(False, steps=150)
    _, loss_on, _ = _train(True, steps=150, energy_window=200)
    assert np.max(np.abs(loss_off - loss_on)) == 0.0


def test_energy_is_tracked_and_learning_does_not_inflate_it():
    """Energy observables are finite and vary; imprinting patterns lowers (does
    not inflate) the Hopfield energy of the learned states."""
    brain, _, deltas = _train(True, steps=260)
    assert np.isfinite(brain.energy)
    assert np.isfinite(deltas).all()
    assert np.any(deltas != 0.0)                 # energy actually moves
    assert deltas.mean() <= 0.05                 # net non-inflating (observed ≈ -0.18)


def test_debt_hook_triggers_extra_structural_plasticity():
    """A low debt threshold (always in debt) provokes more prune/rewire events
    and lowers the learning rate; a high threshold (never in debt) does not."""
    brain_debt, _, _ = _train(True, steps=350, energy_window=100, debt_max=-1e9)
    brain_calm, _, _ = _train(True, steps=350, energy_window=100, debt_max=1e9)
    assert brain_debt._prune_count > brain_calm._prune_count
    assert brain_debt._prune_count >= 1
    assert brain_debt._lr_penalty < 1.0          # sustained debt throttles lr
    assert brain_calm._lr_penalty == 1.0
