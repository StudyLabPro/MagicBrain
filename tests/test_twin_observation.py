"""Tests for the L4→L2 weld (шов 6): the digital twin folds KnowledgeBaseAI's
mastery score as an observation instead of recomputing it.

The twin does not own mastery (KB is source of truth); it assimilates the score
via belief += α·(observation − belief) — the same self-correction equation as
KB's mastery_update, now nested. See docs/REFLEXIVE_VERTICAL_SEAMS.md.
"""
from magicbrain.integration.neural_digital_twin import NeuralDigitalTwin


def test_twin_converges_to_a_stable_observation_and_surprise_decays():
    twin = NeuralDigitalTwin("student-1", observation_gain=0.4)
    surprises = []
    for _ in range(6):
        r = twin.process_interaction_event(
            {"type": "mastery_observed", "topic_id": "T", "score": 0.8}
        )
        surprises.append(abs(r["mastery_surprise"]))
    # Twin mastery approaches the observed value...
    assert 0.7 < twin.mastery_scores["T"] < 0.8
    # ...and the aggregate surprise strictly decays toward zero (ring closes).
    assert all(b < a for a, b in zip(surprises, surprises[1:]))
    assert surprises[-1] < 0.1


def test_twin_does_not_recompute_mastery_and_stays_bounded():
    twin = NeuralDigitalTwin("student-2", observation_gain=0.4)
    # A single observation moves belief by exactly α·(obs − prior) from 0.
    r = twin.process_interaction_event(
        {"type": "student_progress_updated", "topic_id": "A", "score": 1.0}
    )
    assert twin.mastery_scores["A"] == 0.4          # 0 + 0.4·(1.0 − 0)
    assert r["mastery_surprise"] == 1.0             # residual reported upward
    # Bounded to [0, 1] even for extreme corrections.
    twin.mastery_scores["A"] = 0.95
    twin.process_interaction_event(
        {"type": "mastery_observed", "topic_id": "A", "score": 1.0, "alpha": 1.0}
    )
    assert 0.0 <= twin.mastery_scores["A"] <= 1.0


def test_observation_alpha_is_configurable_per_event():
    twin = NeuralDigitalTwin("student-3", observation_gain=0.4)
    r = twin.process_interaction_event(
        {"type": "mastery_observed", "topic_id": "B", "score": 1.0, "alpha": 1.0}
    )
    assert twin.mastery_scores["B"] == 1.0          # α=1 assimilates fully
    assert r["mastery_surprise"] == 1.0
