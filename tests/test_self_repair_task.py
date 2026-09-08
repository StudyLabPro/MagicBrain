"""``benchmark_self_repair`` must return its numbers, not only print them.

The task used to print a report and return ``None``, so a caller (the
playground, a script, a notebook) had to re-implement the benchmark to get the
values. It now returns a dict; the prints stay under ``verbose``.
"""
from __future__ import annotations

from magicbrain import TextBrain
from magicbrain.tasks.self_repair import benchmark_self_repair
from magicbrain.tasks.text_task import build_vocab

GENOME = "30121033102301230112332100123"
TEXT = "the quick brown fox jumps over the lazy dog. " * 30


def _run(**kwargs):
    stoi, itos = build_vocab(TEXT)
    brain = TextBrain(GENOME, len(stoi))
    return benchmark_self_repair(
        brain, TEXT, stoi, itos,
        eval_steps=60, recovery_steps=200, report_every=100,
        damage_frac=0.2, verbose=False, **kwargs,
    )


def test_returns_measurements():
    out = _run(seed_text="")
    for key in ("pre_damage_loss", "post_damage_loss", "final_loss",
                "recovery_curve", "recovery_ratio", "damage_frac",
                "edges_total", "edges_damaged", "wall_time_sec"):
        assert key in out, key
    assert out["damage_frac"] == 0.2
    assert out["edges_total"] > 0
    assert out["edges_damaged"] == int(out["edges_total"] * 0.2)
    assert [p["step"] for p in out["recovery_curve"]] == [100, 200]
    assert out["final_loss"] == out["recovery_curve"][-1]["loss"]
    assert out["sample"] == "", "пустая затравка означает отказ от генерации"


def test_sampling_is_optional_and_bounded():
    out = _run(seed_text="the ", sample_chars=40)
    assert 0 < len(out["sample"]) <= 40 + len("the ")
