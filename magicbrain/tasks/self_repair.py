from __future__ import annotations
import time
import numpy as np
from ..brain import TextBrain
from .text_task import build_vocab
from ..sampling import sample

def benchmark_self_repair(
    brain: TextBrain,
    text: str,
    stoi: dict,
    itos: dict,
    eval_steps: int = 2000,
    recovery_steps: int = 8000,
    damage_frac: float = 0.2,
    report_every: int = 1000,
    seed_text: str = "To be, or not to be",
    sample_chars: int = 300,
    verbose: bool = True,
) -> dict:
    """Damage a fraction of the synapses and measure how far the network recovers.

    Args:
        brain: A (usually pre-trained) network to lesion.
        text: Evaluation/recovery corpus.
        stoi, itos: Vocabulary mappings for ``text``.
        eval_steps: Forward-only steps per evaluation point.
        recovery_steps: Total learning steps after the lesion.
        damage_frac: Fraction of edges zeroed out.
        report_every: Evaluate the loss after every this many recovery steps.
        seed_text: Seed used for the qualitative text sample. Pass ``""`` to
            skip sampling entirely.
        sample_chars: Length of the qualitative sample.
        verbose: Print a human-readable report.

    Returns:
        Dict with ``pre_damage_loss``, ``post_damage_loss``, ``final_loss``,
        ``recovery_curve`` (list of ``{"step", "loss"}``), ``recovery_ratio``
        (share of the damage recovered, 1.0 == back to the pre-damage loss),
        ``damage_frac``, ``edges_total``, ``edges_damaged``, ``sample`` and
        ``wall_time_sec``.
    """
    ids = np.array([stoi[c] for c in text], dtype=np.int32)
    n = len(ids) - 1
    if n < 1:
        raise ValueError("Text too short.")

    if verbose:
        print(f"\nRunning self-repair benchmark (damage={damage_frac:.2f})...")

    def eval_loss(steps: int) -> float:
        losses = []
        start_idx = brain.step % n

        brain.reset_state()

        curr_x = int(ids[start_idx])

        # Warmup
        for _ in range(50):
            brain.forward(curr_x)
            start_idx = (start_idx + 1) % n
            curr_x = int(ids[start_idx])

        for _ in range(steps):
            probs = brain.forward(curr_x)
            target = int(ids[(start_idx + 1) % n])

            p = float(probs[target])
            losses.append(float(-np.log(p + 1e-9)))

            start_idx = (start_idx + 1) % n
            curr_x = target

        return float(np.mean(losses))

    def train_steps(steps: int):
        curr_idx = brain.step % n
        for _ in range(steps):
            x = int(ids[curr_idx])
            y = int(ids[(curr_idx + 1) % n])
            probs = brain.forward(x)
            brain.learn(y, probs)
            curr_idx = (curr_idx + 1) % n

    t0 = time.time()

    # 1. Pre-damage eval
    pre_eval = eval_loss(eval_steps)

    # 2. Damage
    edges_total = int(brain.src.shape[0])
    edges_damaged = int(edges_total * float(np.clip(damage_frac, 0.0, 1.0)))
    brain.damage_edges(damage_frac)

    # 3. Post-damage eval
    post_damage_eval = eval_loss(eval_steps)

    # 4. Recovery
    recovery_curve = []
    done = 0
    remaining = int(recovery_steps)
    while remaining > 0:
        chunk = min(int(report_every), remaining)
        train_steps(chunk)
        remaining -= chunk
        done += chunk
        recovery_curve.append({"step": done, "loss": eval_loss(eval_steps)})

    final_loss = recovery_curve[-1]["loss"] if recovery_curve else post_damage_eval

    gap = post_damage_eval - pre_eval
    recovery_ratio = float((post_damage_eval - final_loss) / gap) if abs(gap) > 1e-9 else None

    if verbose:
        print(
            f"Self-repair(eval): pre={pre_eval:.4f} | post_damage={post_damage_eval:.4f} | "
            f"final={final_loss:.4f}"
        )
        if recovery_curve:
            parts = [f"{p['step']}:{p['loss']:.4f}" for p in recovery_curve]
            print("Recovery curve:")
            print("  " + " | ".join(parts))

    generated = ""
    if seed_text and sample_chars > 0:
        generated = sample(brain, stoi, itos, seed=seed_text, n=sample_chars, temperature=0.75)
        if verbose:
            print("-" * 60)
            print(generated)
            print("-" * 60)

    return {
        "pre_damage_loss": float(pre_eval),
        "post_damage_loss": float(post_damage_eval),
        "final_loss": float(final_loss),
        "recovery_curve": recovery_curve,
        "recovery_ratio": recovery_ratio,
        "damage_frac": float(damage_frac),
        "edges_total": edges_total,
        "edges_damaged": edges_damaged,
        "eval_steps": int(eval_steps),
        "recovery_steps": int(recovery_steps),
        "sample": generated,
        "wall_time_sec": float(time.time() - t0),
    }
