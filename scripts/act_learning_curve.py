#!/usr/bin/env python3
import os
import sys
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

from magicbrain.brain import TextBrain
from magicbrain.integration.act_backend import ACTBackend
from magicbrain.utils import sigmoid


def max_genome() -> str:
    return "3" * 24


def entropy(p: np.ndarray) -> float:
    pp = np.clip(p.astype(np.float64), 1e-12, 1.0)
    return float(-np.sum(pp * np.log(pp)))


def rolling_hit_step(losses: np.ndarray, threshold: float, window: int = 100) -> int:
    L = losses.astype(np.float64)
    if L.size < window:
        return -1
    c = np.cumsum(np.insert(L, 0, 0.0))
    rm = (c[window:] - c[:-window]) / float(window)
    hits = np.where(rm < float(threshold))[0]
    return int(hits[0] + window - 1) if hits.size else -1


def observe_update_metrics(brain: TextBrain, target_id: int, probs: np.ndarray) -> tuple[float, float, float, int]:
    grad = probs.astype(np.float32).copy()
    grad[int(target_id)] -= 1.0

    lr = float(brain.p["lr"]) if hasattr(brain, "p") else 0.0
    lr_out = lr * float(brain.lr_out_mul)
    lr_rec = lr * float(brain.lr_rec_mul)

    state = brain.compute_state().astype(np.float32)
    dR = (lr_out * (state[:, None] * grad[None, :])).astype(np.float32)
    clip_R = int(np.any(np.abs(dR) > float(brain.max_R_update)))
    dR = np.clip(dR, -float(brain.max_R_update), float(brain.max_R_update))

    db = (lr_out * grad).astype(np.float32)
    clip_b = int(np.any(np.abs(db) > float(brain.max_b_update)))
    db = np.clip(db, -float(brain.max_b_update), float(brain.max_b_update))

    pre = brain.trace_fast[brain.src].astype(np.float32)
    post = brain.a[brain.dst].astype(np.float32)
    loss = float(-np.log(float(probs[int(target_id)]) + 1e-9))
    adv = float(brain.loss_ema - loss)
    gain = float(brain.p["dopamine_gain"])
    bias = float(brain.p["dopamine_bias"])
    dopamine = float(sigmoid(gain * adv + bias))
    dW = (lr_rec * dopamine * adv * pre * post).astype(np.float32)
    clip_W = int(np.any(np.abs(dW) > 0.02))
    dW = np.clip(dW, -0.02, 0.02)

    nR = float(np.linalg.norm(dR.ravel(), ord=2))
    nB = float(np.linalg.norm(db.ravel(), ord=2))
    nW = float(np.linalg.norm(dW.ravel(), ord=2))
    clip_any = int((clip_R or clip_b or clip_W))
    return nR, nB, nW, clip_any


def run_training_trace(
    use_act: bool,
    xs: np.ndarray,
    ys: np.ndarray,
    vocab: int,
    seed_override: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    brain = TextBrain(max_genome(), vocab_size=vocab, seed_override=seed_override, use_act=use_act)
    losses = np.empty(xs.shape[0], dtype=np.float64)
    ent = np.empty(xs.shape[0], dtype=np.float64)
    nR = np.empty(xs.shape[0], dtype=np.float64)
    nB = np.empty(xs.shape[0], dtype=np.float64)
    nW = np.empty(xs.shape[0], dtype=np.float64)
    clip = np.empty(xs.shape[0], dtype=np.int32)
    for i in range(xs.shape[0]):
        probs = brain.forward(int(xs[i]))
        nRi, nBi, nWi, clipi = observe_update_metrics(brain, int(ys[i]), probs)
        losses[i] = float(brain.learn(int(ys[i]), probs))
        ent[i] = entropy(probs)
        nR[i] = nRi
        nB[i] = nBi
        nW[i] = nWi
        clip[i] = clipi
    return losses, ent, nR, nB, nW, clip


def act_accumulation_microbenchmark(iters: int = 2_000_000) -> dict:
    act = ACTBackend()

    w0 = np.zeros(1, dtype=np.float32)
    delta = np.full(1, 1e-7, dtype=np.float32)

    w_float = w0.copy()
    for _ in range(int(iters)):
        w_float = w_float + delta

    if act.available:
        w_act = w0.copy()
        for _ in range(int(iters)):
            w_act = act.weight_update(w_act, delta, 1.0)
        return {
            "iters": int(iters),
            "float32_sum": float(w_float[0]),
            "act_sum": float(w_act[0]),
            "act_available": True,
        }

    return {
        "iters": int(iters),
        "float32_sum": float(w_float[0]),
        "act_sum": float("nan"),
        "act_available": False,
    }


def main():
    vocab = 10
    steps = 3000
    stream_seed = 12345

    rng = np.random.default_rng(stream_seed)
    xs = rng.integers(0, vocab, size=steps, dtype=np.int32)
    ys = rng.integers(0, vocab, size=steps, dtype=np.int32)

    seed_override = 424242
    L0, H0, nR0, nB0, nW0, c0 = run_training_trace(False, xs, ys, vocab=vocab, seed_override=seed_override)
    L1, H1, nR1, nB1, nW1, c1 = run_training_trace(True, xs, ys, vocab=vocab, seed_override=seed_override)

    out_csv = os.path.join(script_dir, "act_training_trace.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(
            "step,"
            "loss_no_act,loss_act,"
            "entropy_no_act,entropy_act,"
            "dR_norm_no_act,dR_norm_act,"
            "db_norm_no_act,db_norm_act,"
            "dW_norm_no_act,dW_norm_act,"
            "clip_no_act,clip_act\n"
        )
        for i in range(steps):
            f.write(
                f"{i},"
                f"{L0[i]:.8f},{L1[i]:.8f},"
                f"{H0[i]:.8f},{H1[i]:.8f},"
                f"{nR0[i]:.8f},{nR1[i]:.8f},"
                f"{nB0[i]:.8f},{nB1[i]:.8f},"
                f"{nW0[i]:.8f},{nW1[i]:.8f},"
                f"{int(c0[i])},{int(c1[i])}\n"
            )
    print(f"Wrote {out_csv}")

    print("Final loss (no ACT):", float(np.mean(L0[-200:])))
    print("Final loss (ACT)   :", float(np.mean(L1[-200:])))
    print("Final entropy (no ACT):", float(np.mean(H0[-200:])))
    print("Final entropy (ACT)   :", float(np.mean(H1[-200:])))
    print("Clip rate (no ACT):", float(np.mean(c0)))
    print("Clip rate (ACT)   :", float(np.mean(c1)))

    thr = 2.0
    t0 = rolling_hit_step(L0, thr, window=200)
    t1 = rolling_hit_step(L1, thr, window=200)
    print(f"Convergence time @loss<{thr} (window=200): no ACT={t0}  ACT={t1}")
    print("Var(dW_norm) no ACT:", float(np.var(nW0)))
    print("Var(dW_norm) ACT   :", float(np.var(nW1)))

    micro = act_accumulation_microbenchmark(iters=500_000)
    print("ACT microbenchmark:", micro)

    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure()
        plt.plot(L0, label="no ACT")
        plt.plot(L1, label="ACT")
        plt.yscale("log")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(script_dir, "act_curve.png")
        plt.savefig(out_png, dpi=200)
        print(f"Wrote {out_png}")

        plt.figure()
        plt.plot(H0, label="no ACT")
        plt.plot(H1, label="ACT")
        plt.xlabel("step")
        plt.ylabel("entropy(probs)")
        plt.legend()
        plt.tight_layout()
        out_png2 = os.path.join(script_dir, "act_entropy.png")
        plt.savefig(out_png2, dpi=200)
        print(f"Wrote {out_png2}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
