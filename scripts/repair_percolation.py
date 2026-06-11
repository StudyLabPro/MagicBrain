#!/usr/bin/env python3
import os
import sys
import math
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

from magicbrain.brain import TextBrain


def max_genome() -> str:
    return "3" * 24


def repair_genome() -> str:
    g = list("3" * 24)
    g[20] = "0"
    g[21] = "3"
    g[22] = "3"
    return "".join(g)


def run_stream(brain: TextBrain, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    losses = np.empty(xs.shape[0], dtype=np.float64)
    for i in range(xs.shape[0]):
        probs = brain.forward(int(xs[i]))
        losses[i] = float(brain.learn(int(ys[i]), probs))
    return losses


def recovery_for_frac(
    frac: float,
    runs: int = 30,
    train_steps: int = 1000,
    eval_steps: int = 200,
    repair_steps: int = 1000,
    vocab: int = 10,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    frac = float(np.clip(frac, 0.0, 1.0))
    rng = np.random.default_rng(int(seed))
    rec: list[float] = []
    rewire_fracs: list[float] = []

    for r in range(int(runs)):
        stream_seed = int(rng.integers(0, 2**31 - 1))
        srng = np.random.default_rng(stream_seed)
        xs_train = srng.integers(0, vocab, size=int(train_steps), dtype=np.int32)
        ys_train = srng.integers(0, vocab, size=int(train_steps), dtype=np.int32)
        xs_eval = srng.integers(0, vocab, size=int(eval_steps), dtype=np.int32)
        ys_eval = srng.integers(0, vocab, size=int(eval_steps), dtype=np.int32)
        xs_repair = srng.integers(0, vocab, size=int(repair_steps), dtype=np.int32)
        ys_repair = srng.integers(0, vocab, size=int(repair_steps), dtype=np.int32)
        xs_eval2 = srng.integers(0, vocab, size=int(eval_steps), dtype=np.int32)
        ys_eval2 = srng.integers(0, vocab, size=int(eval_steps), dtype=np.int32)

        brain = TextBrain(repair_genome(), vocab_size=vocab, seed_override=stream_seed, use_act=False)
        src0 = brain.src.copy()
        dst0 = brain.dst.copy()
        run_stream(brain, xs_train, ys_train)
        L0 = float(np.mean(run_stream(brain, xs_eval, ys_eval)))

        brain.damage_edges(frac)

        run_stream(brain, xs_repair, ys_repair)
        L1 = float(np.mean(run_stream(brain, xs_eval2, ys_eval2)))

        rec.append(L0 / (L1 + 1e-12))
        rewire_fracs.append(float(np.mean((brain.src != src0) | (brain.dst != dst0))))

    arr = np.asarray(rec, dtype=np.float64)
    std = float(arr.std(ddof=1) if arr.size > 1 else 0.0)
    ci95 = float(1.96 * std / math.sqrt(max(1, int(runs))))
    return float(arr.mean()), std, ci95, float(np.mean(np.asarray(rewire_fracs, dtype=np.float64)))


def percolation_model(fracs: np.ndarray, fc: float, beta: float, scale: float) -> np.ndarray:
    fr = np.asarray(fracs, dtype=np.float64)
    out = np.zeros_like(fr)
    mask = fr < fc
    out[mask] = scale * np.power(1.0 - fr[mask] / fc, beta)
    return out


def fit_percolation(fracs: np.ndarray, means: np.ndarray) -> tuple[float, float, float]:
    fr = np.asarray(fracs, dtype=np.float64)
    y = np.asarray(means, dtype=np.float64)

    best = (0.0, 0.0, 0.0)
    best_sse = float("inf")

    fc_grid = np.linspace(max(0.15, float(np.max(fr)) * 0.6), 0.95, 81)
    beta_grid = np.linspace(0.05, 3.0, 61)
    scale_grid = np.linspace(max(0.2, float(np.max(y)) * 0.6), float(np.max(y)) * 1.4, 41)

    for fc in fc_grid:
        for beta in beta_grid:
            base = percolation_model(fr, float(fc), float(beta), 1.0)
            if float(np.max(base)) <= 1e-12:
                continue
            for scale in scale_grid:
                pred = float(scale) * base
                sse = float(np.mean((pred - y) ** 2))
                if sse < best_sse:
                    best_sse = sse
                    best = (float(fc), float(beta), float(scale))
    return best


def main():
    fracs = np.round(np.linspace(0.1, 0.8, 8), 2)
    means: list[float] = []
    stds: list[float] = []
    ci95s: list[float] = []
    rewire_means: list[float] = []

    for f in fracs:
        m, s, ci95, rw = recovery_for_frac(float(f), runs=30, train_steps=850, eval_steps=200, repair_steps=850)
        means.append(m)
        stds.append(s)
        ci95s.append(ci95)
        rewire_means.append(rw)
        print(f"frac={float(f):.2f} recovery={m:.3f} ± {s:.3f}  CI95={ci95:.3f}  rewire={rw:.3f}")

    means_arr = np.asarray(means, dtype=np.float64)
    fc, beta, scale = fit_percolation(fracs.astype(np.float64), means_arr)
    print(f"\nFIT: fc={fc:.3f}, beta={beta:.3f}, scale={scale:.3f}")

    out_csv = os.path.join(script_dir, "repair_curve.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("damage_frac,mean,std,ci95,rewire_mean,fit\n")
        for df, m, s, ci95, rw in zip(fracs, means, stds, ci95s, rewire_means):
            fit = float(percolation_model(np.asarray([df], dtype=np.float64), fc, beta, scale)[0])
            f.write(f"{float(df):.2f},{m:.8f},{s:.8f},{ci95:.8f},{rw:.8f},{fit:.8f}\n")
    print(f"Wrote {out_csv}")

    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure()
        plt.errorbar(fracs, means_arr, yerr=np.asarray(stds, dtype=np.float64), fmt="o", label="empirical")
        xs = np.linspace(float(np.min(fracs)), float(np.max(fracs)), 200)
        plt.plot(xs, percolation_model(xs, fc, beta, scale), "--", label="fit")
        plt.xlabel("damage_frac")
        plt.ylabel("recovery_ratio")
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(script_dir, "repair_curve.png")
        plt.savefig(out_png, dpi=200)
        print(f"Wrote {out_png}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
