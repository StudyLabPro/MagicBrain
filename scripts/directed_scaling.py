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

from magicbrain.neurogenesis.pattern_memory import PatternMemory


def capacity_ratio(
    N: int,
    sparsity: float = 0.1,
    reps: int = 3,
    acc_thr: float = 0.9,
    step: int = 5,
) -> tuple[float, float]:
    ratios: list[float] = []
    for rep in range(int(reps)):
        rng = np.random.default_rng(1000 + rep)
        pm = PatternMemory(N=int(N), sparsity=float(sparsity), max_capacity_fraction=0.2)
        curve = pm.capacity_test(rng=rng, step=int(step))
        feasible = [n for n, acc in curve.items() if float(acc) >= float(acc_thr)]
        ratios.append((max(feasible) if feasible else 0) / float(N))
    arr = np.asarray(ratios, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)


def main():
    print("NOTE: Symmetrized directed initialization + Storkey rule (dW is symmetrized).")
    Ns = np.array([256, 512, 768, 1024, 1216], dtype=int)
    acc_thrs = [0.85, 0.90, 0.95]
    stats_by_thr: dict[float, tuple[list[float], list[float], list[float]]] = {}
    for t in acc_thrs:
        stats_by_thr[float(t)] = ([], [], [])

    for N in Ns:
        row = [f"N={int(N)}"]
        for t in acc_thrs:
            m, s = capacity_ratio(int(N), acc_thr=float(t))
            ci95 = 1.96 * s / math.sqrt(3)
            stats_by_thr[float(t)][0].append(m)
            stats_by_thr[float(t)][1].append(s)
            stats_by_thr[float(t)][2].append(ci95)
            row.append(f"thr={t:.2f} C/N={m:.4f} ± {s:.4f} CI95={ci95:.4f}")
        print("  ".join(row))

    means_arr = np.asarray(stats_by_thr[0.90][0], dtype=float)
    a, b = np.polyfit(np.log(Ns.astype(float)), means_arr, 1)
    print("Fit @thr=0.90: (C/N) = a * ln(N) + b")
    print(f"a={a:.6f} b={b:.6f}")

    out_csv = os.path.join(script_dir, "directed_scaling.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(
            "N,"
            "mean_thr085,std_thr085,ci95_thr085,"
            "mean_thr090,std_thr090,ci95_thr090,"
            "mean_thr095,std_thr095,ci95_thr095,"
            "fit_thr090\n"
        )
        for idx, N in enumerate(Ns):
            fit = float(a * math.log(float(N)) + b)
            m85, s85, c85 = (stats_by_thr[0.85][0][idx], stats_by_thr[0.85][1][idx], stats_by_thr[0.85][2][idx])
            m90, s90, c90 = (stats_by_thr[0.90][0][idx], stats_by_thr[0.90][1][idx], stats_by_thr[0.90][2][idx])
            m95, s95, c95 = (stats_by_thr[0.95][0][idx], stats_by_thr[0.95][1][idx], stats_by_thr[0.95][2][idx])
            f.write(
                f"{int(N)},"
                f"{m85:.8f},{s85:.8f},{c85:.8f},"
                f"{m90:.8f},{s90:.8f},{c90:.8f},"
                f"{m95:.8f},{s95:.8f},{c95:.8f},"
                f"{fit:.8f}\n"
            )
    print(f"Wrote {out_csv}")

    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure()
        plt.semilogx(Ns, means_arr, marker="o", linestyle="none", label="empirical (thr=0.90)")
        plt.semilogx(Ns, a * np.log(Ns.astype(float)) + b, linestyle="--", label="fit: a ln N + b")
        plt.xlabel("N")
        plt.ylabel("C/N")
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(script_dir, "directed_scaling.png")
        plt.savefig(out_png, dpi=200)
        print(f"Wrote {out_png}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
