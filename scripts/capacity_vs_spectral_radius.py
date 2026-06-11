#!/usr/bin/env python3
import os
import sys
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

from magicbrain.neurogenesis.pattern_memory import PatternMemory


def effective_spectral_radius(W: np.ndarray, iters: int = 40, seed: int = 0) -> float:
    A = W.astype(np.float64, copy=False)
    n = int(A.shape[0])
    rng = np.random.default_rng(int(seed))

    def rayleigh_max(M: np.ndarray) -> float:
        v = rng.normal(0.0, 1.0, size=n).astype(np.float64)
        v /= (np.linalg.norm(v) + 1e-12)
        for _ in range(int(iters)):
            v = M @ v
            nv = float(np.linalg.norm(v))
            if nv <= 1e-15:
                return 0.0
            v /= nv
        return float(v @ (M @ v))

    lmax = rayleigh_max(A)
    lmin = -rayleigh_max(-A)
    return float(max(abs(lmax), abs(lmin)))


def eval_accuracy(pm: PatternMemory, patterns: list[np.ndarray], sim_thr: float = 0.8) -> float:
    correct = 0
    for i, pat in enumerate(patterns):
        res = pm.recall(pat)
        if res.matched_index == i and float(res.similarity) >= float(sim_thr):
            correct += 1
    return float(correct) / float(max(1, len(patterns)))


def run_curve(N: int, sparsity: float, step: int, max_patterns: int, seed: int = 42) -> list[tuple[int, float, float]]:
    rng = np.random.default_rng(int(seed))
    n_active = max(1, int(N * float(sparsity)))
    out: list[tuple[int, float, float]] = []

    for n_patterns in range(int(step), int(max_patterns) + int(step), int(step)):
        pm = PatternMemory(N=int(N), sparsity=float(sparsity), max_capacity_fraction=0.2)
        patterns: list[np.ndarray] = []
        for _ in range(int(n_patterns)):
            pat = np.zeros(int(N), dtype=np.float32)
            active = rng.choice(int(N), size=int(n_active), replace=False)
            pat[active] = 1.0
            pm._imprint_fast(pat)
            patterns.append(pat)
            pm.patterns.append(pat)

        acc = eval_accuracy(pm, patterns, sim_thr=0.8)
        rho = effective_spectral_radius(pm.W, iters=40, seed=seed + n_patterns)
        out.append((int(n_patterns), float(acc), float(rho)))
        if acc < 0.5:
            break
    return out


def main():
    Ns = [256, 512, 1024, 1216]
    acc_thrs = [0.85, 0.90, 0.95]
    out_csv = os.path.join(script_dir, "capacity_vs_spectral_radius.csv")

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("N,n_patterns,accuracy,spectral_radius\n")
        for N in Ns:
            curve = run_curve(N=int(N), sparsity=0.1, step=5, max_patterns=int(0.2 * int(N)))
            for n, acc, sr in curve:
                f.write(f"{int(N)},{int(n)},{acc:.8f},{sr:.8f}\n")

    print(f"Wrote {out_csv}")

    for N in Ns:
        curve = run_curve(N=int(N), sparsity=0.1, step=5, max_patterns=int(0.2 * int(N)))
        n_arr = np.asarray([x[0] for x in curve], dtype=np.int32)
        acc_arr = np.asarray([x[1] for x in curve], dtype=np.float64)
        sr_arr = np.asarray([x[2] for x in curve], dtype=np.float64)
        if sr_arr.size >= 2 and float(np.std(sr_arr)) > 1e-12:
            corr = float(np.corrcoef(sr_arr, acc_arr)[0, 1])
        else:
            corr = float("nan")
        print(f"\nN={int(N)} corr(spectral_radius, accuracy)={corr:.4f}")
        for t in acc_thrs:
            ok = n_arr[acc_arr >= float(t)]
            cap = int(ok.max()) if ok.size else 0
            sr_at = float(sr_arr[n_arr.tolist().index(cap)]) if cap in n_arr else float("nan")
            print(f"  thr={t:.2f} capacity={cap}  radius_at_capacity={sr_at:.6f}")


if __name__ == "__main__":
    main()

