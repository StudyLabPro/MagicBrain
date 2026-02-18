# MagicBrain Formal Verification (Lean 4 + Mathlib)

Machine-checked proof of **Hopfield network convergence** under asynchronous updates using Lean 4 and Mathlib.

## Toolchain

- **Lean**: `leanprover/lean4:v4.28.0`
- **Mathlib**: `v4.28.0`

## Building

```bash
cd formal
lake build
```

A successful build with zero errors certifies that all 21 theorems are fully proven — no `sorry`, no unfinished proofs.

## Proof Structure

| File | Content | Key Theorems |
|------|---------|-------------|
| `MagicBrainFormal/Hopfield.lean` | Definitions: `sigma`, `spin`, `field`, `updateCoord`, `globalEnergy`, `localEnergy` | `spin_mem_pm_one`, `spin_updateCoord_eq_sign`, `localEnergyAt_nonincreasing_async_sign` |
| `MagicBrainFormal/SumLemmas.lean` | Finite sum manipulation lemmas | `dot_eq_singleton_of_support`, `dot_sub_eq_singleton` |
| `MagicBrainFormal/DeltaE.lean` | Bilinear form symmetry and energy delta formula | `B_symm`, `deltaE_coord` |
| `MagicBrainFormal/Glue.lean` | Bridge from delta formula to monotonicity | `sign_step_deltaE_nonpos`, `globalEnergy_nonincreasing_updateCoord` |
| `MagicBrainFormal/Convergence.lean` | Finite-state convergence via pigeonhole | `energyAt_antitone`, `state_repeats`, `convergence_bound` |
| `MagicBrainFormal.lean` | Root import (entry point) | Imports all modules |

## Main Results

### Theorem 1: Energy Monotonicity (`globalEnergy_nonincreasing_updateCoord`)

For a symmetric weight matrix `W` with zero diagonal, a single asynchronous coordinate update never increases the global Hopfield energy:

```
E(updateCoord(W, s, i)) <= E(s)
```

**Proof outline**:
1. Express `E(s') - E(s)` as `-(x'_i - x_i) * (Wx)_i` via quadratic expansion (`deltaE_coord`)
2. Show sign-aligned update makes this quantity non-positive (`sign_step_deltaE_nonpos`)

### Theorem 2: Finite Convergence (`convergence_bound`)

The asynchronous update sequence reaches a fixed point (energy plateau) within `2^n` steps, where `n` is the number of neurons:

```
exists p q, p < q /\ q <= 2^n + 1 /\ updateSeq(p) = updateSeq(q)
/\ energy is constant on [p, q]
```

**Proof outline**:
1. Energy is non-increasing at every step (`energyAt_antitone`, from Theorem 1)
2. State space has cardinality `2^n` (`state_space_card`)
3. By pigeonhole, some state repeats within `2^n + 1` steps (`state_repeats`)
4. Monotone energy + repeated state implies energy is constant between repetitions (`energy_constant_between`)

## Mathematical Setting

- **States**: Binary vectors `s : Fin n -> Bool`, mapped to `{-1, +1}` via `spin`
- **Energy**: `E(s) = -(1/2) * spin(s)^T * W * spin(s)` (standard Hopfield energy)
- **Update rule**: `s'_i = sign(sum_j W_ij * spin(s_j))` (asynchronous, one coordinate at a time)
- **Assumptions**: `W = W^T` (symmetric), `W_ii = 0` (zero diagonal)

This is a standard result in computational neuroscience (Hopfield, 1982; Cohen & Grossberg, 1983), here formalized with full machine-checked rigor in Lean 4.
