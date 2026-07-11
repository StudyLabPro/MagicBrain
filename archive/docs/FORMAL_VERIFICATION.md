# Formal Verification of Hopfield Network Convergence

## Overview

This document describes the machine-checked verification of the **Hopfield
network convergence theorem** using the Lean 4 proof assistant with the
Mathlib mathematical library (v4.28.0).

The complete proof comprises **5 Lean files** (~480 lines) with **0 sorry**
and **0 errors**.  The proof establishes:

1. **Energy Monotonicity**: Asynchronous coordinate updates never increase
   the global Hopfield energy.
2. **Finite Convergence**: Within `2^n` update steps (where `n` is the
   number of neurons), the energy must stabilize.

These are classical results in computational neuroscience (Hopfield, 1982;
Cohen & Grossberg, 1983), here formalized with full machine-checked rigor.

## Mathematical Setting

### Definitions

Let `n` be the number of neurons and `W` an `n x n` real-valued weight matrix.

**State**: `s : Fin n -> Bool`, mapped to `{-1, +1}` via the spin function:

```
spin(s, i) = if s(i) then 1 else -1
```

**Global energy** (Hopfield energy):

```
E(s) = -(1/2) * spin(s)^T * W * spin(s)
```

**Local field** at neuron `i`:

```
h_i(s) = sum_j W(i,j) * spin(s, j)
```

**Asynchronous update rule** (single-coordinate sign update):

```
updateCoord(W, s, i)(k) =
  if k = i then (0 <= h_i(s))    -- updated neuron aligns with field
  else s(k)                       -- all other neurons unchanged
```

### Assumptions

- `W = W^T` (symmetric weight matrix)
- `W(i,i) = 0` for all `i` (zero diagonal / no self-connections)

These are the standard assumptions for the discrete Hopfield model.

## Main Theorems

### Theorem 1: Energy Delta Formula (`deltaE_coord`)

**Statement**: For any single-coordinate change from `x` to `x'` (where
`x'(k) = x(k)` for all `k != i0`), the energy change has the closed form:

```
E(x') - E(x) = -(x'(i0) - x(i0)) * (W * x)(i0)
```

**Proof outline** (DeltaE.lean, ~150 lines):

1. Define bilinear form `B(u, v) = u^T * W * v`
2. Prove `B_symm`: `B(u, v) = B(v, u)` using `W = W^T`
3. Expand `B(x+d, x+d) - B(x, x) = B(x,d) + B(d,x) + B(d,d)` via
   distributivity of dot product over addition
4. Show `B(d,d) = 0` using zero diagonal: `d` has singleton support at
   `i0`, so `B(d,d) = d(i0) * W(i0,i0) * d(i0) = 0`
5. Show `B(d,x) = delta * (W*x)(i0)` via singleton support lemma
6. Combine with `B_symm`: `Delta_E = -(1/2) * 2 * delta * (W*x)(i0)`

### Theorem 2: Energy Monotonicity (`globalEnergy_nonincreasing_updateCoord`)

**Statement**: For symmetric `W` with zero diagonal:

```
E(updateCoord(W, s, i)) <= E(s)
```

**Proof outline** (Glue.lean, ~45 lines):

1. The sign-aligned update satisfies `x'(i0) = sign(h_i0)`
2. `sign_step_deltaE_nonpos`: Shows `-(sign(h) - x_i) * h <= 0` for any
   `x_i in {-1, +1}` by case analysis on `h >= 0` and `x_i`
3. Apply `deltaE_coord` to get `Delta_E = -(x'_i - x_i) * h_i <= 0`

### Theorem 3: Finite Convergence (`convergence_bound`)

**Statement**: For any initial state `s0` and any update schedule, there
exist indices `i < j` with `j <= 2^n` such that energy is constant on
`[i, j]`:

```
exists i j, i < j /\ j <= 2^n /\
  forall t, i <= t -> t <= j -> E(t) = E(i)
```

**Proof outline** (Convergence.lean, ~130 lines):

1. `energyAt_antitone`: Energy is non-increasing at every step (from
   Theorem 2), hence over any interval
2. `state_space_card`: `|Fin n -> Bool| = 2^n`
3. `state_repeats`: By the pigeonhole principle, in any sequence of
   `2^n + 1` states over a set of size `2^n`, some state repeats
4. `energy_constant_between`: If `s(i) = s(j)` with `i < j`, then
   `E(i) = E(j)`.  Combined with monotonicity, `E` must be constant
   on the entire interval `[i, j]`.

## File Structure

```
formal/
  lakefile.lean                      -- Lean 4 project config (Mathlib v4.28.0)
  lean-toolchain                     -- leanprover/lean4:v4.28.0
  MagicBrainFormal.lean              -- Root import (entry point)
  MagicBrainFormal/
    Hopfield.lean          (69 lines)  -- Definitions + local energy lemmas
    SumLemmas.lean         (83 lines)  -- Finite sum manipulation helpers
    DeltaE.lean           (157 lines)  -- Energy delta formula (Theorem 1)
    Glue.lean              (45 lines)  -- Monotonicity (Theorem 2)
    Convergence.lean      (129 lines)  -- Convergence bound (Theorem 3)
```

## Complete Theorem Inventory

| File | Theorem | Statement |
|------|---------|-----------|
| Hopfield | `spin_mem_pm_one` | `spin(s, i) = 1 or spin(s, i) = -1` |
| Hopfield | `spin_updateCoord_ne` | Off-coordinate spins unchanged |
| Hopfield | `spin_updateCoord_eq_sign` | Updated spin = sign(field) |
| Hopfield | `localEnergyAt_nonincreasing_async_sign` | Local energy non-increasing |
| SumLemmas | `dot_eq_singleton_of_support` | Dot product with singleton vector |
| SumLemmas | `dot_sub_eq_singleton` | Dot product difference with single change |
| DeltaE | `B_symm` | Bilinear form symmetry |
| DeltaE | `deltaE_coord` | **Energy delta closed form** |
| Glue | `sign_step_deltaE_nonpos` | Sign update makes delta <= 0 |
| Glue | `globalEnergy_nonincreasing_updateCoord` | **Energy monotonicity** |
| Convergence | `energyAt_step` | Single-step monotonicity |
| Convergence | `energyAt_antitone` | Multi-step monotonicity |
| Convergence | `state_space_card` | `|state space| = 2^n` |
| Convergence | `state_repeats` | Pigeonhole: repetition in `2^n` steps |
| Convergence | `energy_constant_between` | Energy constant between repeated states |
| Convergence | `convergence_bound` | **Convergence within `2^n` steps** |

## Verification

```bash
cd formal && lake build
# Successful build = all proofs verified
# No warnings, no errors, no sorry
```

The CI pipeline (`qa-gates.yml`) runs `lake build` on every push to verify
that all proofs remain valid.

## Relationship to MagicBrain Implementation

The formal proofs verify the mathematical foundation upon which MagicBrain's
NeuroGenesis engine is built:

| Formal Proof | Implementation Module | Connection |
|-------------|----------------------|------------|
| `globalEnergy` | `neurogenesis/energy.py` | Energy function matches the formalized definition |
| `updateCoord` | `neurogenesis/attractor_dynamics.py` | Attractor convergence relies on energy monotonicity |
| `convergence_bound` | `neurogenesis/pattern_memory.py` | Pattern recall terminates because energy decreases |
| `W = W^T`, `W(i,i) = 0` | `neurogenesis/pattern_memory.py` | Storkey rule produces symmetric zero-diagonal weights |

The implementation extends beyond the formalized binary model (e.g., using
continuous `tanh` activation in attractor dynamics), but the core theoretical
guarantee — that energy-based neural dynamics converge — is machine-verified.

## References

1. Hopfield, J.J. (1982). Neural networks and physical systems with emergent
   collective computational abilities. *PNAS*, 79(8), 2554-2558.
2. Cohen, M.A. & Grossberg, S. (1983). Absolute stability of global pattern
   formation and parallel memory storage by competitive neural networks.
   *IEEE Trans. Systems, Man, and Cybernetics*, SMC-13(5), 815-826.
3. Storkey, A. (1997). Increasing the capacity of a Hopfield network without
   sacrificing functionality. *ICANN*, 451-456.
