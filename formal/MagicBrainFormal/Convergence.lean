/-
  MagicBrainFormal.Convergence — Finite-state convergence of Hopfield networks

  Given the energy monotonicity theorem (Glue.lean), we prove that:
  1. The energy sequence is non-increasing over any update schedule
  2. In any sequence of 2^n updates, some state repeats (pigeonhole)
  3. Energy is constant between repeated states

  These results establish that the Hopfield network has bounded
  dynamics: at most 2^n - 1 strict energy decreases are possible.
-/
import Mathlib
import MagicBrainFormal.Glue

open scoped BigOperators Matrix

namespace MagicBrainFormal

noncomputable section

-- ====================================================================
-- Definitions
-- ====================================================================

/-- State after t asynchronous updates following a schedule. -/
def updateSeq {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ)
    (s₀ : Fin n → Bool) (schedule : ℕ → Fin n) : ℕ → (Fin n → Bool)
  | 0 => s₀
  | t + 1 => updateCoord W (updateSeq W s₀ schedule t) (schedule t)

/-- Energy at step t of the update sequence. -/
def energyAt {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ)
    (s₀ : Fin n → Bool) (schedule : ℕ → Fin n) (t : ℕ) : ℝ :=
  globalEnergy W (updateSeq W s₀ schedule t)

/-- A state is a fixed point if no single-coordinate update changes it. -/
def isFixedPoint {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → Bool) : Prop :=
  ∀ i : Fin n, updateCoord W s i = s

-- ====================================================================
-- Energy monotonicity
-- ====================================================================

/-- Energy is non-increasing at each step (from Glue.lean). -/
theorem energyAt_step {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ)
    (hsym : W = Matrix.transpose W) (hdiag : ∀ i : Fin n, W i i = 0)
    (s₀ : Fin n → Bool) (schedule : ℕ → Fin n) (t : ℕ) :
    energyAt W s₀ schedule (t + 1) ≤ energyAt W s₀ schedule t :=
  globalEnergy_nonincreasing_updateCoord W hsym hdiag _ _

/-- Energy is non-increasing over multiple steps. -/
theorem energyAt_antitone {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ)
    (hsym : W = Matrix.transpose W) (hdiag : ∀ i : Fin n, W i i = 0)
    (s₀ : Fin n → Bool) (schedule : ℕ → Fin n)
    {a b : ℕ} (hab : a ≤ b) :
    energyAt W s₀ schedule b ≤ energyAt W s₀ schedule a := by
  induction b with
  | zero =>
    have : a = 0 := Nat.le_zero.mp hab
    exact this ▸ le_refl _
  | succ b ih =>
    by_cases hab' : a ≤ b
    · exact le_trans (energyAt_step W hsym hdiag s₀ schedule b) (ih hab')
    · have : a = b + 1 := by omega
      exact this ▸ le_refl _

-- ====================================================================
-- Finite state space
-- ====================================================================

/-- The state space Fin n → Bool has exactly 2^n elements. -/
theorem state_space_card (n : ℕ) : Fintype.card (Fin n → Bool) = 2 ^ n := by
  simp [Fintype.card_bool, Fintype.card_fin]

-- ====================================================================
-- Pigeonhole: state repetition bound
-- ====================================================================

/-- In any orbit of length > 2^n, some state must repeat. -/
theorem state_repeats {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ)
    (s₀ : Fin n → Bool) (schedule : ℕ → Fin n) :
    ∃ i j : ℕ, i < j ∧ j ≤ 2 ^ n ∧
      updateSeq W s₀ schedule i = updateSeq W s₀ schedule j := by
  have hcard : Fintype.card (Fin n → Bool) < Fintype.card (Fin (2 ^ n + 1)) := by
    simp [Fintype.card_bool, Fintype.card_fin]
  obtain ⟨a, b, hab, heq⟩ := Fintype.exists_ne_map_eq_of_card_lt
    (fun (t : Fin (2 ^ n + 1)) => updateSeq W s₀ schedule t.val) hcard
  rcases lt_or_gt_of_ne hab with h | h
  · exact ⟨a.val, b.val, h, Nat.lt_succ_iff.mp b.isLt, heq⟩
  · exact ⟨b.val, a.val, h, Nat.lt_succ_iff.mp a.isLt, heq.symm⟩

-- ====================================================================
-- Energy stabilization between repeated states
-- ====================================================================

/-- If two states in the orbit are equal, energy is constant between them. -/
theorem energy_constant_between {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ)
    (hsym : W = Matrix.transpose W) (hdiag : ∀ i : Fin n, W i i = 0)
    (s₀ : Fin n → Bool) (schedule : ℕ → Fin n)
    {i j : ℕ} (_hij : i ≤ j)
    (hrepeat : updateSeq W s₀ schedule i = updateSeq W s₀ schedule j)
    {t : ℕ} (hit : i ≤ t) (htj : t ≤ j) :
    energyAt W s₀ schedule t = energyAt W s₀ schedule i := by
  have h1 := energyAt_antitone W hsym hdiag s₀ schedule hit
  have h2 := energyAt_antitone W hsym hdiag s₀ schedule htj
  have h3 : energyAt W s₀ schedule j = energyAt W s₀ schedule i := by
    unfold energyAt; exact congr_arg (globalEnergy W) hrepeat.symm
  linarith

-- ====================================================================
-- Main convergence theorem
-- ====================================================================

/-- Convergence bound: within 2^n steps, the energy stabilizes for at least
    one interval. Specifically, there exist i < j ≤ 2^n such that for all
    t in [i, j], the energy equals the energy at step i. -/
theorem convergence_bound {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ)
    (hsym : W = Matrix.transpose W) (hdiag : ∀ i : Fin n, W i i = 0)
    (s₀ : Fin n → Bool) (schedule : ℕ → Fin n) :
    ∃ i j : ℕ, i < j ∧ j ≤ 2 ^ n ∧
      ∀ t, i ≤ t → t ≤ j →
        energyAt W s₀ schedule t = energyAt W s₀ schedule i := by
  obtain ⟨i, j, hij, hjbound, hrepeat⟩ := state_repeats W s₀ schedule
  exact ⟨i, j, hij, hjbound, fun t hit htj =>
    energy_constant_between W hsym hdiag s₀ schedule hij.le hrepeat hit htj⟩

end

end MagicBrainFormal
