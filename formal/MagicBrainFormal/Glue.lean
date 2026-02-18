import Mathlib
import MagicBrainFormal.Hopfield
import MagicBrainFormal.DeltaE

open scoped BigOperators
open scoped Matrix

namespace MagicBrainFormal

noncomputable section

theorem sign_step_deltaE_nonpos (h xi : ℝ) (hxi : xi = 1 ∨ xi = -1) :
    - ((if 0 ≤ h then 1 else -1) - xi) * h ≤ 0 := by
  by_cases hh : 0 ≤ h
  · rcases hxi with rfl | rfl <;> simp [hh]; nlinarith
  · have hhlt : h < 0 := lt_of_not_ge hh
    rcases hxi with rfl | rfl <;> simp [hh]; nlinarith [hhlt]

theorem globalEnergy_nonincreasing_updateCoord
    {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ)
    (hsym : W = Matrix.transpose W)
    (hdiag : ∀ i : Fin n, W i i = 0)
    (s : Fin n → Bool) (i0 : Fin n) :
    globalEnergy W (updateCoord W s i0) ≤ globalEnergy W s := by
  classical
  let x : Fin n → ℝ := spin s
  let x' : Fin n → ℝ := spin (updateCoord W s i0)
  have hx : ∀ k : Fin n, k ≠ i0 → x' k = x k := by
    intro k hk
    simpa [x, x'] using (spin_updateCoord_ne (W := W) (s := s) (i := i0) (k := k) hk)
  have hΔ :=
    deltaE_coord (W := W) (hsym := hsym) (hdiag := hdiag) (x := x) (x' := x') (i0 := i0) hx
  have hxi : x i0 = 1 ∨ x i0 = -1 := spin_mem_pm_one (s := s) (i := i0)
  have hx'i0 : x' i0 = if 0 ≤ (W.mulVec x) i0 then 1 else -1 := by
    simpa [x, x', field_eq_mulVec] using (spin_updateCoord_eq_sign (W := W) (s := s) (i0 := i0))
  have hle : globalEnergyVec W x' - globalEnergyVec W x ≤ 0 := by
    have hrhs : - (x' i0 - x i0) * (W.mulVec x) i0 ≤ 0 := by
      simpa [hx'i0] using (sign_step_deltaE_nonpos (h := (W.mulVec x) i0) (xi := x i0) hxi)
    simpa [hΔ] using hrhs
  have : globalEnergyVec W x' ≤ globalEnergyVec W x := by linarith
  simpa [globalEnergy, x, x'] using this

end

end MagicBrainFormal
