import Mathlib
import Mathlib.Data.Matrix.Mul

open scoped BigOperators
open scoped Matrix

namespace MagicBrainFormal

noncomputable section

def sigma (b : Bool) : ℝ :=
  if b then (1 : ℝ) else (-1 : ℝ)

def spin {n : ℕ} (s : Fin n → Bool) : Fin n → ℝ :=
  fun i => sigma (s i)

def field {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) (i : Fin n) : ℝ :=
  ∑ j, W i j * x j

@[simp] lemma field_eq_mulVec {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) (i : Fin n) :
    field W x i = (W.mulVec x) i := by
  rfl

def updateCoord {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → Bool) (i : Fin n) : Fin n → Bool :=
  fun k => if k = i then decide (0 ≤ field W (spin s) i) else s k

def globalEnergyVec {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) : ℝ :=
  (-(1 / 2 : ℝ)) * (x ⬝ᵥ W.mulVec x)

def globalEnergy {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → Bool) : ℝ :=
  globalEnergyVec W (spin s)

def localEnergyAt (h : ℝ) (b : Bool) : ℝ :=
  - sigma b * h

theorem spin_mem_pm_one {n : ℕ} (s : Fin n → Bool) (i : Fin n) :
    spin s i = 1 ∨ spin s i = -1 := by
  cases h : s i <;> simp [spin, sigma, h]

lemma spin_updateCoord_ne {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → Bool) (i k : Fin n) (hk : k ≠ i) :
    spin (updateCoord W s i) k = spin s k := by
  simp [spin, updateCoord, hk]

theorem spin_updateCoord_eq_sign {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → Bool) (i0 : Fin n) :
    spin (updateCoord W s i0) i0 = if 0 ≤ field W (spin s) i0 then 1 else -1 := by
  classical
  by_cases hh : 0 ≤ field W (spin s) i0
  · simp [spin, updateCoord, sigma, hh]
  · simp [spin, updateCoord, sigma, hh]

theorem localEnergyAt_nonincreasing_async_sign (h : ℝ) (b : Bool) :
    localEnergyAt h (decide (0 ≤ h)) ≤ localEnergyAt h b := by
  classical
  by_cases hh : 0 ≤ h
  · cases hb : b <;>
      simp [localEnergyAt, sigma, hh]
  · have hh' : h < 0 := lt_of_not_ge hh
    cases hb : b <;>
      simp [localEnergyAt, sigma, hh, le_of_lt hh']

theorem localEnergy_nonincreasing_async_sign
    {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ)
    (s : Fin n → Bool) (i0 : Fin n) :
    localEnergyAt (field W (spin s) i0) (decide (0 ≤ field W (spin s) i0)) ≤ localEnergyAt (field W (spin s) i0) (s i0) := by
  simpa using localEnergyAt_nonincreasing_async_sign (h := field W (spin s) i0) (b := s i0)

end

end MagicBrainFormal
