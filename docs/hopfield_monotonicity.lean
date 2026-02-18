import Mathlib

open Real

namespace Hopfield

noncomputable section

def energy {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ) : ℝ :=
  (-(1 / 2 : ℝ)) * (∑ i, ∑ j, W i j * s i * s j)

def field {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ) (i : Fin n) : ℝ :=
  ∑ j, W i j * s j

def updateCoord {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ) (i : Fin n) :
    Fin n → ℝ :=
  fun k => if k = i then Real.tanh (field W s i) else s k

theorem energy_nonincreasing_coord
  {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ)
  (hsym : W = Wᵀ)
  (hdiag : ∀ i, W i i = 0)
  (s : Fin n → ℝ) (i : Fin n) :
  energy W (updateCoord W s i) ≤ energy W s := by
  sorry

end Hopfield
