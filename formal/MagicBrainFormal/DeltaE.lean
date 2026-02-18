import Mathlib.Data.Matrix.Mul
import MagicBrainFormal.Hopfield
import MagicBrainFormal.SumLemmas

open scoped BigOperators
open scoped Matrix

namespace MagicBrainFormal

noncomputable section

def B {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ) (u v : Fin n → ℝ) : ℝ :=
  u ⬝ᵥ W.mulVec v

theorem B_symm {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ) (hsym : W = Matrix.transpose W)
    (u v : Fin n → ℝ) : B W u v = B W v u := by
  classical
  -- u ⬝ᵥ (W*ᵥ v) = (u ᵥ* W) ⬝ᵥ v = (Wᵀ*ᵥ u) ⬝ᵥ v = v ⬝ᵥ (W*ᵥ u)
  calc
    B W u v = u ⬝ᵥ W.mulVec v := rfl
    _ = Matrix.vecMul u W ⬝ᵥ v := by
          simpa [B] using (Matrix.dotProduct_mulVec (v := u) (A := W) (w := v))
    _ = (Matrix.transpose W).mulVec u ⬝ᵥ v := by
          -- mulVec_transpose: Wᵀ.mulVec u = vecMul u W
          -- we need it in the opposite direction under dotProduct
          have h := (Matrix.mulVec_transpose (A := W) (x := u)).symm
          exact congrArg (fun t => t ⬝ᵥ v) h
    _ = W.mulVec u ⬝ᵥ v := by
          -- avoid simp recursion on hsym
          have hw : (Matrix.transpose W) = W := by
            exact Eq.symm hsym
          simp [hw]
    _ = v ⬝ᵥ W.mulVec u := by
          simpa using (dotProduct_comm (v := (W.mulVec u)) (w := v))
    _ = B W v u := rfl

theorem globalEnergyVec_eq_B {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ) (x : Fin n → ℝ) :
    globalEnergyVec W x = (-(1 / 2 : ℝ)) * B W x x := by
  rfl

theorem deltaE_coord
    {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ)
    (hsym : W = Matrix.transpose W)
    (hdiag : ∀ i : Fin n, W i i = 0)
    (x x' : Fin n → ℝ) (i0 : Fin n)
    (hx : ∀ k : Fin n, k ≠ i0 → x' k = x k) :
    globalEnergyVec W x' - globalEnergyVec W x =
      - (x' i0 - x i0) * (W.mulVec x) i0 := by
  classical
  let δ : ℝ := x' i0 - x i0
  let d : Fin n → ℝ := fun k => if k = i0 then δ else 0
  have hx' : x' = x + d := by
    funext k
    by_cases hk : k = i0
    · subst hk
      simp [d, δ]
    · have : x' k = x k := hx k hk
      simp [d, hk, this]

  have hdsupp : ∀ k : Fin n, k ≠ i0 → d k = 0 := by
    intro k hk
    simp [d, hk]

  have d_i0 : d i0 = δ := by simp [d]

  have hdW : (W.mulVec d) i0 = W i0 i0 * δ := by
    -- isolate the i0 term in the mulVec sum
    have hsum :
        (W.mulVec d) i0 = (∑ j, W i0 j * d j) := by rfl
    have hsplit :=
      (Finset.sum_erase_add (s := (Finset.univ : Finset (Fin n))) (a := i0)
        (f := fun j => W i0 j * d j) (by simp)).symm
    -- hsplit: sum over univ = term i0 + sum over erase
    have herase : (∑ j ∈ (Finset.univ.erase i0), W i0 j * d j) = 0 := by
      refine Finset.sum_eq_zero ?_
      intro j hj
      have hjne : j ≠ i0 := (Finset.mem_erase.mp hj).1
      simp [hdsupp j hjne]
    calc
      (W.mulVec d) i0
          = ∑ j, W i0 j * d j := by rfl
      _ = (∑ j ∈ (Finset.univ.erase i0), W i0 j * d j) + W i0 i0 * d i0 := by
            set_option linter.unnecessarySimpa false in simpa using hsplit
      _ = W i0 i0 * d i0 + (∑ j ∈ (Finset.univ.erase i0), W i0 j * d j) := by
            ring
      _ = W i0 i0 * δ := by
            simp [d_i0, herase]

  -- Expand (x+d) ⬝ᵥ W*ᵥ (x+d) and cancel x ⬝ᵥ W*ᵥ x.
  have hquad :
      B W (x + d) (x + d) - B W x x =
        (B W x d) + (B W d x) + (B W d d) := by
    have hmul : W.mulVec (x + d) = W.mulVec x + W.mulVec d := by
      simpa using (Matrix.mulVec_add (A := W) (x := x) (y := d))
    calc
      B W (x + d) (x + d) - B W x x
          = ((x + d) ⬝ᵥ W.mulVec (x + d)) - (x ⬝ᵥ W.mulVec x) := rfl
      _ = ((x + d) ⬝ᵥ (W.mulVec x + W.mulVec d)) - (x ⬝ᵥ W.mulVec x) := by
            simp [hmul]
      _ = (((x + d) ⬝ᵥ W.mulVec x) + ((x + d) ⬝ᵥ W.mulVec d)) - (x ⬝ᵥ W.mulVec x) := by
            simp [dotProduct_add]
      _ =
          ((((x ⬝ᵥ W.mulVec x) + (d ⬝ᵥ W.mulVec x))
            + ((x ⬝ᵥ W.mulVec d) + (d ⬝ᵥ W.mulVec d)))
          - (x ⬝ᵥ W.mulVec x)) := by
            simp [add_dotProduct, add_assoc]
      _ = (x ⬝ᵥ W.mulVec d) + (d ⬝ᵥ W.mulVec x) + (d ⬝ᵥ W.mulVec d) := by
            ring
      _ = (B W x d) + (B W d x) + (B W d d) := by
            rfl

  have hBxD : B W x d = B W d x := by
    simpa using (B_symm (W := W) (hsym := hsym) (u := x) (v := d))

  have hBdX : B W d x = δ * (W.mulVec x) i0 := by
    -- dot with singleton-support vector d
    have hsupp : ∀ k : Fin n, k ≠ i0 → d k = 0 := hdsupp
    have := dot_eq_singleton_of_support (v := d) (w := (W.mulVec x)) (a := i0) hsupp
    -- d ⬝ᵥ (W*ᵥ x) = d i0 * (W*ᵥ x) i0 = δ * ...
    simpa [B, d_i0] using this

  have hBdD : B W d d = 0 := by
    have hsupp : ∀ k : Fin n, k ≠ i0 → d k = 0 := hdsupp
    have hd := dot_eq_singleton_of_support (v := d) (w := (W.mulVec d)) (a := i0) hsupp
    -- d ⬝ᵥ (W*ᵥ d) = d i0 * (W*ᵥ d) i0 = δ * (W i0 i0 * δ) = 0 by diag
    have : (W.mulVec d) i0 = W i0 i0 * δ := hdW
    have hdiag0 : W i0 i0 = 0 := hdiag i0
    calc
      B W d d = d ⬝ᵥ W.mulVec d := rfl
      _ = d i0 * (W.mulVec d) i0 := by simpa using hd
      _ = δ * (W i0 i0 * δ) := by simp [d_i0, this]
      _ = 0 := by simp [hdiag0]

  -- now compute ΔE = (-(1/2)) * Δquadratic
  calc
    globalEnergyVec W x' - globalEnergyVec W x
        = (-(1 / 2 : ℝ)) * (B W x' x' - B W x x) := by
            -- expand via definition
            simp [globalEnergyVec_eq_B, sub_eq_add_neg, mul_add]
    _ = (-(1 / 2 : ℝ)) * (B W (x + d) (x + d) - B W x x) := by
            simp [hx']
    _ = (-(1 / 2 : ℝ)) * ((B W x d) + (B W d x) + (B W d d)) := by
            simp [hquad]
    _ = (-(1 / 2 : ℝ)) * ((B W d x) + (B W d x) + 0) := by
            simp [hBxD, hBdD, add_comm]
    _ = - (B W d x) := by
            ring_nf
    _ = - (δ * (W.mulVec x) i0) := by
            simp [hBdX]
    _ = (-δ) * (W.mulVec x) i0 := by
            rw [neg_mul]
    _ = - (x' i0 - x i0) * (W.mulVec x) i0 := by
            simp [δ]

end

end MagicBrainFormal
