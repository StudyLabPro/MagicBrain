import Mathlib.Data.Matrix.Mul

open scoped BigOperators

namespace MagicBrainFormal

noncomputable section

theorem sum_univ_eq_add_sum_erase {ι : Type} [Fintype ι] [DecidableEq ι] {α : Type} [AddCommMonoid α]
    (f : ι → α) (a : ι) :
    (∑ i, f i) = f a + (∑ i ∈ (Finset.univ.erase a), f i) := by
  classical
  have h :=
    (Finset.sum_erase_add (s := (Finset.univ : Finset ι)) (f := f) (a := a) (by simp))
  -- h: (∑ i ∈ univ.erase a, f i) + f a = ∑ i ∈ univ, f i
  calc
    (∑ i, f i) = (∑ i ∈ (Finset.univ : Finset ι), f i) := by simp
    _ = (∑ i ∈ (Finset.univ.erase a), f i) + f a := by simpa using h.symm
    _ = f a + (∑ i ∈ (Finset.univ.erase a), f i) := by
          simpa [add_assoc, add_comm, add_left_comm]

theorem dot_eq_add_sum_erase {ι : Type} [Fintype ι] [DecidableEq ι] {R : Type} [NonUnitalNonAssocSemiring R]
    (v w : ι → R) (a : ι) :
    (dotProduct v w) = v a * w a + (∑ i ∈ (Finset.univ.erase a), v i * w i) := by
  classical
  have h :=
    (Finset.sum_erase_add (s := (Finset.univ : Finset ι)) (f := fun i => v i * w i) (a := a) (by simp))
  -- h: (∑ i ∈ univ.erase a, v i*w i) + v a*w a = ∑ i ∈ univ, v i*w i
  calc
    dotProduct v w = (∑ i ∈ (Finset.univ : Finset ι), v i * w i) := by
      simp [dotProduct]
    _ = (∑ i ∈ (Finset.univ.erase a), v i * w i) + v a * w a := by simpa using h.symm
    _ = v a * w a + (∑ i ∈ (Finset.univ.erase a), v i * w i) := by
          simpa [add_assoc, add_comm, add_left_comm]

theorem dot_eq_singleton_of_support {ι : Type} [Fintype ι] [DecidableEq ι] {R : Type} [NonUnitalNonAssocSemiring R]
    (v w : ι → R) (a : ι) (hzero : ∀ i : ι, i ≠ a → v i = 0) :
    (dotProduct v w) = v a * w a := by
  classical
  have hsplit := dot_eq_add_sum_erase (v := v) (w := w) (a := a)
  -- erase sum is zero due to support restriction
  have herase : (∑ i ∈ (Finset.univ.erase a), v i * w i) = 0 := by
    refine Finset.sum_eq_zero ?_
    intro i hi
    have hine : i ≠ a := (Finset.mem_erase.mp hi).1
    simp [hzero i hine]
  calc
    (dotProduct v w) = v a * w a + (∑ i ∈ (Finset.univ.erase a), v i * w i) := hsplit
    _ = v a * w a := by simpa [herase]

theorem dot_sub_eq_singleton {ι : Type} [Fintype ι] [DecidableEq ι] {R : Type} [Ring R]
    (v v' w : ι → R) (a : ι) (heq : ∀ i : ι, i ≠ a → v' i = v i) :
    (dotProduct v' w) - (dotProduct v w) = (v' a - v a) * w a := by
  classical
  have hsplit' := dot_eq_add_sum_erase (v := v') (w := w) (a := a)
  have hsplit := dot_eq_add_sum_erase (v := v) (w := w) (a := a)
  have herase :
      (∑ i ∈ (Finset.univ.erase a), v' i * w i) = (∑ i ∈ (Finset.univ.erase a), v i * w i) := by
    refine Finset.sum_congr rfl ?_
    intro i hi
    have hine : i ≠ a := (Finset.mem_erase.mp hi).1
    simp [heq i hine]
  calc
    (dotProduct v' w) - (dotProduct v w)
        =
        (v' a * w a + (∑ i ∈ (Finset.univ.erase a), v' i * w i))
        - (v a * w a + (∑ i ∈ (Finset.univ.erase a), v i * w i)) := by
          rw [hsplit', hsplit]
    _ =
        (v' a * w a - v a * w a) +
          ((∑ i ∈ (Finset.univ.erase a), v' i * w i) - (∑ i ∈ (Finset.univ.erase a), v i * w i)) := by
          simpa using
            (add_sub_add_comm
              (v' a * w a)
              (∑ i ∈ (Finset.univ.erase a), v' i * w i)
              (v a * w a)
              (∑ i ∈ (Finset.univ.erase a), v i * w i))
    _ = (v' a - v a) * w a := by
          simp [herase, sub_eq_add_neg, mul_add, add_mul, mul_assoc, mul_left_comm, mul_comm]

end

end MagicBrainFormal
