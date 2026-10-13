import Mathlib

set_option linter.style.header false

/-!
# Compressed arc-flow graph for one capacitated instance type

Formal model of `SpaceTimeFormulation(compressed_flow=True)` restricted to a
single capacitated type `m` (flow conservation decomposes by type).

* Time slots `0 … H`; capacity `N`; `R ≤ N` running jobs, running job `r`
  frees its unit at `rend r ≤ H` (`Instance.rj_arrivals`).
* Jobs `Fin n` with release slot `rel j` (`T[j,m].start`) and processing time
  `p j ≥ 1` (`p_occ[j,m]`). `J` is the set of jobs actually placed on this
  type and `s j` their start slots.
* The graph has event set `E` and node set `Nd` (`_graph.cpp`: events are
  0, H, every release and every running completion; nodes are every
  normal-pattern time plus 0 and H) and one idle ("loss") arc per node `q < H`
  to the next event `nxt q`. `Conserve` is the flow row `flow[m,v]`.

Main results
* `Graph.sound` — a compressed flow respects capacity in every slot;
  `Graph.soundR` the same for real-valued idle flow (the code's continuous `y`).
* `Graph.complete` — a capacity-feasible *left-shifted* schedule has a flow.
* `TypeInst.exists_leftShift` — every feasible schedule left-shifts, starts
  only moving earlier.
* `TypeInst.leftShifted_mem_pattern` — left-shifted starts are normal patterns.
* `Graph.remove_then_leftShift` — removing jobs from a compressed-feasible
  schedule and left-shifting the rest is compressed-feasible, every start in
  the pattern set and no later than before.

Removal *without* the left shift is false. Capacity 1, H = 10, jobs A (p = 3)
and B (p = 2) both released at 0: A@0, B@3 is a compressed flow (A's
completion hands its unit to B at node 3). Move A elsewhere and B@3 is
infeasible: the idle unit at 0 jumps along the loss arc 0 → 10 (the next
event) and never reaches node 3. Left-shifting B to 0 restores feasibility.
-/

open Finset

namespace Fedhpc

/-- One capacitated instance type and the jobs that may run on it. -/
structure TypeInst where
  H : ℕ
  N : ℕ
  n : ℕ
  rel : Fin n → ℕ
  p : Fin n → ℕ
  R : ℕ
  rend : Fin R → ℕ
  hH : 0 < H
  hp : ∀ j, 0 < p j
  hrend : ∀ r, rend r ≤ H
  hRN : R ≤ N

namespace TypeInst

variable (I : TypeInst)

/-- Running jobs still occupying a unit during slot `[u, u+1)`. -/
def run (u : ℕ) : ℕ := ∑ r : Fin I.R, if u < I.rend r then 1 else 0

/-- Running jobs completing at time `v` (ghost arrivals). -/
def rc (v : ℕ) : ℕ := ∑ r : Fin I.R, if I.rend r = v then 1 else 0

/-- Jobs of `J` occupying a unit during slot `[u, u+1)`. -/
def busy (J : Finset (Fin I.n)) (s : Fin I.n → ℕ) (u : ℕ) : ℕ :=
  ∑ j ∈ J, if s j ≤ u ∧ u < s j + I.p j then 1 else 0

def starts (J : Finset (Fin I.n)) (s : Fin I.n → ℕ) (v : ℕ) : ℕ :=
  ∑ j ∈ J, if s j = v then 1 else 0

def ends (J : Finset (Fin I.n)) (s : Fin I.n → ℕ) (v : ℕ) : ℕ :=
  ∑ j ∈ J, if s j + I.p j = v then 1 else 0

/-- Every placed job starts no earlier than its release and ends by `H`. -/
def Valid (J : Finset (Fin I.n)) (s : Fin I.n → ℕ) : Prop :=
  ∀ j ∈ J, I.rel j ≤ s j ∧ s j + I.p j ≤ I.H

/-- Capacity respected in every slot (the full time-grid model). -/
def Feasible (J : Finset (Fin I.n)) (s : Fin I.n → ℕ) : Prop :=
  I.Valid J s ∧ ∀ u, u < I.H → I.run u + I.busy J s u ≤ I.N

/-- No job can start one slot earlier: the slot before it is full. -/
def LeftShifted (J : Finset (Fin I.n)) (s : Fin I.n → ℕ) : Prop :=
  ∀ j ∈ J, I.rel j < s j → I.run (s j - 1) + I.busy J s (s j - 1) = I.N

/-! ### Slot-to-slot bookkeeping -/

lemma run_succ (u : ℕ) : I.run u = I.run (u + 1) + I.rc (u + 1) := by
  unfold run rc
  rw [← sum_add_distrib]
  refine sum_congr rfl (fun r _ => ?_)
  split_ifs <;> omega

lemma busy_succ (J : Finset (Fin I.n)) (s : Fin I.n → ℕ) (u : ℕ) :
    I.busy J s u + I.starts J s (u + 1) = I.busy J s (u + 1) + I.ends J s (u + 1) := by
  unfold busy starts ends
  rw [← sum_add_distrib, ← sum_add_distrib]
  refine sum_congr rfl (fun j _ => ?_)
  have := I.hp j
  split_ifs <;> omega

lemma run_zero : I.run 0 + I.rc 0 = I.R := by
  unfold run rc
  rw [← sum_add_distrib]
  trans ∑ _r : Fin I.R, (1 : ℕ)
  · exact sum_congr rfl (fun r _ => by split_ifs <;> omega)
  · simp

lemma busy_zero (J : Finset (Fin I.n)) (s : Fin I.n → ℕ) :
    I.busy J s 0 = I.starts J s 0 := by
  unfold busy starts
  refine sum_congr rfl (fun j _ => ?_)
  have := I.hp j
  split_ifs <;> omega

lemma ends_zero (J : Finset (Fin I.n)) (s : Fin I.n → ℕ) : I.ends J s 0 = 0 := by
  unfold ends
  refine sum_eq_zero (fun j _ => ?_)
  have := I.hp j
  split_ifs <;> omega

lemma run_last : I.run (I.H - 1) = I.rc I.H := by
  unfold run rc
  refine sum_congr rfl (fun r _ => ?_)
  have := I.hrend r
  have := I.hH
  split_ifs <;> omega

lemma busy_last {J : Finset (Fin I.n)} {s : Fin I.n → ℕ} (hv : I.Valid J s) :
    I.busy J s (I.H - 1) = I.ends J s I.H := by
  unfold busy ends
  refine sum_congr rfl (fun j hj => ?_)
  have := I.hp j
  have := hv j hj
  split_ifs <;> omega

lemma starts_last {J : Finset (Fin I.n)} {s : Fin I.n → ℕ} (hv : I.Valid J s) :
    I.starts J s I.H = 0 := by
  unfold starts
  refine sum_eq_zero (fun j hj => ?_)
  have := I.hp j
  have := hv j hj
  split_ifs <;> omega

lemma busy_mono {J J' : Finset (Fin I.n)} (h : J' ⊆ J) (s : Fin I.n → ℕ) (u : ℕ) :
    I.busy J' s u ≤ I.busy J s u :=
  sum_le_sum_of_subset h

lemma Feasible.mono {J J' : Finset (Fin I.n)} {s : Fin I.n → ℕ} (hf : I.Feasible J s)
    (h : J' ⊆ J) : I.Feasible J' s :=
  ⟨fun j hj => hf.1 j (h hj), fun u hu => (Nat.add_le_add_left (I.busy_mono h s u) _).trans
    (hf.2 u hu)⟩

lemma exists_of_starts {J : Finset (Fin I.n)} {s : Fin I.n → ℕ} {v : ℕ}
    (h : I.starts J s v ≠ 0) : ∃ j ∈ J, s j = v := by
  obtain ⟨j, hj, hne⟩ := exists_ne_zero_of_sum_ne_zero h
  exact ⟨j, hj, by by_contra hc; simp [hc] at hne⟩

lemma exists_of_ends {J : Finset (Fin I.n)} {s : Fin I.n → ℕ} {v : ℕ}
    (h : I.ends J s v ≠ 0) : ∃ j ∈ J, s j + I.p j = v := by
  obtain ⟨j, hj, hne⟩ := exists_ne_zero_of_sum_ne_zero h
  exact ⟨j, hj, by by_contra hc; simp [hc] at hne⟩

lemma exists_of_rc {v : ℕ} (h : I.rc v ≠ 0) : ∃ r, I.rend r = v := by
  obtain ⟨r, _, hne⟩ := exists_ne_zero_of_sum_ne_zero h
  exact ⟨r, by by_contra hc; simp [hc] at hne⟩

lemma rc_eq_zero {v : ℕ} (h : ∀ r, I.rend r ≠ v) : I.rc v = 0 :=
  sum_eq_zero (fun r _ => by simp [h r])

lemma starts_eq_zero {J : Finset (Fin I.n)} {s : Fin I.n → ℕ} {v : ℕ}
    (h : ∀ j ∈ J, s j ≠ v) : I.starts J s v = 0 :=
  sum_eq_zero (fun j hj => by simp [h j hj])

lemma ends_eq_zero {J : Finset (Fin I.n)} {s : Fin I.n → ℕ} {v : ℕ}
    (h : ∀ j ∈ J, s j + I.p j ≠ v) : I.ends J s v = 0 :=
  sum_eq_zero (fun j hj => by simp [h j hj])

/-! ### Left shifting -/

/-- Moving `j` one slot earlier adds occupancy only at slot `s j - 1`. -/
lemma busy_update_le {J : Finset (Fin I.n)} {s : Fin I.n → ℕ} {j : Fin I.n}
    (hj : j ∈ J) (hs : 0 < s j) (u : ℕ) :
    I.busy J (Function.update s j (s j - 1)) u ≤
      I.busy J s u + if u = s j - 1 then 1 else 0 := by
  have key : I.busy J (Function.update s j (s j - 1)) u ≤
      ∑ k ∈ J, ((if s k ≤ u ∧ u < s k + I.p k then 1 else 0) +
        if k = j then (if u = s j - 1 then 1 else 0) else 0) := by
    unfold busy
    refine sum_le_sum (fun k _ => ?_)
    by_cases hk : k = j
    · subst hk
      simp only [Function.update_self, ite_true]
      split_ifs <;> omega
    · simp only [Function.update_of_ne hk, hk, ite_false, add_zero, le_refl]
  rw [sum_add_distrib, sum_ite_eq' J j] at key
  simpa [busy, hj] using key

theorem exists_leftShift {J : Finset (Fin I.n)} :
    ∀ s : Fin I.n → ℕ, I.Feasible J s →
      ∃ s', I.Feasible J s' ∧ I.LeftShifted J s' ∧ ∀ j ∈ J, I.rel j ≤ s' j ∧ s' j ≤ s j := by
  intro s
  induction hsum : ∑ j ∈ J, s j using Nat.strong_induction_on generalizing s with
  | _ total ih =>
  intro hf
  by_cases hl : I.LeftShifted J s
  · exact ⟨s, hf, hl, fun j hj => ⟨(hf.1 j hj).1, le_rfl⟩⟩
  · simp only [LeftShifted, not_forall] at hl
    obtain ⟨j, hj, hrel, hne⟩ := hl
    have hs0 : 0 < s j := by omega
    set s' := Function.update s j (s j - 1) with hs'
    have hslot : I.run (s j - 1) + I.busy J s (s j - 1) < I.N := by
      have := hf.2 (s j - 1) (by have := (hf.1 j hj).2; have := I.hp j; omega)
      omega
    have hf' : I.Feasible J s' := by
      refine ⟨fun k hk => ?_, fun u hu => ?_⟩
      · by_cases hkj : k = j
        · subst hkj
          simp only [hs', Function.update_self]
          have := hf.1 k hk
          omega
        · simpa [hs', Function.update_of_ne hkj] using hf.1 k hk
      · have hb := I.busy_update_le hj hs0 u
        rw [← hs'] at hb
        have hcap := hf.2 u hu
        split_ifs at hb with hu'
        · subst hu'; omega
        · omega
    have hlt : ∑ k ∈ J, s' k < total := by
      rw [← hsum]
      refine sum_lt_sum (fun k _ => ?_) ⟨j, hj, ?_⟩
      · by_cases hkj : k = j
        · subst hkj; simp [hs']
        · simp [hs', Function.update_of_ne hkj]
      · simp [hs']; omega
    obtain ⟨s'', hf'', hl'', hle⟩ := ih _ hlt s' rfl hf'
    refine ⟨s'', hf'', hl'', fun k hk => ⟨(hle k hk).1, (hle k hk).2.trans ?_⟩⟩
    by_cases hkj : k = j
    · subst hkj; simp [hs']
    · simp [hs', Function.update_of_ne hkj]

/-! ### Normal patterns -/

/-- In a left-shifted schedule every start is a normal pattern: the least set
containing releases and running completions, closed under `q ↦ q + p j` for
`q ≥ rel j` (`_graph.cpp`'s closure, which groups jobs by `p` and uses the
smallest release, contains this set). -/
theorem leftShifted_mem_pattern (P : ℕ → Prop)
    (hrelP : ∀ j, P (I.rel j)) (hrendP : ∀ r, P (I.rend r))
    (hclos : ∀ j q, P q → I.rel j ≤ q → q + I.p j ≤ I.H → P (q + I.p j))
    {J : Finset (Fin I.n)} {s : Fin I.n → ℕ}
    (hf : I.Feasible J s) (hl : I.LeftShifted J s) :
    ∀ j ∈ J, P (s j) := by
  suffices h : ∀ t, ∀ j ∈ J, s j = t → P t from fun j hj => h _ j hj rfl
  intro t
  induction t using Nat.strong_induction_on with
  | _ t ih =>
  intro j hj hjt
  subst hjt
  have hv := hf.1 j hj
  rcases Nat.eq_or_lt_of_le hv.1 with heq | hlt
  · rw [← heq]; exact hrelP j
  · have hfull := hl j hj hlt
    have hp := I.hp j
    have h1 : s j - 1 + 1 = s j := by omega
    have hr := I.run_succ (s j - 1)
    have hb := I.busy_succ J s (s j - 1)
    rw [h1] at hr hb
    have hcap := hf.2 (s j) (by omega)
    have hst : I.starts J s (s j) ≠ 0 := by
      have h1 : (if s j = s j then 1 else 0) ≤ I.starts J s (s j) :=
        single_le_sum (f := fun k => if s k = s j then 1 else 0) (fun _ _ => Nat.zero_le _) hj
      simp only [ite_true] at h1
      omega
    by_cases hrc : I.rc (s j) = 0
    · have hend : I.ends J s (s j) ≠ 0 := by omega
      obtain ⟨k, hk, hke⟩ := I.exists_of_ends hend
      have hkv := hf.1 k hk
      have := I.hp k
      rw [← hke]
      exact hclos k (s k) (ih (s k) (by omega) k hk rfl) hkv.1 hkv.2
    · obtain ⟨r, hr⟩ := I.exists_of_rc hrc
      rw [← hr]; exact hrendP r

end TypeInst

/-- The compressed graph of one type (`_graph.cpp`, `compressed_flow=True`). -/
structure Graph (I : TypeInst) where
  E : Finset ℕ
  Nd : Finset ℕ
  nxt : ℕ → ℕ
  hE : E ⊆ Nd
  h0 : 0 ∈ E
  hHE : I.H ∈ E
  hrelE : ∀ j, I.rel j ∈ E
  hrendE : ∀ r, I.rend r ∈ E
  hNd : ∀ v ∈ Nd, v ≤ I.H
  hnxt_lt : ∀ q, q < I.H → q < nxt q
  hnxt_E : ∀ q, q < I.H → nxt q ∈ E
  hnxt_min : ∀ q, q < I.H → ∀ e ∈ E, q < e → nxt q ≤ e

namespace Graph

variable {I : TypeInst} (G : Graph I)

/-- Tails of loss arcs: every node before `H`. -/
def A : Finset ℕ := G.Nd.filter (· < I.H)

/-- Idle flow entering `v` through loss arcs. -/
def inLoss (ℓ : ℕ → ℕ) (v : ℕ) : ℕ := ∑ q ∈ G.A, if G.nxt q = v then ℓ q else 0

/-- Idle flow on loss arcs spanning slot `[u, u+1)`. -/
def arcIdle (ℓ : ℕ → ℕ) (u : ℕ) : ℕ := ∑ q ∈ G.A, if q ≤ u ∧ u < G.nxt q then ℓ q else 0

/-- Flow row `flow[m,v]` of `SpaceTimeFormulation.build(compressed_flow=True)`:
`lhs_wait + arriving = rhs_wait + starting`. -/
def Conserve (J : Finset (Fin I.n)) (s : Fin I.n → ℕ) (ℓ : ℕ → ℕ) (v : ℕ) : Prop :=
  (if v = 0 then I.N - I.R else G.inLoss ℓ v) + I.ends J s v + I.rc v =
    (if v = I.H then I.N else ℓ v) + I.starts J s v

/-- Every job arc of the schedule exists in the graph. -/
def OnNodes (J : Finset (Fin I.n)) (s : Fin I.n → ℕ) : Prop :=
  ∀ j ∈ J, s j ∈ G.Nd ∧ s j + I.p j ∈ G.Nd

/-- The schedule extends to an integral flow of the compressed graph. -/
def FlowFeasible (J : Finset (Fin I.n)) (s : Fin I.n → ℕ) : Prop :=
  I.Valid J s ∧ G.OnNodes J s ∧ ∃ ℓ : ℕ → ℕ, ∀ v ∈ G.Nd, G.Conserve J s ℓ v

lemma mem_A {q : ℕ} : q ∈ G.A ↔ q ∈ G.Nd ∧ q < I.H := by simp [A]

lemma lt_nxt {q : ℕ} (hq : q ∈ G.A) : q < G.nxt q := G.hnxt_lt q ((G.mem_A).1 hq).2

lemma arc_succ (ℓ : ℕ → ℕ) (u : ℕ) (hu : u + 1 < I.H) :
    G.arcIdle ℓ u + (if u + 1 ∈ G.Nd then ℓ (u + 1) else 0) =
      G.arcIdle ℓ (u + 1) + G.inLoss ℓ (u + 1) := by
  have hpt : (if u + 1 ∈ G.Nd then ℓ (u + 1) else 0) =
      ∑ q ∈ G.A, if q = u + 1 then ℓ q else 0 := by
    rw [sum_ite_eq' G.A (u + 1) ℓ]
    simp [G.mem_A, hu]
  rw [hpt]
  unfold arcIdle inLoss
  rw [← sum_add_distrib, ← sum_add_distrib]
  refine sum_congr rfl (fun q hq => ?_)
  have := G.lt_nxt hq
  split_ifs <;> omega

lemma arcIdle_zero (ℓ : ℕ → ℕ) : G.arcIdle ℓ 0 = ℓ 0 := by
  have h0A : 0 ∈ G.A := (G.mem_A).2 ⟨G.hE G.h0, I.hH⟩
  have : G.arcIdle ℓ 0 = ∑ q ∈ G.A, if q = 0 then ℓ q else 0 := by
    unfold arcIdle
    refine sum_congr rfl (fun q hq => ?_)
    have := G.lt_nxt hq
    split_ifs <;> omega
  rw [this, sum_ite_eq' G.A 0 ℓ, ite_eq_left h0A]

lemma inLoss_of_not_mem_E (ℓ : ℕ → ℕ) {v : ℕ} (hv : v ∉ G.E) : G.inLoss ℓ v = 0 := by
  refine sum_eq_zero (fun q hq => ?_)
  have := G.hnxt_E q ((G.mem_A).1 hq).2
  rw [ite_eq_right]
  rintro rfl
  exact hv this

lemma inLoss_of_mem_E (ℓ : ℕ → ℕ) {v : ℕ} (hv : v ∈ G.E) (hv0 : 0 < v) :
    G.inLoss ℓ v = G.arcIdle ℓ (v - 1) := by
  refine sum_congr rfl (fun q hq => ?_)
  have hlt := G.lt_nxt hq
  have hmin := G.hnxt_min q ((G.mem_A).1 hq).2 v hv
  split_ifs with h1 h2 h2 <;> first | rfl | omega

lemma not_mem_Nd_zero {J : Finset (Fin I.n)} {s : Fin I.n → ℕ} (hon : G.OnNodes J s)
    (ℓ : ℕ → ℕ) {v : ℕ} (hv : v ∉ G.Nd) :
    I.starts J s v = 0 ∧ I.ends J s v = 0 ∧ I.rc v = 0 ∧ G.inLoss ℓ v = 0 :=
  ⟨I.starts_eq_zero (fun j hj h => hv (h ▸ (hon j hj).1)),
   I.ends_eq_zero (fun j hj h => hv (h ▸ (hon j hj).2)),
   I.rc_eq_zero (fun r h => hv (h ▸ G.hE (G.hrendE r))),
   G.inLoss_of_not_mem_E ℓ (fun h => hv (G.hE h))⟩

/-! ### Soundness: a compressed flow respects capacity -/

theorem invariant {J : Finset (Fin I.n)} {s : Fin I.n → ℕ} {ℓ : ℕ → ℕ}
    (hon : G.OnNodes J s) (hc : ∀ v ∈ G.Nd, G.Conserve J s ℓ v) :
    ∀ u, u < I.H → I.run u + I.busy J s u + G.arcIdle ℓ u = I.N := by
  intro u
  induction u with
  | zero =>
    intro _
    have c0 := hc 0 (G.hE G.h0)
    unfold Conserve at c0
    rw [ite_eq_left rfl, ite_eq_right (Nat.ne_of_lt I.hH), I.ends_zero] at c0
    have := I.run_zero
    have := I.busy_zero J s
    have := G.arcIdle_zero ℓ
    have := I.hRN
    omega
  | succ u ih =>
    intro hu
    have ih := ih (by omega)
    have hr := I.run_succ u
    have hb := I.busy_succ J s u
    have ha := G.arc_succ ℓ u hu
    by_cases hn : u + 1 ∈ G.Nd
    · have c := hc (u + 1) hn
      unfold Conserve at c
      rw [ite_eq_right (by omega), ite_eq_right (by omega)] at c
      rw [ite_eq_left hn] at ha
      omega
    · obtain ⟨h1, h2, h3, h4⟩ := G.not_mem_Nd_zero hon ℓ hn
      rw [ite_eq_right hn] at ha
      omega

theorem sound {J : Finset (Fin I.n)} {s : Fin I.n → ℕ} (h : G.FlowFeasible J s) :
    I.Feasible J s := by
  obtain ⟨hv, hon, ℓ, hc⟩ := h
  exact ⟨hv, fun u hu => by have := G.invariant hon hc u hu; omega⟩

/-! ### Soundness with continuous idle flow

The code declares the compressed loss variables `y` continuous. With binary
`x`, any real-valued flow satisfying the rows still respects capacity. -/

/-- Real-valued idle flow on loss arcs spanning slot `[u, u+1)`. -/
def arcIdleR (ℓ : ℕ → ℝ) (u : ℕ) : ℝ := ∑ q ∈ G.A, if q ≤ u ∧ u < G.nxt q then ℓ q else 0

/-- Real-valued idle flow entering `v`. -/
def inLossR (ℓ : ℕ → ℝ) (v : ℕ) : ℝ := ∑ q ∈ G.A, if G.nxt q = v then ℓ q else 0

/-- `Conserve` with a real-valued (continuous) idle flow. -/
def ConserveR (J : Finset (Fin I.n)) (s : Fin I.n → ℕ) (ℓ : ℕ → ℝ) (v : ℕ) : Prop :=
  (if v = 0 then ((I.N - I.R : ℕ) : ℝ) else G.inLossR ℓ v) + (I.ends J s v : ℝ) + (I.rc v : ℝ) =
    (if v = I.H then (I.N : ℝ) else ℓ v) + (I.starts J s v : ℝ)

lemma arc_succR (ℓ : ℕ → ℝ) (u : ℕ) (hu : u + 1 < I.H) :
    G.arcIdleR ℓ u + (if u + 1 ∈ G.Nd then ℓ (u + 1) else 0) =
      G.arcIdleR ℓ (u + 1) + G.inLossR ℓ (u + 1) := by
  have hpt : (if u + 1 ∈ G.Nd then ℓ (u + 1) else 0) =
      ∑ q ∈ G.A, if q = u + 1 then ℓ q else 0 := by
    rw [sum_ite_eq' G.A (u + 1) ℓ]
    simp [G.mem_A, hu]
  rw [hpt]
  unfold arcIdleR inLossR
  rw [← sum_add_distrib, ← sum_add_distrib]
  refine sum_congr rfl (fun q hq => ?_)
  have := G.lt_nxt hq
  split_ifs <;> first | (exfalso; omega) | ring

lemma arcIdleR_zero (ℓ : ℕ → ℝ) : G.arcIdleR ℓ 0 = ℓ 0 := by
  have h0A : 0 ∈ G.A := (G.mem_A).2 ⟨G.hE G.h0, I.hH⟩
  have : G.arcIdleR ℓ 0 = ∑ q ∈ G.A, if q = 0 then ℓ q else 0 := by
    unfold arcIdleR
    refine sum_congr rfl (fun q hq => ?_)
    have := G.lt_nxt hq
    split_ifs <;> first | (exfalso; omega) | rfl
  rw [this, sum_ite_eq' G.A 0 ℓ, ite_eq_left h0A]

lemma inLossR_of_not_mem_Nd (ℓ : ℕ → ℝ) {v : ℕ} (hv : v ∉ G.Nd) : G.inLossR ℓ v = 0 := by
  refine sum_eq_zero (fun q hq => ?_)
  have := G.hnxt_E q ((G.mem_A).1 hq).2
  rw [ite_eq_right]
  rintro rfl
  exact hv (G.hE this)

theorem invariantR {J : Finset (Fin I.n)} {s : Fin I.n → ℕ} {ℓ : ℕ → ℝ}
    (hon : G.OnNodes J s) (hc : ∀ v ∈ G.Nd, G.ConserveR J s ℓ v) :
    ∀ u, u < I.H → (I.run u : ℝ) + I.busy J s u + G.arcIdleR ℓ u = I.N := by
  intro u
  induction u with
  | zero =>
    intro _
    have c0 := hc 0 (G.hE G.h0)
    unfold ConserveR at c0
    rw [ite_eq_left rfl, ite_eq_right (Nat.ne_of_lt I.hH), I.ends_zero,
      Nat.cast_sub I.hRN] at c0
    have h1 : (I.run 0 : ℝ) + I.rc 0 = I.R := by exact_mod_cast I.run_zero
    have h2 : (I.busy J s 0 : ℝ) = I.starts J s 0 := by exact_mod_cast I.busy_zero J s
    rw [G.arcIdleR_zero]
    push_cast at c0
    linarith
  | succ u ih =>
    intro hu
    have ih := ih (by omega)
    have hr : (I.run u : ℝ) = I.run (u + 1) + I.rc (u + 1) := by exact_mod_cast I.run_succ u
    have hb : (I.busy J s u : ℝ) + I.starts J s (u + 1) =
        I.busy J s (u + 1) + I.ends J s (u + 1) := by exact_mod_cast I.busy_succ J s u
    have ha := G.arc_succR ℓ u hu
    by_cases hn : u + 1 ∈ G.Nd
    · have c := hc (u + 1) hn
      unfold ConserveR at c
      rw [ite_eq_right (by omega), ite_eq_right (by omega)] at c
      rw [ite_eq_left hn] at ha
      linarith
    · obtain ⟨h1, h2, h3, -⟩ := G.not_mem_Nd_zero hon (fun _ => 0) hn
      rw [ite_eq_right hn, G.inLossR_of_not_mem_Nd ℓ hn] at ha
      rw [h1] at hb; rw [h2] at hb; rw [h3] at hr
      push_cast at hb hr
      linarith

/-- **Soundness, continuous idle flow** — the compressed model as built. -/
theorem soundR {J : Finset (Fin I.n)} {s : Fin I.n → ℕ} {ℓ : ℕ → ℝ}
    (hv : I.Valid J s) (hon : G.OnNodes J s) (hℓ : ∀ q, 0 ≤ ℓ q)
    (hc : ∀ v ∈ G.Nd, G.ConserveR J s ℓ v) : I.Feasible J s := by
  refine ⟨hv, fun u hu => ?_⟩
  have h := G.invariantR hon hc u hu
  have hnn : 0 ≤ G.arcIdleR ℓ u :=
    sum_nonneg (fun q _ => by split_ifs <;> simp [hℓ q])
  have : ((I.run u + I.busy J s u : ℕ) : ℝ) ≤ I.N := by push_cast; linarith
  exact_mod_cast this

/-! ### Completeness: left-shifted schedules have a compressed flow -/

/-- Idle units during slot `[u, u+1)`. -/
def idle (J : Finset (Fin I.n)) (s : Fin I.n → ℕ) (u : ℕ) : ℕ :=
  I.N - (I.run u + I.busy J s u)

/-- The flow put on the loss arc leaving node `q`: all idle units at an
event; only the units freed since the previous slot elsewhere. -/
def lossOf (J : Finset (Fin I.n)) (s : Fin I.n → ℕ) (q : ℕ) : ℕ :=
  if q ∈ G.E then idle J s q else idle J s q - idle J s (q - 1)

/-- Between events no idle unit is ever consumed: a start at a non-event
time `v` is not a release, so (left-shifted) the slot before it is full. -/
lemma idle_mono {J : Finset (Fin I.n)} {s : Fin I.n → ℕ}
    (hf : I.Feasible J s) (hl : I.LeftShifted J s)
    {v : ℕ} (hv0 : 0 < v) (hvE : v ∉ G.E) :
    idle J s (v - 1) ≤ idle J s v := by
  have h1 : v - 1 + 1 = v := by omega
  have hr := I.run_succ (v - 1)
  have hb := I.busy_succ J s (v - 1)
  rw [h1] at hr hb
  have hrc : I.rc v = 0 := I.rc_eq_zero (fun r h => hvE (h ▸ G.hrendE r))
  unfold idle
  by_cases hs : I.starts J s v = 0
  · omega
  · obtain ⟨j, hj, hjv⟩ := I.exists_of_starts hs
    have hrel : I.rel j < s j := by
      rcases Nat.eq_or_lt_of_le (hf.1 j hj).1 with h | h
      · exact absurd (hjv ▸ h ▸ G.hrelE j) hvE
      · exact h
    have := hl j hj hrel
    rw [hjv] at this
    omega

theorem arcIdle_lossOf {J : Finset (Fin I.n)} {s : Fin I.n → ℕ}
    (hf : I.Feasible J s) (hl : I.LeftShifted J s) (hon : G.OnNodes J s) :
    ∀ u, u < I.H → G.arcIdle (G.lossOf J s) u = idle J s u := by
  intro u
  induction u with
  | zero =>
    intro _
    rw [G.arcIdle_zero, lossOf, ite_eq_left G.h0]
  | succ u ih =>
    intro hu
    have ih := ih (by omega)
    have ha := G.arc_succ (G.lossOf J s) u hu
    by_cases hE : u + 1 ∈ G.E
    · rw [ite_eq_left (G.hE hE), G.inLoss_of_mem_E _ hE (by omega), Nat.add_sub_cancel,
        lossOf, ite_eq_left hE] at ha
      omega
    · by_cases hn : u + 1 ∈ G.Nd
      · rw [ite_eq_left hn, G.inLoss_of_not_mem_E _ hE, lossOf, ite_eq_right hE,
          Nat.add_sub_cancel] at ha
        have := G.idle_mono hf hl (v := u + 1) (by omega) hE
        rw [Nat.add_sub_cancel] at this
        omega
      · obtain ⟨h1, h2, h3, h4⟩ := G.not_mem_Nd_zero hon (G.lossOf J s) hn
        rw [ite_eq_right hn] at ha
        have hr := I.run_succ u
        have hb := I.busy_succ J s u
        unfold idle at ih ⊢
        omega

theorem complete {J : Finset (Fin I.n)} {s : Fin I.n → ℕ}
    (hf : I.Feasible J s) (hl : I.LeftShifted J s) (hon : G.OnNodes J s) :
    G.FlowFeasible J s := by
  refine ⟨hf.1, hon, G.lossOf J s, fun v hv => ?_⟩
  have hvH := G.hNd v hv
  have hidle := G.arcIdle_lossOf hf hl hon
  unfold Conserve
  by_cases h0 : v = 0
  · subst h0
    rw [ite_eq_left rfl, ite_eq_right (Nat.ne_of_lt I.hH), I.ends_zero, lossOf, ite_eq_left G.h0]
    have := I.run_zero
    have := I.busy_zero J s
    have := hf.2 0 I.hH
    have := I.hRN
    unfold idle
    omega
  · by_cases hH : v = I.H
    · subst hH
      rw [ite_eq_right h0, ite_eq_left rfl, G.inLoss_of_mem_E _ G.hHE I.hH,
        hidle _ (by have := I.hH; omega),
        ← I.busy_last hf.1, ← I.run_last, I.starts_last hf.1]
      have := hf.2 (I.H - 1) (by have := I.hH; omega)
      unfold idle
      omega
    · rw [ite_eq_right h0, ite_eq_right hH]
      have hvlt : v < I.H := by omega
      have h1 : v - 1 + 1 = v := by omega
      have hr := I.run_succ (v - 1)
      have hb := I.busy_succ J s (v - 1)
      rw [h1] at hr hb
      have hc1 := hf.2 v hvlt
      have hc0 := hf.2 (v - 1) (by omega)
      by_cases hE : v ∈ G.E
      · rw [G.inLoss_of_mem_E _ hE (by omega), hidle _ (by omega), lossOf, ite_eq_left hE]
        unfold idle
        omega
      · have hrc : I.rc v = 0 := I.rc_eq_zero (fun r h => hE (h ▸ G.hrendE r))
        have := G.idle_mono hf hl (v := v) (by omega) hE
        rw [G.inLoss_of_not_mem_E _ hE, lossOf, ite_eq_right hE]
        unfold idle at this ⊢
        omega

/-! ### The corrected removal lemma -/

/-- Any capacity-feasible schedule (e.g. an optimum of the full time-grid
model, after the dominance swaps) left-shifts to a compressed-feasible one
whose starts are normal patterns and no later than before. -/
theorem represent (P : ℕ → Prop)
    (hrelP : ∀ j, P (I.rel j)) (hrendP : ∀ r, P (I.rend r))
    (hclos : ∀ j q, P q → I.rel j ≤ q → q + I.p j ≤ I.H → P (q + I.p j))
    (hPNd : ∀ q, P q → q ≤ I.H → q ∈ G.Nd)
    {J : Finset (Fin I.n)} {s : Fin I.n → ℕ} (hf : I.Feasible J s) :
    ∃ s', G.FlowFeasible J s' ∧ ∀ j ∈ J, P (s' j) ∧ I.rel j ≤ s' j ∧ s' j ≤ s j := by
  obtain ⟨s', hf', hl', hle⟩ := I.exists_leftShift s hf
  have hP := I.leftShifted_mem_pattern P hrelP hrendP hclos hf' hl'
  have hon : G.OnNodes J s' := fun j hj => by
    have hv := hf'.1 j hj
    have := I.hp j
    exact ⟨hPNd _ (hP j hj) (by omega),
      hPNd _ (hclos j _ (hP j hj) hv.1 hv.2) hv.2⟩
  exact ⟨s', G.complete hf' hl' hon, fun j hj => ⟨hP j hj, hle j hj⟩⟩

/-- **Removal lemma for the compressed graph.** Take jobs `J \ J'` off this
type (e.g. move them to an unbounded type). The remaining jobs need not keep
their starts — that can be infeasible — but after a left shift they are
compressed-feasible, at normal-pattern starts, each no later than before. -/
theorem remove_then_leftShift (P : ℕ → Prop)
    (hrelP : ∀ j, P (I.rel j)) (hrendP : ∀ r, P (I.rend r))
    (hclos : ∀ j q, P q → I.rel j ≤ q → q + I.p j ≤ I.H → P (q + I.p j))
    (hPNd : ∀ q, P q → q ≤ I.H → q ∈ G.Nd)
    {J J' : Finset (Fin I.n)} (hJ : J' ⊆ J) {s : Fin I.n → ℕ} (h : G.FlowFeasible J s) :
    ∃ s', G.FlowFeasible J' s' ∧ ∀ j ∈ J', P (s' j) ∧ I.rel j ≤ s' j ∧ s' j ≤ s j :=
  G.represent P hrelP hrendP hclos hPNd ((G.sound h).mono I hJ)

end Graph

end Fedhpc
