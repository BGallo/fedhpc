# fedhpc formal proofs (Lean 4 + Mathlib)

`Proofs/CompressedFlow.lean` — correctness of the compressed arc-flow graph
(`SpaceTimeFormulation(compressed_flow=True)`) for one capacitated type:
soundness (also with continuous idle flow), completeness for left-shifted
schedules, existence of a left shift, normal-pattern membership, and the
corrected job-removal lemma used by the dominance reductions.

Build (Lean toolchain via `elan`, ~/.elan/bin on PATH):

    lake exe cache get   # first time: download prebuilt Mathlib
    lake build

`#print axioms Fedhpc.Graph.remove_then_leftShift` should list only
`propext`, `Classical.choice`, `Quot.sound`.
