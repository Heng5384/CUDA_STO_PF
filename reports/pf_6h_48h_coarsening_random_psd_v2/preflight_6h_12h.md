# 6 h -> 12 h preflight

Authoritative preflight status: `PASS_PF_RANDOM_PSD_6H12H_PREFLIGHT_V2`.

The run uses the qualified fixed timestep `dt_code=0.02` and the same
hash-pinned fixture. The continuous leg has 21,800 steps; its continuous and
checkpoint/restart final checkpoint hashes are identical:

`5c4d857f9cec7e7a11946c44fc4ee87fe54b623292549ddeedca81581b638761`.

The first/last resolved volume diagnostics are `0.0239295448 -> 0.0244209151`
and the reported mean-radius diagnostic is `9.479961 -> 13.939325 nm`.
The maximum per-step total-mass residual is `2.8380e-15` for the continuous
leg and `5.2736e-16` for the restart second leg. The dt/2 comparison was also
materialized to the same endpoint; its final volume diagnostic is `0.0245743267`.

The particle tracker finds 96, 96, 70, 56, 48, 42, and 34 components at
6, 7, 8, 9, 10, 11, and 12 h, respectively, with no merge/split flag. The
dynamic bound-safety counter is retained as a separate diagnostic (see
`initial_state_audit.md`); it is not silently omitted from the qualification
record.
