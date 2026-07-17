# High-throughput PF baseline freeze

## Frozen control

The frozen PF-only control is memory mask `187` (`M1+M2+M4+M5+M6+M8`),
`ctot_jichen_imex_bdf2_active_manifold_v1`, legacy V0 transport, and fixed
`dt=1.953125e-4` code units (`0.008033122153 s`). Thermodynamics, mobility,
phase kinetics, interface width, spatial operators, hard gates, and the Ctot
ownership contract are unchanged.

The current-source `512x1x1` replay completed `8000/8000` accepted steps in
257.256 s with zero hard rejects, retries, and fallbacks. All five endpoint
arrays are byte-identical to the frozen control:

| Field | SHA-256 |
|---|---|
| Ctot | `c2789fa84d4b3017c2836b4c95ae4e60f02ce23b52532188b36086ef66b3d772` |
| Ctot_nm1 | `2a81a5c6e3dacdfb0a768d0aa51adef0e23ce4175940142408d768416073c6e3` |
| phi | `0e48c8b51bfd5c84bf8c7403625fe8f6fcb2ae730259b26c371703db2714fcdb` |
| phi_nm1 | `f0b568bb9dc27e9c82a3919a558bbdf50188057673dcd84792a22cf3a8aa0c78` |
| xB_alpha | `6d22a351a2f631a8a1e75e736ddb5075df4606cab83320543454d8b292fd1a23` |

The 32-cube and 64-cube mask-0/mask-187 endpoints are bitwise equal. A
32-cube continuous-four-step run is also bitwise equal to the corresponding
two-plus-two restart for all five checkpoint fields. Mass, storage, KKT,
energy/work, bounds, finite-state, clipping, and projection gates pass.

## 400-cube evidence

The allocation ledger remains 12.182601 GiB. The measured peak remains
12.900000 GiB, 83.44% of the 15.46 GiB workstation device, so the memory gate
passes. On the final consistent binary, both the planar (`P`) and curved
(`M`) strict-dt16 eight-step runs completed `8/8` macros with zero internal
trial reject and zero fallback. Their three-warmup-excluded means are
`24.1601914` and `30.2429367 s/step`, respectively. The conservative strict
throughput is therefore `0.956231 physical s/GPUh` on the measured curved
state.

Earlier representative runs showed two depth-two subcycles. Those are retained
as forensic evidence, but the final P/M benchmark establishes that they were
not an intrinsic geometry characteristic: they disappeared after the
history-level and three-level BDF2 algebraic reduction-equivalence checks were
made scale aware. Accepted-step physical mass and `sum(divJ)` gates were not
changed.

The cadence implementation only suppresses non-authoritative verbose audit
rows. A separate cadence matrix proves identical endpoint hashes and unchanged
hard-gate execution.

## Large-grid history reduction equivalence

The original BDF2 history preflight compared the separately reduced masses of
two already accepted history levels with a grid-blind absolute `1e-10`
threshold. On the 400-cube planar smoke, pointwise subtraction followed by the
same GPU reduction gives a history delta of `1.0869083411e-9` at a total mass
of `3.296e7`. This is reduction-order precision, not accepted-step mass loss.
The history-only comparison now uses `max(1e-10, 64 ULP(M_history))`, which is
`2.3841857910e-7` at that scale. The source-free accepted-step mass and
`sum(divJ)` hard gates remain unchanged.

The corrected binary reproduced all five frozen endpoint arrays bitwise for
`512x1x1/8000`, `32^3/5`, and `64^3/5`, and reproduced the `4` versus `2+2`
restart endpoint bitwise. A separate 400-cube four-step smoke completed all
four hard-gated macro steps, entered full BDF2 twice, and reported zero history
mass mismatches. Its two depth-two BE recoveries occurred before the separate
three-level identity check was corrected and remain visible only as historical
negative evidence.

The curved 400-cube state exposed the same reduction-scale issue in the
separate three-level BDF2 algebraic mass identity. `M_np1`, `M_n`, and `M_nm1`
were equal at printed double precision, accepted mass error was exactly zero,
and `sum(divJ)=4.55e-13`, while pointwise identity reduction left
`-1.9693347e-10` (`8.229e-18` relative). The identity-only check now shares the
same ULP-aware reduction-equivalence contract. A two-step curved smoke then
completed with zero retry/fallback and the identity passed at unchanged field
values. A second full bitwise small-grid regression also passed. Neither
accepted-step physical mass gate was changed.

Current workstation binary SHA-256 is
`843a69890c64b059c67c4289e93f3b6cd585ad76190619e551c0fe39f079b2fc`.

Status: `PASS_400CUBE_BASELINE_REVALIDATED_AFTER_REDUCTION_EQUIVALENCE_FIX`.

No cluster, GP source, T380 case, elasticity-on case, commit, or push was used.
