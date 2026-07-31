# PF-only conserved composition engine and T380 6 h→48 h pilot

## 1. Executive summary

The PF-only conserved-composition engine passed the numerical, zero-mode,
checkpoint/restart, and performance qualification gates used for this pilot.
The experimentally anchored T380 pilot was completed from 6 h to 48 h on a
deterministic effective resolved-beta population. The population coarsened:
the resolved count fell from 32 to 8, the mean equivalent radius increased
from 7.1303 nm to 11.0950 nm, and the interfacial-area density decreased by
35.4%. The maximum mean conserved-composition error was
`2.8602120671905595e-14`.

The strict all-hour matrix-composition gate is not closed because the 8 h
checkpoint has `x_Ag=0.006889797923205341`, above the registered upper bound
0.0066. A read-only mask audit confirms this is not caused by the exact
interface-exclusion threshold. Therefore the conservative final status is:

`BLOCKED_MATRIX_CONCENTRATION_TREND`

If the matrix-composition gate is evaluated only at the six pre-registered
experimental ages (6, 12, 18, 24, 36, and 48 h), all gates pass and the
conditional status is:

`PASS_T380_6H_48H_COARSENING_TREND_PILOT`

## 2. Frozen source and scope

| Item | Value |
|---|---|
| Branch requested by user | `codex/pf-zero-mode-restart-provenance-v1` |
| Source commit | `6b69895af2d1b86b99c57c5479ff767349c61efe` |
| Solver scope | PF-only conserved composition + `PF_CONSERVED_Y_ZERO_MODE_V1` |
| Temperature | 380 °C |
| Grid | `128^3` |
| Physical box | `128 × 128 × 128 nm^3` |
| `dx` | 1 nm |
| `lambda_sm` | 4 nm |
| Elasticity | OFF |
| GP population/Birth/release/growth | OFF |
| External source/RSMD | OFF |
| New nucleation after t=0 | OFF |
| Direct beta inventory injection | OFF |
| Commit/push | Not performed |

The worktree is detached at the exact commit because the requested branch is
checked out by another worktree. Unrelated dirty and untracked files were
preserved.

## 3. Physical time contract

The converter produced:

```text
dt_code              = 0.02
t_real_unit_s        = 49.54630476715921
dt_physical_s        = 0.9909260953431841
steps_per_hour       = 3633
pilot_elapsed_s      = 151200
```

The pilot therefore covers exactly 42 physical hours from the 6 h effective
fixture to the 48 h endpoint.

## 4. Effective 6 h fixture

The fixture is a validation-only coarse-grained resolved population. It is
not a one-to-one mapping of the APT Ag-rich object density.

```text
mean_C_B_tot          = 0.03
matrix_xB_alpha       = 0.006219279767278563
matrix_xAg_alpha      = 0.0062
target_beta_fraction  = 0.02392954476632684
actual_beta_fraction  = 0.023929544766326843
initial seed count    = 32
```

The periodic seed ladder uses effective radii 6.3, 7.0, and 7.7 nm with a
periodic separation contract. The raw fixture mass error is
`-3.469446951953614e-17`. The field hashes are recorded in
`fixture_hashes.md` and the fixture manifest.

## 5. Qualification evidence

### T400 and short preflight

The frozen branch evidence contains T400 equal-time, dt-refinement,
continuous/restart, zero-mode provenance, and large-grid performance reports.
The raw-fixture preflight status was:

`PASS_PF_6H48H_RAW_FIXTURE_ZERO_MODE_PREFLIGHT_V1`

Continuous and restart checkpoint fields were bytewise identical, with no
clipping, physical mass projection, or bound violation.

### Restart-safe accelerated continuation

The initial serialized chain (Slurm job 72670) reached the 12 h checkpoint.
It was then stopped and job 72676 resumed from the same checkpoint in a new,
non-overwriting output root. The start checkpoint SHA-256 was identical in
both roots:

`4df5e73df52032515492e58fe327c501fbd9fd9ab2ddb57fe18b4c5f2e8c3e02`

The accelerated continuation completed all remaining 36 h and produced 42
unique hourly checkpoints through step 152586.

## 6. Efficiency qualification

The diagnostic AB test (job 72674) measured:

| Diagnostic mode | Wall time per step |
|---|---:|
| Diagnostic off | 0.005325 s |
| Mass diagnostic every 256 steps | 0.005819 s |
| Mass diagnostic every step | 0.122457 s |

The interval-256 and interval-1 states were bytewise equal in the AB
continuous/restart comparison. The completed 12 h→48 h continuation measured
0.005533194444444445 s/step mean and 0.005612 s/step maximum, approximately
22.1 times faster than the diagnostic-every-step path.

## 7. Structural trend results

| Quantity | 6 h | 48 h | Change |
|---|---:|---:|---:|
| Resolved beta count | 32 | 8 | ratio 0.25 |
| Mean equivalent radius | 7.1303 nm | 11.0950 nm | ratio 1.5560 |
| Mean `R^3` | 368.243 nm³ | 1365.772 nm³ | ratio 3.7089 |
| Beta volume fraction | 0.0239295 | 0.0242482 | +1.33% |
| Interface-area density | 0.0149231 nm⁻¹ | 0.00963593 nm⁻¹ | ratio 0.6457 |

The resolved density ratio is a PF coarse-grained population result and is
not required to equal the experimental APT object-density ratio.

## 8. Conservation, zero mode, and matrix composition

```text
max_system_mass_drift  = 2.8602120671905595e-14
max_zero_mode_lambda   = 6.190303219302345e-12
unexpected merge flag  = 0 for all analyzed transitions
clipping               = false
physical projection    = false
GP/source paths        = false
```

At the six registered experimental ages, matrix `x_Ag` stays within
`0.0062 +/- 0.0004`:

| Age | Matrix `x_Ag` |
|---:|---:|
| 6 h | 0.0062000 |
| 12 h | 0.0058774 |
| 18 h | 0.0058774 |
| 24 h | 0.0058774 |
| 36 h | 0.0058774 |
| 48 h | 0.0058774 |

The hourly series has a transient 8 h value of 0.0068898. Recomputing the
same checkpoint with matrix masks `h<0.005` and `h<0.2` gives 0.0068942 and
0.0068865 respectively, so the excursion is not a threshold-only artifact.

## 9. Final decision and limitation

The solver and numerical state handling are qualified. The physical trend
pilot shows the intended coarsening signatures. The only unresolved issue is
whether the short early matrix-composition excursion should be accepted as a
physical transient despite the sparse experimental anchors, or whether a new
validation-only fixture/observation contract is required.

No thermodynamic, mobility, interface, time-conversion, or kinetic parameter
was retuned to remove the excursion. No GP mechanism was introduced.

## 10. Evidence files

- `baseline_freeze.md`
- `solver_contract.md`
- `t400_equal_time_validation.md`
- `dt_refinement.md`
- `restart_validation.md`
- `zero_mode_validation.md`
- `performance_preflight.md`
- `runtime_projection.md`
- `fixture_manifest.json`
- `fixture_hashes.md`
- `particle_trajectories.csv`
- `population_time_series.csv`
- `matrix_composition_time_series.csv`
- `mass_and_zero_mode_time_series.csv`
- `t380_6h_48h_trend_analysis.md`
- `production_candidate_decision.md`
- `first_failure.csv`
- `final_terminal_output.txt`
