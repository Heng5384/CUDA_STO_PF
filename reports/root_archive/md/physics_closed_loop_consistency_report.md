# Physics Closed-Loop Consistency Report

## Verdict

**Consistency verdict: not passed for publication-grade physics.**

The current closed-loop preserves the selected nucleus geometry well, but it does not yet prove that CUDA discretization preserves nucleation physics. Barrier ranking, observed barrier shift, GP-assisted preference, and temperature dependence cannot be validated from the available CUDA outputs.

Formal classification:

```text
D. inconsistent / not scientifically validated model
```

Nuance:

Geometry projection for the single selected case is weakly biased, but the full physics loop is not validated because the discrete CUDA side lacks observed barrier/rate data.

## Data Products

Generated:

- `continuous_physics_summary.csv`
- `cuda_execution_summary.csv`
- `physics_consistency_delta.csv`
- `discretization_bias_report.md`

## Part 1. Continuous Physics Predictions

Source files:

- `Results/orchestrator/T400_xB0p050/selected_nucleus.json`
- `Results/workflows/T400_xB0p050/analysis/nucleation_rate/nucleation_rate_table.csv`

Output:

- `continuous_physics_summary.csv`

Selected continuous prediction:

| Field | Value |
| --- | --- |
| case id | `cntcon_T400_xB0p050_strictref_eyy_sm0p01` |
| T | 400 C |
| xB | 0.05 |
| strain | -0.01 |
| rc_continuous | 1.8245556046158986 nm |
| shape_continuous | spherical |
| barrier_continuous | 291.9433447129304 kBT |

## Part 2. Discrete CUDA Execution Summary

Source files:

- `Results/orchestrator/T400_xB0p050/cuda_launch_config.json`
- `Results/orchestrator/T400_xB0p050/template_mapping_validation_log.csv`

Output:

- `cuda_execution_summary.csv`

Current discrete CUDA representation:

| Field | Value |
| --- | --- |
| template | `generated_cntcon_T400_xB0p050_strictref_eyy_sm0p01_rc1p825` |
| inserted shape | spherical |
| effective rc | 1.8540789999999998 nm |
| launch status | `not_requested` |
| observed barrier | not available |
| nucleation rate proxy | not available |

Important: this is a CUDA-compatible template binding, not an executed CUDA nucleation trajectory.

## Part 3. Continuous vs Discrete Delta

Output:

- `physics_consistency_delta.csv`

Measured delta for the selected case:

| Metric | Value |
| --- | ---: |
| rc error | 0.029523395384101203 nm |
| rc relative error | 1.6181143128447655% |
| shape mismatch score | 0.0 |
| mapping loss total | 0.036904244230126504 |

Unavailable:

- barrier deviation;
- nucleation rate distortion factor;
- ordering flip rate;
- CUDA observed barrier rank.

## Key Diagnostic Checks

### 1. Ranking consistency

Question:

```text
continuous DeltaG* ordering == CUDA observed nucleation ordering?
```

Answer: **not verified**.

Reason:

Only one mapped CUDA launch config exists, and no CUDA observed barrier/rate ordering exists.

### 2. Shape consistency

Question:

```text
optimal continuous nucleus shape approx selected CUDA template?
```

Answer: **yes for the selected case**.

The continuous shape is spherical and the mapped CUDA template is spherical. Shape mismatch score is 0.0.

### 3. Barrier consistency

Question:

```text
does discretization systematically shift DeltaG*?
```

Answer: **not verified**.

Reason:

The mapped CUDA case has no observed barrier. Continuous barrier is 291.9433447129304 kBT, but no discrete CUDA barrier exists to compare.

### 4. GP effect consistency

Question:

```text
does GP still reduce barrier in both representations?
```

Answer: **not verified**.

Reason:

The available GP debug outputs are not paired with the mapped T400/xB0.05 continuous nucleus case. They test GP mass/event logic, not continuous-vs-discrete barrier preservation for this nucleus.

### 5. Temperature dependence consistency

Question:

```text
does continuous-vs-discrete behavior preserve temperature trends?
```

Answer: **not verified**.

Reason:

The current mapped CUDA template validation has only one temperature, T = 400 C.

## Does Discretization Change Physics Predictions?

At geometry level:

```text
No meaningful change detected for the selected case.
```

At full nucleation-physics level:

```text
Cannot be certified.
```

The current discrete template does not recalculate or observe a CUDA barrier; therefore barrier ranking and nucleation-rate predictions remain continuous-side predictions carried into CUDA, not independently validated CUDA outcomes.

## Publication Readiness Assessment

Publication ready: **no**.

Reason:

The implementation is a good engineering closure, but publication-grade physical consistency requires evidence that the mapped CUDA insertion preserves:

- barrier ranking;
- barrier magnitude;
- nucleation rate trend;
- GP-assisted vs non-GP preference;
- temperature dependence.

Those quantities are not available from the current CUDA outputs.

## Required Corrections

1. Run actual scheduled CUDA simulations using the mapped template.
2. Emit `selected_nucleus_log.csv` and a post-insertion energy/barrier diagnostic for each mapped case.
3. Generate at least a small matrix over `T`, `xB`, strain, GP on/off.
4. Replace or validate generated analytic `faceted_family_profiles.csv` against relaxed VTK-derived profiles.
5. Add `maps_from_nucleus_id` to mapped catalog entries so selected mapped aliases can be compared to original continuous nuclei without false validation failures.
6. Recompute ordering inversion and rate distortion after observed CUDA barriers/rates exist.

## Critical Answers

### 1. Does CUDA discretization preserve nucleation physics?

Geometry is preserved for the selected case. Full nucleation physics is not proven preserved.

### 2. Does mapping layer introduce systematic bias?

No systematic geometry bias is detected for the selected case. Barrier/rate bias is unbounded because no observed CUDA barrier/rate exists.

### 3. Is current closed-loop scientifically valid for publication?

No. It is engineering-valid as a closed interface, but not yet publication-valid as a physics-consistent nucleation model.

### 4. What is the dominant error source?

Dominant error source:

```text
missing CUDA-observed barrier/rate validation for generated analytic templates
```

Secondary error source:

```text
sparse template library and generated tanh profiles not validated against relaxed nuclei
```

## Final Classification

```text
D. inconsistent / not scientifically validated model
```

This does not mean the selected geometry is wrong. It means the continuous and CUDA representations are not yet physically comparable on the quantities that determine nucleation conclusions.
