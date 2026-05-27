# Step 27 Observed GP-Zone Diagnostics

## Scope

This step is a **pure GP-only physics diagnostic** for experimentally observed, already-formed GP zones.

What was intentionally **not** changed:

- `storage_exact`
- transport PDE
- elastic solver
- event logic
- stochastic nucleation
- feasibility gate
- throttling
- `two_phase`

No solver logic was modified.

## Files created

- [run_step27_observed_gp_zone_diagnostics.sh](/Users/heng/Documents/GitHub/CUDA_STO_PF/run_step27_observed_gp_zone_diagnostics.sh)
- [reports/step_reports/STEP27_OBSERVED_GP_ZONE_DIAGNOSTICS_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP27_OBSERVED_GP_ZONE_DIAGNOSTICS_REPORT.md)

Remote outputs:

- run root:
  - `/tmp/step27_observed_gp_zone_diagnostics/runs`
- summary CSV:
  - `/tmp/step27_observed_gp_zone_diagnostics/final_key_summary.csv`
- classification map:
  - `/tmp/step27_observed_gp_zone_diagnostics/growth_classification_map.csv`
- plots:
  - `/tmp/step27_observed_gp_zone_diagnostics/plots`

## Setup

Common setup:

- `model_mode = gp_zone`
- `gp_nuc_enabled = 0`
- `gp_to_beta_enabled = 0`
- `y_update_mass_projection_enabled = 0`
- `gp_y_update_mode = storage_exact`
- `gp_elastic_enabled = 1`
- `gp_elastic_active_eta = 1`
- `gp_elastic_active_phi = 0`
- `gp_init_mode = single_sphere`
- `gp_init_mass_mode = local_compensate`
- `dt = 1e-5`
- `steps = 5000`
- `grid = 64^3`
- `T = 27 C`
- `gp_eta_iface_width = 0.20`

Scan:

- background `xB_alpha = 0.03, 0.04, 0.05`
- mature GP seed peak `gp_eta_seed_peak = 0.5, 0.8, 1.0`
- physical GP radius `R_GP = 1, 2, 3, 5 nm`

## Resolution and box-size interpretation

Current raw-field bundle uses:

- `dx = 0.1 nm`
- box size = `64 * 0.1 = 6.4 nm`
- half-box = `3.2 nm`

Therefore:

- `R = 1 nm` corresponds to `10` cells in radius: geometrically resolved
- `R = 2 nm` corresponds to `20` cells in radius: geometrically resolved
- `R = 3 nm` corresponds to `30` cells in radius: geometrically resolved, but **box-marginal** because the diameter is already `6.0 nm`, close to the full `6.4 nm` box
- `R = 5 nm` corresponds to `50` cells in radius: **not representable** as an isolated sphere on this `64^3` box because it exceeds the half-box

So for Step 27 on `64^3`:

- `1 nm` and `2 nm` are resolvable
- `3 nm` is only marginally usable and should be interpreted cautiously
- `5 nm` is **box-limited / numerically infeasible** before any PDE evolution is considered

This is different from an “under-resolved 1 nm” problem. The small radii are resolved in space; the large `5 nm` radius is rejected because the box is too small.

## Main result

All tested mature-GP cases are classified as:

- **numerically infeasible**

That includes every scanned combination at:

- `xB_alpha = 0.03, 0.04, 0.05`
- `eta_peak = 0.5, 0.8, 1.0`
- `R = 1, 2, 3 nm`

and all `5 nm` cases are additionally box-limited.

The failure mode is not NaN/Inf. Instead, what happens is:

1. local-compensated mature-GP initialization succeeds algebraically
2. in many cases the seed amplitude is already scaled down substantially during initialization
3. nevertheless, within the first `20` steps the GP-only dynamics drives:
   - `xB_alpha` to the physical bounds (`~1e-8` and `~1`)
   - `max_abs(dt*divJ)` to very large values
   - early global drift warnings
4. the diagnostic script then stops the case early and marks it infeasible

So this step did **not** find a stable “already-formed mature GP” window on the current `64^3` box and these backgrounds.

## Classification map

| background `xB_alpha` | `peak=0.5, R=1 nm` | `R=2 nm` | `R=3 nm` | `R=5 nm` | `peak=0.8, R=1 nm` | `R=2 nm` | `R=3 nm` | `R=5 nm` | `peak=1.0, R=1 nm` | `R=2 nm` | `R=3 nm` | `R=5 nm` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `0.03` | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible |
| `0.04` | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible |
| `0.05` | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible | numerically infeasible |

## Why they are classified infeasible

For all non-`5 nm` cases:

- the last saved field is already at `step = 20`
- `xB_min_final ≈ 1e-8`
- `xB_max_final ≈ 0.99999999`
- `max_abs(dt*divJ)` ranges from about `4.26e2` up to `1.68e5`
- `gp_closure_error_final` is already in the `1e-4` to `1e-3` range for many cases

This is not a clean “stable mature GP”, “slow growth”, or “dissolution” signature. It is an immediate transport-aggressive state that pushes the composition to the bounds almost at once.

Representative cases:

### `xB_alpha = 0.03`, `eta_peak = 0.5`, `R = 1 nm`

- actual initialized `eta_max = 0.4385` after local compensation
- saved state at `step = 20`
- `xB_min_final = 1e-8`
- `xB_max_final = 0.99999999`
- `local_xB_max_final = 1.0`
- `max_abs(dt*divJ) = 2.38e3`
- early stop reason: `drift_exceeded_1e-2`

### `xB_alpha = 0.04`, `eta_peak = 0.8`, `R = 2 nm`

- actual initialized `eta_max = 0.31497`
- saved state at `step = 20`
- `xB_min_final = 1e-8`
- `xB_max_final = 0.99999999`
- `local_xB_max_final = 1.0`
- `max_abs(dt*divJ) = 7.76e2`
- early stop reason: `drift_exceeded_1e-2`

### `xB_alpha = 0.05`, `eta_peak = 1.0`, `R = 3 nm`

- actual initialized `eta_max = 0.26411`
- saved state at `step = 20`
- `xB_min_final = 1e-8`
- `xB_max_final = 0.99999999`
- `local_xB_max_final = 1.0`
- `max_abs(dt*divJ) = 5.08e3`
- early stop reason: `drift_exceeded_1e-2`

## Local compensation feasibility

This step gives a useful split between **initialization feasibility** and **evolution stability**.

Initialization itself is usually feasible in the narrow algebraic sense:

- `gp_insert_compensation_success = 1`
- `gp_insert_mass_error` stays near numerical tolerance

But for mature GP seeds, local compensation often has to scale the seed down strongly.

Examples:

- `xB=0.03, peak=0.5, R=1 nm`
  - `gp_insert_eta_scaled_factor ≈ 0.877`
  - actual initialized `eta_max ≈ 0.4385`
- `xB=0.05, peak=1.0, R=3 nm`
  - `gp_insert_eta_scaled_factor ≈ 0.264`
  - actual initialized `eta_max ≈ 0.26411`

Across the full matrix:

- `R = 1 nm` often retains about `0.44–0.58` of the requested mature peak
- `R = 2 nm` often drops to about `0.28–0.35`
- `R = 3 nm` often drops to about `0.21–0.26`

So the answer to

> Is local mass compensation feasible for mature GP initialization?

is:

- **yes**, in the sense that the initialization routine can construct a mass-balanced seed
- **no**, in the stronger sense of preserving the requested mature `eta ≈ 1` state

The larger the requested mature GP radius, the more strongly the seed is amplitude-limited by available local Ag inventory.

## Radial-profile interpretation

Step 27 now writes radial profiles for every case under:

- `/tmp/step27_observed_gp_zone_diagnostics/plots/*_radial_profiles.csv`

Representative profile behavior for `xB03_peak0p5_R1nm_5000` at the saved `step = 20`:

- central `eta` is still high
- `phi` remains `0`
- `xB_final(r)` has already developed severe local overshoots, with the center reaching `1.0`

That means the problem is not “beta conversion polluted the GP-only test”.
It is a GP-only transport/composition response to a mature observed-GP initialization that is too demanding for this background/box combination.

## Direct answers to the key physical questions

### 1. Does an already-formed `eta ≈ 1` GP zone of radius `1–5 nm` remain stable?

On the current `64^3`, `dx = 0.1 nm` setup:

- `1–3 nm`: **no**, not as a clean stable GP-only object
- `5 nm`: **cannot be represented** as an isolated object on this box

### 2. Does it dissolve at `xB = 0.03`?

Not in the clean physical sense of a slow, well-resolved dissolution trajectory.

Instead, even the `xB = 0.03` mature-GP cases become transport-aggressive almost immediately and are classified as numerically infeasible.

### 3. Does it grow at `xB = 0.05`?

No stable growth regime was observed.

The `xB = 0.05` mature-GP cases also become infeasible very early, before any trustworthy “growing mature GP” trajectory can be established.

### 4. Is local mass compensation feasible for mature GP initialization?

Yes for algebraic initialization, but often only after strong seed-amplitude reduction.

So it is feasible as a constructor, but it does **not** preserve the intended mature `eta ≈ 1` state for larger radii.

### 5. What GP radius is actually resolvable on the current grid?

On `64^3` with `dx = 0.1 nm`:

- `1 nm`: spatially resolved
- `2 nm`: spatially resolved
- `3 nm`: spatially resolved but box-marginal
- `5 nm`: not box-resolvable

However, “resolved geometrically” is not the same as “stable dynamically”.

### 6. Is mature GP evolution stable without beta conversion?

Not for the scanned `eta_peak = 0.5–1.0`, `R = 1–3 nm`, `xB = 0.03–0.05` cases on this box.

## Interpretation

Step 26B and Step 27 answer different questions:

- Step 26B:
  - can **tiny embryos** become GP?
  - answer: they relax into metastable plateau embryos
- Step 27:
  - what happens if we initialize an **already-formed observed GP zone**?
  - answer: on the current `64^3` box and these backgrounds, the mature GP initialization is either box-limited (`5 nm`) or dynamically infeasible very early (`1–3 nm`)

The important physics interpretation is:

> The current model does not support a clean, long-lived mature-GP-only evolution window for `1–3 nm` observed GP zones on this `64^3`, `dx=0.1 nm` domain with `xB_alpha = 0.03–0.05`.

This does **not** automatically mean the GP model is wrong. It means at least one of these is true:

1. the current box is too small for mature observed-GP diagnostics, especially for `R >= 3 nm`
2. the current local-compensated mature seed is too demanding for the available matrix composition
3. mature GP zones require a different coarse-graining / initialization path than simply scaling up the single-sphere `eta` seed

## Bottom line

Step 27 gives a clear answer:

- tiny embryos from Step 26B are metastable and gentle
- already-formed mature GP seeds from Step 27 are **not** gentle on the current `64^3` setup
- `1 nm` and `2 nm` are geometrically resolvable, but their GP-only evolution is still dynamically infeasible in this scan
- `3 nm` is box-marginal
- `5 nm` is not representable on the current box

So the next physically meaningful step is **not** to compare these mature seeds directly with Step 26B embryos.
The next step should be one of:

1. repeat Step 27 on a larger box (`96^3` or `128^3`) for `R = 3–5 nm`
2. introduce an effective-volume mature-GP initialization specifically for observed GP zones rather than reusing the embryo-style sphere directly
3. diagnose whether the mature-GP instability is primarily box-size driven or local-composition driven by rerunning `R = 1–2 nm` on a larger domain with the same `dx`
