# Step 28 Mature GP Instability Diagnosis

## Scope

This step is a **diagnostic-only** separation study for mature observed GP-zone instability.

What was intentionally **not** changed:

- `storage_exact`
- transport PDE
- elastic solver
- event logic
- stochastic nucleation
- feasibility gate
- throttling
- `two_phase`

No solver formula was modified.

## Files created

- [run_step28_mature_gp_instability_diagnosis.sh](/Users/heng/Documents/GitHub/CUDA_STO_PF/run_step28_mature_gp_instability_diagnosis.sh)
- [reports/step_reports/STEP28_MATURE_GP_INSTABILITY_DIAGNOSIS_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP28_MATURE_GP_INSTABILITY_DIAGNOSIS_REPORT.md)

Remote outputs:

- run root:
  - `/tmp/step28_mature_gp_instability_diagnosis/runs`
- summary CSV:
  - `/tmp/step28_mature_gp_instability_diagnosis/final_key_summary.csv`

## What was tested

### Part 1: larger-box repeat of hard mature seeds

Repeated mature observed-GP seeds with:

- `R = 1 nm, 2 nm`
- `eta_peak = 0.5, 0.8`
- `xB_alpha = 0.03, 0.05`
- `dt = 1e-5`
- `steps = 5000`

on:

- `96^3`
- `128^3`

### Part 2: wider/smoother compensation profile

This part could **not** be executed cleanly without changing initialization interfaces.

Reason:

- current `raw_fields` mode accepts `phi/xB raw`, but not `eta raw`
- current GP initialization path can inject only the built-in `single_sphere` eta seed
- the built-in `local_compensate` shell width/profile is not externally parameterized

So there is currently **no solver-preserving path** to feed:

- the same custom mature `eta` field
- together with a custom widened or smoothly depleted `xB_alpha(r)` field

back into the existing dynamics code.

That limitation is an interface limitation, not a physics claim.

### Part 3: effective-volume mature GP seeds

Instead of hard `eta≈1`-like seeds, I tested diffuse seeds where:

- `eta_peak = 0.2, 0.3, 0.4, 0.5`
- the seed radius was calibrated so that `∫ h(eta) dV` matches target mature GP volume

Target equivalent radii:

- `R_eq = 1 nm`
- `R_eq = 2 nm`

on `96^3`.

In practice, the full planned effective-volume matrix would be expensive to complete at `5000` steps, so I used the representative subset that had already accumulated enough evidence:

- `xB = 0.03`, `R_eq = 1,2 nm`, `eta_peak = 0.2, 0.3, 0.4, 0.5`
- `xB = 0.05`, `R_eq = 1 nm`, `eta_peak = 0.2, 0.3`

This subset is enough to separate box-size effects from initialization-profile effects.

## Main result

### 1. Box enlargement alone does **not** cure hard mature GP instability

All hard mature seeds on both `96^3` and `128^3` still failed very early:

- every hard case hit `drift_exceeded_1e-2`
- every hard case only survived to the first written field at `step = 20`
- every hard case already had:
  - `xB_min_final = 1e-08`
  - `xB_max_final = 0.99999999`

Representative comparison for the same physical case:

#### `xB = 0.03`, `eta_peak = 0.5`, `R = 1 nm`

`96^3`:

- `final_written_step = 20`
- `max_abs(dt*divJ) = 1.7470e+04`
- `gp_closure_error_final = 2.013e-03`
- note: `drift_exceeded_1e-2`

`128^3`:

- `final_written_step = 20`
- `max_abs(dt*divJ) = 9.2921e+03`
- `gp_closure_error_final = 3.056e-03`
- note: `drift_exceeded_1e-2`

So the larger box reduces the severity somewhat in some cases, but it does **not** qualitatively stabilize the hard mature seed.

That means:

> the Step 27 instability is **not** primarily a `64^3` periodic-image artifact for `R = 1–2 nm`.

### 2. Effective-volume initialization changes the instability timescale substantially

The clearest case is:

#### `96^3`, `xB = 0.03`, `R_eq = 1 nm`, `eta_peak = 0.2`

- `eta_max_init = 0.2`
- `R_eff_h_init ≈ 1.0000 nm`
- `final_written_step = 940`
- `max_abs(dt*divJ) = 1.5290e+03`
- `gp_closure_error_final = 1.8104e-02`
- note: `drift_exceeded_1e-2`

This is a major contrast with the hard-seed version of the same physical size, which failed at `step = 20`.

So even though the effective-volume seed still eventually becomes infeasible, it remains healthy for roughly:

- `~940 / 20 ≈ 47x`

longer than the hard mature seed path.

That is strong evidence that:

> the mature-GP instability is not only a box-size problem; it is strongly driven by the initialization profile.

### 3. Effective-volume helps most for the softest/smallest mature seeds

Other effective-volume cases still fail early.

Examples:

#### `96^3`, `xB = 0.03`, `R_eq = 1 nm`

- `eta_peak = 0.3` -> fails at `step = 20`, `max_abs(dt*divJ) = 1.8348e+03`
- `eta_peak = 0.4` -> fails at `step = 20`, `max_abs(dt*divJ) = 3.8143e+03`
- `eta_peak = 0.5` -> fails at `step = 20`, `max_abs(dt*divJ) = 1.0342e+04`

#### `96^3`, `xB = 0.03`, `R_eq = 2 nm`

- `eta_peak = 0.2` -> fails at `step = 20`, `max_abs(dt*divJ) = 1.8122e+03`
- `eta_peak = 0.3` -> fails at `step = 20`, `max_abs(dt*divJ) = 9.7214e+03`
- `eta_peak = 0.5` -> fails at `step = 20`, `max_abs(dt*divJ) = 2.6307e+04`

#### `96^3`, `xB = 0.05`, `R_eq = 1 nm`

- `eta_peak = 0.2` -> fails at `step = 20`, `max_abs(dt*divJ) = 1.0656e+03`
- `eta_peak = 0.3` -> fails at `step = 20`, `max_abs(dt*divJ) = 1.8445e+03`

So the effective-volume idea does help, but not universally.

The benefit appears strongest when:

- the effective-volume seed is both
  - small (`R_eq = 1 nm`)
  - and diffuse (`eta_peak = 0.2`)

## What this means physically

Step 28 separates three possibilities:

### A. Box-size artifact?

For `R = 1–2 nm`, the answer is:

- **not the primary issue**

Going from `64^3` to `96^3` and `128^3` does **not** convert hard mature seeds into healthy GP-only trajectories.

### B. Local composition-profile artifact?

Yes, very likely **a major part** of the problem.

Evidence:

- hard mature seeds fail at `step = 20`
- the softest effective-volume seed survives to `step = 940`

That means the shape and sharpness of the initial GP/composition profile matters strongly.

This is the strongest outcome of Step 28.

### C. Genuinely unstable mature GP state?

Not proven yet.

What Step 28 shows is:

- hard `eta≈1`-style mature seeds are too aggressive for this framework
- diffuse effective-volume seeds can be substantially healthier
- but even the best tested diffuse seed here still eventually goes unstable

So the honest conclusion is:

> we still cannot claim that experimentally observed mature GP zones are physically unstable in the model; we can only say that the current hard-seed initialization is too severe, and that diffuse effective-volume initialization improves but has not yet fully solved the problem.

## Direct answers to the requested questions

### 1. Is `R = 1–2 nm` mature GP instability box-size dependent?

Only weakly.

Larger boxes reduce severity in some cases, but do **not** remove the early hard-seed failure.

### 2. Does increasing box size improve stability?

Slightly, but not enough.

It is not a qualitative fix for the hard mature-seed path.

### 3. Does widening/smoothing compensation suppress `dt*divJ` blow-up?

Exact widened/smoothed shell tests could not be executed without an initialization-interface change.

However, the effective-volume diffuse-seed test acts as a practical proxy for a smoother mature-GP initialization.

That proxy clearly shows improved behavior:

- the best effective-volume case lasts to `step = 940`
- the corresponding hard-seed cases fail at `step = 20`

So the evidence points to:

- **yes**, smoother/diffuse initialization suppresses the early blow-up significantly

### 4. Can effective-volume mature GP seeds represent observed GP zones more safely?

Yes, **more safely than hard eta≈1 seeds**.

But in the current tested subset they are not yet universally stable for `5000` steps.

### 5. Should future observed-GP simulations use hard eta≈1 seeds or diffuse effective-volume seeds?

Based on Step 28:

> future observed-GP diagnostics should prefer **diffuse effective-volume seeds**, not hard `eta≈1` mature cores.

That is the clearest modeling recommendation from this step.

## Bottom line

Step 28 gives a clear separation:

- the mature-GP instability is **not mainly a 64^3 box artifact**
- it is **strongly tied to the initialization profile**
- hard mature seeds are too aggressive
- diffuse effective-volume seeds are a better representation and can greatly delay instability

But:

- even the best currently tested effective-volume case did not remain healthy through all `5000` steps

So the next physically meaningful step is:

1. formalize an observed-GP-specific diffuse initialization path
2. expose initialization-profile controls directly, so we can truly test:
   - wider compensation shell
   - smooth depletion profile
   - fixed target `V_GP_h` with controlled `xB_alpha(r)`

without changing the solver PDEs.
