# Step 26B GP Growth Diagnostics

## Scope

This step is a **pure physics diagnosis** of GP-only single-seed evolution.

What was intentionally left unchanged:

- `storage_exact`
- transport
- elastic solver
- conversion logic
- feasibility gate
- throttling
- stochastic nucleation framework
- `two_phase`

No solver logic was modified.

## Files created

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/run_step26b_gp_growth_diagnostics.sh`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/STEP26B_GP_GROWTH_DIAGNOSTICS_REPORT.md`

Remote outputs:

- run root:
  - `/tmp/step26b_gp_growth_diagnostics/runs`
- summary CSV:
  - `/tmp/step26b_gp_growth_diagnostics/final_key_summary.csv`
- classification map:
  - `/tmp/step26b_gp_growth_diagnostics/growth_classification_map.csv`
- plots:
  - `/tmp/step26b_gp_growth_diagnostics/plots`

## Setup

Common setup for all 12 runs:

- `model_mode = gp_zone`
- `gp_nuc_enabled = 0`
- `gp_to_beta_enabled = 0`
- `y_update_mass_projection_enabled = 0`
- `gp_init_mode = single_sphere`
- `gp_init_mass_mode = local_compensate`
- `dt = 1e-5`
- `steps = 5000`
- `grid = 64^3`
- `T = 27 C`
- `gp_y_update_mode = storage_exact`
- `gp_elastic_enabled = 1`
- `gp_elastic_active_eta = 1`
- `gp_elastic_active_phi = 0`
- seed geometry:
  - `gp_eta_seed_radius = 0.20`
  - `gp_eta_iface_width = 0.20`

Scan matrix:

- background `xB_alpha = 0.03, 0.04, 0.05`
- seed peak `eta_peak = 0.03, 0.05, 0.08, 0.10`

Diagnostics used:

- `eta_max(t)`
- `eta_integral(t)`
- `R_eff_h` from `∫ h(eta) dV`
- `R_eta_proxy` from `∫ eta dV`
- `xB_alpha` min/max and local seed-region `xB`
- `max_abs(dt*divJ)`
- clipping count
- `total_relative_drift`
- `gp_closure_error`
- representative final radial `eta(r)` / `xB(r)` profiles

## Main result

Under the current GP-only model and this temperature/composition window, **none of the tested seeds show sustained GP growth**.

All 12 runs remain numerically healthy:

- no clipping
- no NaN / Inf
- `total_relative_drift` near machine precision
- `gp_closure_error` near machine precision
- `max_abs(dt*divJ)` stays in the safe range `~1e-7` to `~1e-5`

But physically, all 12 runs do the same qualitative thing:

1. the seed peak amplitude relaxes downward by about `4.25%`
2. the total `eta` integral increases slightly by about `0.3%–0.5%`
3. the `h(eta)`-weighted effective volume/radius stays nearly constant, with a tiny decrease
4. local `xB_alpha` enrichment around the seed remains extremely small

This means the seed does **not** disappear, but it also does **not** enter sustained growth. It relaxes into a slightly broader, slightly flatter plateau-like embryo.

## Growth classification map

Using the requested categories:

| background `xB_alpha` | `eta_peak=0.03` | `eta_peak=0.05` | `eta_peak=0.08` | `eta_peak=0.10` |
|---|---|---|---|---|
| `0.03` | metastable plateau | metastable plateau | metastable plateau | metastable plateau |
| `0.04` | metastable plateau | metastable plateau | metastable plateau | metastable plateau |
| `0.05` | metastable plateau | metastable plateau | metastable plateau | metastable plateau |

No case in this matrix qualifies as:

- sustained decay to zero
- slow growth
- runaway

The important nuance is:

- peak amplitude decays slightly
- integrated seed content does not collapse

So the correct label is **metastable plateau**, not true decay.

## Quantitative trends

Across all 12 runs:

- `eta_max_final / eta_max_init ≈ 0.9575`
- `eta_integral_final / eta_integral_init ≈ 1.0033 – 1.0047`
- `R_eff_h_final / R_eff_h_init ≈ 0.99945 – 0.99958`
- local seed-region `xB_alpha` increase is only about `1e-5` to `2e-4`

Representative examples:

### `xB_alpha = 0.03`, `eta_peak = 0.05`

- `eta_max`: `0.04404 -> 0.04217`
- `eta_integral`: `5.97296e-03 -> 5.99474e-03`
- `R_eff_h`: `1.58351e-02 -> 1.58284e-02`
- local `xB_alpha`: `0.02999 -> 0.03002`
- `max_abs(dt*divJ) = 1.21e-06`

Interpretation:

- peak relaxes down
- integrated embryo content slightly broadens
- no sustained enrichment-driven growth

### `xB_alpha = 0.05`, `eta_peak = 0.05`

- `eta_max`: `0.04404 -> 0.04217`
- `eta_integral`: `5.97296e-03 -> 5.99475e-03`
- `R_eff_h`: `1.58351e-02 -> 1.58284e-02`
- local `xB_alpha`: `0.05000 -> 0.05002`
- `max_abs(dt*divJ) = 1.14e-06`

Interpretation:

- essentially the same outcome as `xB=0.03`
- raising the background from `0.03` to `0.05` in this range does not push the seed into growth

### `xB_alpha = 0.05`, `eta_peak = 0.10`

- `eta_max`: `0.08808 -> 0.08433`
- `eta_integral`: `1.19492e-02 -> 1.19894e-02`
- `R_eff_h`: `3.12378e-02 -> 3.12216e-02`
- local `xB_alpha`: `0.04996 -> 0.05016`
- `max_abs(dt*divJ) = 8.42e-06`

Interpretation:

- even the largest seed in this scan does not become supercritical
- it relaxes to the same plateau-like behavior

## Representative radial-profile interpretation

Representative radial profiles were written to:

- `/tmp/step26b_gp_growth_diagnostics/plots/xB03_peak0p05_5000_radial_profiles.csv`
- `/tmp/step26b_gp_growth_diagnostics/plots/xB04_peak0p05_5000_radial_profiles.csv`
- `/tmp/step26b_gp_growth_diagnostics/plots/xB05_peak0p05_5000_radial_profiles.csv`
- `/tmp/step26b_gp_growth_diagnostics/plots/xB05_peak0p10_5000_radial_profiles.csv`

What they show:

- `eta(r)` final is slightly flatter at the center than the initial seed
- the outer profile broadens only very weakly
- `xB(r)` stays almost uniform
- only a tiny enrichment shoulder develops near the seed, far too small to trigger secondary growth

So the GP seed is **not** feeding on a growing local enrichment cloud in this time window.

## Thermodynamic interpretation

Using the same `Δg_nuc_GP` definition as the event model:

- at `T = 27 C`, `Δg_nuc_GP(x,T)` is already negative in all three backgrounds
- approximate background values from the current thermodynamic model:
  - `xB=0.03`: `Δg_nuc_GP ≈ -2169 J/mol`
  - `xB=0.04`: `Δg_nuc_GP ≈ -2178 J/mol`
  - `xB=0.05`: `Δg_nuc_GP ≈ -2137 J/mol`

So chemically, GP formation is favored.

But the runs show that this chemical drive is **not sufficient to make these finite-amplitude embryos grow**.

Why:

1. capillary penalty dominates the early embryo shape relaxation
2. elastic penalty is finite and persistent:
   - mean elastic energy is nonzero
   - it rises with background composition, but does not change the qualitative outcome
3. local enrichment remains tiny:
   - the seed never develops a strong solute cloud that would push it into a growth regime

This means the current model behaves like:

- favorable bulk GP chemistry
- but finite-amplitude embryos in this size/amplitude range relax into a near-stationary diffuse state instead of entering net growth

## Direct answer to the main question

### Does `eta_peak = 0.05` decay, remain metastable, or grow?

For all three backgrounds `xB_alpha = 0.03 / 0.04 / 0.05`:

- it does **not** decay away
- it does **not** show sustained growth
- it settles into a **metastable plateau**

### Is `eta_peak = 0.05` subcritical, near-critical, or supercritical?

Under the current GP-only model and this scan window:

- `eta_peak = 0.05` is **not supercritical**
- it is best interpreted as a **near-critical / metastable embryo**

More precisely:

- a strongly subcritical embryo would shrink away toward zero
- a supercritical embryo would show monotonic radius / volume growth
- this seed does neither
- it relaxes to a diffuse, persistent plateau with almost unchanged integrated size

So the current answer is:

> `eta_peak = 0.05` behaves like a near-critical metastable GP embryo, not a growing supercritical nucleus.

## Broader physical implication

The most important physical result of Step 26B is:

> In the current GP model, tiny inserted GP seeds in the range `eta_peak = 0.03–0.10` and radius `0.20` do not spontaneously enter sustained growth at `xB_alpha = 0.03–0.05`, even though the bulk GP chemical drive is favorable.

That implies at least one of the following is true in the current model:

1. the critical GP size/amplitude is larger than this scan range
2. local enrichment must become much stronger before growth starts
3. capillary and elastic penalties are still dominating at these embryo sizes

## What this means for later modeling

This result is useful for later stochastic event calibration:

- if a nucleation event inserts `eta_peak = 0.05`, that event should currently be interpreted as creating a **metastable embryo**, not a guaranteed growing GP particle
- if the model is expected to show autonomous GP growth after nucleation, later scans should move to:
  - larger seed radius
  - larger seed amplitude
  - higher background `xB_alpha`
  - or refined `gp_delta_g0` / elastic parameter studies

## Bottom line

Step 26B answers the central physics question:

- `eta_peak = 0.05` is **not** supercritical in the current GP-only window
- it behaves as a **near-critical metastable embryo**
- none of the tested seeds show true sustained GP growth
- none decay away either
- local enrichment remains too weak to trigger growth over `5000` steps

