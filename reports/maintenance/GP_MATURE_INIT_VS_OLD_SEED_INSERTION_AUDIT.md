# GP Mature Init vs Old Seed Insertion Audit

This note is a code-and-workflow audit only. It does **not** modify the solver, storage, transport, elastic model, or submit new runs.

## Scope

The question is whether the Step 27/28 mature GP failure is telling us that `eta_peak = 1.0` is intrinsically unphysical, or whether the current GP initialization path is missing the older seed-embedding logic that used a self-consistent `xB` depletion/enrichment profile.

The short answer is:

- the old beta/scheduled insertion workflow did **not** insert only an order-parameter seed;
- it inserted a **profile pair**: a source-derived `phi` profile together with a source-derived relaxed `xB` profile;
- it then performed **smooth local mass compensation** outside the protected seed support;
- the current mature GP initialization does **not** do that;
- so the present Step 27/28 hard-seed failure looks much more like an **initialization-profile problem** than proof that `eta_peak = 1.0` is fundamentally impossible.

## Evidence Reviewed

- Scheduled insertion in [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3705)
- Observed GP initialization in [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:5892)
- GP initialization wrapper in [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:6112)
- GP feasibility limiter in [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:890)
- Scheduled insertion parameters in [pf_params.h](/Users/heng/Documents/GitHub/CUDA_STO_PF/pf_params.h:312) and [pf_params.h](/Users/heng/Documents/GitHub/CUDA_STO_PF/pf_params.h:322)
- Observed GP parameters in [pf_params.h](/Users/heng/Documents/GitHub/CUDA_STO_PF/pf_params.h:102) and [pf_params.h](/Users/heng/Documents/GitHub/CUDA_STO_PF/pf_params.h:113)
- Offline source-profile embedding:
  - [tools/analysis/embed_multiple_reconstructed_nuclei.py](/Users/heng/Documents/GitHub/CUDA_STO_PF/tools/analysis/embed_multiple_reconstructed_nuclei.py:471)
  - [tools/analysis/embed_multiple_reconstructed_nuclei_rescaled.py](/Users/heng/Documents/GitHub/CUDA_STO_PF/tools/analysis/embed_multiple_reconstructed_nuclei_rescaled.py:364)
  - [tools/analysis/embed_multiple_reconstructed_nuclei_rescaled.py](/Users/heng/Documents/GitHub/CUDA_STO_PF/tools/analysis/embed_multiple_reconstructed_nuclei_rescaled.py:1519)
- Pure-Python scheduled insertion dry-run:
  - [tools/analysis/test_scheduled_nucleation_insertions_only.py](/Users/heng/Documents/GitHub/CUDA_STO_PF/tools/analysis/test_scheduled_nucleation_insertions_only.py:640)
- Existing Step 27 diagnosis:
  - [reports/step_reports/STEP27_OBSERVED_GP_ZONE_DIAGNOSTICS_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP27_OBSERVED_GP_ZONE_DIAGNOSTICS_REPORT.md:129)

## 1. Old Seed Insertion Workflow

### 1.1 Runtime scheduled insertion in CUDA

The runtime scheduled insertion path is implemented in [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3705).

What it does:

1. It copies the current target-box `phi` and `xB` fields to host memory and records the total mass before insertion through `compute_mean_xBtot_host(...)` in [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3650).
2. It samples a **source nucleus profile family** rather than building a seed from a single scalar amplitude.
3. It reconstructs both:
   - a `phi` profile, and
   - an `xB` profile referenced to source matrix and interface statistics.
4. It samples a local background `xB_edge` around the target insertion region.
5. It blends the source profile into that local background with a smooth source-box window.
6. It applies a **local compensation shell** outside the new and old protected source boxes.
7. It iterates shell compensation until the total `xBtot` returns to target, subject to clipping bounds.

The key part is the profile mapping, not just the seed amplitude. The code path uses:

- `xb_local = xB_matrix_near + alpha_interface * (xb_prof - xB_matrix_reference)` in [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3824)
- `xB[idx] = fmin(xB[idx], xb_candidate)` in [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3831)

That means the old insertion logic was deliberately conservative:

- it inserted the **lower** of current and source-derived `xB`, which naturally preserves a depletion profile around a precipitate-like object;
- it did **not** hard-set a uniform high-core composition field.

### 1.2 Was the old `phi` seed full amplitude?

Often yes, effectively. The runtime path uses `phi[idx] = fmax(phi[idx], phi_i)` after reconstructing `phi_i` from the source profile family. If the source nucleus already has a near-saturated core, the inserted `phi` core can be near full amplitude.

So the old workflow does **not** support the statement that “full amplitude by itself must fail.” The old workflow allowed strong cores, but only together with a physically consistent `xB` profile.

### 1.3 Did the old workflow have a depletion shell and smooth transition?

Yes.

- Smooth spatial transition came from the source-box window and interface-width rescaling:
  - `scheduled_nuc_scale_interface_width`
  - `scheduled_nuc_scale_xB_profile_width`
  - `scheduled_nuc_scale_geometry`
- Smooth compensation shell came from:
  - `scheduled_nuc_local_comp_inner_nm`
  - `scheduled_nuc_local_comp_outer_nm`
  - `scheduled_nuc_local_comp_taper_nm`
  defined and used around [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3856) and defaults near [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:4879).

The compensation shell is outside the protected seed support and uses a tapered weight, not a single-cell or discontinuous correction.

### 1.4 Did the old workflow use local mass compensation?

Yes. It explicitly iterates a local compensation correction using a shell weight. The older offline rescaled embedding path does the same in [tools/analysis/embed_multiple_reconstructed_nuclei_rescaled.py](/Users/heng/Documents/GitHub/CUDA_STO_PF/tools/analysis/embed_multiple_reconstructed_nuclei_rescaled.py:1519).

This is much more than “subtract one uniform offset.” It is:

- spatially weighted,
- iterative,
- clipped to bounds,
- and checked for compensation failure or excessive clipping.

### 1.5 Did the old workflow directly read source profiles from VTK / dynamic-continue output?

Yes, both the older offline embedding utilities and the scheduled-insertion logic are built around a source-profile concept.

The offline embedding scripts reconstruct `phi` and `xB_local` from source profiles using:

- `rebuild_profile_chunk(...)` in [tools/analysis/embed_multiple_reconstructed_nuclei.py](/Users/heng/Documents/GitHub/CUDA_STO_PF/tools/analysis/embed_multiple_reconstructed_nuclei.py:471)
- `rebuild_rescaled_profile_chunk(...)` in [tools/analysis/embed_multiple_reconstructed_nuclei_rescaled.py](/Users/heng/Documents/GitHub/CUDA_STO_PF/tools/analysis/embed_multiple_reconstructed_nuclei_rescaled.py:364)

and define:

- `xB_matrix_reference`
- `xB_matrix_near`

from source profile statistics in:

- [tools/analysis/embed_multiple_reconstructed_nuclei.py](/Users/heng/Documents/GitHub/CUDA_STO_PF/tools/analysis/embed_multiple_reconstructed_nuclei.py:389)
- [tools/analysis/embed_multiple_reconstructed_nuclei_rescaled.py](/Users/heng/Documents/GitHub/CUDA_STO_PF/tools/analysis/embed_multiple_reconstructed_nuclei_rescaled.py:267)

So the old logic absolutely relied on a **relaxed source profile**, not just a target radius and peak amplitude.

### 1.6 How did the old workflow prevent `xB_alpha` overshoot / clipping?

Several layers were present:

1. The inserted `xB` field was source-derived, not guessed from one scalar.
2. It was blended to a sampled local background `xB_edge`.
3. It used `min(current, candidate)` style update for `xB`, which is naturally depletion-friendly for a precipitate insertion.
4. It clipped explicitly to configured bounds:
   - `scheduled_nuc_xB_min`
   - `scheduled_nuc_xB_max`
5. It ran local-shell compensation iteratively and reported large or suspicious compensation.

This combination strongly reduced the chance that insertion itself would create an immediately pathological profile.

## 2. Current Mature GP Initialization Workflow

### 2.1 `observed_gp_diffuse`

The current mature GP path is `gp_init_mode = observed_gp_diffuse`, implemented in [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:5892).

What it does:

1. It constructs an **analytic radial `eta` profile** using:
   - `gp_obs_target_radius_nm`
   - `gp_obs_eta_peak`
   - `gp_obs_iface_width_nm`
   - `gp_obs_profile_type`
2. It calibrates the profile radius so that `∫ h(eta) dV` matches a target effective volume.
3. It leaves the pre-existing `xB_alpha` field as the starting composition.
4. It computes a compensation weight with one of:
   - `smooth_radial_depletion`
   - `wide_shell`
   - `hybrid`
5. It solves for a **single scalar compensation amplitude** and applies that over the compensation weight.
6. It clips `xB_alpha` to:
   - `gp_obs_min_xB_alpha`
   - `gp_obs_max_xB_alpha`
7. It logs initial gradient diagnostics:
   - `gp_obs_max_abs_grad_xB_init`
   - `gp_obs_estimated_max_abs_grad_mu_init`
   in [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:6106).

What it does **not** do:

- it does not load a relaxed source GP profile from VTK;
- it does not map a source `xB_alpha(r)` or `xBtot(r)` profile;
- it does not sample a real edge background around the insertion region;
- it does not build a profile using source matrix-near and interface-reference concentrations;
- it does not iteratively shape a depletion shell from an already relaxed nucleus field.

In other words, current observed GP initialization is:

- analytic `eta(r)`
- plus scalar radial composition compensation

rather than:

- source-relaxed `eta(r)` and `xB(r)` pair
- plus local shell correction

### 2.2 `single_sphere` GP init with `local_compensate`

The older GP seeding branch in [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:6112) is also simpler than the old beta seed path.

For `gp_init_mode = single_sphere`:

- the seed is analytic `eta_seed = peak * 0.5 * (1 - tanh((r-R)/w))`;
- local compensation is a shell removal around `R ... R + max(3w, dx)`;
- if needed, `gp_eta_mass_limiter = local_clip` invokes feasibility limiting through:
  - `gp_recovered_xB_valid_device(...)` in [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:890)
  - `gp_limited_hGP_device(...)` in [cuda_kernels.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:912)

This feasibility limiter helps keep recovered compositions inside bounds, but it still does **not** construct a realistic source-like `xB` profile.

### 2.3 What does “local_compensate” mean here?

For current GP initialization, “local compensate” is much closer to:

- remove or add one scalar amount over a radial shell weight,

not:

- reconstruct a relaxed `xB` profile with spatially varying core/interface/shell structure.

That difference matters. The old scheduled insertion logic used local compensation only **after** inserting a meaningful source profile. The current GP init asks the scalar compensation shell to do almost all the profile work by itself.

## 3. Exact Differences Between Old Seed Insertion and Current Mature GP Initialization

### 3.1 Features the old workflow had and current GP init does not

1. **Source relaxed profile mapping**
   - Old: yes
   - Current GP init: no

2. **Coupled order-parameter + composition embedding**
   - Old: yes, `phi` and `xB` were inserted together as a matched pair
   - Current GP init: no, `eta` is imposed first, `xB_alpha` is corrected afterward

3. **Background matching from the target box**
   - Old: yes, via edge/background sampling
   - Current GP init: no explicit edge sampling

4. **Smooth source-box blend into ambient composition**
   - Old: yes
   - Current GP init: only generic radial compensation weight

5. **Protected source support + compensation outside it**
   - Old: yes
   - Current GP init: not in the same strong sense

6. **Source-derived depletion shell**
   - Old: yes, inherited from the source `xB` profile and preserved by local compensation
   - Current GP init: no explicit source-derived depletion shell

7. **Anisotropic or profile-family geometry**
   - Old: yes, via family-indexed profile reconstruction
   - Current GP init: no, purely analytic radial profile

### 3.2 What the current GP init is effectively missing

The missing pieces are not small details. The current GP path is missing the exact mechanism that previously made full-amplitude insertions survivable:

- source-profile mapping;
- near-interface composition shaping;
- source-to-target width rescaling for composition;
- local edge/background matching;
- smooth depletion shell inherited from a relaxed source field.

## 4. Why the Hard Mature GP Seed Failed

### 4.1 Judgment among A/B/C/D/E

My current judgment is:

- **Primary:** **B. `xB` profile initialization is not physical**
- **Primary:** **E. no source relaxed profile**
- **Secondary:** **D. compensation shell is too simple / too sharp to replace a real profile**
- **Secondary but not dominant:** **C. box-size can worsen things**
- **Not the main diagnosis:** **A. `eta_peak = 1` is intrinsically wrong**

### 4.2 Why this is the most likely explanation

Step 27 already showed that mature-GP cases become transport-aggressive almost immediately, with `xB` driven to bounds very early, even when the initialization is mass-balanced in an algebraic sense; see [reports/step_reports/STEP27_OBSERVED_GP_ZONE_DIAGNOSTICS_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP27_OBSERVED_GP_ZONE_DIAGNOSTICS_REPORT.md:129).

That is exactly what we would expect if:

- the imposed `eta` field suddenly demands a local partitioning pattern,
- but the initialized `xB_alpha` field is only a crude scalar correction,
- so the initial `grad_mu` is extremely large near the GP core and shell.

The current code even logs `gp_obs_max_abs_grad_xB_init` and an estimated `gp_obs_estimated_max_abs_grad_mu_init` in [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:6106), which is consistent with this concern.

So the failure mode is much more consistent with:

- “we initialized a chemically inconsistent mature GP object”

than with:

- “a mature GP with peak 1 can never exist in principle.”

### 4.3 Why `eta_peak = 1.0` alone is not enough to blame

The old beta seed path already tolerated strong cores because it embedded a **relaxed source field**, not just a high order parameter.

If `eta = 1` were intrinsically the problem, the old insertion logic would also be expected to fail whenever it inserted mature or near-mature source nuclei. That is not what the old design was built around.

The stronger interpretation is:

- `eta_peak = 1.0` becomes dangerous when paired with an unrealistic local `xB_alpha` profile.

## 5. Direct Answer to the Core Question

### Could a self-consistent mature GP profile avoid the Step 27/28 overshoot?

**Theoretically yes, very plausibly.**

If we initialize a mature GP seed with:

- `eta_GP(r)` from the observed/target shape,
- a matched `xB_alpha(r)` or `xBtot(r)` profile,
- a smooth depletion/enrichment shell,
- and local mass compensation outside the protected support,

then the initial `grad_mu` should be much smaller than in the current “analytic eta + scalar compensation” path.

That does **not** guarantee stability. The model might still reject the mature GP state. But it would make `eta_peak = 1.0` a **physically testable** initialization rather than an initialization artifact.

### What is required to make `eta_peak = 1.0` physically testable?

At minimum, the initialization path needs interfaces for:

1. **Reading or constructing a source GP profile**
   - `eta_raw` or `eta_profile(r)`
   - `xB_alpha_raw` or `xBtot_raw`

2. **Separating geometry scaling from composition-profile scaling**
   - analogous to old `scale_geometry`, `scale_interface_width`, `scale_xB_profile_width`

3. **Background matching**
   - sample local matrix composition around the target insertion region

4. **Protected support + outer compensation shell**
   - compensation should not rewrite the inserted core/interface directly

5. **Iterative local mass compensation**
   - not only one scalar compensation pass

6. **Feasibility diagnostics**
   - clipping fraction
   - local `grad_mu`
   - recovered `xB_alpha` validity

## 6. Recommended Design Directions

These are recommendations only. They are **not** implemented here.

### 6.1 `observed_gp_from_profile` mode

Yes, this is the cleanest next abstraction.

It should parallel the old scheduled insertion concept:

- use an observed or reconstructed source GP profile,
- map it into the target box,
- then compensate locally.

This would let us distinguish:

- “the model rejects this mature GP profile”

from:

- “the analytic initializer created an unphysical profile.”

### 6.2 Allow `eta` raw field + `xB` raw field to be read together

Yes. This is strongly recommended.

Reading only `eta` is not enough. The old success path depended on inserting **both** the structural field and the composition field together.

For GP, the minimum useful pair would be one of:

- `eta + xB_alpha`, or
- `eta + xBtot`

The better option depends on which representation is easiest to keep thermodynamically consistent with the current GP closure.

### 6.3 Migrate source-profile rescaling and local shell compensation into GP initialization

Yes. This is the main conceptual migration that appears to be missing.

The old scheduled insertion path already solved a closely related problem:

- insert a mature object into a matrix,
- preserve the source interface/composition structure,
- compensate mass without destroying the near-interface profile.

GP observed initialization currently does not inherit that machinery.

### 6.4 Use an offline Python generator for consistent `eta + xB_alpha + xBtot` VTK

Yes, this is a very good low-risk next step before changing solver internals.

An offline script could:

1. generate or reconstruct a mature GP `eta` field;
2. generate a matched `xB_alpha` or `xBtot` field;
3. apply local-shell compensation and clipping diagnostics;
4. write a consistent VTK set;
5. let CUDA only **read** the prepared fields.

That would let us test the scientific hypothesis first, before deciding whether to add a permanent in-solver GP-from-profile path.

## 7. Bottom-Line Conclusions

## Old seed insertion workflow

The old beta/scheduled insertion workflow was **profile-based**, not amplitude-only. It inserted:

- a source-derived order-parameter profile,
- a source-derived composition profile,
- a smooth transition to the surrounding matrix,
- and a local-shell mass compensation step.

## Current mature GP initialization workflow

The current mature GP initialization is **not** using the old profile-redistribution logic. It constructs:

- an analytic `eta` seed,
- then applies scalar radial compensation to `xB_alpha`.

That is a much cruder initialization path.

## Exact differences

The main missing pieces are:

- no source relaxed profile,
- no coupled `eta + xB` embedding,
- no edge-matched background profile,
- no source-derived depletion shell,
- no old scheduled-insertion profile-rescaling logic.

## Why the hard GP seed failed

The failure is best explained by a chemically inconsistent initialization that produces very large initial composition and chemical-potential gradients. It is **not** best explained by `eta_peak = 1.0` being intrinsically invalid.

## What is required to make `eta_peak = 1.0` mature GP physically testable

We need a profile-consistent initialization path:

- source or reconstructed mature GP profile,
- matched `eta` and `xB` fields,
- smooth depletion/enrichment shell,
- background matching,
- local protected-shell mass compensation,
- and diagnostic clipping / `grad_mu` checks.

## Recommended next step

The most defensible next step is **not** another long production run and not another solver-side hard seed tweak.

The best next step is:

1. build an **offline profile-based mature GP initializer** that writes consistent `eta + xB_alpha + xBtot` VTK fields;
2. optionally add a future `observed_gp_from_profile` mode that reuses the old scheduled-insertion profile logic;
3. then test whether `eta_peak = 1.0` still overshoots when the composition field is physically self-consistent.

That would answer the real scientific question without conflating model physics with initialization artifacts.
