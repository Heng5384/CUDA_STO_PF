# Mapping Binding Final Report

## Final Verdict

Conclusion: **partially closed-loop**.

The continuous-to-discrete mapper is bound into the current orchestrated CUDA launch path, and CUDA can consume the mapped template at runtime. However, the system is not strictly fully closed-loop because manual override flags still exist, CUDA does not directly parse `mapped_template`, and the runtime selector re-selects a `mapped_*` catalog alias rather than the original predicted nucleus object.

## 1. CUDA Runtime Consumption Trace

### Selector resolution

File: `main_cuda.cu`

- `resolve_scheduled_nucleus_from_selector`: lines 3789-3885
- Default scheduled path calls:

```text
python3 nucleus_selector.py select --require-cuda-template --catalog ... --output ...
```

Runtime requirements:

- `cuda_now_uses_predicted_nucleus == true`
- `fallback_triggered == false`
- `selected_nucleus.profile_dir` exists
- `selected_nucleus.source_dyn_dir` exists
- `profile_dir/faceted_family_profiles.csv` exists
- `source_dyn_dir/summary.txt` exists

The resolved paths are copied into:

- `P->scheduled_nuc_profile_dir`
- `P->scheduled_nuc_source_dyn_dir`

### Profile loading

File: `main_cuda.cu`

- `load_scheduled_profile_csv`: lines 3917-4043

CUDA scheduled insertion loads:

```text
P->scheduled_nuc_profile_dir/faceted_family_profiles.csv
P->scheduled_nuc_source_dyn_dir/summary.txt
```

The loader reads profile families from CSV and semiaxes from `summary.txt`.

### Event insertion

File: `main_cuda.cu`

- `apply_scheduled_events_cpu`: lines 5139-5225

The insertion geometry uses:

```text
rt->profile.semiaxes_nm
```

These semiaxes come from the mapped template `summary.txt`.

## 2. Does CUDA Directly Consume `mapped_template`?

No, not literally.

CUDA does **not** parse:

```text
mapped_template
template_geometry_space.json
template_mapping_validation_log.csv
```

CUDA consumes mapper output indirectly through:

```text
mapped_template
  -> selected_nucleus.profile_dir/source_dyn_dir
  -> run-local catalog mapped_* entry
  -> nucleus_selector.py runtime selection
  -> main_cuda.cu P->scheduled_nuc_profile_dir/source_dyn_dir
  -> scheduled insertion profile loader
```

So the binding exists, but it is mediated through selector/catalog fields rather than a native CUDA-side mapper schema.

## 3. Mapping Output Usage

Generated verification files:

- `mapping_ambiguity_report.csv`
- `cuda_binding_verification.csv`

Current validated case:

- run dir: `Results/orchestrator/T400_xB0p050`
- original continuous nucleus: `cntcon_T400_xB0p050_strictref_eyy_sm0p01`
- runtime selector selected: `mapped_cntcon_T400_xB0p050_strictref_eyy_sm0p01`
- mapped template: `generated_cntcon_T400_xB0p050_strictref_eyy_sm0p01_rc1p825`
- `profile_csv_exists`: true
- `summary_exists`: true
- `ready_to_launch`: true
- `manual_override_in_cuda_command`: false
- `runtime_selector_fallback_triggered`: false
- `runtime_selector_cuda_now_uses_predicted_nucleus`: true

This verifies that the current orchestrated launch config is bound to mapper-generated template files.

## 4. Bypass Paths

### Bypass path A: manual nucleus mode

File: `main_cuda.cu`

- CLI parsing: lines 11918-11929
- scheduled validation: lines 12951-12963

Flags:

```text
--use-manual-nucleus
--use_manual_nucleus
--use-manual-nucleus-mode
```

Effect:

If manual mode is true, `resolve_scheduled_nucleus_from_selector()` is skipped and user-supplied `--scheduled-nuc-profile-dir` / `--scheduled-nuc-source-dyn-dir` can bypass the mapper.

### Bypass path B: direct scheduled profile/source flags

File: `main_cuda.cu`

- CLI parsing: lines 11951-11958

Flags:

```text
--scheduled-nuc-source-dyn-dir
--scheduled-nuc-profile-dir
```

Effect:

These are still accepted. In default selector mode, selector resolution overwrites the scheduled dirs. In manual mode, they become the active insertion template and bypass mapper/catalog selection.

### Bypass path C: semiaxes fallback inside profile loader

File: `main_cuda.cu`

- `load_scheduled_profile_csv`: lines 4016-4024

Effect:

If `source_dyn_dir/summary.txt` exists but lacks readable semiaxes, CUDA falls back to a 5 nm spherical geometry. This does not bypass profile loading, but it bypasses mapped geometry fidelity.

### Bypass path D: selector fallback object

File: `nucleus_selector.py`

Effect:

Selector can create `fallback_analytic_sphere`, but current CUDA runtime rejects it in default mode because `fallback_triggered=true` or missing insertable dirs causes a fatal mismatch.

## 5. Uniqueness Check

Output: `mapping_ambiguity_report.csv`

For the current run-local template space:

- continuous rows checked: 53
- rows with unique candidate template under threshold: 53
- rows with multiple candidate templates: 0
- maximum candidates within threshold: 1

Caveat:

This uniqueness result is mostly because the current template library contains one generated template. It verifies no ambiguity in the current run, but it does not prove global uniqueness once a larger faceted template library is added.

## 6. CUDA Binding Verification

Output: `cuda_binding_verification.csv`

Current case result:

```text
cuda_actually_uses_template = true
manual_override_in_cuda_command = false
runtime_selector_fallback_triggered = false
runtime_selector_cuda_now_uses_predicted_nucleus = true
```

Runtime selector check:

```text
python3 nucleus_selector.py select \
  --T 400 --xB 0.05 --strain -0.01 \
  --require-cuda-template \
  --catalog Results/orchestrator/T400_xB0p050/nucleus_catalog.preview.json \
  --output /private/tmp/cuda_runtime_selector_verify.json
```

This selected the mapped catalog entry with the mapper-generated `profile_dir` and `source_dyn_dir`.

## 7. Critical Residual Issue

The runtime selector reports:

```text
validation.selected_shape_consistent_with_lowest_energy_prediction = false
```

Reason:

The selected insertable object is the `mapped_*` catalog alias, while `predicted_lowest_energy_nucleus` remains the original CNT/minimization entry without CUDA dirs. Geometry and rc match, but the validation compares selected-vs-predicted identity/shape semantics too strictly.

Impact:

CUDA runtime itself does not reject this because it checks `cuda_now_uses_predicted_nucleus=true` and `fallback_triggered=false`. Scientifically, the mapping is valid for this case, but the metadata model should be tightened so mapped entries explicitly carry `maps_from_nucleus_id`.

## Required Answers

### 1. Is mapping layer actually used in runtime?

Yes, in the orchestrated default path. Runtime selector chooses the `mapped_*` entry created by the mapper, and CUDA loads that entry's generated profile/source files.

### 2. Is CUDA fully driven by mapped nucleus?

Mostly, but not strictly. CUDA is driven by `selected_nucleus.profile_dir/source_dyn_dir`, which were created by the mapper. CUDA does not directly understand the mapper schema.

### 3. Or is fallback still dominant?

Fallback dominance is low.

The current validated launch has:

- selector fallback: false
- manual override: false
- mapper generated-intermediate fallback: true

This means fallback is not dominating runtime selection, but template generation is still used because no pre-existing real faceted template library matched.

### 4. Is system truly closed-loop or partially bypassed?

Partially closed-loop.

The default orchestrated path is closed:

```text
continuous nucleus -> mapper -> mapped catalog entry -> runtime selector -> CUDA insertion
```

But bypasses still exist:

- manual nucleus mode;
- direct scheduled profile/source flags under manual mode;
- 5 nm spherical semiaxes fallback if summary parsing fails;
- mapped alias metadata is not yet first-class.

## Final Classification

**⚠ partially closed-loop**

The mapper output is actually used by CUDA runtime in the current orchestrated default path, but the architecture still permits manual bypass and relies on catalog alias injection rather than a native CUDA-side `mapped_template` object contract.
