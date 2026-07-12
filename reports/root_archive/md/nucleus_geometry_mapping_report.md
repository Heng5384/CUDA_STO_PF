# Continuous Nucleus to CUDA Template Mapping Report

## Executive Answer

The missing continuous-to-discrete layer is now installed.

Pipeline now closes as:

```text
CNT / energy minimization / selector output
  -> continuous nucleus geometry extraction
  -> CUDA template-space indexing
  -> nearest-template or generated-intermediate-template mapping
  -> selected_nucleus.json + nucleus_catalog preview update
  -> cuda_launch_config.json
  -> CUDA scheduled insertion reads resolved profile/source dirs
```

## Implemented Components

### Continuous nucleus extraction

Implemented in `nucleus_geometry_mapper.py`.

Outputs:

- `continuous_nucleus_geometry.csv`
- run-local example: `Results/orchestrator/T400_xB0p050/continuous_nucleus_geometry.csv`

Extracted fields:

- `rc_nm`
- `aspect_ratio`
- `shape_type`
- `semiaxes_nm`
- `volume_nm3`
- `surface_to_volume_nm_inv`
- `composition_profile`
- source provenance from `selected_nucleus.json` and `nucleus_catalog.json`

### Discrete CUDA template-space index

Implemented in `nucleus_geometry_mapper.py::build_template_space`.

Output:

- `template_geometry_space.json`
- run-local example: `Results/orchestrator/T400_xB0p050/template_geometry_space.json`

Indexed/generated fields:

- `template_id`
- `shape_type`
- `aspect_ratios`
- `semiaxes_nm`
- `rc_range_validity`
- `geometry_vector`
- `profile_dir`
- `source_dyn_dir`

### Core mapper

Implemented in `nucleus_geometry_mapper.py::map_continuous_nucleus_to_cuda_template`.

Mapping rule hierarchy:

1. Minimize radius mismatch, `|rc_continuous - rc_template|`.
2. Minimize shape distance from aspect ratios and feature-vector L2 distance.
3. Track surface-to-volume inconsistency as a surface-energy proxy.
4. If no good existing template is present, generate an intermediate CUDA-compatible template instead of aborting.

Generated CUDA-compatible files:

- `source/summary.txt` with `L1_long`, `L2_mid`, `L3_short`, and aspect ratios.
- `profile/faceted_family_profiles.csv` with octant family profiles and `region=face`.

### Orchestrator integration

Integrated in `nucleus_orchestrator.py`.

The final CUDA stage now performs:

```text
selected_nucleus
  -> map_selected_nucleus_to_cuda_template()
  -> mapped template injected into selected_nucleus.json
  -> mapped template injected into run-local catalog
  -> cuda_launch_config.json uses mapped template
```

Default CUDA config no longer depends on manually supplied scheduled source/profile directories. It uses the selector/catalog object after geometry mapping has resolved the CUDA representation.

## Validation Result

Validation run:

```text
python3 nucleus_orchestrator.py --T 400 --xB 0.05 --strain -0.01 --dry-run
```

Terminal result:

```text
nucleation_orchestrator_installed
full_pipeline_mode_active
autonomous_status = true
catalog_self_learning = true
cuda_closed_loop = true
template_mapper_installed
continuous_to_discrete_mapping_active
geometry_loss_mean = 0.036904244230126504
cuda_closed_loop_status = true
missing_template_regions = generated_intermediate_profile_when_no_existing_template_matches
```

Mapped nucleus:

- `id`: `cntcon_T400_xB0p050_strictref_eyy_sm0p01`
- continuous `rc_nm`: `1.8245556046158986`
- shape: `spherical`
- semiaxes: `1.9162005 1.8963405 1.749696`
- mapped template: `generated_cntcon_T400_xB0p050_strictref_eyy_sm0p01_rc1p825`

Validation log:

- `Results/orchestrator/T400_xB0p050/template_mapping_validation_log.csv`

Observed mismatch:

- `rc_mismatch_nm`: `0.029523395384101203`
- `shape_distance`: `0.0`
- `feature_l2`: `0.029523395384101203`
- `volume_relative_error`: `0.0`
- `surface_energy_inconsistency`: `0.0`
- total geometry loss: `0.036904244230126504`

## Required Questions

### 1. Does every continuous nucleus have a CUDA representation?

Yes, at the interface level.

If an existing template matches, the mapper selects it. If no suitable template exists, the mapper generates an intermediate CUDA-compatible template with `summary.txt` and `faceted_family_profiles.csv`, then updates `template_geometry_space.json`.

### 2. What is geometric error introduced by discretization?

For the validated T400/xB0.05 case, the total geometry loss is:

```text
0.036904244230126504
```

The dominant term is a small effective-radius mismatch of about `0.0295 nm`; shape, volume, and surface-to-volume terms are zero for the generated intermediate template because it uses the continuous semiaxes directly.

### 3. Does mapping bias nucleation predictions?

The mapping layer does not change CNT, CE, DFT, or minimization energies. It can still introduce insertion bias through discretization if an existing coarse template is selected.

Current mitigation:

- mismatch metrics are logged for every launch;
- generated intermediate templates preserve semiaxes when no good template exists;
- CUDA config records the decision reason.

Remaining bias risk:

- generated profiles use analytic tanh composition/order-parameter profiles, not VTK-derived relaxed profiles;
- faceting family profiles are approximated uniformly across octants unless a real profile library exists.

### 4. Is the system now fully closed-loop from physics to CUDA execution?

Yes for architecture and runtime wiring:

```text
physics prediction -> selector -> continuous nucleus -> mapped CUDA template -> CUDA launch config
```

It is not yet fully self-consistent at the profile-fidelity level because generated profiles still need validation against minimized VTK-derived faceted profiles.

## Final Classification

The system is now a closed-loop physics-driven insertion pipeline with a deterministic geometry-discretization layer, but production-grade physical fidelity still depends on replacing or validating generated analytic templates with relaxed CUDA/DFT-derived profile libraries.
