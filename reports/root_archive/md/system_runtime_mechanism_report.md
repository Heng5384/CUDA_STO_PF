# CUDA_STO_PF System Runtime Mechanism Audit

Audit date: 2026-06-22

Scope: read-only audit of the current local code plus historical cluster results under `/data/home/luozhiheng/CUDA_STO_PF`. No PF equations were modified, no simulations were rerun, and no new CNT scans were generated.

## 1. Executive Classification

Current system status: **semi-closed-loop system**.

The codebase contains the pieces needed for:

```text
CNT/minimize scan
  -> dynamic-continue growth/conditioning
  -> profile extraction
  -> selector/manual template resolution
  -> CUDA scheduled insertion
  -> PF evolution
```

However, cluster history shows this chain is complete only for a subset of cases, mainly `T400_xB0p030` scheduled insertion tests with extracted `faceted_family_profiles.csv`. Many minimized and dynamic-continued nuclei exist without CUDA-ready profile directories, so they are not automatically consumable by runtime insertion.

Most important conclusion:

```text
minimize output is not, by itself, a CUDA insertion template.
dynamic-continue output is a physical PF-evolved source nucleus.
CUDA scheduled insertion consumes extracted radial/faceted profiles plus source summary metadata.
```

## 2. Architecture Summary

### 2.1 CNT / minimize workflow

Primary files:

- `tools/analysis/setup_cnt_workflow.py`
- `jobs/submit_cnt_guide_serial.sbatch`
- `main_cuda.cu`
- `tools/analysis/summarize_cnt_scan_from_guide.py`

Runtime mode:

- `main_cuda.cu` parses `--mode=minimize` and sets `P.mode = 1`.
- `--mode=minimize-continue` also sets `P.minimize_continue_from_vtk = 1`.
- The minimizer uses `P.minimize_max_iter` and `P.minimize_dt`.
- Full-model minimization uses `--minimize-full-model`; phi-only minimization avoids composition fields.

Inputs:

- `guide_cnt_scan.csv`
- temperature as `T_C`
- composition as `xB_out`
- strain components
- radius scan cases, including a reference row with `radius_nm = 0.0`
- PF/CNT settings from generated parameter files and CLI flags

Outputs:

- `energy_minimize_*.csv`
- `phi_final_*.vtk`
- `xB_final_*.vtk` when full-model fields are active
- `summary.txt`
- `current_results_master_table_fitted.csv`
- `reference_energy.csv`

Key data contract:

- `setup_cnt_workflow.py` creates a reference row and scan rows.
- `jobs/submit_cnt_guide_serial.sbatch` launches `./main_cuda --mode=minimize`.
- `main_cuda.cu` writes `F_total_CNT_hat = F_surf_hat + F_el_hat + F_chem_CNT_hat`.
- `summarize_cnt_scan_from_guide.py` subtracts the reference energy when available and extracts fitted/discrete critical radii.

Interpretation:

- This is the **physical/numerical energy minimization and CNT scan layer**.
- It identifies critical radius/barrier information, but its VTK outputs are not directly the runtime insertion format.

### 2.2 dynamic-continue workflow

Primary files:

- `tools/analysis/prepare_continue_dynamic_guide.py`
- `jobs/submit_continue_dynamic_guide_serial.sbatch`
- `main_cuda.cu`
- `tools/analysis/summarize_continue_dynamic_from_guide.py`
- `tools/analysis/generate_continue_dynamic_geometry_summaries.py`

Runtime mode:

- `--mode=dynamics-continue` sets `P.mode = 0` and `P.minimize_continue_from_vtk = 1`.
- `--continue-phi-vtk` and `--continue-xB-vtk` provide the minimized source fields.

Inputs:

- `current_results_master_table_fitted.csv`
- `guide_cnt_scan.csv`
- selected minimized `phi_final_*.vtk`
- selected minimized `xB_final_*.vtk`
- chosen start radius, usually near or above fitted/discrete critical radius

Outputs:

- `continue_dyn_1/phi_*.vtk`
- `continue_dyn_1/xB_*.vtk`
- `continue_dyn_1/summary.txt`
- optional growth/geometry summaries

Key data contract:

- `prepare_continue_dynamic_guide.py` selects a source scan row from the CNT/minimize table.
- For rows with `complete_peak_found`, it picks a source radius based on fitted `rc_cnt_fit_nm`, discrete `rc_cnt_nm`, or nearby scan radius.
- `jobs/submit_continue_dynamic_guide_serial.sbatch` launches `./main_cuda --mode=dynamics-continue --continue-phi-vtk ... --continue-xB-vtk ...`.

Interpretation:

- `dynamic-continue` is a **physical PF evolution step**, not only a format conversion.
- It evolves `phi`, `xB`, and `Y` in dynamics mode from a minimized seed.
- It may smooth the interface, grow a supercritical nucleus, and change morphology/topology if the PF dynamics drives it.
- It also acts as a **representation-conditioning bridge**, because its output is better suited for extracting robust insertion profiles than the raw minimized critical seed.

Answer to the explicit role question:

- Physical evolution step: **yes**.
- Representation transformation step: **also yes**, but indirectly through evolved outputs and later profile extraction.
- Pure smoothing/resampling: **no**.
- Can change topology: **yes, in principle**, because it runs real PF dynamics.
- Direct CUDA insertion input: **not directly for scheduled profile insertion**. It must be converted into `faceted_family_profiles.csv` and paired with `summary.txt`.

### 2.3 CUDA scheduled insertion workflow

Primary files:

- `pf_params.h`
- `main_cuda.cu`
- `tools/analysis/extract_faceted_rebuild_profiles.py`
- `tools/analysis/run_dt_scan_scheduled_nucleation_test.py`
- `tools/analysis/run_pure_dynamics_mass_drift_benchmark.py`

Main runtime functions:

- `parse_scheduled_events` in `main_cuda.cu`
- `resolve_scheduled_nucleus_from_selector` in `main_cuda.cu`
- `load_scheduled_profile_csv` in `main_cuda.cu`
- `apply_scheduled_events_cpu` in `main_cuda.cu`

Runtime input contract:

- scheduled event steps
- scheduled event centers in nm
- source dynamic-continue directory containing `summary.txt`
- profile directory containing `faceted_family_profiles.csv`
- current runtime fields: `phi`, `xB`, `Y`

Output:

- updated device fields after host-side embedding
- `scheduled_nucleation_events.csv`
- unified `nucleation_event_log.csv` entries for scheduled insertion
- optional event VTK snapshots after insertion

How insertion works:

1. At scheduled steps, `apply_scheduled_events_cpu` copies `phi`, `xB`, and `Y` from GPU to CPU.
2. It reads semiaxes from `source_dyn_dir/summary.txt`.
3. It samples faceted/profile data from `profile_dir/faceted_family_profiles.csv`.
4. It embeds the nucleus into the current field around the requested center.
5. It computes a local composition compensation shell.
6. It converts updated `xB` back into `Y`.
7. It copies `phi`, `xB`, and `Y` back to the GPU.

Mass/conservation handling:

- It preserves the global `xBtot` target approximately by local compensation.
- Event CSV logs before/after mass, event mass error, relative event mass error, clipping, and profile out-of-range fraction.
- This is an event insertion operator, not a variational PF nucleation step.

Sub-grid and compatibility handling:

- Runtime checks path existence and center bounds.
- It does not provide a full physical `r/dx` quality gate by itself.
- Therefore, sub-grid rejection must be done by the external compatibility/template validation layer.

Manual vs automatic selection:

- Current code supports selector-driven resolution by default when scheduled nucleation is enabled and `--use_manual_nucleus` is not active.
- Current code still supports manual override using `--use_manual_nucleus true` plus `--scheduled-nuc-profile-dir` and `--scheduled-nuc-source-dyn-dir`.
- Cluster historical scheduled tests appear to be profile/manual-template driven, not fully catalog-driven for every historical scan.

### 2.4 GP-assisted event operators

Primary functions:

- `apply_gp_nucleation_event_cpu`
- `apply_gp_to_beta_event_cpu`
- `apply_gp_assisted_scheduled_event_cpu`
- `apply_gp_assisted_stochastic_selection_cpu`

Interpretation:

- These are separate GP/beta event paths.
- `apply_gp_assisted_scheduled_event_cpu` injects compact beta seeds at GP-assisted sites and tracks a GP/matrix mass ledger.
- `apply_gp_assisted_stochastic_selection_cpu` computes a site hazard from local `xB`, curvature proxy, GP site density, and a configured barrier scale.
- These functions do not consume `faceted_family_profiles.csv` and are not the same as CNT/profile scheduled insertion.

## 3. Shared State and Lifecycle Objects

Shared runtime fields:

- `phi`: beta/precipitate order parameter field.
- `xB`: composition field.
- `Y`: transformed composition/logit-like composition state.
- `eta`/GP-related fields: used by GP event paths, not by scheduled profile insertion in the same way.

Independent lifecycle objects:

- `guide_cnt_scan.csv`: offline scan manifest.
- `energy_minimize_*.csv`: per-radius minimization energy trace.
- `current_results_master_table_fitted.csv`: offline critical-radius/barrier summary.
- `guide_continue_dynamic.csv`: selected post-critical continuation manifest.
- `continue_dyn_1/`: evolved dynamic source nucleus.
- `faceted_family_profiles.csv`: representation-transform output used by scheduled insertion.
- `nucleus_catalog.json` / selector output: optional selection layer that resolves a profile/source pair.
- `scheduled_nucleation_events.csv`: runtime insertion audit log.

## 4. End-to-End Flow Diagram

```text
TRACK A: CNT / minimization / source-nucleus construction

setup_cnt_workflow.py
  [numerical scan generation]
  -> guide_cnt_scan.csv

jobs/submit_cnt_guide_serial.sbatch
  [job launch]
  -> main_cuda --mode=minimize
       [physical/numerical energy minimization]
       -> energy_minimize_*.csv
       -> phi_final_*.vtk
       -> xB_final_*.vtk
       -> summary.txt

summarize_cnt_scan_from_guide.py
  [numerical postprocess]
  -> current_results_master_table_fitted.csv
       fields include rc_schur_nm, rc_cnt_nm, rc_cnt_fit_nm, barrier/status

prepare_continue_dynamic_guide.py
  [selection bridge]
  -> guide_continue_dynamic.csv

jobs/submit_continue_dynamic_guide_serial.sbatch
  [job launch]
  -> main_cuda --mode=dynamics-continue
       [physical PF evolution plus representation conditioning]
       -> continue_dyn_1/phi_*.vtk
       -> continue_dyn_1/xB_*.vtk
       -> continue_dyn_1/summary.txt

extract_faceted_rebuild_profiles.py
  [representation transformation]
  -> profile_dir/faceted_family_profiles.csv


TRACK B: CUDA runtime insertion and PF evolution

nucleus_selector.py or manual CLI
  [selection/resolution]
  -> source_dyn_dir + profile_dir

main_cuda --enable-scheduled-nucleation-test
  -> resolve_scheduled_nucleus_from_selector or manual paths
  -> load_scheduled_profile_csv
  -> parse_scheduled_events
  -> apply_scheduled_events_cpu
       [event insertion, CPU roundtrip, composition compensation]
       -> updated phi/xB/Y on GPU
       -> scheduled_nucleation_events.csv
       -> nucleation_event_log.csv

main CUDA dynamics loop
  [physical PF evolution after insertion]
```

Physical steps:

- `main_cuda --mode=minimize`
- `main_cuda --mode=dynamics-continue`
- post-insertion PF dynamics

Numerical/postprocess steps:

- scan generation
- summary/fitting
- guide generation
- geometry summaries

Representation transformation steps:

- dynamic-continue output to faceted profile extraction
- profile embedding into runtime fields

## 5. Cluster Historical Data Integration

Cluster inspected via `cluster-direct` at:

```text
/data/home/luozhiheng/CUDA_STO_PF
```

The requested `uvip-cluster` route required an additional Tailscale SSH web check, but `cluster-direct` reached the same cluster repository for read-only inspection.

### 5.1 Historical data layout

The cluster does not primarily use the newer `/CNT_SCAN_WORKSTATION_RUN/` layout. Historical results are under:

```text
Results/workflows/
```

Observed workflow roots:

```text
Results/workflows/T350_xB0p030
Results/workflows/T400_xB0p030
Results/workflows/T400_xB0p050
```

Counts observed:

| workflow | energy CSV | phi_final | xB_final | master table | continue guide | continue dirs | profile CSV | scheduled events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T350_xB0p030 | 164 | 164 | 163 | no | no | 0 | 0 | 0 |
| T400_xB0p030 | 275 | 275 | 275 | yes | yes | 26 | 2 | 5 |
| T400_xB0p050 | 301 | 300 | 299 | yes | yes | 25 | 0 | 0 |

### 5.2 Minimize outputs

`T400_xB0p030` and `T400_xB0p050` have fitted CNT/minimize master tables:

```text
Results/workflows/T400_xB0p030/cnt_scan/current_results_master_table_fitted.csv
Results/workflows/T400_xB0p050/cnt_scan/current_results_master_table_fitted.csv
```

Example `T400_xB0p030` row:

```text
T_C=400.0
xB_out=0.03
strain=-0.01
rc_schur_nm=3.035828320234268
rc_cnt_nm=2.835828
rc_cnt_fit_nm=2.8436722852518974
status=complete_peak_found
```

Example `T400_xB0p050` row:

```text
T_C=400.0
xB_out=0.05
strain=-0.01
rc_schur_nm=2.2953769983172116
rc_cnt_nm=2.195377
rc_cnt_fit_nm=2.1823650181400387
F_CNT_peak_hat_used=1.5457189999999862e-05
status=complete_peak_found
```

Important schema difference:

- Some `T400_xB0p030` rows have empty `F_CNT_peak_hat_used` even with `complete_peak_found`.
- `T400_xB0p050` rows include nonempty barrier fields.
- This is a historical schema/version inconsistency and must be treated carefully when building a unified catalog.

### 5.3 dynamic-continue outputs

Observed guide files:

```text
Results/workflows/T400_xB0p030/input/guide_continue_dynamic.csv
Results/workflows/T400_xB0p050/input/guide_continue_dynamic.csv
```

Example `T400_xB0p030` dynamic-continue selection:

```text
base_case_tag=cntcon_T400_xB0p030_strictref_exx_sm0p01
T_C=400.0
xB_out=0.03
strain=-0.01
rc_cnt_fit_nm=2.8436722852518974
start_radius_nm=2.935828
source_radius_nm=2.935828
source_phi_final_rel=Results/workflows/T400_xB0p030/raw/.../phi_final_*.vtk
source_xb_final_rel=Results/workflows/T400_xB0p030/raw/.../xB_final_*.vtk
```

Example `T400_xB0p050` dynamic-continue selection:

```text
base_case_tag=cntcon_T400_xB0p050_strictref_exx_sm0p01
T_C=400.0
xB_out=0.05
strain=-0.01
rc_cnt_fit_nm=2.1823650181399423
start_radius_nm=2.295377
source_radius_nm=2.295377
source_phi_final_rel=Results/workflows/T400_xB0p050/raw/.../phi_final_*.vtk
source_xb_final_rel=Results/workflows/T400_xB0p050/raw/.../xB_final_*.vtk
```

Observed dynamic-continue directories include `continue_dyn_1/summary.txt`, `phi_1500.vtk`, and `xB_1500.vtk`.

### 5.4 CUDA-ready profile outputs

Observed profile CSVs only under `T400_xB0p030`:

```text
Results/workflows/T400_xB0p030/analysis_tmp/exx_s0p0025_faceted_profiles_for_sched/faceted_family_profiles.csv
Results/workflows/T400_xB0p030/analysis_tmp/exx_s0p01_faceted_rebuild_profiles_vbox/faceted_family_profiles.csv
```

Sample profile columns:

```text
family,region,u_nm,phi_mean,phi_std,xB_mean,xB_std,sample_count
```

This confirms the data contract used by `load_scheduled_profile_csv`.

### 5.5 Scheduled insertion logs

Observed scheduled event logs only under `T400_xB0p030`, for example:

```text
Results/workflows/T400_xB0p030/scheduled_nuc_test_exx_s0p0025/.../scheduled_nucleation_events.csv
Results/workflows/T400_xB0p030/scheduled_nuc_dt_scan/.../scheduled_nucleation_events.csv
Results/workflows/T400_xB0p030/pure_dynamics_mass_drift_benchmark/.../scheduled_nucleation_events.csv
```

Example event fields:

```text
event_id,step,source_case_label,center_x_nm,center_y_nm,center_z_nm,xB_edge,
M_before_event,M_after_embed_before_comp,M_after_comp,event_mass_error,
relative_event_mass_error,C_local,local_comp_weight_active_fraction,
xB_min_before,xB_max_before,xB_min_after,xB_max_after,phi_max_after,
mean_hphi_after,clip_lower_fraction,clip_upper_fraction,
profile_queries_out_of_range_fraction,overlap_warning
```

Observed event behavior:

- `source_case_label=T400_xB0p030_exx_s0p0025`
- centers such as `(100,100,100)` nm
- relative event mass error logged as `0.0` in sampled scheduled tests
- profile query out-of-range fraction can be high, around `0.56375` in some runs

Interpretation:

- These are real scheduled insertion tests.
- The high out-of-range profile fraction indicates profile coverage/scale mismatch risk, even when mass compensation succeeds.

## 6. Parameter Traceback and Drift Risks

### 6.1 T, xB, strain propagation

Offline scan:

- `setup_cnt_workflow.py` writes `T_C`, `xB_out`, and strain fields into guide rows.
- `jobs/submit_cnt_guide_serial.sbatch` passes those into `main_cuda`.
- `summarize_cnt_scan_from_guide.py` carries them into master tables.

dynamic-continue:

- `prepare_continue_dynamic_guide.py` copies `T_C`, `xB_out`, strain, radii, and source paths into `guide_continue_dynamic.csv`.
- `jobs/submit_continue_dynamic_guide_serial.sbatch` launches the selected source with `--mode=dynamics-continue`.

scheduled insertion:

- Runtime scheduled insertion gets `P.temperature_C`, `P.ic_23d_xB_out`, and strain from `PFParams`.
- Selector mode passes those to `nucleus_selector.py`.
- Manual mode bypasses selector and accepts explicit profile/source directories.

### 6.2 Important temperature convention risk

Cluster history uses workflow names like `T400_xB0p030` and master table values `T_C=400.0`, meaning **400 degrees Celsius** in the current code convention.

Recent workstation requests used temperatures like `350 K`, `380 K`, and `400 K`. If these are written into a catalog without explicit unit normalization, a `T=400` catalog entry can become ambiguous between:

```text
400 C = 673.15 K
400 K = 126.85 C
```

This is the highest-risk parameter drift in the combined historical/current dataset.

### 6.3 Kernel/version drift

Cluster git status shows many modified files compared with origin, including:

```text
main_cuda.cu
cuda_kernels.cu
cuda_kernels.h
pf_params.h
phase_functions.h
physical_inputs.example.json
thermo_utils.h
tools/analysis/*.py
jobs/submit_cnt_guide_serial.sbatch
```

Therefore, historical cluster outputs cannot be assumed to have been generated by exactly the same code as the current local checkout unless a commit hash/manifest is attached to each run.

### 6.4 Grid/spacing trace

Historical cluster workflows use paths such as:

```text
chel_T400_cuda_400x400x400_dt0.03_steps30000_r2.936nm_xB0.030
```

Current insertion code uses `P.dx`, `P.dy`, `P.dz` in nm-space for centers, semiaxes, and profile embedding. For the historical 400-cube runs, critical radii around 2-3 nm are much more compatible with a 0.1 nm-scale grid than the recent sub-nm smoke-test radii around 0.18 nm.

## 7. Missing Bridges

### 7.1 minimize to insertion is incomplete

Missing for many cases:

```text
minimized phi/xB VTK
  -> dynamic-continue source
  -> faceted profile extraction
  -> catalog/template entry
  -> selector-visible CUDA-ready profile/source pair
```

Evidence:

- `T400_xB0p050` has many minimize and dynamic-continue outputs but no detected `faceted_family_profiles.csv`.
- `T350_xB0p030` has minimize outputs but no detected master/continue/profile/scheduled path.
- Only `T400_xB0p030` has observed profile CSVs and scheduled event logs.

### 7.2 dynamic-continue to CUDA template is not automatic everywhere

The code contains `tools/analysis/extract_faceted_rebuild_profiles.py`, and current CUDA points users to it when `faceted_family_profiles.csv` is missing. But historical data show this extraction has only been performed for selected cases.

### 7.3 selector/catalog path depends on real CUDA-compatible templates

`resolve_scheduled_nucleus_from_selector` is strict:

- selector output must include `profile_dir`
- selector output must include `source_dyn_dir`
- `profile_dir/faceted_family_profiles.csv` must exist
- `source_dyn_dir/summary.txt` must exist
- fallback selector output is rejected in default physics-driven scheduled insertion

This is good failure behavior, but it means the closed loop is only active where the catalog contains valid CUDA-compatible profile/source metadata.

### 7.4 runtime quality gates are incomplete

Runtime insertion validates existence and center bounds. It does not fully validate:

- `r/dx` resolution adequacy
- disconnected morphology
- excessive profile out-of-range fraction before insertion
- template/domain mismatch beyond basic metadata

Those checks need to stay in an external compatibility layer or become a preflight gate before launch.

## 8. Direct Answers

### 8.1 How do minimize, dynamic-continue, and CUDA insertion串联?

They are connected through files and manifests:

```text
guide_cnt_scan.csv
  -> energy_minimize_*.csv + phi_final/xB_final VTK
  -> current_results_master_table_fitted.csv
  -> guide_continue_dynamic.csv
  -> continue_dyn_1/phi_*.vtk + xB_*.vtk + summary.txt
  -> faceted_family_profiles.csv
  -> scheduled runtime profile/source dirs
```

The chain is code-supported, but not fully populated for all historical cases.

### 8.2 dynamic-continue 的真实作用是什么?

It is a real PF dynamics continuation of the minimized nucleus, and secondarily a conditioning step that produces a stable source morphology for profile extraction. It is not just smoothing or resampling.

### 8.3 dynamic-continue 输出是否可以直接作为 CUDA insertion input?

For current scheduled profile insertion: **no, not directly**.

It must be transformed into:

```text
source_dyn_dir/summary.txt
profile_dir/faceted_family_profiles.csv
```

The source dynamic directory supplies geometry metadata; the profile directory supplies actual radial/faceted phi/xB profiles.

### 8.4 CUDA insertion 接受什么格式?

For scheduled profile insertion:

- `faceted_family_profiles.csv`
- `summary.txt` with semiaxes or equivalent geometry
- scheduled event centers in nm
- schedule steps
- current simulation `phi/xB/Y` fields

It does not directly ingest arbitrary `phi_final_*.vtk` as a nucleus template at event time.

### 8.5 xB conservation 如何保证?

`apply_scheduled_events_cpu` computes global `xBtot` before insertion, embeds the profile, then applies a local compensation shell to restore the original mass approximately. It logs `M_before_event`, `M_after_embed_before_comp`, `M_after_comp`, and relative mass error.

### 8.6 当前 insertion pipeline 依赖 manual 还是 automatic selection?

Both modes exist.

- Code default can be selector-driven when scheduled nucleation is enabled and manual override is not used.
- Manual override remains available.
- Cluster history shows real scheduled tests with explicit profile/source directories for `T400_xB0p030`, so historical execution is best classified as **manual/profile-driven**, not globally automatic catalog-driven.

## 9. Final System Status

Final classification:

```text
semi-closed-loop system
```

Why not fully closed-loop:

- CNT/minimize and dynamic-continue are connected by guides.
- Dynamic-continue and scheduled insertion are connected only where profile extraction has been performed.
- Selector-driven runtime exists in code but requires catalog entries with real `profile_dir` and `source_dyn_dir`.
- Historical cluster data show this is complete for selected `T400_xB0p030` tests, incomplete for many other cases.

Why not fully open-loop:

- There is an actual executable path from CNT-derived dynamic source to profile-based scheduled CUDA insertion.
- Scheduled event logs confirm profile insertion was run on cluster.
- Runtime logging confirms inserted nuclei are embedded into active PF fields and then evolved.

## 10. Recommended Next Non-Computational Fixes

No rerun is required for these bookkeeping fixes:

1. Add a manifest table linking each historical case:
   `base_case_tag -> energy CSV -> phi/xB final -> continue_dyn_1 -> profile_dir -> scheduled event logs`.
2. Add explicit unit fields:
   `temperature_value`, `temperature_unit`, `temperature_C`, `temperature_K`.
3. Add code/version metadata:
   commit hash, dirty flag, `main_cuda.cu` hash, parameter file hash.
4. Add preflight CUDA-template validation:
   profile exists, summary exists, `r/dx` acceptable, profile out-of-range expected fraction acceptable.
5. Generate missing profile directories for dynamic-continue outputs before treating them as insertion-ready.

## 11. Bottom Line

The project currently implements a working hybrid workflow, but the bridge from minimized/dynamic nuclei to runtime CUDA insertion is only partially populated in historical data. `dynamic-continue` is a real PF evolution stage that prepares a source nucleus; CUDA scheduled insertion consumes extracted profile templates, not raw minimized nuclei. The system is therefore **semi-closed-loop**: physically meaningful and operational for selected profile-backed cases, but not yet a fully automatic end-to-end runtime pipeline for every CNT/minimize result.
