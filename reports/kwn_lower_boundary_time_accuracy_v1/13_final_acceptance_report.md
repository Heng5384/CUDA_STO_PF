# Final acceptance

Top-level status: `BLOCKED_PRODUCTION_TRANSPORT_SCHEME_UPGRADE`.

```json
{
  "ACTIVE_M0_CFL": 0.7689546465160917,
  "ACTIVE_M3_CFL": 0.5995405566018783,
  "BASELINE_REPRODUCED": "PASS_BASELINE_REPRODUCTION",
  "BETA_INITIAL_STATE_IDENTITY": "NOT_RUN_BEFORE_COHORT_EULERIAN_SHARED_OPERATOR_PARITY",
  "BETA_ONLY_DIRECTION": "BLOCKED_PREREQUISITE_GATE",
  "BETA_ONLY_TIMESCALE": "NOT_RUN",
  "BOUNDARY_ACTIVE_CFL": {
    "CFL_4_required_dt_s": 2.6892616479016512e-19,
    "initial_accepted_step": 0.0,
    "observed_physical_Rmin_rate_s_inv": 1.4873971088388063e+19,
    "terminal_state_active": true,
    "terminal_state_rate_s_inv": 1.4873971088388063e+19
  },
  "BOUNDARY_ANALYTIC_BENCHMARK": true,
  "BOUNDARY_GROWTH_VELOCITY_PARITY": "PASS_PHYSICAL_LOWER_BOUNDARY_PARITY",
  "BOUNDARY_INVENTORY_PARITY": "PASS_PHYSICAL_LOWER_BOUNDARY_PARITY",
  "BOUNDARY_NUMBER_FLUX_PARITY": "PASS_PHYSICAL_LOWER_BOUNDARY_PARITY",
  "BRANCH": "codex/kwn-lower-boundary-time-accuracy-v1",
  "COHORT_EULERIAN_CROSSCHECK": "BLOCKED_PREREQUISITE_GATE",
  "COMMIT": "b82fa46af940d7529d2ccaf2f06db3591f4b4ddd",
  "CUDA_RERUN": false,
  "CUMULATIVE_MOL_B_ERROR": "NOT_EVALUATED",
  "CUMULATIVE_NUMBER_DISSOLUTION_ERROR": "NOT_EVALUATED",
  "DIAGNOSTIC_EXPLICIT": "BRUTE_FORCE_DONOR_BOUND_NOT_PRACTICAL",
  "ESTIMATED_48H_COST": {
    "active_CFL_4_projected_48h_steps": 6.425555510183643e+23,
    "active_CFL_4_required_dt_s": 2.6892616479016512e-19,
    "basis": "observed physical-Rmin active-tail state",
    "configured_min_dt_s": 1e-12
  },
  "EULERIAN_AUTHORITY_CONFIG_HASH": "NOT_ASSIGNED",
  "EULERIAN_AUTHORITY_GRID_V2": null,
  "EULERIAN_MAX_RESIDUAL": "NOT_RUN_FINAL_AUTHORITY",
  "EULERIAN_RESTART": "BOUNDARY_UNIT_B7_ONLY",
  "EXPLICIT_VS_COHORT": "NOT_RUN_WHOLE_DOMAIN_DONOR_BOUND_NOT_PRACTICAL",
  "FBETA_ERROR": "NOT_EVALUATED",
  "FINAL_EULERIAN_SOLVER": "CONSERVATIVE_IMPLICIT_UPWIND_FACE_SOLVE_UNQUALIFIED",
  "GLOBAL_MAX_CFL": 2.870792113559984e+19,
  "HISTORICAL_AUTHORITY": "HISTORICAL_SMOOTH_3200_ONLY_V2_NOT_INHERITED",
  "IMPLICIT_ACCURACY_LADDER": "BRUTE_FORCE_CFL_NOT_PRACTICAL",
  "IMPLICIT_VS_COHORT": "BLOCKED_PREREQUISITE_GATE",
  "KEY_REPORTS": [
    "/Users/heng/Documents/GitHub/CUDA_STO_PF-kwn-lower-boundary-time-accuracy-v1/reports/kwn_lower_boundary_time_accuracy_v1/00_baseline_boundary_reproduction.md",
    "/Users/heng/Documents/GitHub/CUDA_STO_PF-kwn-lower-boundary-time-accuracy-v1/reports/kwn_lower_boundary_time_accuracy_v1/01_physical_lower_boundary_contract.md",
    "/Users/heng/Documents/GitHub/CUDA_STO_PF-kwn-lower-boundary-time-accuracy-v1/reports/kwn_lower_boundary_time_accuracy_v1/04_boundary_operator_parity.md",
    "/Users/heng/Documents/GitHub/CUDA_STO_PF-kwn-lower-boundary-time-accuracy-v1/reports/kwn_lower_boundary_time_accuracy_v1/05_courant_definition_and_audit.md",
    "/Users/heng/Documents/GitHub/CUDA_STO_PF-kwn-lower-boundary-time-accuracy-v1/reports/kwn_lower_boundary_time_accuracy_v1/06_implicit_time_accuracy.md",
    "/Users/heng/Documents/GitHub/CUDA_STO_PF-kwn-lower-boundary-time-accuracy-v1/reports/kwn_lower_boundary_time_accuracy_v1/09_cohort_eulerian_crosscheck.md",
    "/Users/heng/Documents/GitHub/CUDA_STO_PF-kwn-lower-boundary-time-accuracy-v1/reports/kwn_lower_boundary_time_accuracy_v1/13_final_acceptance_report.md"
  ],
  "LEGACY_SIX_PARTICLE_EULERIAN_P5": "FAIL_RETAINED",
  "LOCAL_GP_RELEASE_AUTHORIZED": "LOCAL_GP_RELEASE_NOT_AUTHORIZED",
  "LOWER_BOUNDARY_OPERATOR_PARITY": "PASS_LOWER_BOUNDARY_OPERATOR_PARITY",
  "LOWER_TAIL_ACTIVE_CFL": 1.1981215483739291,
  "MEAN_FIELD_PF_GAP": "NOT_RUN",
  "NEXT_ACTION": "Evaluate a conservative characteristic or semi-Lagrangian remap with the same physical-Rmin contract; do not run the cohort crosscheck, authority ladder, PF comparison, or local GP release until it is time-qualified.",
  "N_M0_ERROR": "NOT_EVALUATED",
  "P0_BLOCKERS": [
    "BRUTE_FORCE_CFL_NOT_PRACTICAL_AT_OBSERVED_PHYSICAL_RMIN_ACTIVE_TAIL",
    "BLOCKED_PREREQUISITE_GATE",
    "No time-qualified Eulerian solver/policy exists for the retained 2% cohort gate or v2 authority.",
    "No frozen beta-only PF direction comparison before cohort--Eulerian shared-operator parity passes."
  ],
  "PASSING_ACTIVE_CFL": "NOT_ASSIGNED_UNTIL_COHORT_2_PERCENT_CROSSCHECK",
  "PF_SOURCE_MODIFIED": false,
  "PHYSICAL_RETUNING": false,
  "PHYSICAL_RMIN": 4.724027182871601e-10,
  "PRIMARY_ROOT_CAUSE": "LOWER_BOUNDARY_OPERATOR_MISMATCH_RESOLVED",
  "RMEAN3_ERROR": "NOT_EVALUATED",
  "RMEAN_ERROR": "NOT_EVALUATED",
  "SECONDARY_ROOT_CAUSE": "IMPLICIT_TIME_ACCURACY_UNQUALIFIED_AT_PHYSICAL_RMIN_ACTIVE_BOUNDARY",
  "STATUS": "BLOCKED_PRODUCTION_TRANSPORT_SCHEME_UPGRADE",
  "SV_ERROR": "NOT_EVALUATED",
  "TOP_5_FINDINGS": [
    "The immutable pre-change trace reproduced the old lower-boundary parity failure.",
    "Rmin is frozen as one exact binary64 grid edge for helper, FV face, and cohort event paths.",
    "Raw global CFL and active-population CFL are reported separately; a negligible tail is not silently made an accuracy limiter.",
    "Boundary flux diagnostics are not an additional matrix source; matrix composition is algebraically closed from current beta inventory.",
    "Once the declared lower tail is physically Rmin-active, every requested active-CFL cap falls below min_dt; no 2% crosscheck or PF comparison is claimed."
  ],
  "VALIDATION_CONTRACT_HASH": "d0ff02973ab0f737043e1a40d4f69893a469cbfe2bc4cd22f9e6a410bd0b1333",
  "XMATRIX_ERROR": "NOT_EVALUATED",
  "baseline_historical_Rmin_hex": "0x1.03b4c5a5fbe5ep-31"
}
```
