#!/usr/bin/env bash
set -euo pipefail

# V4 uses the common short-dynamics runner, but pins its preflight and final
# status to the fixed-phi composition-profile contract.  No long production
# continuation is started here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FIXTURE_SCHEMA="${FIXTURE_SCHEMA:-PF_ELASTIC_MULTI_PARTICLE_FIXED_PHI_COMMON_MATRIX_PROFILE_V4}"
export FIXTURE_KIND="${FIXTURE_KIND:-E2_FIXED_PHI_COMMON_MATRIX_CONSTRAINED_PROFILE}"
export FINAL_PROFILE="${FINAL_PROFILE:-FIXED_PHI_FULL_MODEL_CONSERVED_COMPOSITION_MINIMIZATION_V4}"
export HANDOFF_STATUS="${HANDOFF_STATUS:-PASS_CONSERVED_ELASTIC_MULTI_PARTICLE_HANDOFF_RESTART_DT_V4}"
export CASE_TAG="${CASE_TAG:-elastic_multi_particle_fixed_phi_v4}"
export RUNNER_PROVENANCE_EXTRA_REL="${RUNNER_PROVENANCE_EXTRA_REL:-jobs/run_pf_elastic_multi_particle_fixed_phi_qualification_v4.sh}"
exec "${SCRIPT_DIR}/run_pf_elastic_multi_particle_common_matrix_qualification_v3.sh" "$@"
