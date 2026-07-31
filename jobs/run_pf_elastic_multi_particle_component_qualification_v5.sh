#!/usr/bin/env bash
set -euo pipefail

# Short V5 dynamic/restart/dt acceptance only.  The profile builder owns
# initial-state relaxation; this runner evolves the frozen V5 handoff with
# the ordinary selected conserved PF dynamics.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FIXTURE_SCHEMA="${FIXTURE_SCHEMA:-PF_ELASTIC_MULTI_PARTICLE_COMPONENT_VOLUME_PROFILE_V5}"
export FIXTURE_KIND="${FIXTURE_KIND:-E2_COMPONENT_VOLUME_CONSTRAINED_PROFILE}"
export FINAL_PROFILE="${FINAL_PROFILE:-JOINT_PHI_Y_COMPONENT_H_VOLUME_CONSTRAINED_MINIMIZATION_V5}"
export HANDOFF_STATUS="${HANDOFF_STATUS:-PASS_CONSERVED_ELASTIC_MULTI_PARTICLE_HANDOFF_RESTART_DT_V5}"
export CASE_TAG="${CASE_TAG:-elastic_multi_particle_component_v5}"
export RUNNER_PROVENANCE_EXTRA_REL="${RUNNER_PROVENANCE_EXTRA_REL:-jobs/run_pf_elastic_multi_particle_component_qualification_v5.sh}"
exec "${SCRIPT_DIR}/run_pf_elastic_multi_particle_common_matrix_qualification_v3.sh" "$@"
