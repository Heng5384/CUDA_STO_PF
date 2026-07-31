#!/usr/bin/env bash
set -euo pipefail

# The V4 dt=0.01 production-candidate refinement.  It compares 0.01 against
# 0.005 at one identical physical endpoint and includes a 0.01 restart.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DT_CODE="${DT_CODE:-0.01}"
export REFINED_DT_CODE="${REFINED_DT_CODE:-0.005}"
export STEPS="${STEPS:-128}"
export REFINED_STEPS="${REFINED_STEPS:-256}"
export CASE_TAG="${CASE_TAG:-elastic_multi_particle_fixed_phi_v4_dt01}"
export RUNNER_PROVENANCE_EXTRA_REL="jobs/run_pf_elastic_multi_particle_fixed_phi_qualification_dt01_v4.sh"
exec "${SCRIPT_DIR}/run_pf_elastic_multi_particle_fixed_phi_qualification_v4.sh" "$@"
