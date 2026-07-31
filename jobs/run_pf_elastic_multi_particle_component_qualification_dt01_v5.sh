#!/usr/bin/env bash
set -euo pipefail

# V5 second timestep tier: dt=0.01 versus 0.005 at the identical endpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DT_CODE="${DT_CODE:-0.01}"
export REFINED_DT_CODE="${REFINED_DT_CODE:-0.005}"
export STEPS="${STEPS:-128}"
export REFINED_STEPS="${REFINED_STEPS:-256}"
export CASE_TAG="${CASE_TAG:-elastic_multi_particle_component_v5_dt01}"
export RUNNER_PROVENANCE_EXTRA_REL="jobs/run_pf_elastic_multi_particle_component_qualification_dt01_v5.sh"
exec "${SCRIPT_DIR}/run_pf_elastic_multi_particle_component_qualification_v5.sh" "$@"
