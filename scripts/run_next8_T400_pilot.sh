#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${1:-$ROOT/examples/next8_T400_pilot_manifest.json}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "next8_pilot_refused=cluster_environment_detected" >&2
  exit 77
fi

python3 - "$MANIFEST" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("next8_pilot_refused=manifest_missing")
data = json.loads(path.read_text(encoding="utf-8"))
if not data.get("workstation_only") or not data.get("cluster_forbidden"):
    raise SystemExit("next8_pilot_refused=workstation_boundary_missing")
if not data.get("backend_eligible") or not data.get("backend"):
    print("next8_pilot_refused=no_independent_backend")
    print("pilot_cases_completed=0")
    raise SystemExit(78)
if not data.get("execution_authorized"):
    print("next8_pilot_refused=user_authorization_required")
    raise SystemExit(79)
if not isinstance(data.get("delta_phase"), (int, float)) or not isinstance(data.get("delta_diff"), (int, float)):
    print("next8_pilot_refused=force_amplitudes_unresolved")
    raise SystemExit(80)
if not data.get("command") or not data.get("input_hashes"):
    print("next8_pilot_refused=untraceable_command_or_inputs")
    raise SystemExit(81)
raise SystemExit("next8_pilot_refused=runner_not_armed_without_backend_specific_adapter")
PY
