#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 SOURCE_ROOT RUN_ROOT FIXTURE_ROOT OUT_ROOT" >&2
  exit 2
fi

SOURCE_ROOT="$1"
RUN_ROOT="$2"
FIXTURE_ROOT="$3"
OUT_ROOT="$4"

ANALYZER="${RUN_ROOT}/provenance/analyze_pf_246cube_particle_lineage_v1"
AUDITOR="${SOURCE_ROOT}/scripts/audit_pf_246cube_hourly_merge_dissolution_v1.py"
EXPECTED_STEPS="${RUN_ROOT}/provenance/checkpoint_steps.txt"
CAMPAIGN="${RUN_ROOT}/provenance/campaign_manifest.json"

if [[ -e "${OUT_ROOT}" ]]; then
  echo "[fatal] refusing to overwrite output: ${OUT_ROOT}" >&2
  exit 2
fi
for path in \
  "${ANALYZER}" \
  "${AUDITOR}" \
  "${EXPECTED_STEPS}" \
  "${CAMPAIGN}" \
  "${FIXTURE_ROOT}/fixture_manifest.json" \
  "${FIXTURE_ROOT}/initial_components.csv" \
  "${FIXTURE_ROOT}/phi.raw.f64" \
  "${FIXTURE_ROOT}/xB_alpha.raw.f64"
do
  if [[ ! -f "${path}" ]]; then
    echo "[fatal] missing required input: ${path}" >&2
    exit 2
  fi
done

readarray -t CONTRACT < <(
  python3 - "${CAMPAIGN}" "${FIXTURE_ROOT}/fixture_manifest.json" <<'PY'
import hashlib
import json
import pathlib
import sys

campaign_path = pathlib.Path(sys.argv[1])
fixture_path = pathlib.Path(sys.argv[2])
campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
fixture_sha = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
if campaign["fixture_manifest_sha256"] != fixture_sha:
    raise SystemExit("campaign/fixture SHA-256 mismatch")
if int(campaign["final_step"]) != 152585:
    raise SystemExit("unexpected final step")
if float(campaign["dt_physical_s"]) != 0.9909260953431841:
    raise SystemExit("unexpected physical timestep")
if int(campaign["checkpoint_cadence_steps"]) != 3633:
    raise SystemExit("unexpected checkpoint cadence")
if any(
    campaign[key]
    for key in (
        "GP_enabled",
        "GP_birth_enabled",
        "GP_release_enabled",
        "external_source_enabled",
        "new_beta_nucleation_enabled",
    )
):
    raise SystemExit("prohibited physical path is enabled")
print(f"{float(campaign['target_mean_C_Btot']):.17g}")
print(fixture_sha)
PY
)
TARGET_MEAN="${CONTRACT[0]}"
FIXTURE_SHA256="${CONTRACT[1]}"

CHECKPOINT_ARGS=()
while read -r step _rest; do
  [[ -n "${step}" ]] || continue
  checkpoint="${RUN_ROOT}/checkpoints/step_${step}.chk"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "[fatal] missing checkpoint: ${checkpoint}" >&2
    exit 2
  fi
  CHECKPOINT_ARGS+=(--checkpoint "${checkpoint}")
done <"${EXPECTED_STEPS}"

mkdir -p "${OUT_ROOT}/provenance"
sha256sum \
  "${ANALYZER}" \
  "${AUDITOR}" \
  "${EXPECTED_STEPS}" \
  "${CAMPAIGN}" \
  "${FIXTURE_ROOT}/fixture_manifest.json" \
  "${FIXTURE_ROOT}/initial_components.csv" \
  >"${OUT_ROOT}/provenance/input_hashes.sha256"
printf '%s\n' "${FIXTURE_SHA256}" \
  >"${OUT_ROOT}/provenance/fixture_manifest.sha256"

run_tracker() {
  local name="$1"
  local threshold="$2"
  local destination="${OUT_ROOT}/tracker_${name}"
  set +e
  "${ANALYZER}" \
    --initial-phi "${FIXTURE_ROOT}/phi.raw.f64" \
    --initial-xb "${FIXTURE_ROOT}/xB_alpha.raw.f64" \
    "${CHECKPOINT_ARGS[@]}" \
    --out "${destination}" \
    --grid 246 --dx-nm 1 --threshold "${threshold}" \
    --physical-dt-s 0.9909260953431841 --start-age-h 6 \
    --target-mean "${TARGET_MEAN}" --expected-initial-count 96 \
    --allow-dissolution --allow-merge-groups \
    >"${OUT_ROOT}/tracker_${name}.stdout.log" \
    2>"${OUT_ROOT}/tracker_${name}.stderr.log"
  local rc=$?
  set -e
  if [[ "${rc}" -ne 0 && "${rc}" -ne 2 ]]; then
    echo "[fatal] tracker ${name} returned unexpected code ${rc}" >&2
    exit 2
  fi
  for filename in \
    lineage_summary.txt particle_events.csv particle_lineage.csv \
    ensemble_observables.csv status.txt
  do
    if [[ ! -s "${destination}/${filename}" ]]; then
      echo "[fatal] tracker ${name} did not materialize ${filename}" >&2
      exit 2
    fi
  done
}

run_tracker low 0.0001
run_tracker medium 0.001
run_tracker strong 0.005

python3 "${AUDITOR}" \
  --low "${OUT_ROOT}/tracker_low" \
  --medium "${OUT_ROOT}/tracker_medium" \
  --strong "${OUT_ROOT}/tracker_strong" \
  --initial-components "${FIXTURE_ROOT}/initial_components.csv" \
  --expected-steps "${EXPECTED_STEPS}" \
  --out "${OUT_ROOT}/audit"

grep -qx "PASS_246CUBE_HOURLY_MERGE_DISSOLUTION_AUDIT_V1" \
  "${OUT_ROOT}/audit/status.txt"
cp "${OUT_ROOT}/audit/status.txt" "${OUT_ROOT}/status.txt"
cat "${OUT_ROOT}/status.txt"
