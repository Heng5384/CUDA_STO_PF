#!/usr/bin/env bash
set -euo pipefail

# Materialize and audit three non-overwriting 246^3 conditional handoff
# fixtures from a hash-pinned 0.25 nm target-profile library.  The integer
# PSD selection is frozen separately; no profile is scaled or interpolated.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"

SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}}"
HISTORICAL_MANIFEST="${HISTORICAL_MANIFEST:?HISTORICAL_MANIFEST is required}"
LIBRARY_ROOT="${LIBRARY_ROOT:?LIBRARY_ROOT is required}"
SELECTION_PROVENANCE="${SELECTION_PROVENANCE:?SELECTION_PROVENANCE is required}"
INVENTORY_SELECTION="${INVENTORY_SELECTION:?INVENTORY_SELECTION is required}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"

[[ ! -e "${OUT_ROOT}" ]] || {
  echo "[fatal] refusing to overwrite OUT_ROOT: ${OUT_ROOT}" >&2
  exit 2
}
for path in \
  "${HISTORICAL_MANIFEST}" \
  "${LIBRARY_ROOT}/library/library_manifest.json" \
  "${SELECTION_PROVENANCE}" \
  "${INVENTORY_SELECTION}" \
  "${SOURCE_ROOT}/scripts/materialize_pf_246cube_library_handoff_v1.py" \
  "${SOURCE_ROOT}/scripts/audit_pf_246cube_three_seed_fixtures_v1.py"; do
  [[ -f "${path}" ]] || {
    echo "[fatal] missing required input: ${path}" >&2
    exit 2
  }
done
mkdir -p "${OUT_ROOT}/provenance"
exec >"${OUT_ROOT}/driver.stdout.log" 2>"${OUT_ROOT}/driver.stderr.log"

LIBRARY_MANIFEST="${LIBRARY_ROOT}/library/library_manifest.json"
LIBRARY_SHA256="$(sha256sum "${LIBRARY_MANIFEST}" | awk '{print $1}')"
SELECTION_SHA256="$(sha256sum "${SELECTION_PROVENANCE}" | awk '{print $1}')"
SOURCE_TREE_SHA256="$(python3 - "${LIBRARY_MANIFEST}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["source_tree_sha256"])
PY
)"
mapfile -t RADII < <(python3 - "${LIBRARY_MANIFEST}" <<'PY'
import json,sys
for value in json.load(open(sys.argv[1]))["radius_ladder_nm"]:
    print(value)
PY
)

sha256sum \
  "${HISTORICAL_MANIFEST}" \
  "${LIBRARY_MANIFEST}" \
  "${SELECTION_PROVENANCE}" \
  "${INVENTORY_SELECTION}" \
  "${SOURCE_ROOT}/scripts/materialize_pf_246cube_library_handoff_v1.py" \
  "${SOURCE_ROOT}/scripts/audit_pf_246cube_three_seed_fixtures_v1.py" \
  >"${OUT_ROOT}/provenance/input_hashes.sha256"

COMMON_ARGS=(
  --historical-manifest "${HISTORICAL_MANIFEST}"
  --library-root "${LIBRARY_ROOT}"
  --selection-provenance "${SELECTION_PROVENANCE}"
  --inventory-selection "${INVENTORY_SELECTION}"
  --library-manifest-sha256 "${LIBRARY_SHA256}"
  --selection-provenance-sha256 "${SELECTION_SHA256}"
  --registered-radii-nm "${RADII[@]}"
  --expected-source-tree-sha256 "${SOURCE_TREE_SHA256}"
)

for label in replicate_A replicate_B replicate_C; do
  fixture="${OUT_ROOT}/${label}"
  python3 "${SOURCE_ROOT}/scripts/materialize_pf_246cube_library_handoff_v1.py" \
    "${COMMON_ARGS[@]}" --replicate "${label}" --out "${fixture}"
  python3 "${SOURCE_ROOT}/scripts/materialize_pf_246cube_library_handoff_v1.py" \
    "${COMMON_ARGS[@]}" --replicate "${label}" --input-order reverse \
    --compare-to-manifest "${fixture}/fixture_manifest.json"
done

python3 "${SOURCE_ROOT}/scripts/audit_pf_246cube_three_seed_fixtures_v1.py" \
  --fixture "replicate_A=${OUT_ROOT}/replicate_A" \
  --fixture "replicate_B=${OUT_ROOT}/replicate_B" \
  --fixture "replicate_C=${OUT_ROOT}/replicate_C" \
  --materializer \
    "${SOURCE_ROOT}/scripts/materialize_pf_246cube_library_handoff_v1.py" \
  --historical-manifest "${HISTORICAL_MANIFEST}" \
  --library-root "${LIBRARY_ROOT}" \
  --selection-provenance "${SELECTION_PROVENANCE}" \
  --inventory-selection "${INVENTORY_SELECTION}" \
  --library-manifest-sha256 "${LIBRARY_SHA256}" \
  --selection-provenance-sha256 "${SELECTION_SHA256}" \
  --registered-radii-nm "${RADII[@]}" \
  --out "${OUT_ROOT}/audit"

grep -qx 'PASS_246CUBE_THREE_SEED_STATIC_QUALIFICATION_V1' \
  "${OUT_ROOT}/audit/status.txt"
{
  printf 'status=PASS_246CUBE_QUARTER_NM_HANDOFF_STATIC_V1\n'
  printf 'profile_count=%s\n' "${#RADII[@]}"
  printf 'library_manifest_sha256=%s\n' "${LIBRARY_SHA256}"
  printf 'selection_provenance_sha256=%s\n' "${SELECTION_SHA256}"
  printf 'inventory_selection_sha256=%s\n' \
    "$(sha256sum "${INVENTORY_SELECTION}" | awk '{print $1}')"
  printf 'source_tree_sha256=%s\n' "${SOURCE_TREE_SHA256}"
  printf 'target_mean_C_B_tot=0.03\n'
  printf 'profile_scaling_used=false\n'
  printf 'profile_interpolation_used=false\n'
} >"${OUT_ROOT}/status.txt"
cat "${OUT_ROOT}/status.txt"
