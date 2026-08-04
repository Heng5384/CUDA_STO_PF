#!/usr/bin/env bash
set -euo pipefail

# Assemble a non-overwriting selected library from already-qualified entries.
# This is intended for a fail-closed run in which a later radius needed only
# a larger iteration budget.  It never promotes a non-PASS profile.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"

SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:?BASE_RUN_ROOT is required}"
EXTENSION_RUN_ROOT="${EXTENSION_RUN_ROOT:?EXTENSION_RUN_ROOT is required}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
BASE_RADII_NM="${BASE_RADII_NM:-8.0 8.5 9.0 9.5 10.0 10.5 11.0}"
EXTENSION_RADII_NM="${EXTENSION_RADII_NM:-11.5}"
EXPECTED_RADII_NM="${EXPECTED_RADII_NM:-8.0 8.5 9.0 9.5 10.0 10.5 11.0 11.5}"
PROFILE_COUNT="$(wc -w <<<"${EXPECTED_RADII_NM}" | tr -d ' ')"
ALLOW_MIXED_BINARY_SAME_SOURCE="${ALLOW_MIXED_BINARY_SAME_SOURCE:-0}"

if [[ -e "${OUT_ROOT}" ]]; then
  echo "[fatal] refusing to overwrite OUT_ROOT: ${OUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUT_ROOT}/profiles" "${OUT_ROOT}/provenance"

profile_dirs=()
copy_profile() {
  local source_run="$1"
  local radius="$2"
  local tag="R${radius//./p}"
  local source_profile="${source_run}/profiles/${tag}"
  local target_profile="${OUT_ROOT}/profiles/${tag}"
  if [[ ! -f "${source_profile}/profile_manifest.json" ]] ||
     ! grep -q \
       '^profile_status=PASS_ELASTIC_CONSTRAINED_TARGET_PROFILE_V1$' \
       "${source_profile}/final_terminal_output.txt"; then
    echo "[fatal] source profile has no exact PASS: ${source_profile}" >&2
    exit 2
  fi
  if [[ -e "${target_profile}" ]]; then
    echo "[fatal] duplicate selected radius: ${radius}" >&2
    exit 2
  fi
  cp -a "${source_profile}" "${target_profile}"
  profile_dirs+=("${target_profile}")
  {
    printf 'radius_nm=%s\n' "${radius}"
    printf 'source_profile=%s\n' "${source_profile}"
    printf 'source_profile_manifest_sha256=%s\n' \
      "$(sha256sum "${source_profile}/profile_manifest.json" | awk '{print $1}')"
    printf 'selected_profile_manifest_sha256=%s\n' \
      "$(sha256sum "${target_profile}/profile_manifest.json" | awk '{print $1}')"
  } >"${OUT_ROOT}/provenance/${tag}.source.txt"
}

for radius in ${BASE_RADII_NM}; do
  copy_profile "${BASE_RUN_ROOT}" "${radius}"
done
for radius in ${EXTENSION_RADII_NM}; do
  copy_profile "${EXTENSION_RUN_ROOT}" "${radius}"
done

assembler_args=(
  --profiles "${profile_dirs[@]}"
  --expected-radii-nm ${EXPECTED_RADII_NM}
  --out "${OUT_ROOT}/library"
)
if [[ "${ALLOW_MIXED_BINARY_SAME_SOURCE}" == "1" ]]; then
  assembler_args+=(--allow-mixed-binary-same-source)
fi
python3 "${SOURCE_ROOT}/scripts/assemble_pf_elastic_target_profile_library_v1.py" \
  "${assembler_args[@]}" \
  >"${OUT_ROOT}/library_assembly.stdout" \
  2>"${OUT_ROOT}/library_assembly.stderr"
if [[ -s "${OUT_ROOT}/library_assembly.stderr" ]]; then
  echo "[fatal] selected-library assembler produced stderr" >&2
  exit 2
fi
grep -q '^library_status=PASS_ELASTIC_TARGET_PROFILE_LIBRARY_ASSEMBLY_V1$' \
  "${OUT_ROOT}/library/final_terminal_output.txt"

python3 - \
  "${BASE_RUN_ROOT}" \
  "${EXTENSION_RUN_ROOT}" \
  "${OUT_ROOT}/library/library_manifest.json" \
  "${OUT_ROOT}/provenance/selection_provenance.json" <<'PY'
import hashlib
import json
import pathlib
import sys

base, extension, library_path, out = map(pathlib.Path, sys.argv[1:])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

library = json.loads(library_path.read_text())
payload = {
    "schema": "PF_ELASTIC_TARGET_PROFILE_SELECTION_PROVENANCE_V1",
    "selection_status": "PASS_REUSE_ONLY_INDIVIDUALLY_QUALIFIED_PROFILES",
    "base_run_root": str(base),
    "base_run_status": (
        (base / "status.txt").read_text().strip()
        if (base / "status.txt").is_file()
        else "NO_LIBRARY_LEVEL_PASS_PARTIAL_RUN_PRESERVED"
    ),
    "extension_run_root": str(extension),
    "extension_run_status": (extension / "status.txt").read_text().strip(),
    "library_manifest_sha256": sha256(library_path),
    "source_commit": library["source_commit"],
    "source_commit_set": library.get("source_commit_set", [library["source_commit"]]),
    "mixed_source_labels": library.get("mixed_source_labels", False),
    "source_tree_sha256": library["source_tree_sha256"],
    "binary_sha256": library["binary_sha256"],
    "binary_sha256_set": library.get("binary_sha256_set", [library["binary_sha256"]]),
    "mixed_binary_profiles": library.get("mixed_binary_profiles", False),
    "mixed_binary_contract": library.get("mixed_binary_contract", "SINGLE_BINARY"),
    "profile_manifest_sha256": {
        str(row["target_radius_nm"]): row["profile_manifest_sha256"]
        for row in library["profiles"]
    },
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

library_sha="$(
  sha256sum "${OUT_ROOT}/library/library_manifest.json" | awk '{print $1}'
)"
selection_sha="$(
  sha256sum "${OUT_ROOT}/provenance/selection_provenance.json" |
    awk '{print $1}'
)"
{
  printf 'runner_status=PASS_PF_ELASTIC_TARGET_PROFILE_LIBRARY_V1\n'
  printf 'assembly_mode=REUSE_QUALIFIED_PROFILES_WITH_EXTENDED_ITERATION_BUDGET\n'
  printf 'profile_count=%s\n' "${PROFILE_COUNT}"
  printf 'mixed_binary_same_source_allowed=%s\n' \
    "${ALLOW_MIXED_BINARY_SAME_SOURCE}"
  printf 'radii_nm=%s\n' "${EXPECTED_RADII_NM}"
  printf 'elastic_enabled=true\n'
  printf 'gp_enabled=false\n'
  printf 'library_manifest_sha256=%s\n' "${library_sha}"
  printf 'selection_provenance_sha256=%s\n' "${selection_sha}"
} >"${OUT_ROOT}/status.txt"
cat "${OUT_ROOT}/status.txt"
