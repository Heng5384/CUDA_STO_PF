#!/usr/bin/env bash
#SBATCH --job-name=pf246_merge_audit
#SBATCH --partition=serial_pub_3h
#SBATCH --qos=serial_pub_3h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=/data/home/luozhiheng/tmp/pf246_merge_audit_%j.out
#SBATCH --error=/data/home/luozhiheng/tmp/pf246_merge_audit_%j.err

set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT is required}"
FIXTURE_ROOT="${FIXTURE_ROOT:?FIXTURE_ROOT is required}"
STAGE7_ROOT="${STAGE7_ROOT:?STAGE7_ROOT is required}"
SCREEN_ROOT="${SCREEN_ROOT:?SCREEN_ROOT is required}"
AUDIT_ROOT="${AUDIT_ROOT:?AUDIT_ROOT is required}"

cd "${SOURCE_ROOT}"
source jobs/_site_env.sh
exec env \
  SOURCE_ROOT="${SOURCE_ROOT}" \
  FIXTURE_ROOT="${FIXTURE_ROOT}" \
  STAGE7_ROOT="${STAGE7_ROOT}" \
  SCREEN_ROOT="${SCREEN_ROOT}" \
  AUDIT_ROOT="${AUDIT_ROOT}" \
  TRACKER_LDFLAGS="-lstdc++fs" \
  bash jobs/run_pf_246cube_merge_aware_audit_v1.sh
