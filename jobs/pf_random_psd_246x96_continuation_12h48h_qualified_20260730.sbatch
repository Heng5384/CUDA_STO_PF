#!/usr/bin/env bash
# Qualified PF-only random-PSD V2 continuation, 12 h -> 48 h.
# The binary, parameters, fixture, and checkpoint are the same provenance as
# the completed 6 h -> 12 h preflight; elasticity and all source paths remain off.
set -euo pipefail

OUT=/data/home/luozhiheng/tmp/pf_random_psd_v2_246x96_continuation_12h48h_qualified_20260730
SRC=/data/home/luozhiheng/tmp/codex_pf_random_psd_v2_246x96_6b69895_20260730
CHK=/data/home/luozhiheng/tmp/pf_random_psd_v2_preflight_246x96_6h12h_20260730/continuous_final.chk
PARAM=/data/home/luozhiheng/tmp/pf_random_psd_v2_preflight_246x96_6h12h_20260730/pf_input.params
FIXTURE=/data/home/luozhiheng/tmp/codex_pf_random_psd_v2_246x96_6b69895_20260730/data/qualification/pf_6h_48h_random_psd_v2_246x96

mkdir -p "$OUT"
test -s "$SRC/main_cuda"
test -s "$CHK"
cp "$SRC/main_cuda" "$OUT/main_cuda"
cp "$PARAM" "$OUT/pf_input.params"
cp "$FIXTURE/fixture_manifest.json" "$OUT/fixture_manifest.json"
sha256sum "$OUT/main_cuda" "$OUT/pf_input.params" "$OUT/fixture_manifest.json" "$CHK" > "$OUT/input_hashes.sha256"
cd "$OUT"
exec "$OUT/main_cuda" 246 246 246 0.02 152600 3633 256 0 \
  --pf-param-file "$OUT/pf_input.params" \
  --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1 \
  --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1 \
  --pf-zero-mode-tol-rel 1e-12 \
  --pf-zero-mode-max-iter 24 \
  --pf-restart-from "$CHK" \
  --pf-checkpoint-every 3633 \
  --pf-checkpoint-path "$OUT/checkpoint_12h_to_48h.chk" \
  --enable-dynamics-mass-diagnostics \
  --dynamics-mass-diag-interval 256
