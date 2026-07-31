#!/usr/bin/env bash
# Non-overwriting 246^3 elastic-on continuation from the 72834 12 h checkpoint to 48 h.
set -euo pipefail

OUT=/data/home/luozhiheng/tmp/pf_random_psd_v2_246x96_elastic_12h48h_20260730
SRC=/data/home/luozhiheng/tmp/codex_pf_random_psd_v2_fixture_ingress_v1_20260730
CHK=/data/home/luozhiheng/tmp/pf_random_psd_v2_246x96_elastic_6h12h_dt0p02_20260730/checkpoint_6h_to_12h_elastic.chk
PARAM=/data/home/luozhiheng/tmp/pf_random_psd_v2_preflight_246x96_6h12h_20260730/pf_input.params

mkdir -p "$OUT"
test -s "$CHK"
cp "$SRC/main_cuda" "$OUT/main_cuda"
cp "$PARAM" "$OUT/pf_input.params"
sha256sum "$OUT/main_cuda" "$OUT/pf_input.params" "$CHK" > "$OUT/input_hashes.sha256"
cd "$OUT"
exec "$OUT/main_cuda" 246 246 246 0.02 152600 3633 256 1 \
  --pf-param-file "$OUT/pf_input.params" \
  --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1 \
  --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1 \
  --pf-zero-mode-tol-rel 1e-12 \
  --pf-zero-mode-max-iter 24 \
  --pf-restart-from "$CHK" \
  --pf-checkpoint-every 3633 \
  --pf-checkpoint-path "$OUT/checkpoint_12h_to_48h_elastic.chk" \
  --enable-dynamics-mass-diagnostics \
  --dynamics-mass-diag-interval 256
