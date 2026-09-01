# 12 Reproduction commands

Run from the isolated worktree and use a new non-overwriting output root for every CUDA submission.

```bash
cd /Users/heng/Documents/GitHub/CUDA_STO_PF-kwn-pf-cuda-runtime-closure-v1
PYTHONPATH=src python3 -m unittest tests.kwn.test_beta_only_same_contract_control \
  tests.kwn.test_conservative_positivity_repair tests.kwn.test_rmin_boundary \
  tests.kwn.test_numerical_gates
PYTHONPATH=src python3 scripts/diagnose_kwn_positivity_failure.py \
  --output-root outputs/kwn_pf_cuda_runtime_closure_v1
PYTHONPATH=src python3 scripts/run_kwn_conservative_qualification.py \
  --output-root outputs/kwn_pf_cuda_runtime_closure_v1/kwn_conservative_qualification

# After a fresh, clean CUDA R4 job completes and compact audit is copied locally:
PYTHONPATH=src python3 scripts/render_kwn_pf_cuda_runtime_closure_v1.py \
  --cuda-run-root /Users/heng/Documents/GitHub/CUDA_STO_PF-kwn-pf-cuda-runtime-closure-v1/outputs/kwn_pf_cuda_runtime_closure_v1/cuda_ae_cluster_r4_097a69582ca3_20260901T080254Z \
  --output-root outputs/kwn_pf_cuda_runtime_closure_v1 \
  --report-root /Users/heng/Documents/GitHub/CUDA_STO_PF-kwn-pf-cuda-runtime-closure-v1/reports/kwn_pf_cuda_runtime_closure_v1
```

Do not run the beta-only KWN–PF comparator unless both prerequisite statuses are exact `PASS_CUDA_AE_SMOKE` and `PASS_KWN_CONSERVATIVE_POSITIVITY`; this evidence package intentionally writes a `NOT_RUN` row instead.
