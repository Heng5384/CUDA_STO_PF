# Reproduction commands

```bash
PYTHONPATH=src python3 -m unittest tests.kwn.test_population_metrics tests.kwn.test_cohort_solver tests.kwn.test_rmin_boundary
BASELINE_OUTPUT_ROOT=/absolute/path/to/isolated/outputs/kwn_discrete_cohort_comparison_v1
PYTHONPATH=src python3 scripts/run_kwn_shared_operator_parity_v1.py all --baseline-output-root "$BASELINE_OUTPUT_ROOT"
```

This run used `/Users/heng/Documents/GitHub/CUDA_STO_PF-shared-op-baseline.LKWl2o/outputs/kwn_discrete_cohort_comparison_v1` as its isolated exact-replay evidence.  The baseline command is read-only with respect to frozen historical artifacts; this task does not run CUDA/PF or alter physical inputs.
