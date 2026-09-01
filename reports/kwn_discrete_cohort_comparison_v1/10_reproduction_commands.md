# Reproduction commands

From this worktree:

```bash
PYTHONPATH=src python3 -m unittest tests.kwn.test_cohort_solver
python3 scripts/run_kwn_discrete_cohort_comparison_v1.py all --radius-grid-output-root /Users/heng/Documents/GitHub/CUDA_STO_PF-kwn-radius-grid-convergence-v1/outputs/kwn_radius_grid_convergence_v1
```

The second command reads the previous radius-grid artifacts only; it does not rerun CUDA/PF A–E or change their sources.
