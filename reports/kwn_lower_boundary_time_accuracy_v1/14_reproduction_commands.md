# Reproduction commands

```bash
# Finite lower-boundary gates and common 0--0.1 h active-CFL ladder
PYTHONPATH=src python3 scripts/run_kwn_lower_boundary_time_accuracy_v1.py all

# Explicitly authorize the expensive exact canonical cohort path
PYTHONPATH=src python3 scripts/run_kwn_lower_boundary_time_accuracy_v1.py crosscheck --allow-long-cohort --cohort-points-per-cell 2
```

Executed command: `scripts/run_kwn_lower_boundary_time_accuracy_v1.py all --max-wall-s 140 --max-steps 250000`.  This runner does not invoke CUDA/PF.
