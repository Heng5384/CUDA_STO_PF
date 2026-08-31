# Reproduction commands

Run from the isolated worktree root.  The configuration files are JSON-subset YAML because the local environment has no PyYAML dependency.

```bash
cd /Users/heng/Documents/GitHub/CUDA_STO_PF-kwn-pf-mvp-v1
PYTHONPATH=src python3 -m unittest discover -s tests/kwn -v
PYTHONPATH=src python3 -m unittest discover -s tests/coupling -v
PYTHONPATH=src python3 scripts/run_beta_only_consistency.py
PYTHONPATH=src python3 scripts/run_gp_feasibility_sweep.py
PYTHONPATH=src python3 scripts/build_kwn_pf_handoff.py
python3 scripts/run_pf_handoff_smoke.py
PYTHONPATH=src python3 scripts/make_kwn_pf_report.py
```

The PF smoke wrapper is an authority gate.  A `NOT_RUN_*` result is the intended reproducible outcome while the P0 contract conflict and partial state closure remain.
