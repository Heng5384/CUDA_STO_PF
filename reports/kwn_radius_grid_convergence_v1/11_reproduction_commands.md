# Reproduction commands

```bash
PYTHONPATH=src python3 scripts/run_kwn_radius_grid_convergence_v1.py baseline
PYTHONPATH=src python3 scripts/run_kwn_radius_grid_convergence_v1.py initial-audit
for bins in 100 200 400 800 1600; do
  PYTHONPATH=src python3 scripts/run_kwn_radius_grid_convergence_v1.py run-grid --scenario fixture --bins $bins --schedule uniform --run-id ladder_fixture_${bins}_uniform
  PYTHONPATH=src python3 scripts/run_kwn_radius_grid_convergence_v1.py run-grid --scenario smooth --bins $bins --schedule uniform --run-id ladder_smooth_${bins}_uniform
done
PYTHONPATH=src python3 scripts/run_kwn_radius_grid_convergence_v1.py inspect
# Run 3200 fixture/smooth only if inspect declares it required.
PYTHONPATH=src python3 scripts/run_kwn_radius_grid_convergence_v1.py requalify
PYTHONPATH=src python3 scripts/run_kwn_radius_grid_convergence_v1.py compare-pf
PYTHONPATH=src python3 scripts/run_kwn_radius_grid_convergence_v1.py assemble
```
