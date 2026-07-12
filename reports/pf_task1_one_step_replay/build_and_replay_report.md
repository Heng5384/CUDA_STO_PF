# Task 1 Build and Replay Report

- Clean pre-instrument build: PASS (`nvcc 12.9.86`, `sm_120`, RTX 5080).
- Host storage formula test: PASS.
- Thermodynamic directional derivative test: PASS (`mu` error `2.12e-10`, fixed-C phase derivative error `9.85e-11`).
- Pre-instrument 8^3 PF-only L run: completed; exposed `-3.840808e-2` relative one-step storage drift.
- Post-instrument build: PASS; only pre-existing unused-symbol and thermo extern warnings.
- Post-instrument `make test` memory-ledger smoke: PASS.
- Post-clean-build L/X/Q replay script: PASS (all three runs completed and emitted ten trace stages).
- Trace-off primary field equivalence: PASS, byte-identical `phi_1.vtk`, `xB_1.vtk`, and `xBtot_1.vtk`.
- L replay: first failure S5; accepted with mass loss.
- X replay: first failure S5; accepted with mass loss.
- Q replay: S5 proposal infeasible, transaction rollback; final mass conserved to roundoff.
- Architecture validation claim: NOT MADE.
