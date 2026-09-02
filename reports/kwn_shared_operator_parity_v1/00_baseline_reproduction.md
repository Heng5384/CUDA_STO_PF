# Baseline reproduction

Status: `PASS_BASELINE_REPRODUCTION`.  The isolated rerun reproduced every registered CSV metric and all full-time maximum errors exactly. The observed validation contract is `d0ff02973ab0f737043e1a40d4f69893a469cbfe2bc4cd22f9e6a410bd0b1333`.

The solver-config hash differs only because the isolated worktree has a different absolute contract path; the sorted metric CSV and all registered errors are byte-identical.

| Metric | maximum relative error | time (h) |
|---|---:|---:|
| M0_m3 | 2.43207713% | 48 |
| M1_m2 | 1.11022839% | 48 |
| M2_m | 0.535822292% | 48 |
| M3_dimensionless | 0.0222298019% | 48 |
| N_m0_m3 | 2.43207713% | 48 |
| Rmean3_m3 | 2.39603355% | 48 |
| Rmean_m | 1.29046367% | 48 |
| Sv_m_inv | 0.535822292% | 48 |
| cumulative_dissolution_inventory_mol_m3 | 21.2187061% | 0.1 |
| f_beta | 0.0222298019% | 48 |
| matrix_xB | 0.108397066% | 48 |
