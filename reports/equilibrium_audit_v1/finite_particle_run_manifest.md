# Finite-particle workstation run manifest

All runs in this manifest use the clean branch source `codex/pf-zero-mode-restart-provenance-v1` at commit `6b69895af2d1b86b99c57c5479ff767349c61efe`, binary SHA-256 `11d073a272a0b7fa668e40037bc1f96b8389909f95c74d6ad9a5db11236e01af`, and the frozen parameter template SHA-256 `e69620fab71521a9b28659ea05660e7545bb5dae24fd21a9e9074f241c78c3aa`. Each case made a private copy of that parameter template and changed only `ic_23d_xB_out` for the registered far-field bracket or the grid/radius command-line geometry. GP, birth, release and new-beta nucleation remained OFF.

| family | radii (nm) | boxes (nm) | endpoint | zero-mode |
|---|---|---|---|---|
| nonelastic bracket | 6, 8, 10, 12, 14, 16, 20 | 64, 128, 80, 96, 112, 128, 160 | 0.634–5.074 s | PASS |
| nonelastic box checks | 8, 10, 14 | 96/128/160, 80/96, 112/128 | 1.268 s | PASS |
| elastic bracket/center | 8, 10, 12, 14, 20 | 128, 80, 96, 112, 160 | 0.634–1.268 s | PASS |
| restart | R=8 | 128 | step 64 → 128 | PASS bytewise |
| dt refinement | nonelastic R=8; elastic R=10 | 128; 80 | equal physical endpoint | PASS short local |
| long dynamic extension | nonelastic R=8, low endpoint xAg=0.0058 | 128 | 40.588 s (8192 steps) | PASS zero-mode; dynamic sign only |

The authoritative output roots are under `/home/zhiheng/tmp/codex_equilibrium_audit_v1_branch_20260729/`; the per-case `run.log` files contain the startup contract, `R_input`, `vf_precip`, `R_avg_nm`, wall time and `PF_ZERO_MODE_FINAL_AUDIT`. The run windows are intentionally short dynamic windows. They establish conservation, restart and observed bracket behavior; they do not by themselves establish an equilibrium root.

The `dt=0.02` production timestep was not substituted for these tests: this audit resolves interface/profile transients and uses `dt=1e-4` (with a controlled `2e-4` refinement), while `dt=0.02` corresponds to about `0.990926 s` physical per macrostep under the frozen unit conversion.
