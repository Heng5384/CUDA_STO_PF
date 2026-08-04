# 400³ fixed-χB=0.03 elastic PSD-sign pilot: completion contract

This document defines the evidence required before the running V5 pilot may be
called complete.  It is an audit checklist only; it does not alter the PF
binary, fixture, parameters, checkpoint chain, or physical model.

## Frozen identity

| Item | Required evidence |
|---|---|
| Production root | `/data/home/luozhiheng/tmp/pf_400cube_6h48h_fixed_xb03_elastic_psd_sign_pilot_v5_20260802` |
| Job chain | Production `73850`, read-only tail `73851` |
| Grid and physical system | 400³, `dx=1 nm`, 380 °C, `mean_C_Btot=0.03` |
| Source terms | GP, GP Birth, GP release, external source, and new β nucleation all disabled |
| Elasticity | `ELASTIC_WARM_START_RESIDUAL_V1` enabled |
| Time path | `dt_code=0.02`; 6 h to 48 h; final step `152585` |
| Fixture hash | `b8b59357b659078e51077330f4fe1c31caf302c2605e809c71d8e41b18771b13` |
| Profile-library hash | `a1b0dba740113cdfa97cd86c762030fe190fe23036487a82afff6df7770118da` |
| Parameter hash | `3571c917cb5696f85ff5a2dbe9d43e7c7ed6c1634c1b13f9a2777a0b2aebadc3` |
| Binary hash | `759956a89780db9a19ccd51b4115319de47463a3fd7b8d446a022a74c9beeb3b` |

## Required completion evidence

| Requirement | Authoritative artifact / exact condition | Status while job is running |
|---|---|---|
| Complete trajectory | Root `status.txt` is exactly `PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_PILOT_V1` | Pending |
| End point | 44 checkpoint/science endpoints, including final `step_152585` | Pending |
| Deterministic provenance | `provenance/input_hashes.sha256`, `campaign_manifest.json`, environment/device record, and `analysis_hashes.sha256` | Partially frozen; final chain pending |
| Checkpoint integrity | `provenance/checkpoint_chain.sha256` validates every checkpoint | Pending |
| Solver integrity | Every segment has exact segment PASS, empty stderr, zero-mode PASS, and restart provenance after the first segment | Pending |
| Conservation/bounds | Final audit gates `mass_conservation`, `zero_mode`, and `finite_bounds` are true | Pending |
| Identity | Three-threshold periodic merge-aware audit is PASS; unresolved merge/split remains fail-closed | Pending |
| Global microstructure | Hourly full PSD, `N_v`, mean radius, `S_v`, `M_6`, β fraction, and far-field matrix Ag are emitted | Pending |
| No-dislocation transport | `transport/status.txt` is exactly `PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_TRANSPORT_V1` | Pending |
| Required transport modes | 300 K full-PSD calculation at fixed 6 h matrix and PF time-varying matrix; `A_N=1.5`, `S11=S13=0`, no Yu refit scale | Pending |
| PSD sufficiency | Direct full PSD is compared with `N_v+R`, `S_v`, and `S_v+M_6` | Pending |
| Scientific sign | `delta_kappa_PSD_300K` is reported as positive, negative, or zero within numerical precision without preselection | Pending |
| Figures | 6 h/48 h PSD comparison and 6–48 h 300 K conditional κ trajectory | Pending |

## Scope of a final PASS

An exact pilot PASS establishes one direct, 400 nm, full-elastic conditional
trajectory.  The user explicitly authorized skipping one-step/restart and
short-physical smoke qualification for this pilot.  Therefore a PASS must not
be described as box-size convergence, ensemble qualification, an absolute
experimental thermal-conductivity reconstruction, or a prediction of first
β nucleation.
