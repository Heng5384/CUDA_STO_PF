# Baseline reproduction

Status: `PASS_RADIUS_GRID_BASELINE_REPRODUCTION`. The legacy unsafe diagnostic was replayed from the frozen initial entries; production transport remained the repaired implicit face solve.

| Check | Result |
| --- | --- |
| Strict legacy failure | PASS |
| Repaired historical P5 | PASS |
| 200 schedule | sequential 0/0.1/1/3/6/12/24/48 h |
| 400 schedule | direct 0→48 h |

The legacy acceptance comparison uses a predeclared 5e-14 relative tolerance, because the frozen replay differs only at floating-point ulps while failure step, bin and timestep are exact.

Recorded baseline contract: solver `CONSERVATIVE_IMPLICIT_UPWIND_FACE_SOLVE`, source-config hash `88d2398bf5ca28e6d40f3f1807ceb5db8550374bcf66640415719f91d12fd88d`, Rmin `4.72403e-10` m, Rmax `1e-07` m, 200 radius bins, radius-edge SHA-256 `81b66983cd80d131c48b538c932a12f36125bc8aff10d0c2db7209f82a613c41`, fixture PSD source hash `7a3ab71c5a8ad622a456ddd2cdcd7fa215a91ff4fd4121a64f2a5ff46a849827`, and lower-bound treatment `CONSERVATIVE_IMPLICIT_UPWIND_RMIN_OUTFLOW; global ledger recovers matrix inventory from fixed-pivot beta M3 without a clamp`. The run manifest records the exact timestep policy and output moment definitions.
