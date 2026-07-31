# Multi-particle fixture V2 completion audit

This is the V1-goal evidence folder with a versioned V2 composition-contract
addendum. The V1 direct `delta_C` sum is preserved as failed evidence: its
six-profile reconstruction produced invalid `xB` and would have required
clipping. No V1 evidence was overwritten.

## Frozen inputs

| item | value |
|---|---|
| selected library manifest | `58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe` |
| selection provenance | `56c44d8f72b27bb462dffe89b59cb2fcb2ff0bf8807dec9d9cac31d2bd7fcbe3` |
| source tree | `f7855699addf98f9d5aed03af876d851d524fb62c98f5a48df78d1fe561a2a75` |
| binary | `55cf917df94fcf01373d62f54f9ab99715975863ad5dcfb95baa8a517460cda1` |
| physical contract | 380 C, 96³ at 1 nm, lambda 4 nm, periodic fixed-cell elasticity, identity orientation, GP/source/new nucleation off |

## V2 field construction

The phase union is `phi=1-product(1-phi_j)`. Each profile contributes only
`delta_x_j=xB_alpha_j-xB_far_j`; E2 uses
`C=h+(1-h)*(xB_matrix+sum(delta_x_j))`. Absolute `xB` is never superposed.
There is no interpolation, rotation, scaling, or E2 clipping.

The six-particle engineering fixture uses radii 8.0 nm (2), 9.5 nm (2), and
10.5 nm (2), hash-pinned integer-grid placement seed `2026073091`. Its closest
periodic pair has 8 nm separation margin. It is not an experimental 6 h PSD
candidate.

The E2 V2 fixture manifest SHA-256 is
`bbac0b9ce0521fb525f76d7bea90da1fc6352cb5bac62cda86a06d27d5e54093`; the
S0 sensitivity fixture is
`8b9b9ddf4c4b7fda2f701b14451e916682519a0123d86ea3623a20d767a0412f`.
Both have exact total inventory 26542.08, relative error
`1.3706457094137737e-16`, beta volume fraction `0.0239282691204922`, and
matrix `xB=0.006222777982325975` (`xAg=0.00620347655366995`).

S0 uses an explicit conservative phase-storage projection only because its
analytic core has a different alpha geometry. It preserves the E2 correction
sum exactly, discards zero mass, and records 390 bounded storage cells.

## Gates

| stage | outcome |
|---|---|
| static V2 contracts | PASS, 19/19 |
| deterministic order/materialization | PASS bytewise |
| E2/S0 matched inventory, h volume, baseline | PASS |
| E2 one-step raw-field handoff | PASS; min overlap 0.9952431 |
| mass / zero mode / prohibited paths / clipping | PASS |
| restart | PASS bytewise; endpoint SHA `ec81328bba95ac1864b1b40c359b4bbcc2555e30c62b98ada2f068950b075963` |
| six-particle identities | PASS; no merge or split |
| finite elastic diagnostic provenance | PASS |
| dt 0.02 vs 0.01 | FAIL only in xB: MAE `1.114535459033156e-4` > `5e-5` |

The phi L1 (`8.263201696777277e-4`) and maximum semi-axis difference
(`3.926911327320605e-4`) pass. The xB discrepancy is also present in the
far matrix (`1.1499033009521266e-4`), so it cannot be dismissed as a beta-core
storage artifact. A read-only 0.01 vs 0.005 diagnostic still gives
`1.0431280129344037e-4` xB MAE.

S0/E2 high-cadence startup and 6–12 h preflight were not launched because the
pre-registered dt gate failed. No full 6–48 h production, commit, push,
physical retuning, or GP path occurred. Materialization used frozen cluster
artifacts; dynamics used the workstation.
