# KWN–PF MVP v1 final acceptance

Top-level status: `PARTIAL_PF_STATE_NOT_CLOSED`

| question | evidence-grounded answer |
|---|---|
| 1. KWN backend | `INTERNAL_KWN_BACKEND_SELECTED`; Kawin was unavailable/unpinned, while the internal finite-volume backend is fully owned and tested. |
| 2. Strict KWN conservation | Yes for the implemented solver: N1–N7, N8 observation mapping, and the all-state ledger invariant pass; residual target is <= `1e-10` and the handoff package residual is `0`. |
| 3. beta-only PF consistency | `P0_CONTRACT_CONFLICT`: no valid same-contract PF/KWN comparison was run. |
| 4. source of beta-only differences | Not determinable yet; thermodynamic authority, PF-consistent diffusivity, elastic/spatial competition, and full PF PSD inputs are not simultaneously available. |
| 5. 6 h GP-like population | Prescribed source runs; effective-CNT has 1 soft-constraint hit(s) of 64, not a predictive calibration. |
| 6. unidentifiable GP parameters | `xB_g`, gamma_g, site density, attachment, diffusivity scale, and elastic penalty remain non-identified exploratory parameters. |
| 7. full 6 h mapping | No.  The matrix/resolved-beta portions have declared targets, but the complete four-bucket state is not PF-closed. |
| 8. unrepresentable inventory | GP inventory is non-zero in the emitted package (`7.0843227289824483e+01` mol B m⁻³); PF also lacks a persistent sub-grid beta state even though that bucket is zero for this prescribed source. |
| 9. PF pulse/seed dissolution | `NOT_RUN_P0_CONTRACT_CONFLICT`: no PF run occurred, so no transient improvement or degradation is claimed. |
| 10. next-step decision | Resolve/hash-bind the PF thermodynamic contract and specify independent PF GP/sub-grid-beta state variables before any PF smoke run. Do not start a larger PF case or concurrent coupling. |

## Handoff evidence

The v1 package was generated from commit `369867741ebfb6b3b1867b3ec4e7727994e284a4` with source config hash `2aa878559ca897fdbb80b384791280eac2d642dc6b175b116fb109abcfe8da43`. Its audited status is `PARTIAL_PF_STATE_NOT_CLOSED`, package-level relative residual `0.000e+00`, and PF raw initialization was not emitted.
