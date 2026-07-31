# Profile A/B materialization diagnostic (T380, GP OFF)

This diagnostic compares the clean-branch analytic seed at step 0 (profile A)
with the separately generated full-model-minimize output (profile B candidate)
on workstation `fuxin`.  Both fields are 128^3, centered at `(64,64,64)` nm,
with the same nominal R=8 nm geometry and the frozen analytic far-field
composition.  No production parameter was changed.

The comparison is intentionally **not** a dynamic A/B qualification: the
clean commit `6b69895af2d1b86b99c57c5479ff767349c61efe` rejects
`PF_CONSERVED_Y_ZERO_MODE_V1` combined with minimize/raw/VTK initialization,
so profile B cannot be handed to the conserved zero-mode dynamics without a
new validated materialization contract.

## Direct field comparison

| quantity | A: analytic seed | B: full-model minimize |
|---|---:|---:|
| mean phi | 0.00118035662651062 | 0.00112323957443237 |
| mean h(phi) | 0.00106051784447337 | 0.00102265319223610 |
| mean xB | 0.00466000000000002 | 0.00466870992660523 |
| phi range | [0, 0.99966] | [0, 0.99632] |
| xB range | [0.00466, 0.00466] | [0.00466, 0.00565] |

Absolute differences are `mean(|Δphi|)=5.85536e-5`, `max(|Δphi|)=0.02746`,
`mean(|ΔxB|)=8.70993e-6`, and `max(|ΔxB|)=9.90e-4`.  The fraction of voxels
with `|Δphi|>0.01` is `2.50530e-3`; the mean h-volume changes by `-3.57039%`.

These numbers demonstrate that the minimized profile is a measurable profile
variant, not byte-identical initialization.  They do not establish which
profile is physically preferred, because the B field was generated in a
different (non-zero-mode) minimization epoch.

## Provenance

- branch: `codex/pf-zero-mode-restart-provenance-v1`
- commit: `6b69895af2d1b86b99c57c5479ff767349c61efe`
- clean branch binary: SHA-256 `11d073a272a0b7fa668e40037bc1f96b8389909f95c74d6ad9a5db11236e01af`
- complete workstation JSON (including radial profiles):
  `/home/zhiheng/tmp/codex_equilibrium_audit_v1_branch_20260729/profile_AB_materialization_diagnostic.json`

Status remains `BLOCKED_PROFILE_EQUILIBRATION_METHOD` for the requested
zero-mode dynamic A/B and V2-B/C runs.  This artifact is evidence for the
method decision, not a production initialization file.
