# KWN-to-PF handoff mass audit

Status: `PARTIAL_PF_STATE_NOT_CLOSED`

## Package

- schema: `kwn_pf_handoff_v1`
- package: `outputs/kwn_pf_mvp_v1/kwn_pf_handoff_6h`
- inventory basis: `mol_B_per_m3`
- PF raw initialization emitted: `false`
- PF raw initialization allowed: `false`

## Four-bucket ledger

| bucket | mol B m⁻³ | destination |
|---|---:|---|
| `C_B_matrix` | 1.2010356487774848e+02 | PF matrix xB_alpha |
| `C_B_GP` | 7.0843227289824483e+01 | package-only retained inventory |
| `C_B_beta_subgrid` | 0.0000000000000000e+00 | package-only retained inventory |
| `C_B_beta_resolved` | 0.0000000000000000e+00 | PF resolved-beta geometry |
| `C_B_total` | 1.9094679216757297e+02 | source total |

Package-level relative accounting residual: `0.000e+00`.

The ledger closes only at package level when status is `PARTIAL_PF_STATE_NOT_CLOSED`: unmapped GP/sub-grid beta material remains explicitly retained in the package and is not transferred into `xB_alpha` or resolved `phi`.  This package therefore does not authorize a PF/CUDA run.

## Provenance

`metadata.json` records the source commit, KWN configuration hash, backend, analysis-script hash, fixture hash, and the explicit absence of a compiled KWN binary.
