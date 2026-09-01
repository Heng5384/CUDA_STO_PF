# Read-only baseline reproduction

Read-only source: `/Users/heng/Documents/GitHub/CUDA_STO_PF-kwn-radius-grid-convergence-v1/outputs/kwn_radius_grid_convergence_v1`.  The report-producing clean commit is `2f67a34751af1312eaf63f635f4a5be9166bf03a`; the older `git_head` in runtime provenance is retained only as the disclosed launch-time baseline.

Status: `PASS_RADIUS_GRID_BASELINE_REPRODUCTION`.

| Scenario | 1600→3200 N | Rmean | Rmean3 | Sv | fβ | matrix xB |
|---|---:|---:|---:|---:|---:|---:|
| smooth | 1.34708% | 0.358792% | 1.3747% | 0.53817% | 0.00910839% | 0.0444147% |
| fixture | 7.6838% | 2.5068% | 8.39424% | 2.78188% | 0.0654406% | 0.314198% |

The immutable 48 h positivity/conservation status remains `PASS_KWN_POSITIVITY_CONSERVATION_48H`, with maximum residual `1.554061176041917e-16`.
