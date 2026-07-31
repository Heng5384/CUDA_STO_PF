# 256-step dynamic and restart validation

Each replicate used the frozen production timestep:

```text
dt_code=0.02
dt_physical_s=0.9909260953431841
continuous=256 steps
restart=128 steps + checkpoint + 128 steps
physical_time_s=253.67708040785513
endpoint_age_h=6.070465855668848
```

The continuous and restarted final checkpoint files are bytewise identical
within every replicate. This equality covers `phi`, `Y`, `xB`,
`dY_dt_prev`, the zero-mode state, elastic runtime state, target ledger, and
checkpoint provenance.

| replicate | particle count | beta volume fraction | mean radius (nm) | \(S_v\) (nm⁻¹) | \(M_6\) (nm³) | far-field \(x_{Ag}\), \(d\ge3\lambda\) |
|---|---:|---:|---:|---:|---:|---:|
| A | 96 | 0.0225777602281 | 9.33943615104 | 0.00712954427026 | 4.84818788259 | 0.00755365665966 |
| B | 96 | 0.0225637485232 | 9.33723719752 | 0.00712635948808 | 4.84428546982 | 0.00757055495536 |
| C | 96 | 0.0225876191405 | 9.34156608320 | 0.00713217640067 | 4.84971883615 | 0.00754366682605 |

The registered far-field result uses `phi<0.5` and exact periodic Euclidean
distance \(d\ge3\lambda=12\) nm. The \(d\ge4\lambda\) comparator is also
retained in the machine-readable trajectory. The periodic distance
implementation was checked against SciPy's tiled Euclidean distance transform
and matched the eligible fractions and concentration means to floating-point
roundoff.

| replicate | normalized \(\phi\) L1 from initial | full-field \(x_B\) MAE | mass relative error | continuous checkpoint SHA-256 |
|---|---:|---:|---:|---|
| A | 0.0350945671851 | 0.000788860763693 | 6.47753786138e-14 | `6ee3fb4912740af2b6e72a9a3e848af3c35139bcccf99cd88fb3dc44fb9dd8e8` |
| B | 0.0358409778077 | 0.000803067417791 | 2.80215420563e-13 | `d3dd2c8e185ad234ad459630227aaa191b1bb3d44f26289e5a28d13a39c3f9d3` |
| C | 0.0343798709959 | 0.000778912201505 | 1.06090861553e-13 | `ed0a5039d205e622bd6cbd0decb7bd309f6907f694857bb3c9d0b7deccae0175` |

No particle disappeared, merged, split, or appeared without an overlap parent
during this gate. All final fields were finite and within the solver
contract; GP, external-source, and new-beta event markers were absent.

```text
restart_status=PASS_BYTEWISE_ALL_THREE
particle_identity_status=PASS_96_OF_96_ALL_THREE
mass_and_zero_mode_status=PASS
```
