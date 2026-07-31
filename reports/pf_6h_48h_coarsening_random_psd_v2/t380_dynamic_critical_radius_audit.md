# T380 dynamic critical-radius audit

The exact-commit source-free bracket was completed on `gpu_uvip` using the
qualified T380 PF solver, `dx=1 nm`, `lambda_sm=4 nm`,
`matrix_xB_alpha=0.006219279767278563`, `dt_code=0.02`, elasticity off, and
all GP/nucleation paths off.

The 4096-step window corresponds to 4058.83329 s of model time. The radius
classifications were:

| requested radius (nm) | final-minus-initial radius (nm) | classification |
|---:|---:|---|
| 4 | -3.49420 | dissolving |
| 6 | -5.81798 | dissolving |
| 8 | -1.27143 | dissolving |
| 10 | +0.51290 | growing |
| 12 | +0.85836 | growing |
| 16 | +0.56144 | growing |
| 20 | +0.25666 | growing |
| 24 | +0.44581 | growing |
| 32 | +0.35337 | growing |

The recovered dynamic boundary is therefore:

- largest clearly dissolving radius: **8 nm**;
- smallest clearly growing radius: **10 nm**;
- dynamic critical-radius interval: **8–10 nm**;
- working radius for the continuous fixture: **9 nm**;
- minimum reliably resolved radius in this bracket: **4 nm**.

This boundary is a source-free growth/dissolution result and is not fitted to
the V1 population. The random-PSD fixture uses the registered bounds
`0.8*9=7.2 nm` and `1.4*9=12.6 nm`.

Evidence: `critical_radius_bracket.csv` and
`critical_radius_bracket.json` in this report root; cluster status was
`PASS_T380_DYNAMIC_CRITICAL_RADIUS_BRACKET_V2`.

`t380_dynamic_critical_radius_status=PASS`
