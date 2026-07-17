# Physical-time cost projection

Based only on the actual representative 400^3 benchmark: `dt_phys=0.008033122153` s and
`76.33214787983646` wall s/accepted step.  No 512x1 timing is used.

| Physical hours | GPU wall hours | GPU wall days |
|---:|---:|---:|
| 1 | 9502.177 | 395.924 |
| 10 | 95021.769 | 3959.240 |
| 50 | 475108.846 | 19796.202 |

Excluded: GP source, large field output, particle analysis, adaptive dt, KWN handoff and elastic mechanics.
The estimate is a short-run projection and inherits the measured nonlinear-iteration/event mix.
