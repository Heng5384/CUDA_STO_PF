# Equal-time accuracy

The formal V2 residual-gate qualification ended in `FAIL_V2_MATERIAL_TRANSPORT_DEFECT`.
Therefore no relaxed gate is available for quantitative production acceptance and
G10 results remain solver-development diagnostics only.

The 20-step dt/16 V1--V4 checkpoints are bitwise identical to V0 for both Ctot
and phi. Long-window V5 is not equivalent because it invokes fallback: at dt/16
the endpoint differences relative to V0 are Ctot Linf
`4.64017885886125470e-05`
and phi Linf
`2.96478234095287618e-05`.
These differences are reported, not accepted as qualified error.
