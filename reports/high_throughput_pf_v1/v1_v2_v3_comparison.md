# V1 / V2 / V3 comparison

| Contract | Scientific quantity | Independent holdout | Status |
|---|---|---|---|
| V1 local peak defect-rate gate | Peak local absolute residual accumulation | Historical | `FAIL_PREREGISTERED_LOCAL_PEAK_DEFECT_RATE_GATE` |
| V2 material transport defect | Included sign-coherence as a hard discriminator | Historical | `FAIL_V2_MATERIAL_TRANSPORT_DEFECT` |
| V3 physically scaled signed defect | Net accumulated residual relative to actual material transfer | New sixth window | `PASS_PHYSICALLY_SCALED_SIGNED_DEFECT_GATE_V3` |

V1 and V2 are preserved exactly as failed diagnostic routes. V3 does not
reinterpret a high sign-coherence ratio as a physical bias when the underlying
interface defect is negligible relative to transported material. It instead
normalizes the signed defect by the accepted composition transfer and compares
the candidate directly with the strict trajectory.

For `G9/dt4`, the old interface sign-coherence diagnostic is
`0.174891271198257`, while the physically normalized interface defect is only
`4.29739643823258e-10`. Since `eta_interface=9.79778165777453e-09`, the old
ratio is not materially interpretable and is not a V3 failure.

No transport, phase, thermodynamic, mobility, or acceptance equation changed
when V3 was adopted.
