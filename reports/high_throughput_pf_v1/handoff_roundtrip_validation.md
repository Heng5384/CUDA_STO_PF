# PF-to-particle handoff roundtrip validation

A real accepted 32-cube PF checkpoint was converted offline using
`q_alpha=Ctot-h(phi)` and particle inventory `integral h(phi) v_B`. One resolved
object was identified. The mapping never wrote back to PF fields and GP inventory
was zero and counted once.

| Check | Result | Gate |
|---|---:|---:|
| global inventory relative error | `2.82816e-13` | `1e-12` |
| particle beta partition relative error | `3.97557e-13` | `1e-6` |
| matrix inventory reconstruction error | roundoff | `1e-6` |
| particle count | `1 -> 1` | unchanged |
| h-weighted equivalent radius | `8.09842 nm` | recorded exactly |
| duplicate GP inventory | none | none |

Threshold radii at phi 0.4/0.5/0.6 are 8.43849, 7.95541, and 7.60031 nm.
This passes the ledger roundtrip but does not alone establish temporal handoff
eligibility or calibrate a KWN model.

Status: `PASS_PF_PARTICLE_HANDOFF_ROUNDTRIP_LEDGER`.
