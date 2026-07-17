# Minimal PF/particle overlap

## Contract

The default-off particle oracle uses parameters derived from unchanged T400
runtime inputs. None were fitted to these trajectories. It conserves its own
matrix-plus-particle ledger exactly and has no nucleation, merging, elasticity,
GP release, or direct PF write authority.

The first two-particle PF attempt at fixed G9/dt4 exhausted event subcycling at
step 27. Every failed trial restored the accepted state and history bitwise.
Because overlap fidelity does not require the throughput candidate's dt, the
unchanged ensemble was rerun with the qualified strict dt16 reference. No
residual, mass, KKT, or physics parameter was relaxed.

| Case | PF steps | Physical time | Radius error | Phase error | PF/oracle count | Extinction order | Result |
|---|---:|---:|---:|---:|---:|---|---|
| two particle | 200 | 1.60662 s | 0.0576% | 0.146% | 2/2 | not applicable | PASS |
| three particle/extinction | 1200 | 9.63975 s | 8.404% | 0.446% | 3/3 | no registered extinction in either trajectory | FAIL |

The PF handoff ledgers close to `2.99e-13` and `2.87e-13` relative for the two
final states. The oracle mass error is zero to reported precision. Thus the
failure is neither a handoff mass error nor a PF source-free mass error. It is
the preregistered trend overlap: the smallest-particle radius differs by more
than 5%, and the requested extinction-order comparison was not reached.

No parameter was retuned after seeing the result. The oracle is therefore not
authorized as a quantitative long-time model.

Status: `PF_ORACLE_OVERLAP_FAILED`.

Recommended classification: `REFINE_PF_TO_PARTICLE_STATE_MAPPING_BEFORE_GP_CAMPAIGN`.
