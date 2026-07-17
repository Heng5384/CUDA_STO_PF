# Transactional residual observer validation

## Verdict

`PASS_TRANSACTIONAL_OBSERVER_V2`

The accepted-residual observer is default-off and does not feed any host value
back to a CUDA kernel, nonlinear solve, retry predicate, or accepted state. Its
event path now follows the same macro transaction boundary as the solver:

1. accepted BE substeps remain in a host-only pending ledger;
2. `CTOT_BDF2_EVENT_MACRO_READY` commits the pending rows and defect;
3. a deeper subcycle retry clears the pending ledger before macro rollback;
4. only macro-committed substeps contribute to `E_R`, `D_i`, or observed time.

## Event-bearing on/off replay

Case: `G10 + dt1`, 360 macro steps, gate `1e-10`. This interval crosses the
first depth-2 and depth-4 event subcycles.

| Check | Result |
|---|---|
| Workstation build | PASS |
| Binary SHA-256 | `3986440d157f29c5ab53387345f67bbedf7c264b1a5136b04461cec32a954b0b` |
| Observer schema | `CTOT_TRANSPORT_GATE_TRAJECTORY_V2_TRANSACTIONAL` |
| Valid committed substep rows | 364 |
| Expected time code | 1.125 |
| Observed time code | 1.125000000000007 |
| Time closure error | 7.105427357601002e-15 |
| `Ctot` diagnostics on/off | byte-identical |
| `phi` diagnostics on/off | byte-identical |
| `xB_alpha` diagnostics on/off | byte-identical |
| `Ctot_nm1` diagnostics on/off | byte-identical |
| `phi_nm1` diagnostics on/off | byte-identical |

The row count exceeds the macro-step count because valid BE substeps are
retained individually. The physical-time sum remains exact because partial
subcycles that were later rolled back are absent.
