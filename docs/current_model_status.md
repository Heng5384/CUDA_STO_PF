# Current Model Status

## Composition update modes

| Mode | Current status | Interpretation |
|---|---|---|
| L: `lagged_rhs` | Failed diagnostic reference | Lagged-history positive feedback is the identified failure mechanism. Keep for regression and comparison. |
| X: `x_transport_projection_split` | Bounded diagnostic, not accepted | Global projection dominates, so local phase-storage semantics fail even when the trajectory remains bounded. |
| Q: `q_transport_projection_split` | Formula-level preferred candidate, runtime not accepted | Current flux discretization does not preserve the capacity bound. Keep as the leading candidate and regression target. |

No composition mode is currently documented as production accepted.

## RSMD / S3 source component

The exact-exponential S3 source transaction, fixed physical source cadence, headroom-weighted redistribution, GP-ledger deduction, source-only validation, history synchronization, and regional composition diagnostics are retained as a validated component.

Current gate:

```text
S3_source_component_frozen=true
S3_reintegration_allowed=false
```

S3 remains frozen until the PF-only baseline and composition architecture acceptance gates pass. A validated source-only transaction is not equivalent to accepted production reintegration.

## Minimal verification

- Build: `make main_cuda` with the CUDA environment described in `README.md`.
- Unit-conversion smoke: `bash scripts/smoke_test_unit_conversion.sh`.
- Formula/operator smoke: `python3 scripts/test_pf_storage_exact_formula.py` and `python3 scripts/test_pf_x_q_operators.py`.
- Report tools should support `--help` or import without requiring a large simulation.

Accepted seed/profile libraries and T380/T400 parameter inputs remain under `data/`, `shape_library/`, and `params/`; cleanup must not treat them as generated scratch data.
