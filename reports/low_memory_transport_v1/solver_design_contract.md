# Low-memory transport V1 design contract

## Selector

`ctot_transport_efficiency_mode` is versioned and defaults to `LEGACY_CURRENT`. `LOW_MEMORY_GLOBALIZATION_V1` is restricted to the active-manifold IMEX-BDF2 integrator with `adaptive_logit_feasible_ctot_v1`. Legacy mode requires a zero feature mask.

Implemented feature bits are:

- `PLATEAU_ESCALATION`
- `EXACT_FRACTION_TO_BOUNDARY`
- `HYBRID_L2_LINF_MERIT`
- `ADAPTIVE_SPECTRAL_SCALAR`

`INTERNAL_HOMOTOPY` is reserved but rejected by parameter validation because Stage 7 has not been implemented. This prevents a silent no-op configuration.

## Mathematical invariants

The residual, cold audit and final accepted root are unchanged. Fraction-to-boundary acts only on the search direction and computes local storage using the authoritative cancellation-resistant form

`q=(C-v_B)+(1-h)v_B`.

It selects `lambda_0=min(1,0.995*lambda_feasible)`. It does not clip or project `Ctot`.

The hybrid merit changes trial acceptance only. The final cold `Linf` gate remains unchanged. The adaptive scalar changes only the matrix-free FFT preconditioner and is bounded to `[0.1,10]` times each baseline scalar.

## Memory contract

No sparse matrix or Krylov basis is introduced. All implemented features use scalar host state plus existing full-grid scratch. The fraction-to-boundary reduction reuses the current real scratch. Persistent full-grid fields added: zero.

## Restart contract

Mode and feature mask are written to checkpoint metadata. Mismatch fails closed unless explicit diagnostic migration is enabled. Diagnostic migration is logged and is not the production default.
