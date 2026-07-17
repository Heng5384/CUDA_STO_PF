# Plateau detector validation

The nine-value window implements the frozen `R_k/R_(k-8) >= 0.98` rule and four
consecutive sub-one-percent improvements. Host tests pass. Runtime tests show
that the detector exits V1 at step 1325 instead of burning the full nonlinear
budget, while never accepting a residual above the unchanged gate. With later
strategies enabled it escalates rather than accepting. Status:
`IMPLEMENTED_DEFAULT_OFF_FAIL_CLOSED_VALIDATED`.
