# Frozen reject replay

The catalog contains 31 historical reject records, but the repository evidence
does not contain the exact predecessor `Ctot`, `phi`, BDF2 history, active set,
and mechanics arrays for all 31 records. Exact state replay is therefore not
possible and no convergence count is fabricated.

The available replacement evidence is trajectory recurrence from the common
frozen initial state. It reproduces the same globalization failure class:
`transport_line_search_stagnation`, repeated internal fallback at coarse dt,
and bitwise rollback when a trial is rejected. This evidence is sufficient to
reject V1--V5 as production candidates, but it is not labelled an exact 31-state
replay. Status: `INCOMPLETE_EXACT_STATE_PAYLOADS_NOT_RECORDED`.
