# Zero-mode validation

The selected scalar zero-mode correction is host Newton with bisection
fallback, fed by persistent paired FP64 device reductions. The existing
64^3/128^3/195^3 timing matrix and 400^3/512^3 matrix both report final mass
errors at roundoff and no prohibited path. The current pilot's first segment
also reports PF_ZERO_MODE_FINAL_AUDIT status=PASS.
