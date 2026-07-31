# Numerical validation

This file is generated from the frozen JSON inputs. No fit is performed.

## Summary

- Full rates finite and positive: `True`
- Conductivities finite and positive: `True`
- Maximum 256/512-point refinement difference: `6.613412e-14`
- Maximum Gauss/adaptive-quad difference: `6.806179e-14`
- Gamma-prime independent relative check: `8.436288e-04`
- In-process deterministic repeat: `True`
- Asymptotic rate powers pass: `True`
- Precipitate crossover continuity pass: `True`

## Source limitations

- SI Eq. (S1) publishes only the combined Normal-plus-Umklapp rate.
- The main text publishes a single relaxation-time integral; no Callaway second term is present.
- Table S2 prints `m^-3` for atomic volumes `Vm` and `Vi`. The implementation uses `m^3/atom`, as required by Eq. (S15) and dimensional consistency.
- A zero published defect density produces zero rate and infinite individual relaxation time; positivity is tested on the full active total rate.

## Isolation

- PF data used: `false`
- Sheskin data used: `false`
- Parameter refit: `false`

## Cross-process output determinism

The full implementation was run in two separate Python processes into
`tmp/yu2024_repeat_a` and `tmp/yu2024_repeat_b`. A recursive byte comparison
found no differences, including the CSV, JSON, Markdown, and PNG outputs.
The frozen hashes are recorded in `determinism_audit.json`.
