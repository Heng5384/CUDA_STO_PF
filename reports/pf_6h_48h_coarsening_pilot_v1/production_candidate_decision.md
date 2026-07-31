# Production-candidate decision

## Status

`BLOCKED_MATRIX_CONCENTRATION_TREND`

The result is dual-reported because the experiment provides composition
anchors only at 6 h and 48 h. Under a gate evaluated only at the explicitly
registered ages (6, 12, 18, 24, 36, 48 h), the numerical/coarsening result is
`PASS_T380_6H_48H_COARSENING_TREND_PILOT`. Under the stricter contract that
every hourly checkpoint must remain inside the APT uncertainty band, the
authoritative conservative decision is the status above.

## Gates

| Gate | Result |
|---|---|
| PF-only paths | PASS; GP/source/nucleation paths disabled |
| T400 equal-time/restart/zero-mode qualification | PASS in frozen branch evidence |
| Effective fixture mass and separation | PASS |
| 6 h→48 h checkpoint chain | PASS; 42 unique hourly checkpoints |
| Start-of-continuation checkpoint identity | PASS; SHA-256 bytewise equal |
| Conserved mass | PASS; max mean error 2.86e-14 |
| Resolved count trend | PASS; 32→8 |
| Size trend | PASS; mean radius 7.1303→11.0950 nm |
| Interface-area trend | PASS; 0.0149231→0.00963593 nm^-1 |
| Registered-age matrix Ag band | PASS |
| All-hour matrix Ag band | BLOCKED; first failure at 8 h, xAg=0.00688980 |

## Performance qualification

The diagnostic AB test measured (128^3, T380, dt_code=0.02):

- diagnostic off: 0.005325 s/step;
- mass diagnostic every 256 steps: 0.005819 s/step;
- mass diagnostic every step: 0.122457 s/step.

The accelerated 12→48 h continuation measured a mean 0.0055332 s/step and
maximum 0.005612 s/step. The cadence change is accepted only because the AB
continuous/restart checkpoints were bytewise equal.

The 8 h excursion remains when the matrix mask is varied from `h<0.005` to
`h<0.2` (`xAg=0.0068942` to `0.0068865`), so it is not an
interface-threshold artifact.

## Recommendation

Do not promote this fixture as a final production coarsening model yet. Keep
the numerical engine and accelerated diagnostic cadence, but resolve the
8 h matrix-composition excursion with a new validation-only fixture or an
explicitly justified observation contract. Do not change thermodynamics,
mobility, interface parameters, or time conversion to force the gate.
