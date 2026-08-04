# Legacy versus exact conflict audit

| item | legacy/runtime evidence | exact candidate evidence | conflict / disposition |
|---|---|---|---|
| interaction parameter | `L(T)=41212.9-18.05T` J/mol | candidate `DeltaH≈41504.291`, `DeltaS≈18.469277` | exact mapping to `L(T)` not supplied; unresolved |
| 380 °C L | `29423.5425 J/mol` at `653.15 K` | not independently available | unresolved |
| 380 °C root | `xB=0.004664951821454188` | prompt candidate `xB≈0.0046492610` | difference is small but source/objective unavailable; cannot dismiss |
| composition variable | runtime pseudo-binary `xB` | fit variable not documented | unresolved |
| conversion | exact algebraic host conversion `xAg=2xB/(2+xB)` and inverse `xB=2xAg/(2-xAg)` appears in code | exact-fit conversion source absent | implementation path exists, contract authority absent |
| experimental solvus points | not present in exact-fit files; current fit summary is `NOT_FIT` | prompt says four points exist | source conflict / missing evidence |
| temperature basis | existing runtime/audit uses K internally and `380 °C→653.15 K` | candidate says `653.15 K`, but no fit report | consistent numerically, not provenance-complete |
| covariance / uncertainty | `thermodynamic_parameter_uncertainty.md` says unavailable | not supplied | exact refit cannot be validated |
| publication status | legacy only | exact not proven | `BLOCKED_CONTRACT_CONFLICT` |

No production source, result metadata, or old result file was modified during this audit.

## Audit revision 2: calibrated exact candidate is now evidenced

The external fit is no longer prompt-only. Its independently reproduced values
at `T=653.15 K` are:

| Contract | L (J/mol) | xB,eq | Ag total (at.%) | status |
|---|---:|---:|---:|---|
| Exact four-point calibration candidate | 29441.0830372 | 0.0046492610055 | 0.4638478257 | candidate exact; not deployed |
| Existing production runtime | 29423.5425 | 0.0046649518215 | 0.4654096254 | legacy runtime |

The difference is therefore a real contract difference, not a temperature or
conversion typo. The exact candidate has reproducible raw points and code, but
its residual artifact, full source citation/digitization uncertainty and matched
PF downstream impact are not frozen. Existing results remain legacy or
mixed/unknown and must not be relabeled.
