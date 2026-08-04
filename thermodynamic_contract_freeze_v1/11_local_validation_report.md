# Local validation report

Status: `BLOCKED_CONTRACT_CONFLICT`

## Executed read-only checks

- Local repository and current branch identified.
- Legacy source equations found in `thermo_utils.h` and `Unit_Psedobinary.py`.
- Existing legacy 380 °C root reproduced in the registered audit CSV.
- Exact-fit report, four-point source data, parameter CSV and residual CSV were not found.
- Local binary `main_cuda` is absent, so no new CPU/GPU runtime identity test was run.

## Validation status

| test | status | reason |
|---|---|---|
| exact contract schema | NOT_RUN | exact contract has no authoritative values |
| four-point fit reproduction | BLOCKED | four source points and fit output absent |
| exact equilibrium root | BLOCKED | candidate root is prompt-only |
| exact conversion round-trip | NOT_RUN | conversion formula exists, exact-fit authority absent |
| free-energy derivative | NOT_RUN | no exact contract |
| driving-force sign | NOT_RUN | no exact contract |
| legacy-vs-exact sensitivity | NOT_RUN | exact coefficients unavailable |
| runtime contract identity | NOT_RUN | no local binary and no contract hash |
| checkpoint compatibility | NOT_RUN | no contract-aware checkpoint implementation |
| cross-platform reproducibility | BLOCKED | environments have different dirty commits/binaries |
| registered legacy static equation audit | PASS | existing `reports/equilibrium_audit_v1` evidence |

No production or result file was changed.

## Audit revision 2: calibration-path validation

The external calibration source was independently checked without rewriting its
outputs. The four raw rows, exact conversion and linearization reproduce:

```text
Delta_H=41504.29119633958 J/mol
Delta_S=18.469276826409214 J/(mol K)
L(653.15 K)=29441.083037170407 J/mol
xB_eq(653.15 K)=0.004649261005504821
xAg_total_eq(653.15 K)=0.46384782574611727 at.%
```

| test | status | reason |
|---|---|---|
| calibration raw-source hash check | PASS | hash-pinned `Solubility_extract.csv` found |
| four-point fit reproduction | PASS | independent calculation matches candidate exact output |
| exact conversion round-trip | PASS | algebraic forward/inverse identity |
| exact equilibrium root | PASS | independent low-concentration root at 653.15 K |
| pointwise residual artifact | BLOCKED | no immutable residual CSV in calibration workspace |
| legacy-vs-exact PF sensitivity | NOT_RUN | no matched PF states evaluated |
| local runtime contract identity | BLOCKED | production source remains legacy and local binary absent |
| workstation/cluster identity | BLOCKED | dirty, different commits/binaries and no deployed contract hash |

Therefore the exact candidate is reproducible but not yet an accepted
cross-environment production/publication contract.
