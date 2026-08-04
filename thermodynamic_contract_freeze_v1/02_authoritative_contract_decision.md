# Authoritative contract decision

## Decision

```text
BLOCKED_CONTRACT_CONFLICT
```

## Evidence-based determination

1. The only complete and executable thermodynamic contract found is the legacy pseudo-binary regular-solution contract in `thermo_utils.h` and `Unit_Psedobinary.py`.
2. The repository does not contain the requested four-point experimental solvus source, exact-fit report, exact-fit parameter CSV, pointwise residuals, sensitivity summary, downstream impact report, or fit covariance.
3. `reports/equilibrium_audit_v1/thermodynamic_fit_reconstruction.md` explicitly says the original real-unit source files are absent and that the available rows were not refit.
4. The prompt candidate values (`DeltaH`, `DeltaS`, `xB_eq≈0.0046492610`) therefore cannot be promoted to authority.
5. The legacy 380 °C root is reproducible: `T=653.15 K`, `xB_eq=0.004664951821454188`, `xAg_eq=0.004654096254055399`.
6. The local and remote source trees contain different commits, dirty worktrees, and different binaries. No cross-environment contract hash exists.

## Required answers

| question | answer |
|---|---|
| exact-fit input data and raw provenance | not found; blocked |
| fit variable and exact conversion | not established for candidate exact fit; legacy code uses pseudo-binary `xB` and exact `xAg=2*xB/(2+xB)` in host paths |
| fit expression and coefficient units | not established for exact fit; legacy expression is `L(T)=41212.9-18.05T` J/mol with T in K |
| 380 °C temperature | `653.15 K` in existing audit/runtime reports; remote production configs are not contract-hashed |
| exact root | not authoritative; candidate `0.0046492610` is unverified |
| legacy root | `xB=0.004664951821454188`, `xAg=0.004654096254055399` |
| existing 246³ contract | legacy or mixed/unknown by result path; no contract hash in provenance |
| future 400³ contract | must not start under current evidence; exact contract unresolved |
| publication contract | cannot be frozen yet |
| rerun decision | indeterminate until exact evidence is restored and matched sensitivity is run |

## Blocking evidence required before Stage B

- authoritative four-point solvus data and source reference;
- exact-fit report and terminal output;
- exact-fit parameter/residual/sensitivity/downstream CSVs;
- definition of whether fit is in `xAg` or `xB` and exact conversion;
- independent CPU/Python/GPU root reproduction;
- explicit decision on legacy-result reuse versus rerun.

## Audit revision 2: external calibration path found

The requested calibration path is present and materially changes the evidence
inventory. `outputs_exact_4pt_final/exact_4pt_final_terminal_output.txt` reports
`PASS_FINAL_EXACT_4POINT_FIT`, and the fit is independently reproducible from
the four-row raw CSV. The exact candidate is:

```text
Delta_H=41504.29119633958 J/mol
Delta_S=18.469276826409214 J/(mol K)
L(653.15 K)=29441.083037170407 J/mol
xB_eq(653.15 K)=0.004649261005504821
xAg_total_eq(653.15 K)=0.46384782574611727 at.%
```

This does not yet change the decision to `BLOCKED_CONTRACT_CONFLICT` because:

1. the calibration workspace's own conflict audit says the manuscript/runtime
   coefficient contract remains unselected;
2. no immutable pointwise residual CSV or covariance/sensitivity artifact is
   present beside the candidate final outputs;
3. the exact-fit source is an external unversioned workspace and is not bound
   to the PF source commit, binary, fixture manifests or checkpoint identity;
4. existing 246³ and current 400³ runtime paths still carry legacy literals;
5. no matched exact-vs-legacy PF sensitivity or rerun evidence exists.

Accordingly the exact fit is now classified as
`AUTHORITATIVE_EXACT_FIT_OUTPUT candidate`, not as the deployed publication
contract. The next safe action is a signed contract decision plus residual and
downstream-impact artifacts; only then may Stage B modify a clean worktree.
