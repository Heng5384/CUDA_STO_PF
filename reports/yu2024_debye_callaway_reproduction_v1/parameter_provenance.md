# Parameter provenance

The authoritative computation inputs are the two JSON files under `data/qualification/yu2024_transport_v1/`. The CSV table in this report adds human-readable provenance and classification. No runtime value is obtained from PF output, Sheskin data, an optimizer, or a hidden constant.

## Classification summary

- `A_N = 1.5` is the only value explicitly labelled fitted by Yu.
- `epsilon = 65`, `C = 0.96`, `gamma`, `v`, `vL`, `vT`, `Theta_D`, and the cited elastic quantities are literature inputs.
- State-specific lattice constant, grain size, interstitial Ag fraction, number densities, and dislocation density are measured or registered by Yu.
- Atomic volume, `Gamma`, density contrast, `alpha`, `beta`, and the independent `gamma_prime` check are derived only from published values.
- No value is classified `DIGITIZED_FROM_FIGURE` in the scientific parameter table. Figure digitization is used only to reconstruct the published model curve for comparison.

## Fail-closed ambiguities

### Point-defect mass input

SI Eq. S4 asks for a host-impurity mass difference. Table S2 labels `Delta M_i = 107.87 g/mol` as that difference, but 107.87 g/mol is the Ag atomic mass. The implementation does not silently replace it with `207.2 - 107.87`; it uses the literal table number and records the parameter as `MISSING_OR_AMBIGUOUS`. This literal choice reproduces the AQ curve closely but does not resolve the semantic inconsistency.

### Atomic-volume units

Table S2 prints `m^-3` for `Vm` and `Vi`. Their magnitudes, definitions, Eq. S15, and Eq. S14 dimensional balance require `m^3/atom`. This is treated as a published unit typo and corrected explicitly.

### Normal and Umklapp separation

The source never gives separate coefficients or rates. Only their sum in Eq. S1 is computable. Separate Normal and Umklapp contributions therefore remain `MISSING_OR_AMBIGUOUS` and are not fabricated.

### Full Callaway second term

No second term is present in Yu Eq. (3) or the SI. The reproduced object is Yu's published single-relaxation-time effective model, not a textbook two-term Callaway model.

### Annealing temperature

The main experimental section gives 653 K; SI Table S2 gives 655 K. The directly published `gamma_prime = 2.62` is used. The 655 K value is used only to verify Eq. S14, so the discrepancy does not introduce a fitted freedom.

## Refit audit

No parameter was changed to reduce residuals. In particular, the large 48 h residual was retained. The diagnostic comparison of published Eq. S12 and the required Ag-decorated Eq. S13 is discussed in the reproduction report, but neither alternative was fitted.
