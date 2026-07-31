# Source audit

## Frozen source decision

The path labelled "Yu paper" in the task points to Sheskin et al. 2018, not Yu et al. 2024. It is therefore frozen only as an excluded source and is not used in any equation, parameter, data, or comparison.

The accepted main article is Yuan Yu et al., "Ostwald Ripening of Ag2Te Precipitates in Thermoelectric PbTe: Effects of Crystallography, Dislocations, and Interatomic Bonding", *Advanced Energy Materials* 14 (2024) 2304442, DOI 10.1002/aenm.202304442. The local `AEM Yu 2024 SM.pdf` is its matching Supplementary Information. The hashes and absolute paths are in `source_manifest.json`.

## State definitions

Both states use nominal composition `(PbTe)0.97(Ag2Te)0.03`. The melt was heated to 1273 K over 12 h, held 6 h, cooled to 973 K, homogenized for 48 h, and iced-water quenched. The ingot was ground and hot pressed at 923 K for 30 min, with 45 MPa applied for 15 min under flowing Ar-7% H2, followed by iced-water quenching. This is AQ.

The 48 h state starts from the AQ hot-pressed pellet and was annealed at 653 K for 48 h in a sealed quartz ampoule under 120 torr Ar-7% H2, followed by iced-water quenching. Main text page 10 gives 653 K; SI Table S2 gives 655 K for the `gamma_prime` calculation. Both values are preserved. The implementation uses 655 K only in the independent check of SI Eq. S14 and uses the directly published `gamma_prime = 2.62` in the scattering rate.

## Thermal-conductivity sources

- Experimental AQ and 48 h lattice thermal conductivities are tabulated in SI Table S1. They were transcribed directly; no figure digitization or smoothing was used.
- Yu's calculated model curves are not tabulated. The two dashed curves in main-text Figure 6b were digitized from a 240 dpi Poppler render. Raw connected-component pixels, calibration, hashes, interpolation, and uncertainty are frozen under `data/qualification/yu2024_transport_v1/`.
- Figures 6c and 6d give the 300 K spectral decomposition. They are used only as a visual check because the curves are not tabulated.

## Source-level discrepancies

1. The task-supplied paper path is the wrong article and is excluded.
2. Main text calls the model "Debye-Callaway", but Eq. (3) is a single relaxation-time integral. No Callaway second term is published.
3. SI Eq. S1 publishes only the sum of Normal and Umklapp rates. It does not provide independently evaluable `tau_N` and `tau_U`.
4. SI Table S2 prints `m^-3` for `Vm` and `Vi`, although Eq. S15 and the values require `m^3/atom`.
5. SI Table S2 labels 107.87 g/mol as the mass *difference* between impurity and host, although 107.87 g/mol is the Ag atomic mass. The implementation uses the literal table value in Eq. S4 and marks the meaning ambiguous.
6. The earlier Stage1B audit transcribed the `(6 pi^2)^(1/3)` factor in SI Eq. S1 as a multiplier. Visual inspection of SI page 14 shows it is a denominator. The implementation uses the denominator.
7. Applying all published 48 h parameters to the published equations does not reproduce Figure 6b. This is a model/source mismatch, not an integration error.

## Gate result

The SI exists and the published equation set is recoverable. Source completeness is sufficient to attempt reproduction, but it is not sufficient for a passing freeze because the 48 h published-parameter calculation fails the published model curve. No parameter was refitted.
