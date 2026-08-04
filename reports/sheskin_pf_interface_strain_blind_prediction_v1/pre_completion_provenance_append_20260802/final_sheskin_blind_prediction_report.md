# Sheskin AQ-background + PF interface/strain blind-prediction report

## Outcome

`FAIL_RESOLVED_ONLY_AFTER_INTERFACE_AND_STRAIN_TEST`

The resolved PF microstructure has strong geometric and spectral evolution, and the qualified interface model correctly predicts the **sign** of recovery for every retained AQ-background member. It does not predict the required magnitude. At 573.15 K, the nonprobabilistic background-envelope interface result spans `0.836369`--`0.911414` W m^-1 K^-1 at 48 h, with only `2.115`--`7.268%` recovery. The experiment is 1.03 W m^-1 K^-1 and the preregistered recovery gate is 10--30%. No retained member reaches that gate.

The parameter-free scalar Born strain channel is far weaker than the host/background rates and retains the M0 negative trend. Adding it to MI changes the endpoint negligibly. Therefore neither resolved interface scattering nor resolved coherent hydrostatic-strain scattering closes the experimental gap under the frozen contracts.

## Evidence chain

- Density-only baseline reproduced exactly: 6 h `1.2980916621210676`, 48 h `1.2948991542326225` W m^-1 K^-1.
- AQ selected H2 background MAPE envelope: `4.0451`--`4.1457%`; however all one-parameter AQ candidates have poor absolute chi-square fit. The background is an empirical time-invariant envelope, not a named defect.
- Mechanics replay: checkpoint-warm online/offline fields are byte-identical with zero energy/strain/stress error. The zero-initialized sensitivity path failed the frozen strain/stress tolerances and is forbidden for authority use.
- Historical replay: 18 accepted-field states, maximum solver residual `9.11258e-07`, no PF/time advance and no checkpoint write.
- Interface area ratio `Sv_48/Sv_6 = 0.395306` (`STRONG_LEVERAGE`).
- Total hydrostatic variance ratio `1.28336`, but mid/high-q ratios are `0.306247` and `0.458338`: `SPECTRAL_REDISTRIBUTION_LEVERAGE`.
- Interface alpha range `2.73484`--`10`; `16/18` background members are strictly inside the frozen `[0.1,10]` interval.
- The freeze manifest was generated from AQ + 6 h only. The 48 h evaluation verified all frozen hashes and performed no refit.

## 573.15 K frozen results

| Model | equal-weight diagnostic mean 6 h | equal-weight diagnostic mean 48 h | change | background-member recovery envelope |
|---|---:|---:|---:|---:|
| M0 | 0.873345 | 0.871682 | -0.190% | -0.205 to -0.187% |
| MI | 0.820981 | 0.849143 | 3.430% | 2.115 to 7.268% |
| MS | 0.873161 | 0.871528 | -0.187% | -0.200 to -0.184% |
| MIS | 0.820883 | 0.849037 | 3.430% | 2.115 to 7.267% |

The equal-weight means average A/B/C and the 18 retained background members only as a deterministic diagnostic. The background members are an uncertainty envelope, not a probability distribution; the min/median/max member results in `full_temperature_comparison.csv` are the authoritative propagation.

## Interpretation boundary

Figure 5c is measured **total** thermal conductivity, while the calculation uses an effective Debye lattice-style relaxation-time reconstruction. The identities are kept explicit. This study does not claim absolute experimental lattice-thermal-conductivity reproduction, first-principles prediction, or a PF dislocation-density prediction.

The correct next action is not to retune the PF PSD, eigenstrain, thermodynamics, `A_N`, or Yu dislocation scale. Within the registered resolved-particle-only scope, the mechanism test is complete and fails the magnitude gates. Any further route requires independently constrained physics outside this model, such as polarization-resolved PbTe/Ag2Te interface transmission or separately evidenced unresolved defects; it must be a new contract rather than a fit to this 48 h endpoint.
