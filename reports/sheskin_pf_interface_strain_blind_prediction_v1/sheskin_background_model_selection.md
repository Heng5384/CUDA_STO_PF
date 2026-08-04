# Sheskin AQ background model selection

Status: `SHESKIN_BACKGROUND_IDENTIFIED`.

Only the seven selected AQ measured-total points were used. No 6 h or 48 h value was loaded by the fitter. Each candidate has exactly one common, temperature-independent parameter.
Absolute goodness-of-fit audit: `POOR_ABSOLUTE_FIT_ALL_CANDIDATES_CHI2_P_LT_0P01`. AICc identifies the best member of the pre-registered candidate set; it does not by itself establish that the winning one-parameter curve explains residuals at the assigned 0.015 W m^-1 K^-1 digitization scale.

The Sheskin AQ measurements do not define a unique PSD. The retained envelope uses the Table 1 TEM/APT density at central and +/-1 sigma values, the reported <=10 nm small-object diameter bound, and the 37:6 small/elongated count only as a single-reconstruction diagnostic. SE/FIB density is not added. Yu AQ particle populations are not used.

## Candidate summary

| model | AICc min/median/max | AQ MAPE min/median/max (%) | LOTO MAPE (%) |
|---|---:|---:|---:|
| H1_constant | 97.152 / 97.215 / 97.307 | 4.697 / 4.699 / 4.701 | 5.329 |
| H2_omega2 | 76.074 / 76.078 / 78.440 | 4.045 / 4.046 / 4.146 | 4.680 |
| H3_omega4 | 80.785 / 99.950 / 102.715 | 4.201 / 4.766 / 4.827 | 5.236 |
| H4_host_scale | 240.796 / 355.747 / 362.313 | 7.805 / 9.392 / 9.469 | 10.425 |

Winning models across AQ structure cases: `H2_omega2`.
Models retained by the pre-registered Delta-AICc <= 2 rule in at least one admissible case: `H2_omega2`.

H1-H3 are empirical frequency forms and are not assigned to a specific defect. H4 scales the frozen Yu phonon-phonon term but is not interpreted as a revised Yu parameter. Every fitted amplitude is nonnegative; no independent experimental upper bound is available, so physicality is limited to sign and optimizer-bound checks.
