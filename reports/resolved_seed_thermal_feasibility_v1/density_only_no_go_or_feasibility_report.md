# Fixed-inventory resolved-particle density-only feasibility

The scan fixes the Method-1 effective h-volume at `356237.610167353356 nm³` and uses a fixed matrix `xAg=0.0062`. For each lognormal population, the effective count is obtained from the fixed h-volume divided by the population's full third moment; it is not inferred from an arbitrary count.

| best diagnostic case | value |
|---|---|
| CV | 0.3 |
| R6_nm | 15 |
| R48_nm | 17 |
| N6_effective | 19.4579860797 |
| N48_effective | 13.3667215589 |
| kappa_6h_W_mK | 1.29615716044 |
| kappa_48h_W_mK | 1.29611090645 |
| delta_kappa_W_mK | -4.62539905273e-05 |
| relative_change_percent | -0.00356854800784 |
| objective_J | 0.342260270765 |

Positive-trend combinations exist: `True`. Both endpoints within the pre-registered ±5% diagnostic gate: `False`.

**No-go conclusion:** the current 6 h state does not fail merely because it has 96 particles with a mean radius near 9.54 nm. Within the PF-resolved radius range scanned here, changing only resolved particle count and size under fixed inventory does not reproduce both experimental conductivity endpoints using the frozen density-contrast scattering model.
