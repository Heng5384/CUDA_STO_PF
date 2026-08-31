# Effective GP-KWN feasibility

Status: `EFFECTIVE_CNT_SOFT_CONSTRAINT_FEASIBLE_NOT_IDENTIFIED`

The prescribed-source route is numerically active solely to test two-population mass exchange and handoff. `PRESCRIBED_SOURCE_IS_NOT_A_GP_NUCLEATION_PREDICTION`.

The effective-CNT sweep retained all 64 deterministic Latin-hypercube parameter sets: 1 met all soft constraints, 22 missed at least one soft constraint, and 41 hit an explicit solver/inventory boundary. `xB_g`, gamma_g, site density, attachment, diffusivity scale, and elastic penalty remain effective/exploratory parameters rather than identified GP thermodynamics. A soft-constraint hit is not a calibrated physical GP nucleation prediction. Yu 2024 is a holdout plausibility envelope, not a joint fit.

The only soft-constraint hit is parameter set 47 (gamma_g=0.0018366636892788134, xB_g=0.0985849139411075, D_scale_g=0.0031365193932353453).

## Non-identifiability and failure interpretation

Only 23 completed effective-CNT trajectories have finite terminal observables, and only 1 met all six soft constraints. That sampling evidence cannot identify a unique GP composition, interface energy, site density, attachment prefactor, diffusivity scale, or elastic penalty. `gp_parameter_correlation.csv` provides descriptive finite-sample Pearson correlations only; it excludes the retained solver/inventory failures rather than imputing them. The 41 solver/inventory failures are useful evidence that some exploratory priors overconsume the shared reservoir, not data to hide or cap.

Outputs: `gp_parameter_sweep.csv`, `gp_parameter_correlation.csv`, `gp_trajectories.csv`, and population PSD heatmaps.
