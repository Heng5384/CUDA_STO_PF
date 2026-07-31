# T380 6 h→48 h coarsening-trend pilot

## Scope and provenance

This is the PF-only conserved-composition pilot on a deterministic effective
resolved-beta fixture. GP population, GP Birth, GP release, GP growth,
external source, scheduled nucleation, elastic relaxation, and direct beta
inventory injection were disabled.

The original 6 h→12 h chain was started as Slurm job 72670 and stopped after
the 12 h checkpoint. The continuation (job 72676) resumed from the exact
12 h checkpoint and ran to 48 h. The two copies of
`step_21798.chk` have SHA-256
`4df5e73df52032515492e58fe327c501fbd9fd9ab2ddb57fe18b4c5f2e8c3e02`.

The acceleration changes only the diagnostic cadence after the validated
AB test: `dynamics_mass_diag_interval=256` instead of `1`. The AB report
shows bytewise-equal checkpoints and the same mean mass error for interval 1
and interval 256.

## Observed trend

The 6 h fixture contains 32 effective resolved seeds. The periodic
`phi>0.5` observer reports 24 particles at 7 h, 16 at 8 h, and 8 at 9 h;
the count remains 8 through 48 h. The initial mean equivalent radius is
7.13034 nm and the final mean is 11.09497 nm (ratio 1.5560). The mean
third-radius moment ratio is 3.7089. Interface-area density decreases from
0.0149231 nm^-1 at the fixture to 0.00963593 nm^-1 at 48 h (ratio 0.6457).

The resolved-density ratio is therefore 8/32 = 0.25. This is a
coarse-grained PF population ratio and is not compared as a one-to-one APT
object-density ratio.

## Conservation and matrix composition

The maximum absolute mean conserved-composition error over all checkpoints is
`2.8602120671905595e-14`; all checkpoint audits report
`PF_ZERO_MODE_FINAL_AUDIT status=PASS`. No clipping, physical global mass
projection, GP path, or external source was observed. The maximum observed
zero-mode correction was `6.190303219302345e-12`.

At the registered reporting ages 6, 12, 18, 24, 36, and 48 h, matrix
`x_Ag` is approximately 0.0062 at 6 h and 0.0058774 at 12–48 h, within the
registered 0.0062 +/- 0.0004 band. The hourly diagnostic series also shows a
transient 8 h value of 0.0068898, above the upper bound 0.0066, before
returning to the band at 9 h. This excursion is the first failure of the
all-hour matrix-composition gate and is not hidden by selecting only the six
registered ages.

As a read-only observation audit, the 8 h checkpoint gives `x_Ag=0.0068942`
with an `h<0.005` matrix mask and `x_Ag=0.0068865` with an `h<0.2` mask.
The excursion is therefore insensitive to the exact interface-exclusion
threshold; it is not a threshold-only artifact.

## Decision

The numerical conservation/restart and the structural coarsening trends pass,
but the all-hour matrix-composition gate is not closed. The pilot is therefore
classified as `BLOCKED_MATRIX_CONCENTRATION_TREND`, pending an explicitly
audited treatment of the early transient (for example, a better validated
effective fixture or a separately justified observation window). No physical
parameter was retuned in response.

If the gate is evaluated only at the six pre-registered experimental ages,
the corresponding conditional status is
`PASS_T380_6H_48H_COARSENING_TREND_PILOT`; this is reported separately so the
unmeasured 8 h transient is not silently discarded.

Evidence tables are in `population_time_series.csv`,
`particle_trajectories.csv`, `matrix_composition_time_series.csv`, and
`mass_and_zero_mode_time_series.csv`.
