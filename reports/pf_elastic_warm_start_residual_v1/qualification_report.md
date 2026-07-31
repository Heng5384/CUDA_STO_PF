# PF elastic warm-start and residual-control qualification

Date: 2026-07-31

Final status:

```text
PASS_ELASTIC_WARM_START_RESIDUAL_V1
```

## Scope

This is a numerical-solver optimization for the current post-nucleation,
resolved-beta PF path.  GP, GP Birth, GP release, external sources, and new
beta nucleation remained disabled.  No thermodynamic, kinetic, elastic, grid,
time-step, fixture, or other physical parameter was retuned.

The comparison used the same hash-pinned 246^3 replicate-B fixture at T380,
`dt_code=0.02`, for 256 macrosteps.  The fixed-iteration reference and the
accelerated candidate used the same executable:

```text
main_cuda_sha256=93e4e6159184de783df406e9c6329459ab0d789026f17adc9bb63757e1f3f385
main_cuda.cu_sha256=e55277be8bb46355f8b0ef4b58e1bebf4f8a5f47331c91e8a3823060cd3f7340
```

## Solver change

The candidate:

- keeps the converged elastic displacement field in GPU k-space and uses it
  as the next macrostep's initial state;
- computes a Hermitian-weighted relative fixed-point displacement residual on
  the GPU;
- transfers only the two scalar residual sums to the host;
- requires at least two fixed-point updates;
- stops when the relative residual is at most `1e-6`;
- fails closed at a hard cap of 32 updates;
- stores the complete warm state and its time level in a checksummed V4
  checkpoint with a solver-parameter fingerprint.

The legacy fixed-iteration path remains the default when the two new switches
are off and continues to write V3 checkpoints.

## Preserved first failure

The first candidate used a hard cap of 20.  It stopped at the first cold-start
step:

```text
iterations=20
relative_residual=1.83728973425113764e-06
tolerance=1.0e-06
```

The residual requirement was not weakened.  The final candidate raised only
the numerical safety cap to 32.  Its cold-start step converged in 22 updates
with residual `7.47906089762138573e-07`.  The failed V1 output remains at:

```text
/home/zhiheng/tmp/pf_elastic_warm_start_residual_v1_20260731
```

## Final performance

| metric | fixed iteration | warm-start/residual | result |
|---|---:|---:|---:|
| wall seconds | 326.958 | 165.050 | PASS |
| seconds/macrostep | 1.277179688 | 0.644726563 | PASS |
| speedup | 1.0x | 1.980963344x | PASS |
| mean elastic updates | 19 reference updates | 4.59765625 | PASS |
| maximum candidate updates | — | 22, cold start | PASS |
| mean warm-step updates | — | 4.529411765 | PASS |
| nonconverged steps | — | 0/256 | PASS |
| peak VRAM | 4680 MiB | 4854 MiB | +174 MiB |

The warm-state buffers increase peak VRAM by 174 MiB.  GPU utilization
remained high: 98.77% for the reference and 97.38% for the candidate.

## Numerical and scientific agreement

All 96 registered particle identities remained resolved.  Relative or absolute
candidate/reference differences at the common endpoint were:

```text
phi_normalized_L1=6.381899413146483e-07
Y_mean_absolute=3.545371682836073e-07
xB_mean_absolute=2.682764013711608e-09
dYdt_mean_absolute=9.143895141137777e-08

beta_volume_fraction_relative=1.1573401665436801e-07
mean_radius_relative=4.68456401790241e-08
Sv_relative=8.534742927856135e-08
M6_relative=1.6799146011998583e-07
matrix_xAg_absolute=2.6716003567392455e-09
mean_elastic_energy_relative=2.920337715577279e-07
```

The candidate mass-relative error was `1.764705485777036e-13`, inside the
registered `1e-10` tier.

## V4 restart

Continuous 256 steps and 128 + checkpoint/restart + 128 steps produced
byte-identical V4 checkpoints:

```text
sha256=0cab6cd5f3f8965f99b4b0864afacd84b63f30a863346f343d6faf8a0574151e
checkpoint_bytes=656478472
restart_status=PASS_BYTEWISE
```

The V4 provenance includes the elastic solver mode, solver fingerprint,
displacement k-space state, source-field time level, last iteration count, and
last relative residual.  V2/V3 reading remains supported, but a V2/V3
checkpoint is rejected when the accelerated solver explicitly requires the V4
elastic state.

## Evidence

The final non-overwriting workstation root is:

```text
/home/zhiheng/tmp/pf_elastic_warm_start_residual_v2_20260731
```

The local frozen evidence in this directory includes the final audit JSON,
terminal markers, residual trace, mass diagnostics, baseline audit, parameter
files, and input hashes.  The large checkpoints were intentionally not copied
into Git.

