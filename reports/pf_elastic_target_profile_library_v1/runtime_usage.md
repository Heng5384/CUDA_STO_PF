# Elastic target-profile V1 runtime usage

## What is loadable

Each registered radius directory is a complete single-particle raw-field
fixture.  The authoritative runtime inputs are:

- `phi.raw.f64`;
- `xB_alpha.raw.f64`;
- `init_meta.json`;
- `profile_manifest.json`.

The manifest pins every field hash and records the exact source, binary,
eigenstrain, orientation, volume, composition, and mass ledger.  VTK is not
required at runtime.

The raw `xB_alpha` field is a direct-load fixture only for the exact isolated
grid recorded by the manifest.  It is not a portable absolute matrix
baseline: fixing one particle's inventory in different box volumes produces
a uniform \(O(L^{-3})\) concentration offset.  The portable composition
quantity is:

\[
\delta C_{\rm relax}=(1-h(\phi))
\left(x_B^\alpha-x_{B,\rm far}^\alpha\right),
\]

stored as `delta_C_relaxation.raw.f64`.  An arbitrary-domain or
multi-particle materializer must combine this correction with that target
system's registered matrix baseline and then close its exact total inventory
with the conserved zero mode.

## Single-profile dynamic invocation

For an exact registered entry on the same grid:

```bash
main_cuda 96 96 96 0.02 N N 1 1 \
  --pf-param-file PF_PARAMS \
  --mode dynamics \
  --init-mode raw_fields \
  --init-phi-raw PROFILE/phi.raw.f64 \
  --init-xB-raw PROFILE/xB_alpha.raw.f64 \
  --init-meta PROFILE/init_meta.json \
  --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1 \
  --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1 \
  --pf-zero-mode-tol-rel 1e-12 \
  --pf-zero-mode-max-iter 24 \
  --pf-checkpoint-every N \
  --pf-checkpoint-path OUTPUT/final.chk
```

The production parameter file must keep elasticity enabled and every GP,
external-source, and new-beta-nucleation path disabled.

## Restart

Fresh initialization may use the hash-pinned raw fields once.  Continuation
must use the checksummed zero-mode checkpoint:

```bash
main_cuda 96 96 96 0.02 FINAL_STEP FINAL_STEP 1 1 \
  --pf-param-file PF_PARAMS \
  --mode dynamics \
  --pf-restart-from OUTPUT/half.chk \
  --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1 \
  --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1 \
  --pf-zero-mode-tol-rel 1e-12 \
  --pf-zero-mode-max-iter 24 \
  --pf-checkpoint-every FINAL_STEP \
  --pf-checkpoint-path OUTPUT/final.chk
```

Using raw fields as an ad hoc continuation format is prohibited.

## Current boundary

V1 qualifies exact radii 8.0–11.5 nm in 0.5 nm increments for one fixed
orientation variant.  It does not yet authorize:

- interpolation between radii;
- rotation into other crystallographic variants;
- overlay of several isolated fields into one multi-particle fixture.

It also does not authorize copying an isolated profile's absolute
`xB_alpha.raw.f64` into a differently sized box.  Shape and
`delta_C_relaxation` passed the portable finite-box audit; the absolute
single-particle matrix offset did not.

Those operations require a separate deterministic materializer that proves
periodic non-overlap, order independence, exact total inventory, source
profile provenance, and a multi-particle dynamic handoff.  Nearest-radius
substitution is not silently allowed.
