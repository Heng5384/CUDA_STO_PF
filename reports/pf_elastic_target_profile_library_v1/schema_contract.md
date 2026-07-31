# PF elastic target-profile library V1 contract

## Scientific role

This library contains isolated, resolved-beta profiles prepared for the
post-nucleation 6–48 h study.  Every entry is an offline constrained
equilibrium calculation at 380 °C with the frozen production contract:

- \(dx=1\) nm;
- \(\lambda_{\rm sm}=4\) nm;
- periodic fixed-cell elasticity;
- eigenstrain
  \((\epsilon_{xx}^0,\epsilon_{yy}^0,\epsilon_{zz}^0)
  =(0.046,-0.022,-0.017)\);
- identity/[100] orientation variant;
- no GP, GP Birth, GP release, external source, or new beta nucleation.

An entry is an initialization artifact, not a dynamic ageing result and not
an experimental particle-size distribution.

## Constrained minimization

For each registered equivalent radius, the solver minimizes

\[
F_{\rm chem}+F_{\rm interface}+F_{\rm elastic}
+\lambda_V\left(\sum h(\phi)-V_h^\star\right)
+\lambda_M\left(\sum C_B-C_B^\star\right).
\]

The volume target is the registered sphere-equivalent \(h\)-volume.  The mass
target is frozen from the exact initial canonical ledger:

\[
C_B=(1-h(\phi))x_B^\alpha+v_Bh(\phi).
\]

The mass constraint is applied after every phase-volume projection.  It is a
separate offline minimization path and does not replace or modify the
production Ji-Chen zero mode.

## Required convergence gates

An entry is loadable only when all of the following hold for 20 consecutive
iterations:

- the registered gradient-flow pseudo-time has reached at least 300;
- projected, bound-aware KKT residual \(<10^{-3}\);
- \(d\phi/dt<1\times10^{-5}\) through the direct-rate branch, or
  \(d\phi/dt<5\times10^{-5}\) through the registered
  energy-plateau branch;
- \(dY/dt<10^{-4}\) through the registered energy-plateau branch;
- relative total-energy change \(<10^{-8}\);
- relative \(h\)-volume error \(<2\times10^{-4}\);
- relative canonical mass error \(<10^{-12}\).

Reaching `max_iter` is not a pass.  It emits
`MINIMIZE_TARGET_PROFILE_CONVERGENCE_FINAL_AUDIT status=FAIL` and cannot
materialize a library entry.

### Plateau-rate contract revision

The first contract used \(4\times10^{-5}\).  At R=11.5 nm with
`minimize_dt=0.025`, 12000 and 16000 iterations independently reached a
projection-cycle floor of approximately \(4.0181\times10^{-5}\), while the
projected KKT, energy, volume, and mass gates passed.  The fields between
those endpoints differed by only \(8.29\times10^{-5}\) normalized phi L1,
\(2.87\times10^{-9}\) mean absolute \(x_B\), and
\(6.65\times10^{-8}\) relative h-volume.

The rate contract was therefore revised uniformly to
\(1\times10^{-5}\) on the direct branch and \(5\times10^{-5}\) on the
energy-plateau branch, before generating the selected R=11.5 nm entry.  This
is a numerical stopping-contract correction, not a physical parameter
change.  The seven previously qualified entries already satisfy the revised
contract.  The earlier failed roots remain preserved.  KKT, energy, volume,
composition-rate, and mass gates were not relaxed.

### Timestep-independent stopping depth

The initial implementation allowed the energy-plateau branch to stop as soon
as its per-step relative energy change crossed the threshold.  A direct
\(R=10.5\) nm audit showed that dt=0.05, 0.025, and 0.0125 then stopped at
different pseudo-times (approximately 205, 176, and 151), because a smaller
step naturally produces a smaller per-step energy change.  This is a
stopping-time defect, not physical timestep dependence.

The selected-library runner therefore also requires:

```text
minimize_min_pseudo_time=300
```

before the consecutive convergence count may begin.  The value is an
offline gradient-flow relaxation depth and has no physical-time meaning.
All original convergence, KKT, volume, mass, and field-comparison gates
remain active.  Earlier unequal-pseudo-time comparisons remain preserved as
failed historical evidence and are not promoted.

For timestep refinement, both endpoints must reach the common minimum depth.
The permitted difference between their eventual convergence times is the
larger of one consecutive-step window and 2.5% of that minimum depth.  This
separates harmless projection-cycle stopping latency from the earlier
29-pseudo-time under-relaxation mismatch.  It does not relax any field,
shape, KKT, energy, volume, composition, or mass threshold.

## V1 radius ladder

The registered V1 ladder is:

```text
8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5 nm
```

It brackets the current random-PSD V2 6 h population
(8.17–10.87 nm).  The initial isolated minimization candidate is \(96^3\).
The registered \(96^3\), \(128^3\), \(160^3\), and \(192^3\) finite-size
series passes for shape, integrated elastic energy, and the
far-field-subtracted local composition correction.  The absolute
single-particle matrix offset follows the expected \(L^{-3}\) inventory
scaling and is not portable.  Consequently, an exact same-grid entry may
load its raw absolute composition, whereas a different-sized or
multi-particle consumer must use the portable local correction, reconstruct
its own matrix baseline, and close the target inventory with the conserved
zero mode.

## Entry fields

Each profile directory contains float64 little-endian C-order fields:

- `phi.raw.f64`;
- `xB_alpha.raw.f64`;
- `Y.raw.f64`;
- `C_B_tot.raw.f64`;
- `h_phi.raw.f64`;
- `delta_C_relaxation.raw.f64`;
- `dY_dt_prev.raw.f64`.

The manifest also records:

- target and actual \(h\)-volume equivalent radius;
- periodic centroid;
- shape tensor, principal axes, ellipsoid semi-axes, and major/minor ratio;
- far-field matrix-composition statistics;
- exact mass ledger and constraint trace;
- source, runtime-parameter, binary, and every field hash.

## Interpolation boundary

V1 qualifies the registered discrete ladder.  Interpolation between entries
is not silently authorized.  Until a leave-one-out comparison against direct
minimizations passes, consumers must use an exact registered entry or fail
closed.

## Multi-particle boundary

The library does not itself define a 6 h PSD or spatial realization.  A
multi-particle fixture must separately prove:

- periodic non-overlap;
- deterministic placement and order invariance;
- exact total \(h\)-volume and canonical mass;
- source-profile hash provenance;
- no initial profile jump under an elastic short-window continuation;
- fail-closed particle identity tracking.
