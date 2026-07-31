# Elastic target-profile V1 qualification plan

## Workstation qualification role

The workstation first generated the eight-radius implementation proof with:

```text
grid=96^3
minimize_dt=0.1
elastic=true
orientation=variant_100_identity
```

It then ran the \(96^3\), \(128^3\), \(160^3\), and \(192^3\) finite-box
series and the dynamic handoff/restart cases.  It is not required to
duplicate the final cluster eight-radius ladder solely for a device
comparison.  Once the selected cluster library is frozen, the workstation
loads its small, middle, and large entries for the final independent dynamic
qualification.

## Cluster timestep refinement

Generate the same ladder independently on `gpu_uvip` with:

```text
grid=96^3
minimize_dt=0.05 and 0.025
minimize_min_pseudo_time=300
elastic=true
orientation=variant_100_identity
```

Compare radius by radius:

- equivalent \(h\)-volume radius;
- ellipsoid semi-axes and major/minor ratio;
- periodic centroid;
- far-field \(x_B^\alpha\);
- normalized \(\phi\), \(x_B\), and canonical-mass differences after
  confirming common runtime-source hashes.

The two cluster binaries and every run root remain separately identified.
The workstation and cluster are assigned complementary work to reduce
elapsed qualification time; cross-device byte equality is not a production
gate.

The timestep comparison is valid only when both entries record at least the
common minimum pseudo-time.  Their converged pseudo-times may differ by no
more than the larger of one registered consecutive-step window and 2.5% of
the common minimum relaxation depth.  The relative allowance accounts for
the phase of the projection-cycle plateau; it remains much smaller than the
29-pseudo-time mismatch in the preserved unequal-depth failure.  Field,
shape, composition, mass, and KKT gates are unchanged and remain the primary
endpoint tests.

## Finite-box qualification

Use the largest and most anisotropic registered entry, \(R=11.5\) nm, on
the \(96^3\), \(128^3\), \(160^3\), and \(192^3\) isolated periodic boxes.
Separately report:

- equivalent radius, semi-axes, axis ratio, and centered local \(\phi\);
- box-integrated elastic energy;
- absolute far-field \(x_B^\alpha\);
- far-field-subtracted local composition correction;
- whether
  \((x_B^{\rm far}-x_B^{\rm registered})L^3\) is box invariant.

The absolute raw \(x_B\) field must not be declared portable merely because
the far-field-subtracted correction passes.  Arbitrary-domain consumers must
reconstruct the target system's matrix baseline and close its exact total
inventory through the conserved zero mode.

## Short continuation acceptance

At minimum one small, one middle, and one large profile must be loaded into
the production elastic dynamic path and run with production `dt` plus
`dt/2`.  Require:

- a separate one-production-step handoff probe, evaluated with
  volume-normalized semi-axes, axis ratio, and periodic centroid, must retain
  the minimized elastic shape within 2%, 2%, and 0.1 nm respectively;
- particle growth or shrinkage over the longer continuation is reported as
  physical evolution, not mislabeled as an initialization jump;
- at the common physical endpoint, `dt` versus `dt/2` must have normalized
  \(\phi\) L1 difference at most 2%, semi-axis difference at most 2%, and
  mean absolute \(x_B^\alpha\) difference at most \(5\times10^{-5}\);
  \(2\times10^{-5}\) remains the preferred (non-hard) composition grade;
- no mass or zero-mode tier change;
- continuous/restart equality;
- no GP or new-nucleation path;
- exactly one periodic six-neighbour component at `h(phi)>1e-4` in every
  state, at least 90% support overlap with the initial particle, and no
  unexpected merge/split.

Only after this gate may a multi-particle 6 h fixture cite the library as its
initial-state source.
