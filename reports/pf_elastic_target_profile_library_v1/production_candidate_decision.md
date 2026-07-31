# Elastic constrained target-profile library production decision

## Final decision

```text
final_status=PASS_PF_ELASTIC_TARGET_PROFILE_LIBRARY_ENGINEERING_V1
selected_library_status=PASS_PF_ELASTIC_TARGET_PROFILE_LIBRARY_V1
selected_library_manifest_sha256=58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe
energy_provenance_status=PASS_ELASTIC_TARGET_PROFILE_ENERGY_PROVENANCE_V1
cluster_dt_refinement_status=PASS_ELASTIC_TARGET_PROFILE_DT_AND_DEVICE_REFINEMENT_V1
finite_box_status=PASS_ELASTIC_TARGET_PROFILE_PORTABLE_CORRECTION_BOX_SIZE_V1
dynamic_handoff_status=PASS_ELASTIC_TARGET_PROFILE_DYNAMIC_RESTART_AND_DT_V1
restart_status=PASS_BYTEWISE
particle_identity_status=PASS_ELASTIC_TARGET_PROFILE_SINGLE_PARTICLE_IDENTITY_V1
```

The exact registered entries are qualified as initialization artifacts for
the elastic, post-nucleation resolved-beta study.

## What was qualified

- T=380 °C, \(dx=1\) nm, \(\lambda_{\rm sm}=4\) nm;
- periodic fixed-cell elasticity;
- identity/[100] orientation;
- eigenstrain \((0.046,-0.022,-0.017,0,0,0)\);
- equivalent radii 8.0–11.5 nm in 0.5 nm increments;
- exact constrained \(h(\phi)\) volume and canonical B inventory;
- authoritative float64 raw \(\phi\), \(x_B^\alpha\), canonical composition,
  and portable far-subtracted composition correction;
- selected `minimize_dt=0.025` ladder against an independent `dt=0.05`
  ladder;
- finite-box behavior on 96³, 128³, 160³, and 192³;
- one-step dynamic handoff, production-dt/refined-dt continuation,
  bytewise checkpoint/restart, zero-mode provenance, and particle identity
  for R=8.0, 9.5, and 11.5 nm.

No GP, GP Birth, GP release, external source, or new beta nucleation path was
enabled.

## Numerical grades

- Selected library profile count: 8/8.
- Largest cluster refinement canonical-field L1:
  \(8.68\times10^{-5}\), below \(10^{-4}\).
- Largest cluster refinement mean \(|\Delta x_B|\):
  \(4.57\times10^{-7}\).
- Largest selected-profile mass error:
  \(3.54\times10^{-16}\).
- Dynamic mass error:
  \(4.57\times10^{-16}\) to \(8.73\times10^{-15}\).
- Dynamic continuous/restart: bytewise for all three registered probes.
- Dynamic dt composition differences:
  \(3.91\times10^{-5}\) to \(4.38\times10^{-5}\).

The dynamic composition refinement passes the hard
\(5\times10^{-5}\) gate but does not receive the preferred
\(2\times10^{-5}\) grade. This is recorded as a numerical grade, not hidden
or relabeled as failure.

## Finite-box interpretation

The 160³→192³ comparison changes:

- major/minor axis ratio by 0.0998%;
- the largest semi-axis by 0.0577%;
- centered local \(\phi\) by 0.0814%;
- box-integrated elastic energy by 0.3952%.

The far-subtracted local composition passes. Absolute isolated-box
\(x_B^\alpha\) differs because fixing one particle's inventory produces the
expected \(L^{-3}\) uniform matrix offset. Therefore:

- `delta_C_relaxation` is the portable composition artifact;
- raw absolute `xB_alpha` is qualified only for exact same-grid direct load;
- a target multi-particle box must reconstruct its registered matrix
  baseline and close the exact total inventory with the conserved zero mode.

## Explicit boundaries

This decision does not qualify:

- interpolation between registered radii;
- rotated orientation variants;
- analytic multi-particle overlay;
- independent superposition of elastic or absolute-composition fields;
- an experimental 6 h PSD fixture;
- a full 6–48 h multi-particle production run.

Those are separate materialization and scientific-qualification tasks.

## Resource use

The cluster generated and refined the final eight-radius library. The
workstation independently handled the four-box audit and dynamic/restart/dt
probes using transferred, hash-verified selected profiles. It was not used
to duplicate the full final ladder solely for device comparison.

## Recommended next action

Build a deterministic multi-particle 6 h fixture materializer that places
only exact registered entries, combines their bounded phase fields and
portable `delta_C_relaxation` contributions, solves the target-box matrix
baseline/zero mode for the frozen total inventory, and then qualifies
overlap, mass, dt, restart, merge-aware identity, and elastic-energy
provenance before any new 6–48 h statistical production.
