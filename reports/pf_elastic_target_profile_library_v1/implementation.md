# Elastic constrained target-profile implementation

## Runtime path

The implementation extends full-model `minimize` mode with an offline-only
canonical mass constraint.  The algorithm for one iteration is:

1. solve periodic mechanical equilibrium for the current beta shape;
2. advance the phase field under chemical, interface, and elastic driving;
3. enforce the registered \(h(\phi)\)-volume;
4. advance the conserved composition field;
5. solve one scalar zero-mode shift of \(Y\) so the exact canonical B ledger
   equals its frozen target;
6. evaluate energy, projected KKT residual, field rates, volume, and mass;
7. prevent the convergence counter from starting before the common
   gradient-flow pseudo-time 300;
8. stop only after all registered gates then hold consecutively.

The scalar mass solve is monotone and uses Newton/bisection safeguards.  It
does not assemble a spatial matrix, use a global mass projection during
production dynamics, or enable any GP path.

## Why a sphere is still accepted as the seed

The sphere is only a deterministic initial guess with the requested
equivalent volume.  It is not copied directly into the 6 h fixture.  The
elastic minimizer is free to change aspect ratio and principal directions
while volume and total B remain fixed.  The materialized result—not the
analytic sphere—is the library entry.

## Exact outputs

The final CUDA state is written as raw float64 arrays before VTK formatting.
The materializer reconstructs

\[
C_B=(1-h)x_B^\alpha+v_Bh
\]

and verifies it against the final constraint marker and trace.  ASCII VTK is
diagnostic only and is not the authoritative library source.

The materializer also stores the far-field-subtracted composition
correction

\[
\delta C_{\rm relax}=(1-h)(x_B^\alpha-x_{B,\rm far}^\alpha).
\]

This is the portable composition part of an isolated entry.  The absolute
`xB_alpha` raw field is retained for exact same-grid single-particle loading,
but its uniform matrix offset is not portable across box volumes.

## Code surfaces

- `main_cuda.cu`, `pf_params.h`: constrained minimization controls, final
  audits, and exact raw output;
- `cuda_kernels.cu`, `cuda_kernels.h`: bound-aware projected KKT residual;
- `scripts/materialize_pf_elastic_target_profile_v1.py`: field and geometry
  materialization;
- `scripts/assemble_pf_elastic_target_profile_library_v1.py`: radius-ladder
  and invariant validation;
- `scripts/test_pf_elastic_target_profile_v1.py`: host/static/materializer
  regression tests;
- `scripts/audit_pf_elastic_target_profile_energy_v1.py`: positive
  radius-resolved elastic-energy endpoint and eigenstrain provenance;
- `scripts/audit_pf_elastic_target_profile_identity_v1.py`: periodic,
  fail-closed single-particle identity through handoff, dt refinement, and
  restart;
- `scripts/audit_pf_elastic_target_profile_box_series_v1.py`: multi-box
  shape, integrated-energy, portable-composition, and \(L^{-3}\) inventory
  offset audit;
- `jobs/run_pf_elastic_target_profile_library_v1.sh`: non-overwriting
  workstation/cluster runner;
- `jobs/submit_pf_elastic_target_profile_library_v1.sbatch`: cluster
  qualification wrapper.
