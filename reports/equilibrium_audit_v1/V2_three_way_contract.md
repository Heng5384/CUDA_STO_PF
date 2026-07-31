# V2 three-way contract

V2-A is the frozen analytic tanh/elastic-OFF reference; V2-B changes only to local-equilibrium profiles with elasticity OFF; V2-C is the same local profiles with elasticity ON. Centers, radii, inventory, composition, timestep, output cadence and particle identities must be identical.

Current execution status: V2-A remains frozen historical evidence.  The clean
branch's zero-mode provenance guard still rejects the required local-profile
minimize/raw/VTK handoff.  An isolated extension nevertheless materialized and
ran matched 96^3 V2-B/C cases; those validation-only results are recorded in
`multi_particle_v2_results_v1.md` and are not inferred from V2-A or promoted
to clean production evidence.
