# Physical lower-boundary operator parity

Physical-quantity status: `PASS_PHYSICAL_LOWER_BOUNDARY_PARITY`.  Composite lower-boundary gate: `PASS_LOWER_BOUNDARY_OPERATOR_PARITY`.

| quantity | left | right | relative_error | pass_1e-12 |
|---|---|---|---|---|
| growth_velocity_eulerian_vs_helper | -11767189.594675226 | -11767189.594675226 | 0.0 | True |
| growth_velocity_fv_face_vs_helper | -11767189.594675226 | -11767189.594675226 | 0.0 | True |
| number_flux_fv_vs_helper | 3.2876747406250725e-60 | 3.2876747406250725e-60 | 0.0 | True |
| growth_velocity_cohort_vs_helper | -11767189.594675226 | -11767189.594675226 | 0.0 | True |
| number_flux_cohort_vs_helper | 3.2876747406250725e-60 | 3.2876747406250725e-60 | 0.0 | True |
| cohort_first_cell_density_vs_eulerian | 2.7939336866916626e-67 | 2.7939336866916626e-67 | 0.0 | True |
| particle_volume_cohort_vs_helper | 4.415966530881012e-28 | 4.415966530881012e-28 | 0.0 | True |
| particle_beta_moles_cohort_vs_helper | 1.0768286305154996e-23 | 1.0768286305154996e-23 | 0.0 | True |
| particle_B_moles_cohort_vs_helper | 1.0768286305154996e-23 | 1.0768286305154996e-23 | 0.0 | True |
| volume_flux_cohort_vs_helper | 1.4518261619023232e-87 | 1.4518261619023232e-87 | 0.0 | True |
| mol_B_flux_cohort_vs_helper | 3.540262288527697e-83 | 3.540262288527697e-83 | 0.0 | True |

The Eulerian lower face is accepted only when its actual finite-volume face velocity, number flux, inventory price, and cohort event/operator path all agree with the shared Rmin helper.
