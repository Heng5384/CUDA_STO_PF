# PF X/Q Operator Unit Tests

Formula-level tests G1-G7 were run before the expanded runtime matrix.

Rows: `22`; failed rows: `4`.

The intentional X failure in G4 is the audited production gap: current Mode X does not perform local phase-storage transfer. The X conditioning failures below alpha=0.1 document why its support rule is numerical protection rather than a physical closure.

| test | mode | pass | metric | note |
|---|---|---|---:|---|
| G1_uniform_equilibrium | X | True | 0 | zero divJ and static h leave x unchanged |
| G1_uniform_equilibrium | Q | True | 0 | zero divJ and static h leave q unchanged |
| G2_static_diffuse_interface | X | True | 0 | uniform mu gives zero flux |
| G2_static_diffuse_interface | Q | True | 0 | uniform mu gives zero flux |
| G3_pure_transport_sinusoid | Q | True | 1.0842e-18 | discrete Fourier decay matches explicit q update |
| G4_pure_phi_storage | Q | True | 5.55112e-17 | q_new=q_old-Delta h closes local total storage |
| G4_pure_phi_storage | X_current | False | 0.008 | current X leaves x fixed during phi and requires later global projection |
| G5_moving_interface_round_trip | Q | True | 2.77556e-17 | feasible local storage transfer is reversible |
| G6_h_to_one_conditioning | X | True | 1 | X inversion condition scales as 1/(1-h) |
| G6_h_to_one_conditioning | Q | True | 1 | Q evolution has no storage division |
| G6_h_to_one_conditioning | X | True | 2 | X inversion condition scales as 1/(1-h) |
| G6_h_to_one_conditioning | Q | True | 1 | Q evolution has no storage division |
| G6_h_to_one_conditioning | X | True | 10 | X inversion condition scales as 1/(1-h) |
| G6_h_to_one_conditioning | Q | True | 1 | Q evolution has no storage division |
| G6_h_to_one_conditioning | X | False | 20 | X inversion condition scales as 1/(1-h) |
| G6_h_to_one_conditioning | Q | True | 1 | Q evolution has no storage division |
| G6_h_to_one_conditioning | X | False | 100 | X inversion condition scales as 1/(1-h) |
| G6_h_to_one_conditioning | Q | True | 1 | Q evolution has no storage division |
| G6_h_to_one_conditioning | X | False | 1000 | X inversion condition scales as 1/(1-h) |
| G6_h_to_one_conditioning | Q | True | 1 | Q evolution has no storage division |
| G7_projection_identity | projection | True | 1.04083e-17 | satisfied state is unchanged |
| G7_projection_idempotence | projection | True | 6.93889e-18 | P(P(x))=P(x) |
