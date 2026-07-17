# Production candidate decision

Selected evaluated candidate: `G9+dt4`.
Qualified production candidate: `NONE`.
Final V2 status: `FAIL_V2_MATERIAL_TRANSPORT_DEFECT`.

The preregistered V1 status remains independent and is not rewritten.

## Gate attribution

- Global normalized defect: `PASS`.
- Interface normalized defect: `PASS`.
- Material-cell normalized defect: `PASS`.
- Signed-bias gate: `FAIL`.
- Signed-growth trend gate: `PASS`.
- QoI: `PASS`; numerical/retry: `PASS`.
- Holdout: `FAIL` with `b_interface=3.430660e-03`.

The candidate is rejected because interface-local residual defects retain a systematic sign well above the frozen `1e-3` fraction limit. The absolute defect remains tiny relative to real interface evolution, but V2 does not permit that magnitude result to override a signed-bias hard-gate failure.
