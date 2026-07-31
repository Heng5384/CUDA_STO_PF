# Fixture-artifact resolution plan

V1's rapid 32 -> 8 synchronized collapse is attributable to its 4x4x2
periodic lattice, three repeated radii, and only 0.214 nm minimum clearance.
V2 removes those artifacts by using 96 unique deterministic quantiles and a
periodic hard-core random population with 4.896852 nm minimum clearance.

The 6--12 h tracker already shows non-synchronized component loss (96, 96, 70,
56, 48, 42, 34) with no merge/split event. The 12--48 h continuation is still
required to demonstrate that this improvement persists through the registered
48 h endpoint.

`fixture_artifact_resolution_status=PRELIMINARY_PASS_6H12H_PENDING_48H`
