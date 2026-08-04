# V3 absolute endpoint necessary-condition audit

`PASS_V3_ABSOLUTE_ENDPOINT_NECESSARY_CONDITION`

All resolved density/stiffness, interface, conversion, roughness, coherent-strain and damping rates were set exactly to zero. The retained base contains only the selected host, grain boundary, matrix point defects, AQ-only time-invariant background and the registered A/B/C matrix composition.

- retained AQ background members: `103`
- endpoint-passing members: `14`
- endpoint-passing `SOURCE_LITERAL_BOUND` AQ cases: `0`
- all-member 573.15 K upper 6 h min/median/max: `0.855524 / 0.885191 / 1.420703` W m^-1 K^-1
- all-member 573.15 K upper 48 h min/median/max: `0.855322 / 0.884579 / 1.419865` W m^-1 K^-1
- H-P0 48 h maximum: `0.982042` W m^-1 K^-1 (all H-P0 members fail 1.03)

The mathematical hard cap is removed only by some H-P2 members. Every such member remains `PROVISIONAL_HOST_ENVELOPE`, and every passing AQ structure case is a `SINGLE_APT_COUNT_FRACTION_DIAGNOSTIC` rather than a source-literal two-endmember bound. Therefore this PASS means only that the broad replacement-model diagnostic envelope is not excluded by the endpoint inequality. Rebuilding the host/background contract removes the mathematical hard cap, but does not prove that physically realistic Ag2Te parameters will produce the required recovery.
