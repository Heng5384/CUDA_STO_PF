#ifndef PF_AUXILIARY_HANDOFF_V2_H
#define PF_AUXILIARY_HANDOFF_V2_H

#include <string>

#include "pf_zero_mode_checkpoint.h"

namespace pf_auxiliary_handoff_v2 {

// Plain-text companion schema emitted with the validation-only, fixture-
// conditioned KWN--PF handoff v2 package.  It materializes only the compact
// frozen GP/sub-grid-beta state; it neither owns nor initializes PF fields.
constexpr const char* kAuxiliaryHandoffSidecarSchemaV1 =
    "PF_AUXILIARY_HANDOFF_V2_SIDECAR_V1";

// All four hashes are consumed as a single immutable identity.  This keeps the
// fresh-run CUDA adapter from accepting a package that belongs to the right
// thermodynamic contract but a different source handoff or fixture.
struct AuxiliaryHandoffIdentity {
    std::string validation_contract_hash;
    std::string source_handoff_hash;
    std::string package_hash;
    std::string fixture_hash;
};

// Parse a deterministic sidecar into the checkpoint-owned storage type.
// The complete expected identity is a restart identity, not optional
// annotation: any mismatch is rejected before state is returned.  Each Python
// n(R) bin supplied in m^-4 is integrated over its radius width into the
// returned AuxiliaryPopulationBin::number_density_m3 value.  The continuum
// bin count remains zero because it is not a discrete PF particle count.
bool read_auxiliary_population_handoff_v2(
    const std::string& path,
    const AuxiliaryHandoffIdentity& expected_identity,
    pf_zero_mode::AuxPopulationState* state,
    std::string* error);

}  // namespace pf_auxiliary_handoff_v2

#endif  // PF_AUXILIARY_HANDOFF_V2_H
