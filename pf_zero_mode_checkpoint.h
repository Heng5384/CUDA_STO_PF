#ifndef PF_ZERO_MODE_CHECKPOINT_H
#define PF_ZERO_MODE_CHECKPOINT_H

#include <cstdint>
#include <string>
#include <vector>

namespace pf_zero_mode {

constexpr const char* kModeOff = "OFF";
constexpr const char* kModeV1 = "PF_CONSERVED_Y_ZERO_MODE_V1";
constexpr const char* kHostBackendV1 = "HOST_NEWTON_BISECTION_V1";
constexpr const char* kExplicitContextNV1 = "SM_EXPLICIT_CONTEXT_N_V1";
constexpr const char* kReactionTangentNV1 = "SM_TANGENT_N_V1";

struct Provenance {
    std::string zero_mode;
    std::string backend;
    std::string composition_mode;
    std::string y_update_mode;
    std::string explicit_context;
    std::string reaction_discretization;
    std::uint64_t parameter_fingerprint = 0U;
};

struct RuntimeState {
    double target_mass_code = 0.0;
    double last_lambda = 0.0;
    double last_residual_code = 0.0;
    double last_derivative_code = 0.0;
    std::uint64_t last_iterations = 0U;
    std::uint64_t accepted_zero_mode_steps = 0U;
};

struct Checkpoint {
    std::uint64_t accepted_step = 0U;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    double dt_code = 0.0;
    double temperature_K = 0.0;
    Provenance provenance;
    RuntimeState zero_mode;
    std::vector<double> phi;
    std::vector<double> Y;
    std::vector<double> xB;
    std::vector<double> dY_dt_prev;
};

// The file is written through a sibling temporary path and atomically renamed
// only after the complete payload checksum has been finalized.
bool write_checkpoint(const std::string& path, const Checkpoint& checkpoint,
                      std::string* error);

// expected_provenance is a restart identity, not a hint.  A mismatch is
// rejected before any field is returned to the caller.
bool read_checkpoint(const std::string& path,
                     const Provenance& expected_provenance,
                     Checkpoint* checkpoint,
                     std::string* error);

bool validate_checkpoint(const Checkpoint& checkpoint, std::string* error);

std::uint64_t fnv1a64(const void* data, std::size_t size,
                      std::uint64_t seed = 1469598103934665603ULL);

}  // namespace pf_zero_mode

#endif  // PF_ZERO_MODE_CHECKPOINT_H
