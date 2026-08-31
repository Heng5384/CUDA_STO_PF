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
constexpr const char* kLegacyInitialStateClass = "LEGACY_UNSPECIFIED";
constexpr const char* kElasticStateNotRequired =
    "ELASTIC_RUNTIME_STATE_NOT_REQUIRED";
constexpr const char* kElasticWarmStartResidualV1 =
    "ELASTIC_WARM_START_RESIDUAL_V1";
constexpr const char* kConditionalHandoffV1 =
    "MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1";
constexpr const char* kLegacyZeroAux = "LEGACY_ZERO_AUX";
constexpr const char* kAuxiliaryStateStorageOnlyV1 =
    "AUXILIARY_POPULATION_STORAGE_ONLY_V1";
constexpr const char* kAuxiliaryInventoryUnitsMolB = "mol_B";
constexpr const char* kLegacyUnboundValidationContractHash =
    "LEGACY_UNBOUND_VALIDATION_CONTRACT";
constexpr std::uint32_t kAuxiliaryPopulationSchemaV1 = 1U;

struct Provenance {
    std::string zero_mode;
    std::string backend;
    std::string composition_mode;
    std::string y_update_mode;
    std::string explicit_context;
    std::string reaction_discretization;
    // Version-3 checkpoint identity.  Legacy callers retain the explicit
    // LEGACY_UNSPECIFIED sentinel and can still read historical V2 files.
    std::string initial_state_class = kLegacyInitialStateClass;
    std::string fixture_manifest_sha256 = kLegacyInitialStateClass;
    std::string profile_library_manifest_sha256 =
        kLegacyInitialStateClass;
    // SHA-256 of the common PF/KWN validation contract.  V2/V3/V4 files do
    // not contain this identity and are represented explicitly by the legacy
    // unbound sentinel on read; they cannot be restarted as a current hash.
    std::string validation_contract_hash =
        kLegacyUnboundValidationContractHash;
    std::string elastic_solver_mode = kElasticStateNotRequired;
    std::uint64_t elastic_solver_fingerprint = 0U;
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

struct ElasticRuntimeState {
    bool present = false;
    // The warm state at a checkpoint written after accepted step n is the
    // converged mechanical solution used at the start of step n, hence it is
    // associated with source field step n-1.
    std::uint64_t source_field_step = 0U;
    std::uint64_t last_iterations = 0U;
    double last_relative_residual = 0.0;
    // Packed ux,uy,uz complex-float arrays.  Each complex value is stored as
    // two adjacent floats, so the vector length is 6*k_element_count.
    std::vector<float> displacement_k;
};

// A compact, host-side population record.  Inventory is the absolute amount
// of B assigned to this bin, in mol_B; it is not a concentration and is never
// implicitly injected into the local PF composition field.
struct AuxiliaryPopulationBin {
    double radius_lower_m = 0.0;
    double radius_upper_m = 0.0;
    double number_density_m3 = 0.0;
    std::uint64_t count = 0U;
    double xB = 0.0;
    double molar_volume_m3_mol = 0.0;
    double inventory_mol = 0.0;
};

// This state is deliberately storage-only in v1.  It is persisted with the
// checkpoint and participates in the four-bucket ledger, but it does not
// alter phi, xB, chemical potentials, or PF dynamics.  A default-constructed
// instance is an explicit legacy zero-aux state; readers of V2/V3/V4 files
// return exactly that state rather than inferring any missing inventory.
struct AuxPopulationState {
    std::uint32_t schema_version = 0U;
    bool present = false;
    bool frozen = true;
    std::string state = kLegacyZeroAux;
    // SHA-256 of the canonical validation thermodynamic/kinetic contract.
    std::string validation_contract_hash;
    // SHA-256 of the package that prescribed this auxiliary population.
    std::string source_handoff_hash;
    std::string units = kAuxiliaryInventoryUnitsMolB;
    double Q_B_GP_mol = 0.0;
    double Q_B_beta_subgrid_mol = 0.0;
    std::string gp_population_provenance;
    std::string beta_subgrid_population_provenance;
    std::vector<AuxiliaryPopulationBin> gp_bins;
    std::vector<AuxiliaryPopulationBin> beta_subgrid_bins;
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
    ElasticRuntimeState elastic;
    AuxPopulationState aux;
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

// These helpers expose the compact auxiliary ledger without making callers
// reimplement a potentially inconsistent sum over retained PSD bins.
double auxiliary_population_inventory_mol(
    const std::vector<AuxiliaryPopulationBin>& bins);
double auxiliary_total_inventory_mol(const AuxPopulationState& state);

std::uint64_t fnv1a64(const void* data, std::size_t size,
                      std::uint64_t seed = 1469598103934665603ULL);

}  // namespace pf_zero_mode

#endif  // PF_ZERO_MODE_CHECKPOINT_H
