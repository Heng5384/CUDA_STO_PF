#include "pf_zero_mode_checkpoint.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>

namespace pf_zero_mode {
namespace {

constexpr std::uint32_t kVersionV2 = 2U;
constexpr std::uint32_t kVersionV3 = 3U;
constexpr std::uint32_t kVersionV4 = 4U;
constexpr std::uint32_t kVersionV5 = 5U;
constexpr std::uint32_t kVersionV6 = 6U;
constexpr char kMagicV2[8] = {'P', 'F', 'Z', 'M', 'C', 'H', 'K', '2'};
constexpr char kMagicV3[8] = {'P', 'F', 'Z', 'M', 'C', 'H', 'K', '3'};
constexpr char kMagicV4[8] = {'P', 'F', 'Z', 'M', 'C', 'H', 'K', '4'};
constexpr char kMagicV5[8] = {'P', 'F', 'Z', 'M', 'C', 'H', 'K', '5'};
constexpr char kMagicV6[8] = {'P', 'F', 'Z', 'M', 'C', 'H', 'K', '6'};
constexpr std::size_t kSelectorBytes = 64U;
constexpr std::size_t kIdentityBytes = 96U;

struct DiskPrefix {
    char magic[8];
    std::uint32_t version;
    std::uint32_t header_bytes;
};

// Historical format retained for read compatibility.  New writes use V3.
struct DiskHeaderV2 {
    char magic[8];
    std::uint32_t version;
    std::uint32_t header_bytes;
    std::uint64_t element_count;
    std::uint64_t accepted_step;
    std::int32_t nx;
    std::int32_t ny;
    std::int32_t nz;
    std::int32_t reserved;
    double dt_code;
    double temperature_K;
    double target_mass_code;
    double last_lambda;
    double last_residual_code;
    double last_derivative_code;
    std::uint64_t last_iterations;
    std::uint64_t accepted_zero_mode_steps;
    std::uint64_t parameter_fingerprint;
    char zero_mode[kSelectorBytes];
    char backend[kSelectorBytes];
    char composition_mode[kSelectorBytes];
    char y_update_mode[kSelectorBytes];
    char explicit_context[kSelectorBytes];
    char reaction_discretization[kSelectorBytes];
    std::uint64_t payload_checksum;
};

struct DiskHeaderV3 {
    char magic[8];
    std::uint32_t version;
    std::uint32_t header_bytes;
    std::uint64_t element_count;
    std::uint64_t accepted_step;
    std::int32_t nx;
    std::int32_t ny;
    std::int32_t nz;
    std::int32_t reserved;
    double dt_code;
    double temperature_K;
    double target_mass_code;
    double last_lambda;
    double last_residual_code;
    double last_derivative_code;
    std::uint64_t last_iterations;
    std::uint64_t accepted_zero_mode_steps;
    std::uint64_t parameter_fingerprint;
    char zero_mode[kSelectorBytes];
    char backend[kSelectorBytes];
    char composition_mode[kSelectorBytes];
    char y_update_mode[kSelectorBytes];
    char explicit_context[kSelectorBytes];
    char reaction_discretization[kSelectorBytes];
    char initial_state_class[kIdentityBytes];
    char fixture_manifest_sha256[kIdentityBytes];
    char profile_library_manifest_sha256[kIdentityBytes];
    std::uint64_t payload_checksum;
};

struct DiskHeaderV4 {
    char magic[8];
    std::uint32_t version;
    std::uint32_t header_bytes;
    std::uint64_t element_count;
    std::uint64_t k_element_count;
    std::uint64_t accepted_step;
    std::int32_t nx;
    std::int32_t ny;
    std::int32_t nz;
    std::int32_t elastic_state_present;
    double dt_code;
    double temperature_K;
    double target_mass_code;
    double last_lambda;
    double last_residual_code;
    double last_derivative_code;
    std::uint64_t last_iterations;
    std::uint64_t accepted_zero_mode_steps;
    std::uint64_t parameter_fingerprint;
    std::uint64_t elastic_solver_fingerprint;
    std::uint64_t elastic_source_field_step;
    std::uint64_t elastic_last_iterations;
    double elastic_last_relative_residual;
    char zero_mode[kSelectorBytes];
    char backend[kSelectorBytes];
    char composition_mode[kSelectorBytes];
    char y_update_mode[kSelectorBytes];
    char explicit_context[kSelectorBytes];
    char reaction_discretization[kSelectorBytes];
    char elastic_solver_mode[kSelectorBytes];
    char initial_state_class[kIdentityBytes];
    char fixture_manifest_sha256[kIdentityBytes];
    char profile_library_manifest_sha256[kIdentityBytes];
    std::uint64_t payload_checksum;
};

struct DiskPopulationBinV1 {
    double radius_lower_m;
    double radius_upper_m;
    double number_density_m3;
    std::uint64_t count;
    double xB;
    double molar_volume_m3_mol;
    double inventory_mol;
};

// Historical V5 layout retained for read compatibility.  It adds a compact
// host-side auxiliary-population payload to V4, but has no package identity.
// The auxiliary bins are serialized after the field/elastic payload so their
// complete PSD is covered by the same payload checksum.
struct DiskHeaderV5 {
    char magic[8];
    std::uint32_t version;
    std::uint32_t header_bytes;
    std::uint64_t element_count;
    std::uint64_t k_element_count;
    std::uint64_t accepted_step;
    std::int32_t nx;
    std::int32_t ny;
    std::int32_t nz;
    std::int32_t elastic_state_present;
    std::uint32_t aux_state_present;
    std::uint32_t aux_schema_version;
    std::uint32_t aux_frozen;
    std::uint32_t reserved;
    double dt_code;
    double temperature_K;
    double target_mass_code;
    double last_lambda;
    double last_residual_code;
    double last_derivative_code;
    std::uint64_t last_iterations;
    std::uint64_t accepted_zero_mode_steps;
    std::uint64_t parameter_fingerprint;
    std::uint64_t elastic_solver_fingerprint;
    std::uint64_t elastic_source_field_step;
    std::uint64_t elastic_last_iterations;
    double elastic_last_relative_residual;
    std::uint64_t gp_bin_count;
    std::uint64_t beta_subgrid_bin_count;
    double Q_B_GP_mol;
    double Q_B_beta_subgrid_mol;
    char zero_mode[kSelectorBytes];
    char backend[kSelectorBytes];
    char composition_mode[kSelectorBytes];
    char y_update_mode[kSelectorBytes];
    char explicit_context[kSelectorBytes];
    char reaction_discretization[kSelectorBytes];
    char elastic_solver_mode[kSelectorBytes];
    char auxiliary_state[kSelectorBytes];
    char auxiliary_units[kSelectorBytes];
    char initial_state_class[kIdentityBytes];
    char fixture_manifest_sha256[kIdentityBytes];
    char profile_library_manifest_sha256[kIdentityBytes];
    // Checkpoint-level contract identity.  Active auxiliary state is required
    // to bind to this same hash, rather than carrying a divergent copy.
    char validation_contract_hash[kIdentityBytes];
    char source_handoff_hash[kIdentityBytes];
    char gp_population_provenance[kIdentityBytes];
    char beta_subgrid_population_provenance[kIdentityBytes];
    std::uint64_t payload_checksum;
};

// V6 is the sole write format.  It retains the V5 compact auxiliary payload
// and adds the canonical package hash required to prove a restart still refers
// to the exact fixture-conditioned handoff package.  V5 remains readable as
// a legacy auxiliary format whose missing package identity cannot be reissued.
struct DiskHeaderV6 {
    char magic[8];
    std::uint32_t version;
    std::uint32_t header_bytes;
    std::uint64_t element_count;
    std::uint64_t k_element_count;
    std::uint64_t accepted_step;
    std::int32_t nx;
    std::int32_t ny;
    std::int32_t nz;
    std::int32_t elastic_state_present;
    std::uint32_t aux_state_present;
    std::uint32_t aux_schema_version;
    std::uint32_t aux_frozen;
    std::uint32_t reserved;
    double dt_code;
    double temperature_K;
    double target_mass_code;
    double last_lambda;
    double last_residual_code;
    double last_derivative_code;
    std::uint64_t last_iterations;
    std::uint64_t accepted_zero_mode_steps;
    std::uint64_t parameter_fingerprint;
    std::uint64_t elastic_solver_fingerprint;
    std::uint64_t elastic_source_field_step;
    std::uint64_t elastic_last_iterations;
    double elastic_last_relative_residual;
    std::uint64_t gp_bin_count;
    std::uint64_t beta_subgrid_bin_count;
    double Q_B_GP_mol;
    double Q_B_beta_subgrid_mol;
    char zero_mode[kSelectorBytes];
    char backend[kSelectorBytes];
    char composition_mode[kSelectorBytes];
    char y_update_mode[kSelectorBytes];
    char explicit_context[kSelectorBytes];
    char reaction_discretization[kSelectorBytes];
    char elastic_solver_mode[kSelectorBytes];
    char auxiliary_state[kSelectorBytes];
    char auxiliary_units[kSelectorBytes];
    char initial_state_class[kIdentityBytes];
    char fixture_manifest_sha256[kIdentityBytes];
    char profile_library_manifest_sha256[kIdentityBytes];
    char validation_contract_hash[kIdentityBytes];
    char source_handoff_hash[kIdentityBytes];
    char package_handoff_hash[kIdentityBytes];
    char gp_population_provenance[kIdentityBytes];
    char beta_subgrid_population_provenance[kIdentityBytes];
    std::uint64_t payload_checksum;
};

bool set_error(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

bool finite_vector(const std::vector<double>& values) {
    for (double value : values) {
        if (!std::isfinite(value)) return false;
    }
    return true;
}

bool finite_float_vector(const std::vector<float>& values) {
    for (float value : values) {
        if (!std::isfinite(value)) return false;
    }
    return true;
}

bool copy_selector(char* destination, const std::string& value,
                   const char* label, std::string* error) {
    if (value.empty() || value.size() >= kSelectorBytes) {
        return set_error(error, std::string(label) + " is empty or too long");
    }
    std::memset(destination, 0, kSelectorBytes);
    std::memcpy(destination, value.data(), value.size());
    return true;
}

bool copy_identity(char* destination, const std::string& value,
                   const char* label, std::string* error) {
    if (value.empty() || value.size() >= kIdentityBytes) {
        return set_error(error, std::string(label) + " is empty or too long");
    }
    std::memset(destination, 0, kIdentityBytes);
    std::memcpy(destination, value.data(), value.size());
    return true;
}

bool write_exact(FILE* fp, const void* data, std::size_t size) {
    return size == 0U || std::fwrite(data, 1U, size, fp) == size;
}

bool read_exact(FILE* fp, void* data, std::size_t size) {
    return size == 0U || std::fread(data, 1U, size, fp) == size;
}

bool selector_equal(const char* disk, const std::string& expected) {
    return expected.size() < kSelectorBytes &&
           std::strncmp(disk, expected.c_str(), kSelectorBytes) == 0 &&
           disk[expected.size()] == '\0';
}

bool identity_equal(const char* disk, const std::string& expected) {
    return expected.size() < kIdentityBytes &&
           std::strncmp(disk, expected.c_str(), kIdentityBytes) == 0 &&
           disk[expected.size()] == '\0';
}

bool read_selector_value(const char* disk, std::string* value) {
    for (std::size_t index = 0U; index < kSelectorBytes; ++index) {
        if (disk[index] == '\0') {
            if (index == 0U) return false;
            value->assign(disk, index);
            return true;
        }
    }
    return false;
}

bool read_identity_value(const char* disk, std::string* value) {
    for (std::size_t index = 0U; index < kIdentityBytes; ++index) {
        if (disk[index] == '\0') {
            if (index == 0U) return false;
            value->assign(disk, index);
            return true;
        }
    }
    return false;
}

bool legacy_identity_expected(const Provenance& provenance) {
    return provenance.initial_state_class == kLegacyInitialStateClass &&
           provenance.fixture_manifest_sha256 == kLegacyInitialStateClass &&
           provenance.profile_library_manifest_sha256 ==
               kLegacyInitialStateClass;
}

bool elastic_state_expected(const Provenance& provenance) {
    return provenance.elastic_solver_mode != kElasticStateNotRequired ||
           provenance.elastic_solver_fingerprint != 0U;
}

bool finite_auxiliary_bin(const AuxiliaryPopulationBin& bin) {
    return std::isfinite(bin.radius_lower_m) &&
           std::isfinite(bin.radius_upper_m) &&
           std::isfinite(bin.number_density_m3) && std::isfinite(bin.xB) &&
           std::isfinite(bin.molar_volume_m3_mol) &&
           std::isfinite(bin.inventory_mol) && bin.radius_lower_m >= 0.0 &&
           bin.radius_upper_m > bin.radius_lower_m &&
           bin.number_density_m3 >= 0.0 && bin.xB >= 0.0 && bin.xB <= 1.0 &&
           bin.molar_volume_m3_mol > 0.0 && bin.inventory_mol >= 0.0;
}

bool inventories_match(double declared, double summed) {
    const double scale =
        std::fmax(1.0e-30,
                  std::fmax(std::fabs(declared), std::fabs(summed)));
    return std::fabs(declared - summed) <= 1.0e-12 * scale;
}

bool legacy_zero_aux(const AuxPopulationState& aux) {
    return !aux.present && aux.schema_version == 0U && aux.frozen &&
           aux.state == kLegacyZeroAux &&
           aux.validation_contract_hash.empty() &&
           aux.source_handoff_hash.empty() &&
           aux.package_handoff_hash.empty() &&
           aux.units == kAuxiliaryInventoryUnitsMolB &&
           aux.Q_B_GP_mol == 0.0 && aux.Q_B_beta_subgrid_mol == 0.0 &&
           aux.gp_population_provenance.empty() &&
           aux.beta_subgrid_population_provenance.empty() &&
           aux.gp_bins.empty() && aux.beta_subgrid_bins.empty();
}

bool validate_auxiliary_state(const AuxPopulationState& aux,
                              std::string* error) {
    if (!aux.present) {
        if (!legacy_zero_aux(aux)) {
            return set_error(
                error,
                "absent auxiliary state must be the explicit legacy zero state");
        }
        return true;
    }
    if (aux.schema_version != kAuxiliaryPopulationSchemaV1 ||
        !aux.frozen || aux.state.empty() ||
        aux.validation_contract_hash.empty() || aux.source_handoff_hash.empty() ||
        aux.units != kAuxiliaryInventoryUnitsMolB ||
        aux.gp_population_provenance.empty() ||
        aux.beta_subgrid_population_provenance.empty() ||
        !std::isfinite(aux.Q_B_GP_mol) ||
        !std::isfinite(aux.Q_B_beta_subgrid_mol) || aux.Q_B_GP_mol < 0.0 ||
        aux.Q_B_beta_subgrid_mol < 0.0) {
        return set_error(error, "auxiliary population state is incomplete");
    }
    for (const AuxiliaryPopulationBin& bin : aux.gp_bins) {
        if (!finite_auxiliary_bin(bin)) {
            return set_error(error, "GP auxiliary population bin is invalid");
        }
    }
    for (const AuxiliaryPopulationBin& bin : aux.beta_subgrid_bins) {
        if (!finite_auxiliary_bin(bin)) {
            return set_error(
                error, "subgrid-beta auxiliary population bin is invalid");
        }
    }
    if ((aux.Q_B_GP_mol > 0.0 && aux.gp_bins.empty()) ||
        (aux.Q_B_beta_subgrid_mol > 0.0 &&
         aux.beta_subgrid_bins.empty()) ||
        !inventories_match(aux.Q_B_GP_mol,
                           auxiliary_population_inventory_mol(aux.gp_bins)) ||
        !inventories_match(
            aux.Q_B_beta_subgrid_mol,
            auxiliary_population_inventory_mol(aux.beta_subgrid_bins))) {
        return set_error(error, "auxiliary population inventory ledger is inconsistent");
    }
    return true;
}

DiskPopulationBinV1 to_disk_bin(const AuxiliaryPopulationBin& bin) {
    DiskPopulationBinV1 disk = {};
    disk.radius_lower_m = bin.radius_lower_m;
    disk.radius_upper_m = bin.radius_upper_m;
    disk.number_density_m3 = bin.number_density_m3;
    disk.count = bin.count;
    disk.xB = bin.xB;
    disk.molar_volume_m3_mol = bin.molar_volume_m3_mol;
    disk.inventory_mol = bin.inventory_mol;
    return disk;
}

AuxiliaryPopulationBin from_disk_bin(const DiskPopulationBinV1& disk) {
    AuxiliaryPopulationBin bin;
    bin.radius_lower_m = disk.radius_lower_m;
    bin.radius_upper_m = disk.radius_upper_m;
    bin.number_density_m3 = disk.number_density_m3;
    bin.count = disk.count;
    bin.xB = disk.xB;
    bin.molar_volume_m3_mol = disk.molar_volume_m3_mol;
    bin.inventory_mol = disk.inventory_mol;
    return bin;
}

}  // namespace

std::uint64_t fnv1a64(const void* data, std::size_t size, std::uint64_t seed) {
    const unsigned char* bytes = static_cast<const unsigned char*>(data);
    std::uint64_t hash = seed;
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= static_cast<std::uint64_t>(bytes[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

double auxiliary_population_inventory_mol(
    const std::vector<AuxiliaryPopulationBin>& bins) {
    double total = 0.0;
    for (const AuxiliaryPopulationBin& bin : bins) {
        total += bin.inventory_mol;
    }
    return total;
}

double auxiliary_total_inventory_mol(const AuxPopulationState& state) {
    return state.Q_B_GP_mol + state.Q_B_beta_subgrid_mol;
}

bool validate_checkpoint(const Checkpoint& checkpoint, std::string* error) {
    if (checkpoint.nx <= 0 || checkpoint.ny <= 0 || checkpoint.nz <= 0) {
        return set_error(error, "checkpoint grid dimensions are invalid");
    }
    const std::uint64_t count =
        static_cast<std::uint64_t>(checkpoint.nx) *
        static_cast<std::uint64_t>(checkpoint.ny) *
        static_cast<std::uint64_t>(checkpoint.nz);
    if (count > static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())) {
        return set_error(error, "checkpoint element count overflows size_t");
    }
    if (checkpoint.phi.size() != count || checkpoint.Y.size() != count ||
        checkpoint.xB.size() != count ||
        checkpoint.dY_dt_prev.size() != count) {
        return set_error(error, "checkpoint field sizes do not match the grid");
    }
    if (!(checkpoint.dt_code > 0.0) || !std::isfinite(checkpoint.dt_code) ||
        !(checkpoint.temperature_K > 0.0) ||
        !std::isfinite(checkpoint.temperature_K)) {
        return set_error(error, "checkpoint time/temperature is invalid");
    }
    if (checkpoint.provenance.zero_mode != kModeV1 ||
        checkpoint.provenance.backend.empty() ||
        checkpoint.provenance.composition_mode.empty() ||
        checkpoint.provenance.y_update_mode.empty() ||
        checkpoint.provenance.explicit_context.empty() ||
        checkpoint.provenance.reaction_discretization.empty() ||
        checkpoint.provenance.initial_state_class.empty() ||
        checkpoint.provenance.fixture_manifest_sha256.empty() ||
        checkpoint.provenance.profile_library_manifest_sha256.empty() ||
        checkpoint.provenance.validation_contract_hash.empty() ||
        checkpoint.provenance.parameter_fingerprint == 0U) {
        return set_error(error, "checkpoint zero-mode provenance is incomplete");
    }
    const bool expects_elastic =
        elastic_state_expected(checkpoint.provenance);
    if (expects_elastic != checkpoint.elastic.present) {
        return set_error(
            error,
            "checkpoint elastic state/provenance presence mismatch");
    }
    if (checkpoint.elastic.present) {
        const std::uint64_t k_count =
            static_cast<std::uint64_t>(checkpoint.nx) *
            static_cast<std::uint64_t>(checkpoint.ny) *
            static_cast<std::uint64_t>(checkpoint.nz / 2 + 1);
        if (checkpoint.provenance.elastic_solver_mode !=
                kElasticWarmStartResidualV1 ||
            checkpoint.provenance.elastic_solver_fingerprint == 0U ||
            checkpoint.accepted_step == 0U ||
            checkpoint.elastic.source_field_step + 1U !=
                checkpoint.accepted_step ||
            checkpoint.elastic.displacement_k.size() != 6U * k_count ||
            !std::isfinite(checkpoint.elastic.last_relative_residual) ||
            checkpoint.elastic.last_relative_residual < 0.0 ||
            !finite_float_vector(checkpoint.elastic.displacement_k)) {
            return set_error(error,
                             "checkpoint elastic runtime state is invalid");
        }
    } else if (!checkpoint.elastic.displacement_k.empty()) {
        return set_error(
            error,
            "checkpoint has elastic displacement data without provenance");
    }
    if (!validate_auxiliary_state(checkpoint.aux, error)) return false;
    if (checkpoint.aux.present &&
        checkpoint.aux.validation_contract_hash !=
            checkpoint.provenance.validation_contract_hash) {
        return set_error(
            error,
            "auxiliary contract hash differs from checkpoint provenance");
    }
    if (!std::isfinite(checkpoint.zero_mode.target_mass_code) ||
        !std::isfinite(checkpoint.zero_mode.last_lambda) ||
        !std::isfinite(checkpoint.zero_mode.last_residual_code) ||
        !std::isfinite(checkpoint.zero_mode.last_derivative_code) ||
        !finite_vector(checkpoint.phi) || !finite_vector(checkpoint.Y) ||
        !finite_vector(checkpoint.xB) ||
        !finite_vector(checkpoint.dY_dt_prev)) {
        return set_error(error, "checkpoint contains a non-finite value");
    }
    return true;
}

bool write_checkpoint(const std::string& path, const Checkpoint& checkpoint,
                      std::string* error) {
    if (!validate_checkpoint(checkpoint, error)) return false;
    if (checkpoint.provenance.validation_contract_hash ==
        kLegacyUnboundValidationContractHash) {
        return set_error(
            error,
            "V6 checkpoint requires a bound validation contract hash");
    }
    DiskHeaderV6 header = {};
    std::memcpy(header.magic, kMagicV6, sizeof(kMagicV6));
    header.version = kVersionV6;
    header.header_bytes = sizeof(DiskHeaderV6);
    header.element_count = checkpoint.phi.size();
    header.k_element_count = checkpoint.elastic.present
                                 ? checkpoint.elastic.displacement_k.size() / 6U
                                 : 0U;
    header.accepted_step = checkpoint.accepted_step;
    header.nx = checkpoint.nx;
    header.ny = checkpoint.ny;
    header.nz = checkpoint.nz;
    header.elastic_state_present = checkpoint.elastic.present ? 1 : 0;
    header.aux_state_present = checkpoint.aux.present ? 1U : 0U;
    header.aux_schema_version = checkpoint.aux.schema_version;
    header.aux_frozen = checkpoint.aux.frozen ? 1U : 0U;
    header.dt_code = checkpoint.dt_code;
    header.temperature_K = checkpoint.temperature_K;
    header.target_mass_code = checkpoint.zero_mode.target_mass_code;
    header.last_lambda = checkpoint.zero_mode.last_lambda;
    header.last_residual_code = checkpoint.zero_mode.last_residual_code;
    header.last_derivative_code = checkpoint.zero_mode.last_derivative_code;
    header.last_iterations = checkpoint.zero_mode.last_iterations;
    header.accepted_zero_mode_steps =
        checkpoint.zero_mode.accepted_zero_mode_steps;
    header.parameter_fingerprint =
        checkpoint.provenance.parameter_fingerprint;
    header.elastic_solver_fingerprint =
        checkpoint.provenance.elastic_solver_fingerprint;
    header.elastic_source_field_step =
        checkpoint.elastic.source_field_step;
    header.elastic_last_iterations = checkpoint.elastic.last_iterations;
    header.elastic_last_relative_residual =
        checkpoint.elastic.last_relative_residual;
    header.gp_bin_count = checkpoint.aux.gp_bins.size();
    header.beta_subgrid_bin_count = checkpoint.aux.beta_subgrid_bins.size();
    header.Q_B_GP_mol = checkpoint.aux.Q_B_GP_mol;
    header.Q_B_beta_subgrid_mol = checkpoint.aux.Q_B_beta_subgrid_mol;
    if (!copy_selector(header.zero_mode, checkpoint.provenance.zero_mode,
                       "zero_mode", error) ||
        !copy_selector(header.backend, checkpoint.provenance.backend,
                       "backend", error) ||
        !copy_selector(header.composition_mode,
                       checkpoint.provenance.composition_mode,
                       "composition_mode", error) ||
        !copy_selector(header.y_update_mode,
                       checkpoint.provenance.y_update_mode,
                       "y_update_mode", error) ||
        !copy_selector(header.explicit_context,
                       checkpoint.provenance.explicit_context,
                       "explicit_context", error) ||
        !copy_selector(header.reaction_discretization,
                       checkpoint.provenance.reaction_discretization,
                       "reaction_discretization", error) ||
        !copy_selector(header.elastic_solver_mode,
                       checkpoint.provenance.elastic_solver_mode,
                       "elastic_solver_mode", error) ||
        !copy_identity(header.initial_state_class,
                       checkpoint.provenance.initial_state_class,
                       "initial_state_class", error) ||
        !copy_identity(header.fixture_manifest_sha256,
                       checkpoint.provenance.fixture_manifest_sha256,
                       "fixture_manifest_sha256", error) ||
        !copy_identity(header.profile_library_manifest_sha256,
                       checkpoint.provenance.profile_library_manifest_sha256,
                       "profile_library_manifest_sha256", error) ||
        !copy_identity(header.validation_contract_hash,
                       checkpoint.provenance.validation_contract_hash,
                       "validation_contract_hash", error)) {
        return false;
    }
    if (checkpoint.aux.present &&
        (!copy_selector(header.auxiliary_state, checkpoint.aux.state,
                        "auxiliary_state", error) ||
         !copy_selector(header.auxiliary_units, checkpoint.aux.units,
                        "auxiliary_units", error) ||
         !copy_identity(header.source_handoff_hash,
                        checkpoint.aux.source_handoff_hash,
                        "source_handoff_hash", error) ||
         !copy_identity(header.package_handoff_hash,
                        checkpoint.aux.package_handoff_hash,
                        "package_handoff_hash", error) ||
         !copy_identity(header.gp_population_provenance,
                        checkpoint.aux.gp_population_provenance,
                        "gp_population_provenance", error) ||
         !copy_identity(header.beta_subgrid_population_provenance,
                        checkpoint.aux.beta_subgrid_population_provenance,
                        "beta_subgrid_population_provenance", error))) {
        return false;
    }

    const std::string temporary = path + ".tmp";
    FILE* fp = std::fopen(temporary.c_str(), "wb+");
    if (!fp) {
        return set_error(error, "cannot open checkpoint temporary file: " +
                                std::string(std::strerror(errno)));
    }
    std::uint64_t checksum = fnv1a64(&header, sizeof(header));
    bool ok = write_exact(fp, &header, sizeof(header));
    const std::vector<double>* fields[] = {
        &checkpoint.phi, &checkpoint.Y, &checkpoint.xB,
        &checkpoint.dY_dt_prev};
    for (const std::vector<double>* field : fields) {
        const std::size_t bytes = field->size() * sizeof(double);
        if (ok) ok = write_exact(fp, field->data(), bytes);
        checksum = fnv1a64(field->data(), bytes, checksum);
    }
    const std::size_t elastic_bytes =
        checkpoint.elastic.displacement_k.size() * sizeof(float);
    if (ok) {
        ok = write_exact(fp, checkpoint.elastic.displacement_k.data(),
                         elastic_bytes);
    }
    checksum = fnv1a64(checkpoint.elastic.displacement_k.data(),
                       elastic_bytes, checksum);
    const std::vector<AuxiliaryPopulationBin>* populations[] = {
        &checkpoint.aux.gp_bins, &checkpoint.aux.beta_subgrid_bins};
    for (const std::vector<AuxiliaryPopulationBin>* population : populations) {
        for (const AuxiliaryPopulationBin& bin : *population) {
            const DiskPopulationBinV1 disk = to_disk_bin(bin);
            if (ok) ok = write_exact(fp, &disk, sizeof(disk));
            checksum = fnv1a64(&disk, sizeof(disk), checksum);
        }
    }
    if (ok) {
        header.payload_checksum = checksum;
        ok = std::fseek(fp, 0L, SEEK_SET) == 0 &&
             write_exact(fp, &header, sizeof(header)) &&
             std::fflush(fp) == 0;
    }
    if (std::fclose(fp) != 0) ok = false;
    if (!ok) {
        std::remove(temporary.c_str());
        return set_error(error, "checkpoint V6 serialization failed");
    }
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        const std::string detail = std::strerror(errno);
        std::remove(temporary.c_str());
        return set_error(error,
                         "checkpoint V6 atomic rename failed: " + detail);
    }
    return true;
}

bool read_checkpoint(const std::string& path,
                     const Provenance& expected_provenance,
                     Checkpoint* checkpoint,
                     std::string* error) {
    if (!checkpoint) return set_error(error, "checkpoint output is null");
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        return set_error(error, "cannot open checkpoint: " +
                                std::string(std::strerror(errno)));
    }
    DiskPrefix prefix = {};
    bool ok = read_exact(fp, &prefix, sizeof(prefix));
    if (!ok || std::fseek(fp, 0L, SEEK_SET) != 0) {
        std::fclose(fp);
        return set_error(error, "checkpoint header/version mismatch");
    }

    std::uint64_t element_count = 0U;
    std::uint64_t accepted_step = 0U;
    std::int32_t nx = 0, ny = 0, nz = 0;
    double dt_code = 0.0, temperature_K = 0.0;
    double target_mass_code = 0.0, last_lambda = 0.0;
    double last_residual_code = 0.0, last_derivative_code = 0.0;
    std::uint64_t last_iterations = 0U;
    std::uint64_t accepted_zero_mode_steps = 0U;
    std::uint64_t k_element_count = 0U;
    std::uint64_t elastic_source_field_step = 0U;
    std::uint64_t elastic_last_iterations = 0U;
    double elastic_last_relative_residual = 0.0;
    bool has_elastic_state = false;
    std::uint64_t gp_bin_count = 0U;
    std::uint64_t beta_subgrid_bin_count = 0U;
    bool has_auxiliary_state = false;
    AuxPopulationState stored_aux;
    std::uint64_t expected_checksum = 0U;
    std::uint64_t checksum = 0U;

    if (std::memcmp(prefix.magic, kMagicV6, sizeof(kMagicV6)) == 0 &&
        prefix.version == kVersionV6 &&
        prefix.header_bytes == sizeof(DiskHeaderV6)) {
        DiskHeaderV6 header = {};
        ok = read_exact(fp, &header, sizeof(header));
        const bool header_has_elastic = header.elastic_state_present == 1;
        const bool header_has_auxiliary = header.aux_state_present == 1U;
        if (!ok || (header.elastic_state_present != 0 &&
                    header.elastic_state_present != 1) ||
            header.aux_state_present > 1U || header.aux_frozen > 1U ||
            elastic_state_expected(expected_provenance) !=
                header_has_elastic ||
            !selector_equal(header.zero_mode,
                            expected_provenance.zero_mode) ||
            !selector_equal(header.backend,
                            expected_provenance.backend) ||
            !selector_equal(header.composition_mode,
                            expected_provenance.composition_mode) ||
            !selector_equal(header.y_update_mode,
                            expected_provenance.y_update_mode) ||
            !selector_equal(header.explicit_context,
                            expected_provenance.explicit_context) ||
            !selector_equal(header.reaction_discretization,
                            expected_provenance.reaction_discretization) ||
            !selector_equal(header.elastic_solver_mode,
                            expected_provenance.elastic_solver_mode) ||
            !identity_equal(header.initial_state_class,
                            expected_provenance.initial_state_class) ||
            !identity_equal(header.fixture_manifest_sha256,
                            expected_provenance.fixture_manifest_sha256) ||
            !identity_equal(
                header.profile_library_manifest_sha256,
                expected_provenance.profile_library_manifest_sha256) ||
            !identity_equal(header.validation_contract_hash,
                            expected_provenance.validation_contract_hash) ||
            header.parameter_fingerprint !=
                expected_provenance.parameter_fingerprint ||
            header.elastic_solver_fingerprint !=
                expected_provenance.elastic_solver_fingerprint) {
            std::fclose(fp);
            return set_error(error,
                             "checkpoint V6 elastic/zero-mode provenance mismatch");
        }
        if (header_has_auxiliary) {
            AuxPopulationState aux;
            aux.schema_version = header.aux_schema_version;
            aux.present = true;
            aux.frozen = header.aux_frozen == 1U;
            aux.validation_contract_hash =
                expected_provenance.validation_contract_hash;
            aux.Q_B_GP_mol = header.Q_B_GP_mol;
            aux.Q_B_beta_subgrid_mol = header.Q_B_beta_subgrid_mol;
            if (!read_selector_value(header.auxiliary_state, &aux.state) ||
                !read_selector_value(header.auxiliary_units, &aux.units) ||
                !read_identity_value(header.source_handoff_hash,
                                     &aux.source_handoff_hash) ||
                !read_identity_value(header.package_handoff_hash,
                                     &aux.package_handoff_hash) ||
                !read_identity_value(header.gp_population_provenance,
                                     &aux.gp_population_provenance) ||
                !read_identity_value(
                    header.beta_subgrid_population_provenance,
                    &aux.beta_subgrid_population_provenance)) {
                std::fclose(fp);
                return set_error(error,
                                 "checkpoint V6 auxiliary metadata is invalid");
            }
            stored_aux = std::move(aux);
            gp_bin_count = header.gp_bin_count;
            beta_subgrid_bin_count = header.beta_subgrid_bin_count;
            has_auxiliary_state = true;
        } else if (header.aux_schema_version != 0U ||
                   header.aux_frozen != 1U || header.gp_bin_count != 0U ||
                   header.beta_subgrid_bin_count != 0U ||
                   header.Q_B_GP_mol != 0.0 ||
                   header.Q_B_beta_subgrid_mol != 0.0) {
            std::fclose(fp);
            return set_error(error,
                             "checkpoint V6 absent auxiliary state is not zero");
        }
        element_count = header.element_count;
        k_element_count = header.k_element_count;
        accepted_step = header.accepted_step;
        nx = header.nx;
        ny = header.ny;
        nz = header.nz;
        dt_code = header.dt_code;
        temperature_K = header.temperature_K;
        target_mass_code = header.target_mass_code;
        last_lambda = header.last_lambda;
        last_residual_code = header.last_residual_code;
        last_derivative_code = header.last_derivative_code;
        last_iterations = header.last_iterations;
        accepted_zero_mode_steps = header.accepted_zero_mode_steps;
        elastic_source_field_step = header.elastic_source_field_step;
        elastic_last_iterations = header.elastic_last_iterations;
        elastic_last_relative_residual = header.elastic_last_relative_residual;
        has_elastic_state = header_has_elastic;
        expected_checksum = header.payload_checksum;
        header.payload_checksum = 0U;
        checksum = fnv1a64(&header, sizeof(header));
    } else if (std::memcmp(prefix.magic, kMagicV5, sizeof(kMagicV5)) == 0 &&
               prefix.version == kVersionV5 &&
               prefix.header_bytes == sizeof(DiskHeaderV5)) {
        DiskHeaderV5 header = {};
        ok = read_exact(fp, &header, sizeof(header));
        const bool header_has_elastic = header.elastic_state_present == 1;
        const bool header_has_auxiliary = header.aux_state_present == 1U;
        if (!ok || (header.elastic_state_present != 0 &&
                    header.elastic_state_present != 1) ||
            header.aux_state_present > 1U || header.aux_frozen > 1U ||
            elastic_state_expected(expected_provenance) !=
                header_has_elastic ||
            !selector_equal(header.zero_mode,
                            expected_provenance.zero_mode) ||
            !selector_equal(header.backend,
                            expected_provenance.backend) ||
            !selector_equal(header.composition_mode,
                            expected_provenance.composition_mode) ||
            !selector_equal(header.y_update_mode,
                            expected_provenance.y_update_mode) ||
            !selector_equal(header.explicit_context,
                            expected_provenance.explicit_context) ||
            !selector_equal(header.reaction_discretization,
                            expected_provenance.reaction_discretization) ||
            !selector_equal(header.elastic_solver_mode,
                            expected_provenance.elastic_solver_mode) ||
            !identity_equal(header.initial_state_class,
                            expected_provenance.initial_state_class) ||
            !identity_equal(header.fixture_manifest_sha256,
                            expected_provenance.fixture_manifest_sha256) ||
            !identity_equal(
                header.profile_library_manifest_sha256,
                expected_provenance.profile_library_manifest_sha256) ||
            !identity_equal(header.validation_contract_hash,
                            expected_provenance.validation_contract_hash) ||
            header.parameter_fingerprint !=
                expected_provenance.parameter_fingerprint ||
            header.elastic_solver_fingerprint !=
                expected_provenance.elastic_solver_fingerprint) {
            std::fclose(fp);
            return set_error(error,
                             "checkpoint V5 elastic/zero-mode provenance mismatch");
        }
        if (header_has_auxiliary) {
            AuxPopulationState aux;
            aux.schema_version = header.aux_schema_version;
            aux.present = true;
            aux.frozen = header.aux_frozen == 1U;
            aux.validation_contract_hash =
                expected_provenance.validation_contract_hash;
            aux.Q_B_GP_mol = header.Q_B_GP_mol;
            aux.Q_B_beta_subgrid_mol = header.Q_B_beta_subgrid_mol;
            if (!read_selector_value(header.auxiliary_state, &aux.state) ||
                !read_selector_value(header.auxiliary_units, &aux.units) ||
                !read_identity_value(header.source_handoff_hash,
                                     &aux.source_handoff_hash) ||
                !read_identity_value(header.gp_population_provenance,
                                     &aux.gp_population_provenance) ||
                !read_identity_value(
                    header.beta_subgrid_population_provenance,
                    &aux.beta_subgrid_population_provenance)) {
                std::fclose(fp);
                return set_error(error,
                                 "checkpoint V5 auxiliary metadata is invalid");
            }
            stored_aux = std::move(aux);
            gp_bin_count = header.gp_bin_count;
            beta_subgrid_bin_count = header.beta_subgrid_bin_count;
            has_auxiliary_state = true;
        } else if (header.aux_schema_version != 0U ||
                   header.aux_frozen != 1U || header.gp_bin_count != 0U ||
                   header.beta_subgrid_bin_count != 0U ||
                   header.Q_B_GP_mol != 0.0 ||
                   header.Q_B_beta_subgrid_mol != 0.0) {
            std::fclose(fp);
            return set_error(error,
                             "checkpoint V5 absent auxiliary state is not zero");
        }
        element_count = header.element_count;
        k_element_count = header.k_element_count;
        accepted_step = header.accepted_step;
        nx = header.nx;
        ny = header.ny;
        nz = header.nz;
        dt_code = header.dt_code;
        temperature_K = header.temperature_K;
        target_mass_code = header.target_mass_code;
        last_lambda = header.last_lambda;
        last_residual_code = header.last_residual_code;
        last_derivative_code = header.last_derivative_code;
        last_iterations = header.last_iterations;
        accepted_zero_mode_steps = header.accepted_zero_mode_steps;
        elastic_source_field_step = header.elastic_source_field_step;
        elastic_last_iterations = header.elastic_last_iterations;
        elastic_last_relative_residual = header.elastic_last_relative_residual;
        has_elastic_state = header_has_elastic;
        expected_checksum = header.payload_checksum;
        header.payload_checksum = 0U;
        checksum = fnv1a64(&header, sizeof(header));
    } else if (std::memcmp(prefix.magic, kMagicV4, sizeof(kMagicV4)) == 0 &&
        prefix.version == kVersionV4 &&
        prefix.header_bytes == sizeof(DiskHeaderV4)) {
        DiskHeaderV4 header = {};
        ok = read_exact(fp, &header, sizeof(header));
        if (!ok || header.elastic_state_present != 1 ||
            expected_provenance.validation_contract_hash !=
                kLegacyUnboundValidationContractHash ||
            !elastic_state_expected(expected_provenance) ||
            !selector_equal(header.zero_mode,
                            expected_provenance.zero_mode) ||
            !selector_equal(header.backend,
                            expected_provenance.backend) ||
            !selector_equal(header.composition_mode,
                            expected_provenance.composition_mode) ||
            !selector_equal(header.y_update_mode,
                            expected_provenance.y_update_mode) ||
            !selector_equal(header.explicit_context,
                            expected_provenance.explicit_context) ||
            !selector_equal(header.reaction_discretization,
                            expected_provenance.reaction_discretization) ||
            !selector_equal(header.elastic_solver_mode,
                            expected_provenance.elastic_solver_mode) ||
            !identity_equal(header.initial_state_class,
                            expected_provenance.initial_state_class) ||
            !identity_equal(header.fixture_manifest_sha256,
                            expected_provenance.fixture_manifest_sha256) ||
            !identity_equal(
                header.profile_library_manifest_sha256,
                expected_provenance.profile_library_manifest_sha256) ||
            header.parameter_fingerprint !=
                expected_provenance.parameter_fingerprint ||
            header.elastic_solver_fingerprint !=
                expected_provenance.elastic_solver_fingerprint) {
            std::fclose(fp);
            return set_error(
                error,
                "checkpoint V4 elastic/zero-mode provenance mismatch");
        }
        element_count = header.element_count;
        k_element_count = header.k_element_count;
        accepted_step = header.accepted_step;
        nx = header.nx;
        ny = header.ny;
        nz = header.nz;
        dt_code = header.dt_code;
        temperature_K = header.temperature_K;
        target_mass_code = header.target_mass_code;
        last_lambda = header.last_lambda;
        last_residual_code = header.last_residual_code;
        last_derivative_code = header.last_derivative_code;
        last_iterations = header.last_iterations;
        accepted_zero_mode_steps =
            header.accepted_zero_mode_steps;
        elastic_source_field_step =
            header.elastic_source_field_step;
        elastic_last_iterations =
            header.elastic_last_iterations;
        elastic_last_relative_residual =
            header.elastic_last_relative_residual;
        has_elastic_state = true;
        expected_checksum = header.payload_checksum;
        header.payload_checksum = 0U;
        checksum = fnv1a64(&header, sizeof(header));
    } else if (std::memcmp(prefix.magic, kMagicV3, sizeof(kMagicV3)) == 0 &&
        prefix.version == kVersionV3 &&
        prefix.header_bytes == sizeof(DiskHeaderV3)) {
        if (elastic_state_expected(expected_provenance) ||
            expected_provenance.validation_contract_hash !=
                kLegacyUnboundValidationContractHash) {
            std::fclose(fp);
            return set_error(
                error,
                "checkpoint V3 lacks required elastic runtime or contract state");
        }
        DiskHeaderV3 header = {};
        ok = read_exact(fp, &header, sizeof(header));
        if (!ok ||
            !selector_equal(header.zero_mode, expected_provenance.zero_mode) ||
            !selector_equal(header.backend, expected_provenance.backend) ||
            !selector_equal(header.composition_mode,
                            expected_provenance.composition_mode) ||
            !selector_equal(header.y_update_mode,
                            expected_provenance.y_update_mode) ||
            !selector_equal(header.explicit_context,
                            expected_provenance.explicit_context) ||
            !selector_equal(header.reaction_discretization,
                            expected_provenance.reaction_discretization) ||
            !identity_equal(header.initial_state_class,
                            expected_provenance.initial_state_class) ||
            !identity_equal(header.fixture_manifest_sha256,
                            expected_provenance.fixture_manifest_sha256) ||
            !identity_equal(
                header.profile_library_manifest_sha256,
                expected_provenance.profile_library_manifest_sha256) ||
            header.parameter_fingerprint !=
                expected_provenance.parameter_fingerprint) {
            std::fclose(fp);
            return set_error(error, "checkpoint provenance mismatch");
        }
        element_count = header.element_count;
        accepted_step = header.accepted_step;
        nx = header.nx;
        ny = header.ny;
        nz = header.nz;
        dt_code = header.dt_code;
        temperature_K = header.temperature_K;
        target_mass_code = header.target_mass_code;
        last_lambda = header.last_lambda;
        last_residual_code = header.last_residual_code;
        last_derivative_code = header.last_derivative_code;
        last_iterations = header.last_iterations;
        accepted_zero_mode_steps = header.accepted_zero_mode_steps;
        expected_checksum = header.payload_checksum;
        header.payload_checksum = 0U;
        checksum = fnv1a64(&header, sizeof(header));
    } else if (std::memcmp(prefix.magic, kMagicV2, sizeof(kMagicV2)) == 0 &&
               prefix.version == kVersionV2 &&
               prefix.header_bytes == sizeof(DiskHeaderV2)) {
        if (!legacy_identity_expected(expected_provenance) ||
            expected_provenance.validation_contract_hash !=
                kLegacyUnboundValidationContractHash ||
            elastic_state_expected(expected_provenance)) {
            std::fclose(fp);
            return set_error(
                error,
                "legacy checkpoint lacks required fixture/library/elastic/contract provenance");
        }
        DiskHeaderV2 header = {};
        ok = read_exact(fp, &header, sizeof(header));
        if (!ok ||
            !selector_equal(header.zero_mode, expected_provenance.zero_mode) ||
            !selector_equal(header.backend, expected_provenance.backend) ||
            !selector_equal(header.composition_mode,
                            expected_provenance.composition_mode) ||
            !selector_equal(header.y_update_mode,
                            expected_provenance.y_update_mode) ||
            !selector_equal(header.explicit_context,
                            expected_provenance.explicit_context) ||
            !selector_equal(header.reaction_discretization,
                            expected_provenance.reaction_discretization) ||
            header.parameter_fingerprint !=
                expected_provenance.parameter_fingerprint) {
            std::fclose(fp);
            return set_error(error, "checkpoint zero-mode provenance mismatch");
        }
        element_count = header.element_count;
        accepted_step = header.accepted_step;
        nx = header.nx;
        ny = header.ny;
        nz = header.nz;
        dt_code = header.dt_code;
        temperature_K = header.temperature_K;
        target_mass_code = header.target_mass_code;
        last_lambda = header.last_lambda;
        last_residual_code = header.last_residual_code;
        last_derivative_code = header.last_derivative_code;
        last_iterations = header.last_iterations;
        accepted_zero_mode_steps = header.accepted_zero_mode_steps;
        expected_checksum = header.payload_checksum;
        header.payload_checksum = 0U;
        checksum = fnv1a64(&header, sizeof(header));
    } else {
        std::fclose(fp);
        return set_error(error, "checkpoint header/version mismatch");
    }

    if (element_count == 0U ||
        element_count >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max() / sizeof(double))) {
        std::fclose(fp);
        return set_error(error, "checkpoint element count is invalid");
    }
    if (has_elastic_state) {
        const std::uint64_t expected_k_count =
            static_cast<std::uint64_t>(nx) *
            static_cast<std::uint64_t>(ny) *
            static_cast<std::uint64_t>(nz / 2 + 1);
        if (k_element_count == 0U ||
            k_element_count != expected_k_count ||
            k_element_count >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max() /
                    (6U * sizeof(float)))) {
            std::fclose(fp);
            return set_error(
                error,
                "checkpoint elastic k-space element count is invalid");
        }
    }
    if (has_auxiliary_state &&
        (gp_bin_count > static_cast<std::uint64_t>(
                             std::numeric_limits<std::size_t>::max() /
                             sizeof(DiskPopulationBinV1)) ||
         beta_subgrid_bin_count > static_cast<std::uint64_t>(
                                      std::numeric_limits<std::size_t>::max() /
                                      sizeof(DiskPopulationBinV1)))) {
        std::fclose(fp);
        return set_error(error, "checkpoint auxiliary bin count is invalid");
    }

    Checkpoint loaded;
    loaded.accepted_step = accepted_step;
    loaded.nx = nx;
    loaded.ny = ny;
    loaded.nz = nz;
    loaded.dt_code = dt_code;
    loaded.temperature_K = temperature_K;
    loaded.provenance = expected_provenance;
    loaded.zero_mode.target_mass_code = target_mass_code;
    loaded.zero_mode.last_lambda = last_lambda;
    loaded.zero_mode.last_residual_code = last_residual_code;
    loaded.zero_mode.last_derivative_code = last_derivative_code;
    loaded.zero_mode.last_iterations = last_iterations;
    loaded.zero_mode.accepted_zero_mode_steps =
        accepted_zero_mode_steps;
    loaded.elastic.present = has_elastic_state;
    loaded.elastic.source_field_step =
        elastic_source_field_step;
    loaded.elastic.last_iterations =
        elastic_last_iterations;
    loaded.elastic.last_relative_residual =
        elastic_last_relative_residual;
    loaded.aux = stored_aux;
    loaded.phi.resize(element_count);
    loaded.Y.resize(element_count);
    loaded.xB.resize(element_count);
    loaded.dY_dt_prev.resize(element_count);
    std::vector<double>* fields[] = {
        &loaded.phi, &loaded.Y, &loaded.xB, &loaded.dY_dt_prev};
    for (std::vector<double>* field : fields) {
        const std::size_t bytes = field->size() * sizeof(double);
        if (ok) ok = read_exact(fp, field->data(), bytes);
        if (ok) checksum = fnv1a64(field->data(), bytes, checksum);
    }
    if (has_elastic_state) {
        loaded.elastic.displacement_k.resize(
            static_cast<std::size_t>(6U * k_element_count));
        const std::size_t bytes =
            loaded.elastic.displacement_k.size() * sizeof(float);
        if (ok) {
            ok = read_exact(
                fp, loaded.elastic.displacement_k.data(), bytes);
        }
        if (ok) {
            checksum = fnv1a64(
                loaded.elastic.displacement_k.data(), bytes, checksum);
        }
    }
    if (has_auxiliary_state) {
        std::vector<AuxiliaryPopulationBin>* populations[] = {
            &loaded.aux.gp_bins, &loaded.aux.beta_subgrid_bins};
        const std::uint64_t bin_counts[] = {
            gp_bin_count, beta_subgrid_bin_count};
        for (std::size_t population_index = 0U;
             population_index < 2U; ++population_index) {
            std::vector<AuxiliaryPopulationBin>* population =
                populations[population_index];
            population->resize(
                static_cast<std::size_t>(bin_counts[population_index]));
            for (std::size_t bin_index = 0U;
                 bin_index < population->size(); ++bin_index) {
                DiskPopulationBinV1 disk = {};
                if (ok) ok = read_exact(fp, &disk, sizeof(disk));
                if (ok) checksum = fnv1a64(&disk, sizeof(disk), checksum);
                (*population)[bin_index] = from_disk_bin(disk);
            }
        }
    }
    unsigned char trailing = 0U;
    if (ok && std::fread(&trailing, 1U, 1U, fp) != 0U) ok = false;
    std::fclose(fp);
    if (!ok || checksum != expected_checksum) {
        return set_error(error, "checkpoint payload checksum mismatch");
    }
    if (!validate_checkpoint(loaded, error)) return false;
    *checkpoint = std::move(loaded);
    return true;
}

}  // namespace pf_zero_mode
