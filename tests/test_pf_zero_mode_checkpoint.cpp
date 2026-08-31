#include "../pf_zero_mode_checkpoint.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

namespace {

bool exact_equal(const pf_zero_mode::AuxiliaryPopulationBin& left,
                 const pf_zero_mode::AuxiliaryPopulationBin& right) {
    return left.radius_lower_m == right.radius_lower_m &&
           left.radius_upper_m == right.radius_upper_m &&
           left.number_density_m3 == right.number_density_m3 &&
           left.count == right.count && left.xB == right.xB &&
           left.molar_volume_m3_mol == right.molar_volume_m3_mol &&
           left.inventory_mol == right.inventory_mol;
}

bool exact_equal(const std::vector<pf_zero_mode::AuxiliaryPopulationBin>& left,
                 const std::vector<pf_zero_mode::AuxiliaryPopulationBin>& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0U; index < left.size(); ++index) {
        if (!exact_equal(left[index], right[index])) return false;
    }
    return true;
}

bool exact_equal(const pf_zero_mode::AuxPopulationState& left,
                 const pf_zero_mode::AuxPopulationState& right) {
    return left.schema_version == right.schema_version &&
           left.present == right.present && left.frozen == right.frozen &&
           left.state == right.state &&
           left.validation_contract_hash == right.validation_contract_hash &&
           left.source_handoff_hash == right.source_handoff_hash &&
           left.units == right.units && left.Q_B_GP_mol == right.Q_B_GP_mol &&
           left.Q_B_beta_subgrid_mol == right.Q_B_beta_subgrid_mol &&
           left.gp_population_provenance == right.gp_population_provenance &&
           left.beta_subgrid_population_provenance ==
               right.beta_subgrid_population_provenance &&
           exact_equal(left.gp_bins, right.gp_bins) &&
           exact_equal(left.beta_subgrid_bins, right.beta_subgrid_bins);
}

bool exact_equal(const pf_zero_mode::Checkpoint& left,
                 const pf_zero_mode::Checkpoint& right) {
    return left.accepted_step == right.accepted_step &&
           left.nx == right.nx && left.ny == right.ny &&
           left.nz == right.nz && left.dt_code == right.dt_code &&
           left.temperature_K == right.temperature_K &&
           left.provenance.zero_mode == right.provenance.zero_mode &&
           left.provenance.backend == right.provenance.backend &&
           left.provenance.composition_mode ==
               right.provenance.composition_mode &&
           left.provenance.y_update_mode ==
               right.provenance.y_update_mode &&
           left.provenance.explicit_context ==
               right.provenance.explicit_context &&
           left.provenance.reaction_discretization ==
               right.provenance.reaction_discretization &&
           left.provenance.initial_state_class ==
               right.provenance.initial_state_class &&
           left.provenance.fixture_manifest_sha256 ==
               right.provenance.fixture_manifest_sha256 &&
           left.provenance.profile_library_manifest_sha256 ==
               right.provenance.profile_library_manifest_sha256 &&
           left.provenance.validation_contract_hash ==
               right.provenance.validation_contract_hash &&
           left.provenance.elastic_solver_mode ==
               right.provenance.elastic_solver_mode &&
           left.provenance.elastic_solver_fingerprint ==
               right.provenance.elastic_solver_fingerprint &&
           left.provenance.parameter_fingerprint ==
               right.provenance.parameter_fingerprint &&
           left.zero_mode.target_mass_code ==
               right.zero_mode.target_mass_code &&
           left.zero_mode.last_lambda == right.zero_mode.last_lambda &&
           left.zero_mode.last_residual_code ==
               right.zero_mode.last_residual_code &&
           left.zero_mode.last_derivative_code ==
               right.zero_mode.last_derivative_code &&
           left.zero_mode.last_iterations ==
               right.zero_mode.last_iterations &&
           left.zero_mode.accepted_zero_mode_steps ==
               right.zero_mode.accepted_zero_mode_steps &&
           left.elastic.present == right.elastic.present &&
           left.elastic.source_field_step ==
               right.elastic.source_field_step &&
           left.elastic.last_iterations ==
               right.elastic.last_iterations &&
           left.elastic.last_relative_residual ==
               right.elastic.last_relative_residual &&
           left.elastic.displacement_k ==
               right.elastic.displacement_k &&
           exact_equal(left.aux, right.aux) &&
           left.phi == right.phi && left.Y == right.Y &&
           left.xB == right.xB &&
           left.dY_dt_prev == right.dY_dt_prev;
}

constexpr std::size_t kSelectorBytes = 64U;
constexpr std::size_t kIdentityBytes = 96U;

struct LegacyDiskHeaderV2 {
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

struct LegacyDiskHeaderV3 {
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

struct LegacyDiskHeaderV4 {
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

bool copy_text(char* destination, std::size_t bytes,
               const std::string& source) {
    if (source.empty() || source.size() >= bytes) return false;
    std::memset(destination, 0, bytes);
    std::memcpy(destination, source.data(), source.size());
    return true;
}

template <typename Header>
bool fill_legacy_common(Header* header,
                        const pf_zero_mode::Checkpoint& source) {
    header->element_count = source.phi.size();
    header->accepted_step = source.accepted_step;
    header->nx = source.nx;
    header->ny = source.ny;
    header->nz = source.nz;
    header->dt_code = source.dt_code;
    header->temperature_K = source.temperature_K;
    header->target_mass_code = source.zero_mode.target_mass_code;
    header->last_lambda = source.zero_mode.last_lambda;
    header->last_residual_code = source.zero_mode.last_residual_code;
    header->last_derivative_code = source.zero_mode.last_derivative_code;
    header->last_iterations = source.zero_mode.last_iterations;
    header->accepted_zero_mode_steps =
        source.zero_mode.accepted_zero_mode_steps;
    header->parameter_fingerprint = source.provenance.parameter_fingerprint;
    return copy_text(header->zero_mode, kSelectorBytes,
                     source.provenance.zero_mode) &&
           copy_text(header->backend, kSelectorBytes,
                     source.provenance.backend) &&
           copy_text(header->composition_mode, kSelectorBytes,
                     source.provenance.composition_mode) &&
           copy_text(header->y_update_mode, kSelectorBytes,
                     source.provenance.y_update_mode) &&
           copy_text(header->explicit_context, kSelectorBytes,
                     source.provenance.explicit_context) &&
           copy_text(header->reaction_discretization, kSelectorBytes,
                     source.provenance.reaction_discretization);
}

template <typename Header>
bool write_legacy_payload(const std::string& path, Header header,
                          const pf_zero_mode::Checkpoint& source,
                          bool write_elastic) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) return false;
    std::uint64_t checksum = pf_zero_mode::fnv1a64(&header, sizeof(header));
    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
    const std::vector<double>* fields[] = {
        &source.phi, &source.Y, &source.xB, &source.dY_dt_prev};
    for (const std::vector<double>* field : fields) {
        const std::size_t bytes = field->size() * sizeof(double);
        stream.write(reinterpret_cast<const char*>(field->data()), bytes);
        checksum = pf_zero_mode::fnv1a64(field->data(), bytes, checksum);
    }
    if (write_elastic) {
        const std::size_t bytes =
            source.elastic.displacement_k.size() * sizeof(float);
        stream.write(reinterpret_cast<const char*>(
                         source.elastic.displacement_k.data()),
                     bytes);
        checksum = pf_zero_mode::fnv1a64(
            source.elastic.displacement_k.data(), bytes, checksum);
    }
    header.payload_checksum = checksum;
    stream.seekp(0);
    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
    stream.close();
    return static_cast<bool>(stream);
}

bool write_legacy_v2(const std::string& path,
                     const pf_zero_mode::Checkpoint& source) {
    LegacyDiskHeaderV2 header = {};
    std::memcpy(header.magic, "PFZMCHK2", sizeof(header.magic));
    header.version = 2U;
    header.header_bytes = sizeof(header);
    return fill_legacy_common(&header, source) &&
           write_legacy_payload(path, header, source, false);
}

bool write_legacy_v3(const std::string& path,
                     const pf_zero_mode::Checkpoint& source) {
    LegacyDiskHeaderV3 header = {};
    std::memcpy(header.magic, "PFZMCHK3", sizeof(header.magic));
    header.version = 3U;
    header.header_bytes = sizeof(header);
    if (!fill_legacy_common(&header, source) ||
        !copy_text(header.initial_state_class, kIdentityBytes,
                   source.provenance.initial_state_class) ||
        !copy_text(header.fixture_manifest_sha256, kIdentityBytes,
                   source.provenance.fixture_manifest_sha256) ||
        !copy_text(header.profile_library_manifest_sha256, kIdentityBytes,
                   source.provenance.profile_library_manifest_sha256)) {
        return false;
    }
    return write_legacy_payload(path, header, source, false);
}

bool write_legacy_v4(const std::string& path,
                     const pf_zero_mode::Checkpoint& source) {
    LegacyDiskHeaderV4 header = {};
    std::memcpy(header.magic, "PFZMCHK4", sizeof(header.magic));
    header.version = 4U;
    header.header_bytes = sizeof(header);
    if (!fill_legacy_common(&header, source) || !source.elastic.present ||
        !copy_text(header.elastic_solver_mode, kSelectorBytes,
                   source.provenance.elastic_solver_mode) ||
        !copy_text(header.initial_state_class, kIdentityBytes,
                   source.provenance.initial_state_class) ||
        !copy_text(header.fixture_manifest_sha256, kIdentityBytes,
                   source.provenance.fixture_manifest_sha256) ||
        !copy_text(header.profile_library_manifest_sha256, kIdentityBytes,
                   source.provenance.profile_library_manifest_sha256)) {
        return false;
    }
    header.k_element_count = source.elastic.displacement_k.size() / 6U;
    header.elastic_state_present = 1;
    header.elastic_solver_fingerprint =
        source.provenance.elastic_solver_fingerprint;
    header.elastic_source_field_step = source.elastic.source_field_step;
    header.elastic_last_iterations = source.elastic.last_iterations;
    header.elastic_last_relative_residual =
        source.elastic.last_relative_residual;
    return write_legacy_payload(path, header, source, true);
}

bool is_legacy_zero_aux(const pf_zero_mode::AuxPopulationState& aux) {
    return !aux.present && aux.schema_version == 0U && aux.frozen &&
           aux.state == pf_zero_mode::kLegacyZeroAux &&
           aux.validation_contract_hash.empty() &&
           aux.source_handoff_hash.empty() &&
           aux.Q_B_GP_mol == 0.0 && aux.Q_B_beta_subgrid_mol == 0.0 &&
           aux.gp_bins.empty() && aux.beta_subgrid_bins.empty();
}

}  // namespace

int main() {
    const std::string path = "/tmp/pf_zero_mode_checkpoint_test.chk";
    const std::string elastic_path =
        "/tmp/pf_zero_mode_checkpoint_elastic_test.chk";
    const std::string auxiliary_path =
        "/tmp/pf_zero_mode_checkpoint_auxiliary_test.chk";
    const std::string legacy_v2_path =
        "/tmp/pf_zero_mode_checkpoint_legacy_v2_test.chk";
    const std::string legacy_v3_path =
        "/tmp/pf_zero_mode_checkpoint_legacy_v3_test.chk";
    const std::string legacy_v4_path =
        "/tmp/pf_zero_mode_checkpoint_legacy_v4_test.chk";
    std::remove(path.c_str());
    std::remove((path + ".tmp").c_str());
    std::remove(elastic_path.c_str());
    std::remove((elastic_path + ".tmp").c_str());
    std::remove(auxiliary_path.c_str());
    std::remove((auxiliary_path + ".tmp").c_str());
    std::remove(legacy_v2_path.c_str());
    std::remove(legacy_v3_path.c_str());
    std::remove(legacy_v4_path.c_str());

    pf_zero_mode::Checkpoint source;
    source.accepted_step = 17U;
    source.nx = 2;
    source.ny = 2;
    source.nz = 2;
    source.dt_code = 0.02;
    source.temperature_K = 673.15;
    source.provenance.zero_mode = pf_zero_mode::kModeV1;
    source.provenance.backend = pf_zero_mode::kHostBackendV1;
    source.provenance.composition_mode = "legacy";
    source.provenance.y_update_mode = "lagged_rhs";
    source.provenance.explicit_context =
        pf_zero_mode::kExplicitContextNV1;
    source.provenance.reaction_discretization =
        pf_zero_mode::kReactionTangentNV1;
    source.provenance.initial_state_class =
        pf_zero_mode::kConditionalHandoffV1;
    source.provenance.fixture_manifest_sha256 =
        "3df27d489abeb808f3e72e75d4910ee53a7f490d702af9f1ed17f5d20d03f82b";
    source.provenance.profile_library_manifest_sha256 =
        "58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe";
    source.provenance.validation_contract_hash =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    source.provenance.parameter_fingerprint = 0x123456789abcdef0ULL;
    source.zero_mode.target_mass_code = 0.24;
    source.zero_mode.last_lambda = -1.25e-9;
    source.zero_mode.last_residual_code = 3.0e-14;
    source.zero_mode.last_derivative_code = 0.18;
    source.zero_mode.last_iterations = 2U;
    source.zero_mode.accepted_zero_mode_steps = 17U;
    for (int i = 0; i < 8; ++i) {
        source.phi.push_back(0.01 * i);
        source.Y.push_back(-4.0 + 0.1 * i);
        source.xB.push_back(0.02 + 0.001 * i);
        source.dY_dt_prev.push_back(1.0e-6 * i);
    }

    std::string error;
    if (!pf_zero_mode::write_checkpoint(path, source, &error)) {
        std::cerr << error << "\n";
        return 1;
    }
    pf_zero_mode::Checkpoint loaded;
    if (!pf_zero_mode::read_checkpoint(
            path, source.provenance, &loaded, &error) ||
        !exact_equal(source, loaded)) {
        std::cerr << "round-trip failed: " << error << "\n";
        return 1;
    }

    pf_zero_mode::Provenance mismatch = source.provenance;
    mismatch.backend = "UNQUALIFIED_BACKEND";
    if (pf_zero_mode::read_checkpoint(path, mismatch, &loaded, &error)) {
        std::cerr << "provenance mismatch was not rejected\n";
        return 1;
    }
    mismatch = source.provenance;
    mismatch.explicit_context = "LEGACY_MIXED_PHI_NP1_V1";
    if (pf_zero_mode::read_checkpoint(path, mismatch, &loaded, &error)) {
        std::cerr << "time-level mismatch was not rejected\n";
        return 1;
    }
    mismatch = source.provenance;
    mismatch.reaction_discretization = "FINITE_DELTA_H_REACTION_V1";
    if (pf_zero_mode::read_checkpoint(path, mismatch, &loaded, &error)) {
        std::cerr << "reaction-discretization mismatch was not rejected\n";
        return 1;
    }
    mismatch = source.provenance;
    mismatch.fixture_manifest_sha256 =
        "0df27d489abeb808f3e72e75d4910ee53a7f490d702af9f1ed17f5d20d03f82b";
    if (pf_zero_mode::read_checkpoint(path, mismatch, &loaded, &error)) {
        std::cerr << "fixture provenance mismatch was not rejected\n";
        return 1;
    }
    mismatch = source.provenance;
    mismatch.profile_library_manifest_sha256 =
        "08803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe";
    if (pf_zero_mode::read_checkpoint(path, mismatch, &loaded, &error)) {
        std::cerr << "library provenance mismatch was not rejected\n";
        return 1;
    }
    mismatch = source.provenance;
    mismatch.validation_contract_hash =
        "f123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    if (pf_zero_mode::read_checkpoint(path, mismatch, &loaded, &error)) {
        std::cerr << "validation-contract mismatch was not rejected\n";
        return 1;
    }
    mismatch = source.provenance;
    mismatch.initial_state_class = pf_zero_mode::kLegacyInitialStateClass;
    if (pf_zero_mode::read_checkpoint(path, mismatch, &loaded, &error)) {
        std::cerr << "initial-state class mismatch was not rejected\n";
        return 1;
    }

    pf_zero_mode::Checkpoint elastic_source = source;
    elastic_source.provenance.elastic_solver_mode =
        pf_zero_mode::kElasticWarmStartResidualV1;
    elastic_source.provenance.elastic_solver_fingerprint =
        0xfedcba9876543210ULL;
    elastic_source.elastic.present = true;
    elastic_source.elastic.source_field_step =
        elastic_source.accepted_step - 1U;
    elastic_source.elastic.last_iterations = 4U;
    elastic_source.elastic.last_relative_residual = 7.5e-7;
    // 2*2*(2/2+1)=8 complex values per component, packed as six
    // real/imag float lanes across ux,uy,uz.
    elastic_source.elastic.displacement_k.resize(6U * 8U);
    for (std::size_t i = 0;
         i < elastic_source.elastic.displacement_k.size(); ++i) {
        elastic_source.elastic.displacement_k[i] =
            static_cast<float>(0.001 * static_cast<double>(i));
    }
    if (!pf_zero_mode::write_checkpoint(
            elastic_path, elastic_source, &error)) {
        std::cerr << "elastic V5 write failed: " << error << "\n";
        return 1;
    }
    {
        std::ifstream stream(elastic_path, std::ios::binary);
        char magic[8] = {};
        std::uint32_t version = 0U;
        std::uint32_t header_bytes = 0U;
        stream.read(magic, sizeof(magic));
        stream.read(
            reinterpret_cast<char*>(&version), sizeof(version));
        stream.read(
            reinterpret_cast<char*>(&header_bytes),
            sizeof(header_bytes));
        if (!stream ||
            std::string(magic, sizeof(magic)) != "PFZMCHK5" ||
            version != 5U || header_bytes == 0U) {
            std::cerr << "elastic V5 disk prefix mismatch\n";
            return 1;
        }
    }
    pf_zero_mode::Checkpoint elastic_loaded;
    if (!pf_zero_mode::read_checkpoint(
            elastic_path, elastic_source.provenance,
            &elastic_loaded, &error) ||
        !exact_equal(elastic_source, elastic_loaded)) {
        std::cerr << "elastic V5 round-trip failed: " << error << "\n";
        return 1;
    }
    mismatch = elastic_source.provenance;
    mismatch.elastic_solver_fingerprint ^= 0x1ULL;
    if (pf_zero_mode::read_checkpoint(
            elastic_path, mismatch, &elastic_loaded, &error)) {
        std::cerr << "elastic solver provenance mismatch was not rejected\n";
        return 1;
    }
    mismatch = source.provenance;
    if (pf_zero_mode::read_checkpoint(
            elastic_path, mismatch, &elastic_loaded, &error)) {
        std::cerr << "elastic V5 state was accepted by a legacy solver\n";
        return 1;
    }

    pf_zero_mode::Checkpoint auxiliary_source = source;
    auxiliary_source.aux.present = true;
    auxiliary_source.aux.schema_version =
        pf_zero_mode::kAuxiliaryPopulationSchemaV1;
    auxiliary_source.aux.frozen = true;
    auxiliary_source.aux.state =
        pf_zero_mode::kAuxiliaryStateStorageOnlyV1;
    auxiliary_source.aux.validation_contract_hash =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    auxiliary_source.aux.source_handoff_hash =
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    auxiliary_source.aux.gp_population_provenance =
        "PRESCRIBED_SOURCE_KWN_GP_V1";
    auxiliary_source.aux.beta_subgrid_population_provenance =
        "PRESCRIBED_SOURCE_KWN_SUBGRID_BETA_V1";
    pf_zero_mode::AuxiliaryPopulationBin gp_low;
    gp_low.radius_lower_m = 1.0e-9;
    gp_low.radius_upper_m = 2.0e-9;
    gp_low.number_density_m3 = 2.5e22;
    gp_low.count = 7U;
    gp_low.xB = 0.12;
    gp_low.molar_volume_m3_mol = 4.1009e-5;
    gp_low.inventory_mol = 2.0e-18;
    pf_zero_mode::AuxiliaryPopulationBin gp_high = gp_low;
    gp_high.radius_lower_m = 2.0e-9;
    gp_high.radius_upper_m = 4.0e-9;
    gp_high.count = 3U;
    gp_high.inventory_mol = 3.0e-18;
    pf_zero_mode::AuxiliaryPopulationBin beta_subgrid = gp_high;
    beta_subgrid.radius_lower_m = 4.0e-9;
    beta_subgrid.radius_upper_m = 6.0e-9;
    beta_subgrid.count = 2U;
    beta_subgrid.xB = 1.0;
    beta_subgrid.inventory_mol = 5.0e-18;
    auxiliary_source.aux.gp_bins.push_back(gp_low);
    auxiliary_source.aux.gp_bins.push_back(gp_high);
    auxiliary_source.aux.beta_subgrid_bins.push_back(beta_subgrid);
    auxiliary_source.aux.Q_B_GP_mol =
        pf_zero_mode::auxiliary_population_inventory_mol(
            auxiliary_source.aux.gp_bins);
    auxiliary_source.aux.Q_B_beta_subgrid_mol =
        pf_zero_mode::auxiliary_population_inventory_mol(
            auxiliary_source.aux.beta_subgrid_bins);
    if (std::fabs(pf_zero_mode::auxiliary_total_inventory_mol(
                      auxiliary_source.aux) -
                  1.0e-17) >
        1.0e-30) {
        std::cerr << "auxiliary ledger helper returned the wrong total\n";
        return 1;
    }
    pf_zero_mode::Checkpoint invalid_auxiliary = auxiliary_source;
    invalid_auxiliary.aux.Q_B_GP_mol = 1.0e-18;
    if (pf_zero_mode::validate_checkpoint(invalid_auxiliary, &error)) {
        std::cerr << "inconsistent auxiliary inventory was not rejected\n";
        return 1;
    }
    if (!pf_zero_mode::write_checkpoint(
            auxiliary_path, auxiliary_source, &error)) {
        std::cerr << "nonzero auxiliary V5 write failed: " << error << "\n";
        return 1;
    }
    pf_zero_mode::Checkpoint auxiliary_loaded;
    if (!pf_zero_mode::read_checkpoint(
            auxiliary_path, auxiliary_source.provenance, &auxiliary_loaded,
            &error) ||
        !exact_equal(auxiliary_source, auxiliary_loaded)) {
        std::cerr << "nonzero auxiliary V5 round-trip failed: " << error
                  << "\n";
        return 1;
    }

    pf_zero_mode::Checkpoint legacy_v2_source = source;
    legacy_v2_source.provenance.initial_state_class =
        pf_zero_mode::kLegacyInitialStateClass;
    legacy_v2_source.provenance.fixture_manifest_sha256 =
        pf_zero_mode::kLegacyInitialStateClass;
    legacy_v2_source.provenance.profile_library_manifest_sha256 =
        pf_zero_mode::kLegacyInitialStateClass;
    legacy_v2_source.provenance.validation_contract_hash =
        pf_zero_mode::kLegacyUnboundValidationContractHash;
    if (pf_zero_mode::write_checkpoint(
            "/tmp/pf_zero_mode_checkpoint_legacy_relabel.chk",
            legacy_v2_source, &error)) {
        std::cerr << "legacy-unbound state was incorrectly reissued as V5\n";
        return 1;
    }
    if (!write_legacy_v2(legacy_v2_path, legacy_v2_source) ||
        !pf_zero_mode::read_checkpoint(
            legacy_v2_path, legacy_v2_source.provenance, &loaded, &error) ||
        !exact_equal(legacy_v2_source, loaded) || !is_legacy_zero_aux(loaded.aux)) {
        std::cerr << "legacy V2 zero-aux recovery failed: " << error << "\n";
        return 1;
    }
    pf_zero_mode::Checkpoint legacy_v3_source = source;
    legacy_v3_source.provenance.validation_contract_hash =
        pf_zero_mode::kLegacyUnboundValidationContractHash;
    if (!write_legacy_v3(legacy_v3_path, legacy_v3_source) ||
        !pf_zero_mode::read_checkpoint(
            legacy_v3_path, legacy_v3_source.provenance, &loaded, &error) ||
        !exact_equal(legacy_v3_source, loaded) ||
        !is_legacy_zero_aux(loaded.aux)) {
        std::cerr << "legacy V3 zero-aux recovery failed: " << error << "\n";
        return 1;
    }
    if (pf_zero_mode::read_checkpoint(
            legacy_v3_path, source.provenance, &loaded, &error)) {
        std::cerr << "legacy V3 was accepted as a current contract\n";
        return 1;
    }
    pf_zero_mode::Checkpoint legacy_v4_source = elastic_source;
    legacy_v4_source.provenance.validation_contract_hash =
        pf_zero_mode::kLegacyUnboundValidationContractHash;
    if (!write_legacy_v4(legacy_v4_path, legacy_v4_source) ||
        !pf_zero_mode::read_checkpoint(
            legacy_v4_path, legacy_v4_source.provenance, &elastic_loaded,
            &error) ||
        !exact_equal(legacy_v4_source, elastic_loaded) ||
        !is_legacy_zero_aux(elastic_loaded.aux)) {
        std::cerr << "legacy V4 zero-aux recovery failed: " << error << "\n";
        return 1;
    }

    {
        std::fstream stream(path, std::ios::in | std::ios::out |
                                     std::ios::binary);
        stream.seekp(-1, std::ios::end);
        char byte = 0;
        stream.read(&byte, 1);
        stream.clear();
        stream.seekp(-1, std::ios::end);
        byte ^= 0x1;
        stream.write(&byte, 1);
    }
    if (pf_zero_mode::read_checkpoint(
            path, source.provenance, &loaded, &error)) {
        std::cerr << "corrupted checkpoint was not rejected\n";
        return 1;
    }
    std::remove(path.c_str());
    std::remove(elastic_path.c_str());
    std::remove(auxiliary_path.c_str());
    std::remove(legacy_v2_path.c_str());
    std::remove(legacy_v3_path.c_str());
    std::remove(legacy_v4_path.c_str());
    std::cout << "PASS_PF_ZERO_MODE_CHECKPOINT_PROVENANCE_V2_TO_V5_AUX\n";
    return 0;
}
