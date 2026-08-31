#include "pf_auxiliary_handoff_v2.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace pf_auxiliary_handoff_v2 {
namespace {

constexpr const char* kAuxiliaryState =
    "AUXILIARY_POPULATION_STORAGE_ONLY_V1";
constexpr const char* kAuxiliaryUnits = "mol_B";
constexpr const char* kDynamicsExcluded =
    "STORAGE_ONLY_NO_PF_FIELD_MUTATION";
constexpr const char* kPopulationProvenance =
    "PRESCRIBED_NON_PREDICTIVE_STORAGE_CONTROL|"
    "compact_host_population_state_v2";
constexpr const char* kPopulationDisclaimer =
    "FIXTURE_CONDITIONED_PRESCRIBED_SOURCE_IS_NOT_A_GP_NUCLEATION_PREDICTION";
constexpr double kLedgerRelativeTolerance = 1.0e-10;

struct ContinuumBin {
    double radius_lower_m = 0.0;
    double radius_upper_m = 0.0;
    double number_density_per_m4 = 0.0;
    double inventory_mol = 0.0;
};

bool set_error(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

bool is_lower_hex_sha256(const std::string& value) {
    if (value.size() != 64U) return false;
    return value.find_first_not_of("0123456789abcdef") == std::string::npos;
}

bool parse_finite_double(const std::string& text, double* value) {
    if (text.empty() || !value) return false;
    errno = 0;
    char* end = nullptr;
    const double parsed = std::strtod(text.c_str(), &end);
    if (errno == ERANGE || end == text.c_str() || end == nullptr ||
        *end != '\0' || !std::isfinite(parsed)) {
        return false;
    }
    *value = parsed;
    return true;
}

bool parse_count(const std::string& text, std::uint64_t* value) {
    if (text.empty() || !value) return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || end == nullptr ||
        *end != '\0') {
        return false;
    }
    *value = static_cast<std::uint64_t>(parsed);
    return true;
}

bool parse_bin(const std::string& text, ContinuumBin* bin) {
    if (!bin) return false;
    std::vector<std::string> fields;
    std::size_t begin = 0U;
    while (true) {
        const std::size_t comma = text.find(',', begin);
        fields.push_back(text.substr(begin, comma - begin));
        if (comma == std::string::npos) break;
        begin = comma + 1U;
    }
    if (fields.size() != 4U) return false;
    return parse_finite_double(fields[0], &bin->radius_lower_m) &&
           parse_finite_double(fields[1], &bin->radius_upper_m) &&
           parse_finite_double(fields[2], &bin->number_density_per_m4) &&
           parse_finite_double(fields[3], &bin->inventory_mol);
}

bool inventories_match(double declared, double summed) {
    const double scale = std::fmax(
        1.0e-30, std::fmax(std::fabs(declared), std::fabs(summed)));
    return std::fabs(declared - summed) <= 1.0e-12 * scale;
}

bool build_bins(const std::vector<ContinuumBin>& source,
                std::uint64_t declared_count, double box_volume_m3,
                double x_b, double vm_m3_mol, double declared_inventory_mol,
                const char* label,
                std::vector<pf_zero_mode::AuxiliaryPopulationBin>* destination,
                std::string* error) {
    if (source.size() != declared_count) {
        return set_error(error, std::string(label) + " bin count mismatch");
    }
    if (declared_inventory_mol > 0.0 && source.empty()) {
        return set_error(error,
                         std::string(label) + " nonzero inventory has no bins");
    }
    destination->clear();
    destination->reserve(source.size());
    double summed_inventory_mol = 0.0;
    for (const ContinuumBin& input : source) {
        if (input.radius_lower_m <= 0.0 ||
            input.radius_upper_m <= input.radius_lower_m ||
            input.number_density_per_m4 < 0.0 || input.inventory_mol < 0.0) {
            return set_error(error, std::string(label) + " bin is invalid");
        }
        const double width_m = input.radius_upper_m - input.radius_lower_m;
        const double number_density_m3 = input.number_density_per_m4 * width_m;
        const double centre_m = 0.5 *
                                (input.radius_lower_m + input.radius_upper_m);
        const double particle_volume_m3 =
            4.0 * std::acos(-1.0) / 3.0 * centre_m * centre_m * centre_m;
        const double recomputed_inventory_mol =
            number_density_m3 * particle_volume_m3 * box_volume_m3 * x_b /
            vm_m3_mol;
        if (!std::isfinite(number_density_m3) ||
            !std::isfinite(recomputed_inventory_mol) || number_density_m3 < 0.0 ||
            !inventories_match(input.inventory_mol,
                               recomputed_inventory_mol)) {
            return set_error(error,
                             std::string(label) +
                                 " bin inventory does not match its PSD");
        }
        pf_zero_mode::AuxiliaryPopulationBin converted;
        converted.radius_lower_m = input.radius_lower_m;
        converted.radius_upper_m = input.radius_upper_m;
        // A continuum KWN density integrated over ΔR.  It intentionally does
        // not claim an integer resolved-PF particle count.
        converted.number_density_m3 = number_density_m3;
        converted.count = 0U;
        converted.xB = x_b;
        converted.molar_volume_m3_mol = vm_m3_mol;
        converted.inventory_mol = input.inventory_mol;
        destination->push_back(converted);
        summed_inventory_mol += input.inventory_mol;
    }
    if (!inventories_match(declared_inventory_mol, summed_inventory_mol)) {
        return set_error(error,
                         std::string(label) + " inventory does not close bins");
    }
    return true;
}

bool scalar_value(const std::map<std::string, std::string>& values,
                  const char* key, std::string* value, std::string* error) {
    const auto iterator = values.find(key);
    if (iterator == values.end() || iterator->second.empty()) {
        return set_error(error, std::string("missing sidecar field: ") + key);
    }
    *value = iterator->second;
    return true;
}

bool scalar_double(const std::map<std::string, std::string>& values,
                   const char* key, double* value, std::string* error) {
    std::string text;
    if (!scalar_value(values, key, &text, error)) return false;
    if (!parse_finite_double(text, value)) {
        return set_error(error, std::string("invalid sidecar number: ") + key);
    }
    return true;
}

bool scalar_count(const std::map<std::string, std::string>& values,
                  const char* key, std::uint64_t* value, std::string* error) {
    std::string text;
    if (!scalar_value(values, key, &text, error) || !parse_count(text, value)) {
        return set_error(error, std::string("invalid sidecar count: ") + key);
    }
    return true;
}

}  // namespace

bool read_auxiliary_population_handoff_v2(
    const std::string& path,
    const AuxiliaryHandoffIdentity& expected_identity,
    pf_zero_mode::AuxPopulationState* state,
    std::string* error) {
    if (!state) return set_error(error, "auxiliary handoff destination is null");
    if (!is_lower_hex_sha256(expected_identity.validation_contract_hash) ||
        !is_lower_hex_sha256(expected_identity.source_handoff_hash) ||
        !is_lower_hex_sha256(expected_identity.package_hash) ||
        !is_lower_hex_sha256(expected_identity.fixture_hash)) {
        return set_error(error, "expected auxiliary handoff hash is invalid");
    }

    std::ifstream stream(path);
    if (!stream) return set_error(error, "cannot open auxiliary handoff sidecar");
    const std::set<std::string> scalar_keys = {
        "schema_version",
        "validation_contract_hash",
        "source_handoff_hash",
        "package_hash",
        "fixture_hash",
        "state",
        "frozen",
        "units",
        "dynamics",
        "box_volume_m3",
        "Q_B_total_mol",
        "Q_B_matrix_mol",
        "Q_B_beta_resolved_fixed_mol",
        "Q_B_GP_mol",
        "Q_B_beta_subgrid_mol",
        "Q_B_bucket_sum_mol",
        "residual_mol",
        "relative_residual",
        "gp_population_provenance",
        "gp_population_disclaimer",
        "gp_xB",
        "gp_Vm_m3_mol",
        "gp_bin_count",
        "beta_subgrid_population_provenance",
        "beta_subgrid_population_disclaimer",
        "beta_subgrid_xB",
        "beta_subgrid_Vm_m3_mol",
        "beta_subgrid_bin_count"};
    std::map<std::string, std::string> values;
    std::vector<ContinuumBin> gp_source_bins;
    std::vector<ContinuumBin> subgrid_source_bins;
    std::string line;
    while (std::getline(stream, line)) {
        const std::size_t separator = line.find('=');
        if (line.empty() || separator == std::string::npos || separator == 0U ||
            separator + 1U >= line.size() ||
            line.find('=', separator + 1U) != std::string::npos) {
            return set_error(error, "auxiliary handoff sidecar line is malformed");
        }
        const std::string key = line.substr(0U, separator);
        const std::string value = line.substr(separator + 1U);
        if (key == "gp_bin") {
            ContinuumBin bin;
            if (!parse_bin(value, &bin)) {
                return set_error(error, "GP sidecar bin is malformed");
            }
            gp_source_bins.push_back(bin);
        } else if (key == "beta_subgrid_bin") {
            ContinuumBin bin;
            if (!parse_bin(value, &bin)) {
                return set_error(error, "subgrid sidecar bin is malformed");
            }
            subgrid_source_bins.push_back(bin);
        } else if (scalar_keys.count(key) != 0U && values.emplace(key, value).second) {
            continue;
        } else {
            return set_error(error, "auxiliary handoff sidecar has an unknown or duplicate field");
        }
    }
    if (!stream.eof()) return set_error(error, "cannot read auxiliary handoff sidecar");

    std::string schema;
    std::string contract_hash;
    std::string source_handoff_hash;
    std::string package_hash;
    std::string fixture_hash;
    std::string storage_state;
    std::string frozen;
    std::string units;
    std::string dynamics;
    std::string gp_provenance;
    std::string gp_disclaimer;
    std::string subgrid_provenance;
    std::string subgrid_disclaimer;
    if (!scalar_value(values, "schema_version", &schema, error) ||
        !scalar_value(values, "validation_contract_hash", &contract_hash, error) ||
        !scalar_value(values, "source_handoff_hash", &source_handoff_hash, error) ||
        !scalar_value(values, "package_hash", &package_hash, error) ||
        !scalar_value(values, "fixture_hash", &fixture_hash, error) ||
        !scalar_value(values, "state", &storage_state, error) ||
        !scalar_value(values, "frozen", &frozen, error) ||
        !scalar_value(values, "units", &units, error) ||
        !scalar_value(values, "dynamics", &dynamics, error) ||
        !scalar_value(values, "gp_population_provenance", &gp_provenance, error) ||
        !scalar_value(values, "gp_population_disclaimer", &gp_disclaimer, error) ||
        !scalar_value(values, "beta_subgrid_population_provenance",
                      &subgrid_provenance, error) ||
        !scalar_value(values, "beta_subgrid_population_disclaimer",
                      &subgrid_disclaimer, error)) {
        return false;
    }
    if (schema != kAuxiliaryHandoffSidecarSchemaV1 ||
        !is_lower_hex_sha256(contract_hash) ||
        !is_lower_hex_sha256(source_handoff_hash) ||
        !is_lower_hex_sha256(package_hash) || !is_lower_hex_sha256(fixture_hash) ||
        contract_hash != expected_identity.validation_contract_hash ||
        source_handoff_hash != expected_identity.source_handoff_hash ||
        package_hash != expected_identity.package_hash ||
        fixture_hash != expected_identity.fixture_hash ||
        storage_state != kAuxiliaryState || frozen != "true" ||
        units != kAuxiliaryUnits || dynamics != kDynamicsExcluded ||
        gp_provenance != kPopulationProvenance ||
        subgrid_provenance != kPopulationProvenance ||
        gp_disclaimer != kPopulationDisclaimer ||
        subgrid_disclaimer != kPopulationDisclaimer) {
        return set_error(error, "auxiliary handoff schema or provenance mismatch");
    }

    double box_volume_m3 = 0.0;
    double q_total_mol = 0.0;
    double q_matrix_mol = 0.0;
    double q_resolved_mol = 0.0;
    double q_gp_mol = 0.0;
    double q_subgrid_mol = 0.0;
    double q_bucket_sum_mol = 0.0;
    double residual_mol = 0.0;
    double relative_residual = 0.0;
    double gp_x_b = 0.0;
    double gp_vm_m3_mol = 0.0;
    double subgrid_x_b = 0.0;
    double subgrid_vm_m3_mol = 0.0;
    std::uint64_t gp_bin_count = 0U;
    std::uint64_t subgrid_bin_count = 0U;
    if (!scalar_double(values, "box_volume_m3", &box_volume_m3, error) ||
        !scalar_double(values, "Q_B_total_mol", &q_total_mol, error) ||
        !scalar_double(values, "Q_B_matrix_mol", &q_matrix_mol, error) ||
        !scalar_double(values, "Q_B_beta_resolved_fixed_mol", &q_resolved_mol,
                       error) ||
        !scalar_double(values, "Q_B_GP_mol", &q_gp_mol, error) ||
        !scalar_double(values, "Q_B_beta_subgrid_mol", &q_subgrid_mol, error) ||
        !scalar_double(values, "Q_B_bucket_sum_mol", &q_bucket_sum_mol,
                       error) ||
        !scalar_double(values, "residual_mol", &residual_mol, error) ||
        !scalar_double(values, "relative_residual", &relative_residual, error) ||
        !scalar_double(values, "gp_xB", &gp_x_b, error) ||
        !scalar_double(values, "gp_Vm_m3_mol", &gp_vm_m3_mol, error) ||
        !scalar_double(values, "beta_subgrid_xB", &subgrid_x_b, error) ||
        !scalar_double(values, "beta_subgrid_Vm_m3_mol", &subgrid_vm_m3_mol,
                       error) ||
        !scalar_count(values, "gp_bin_count", &gp_bin_count, error) ||
        !scalar_count(values, "beta_subgrid_bin_count", &subgrid_bin_count,
                      error)) {
        return false;
    }
    if (box_volume_m3 <= 0.0 || q_total_mol <= 0.0 || q_matrix_mol < 0.0 ||
        q_resolved_mol < 0.0 || q_gp_mol < 0.0 || q_subgrid_mol < 0.0 ||
        q_bucket_sum_mol < 0.0 || relative_residual < 0.0 ||
        relative_residual > kLedgerRelativeTolerance || gp_x_b < 0.0 ||
        gp_x_b > 1.0 || subgrid_x_b < 0.0 || subgrid_x_b > 1.0 ||
        gp_vm_m3_mol <= 0.0 || subgrid_vm_m3_mol <= 0.0) {
        return set_error(error, "auxiliary handoff numeric domain is invalid");
    }

    std::vector<pf_zero_mode::AuxiliaryPopulationBin> gp_bins;
    std::vector<pf_zero_mode::AuxiliaryPopulationBin> subgrid_bins;
    if (!build_bins(gp_source_bins, gp_bin_count, box_volume_m3, gp_x_b,
                    gp_vm_m3_mol, q_gp_mol, "GP", &gp_bins, error) ||
        !build_bins(subgrid_source_bins, subgrid_bin_count, box_volume_m3,
                    subgrid_x_b, subgrid_vm_m3_mol, q_subgrid_mol,
                    "subgrid-beta", &subgrid_bins, error)) {
        return false;
    }

    const double bucket_sum_mol =
        q_matrix_mol + q_resolved_mol + q_gp_mol + q_subgrid_mol;
    const double recomputed_residual_mol = q_total_mol - bucket_sum_mol;
    if (!inventories_match(q_bucket_sum_mol, bucket_sum_mol) ||
        !inventories_match(q_total_mol, bucket_sum_mol) ||
        !inventories_match(residual_mol, recomputed_residual_mol)) {
        return set_error(error, "auxiliary handoff four-bucket ledger does not close");
    }

    pf_zero_mode::AuxPopulationState parsed;
    parsed.schema_version = pf_zero_mode::kAuxiliaryPopulationSchemaV1;
    parsed.present = true;
    parsed.frozen = true;
    parsed.state = kAuxiliaryState;
    parsed.validation_contract_hash = contract_hash;
    parsed.source_handoff_hash = source_handoff_hash;
    parsed.package_handoff_hash = package_hash;
    parsed.units = kAuxiliaryUnits;
    parsed.Q_B_GP_mol = q_gp_mol;
    parsed.Q_B_beta_subgrid_mol = q_subgrid_mol;
    parsed.gp_population_provenance = gp_provenance;
    parsed.beta_subgrid_population_provenance = subgrid_provenance;
    parsed.gp_bins = std::move(gp_bins);
    parsed.beta_subgrid_bins = std::move(subgrid_bins);
    *state = std::move(parsed);
    return true;
}

}  // namespace pf_auxiliary_handoff_v2
