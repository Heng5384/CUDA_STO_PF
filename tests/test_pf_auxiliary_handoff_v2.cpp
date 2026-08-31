#include "../pf_auxiliary_handoff_v2.h"
#include "../generated/pf_kwn_validation_contract_v1.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

constexpr const char* kContractHash = PF_KWN_VALIDATION_CONTRACT_HASH;
constexpr const char* kSourceHandoffHash =
    "1111111111111111111111111111111111111111111111111111111111111111";
constexpr const char* kPackageHash =
    "2222222222222222222222222222222222222222222222222222222222222222";
constexpr const char* kFixtureHash =
    "3333333333333333333333333333333333333333333333333333333333333333";

bool require(bool condition, const char* message) {
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

std::string number(double value) {
    std::ostringstream stream;
    stream << std::scientific << std::setprecision(17) << value;
    return stream.str();
}

double bin_inventory(double lower_m, double upper_m,
                     double number_density_per_m4, double box_volume_m3,
                     double x_b, double vm_m3_mol) {
    const double width_m = upper_m - lower_m;
    const double centre_m = 0.5 * (lower_m + upper_m);
    const double particle_volume_m3 =
        4.0 * std::acos(-1.0) / 3.0 * centre_m * centre_m * centre_m;
    return number_density_per_m4 * width_m * particle_volume_m3 *
           box_volume_m3 * x_b / vm_m3_mol;
}

bool write_sidecar(const std::string& path, bool corrupt_ledger) {
    const double box_volume_m3 = 1.0e-18;
    const double gp_lower_m = 0.75e-9;
    const double gp_upper_m = 1.25e-9;
    const double gp_density_per_m4 = 2.0e31;
    const double gp_x_b = 0.03;
    const double gp_vm_m3_mol = 2.5e-5;
    const double gp_inventory_mol =
        bin_inventory(gp_lower_m, gp_upper_m, gp_density_per_m4,
                      box_volume_m3, gp_x_b, gp_vm_m3_mol);
    const double subgrid_lower_m = 1.75e-9;
    const double subgrid_upper_m = 2.25e-9;
    const double subgrid_density_per_m4 = 1.0e30;
    const double subgrid_x_b = 1.0;
    const double subgrid_vm_m3_mol = 2.7e-5;
    const double subgrid_inventory_mol =
        bin_inventory(subgrid_lower_m, subgrid_upper_m,
                      subgrid_density_per_m4, box_volume_m3, subgrid_x_b,
                      subgrid_vm_m3_mol);
    const double q_matrix_mol = 2.0e-18;
    const double q_resolved_mol = 3.0e-18;
    const double bucket_sum_mol = q_matrix_mol + q_resolved_mol +
                                  gp_inventory_mol + subgrid_inventory_mol;
    const double q_total_mol =
        corrupt_ledger ? bucket_sum_mol * 1.01 : bucket_sum_mol;

    std::ofstream stream(path, std::ios::trunc);
    if (!stream) return false;
    stream << "schema_version=PF_AUXILIARY_HANDOFF_V2_SIDECAR_V1\n"
           << "validation_contract_hash=" << kContractHash << '\n'
           << "source_handoff_hash=" << kSourceHandoffHash << '\n'
           << "package_hash=" << kPackageHash << '\n'
           << "fixture_hash=" << kFixtureHash << '\n'
           << "state=AUXILIARY_POPULATION_STORAGE_ONLY_V1\n"
           << "frozen=true\n"
           << "units=mol_B\n"
           << "dynamics=STORAGE_ONLY_NO_PF_FIELD_MUTATION\n"
           << "box_volume_m3=" << number(box_volume_m3) << '\n'
           << "Q_B_total_mol=" << number(q_total_mol) << '\n'
           << "Q_B_matrix_mol=" << number(q_matrix_mol) << '\n'
           << "Q_B_beta_resolved_fixed_mol=" << number(q_resolved_mol)
           << '\n'
           << "Q_B_GP_mol=" << number(gp_inventory_mol) << '\n'
           << "Q_B_beta_subgrid_mol=" << number(subgrid_inventory_mol)
           << '\n'
           << "Q_B_bucket_sum_mol=" << number(bucket_sum_mol) << '\n'
           << "residual_mol=" << number(q_total_mol - bucket_sum_mol)
           << '\n'
           << "relative_residual="
           << number(std::fabs(q_total_mol - bucket_sum_mol) / q_total_mol)
           << '\n'
           << "gp_population_provenance="
              "PRESCRIBED_NON_PREDICTIVE_STORAGE_CONTROL|"
              "compact_host_population_state_v2\n"
           << "gp_population_disclaimer="
              "FIXTURE_CONDITIONED_PRESCRIBED_SOURCE_IS_NOT_A_GP_NUCLEATION_PREDICTION\n"
           << "gp_xB=" << number(gp_x_b) << '\n'
           << "gp_Vm_m3_mol=" << number(gp_vm_m3_mol) << '\n'
           << "gp_bin_count=1\n"
           << "gp_bin=" << number(gp_lower_m) << ',' << number(gp_upper_m)
           << ',' << number(gp_density_per_m4) << ','
           << number(gp_inventory_mol) << '\n'
           << "beta_subgrid_population_provenance="
              "PRESCRIBED_NON_PREDICTIVE_STORAGE_CONTROL|"
              "compact_host_population_state_v2\n"
           << "beta_subgrid_population_disclaimer="
              "FIXTURE_CONDITIONED_PRESCRIBED_SOURCE_IS_NOT_A_GP_NUCLEATION_PREDICTION\n"
           << "beta_subgrid_xB=" << number(subgrid_x_b) << '\n'
           << "beta_subgrid_Vm_m3_mol=" << number(subgrid_vm_m3_mol)
           << '\n'
           << "beta_subgrid_bin_count=1\n"
           << "beta_subgrid_bin=" << number(subgrid_lower_m) << ','
           << number(subgrid_upper_m) << ','
           << number(subgrid_density_per_m4) << ','
           << number(subgrid_inventory_mol) << '\n';
    stream.close();
    return static_cast<bool>(stream);
}

pf_auxiliary_handoff_v2::AuxiliaryHandoffIdentity expected_identity() {
    pf_auxiliary_handoff_v2::AuxiliaryHandoffIdentity identity;
    identity.validation_contract_hash = kContractHash;
    identity.source_handoff_hash = kSourceHandoffHash;
    identity.package_hash = kPackageHash;
    identity.fixture_hash = kFixtureHash;
    return identity;
}

}  // namespace

int main() {
    const std::string path = "/tmp/pf_auxiliary_handoff_v2_test.txt";
    std::remove(path.c_str());
    if (!require(write_sidecar(path, false), "write valid sidecar")) return 1;

    pf_zero_mode::AuxPopulationState state;
    std::string error;
    const auto identity = expected_identity();
    if (!require(pf_auxiliary_handoff_v2::read_auxiliary_population_handoff_v2(
                     path, identity, &state, &error),
                 "read valid sidecar")) {
        std::cerr << error << '\n';
        return 1;
    }
    const double expected_gp_density_m3 = 2.0e31 * (1.25e-9 - 0.75e-9);
    if (!require(state.present && state.frozen &&
                     state.schema_version ==
                         pf_zero_mode::kAuxiliaryPopulationSchemaV1,
                 "materialized state identity") ||
        !require(state.validation_contract_hash == kContractHash &&
                     state.source_handoff_hash == kSourceHandoffHash &&
                     state.package_handoff_hash == kPackageHash,
                 "materialized state hash bind") ||
        !require(state.gp_bins.size() == 1U &&
                     state.beta_subgrid_bins.size() == 1U,
                 "materialized PSD bin counts") ||
        !require(state.gp_bins.front().count == 0U,
                 "continuum GP bin has no discrete particle count") ||
        !require(std::fabs(state.gp_bins.front().number_density_m3 -
                           expected_gp_density_m3) <=
                     1.0e-12 * expected_gp_density_m3,
                 "m^-4 to integrated m^-3 conversion") ||
        !require(state.Q_B_GP_mol == state.gp_bins.front().inventory_mol &&
                     state.Q_B_beta_subgrid_mol ==
                         state.beta_subgrid_bins.front().inventory_mol,
                 "per-bin inventories are preserved")) {
        std::remove(path.c_str());
        return 1;
    }

    auto wrong_package = identity;
    wrong_package.package_hash.assign(64U, '4');
    if (!require(!pf_auxiliary_handoff_v2::read_auxiliary_population_handoff_v2(
                     path, wrong_package, &state, &error),
                 "package identity mismatch fails closed")) {
        std::remove(path.c_str());
        return 1;
    }
    if (!require(write_sidecar(path, true), "write corrupted-ledger sidecar") ||
        !require(!pf_auxiliary_handoff_v2::read_auxiliary_population_handoff_v2(
                     path, identity, &state, &error),
                 "ledger mismatch fails closed")) {
        std::remove(path.c_str());
        return 1;
    }
    std::remove(path.c_str());
    std::cout << "PASS_PF_AUXILIARY_HANDOFF_V2_HOST_MATERIALIZER\n";
    return 0;
}
