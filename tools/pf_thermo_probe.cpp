// Host-only numerical view of the generated PF/KWN validation contract.
//
// This deliberately enters through the active PF thermodynamic wrapper. That
// wrapper includes the generated contract view, so a successful probe covers
// both the generated coefficients and the PF-facing call path.

#include "thermo_utils.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<double> parse_csv(const std::string& text, const char* flag) {
    std::vector<double> values;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        char* end = NULL;
        const double value = std::strtod(token.c_str(), &end);
        if (end == token.c_str() || *end != '\0' || !(value > 0.0)) {
            throw std::runtime_error(std::string("invalid positive value for ") + flag);
        }
        values.push_back(value);
    }
    if (values.empty()) {
        throw std::runtime_error(std::string("empty list for ") + flag);
    }
    return values;
}

const char* require_value(int argc, char** argv, int* index, const char* flag) {
    if (*index + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + flag);
    }
    ++*index;
    return argv[*index];
}

}  // namespace

int main(int argc, char** argv) {
    try {
        double temperature_K = 653.15;
        std::vector<double> x_values = {
            1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 0.004, 0.00465, 0.005,
            0.01, 0.02, 0.05};
        std::vector<double> radii_nm = {2.0, 5.0, 10.0, 20.0, 50.0};

        for (int index = 1; index < argc; ++index) {
            const std::string flag(argv[index]);
            if (flag == "--temperature-k") {
                const char* value = require_value(argc, argv, &index, "--temperature-k");
                char* end = NULL;
                temperature_K = std::strtod(value, &end);
                if (end == value || *end != '\0' || !(temperature_K > 0.0)) {
                    throw std::runtime_error("--temperature-k must be positive");
                }
            } else if (flag == "--x-values") {
                x_values = parse_csv(require_value(argc, argv, &index, "--x-values"), "--x-values");
            } else if (flag == "--radii-nm") {
                radii_nm = parse_csv(require_value(argc, argv, &index, "--radii-nm"), "--radii-nm");
            } else if (flag == "--expected-contract-hash") {
                const std::string expected(
                    require_value(argc, argv, &index, "--expected-contract-hash"));
                if (expected != PF_KWN_VALIDATION_CONTRACT_HASH) {
                    throw std::runtime_error("compiled contract hash does not match expected hash");
                }
            } else if (flag == "--help") {
                std::cout << "Usage: pf_thermo_probe [--temperature-k T] [--x-values a,b] "
                             "[--radii-nm a,b] [--expected-contract-hash sha256]\n";
                return 0;
            } else {
                throw std::runtime_error("unknown option: " + flag);
            }
        }

        const double solvus = solve_x_eq_device(temperature_K);
        const double diffusivity = D_Ag_in_PbTe_m2_per_s(temperature_K);
        std::cout << std::setprecision(17);
        std::cout
            << "contract_hash,temperature_K,xB,radius_m,G_alpha_J_mol,mu_A_J_mol,"
               "mu_B_J_mol,dGdx_J_mol,d2Gdx2_J_mol,driving_force_beta_J_mol,"
               "D_alpha_m2_s,planar_solvus_xB,curvature_equilibrium_xB,"
               "Vm_alpha_m3_mol,Vm_beta_m3_mol\n";
        for (double xB : x_values) {
            for (double radius_nm : radii_nm) {
                const double radius_m = radius_nm * 1.0e-9;
                std::cout << PF_KWN_VALIDATION_CONTRACT_HASH << ',' << temperature_K << ','
                          << xB << ',' << radius_m << ','
                          << pf_kwn_G_alpha(temperature_K, xB) << ','
                          << mu_PbTe_calphad(temperature_K, xB) << ','
                          << mu_Ag2Te_calphad(temperature_K, xB) << ','
                          << (mu_Ag2Te_calphad(temperature_K, xB) -
                              mu_PbTe_calphad(temperature_K, xB)) << ','
                          << (dmu_Ag2Te_calphad_dx(temperature_K, xB) -
                              dmu_PbTe_calphad_dx(temperature_K, xB)) << ','
                          << pf_kwn_beta_driving_force(temperature_K, xB) << ','
                          << diffusivity << ',' << solvus << ','
                          << pf_kwn_curvature_equilibrium(temperature_K, radius_m, 0.0) << ','
                          << PF_KWN_VM_ALPHA_M3_PER_MOL << ','
                          << PF_KWN_VM_BETA_M3_PER_MOL << '\n';
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pf_thermo_probe: " << error.what() << '\n';
        return 2;
    }
}
