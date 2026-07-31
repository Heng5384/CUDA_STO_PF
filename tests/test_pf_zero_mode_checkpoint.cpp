#include "../pf_zero_mode_checkpoint.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

namespace {

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
           left.phi == right.phi && left.Y == right.Y &&
           left.xB == right.xB &&
           left.dY_dt_prev == right.dY_dt_prev;
}

}  // namespace

int main() {
    const std::string path = "/tmp/pf_zero_mode_checkpoint_test.chk";
    const std::string elastic_path =
        "/tmp/pf_zero_mode_checkpoint_elastic_test.chk";
    std::remove(path.c_str());
    std::remove((path + ".tmp").c_str());
    std::remove(elastic_path.c_str());
    std::remove((elastic_path + ".tmp").c_str());

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
        std::cerr << "elastic V4 write failed: " << error << "\n";
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
            std::string(magic, sizeof(magic)) != "PFZMCHK4" ||
            version != 4U || header_bytes != 904U) {
            std::cerr << "elastic V4 disk prefix mismatch\n";
            return 1;
        }
    }
    pf_zero_mode::Checkpoint elastic_loaded;
    if (!pf_zero_mode::read_checkpoint(
            elastic_path, elastic_source.provenance,
            &elastic_loaded, &error) ||
        !exact_equal(elastic_source, elastic_loaded)) {
        std::cerr << "elastic V4 round-trip failed: " << error << "\n";
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
        std::cerr << "elastic V4 state was accepted by a legacy solver\n";
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
    std::cout << "PASS_PF_ZERO_MODE_CHECKPOINT_PROVENANCE_V3_V4\n";
    return 0;
}
