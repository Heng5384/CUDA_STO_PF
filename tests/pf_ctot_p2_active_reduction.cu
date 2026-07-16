#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "phase_pdas_reduction.h"

#define CUDA_TEST(call) do { \
    const cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s failed: %s\n", #call, \
                     cudaGetErrorString(error)); \
        return 2; \
    } \
} while (0)

static int run_case(int n, int repeat_count) {
    const double alpha = 5.0e-5;
    const double v_B = 1.0;
    const double x_min = 0.0;
    const double x_max = 1.0 - 1.0e-12;
    const double bound_tol = 1.0e-12;
    std::vector<double> phi(n, 0.5);
    std::vector<double> ctot(n, 0.5);
    std::vector<double> residual(n, 0.0);
    std::vector<double> previous(n, (double)PHASE_KKT_FREE);
    std::vector<double> active(n, NAN);

    for (int i = 0; i < n; ++i) {
        switch (i % 8) {
        case 0: residual[i] = 1.0e8; break;
        case 1: residual[i] = -1.0e8; break;
        case 2: residual[i] = 0.25; break;
        case 3: previous[i] = (double)PHASE_KKT_LOWER_ACTIVE; break;
        case 4: ctot[i] = NAN; break;
        case 5: phi[i] = NAN; break;
        case 6: residual[i] = NAN; break;
        default: previous[i] = (double)PHASE_KKT_UPPER_ACTIVE; break;
        }
    }

    PhasePdasTrialDecisionPacket expected = {};
    for (int i = 0; i < n; ++i) {
        const PhaseKktBounds bounds = phase_kkt_bounds_from_ctot(
            ctot[i], v_B, x_min, x_max, bound_tol);
        const int code = phase_kkt_active_code(
            phi[i], residual[i], alpha, bounds);
        expected.lower_count += code == PHASE_KKT_LOWER_ACTIVE;
        expected.upper_count += code == PHASE_KKT_UPPER_ACTIVE;
        expected.free_count += code == PHASE_KKT_FREE;
        expected.invalid_count += code == PHASE_KKT_INVALID;
        expected.nonfinite_count += !std::isfinite(phi[i]) ||
                                    !std::isfinite(residual[i]) ||
                                    !std::isfinite(ctot[i]);
        expected.bound_failure_count += !bounds.valid;
        if (code != PHASE_KKT_INVALID && (int)std::llrint(previous[i]) != code)
            expected.active_change_count += 1;
    }
    expected.active_set_stable = expected.active_change_count == 0;
    expected.trial_valid = expected.invalid_count == 0;

    double *d_phi = nullptr, *d_ctot = nullptr, *d_residual = nullptr;
    double *d_previous = nullptr, *d_active = nullptr;
    PhasePdasTrialBlockSummary *d_summaries = nullptr;
    PhasePdasTrialDecisionPacket *d_packet = nullptr;
    const size_t bytes = (size_t)n * sizeof(double);
    const int blocks = (n + PHASE_PDAS_TRIAL_REDUCTION_THREADS - 1) /
                       PHASE_PDAS_TRIAL_REDUCTION_THREADS;
    CUDA_TEST(cudaMalloc(&d_phi, bytes));
    CUDA_TEST(cudaMalloc(&d_ctot, bytes));
    CUDA_TEST(cudaMalloc(&d_residual, bytes));
    CUDA_TEST(cudaMalloc(&d_previous, bytes));
    CUDA_TEST(cudaMalloc(&d_active, bytes));
    CUDA_TEST(cudaMalloc(&d_summaries,
                         (size_t)blocks * sizeof(PhasePdasTrialBlockSummary)));
    CUDA_TEST(cudaMalloc(&d_packet, sizeof(PhasePdasTrialDecisionPacket)));
    CUDA_TEST(cudaMemcpy(d_phi, phi.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_TEST(cudaMemcpy(d_ctot, ctot.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_TEST(cudaMemcpy(d_residual, residual.data(), bytes,
                         cudaMemcpyHostToDevice));
    CUDA_TEST(cudaMemcpy(d_previous, previous.data(), bytes,
                         cudaMemcpyHostToDevice));

    PhasePdasTrialDecisionPacket first = {};
    for (int repeat = 0; repeat < repeat_count; ++repeat) {
        ctot_phase_pdas_compare_trial_active_kernel<<<
            blocks, PHASE_PDAS_TRIAL_REDUCTION_THREADS>>>(
            d_phi, d_ctot, d_residual, d_previous, d_active, alpha, v_B,
            x_min, x_max, bound_tol, d_summaries, n);
        ctot_phase_pdas_reduce_trial_summaries_kernel<<<
            1, PHASE_PDAS_TRIAL_REDUCTION_THREADS>>>(
            d_summaries, blocks, d_packet);
        CUDA_TEST(cudaGetLastError());
        PhasePdasTrialDecisionPacket actual = {};
        CUDA_TEST(cudaMemcpy(&actual, d_packet, sizeof(actual),
                             cudaMemcpyDeviceToHost));
        if (repeat == 0) first = actual;
        if (std::memcmp(&actual, &first, sizeof(actual)) != 0 ||
            std::memcmp(&actual, &expected, sizeof(actual)) != 0) {
            std::fprintf(stderr, "packet mismatch n=%d repeat=%d\n", n, repeat);
            return 1;
        }
    }
    CUDA_TEST(cudaMemcpy(active.data(), d_active, bytes,
                         cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        const PhaseKktBounds bounds = phase_kkt_bounds_from_ctot(
            ctot[i], v_B, x_min, x_max, bound_tol);
        const int expected_code = phase_kkt_active_code(
            phi[i], residual[i], alpha, bounds);
        if ((int)std::llrint(active[i]) != expected_code) {
            std::fprintf(stderr, "active code mismatch index=%d\n", i);
            return 1;
        }
    }

    cudaFree(d_phi);
    cudaFree(d_ctot);
    cudaFree(d_residual);
    cudaFree(d_previous);
    cudaFree(d_active);
    cudaFree(d_summaries);
    cudaFree(d_packet);
    return 0;
}

int main() {
    if (run_case(1031, 10) != 0) return 1;
    if (run_case(1 << 20, 3) != 0) return 1;
    std::printf("P2_ACTIVE_REDUCTION_PASS exact_packets=1 deterministic=1 "
                "per_cell_active_codes_exact=1\n");
    return 0;
}
