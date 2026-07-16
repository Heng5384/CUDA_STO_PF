#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define THERMO_UTILS_DEFINE_GLOBALS
#include "phase_functions.h"
#include "thermo_utils.h"
#include "cuda_kernels.h"
#include <cuda_runtime.h>

#define CUDA_OK(call) do { \
    const cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(err_)); \
        return false; \
    } \
} while (0)

struct CaseSpec {
    const char *name;
    bool anisotropic;
    bool beta_slab;
    bool smooth_interface;
    bool circular;
    bool correction;
    bool constant_mu;
    bool fourier_mu;
    bool constant_x;
};

static int index3(int i, int j, int k, int Ny, int Nz) {
    return (i * Ny + j) * Nz + k;
}

static bool run_case(const CaseSpec &spec, std::FILE *csv) {
    const int Nx = 10, Ny = 9, Nz = 8;
    const int n = Nx * Ny * Nz;
    const double dx = spec.anisotropic ? 0.7 : 1.0;
    const double dy = spec.anisotropic ? 1.1 : 1.0;
    const double dz = spec.anisotropic ? 1.6 : 1.0;
    const double volume = dx * dy * dz;
    const double spacings[3] = {dx, dy, dz};
    std::vector<double> mu(n), phi(n, 0.0), phi_old(n, 0.0), x(n, 0.03);
    std::mt19937_64 rng(20260713);
    std::uniform_real_distribution<double> random_mu(-1.0, 1.0);
    std::uniform_real_distribution<double> random_x(0.01, 0.12);
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                const int p = index3(i, j, k, Ny, Nz);
                const double rx = (i - 0.5 * (Nx - 1)) * dx;
                const double ry = (j - 0.5 * (Ny - 1)) * dy;
                const double rz = (k - 0.5 * (Nz - 1)) * dz;
                const double radius = std::sqrt(rx * rx + ry * ry + rz * rz);
                if (spec.constant_mu) {
                    mu[p] = 0.375;
                } else if (spec.fourier_mu) {
                    mu[p] = std::sin(2.0 * M_PI * i / Nx) +
                            0.35 * std::cos(2.0 * M_PI * j / Ny);
                } else {
                    mu[p] = random_mu(rng);
                }
                if (!spec.constant_x) x[p] = random_x(rng);
                if (spec.beta_slab) {
                    phi[p] = (i >= 3 && i <= 5) ? 1.0 : 0.0;
                } else if (spec.smooth_interface) {
                    phi[p] = 0.5 * (1.0 - std::tanh((rx - 0.25) / 1.2));
                } else if (spec.circular) {
                    phi[p] = 0.5 * (1.0 - std::tanh((radius - 2.4) / 0.9));
                }
                phi_old[p] = phi[p];
                if (spec.correction) {
                    const double shifted = radius - 2.4 + 1.0e-4;
                    phi_old[p] = 0.5 * (1.0 - std::tanh(shifted / 0.9));
                }
            }
        }
    }

    double *d_mu = nullptr, *d_phi = nullptr, *d_phi_old = nullptr;
    double *d_x = nullptr, *d_fx = nullptr, *d_fy = nullptr;
    double *d_fz = nullptr, *d_div = nullptr, *d_stats = nullptr;
    CUDA_OK(cudaMalloc(&d_mu, n * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_phi, n * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_phi_old, n * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_fx, n * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_fy, n * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_fz, n * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_div, n * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_stats, CTOT_TRANSPORT_STATS_COUNT * sizeof(double)));
    CUDA_OK(cudaMemcpy(d_mu, mu.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_phi, phi.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_phi_old, phi_old.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_x, x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double> stats(CTOT_TRANSPORT_STATS_COUNT, 0.0);
    stats[CTOT_TRANSPORT_MIN_M] = HUGE_VAL;
    CUDA_OK(cudaMemcpy(d_stats, stats.data(), stats.size() * sizeof(double),
                       cudaMemcpyHostToDevice));

    double *faces[3] = {d_fx, d_fy, d_fz};
    for (int axis = 0; axis < 3; ++axis) {
        launch_ctot_fv_positive_face_flux_kernel(
            d_mu, d_phi, d_x, faces[axis], Nx, Ny, Nz, axis,
            spacings[axis], 1.0e-2, 1.0, 0.0, 1.0, 673.15, 1.3779024e5,
            1.0e-12, 0.0, d_stats, n);
    }
    if (spec.correction) {
        for (int axis = 0; axis < 3; ++axis) {
            launch_ctot_add_antitrapping_face_flux_kernel(
                d_phi, d_phi_old, d_x, faces[axis], Nx, Ny, Nz, axis,
                dx, dy, dz, 0.6, 1.0e-3, 1.0, d_mu, d_stats, n);
        }
    }
    launch_ctot_fv_divergence_kernel(d_fx, d_fy, d_fz, d_div,
                                      Nx, Ny, Nz, dx, dy, dz, n);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaGetLastError());

    std::vector<double> fx(n), fy(n), fz(n), div(n);
    CUDA_OK(cudaMemcpy(fx.data(), d_fx, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(fy.data(), d_fy, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(fz.data(), d_fz, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(div.data(), d_div, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(stats.data(), d_stats, stats.size() * sizeof(double),
                       cudaMemcpyDeviceToHost));

    const std::vector<double> *host_faces[3] = {&fx, &fy, &fz};
    long double cell_power = 0.0L, face_pairing = 0.0L;
    long double mass_rate = 0.0L;
    double max_flux_beta_face = 0.0;
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                const int p = index3(i, j, k, Ny, Nz);
                cell_power += volume * (long double)mu[p] * div[p];
                mass_rate += volume * (long double)div[p];
                for (int axis = 0; axis < 3; ++axis) {
                    int ip = i, jp = j, kp = k;
                    if (axis == 0) ip = (i + 1) % Nx;
                    else if (axis == 1) jp = (j + 1) % Ny;
                    else kp = (k + 1) % Nz;
                    const int q = index3(ip, jp, kp, Ny, Nz);
                    const double grad = (mu[q] - mu[p]) / spacings[axis];
                    const double flux = (*host_faces[axis])[p];
                    face_pairing += volume * (long double)grad * flux;
                    if (phi[p] == 1.0 || phi[q] == 1.0)
                        max_flux_beta_face = std::fmax(max_flux_beta_face,
                                                      std::fabs(flux));
                }
            }
        }
    }
    const long double adjoint_defect = cell_power + face_pairing;
    const long double scale = std::fmax(
        1.0L, std::fmax(std::fabs(cell_power), std::fabs(face_pairing)));
    const double adjoint_rel = (double)(std::fabs(adjoint_defect) / scale);
    const double mass_rel = (double)(std::fabs(mass_rate) /
                                     std::fmax(1.0L, std::fabs(face_pairing)));
    const bool base_dissipation_ok = spec.correction || face_pairing >= -1.0e-13;
    const bool endpoint_ok = !spec.beta_slab || max_flux_beta_face <= 1.0e-15;
    const bool finite_ok = stats[CTOT_TRANSPORT_NONFINITE_COUNT] == 0.0 &&
                           stats[CTOT_TRANSPORT_MOBILITY_FAILURE_COUNT] == 0.0;
    const bool pass = adjoint_rel <= 2.0e-13 && mass_rel <= 2.0e-13 &&
                      base_dissipation_ok && endpoint_ok && finite_ok;
    std::fprintf(csv,
        "%s,%.17e,%.17e,%.17Le,%.17Le,%.17e,%.17e,%.17e,%.0f,%.0f,%s\n",
        spec.name, adjoint_rel, mass_rel, cell_power, face_pairing,
        stats[CTOT_TRANSPORT_DISSIPATION_SUM],
        stats[CTOT_TRANSPORT_ANTITRAPPING_WORK_SUM], max_flux_beta_face,
        stats[CTOT_TRANSPORT_ZERO_FACE_COUNT],
        stats[CTOT_TRANSPORT_MOBILITY_FAILURE_COUNT], pass ? "PASS" : "FAIL");
    std::printf("%s=%s adjoint_rel=%.3e mass_rel=%.3e\n",
                spec.name, pass ? "PASS" : "FAIL", adjoint_rel, mass_rel);

    cudaFree(d_mu); cudaFree(d_phi); cudaFree(d_phi_old); cudaFree(d_x);
    cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz); cudaFree(d_div);
    cudaFree(d_stats);
    return pass;
}

int main(int argc, char **argv) {
    int device_count = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (count_error != cudaSuccess || device_count < 1) {
        std::fprintf(stderr, "no CUDA device available\n");
        return 2;
    }
    sync_thermo_runtime_flag_cuda_kernels(1);
    const char *output = argc > 1 ? argv[1] : "prompt7g_cuda_adjoint_identity.csv";
    std::FILE *csv = std::fopen(output, "w");
    if (!csv) return 2;
    std::fprintf(csv,
        "case,adjoint_relative_defect,global_mass_relative_defect,cell_power,"
        "face_pairing,kernel_base_dissipation_sum,correction_work_sum,"
        "max_flux_touching_exact_beta,zero_face_count,mobility_failure_count,status\n");
    const CaseSpec cases[] = {
        {"constant_mu_constant_mobility", false, false, false, false, false, true, false, true},
        {"single_fourier_mode", false, false, false, false, false, false, true, true},
        {"random_mu_constant_mobility", false, false, false, false, false, false, false, true},
        {"random_mu_variable_mobility", false, false, false, false, false, false, false, false},
        {"anisotropic_spacing", true, false, false, false, false, false, false, false},
        {"pure_beta_slab", false, true, false, false, false, false, false, false},
        {"smooth_matrix_interface_beta", false, false, true, false, false, false, false, false},
        {"resolved_circular_equilibrium", false, false, false, true, false, false, true, false},
        {"resolved_circular_dissolution_off", false, false, false, true, false, false, false, false},
        {"resolved_circular_dissolution_on", false, false, false, true, true, false, false, false},
    };
    bool pass = true;
    for (const CaseSpec &spec : cases) pass = run_case(spec, csv) && pass;
    std::fclose(csv);
    std::printf("cuda_adjoint_identity_gate=%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
