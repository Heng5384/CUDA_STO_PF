#include <cmath>
#include <cstdio>
#include <vector>

#define THERMO_UTILS_DEFINE_GLOBALS
#include "phase_functions.h"
#include "thermo_utils.h"
#include <cuda_runtime.h>
#include <cufft.h>

#define CUDA_OK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return 2; } } while (0)
#define CUFFT_OK(call) do { cufftResult e_ = (call); if (e_ != CUFFT_SUCCESS) { \
    std::fprintf(stderr, "cuFFT error %s:%d: %d\n", __FILE__, __LINE__, (int)e_); return 2; } } while (0)

__global__ void formula_kernel(const double *phi, const double *x, double *out, int n,
                               double T, double scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[5*i+0] = h_of_phi(phi[i]);
    out[5*i+1] = h_prime_of_phi(phi[i]);
    out[5*i+2] = mu_A_dimless(x[i], T, scale);
    out[5*i+3] = mu_B_dimless(x[i], T, scale);
    out[5*i+4] = gamma_thermo_nonlinear(x[i], h_of_phi(phi[i]), 1.0, 0.0, 1.0, T, scale);
}

__global__ void spectral_derivative(cufftDoubleComplex *f, int n, double length, int order) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k > n/2) return;
    const double wave = 2.0 * M_PI * k / length;
    cufftDoubleComplex z = f[k];
    if (order == 1) {
        f[k].x = -wave * z.y;
        f[k].y =  wave * z.x;
    } else {
        f[k].x = -wave * wave * z.x;
        f[k].y = -wave * wave * z.y;
    }
    if (k == 0) { f[k].x = 0.0; f[k].y = 0.0; }
}

int main() {
    int device_count = 0;
    CUDA_OK(cudaGetDeviceCount(&device_count));
    if (device_count < 1) { std::fprintf(stderr, "no CUDA device\n"); return 2; }
    const int thermo_flag = 1;
    h_thermo_convex_extrapolation_enabled = thermo_flag;
    CUDA_OK(cudaMemcpyToSymbol(d_thermo_convex_extrapolation_enabled, &thermo_flag, sizeof(int)));

    const int n = 8;
    std::vector<double> phi(n), x(n), got(5*n);
    const double x_values[n] = {0.0078305391025, 0.03, 0.079, 0.089,
                                0.091, 0.12, 0.20, 0.3529411764705882};
    for (int i = 0; i < n; ++i) { phi[i] = i / 7.0; x[i] = x_values[i]; }
    double *d_phi = nullptr, *d_x = nullptr, *d_out = nullptr;
    CUDA_OK(cudaMalloc(&d_phi, n*sizeof(double)));
    CUDA_OK(cudaMalloc(&d_x, n*sizeof(double)));
    CUDA_OK(cudaMalloc(&d_out, 5*n*sizeof(double)));
    CUDA_OK(cudaMemcpy(d_phi, phi.data(), n*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_x, x.data(), n*sizeof(double), cudaMemcpyHostToDevice));
    formula_kernel<<<1, 32>>>(d_phi, d_x, d_out, n, 673.15, 1.3779024e5);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaMemcpy(got.data(), d_out, 5*n*sizeof(double), cudaMemcpyDeviceToHost));
    double parity_max = 0.0;
    for (int i = 0; i < n; ++i) {
        const double expected[5] = {
            h_of_phi(phi[i]), h_prime_of_phi(phi[i]),
            mu_A_dimless(x[i], 673.15, 1.3779024e5),
            mu_B_dimless(x[i], 673.15, 1.3779024e5),
            gamma_thermo_nonlinear(x[i], h_of_phi(phi[i]), 1.0, 0.0, 1.0, 673.15, 1.3779024e5)};
        for (int j = 0; j < 5; ++j) parity_max = fmax(parity_max, fabs(got[5*i+j] - expected[j]));
    }
    CUDA_OK(cudaFree(d_phi)); CUDA_OK(cudaFree(d_x)); CUDA_OK(cudaFree(d_out));

    const int N = 32, mode = 3;
    const double L = (double)N;
    std::vector<double> signal(N), roundtrip(N), deriv(N), lap(N);
    for (int i = 0; i < N; ++i) signal[i] = sin(2.0*M_PI*mode*i/N);
    double *d_real = nullptr;
    cufftDoubleComplex *d_freq = nullptr;
    CUDA_OK(cudaMalloc(&d_real, N*sizeof(double)));
    CUDA_OK(cudaMalloc(&d_freq, (N/2+1)*sizeof(cufftDoubleComplex)));
    cufftHandle r2c, c2r;
    CUFFT_OK(cufftPlan1d(&r2c, N, CUFFT_D2Z, 1));
    CUFFT_OK(cufftPlan1d(&c2r, N, CUFFT_Z2D, 1));
    CUDA_OK(cudaMemcpy(d_real, signal.data(), N*sizeof(double), cudaMemcpyHostToDevice));
    CUFFT_OK(cufftExecD2Z(r2c, d_real, d_freq));
    CUFFT_OK(cufftExecZ2D(c2r, d_freq, d_real));
    CUDA_OK(cudaMemcpy(roundtrip.data(), d_real, N*sizeof(double), cudaMemcpyDeviceToHost));
    double roundtrip_max = 0.0;
    for (int i = 0; i < N; ++i) roundtrip_max = fmax(roundtrip_max, fabs(roundtrip[i]/N-signal[i]));

    CUDA_OK(cudaMemcpy(d_real, signal.data(), N*sizeof(double), cudaMemcpyHostToDevice));
    CUFFT_OK(cufftExecD2Z(r2c, d_real, d_freq));
    spectral_derivative<<<1, 32>>>(d_freq, N, L, 1);
    CUFFT_OK(cufftExecZ2D(c2r, d_freq, d_real));
    CUDA_OK(cudaMemcpy(deriv.data(), d_real, N*sizeof(double), cudaMemcpyDeviceToHost));
    double derivative_max = 0.0;
    const double wave = 2.0*M_PI*mode/L;
    for (int i = 0; i < N; ++i)
        derivative_max = fmax(derivative_max, fabs(deriv[i]/N-wave*cos(2.0*M_PI*mode*i/N)));

    CUDA_OK(cudaMemcpy(d_real, signal.data(), N*sizeof(double), cudaMemcpyHostToDevice));
    CUFFT_OK(cufftExecD2Z(r2c, d_real, d_freq));
    spectral_derivative<<<1, 32>>>(d_freq, N, L, 2);
    cufftDoubleComplex k0;
    CUDA_OK(cudaMemcpy(&k0, d_freq, sizeof(k0), cudaMemcpyDeviceToHost));
    CUFFT_OK(cufftExecZ2D(c2r, d_freq, d_real));
    CUDA_OK(cudaMemcpy(lap.data(), d_real, N*sizeof(double), cudaMemcpyDeviceToHost));
    double lap_max = 0.0;
    for (int i = 0; i < N; ++i)
        lap_max = fmax(lap_max, fabs(lap[i]/N+wave*wave*signal[i]));

    std::printf("cuda_formula_parity_max=%.17g\n", parity_max);
    std::printf("fft_roundtrip_max=%.17g\n", roundtrip_max);
    std::printf("fft_derivative_sign_max=%.17g\n", derivative_max);
    std::printf("fft_laplacian_max=%.17g\n", lap_max);
    std::printf("fft_k0_divergence_abs=%.17g\n", hypot(k0.x, k0.y));
    const bool pass = parity_max <= 1e-10 && roundtrip_max <= 1e-12 &&
                      derivative_max <= 1e-11 && lap_max <= 1e-11 && hypot(k0.x,k0.y) <= 1e-15;
    std::printf("cuda_fft_gate=%s\n", pass ? "PASS" : "BLOCKED");

    cufftDestroy(r2c); cufftDestroy(c2r);
    cudaFree(d_real); cudaFree(d_freq);
    return pass ? 0 : 1;
}
