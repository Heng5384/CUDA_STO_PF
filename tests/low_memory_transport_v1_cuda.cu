#include "../low_memory_transport_v1_utils.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

__global__ void evaluate_fraction_to_boundary(
    const double *C, const double *h, const double *alpha,
    const double *direction, double *out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = ctot_lmt_fraction_to_boundary_local(
            C[i], h[i], alpha[i], direction[i], 1.0, 1.0e-10);
    }
}

int main()
{
    const std::vector<double> C = {0.5, 0.5, 0.2, 0.8, 0.1};
    const std::vector<double> h = {0.2, 0.2, 0.0, 0.0, 1.0};
    const std::vector<double> alpha = {0.8, 0.8, 1.0, 1.0, 0.0};
    const std::vector<double> direction = {0.6, -1.0, 0.2, -0.1, 2.0};
    const int n = static_cast<int>(C.size());
    double *d_C = nullptr, *d_h = nullptr, *d_alpha = nullptr;
    double *d_direction = nullptr, *d_out = nullptr;
    cudaMalloc(&d_C, n * sizeof(double));
    cudaMalloc(&d_h, n * sizeof(double));
    cudaMalloc(&d_alpha, n * sizeof(double));
    cudaMalloc(&d_direction, n * sizeof(double));
    cudaMalloc(&d_out, n * sizeof(double));
    cudaMemcpy(d_C, C.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, alpha.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_direction, direction.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    evaluate_fraction_to_boundary<<<1, 32>>>(
        d_C, d_h, d_alpha, d_direction, d_out, n);
    std::vector<double> gpu(n, 0.0);
    cudaMemcpy(gpu.data(), d_out, n * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        const double host = ctot_lmt_fraction_to_boundary_local(
            C[i], h[i], alpha[i], direction[i], 1.0, 1.0e-10);
        if (std::fabs(host - gpu[i]) > 1.0e-14) return 2;
    }
    cudaFree(d_C);
    cudaFree(d_h);
    cudaFree(d_alpha);
    cudaFree(d_direction);
    cudaFree(d_out);
    std::puts("PASS_LOW_MEMORY_TRANSPORT_V1_CUDA_PARITY");
    return 0;
}
