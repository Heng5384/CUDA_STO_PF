#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            fprintf(stderr, "cuFFT error at %s:%d: error code %d\n", __FILE__, __LINE__, err); \
            exit(1); \
        } \
    } while(0)

// 常数定义
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Optimization: Scratch Arena - 字节级临时分配，按 phase 重置
typedef struct {
    void *base;
    size_t capacity;
    size_t offset;
#ifdef SCRATCH_ARENA_DEBUG
    size_t peak_offset;
    const char *tag;
#endif
} ScratchArena;

void scratch_arena_init(ScratchArena *arena, void *base, size_t capacity, const char *tag);
void *scratch_arena_alloc(ScratchArena *arena, size_t bytes, size_t alignment);
void scratch_arena_reset(ScratchArena *arena);
#ifdef SCRATCH_ARENA_DEBUG
void scratch_arena_assert_ok(const ScratchArena *arena);
#endif

// k空间结构体（CUDA版本）
typedef struct {
    double *d_k2;      // GPU上的k^2数组
    double *d_k4;      // GPU上的k^4数组
    int Nx, Ny, Nz, NzC;
    double dx, dy, dz;
} KSpace_CUDA;

// 初始化k空间（在GPU上）
void kspace_build_cuda(KSpace_CUDA *KS, int Nx, int Ny, int Nz, 
                       double dx, double dy, double dz, int build_k4 = 1);

// 释放k空间资源
void kspace_free_cuda(KSpace_CUDA *KS);

// 辅助函数：计算kx（设备端可用）
__device__ __host__ static inline double kx_wrap(int i, int Nx, double dx) {
    int ii = (i <= Nx/2) ? i : i - Nx;
    return 2.0 * M_PI * ii / (dx * Nx);
}

// 辅助函数：计算ky（设备端可用）
__device__ __host__ static inline double ky_wrap(int j, int Ny, double dy) {
    int jj = (j <= Ny/2) ? j : j - Ny;
    return 2.0 * M_PI * jj / (dy * Ny);
}

// 辅助函数：计算kz（设备端可用）
__device__ __host__ static inline double kz_wrap(int k, int Nz, double dz) {
    return 2.0 * M_PI * k / (dz * Nz);
}

#endif // CUDA_COMMON_H
