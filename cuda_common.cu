#include "cuda_common.h"
#include <cuda_runtime.h>
#include <string.h>
#include <assert.h>

// Optimization: Scratch Arena 实现
void scratch_arena_init(ScratchArena *arena, void *base, size_t capacity, const char *tag) {
    arena->base = base;
    arena->capacity = capacity;
    arena->offset = 0;
#ifdef SCRATCH_ARENA_DEBUG
    arena->peak_offset = 0;
    arena->tag = tag;
#endif
}

void *scratch_arena_alloc(ScratchArena *arena, size_t bytes, size_t alignment) {
    if (bytes == 0) return arena->base;
    size_t mask = alignment - 1;
    size_t aligned = (arena->offset + mask) & ~mask;
    if (aligned + bytes > arena->capacity) {
#ifdef SCRATCH_ARENA_DEBUG
        fprintf(stderr, "[ScratchArena] overflow: %s offset=%zu + bytes=%zu > capacity=%zu\n",
                arena->tag ? arena->tag : "?", arena->offset, bytes, arena->capacity);
        assert(0);
#else
        return NULL;
#endif
    }
    void *ptr = (char *)arena->base + aligned;
    arena->offset = aligned + bytes;
#ifdef SCRATCH_ARENA_DEBUG
    if (arena->offset > arena->peak_offset) arena->peak_offset = arena->offset;
    fprintf(stderr, "[ScratchArena] %s alloc %zu bytes -> ptr %p, offset now %zu\n",
            arena->tag ? arena->tag : "?", bytes, ptr, arena->offset);
#endif
    return ptr;
}

void scratch_arena_reset(ScratchArena *arena) {
    arena->offset = 0;
}

#ifdef SCRATCH_ARENA_DEBUG
void scratch_arena_assert_ok(const ScratchArena *arena) {
    assert(arena->offset <= arena->capacity);
}
#endif

// CUDA kernel: 在GPU上构建k空间数组
__global__ void kspace_build_kernel(double *k2, double *k4, 
                                    int Nx, int Ny, int Nz, int NzC,
                                    double dx, double dy, double dz,
                                    int local_n0, int local_0_start) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = local_n0 * Ny * NzC;
    
    if (idx < total) {
        int i = idx / (Ny * NzC);
        int remainder = idx % (Ny * NzC);
        int j = remainder / NzC;
        int k = remainder % NzC;
        
        int gi = i + local_0_start;
        double kx = kx_wrap(gi, Nx, dx);
        double ky = ky_wrap(j, Ny, dy);
        double kz = kz_wrap(k, Nz, dz);
        
        double k2v = kx * kx + ky * ky + kz * kz;
        k2[idx] = k2v;
        k4[idx] = k2v * k2v;
    }
}

// 初始化k空间（在GPU上）
void kspace_build_cuda(KSpace_CUDA *KS, int Nx, int Ny, int Nz, 
                       double dx, double dy, double dz) {
    KS->Nx = Nx;
    KS->Ny = Ny;
    KS->Nz = Nz;
    KS->dx = dx;
    KS->dy = dy;
    KS->dz = dz;
    KS->NzC = Nz / 2 + 1;
    
    // 对于单GPU版本，local_n0 = Nx, local_0_start = 0
    int local_n0 = Nx;
    int local_0_start = 0;
    
    size_t nloc = (size_t)local_n0 * (size_t)Ny * (size_t)KS->NzC;
    size_t size_k = nloc * sizeof(double);
    
    // 分配GPU内存
    CUDA_CHECK(cudaMalloc(&KS->d_k2, size_k));
    CUDA_CHECK(cudaMalloc(&KS->d_k4, size_k));
    
    // 启动kernel
    int threads_per_block = 256;
    int blocks = (nloc + threads_per_block - 1) / threads_per_block;
    
    kspace_build_kernel<<<blocks, threads_per_block>>>(
        KS->d_k2, KS->d_k4,
        Nx, Ny, Nz, KS->NzC,
        dx, dy, dz,
        local_n0, local_0_start
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

// 释放k空间资源
void kspace_free_cuda(KSpace_CUDA *KS) {
    if (KS->d_k2) {
        CUDA_CHECK(cudaFree(KS->d_k2));
        KS->d_k2 = NULL;
    }
    if (KS->d_k4) {
        CUDA_CHECK(cudaFree(KS->d_k4));
        KS->d_k4 = NULL;
    }
}

