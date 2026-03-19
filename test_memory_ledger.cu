#include <stdio.h>
#include <stdint.h>
#include <cufft.h>

// 复制 print_memory_ledger 函数
static void print_memory_ledger(
    size_t size_r, size_t size_k, size_t size_r_float, size_t size_k_float,
    size_t scratch_k_bytes, size_t scratch_r_bytes,
    int elastic_enabled, size_t total_r, size_t total_k) {
    size_t resident = 0;
#define LEDGER(name, elem, elem_sz) do { \
    size_t _b = (elem) * (elem_sz); \
    printf("  %-24s %14zu elem  %12zu bytes\n", (name), (size_t)(elem), _b); \
    resident += _b; \
} while(0)
    printf("\n=== Memory Ledger (resident arrays) ===\n");
    LEDGER("d_phi_r", total_r, sizeof(double));
    LEDGER("d_Y_r", total_r, sizeof(double));
    LEDGER("d_xB_r", total_r, sizeof(double));
    LEDGER("d_phi_rhs_r(=d_lapY_r)", total_r, sizeof(double));
    LEDGER("d_phi_n_saved", total_r, sizeof(double));
    LEDGER("d_Y_n_saved", total_r, sizeof(double));
    LEDGER("d_dY_dt_prev_r", total_r, sizeof(double));
    LEDGER("d_mu_x_r(=d_Y_rhs_r)", total_r, sizeof(double));
    LEDGER("d_divJ_r(=d_xB_prev_r)", total_r, sizeof(double));
    LEDGER("d_phi_k", total_k, sizeof(cufftDoubleComplex));
    LEDGER("d_phi_rhs_k(=d_mu_x_k=d_Y_rhs_k)", total_k, sizeof(cufftDoubleComplex));
    LEDGER("d_Y_k", total_k, sizeof(cufftDoubleComplex));
    LEDGER("d_divJ_k", total_k, sizeof(cufftDoubleComplex));
    LEDGER("d_scratch_k_double", scratch_k_bytes / sizeof(cufftDoubleComplex), sizeof(cufftDoubleComplex));
    LEDGER("d_scratch_r_double", scratch_r_bytes / sizeof(double), sizeof(double));
    if (elastic_enabled) {
        // Optimization(4): d_uxx0_r..d_uyz0_r 已移除，eps0 现场计算
        LEDGER("d_uxx_r..d_uyz_r (6)", 6 * total_r, sizeof(float));
        LEDGER("d_sigma_*_r (6)", 6 * total_r, sizeof(float));
        // Optimization(3): d_uxx0_k..d_uyz0_k 已移除，复用 d_uxx_k..d_uyz_k 作为临时 eigenstrain_k
        LEDGER("d_ux_k,d_uy_k,d_uz_k", 3 * total_k, sizeof(cufftComplex));
        LEDGER("d_uxx_k..d_uyz_k (6)", 6 * total_k, sizeof(cufftComplex));
        // d_elastic_tmp_r, d_elastic_tmp_k 已移除，弹性 FFT 直接 out-of-place
    }
    printf("  ----------------------------------------\n");
    printf("  Total resident (est.)            %12zu bytes (%.2f MB) (%.2f GB)\n", 
           resident, resident / (1024.0*1024.0), resident / (1024.0*1024.0*1024.0));
    printf("  Peak estimate (resident+transient) ~ same (scratch reused)\n");
    printf("=== End Memory Ledger ===\n\n");
#undef LEDGER
}

int main(int argc, char **argv) {
    // 测试不同网格尺寸
    int test_cases[][3] = {
        {256, 256, 256},
        {512, 512, 512},
        {600, 600, 600},
        {1024, 1024, 1024}
    };
    
    printf("========================================\n");
    printf("显存账本剖析 - 不同网格尺寸\n");
    printf("========================================\n\n");
    
    for (int i = 0; i < 4; i++) {
        int Nx = test_cases[i][0];
        int Ny = test_cases[i][1];
        int Nz = test_cases[i][2];
        int NzC = Nz / 2 + 1;
        
        size_t total_r = (size_t)Nx * (size_t)Ny * (size_t)Nz;
        size_t total_k = (size_t)Nx * (size_t)Ny * (size_t)NzC;
        
        size_t size_r = total_r * sizeof(double);
        size_t size_k = total_k * sizeof(cufftDoubleComplex);
        size_t size_r_float = total_r * sizeof(float);
        size_t size_k_float = total_k * sizeof(cufftComplex);
        size_t scratch_k_double_bytes = size_k;
        size_t scratch_r_double_bytes = 2 * size_r;
        
        printf("\n【网格尺寸: %dx%dx%d】\n", Nx, Ny, Nz);
        printf("total_r = %zu, total_k = %zu\n", total_r, total_k);
        
        // 仅相场模式
        printf("\n--- 仅相场模式 (elastic_enabled=0) ---\n");
        print_memory_ledger(size_r, size_k, size_r_float, size_k_float,
                           scratch_k_double_bytes, scratch_r_double_bytes,
                           0, total_r, total_k);
        
        // 相场+弹性模式
        printf("\n--- 相场+弹性模式 (elastic_enabled=1) ---\n");
        print_memory_ledger(size_r, size_k, size_r_float, size_k_float,
                           scratch_k_double_bytes, scratch_r_double_bytes,
                           1, total_r, total_k);
    }
    
    return 0;
}
