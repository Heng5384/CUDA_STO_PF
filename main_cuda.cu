/**
 * CUDA版本相场模拟主程序
 * 集成所有CUDA kernel，实现完整的相场模拟
 */

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cuComplex.h>

#include "cuda_common.h"
#include "pf_params.h"
#include "phase_functions.h"
#include "thermo_utils.h"
#include "cuda_kernels.h"
#include "io_vtk_cuda.h"

typedef enum {
    VTK_NAME_INIT = 0,
    VTK_NAME_STEP = 1,
    VTK_NAME_FINAL = 2
} VtkNameKind;

// 统一生成 VTK 输出路径（按模式分离）
// - dynamic(mode=0): 输出到 output_dir，文件名不带 case
// - minimize(mode=1): 输出到 case_output_dir，文件名带 case_tag
static void build_case_vtk_path(char *out, size_t out_size,
                                const char *output_dir,
                                const char *case_output_dir,
                                const char *field,
                                VtkNameKind kind,
                                int step,
                                const char *case_tag,
                                int mode) {
    const char *tag = (case_tag && case_tag[0] != '\0') ? case_tag : "case_unknown";
    if (!out || out_size == 0) return;
    if (!field) {
        out[0] = '\0';
        return;
    }

    if (mode == 0) {
        // dynamic: no case suffix, use output_dir
        if (!output_dir) {
            out[0] = '\0';
            return;
        }
        if (kind == VTK_NAME_INIT) {
            snprintf(out, out_size, "%s/%s_init.vtk", output_dir, field);
        } else if (kind == VTK_NAME_FINAL) {
            snprintf(out, out_size, "%s/%s_final.vtk", output_dir, field);
        } else {
            snprintf(out, out_size, "%s/%s_%d.vtk", output_dir, field, step);
        }
    } else {
        // minimize: with case suffix, use case_output_dir
        if (!case_output_dir) {
            out[0] = '\0';
            return;
        }
        if (kind == VTK_NAME_INIT) {
            snprintf(out, out_size, "%s/%s_init_%s.vtk", case_output_dir, field, tag);
        } else if (kind == VTK_NAME_FINAL) {
            snprintf(out, out_size, "%s/%s_final_%s.vtk", case_output_dir, field, tag);
        } else {
            snprintf(out, out_size, "%s/%s_%d_%s.vtk", case_output_dir, field, step, tag);
        }
    }
}

static double wall_time_sec_monotonic(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0.0;
    }
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void format_hms(double seconds, int *hh, int *mm, int *ss) {
    if (seconds < 0) seconds = 0;
    long s = (long)(seconds + 0.5);
    *hh = (int)(s / 3600);
    *mm = (int)((s % 3600) / 60);
    *ss = (int)(s % 60);
}

static const char *log_enabled_cn(int enabled) {
    return enabled ? "启用" : "禁用";
}

static void log_section_header(const char *title) {
    printf("\n========== %s ==========\n", title ? title : "Section");
}

static void log_kv_text(const char *key, const char *fmt, ...) {
    char value[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(value, sizeof(value), fmt, ap);
    va_end(ap);
    printf("  %-24s : %s\n", key ? key : "item", value);
}

typedef struct {
    int have_dx, have_dy, have_dz;
    int have_temperature_C, have_mu_reference_scale;
    int have_W, have_kappa_phi, have_L_phi;
    int have_D_alpha, have_D_compound;
    int have_v_A, have_v_B;
    int have_Vm_compound, have_Vm_alpha_0, have_dVm_alpha_dxB;
    int have_ic_vf_init_phi, have_ic_vf_target_phi, have_ic_phi_iface_w;
    int have_gamma_Jm2, have_lambda_sm_m, have_Vm_alpha_0_phys_m3mol;
    int have_elastic_shift_dimless, have_eps_iso_over_vB;
    uint32_t eigen_mask;
    uint32_t s_mask;
    uint32_t sp_mask;
} PFParamOverridePresence;

#define PF_VOIGT21_FULL_MASK ((uint32_t)((1u << 21) - 1u))
#define PF_EIGEN_FULL_MASK   ((uint32_t)((1u << 6) - 1u))

static int voigt21_suffix_index(const char *suffix) {
    static const char *const suffixes[21] = {
        "11", "12", "13", "14", "15", "16",
        "22", "23", "24", "25", "26",
        "33", "34", "35", "36",
        "44", "45", "46",
        "55", "56",
        "66"
    };
    if (!suffix) return -1;
    for (int i = 0; i < 21; ++i) {
        if (strcmp(suffix, suffixes[i]) == 0) return i;
    }
    return -1;
}

static void mark_pfparams_presence(PFParamOverridePresence *presence, const char *key) {
    if (!presence || !key) return;
    if (strcmp(key, "dx") == 0) { presence->have_dx = 1; return; }
    if (strcmp(key, "dy") == 0) { presence->have_dy = 1; return; }
    if (strcmp(key, "dz") == 0) { presence->have_dz = 1; return; }
    if (strcmp(key, "temperature_C") == 0) { presence->have_temperature_C = 1; return; }
    if (strcmp(key, "mu_reference_scale") == 0) { presence->have_mu_reference_scale = 1; return; }
    if (strcmp(key, "W") == 0) { presence->have_W = 1; return; }
    if (strcmp(key, "kappa_phi") == 0) { presence->have_kappa_phi = 1; return; }
    if (strcmp(key, "L_phi") == 0) { presence->have_L_phi = 1; return; }
    if (strcmp(key, "D_alpha") == 0) { presence->have_D_alpha = 1; return; }
    if (strcmp(key, "D_compound") == 0) { presence->have_D_compound = 1; return; }
    if (strcmp(key, "v_A") == 0) { presence->have_v_A = 1; return; }
    if (strcmp(key, "v_B") == 0) { presence->have_v_B = 1; return; }
    if (strcmp(key, "Vm_compound") == 0) { presence->have_Vm_compound = 1; return; }
    if (strcmp(key, "Vm_alpha_0") == 0) { presence->have_Vm_alpha_0 = 1; return; }
    if (strcmp(key, "dVm_alpha_dxB") == 0) { presence->have_dVm_alpha_dxB = 1; return; }
    if (strcmp(key, "ic_vf_init_phi") == 0) { presence->have_ic_vf_init_phi = 1; return; }
    if (strcmp(key, "ic_vf_target_phi") == 0) { presence->have_ic_vf_target_phi = 1; return; }
    if (strcmp(key, "ic_phi_iface_w") == 0) { presence->have_ic_phi_iface_w = 1; return; }
    if (strcmp(key, "gamma_Jm2") == 0) { presence->have_gamma_Jm2 = 1; return; }
    if (strcmp(key, "lambda_sm_m") == 0) { presence->have_lambda_sm_m = 1; return; }
    if (strcmp(key, "Vm_alpha_0_phys_m3mol") == 0) { presence->have_Vm_alpha_0_phys_m3mol = 1; return; }
    if (strcmp(key, "elastic_shift_dimless") == 0) { presence->have_elastic_shift_dimless = 1; return; }
    if (strcmp(key, "eps_iso_over_vB") == 0) { presence->have_eps_iso_over_vB = 1; return; }

    if (strcmp(key, "eps_xx00") == 0) { presence->eigen_mask |= (1u << 0); return; }
    if (strcmp(key, "eps_yy00") == 0) { presence->eigen_mask |= (1u << 1); return; }
    if (strcmp(key, "eps_zz00") == 0) { presence->eigen_mask |= (1u << 2); return; }
    if (strcmp(key, "eps_yz00") == 0) { presence->eigen_mask |= (1u << 3); return; }
    if (strcmp(key, "eps_xz00") == 0) { presence->eigen_mask |= (1u << 4); return; }
    if (strcmp(key, "eps_xy00") == 0) { presence->eigen_mask |= (1u << 5); return; }

    if (strncmp(key, "S_p_", 4) == 0) {
        int idx = voigt21_suffix_index(key + 4);
        if (idx >= 0) presence->sp_mask |= (1u << idx);
        return;
    }
    if (strncmp(key, "S_", 2) == 0) {
        int idx = voigt21_suffix_index(key + 2);
        if (idx >= 0) presence->s_mask |= (1u << idx);
        return;
    }
}

static int validate_required_pfparams_presence(const PFParamOverridePresence *presence, const char *path) {
    int ok = 1;
#define REQUIRE_ONE(field, label) \
    do { \
        if (!(presence)->field) { \
            fprintf(stderr, "[fatal] %s 缺少必需物理参数: %s\n", (path ? path : "<pf-param-file>"), (label)); \
            ok = 0; \
        } \
    } while (0)

    REQUIRE_ONE(have_dx, "dx");
    REQUIRE_ONE(have_dy, "dy");
    REQUIRE_ONE(have_dz, "dz");
    REQUIRE_ONE(have_temperature_C, "temperature_C");
    REQUIRE_ONE(have_mu_reference_scale, "mu_reference_scale");
    REQUIRE_ONE(have_W, "W");
    REQUIRE_ONE(have_kappa_phi, "kappa_phi");
    REQUIRE_ONE(have_L_phi, "L_phi");
    REQUIRE_ONE(have_D_alpha, "D_alpha");
    REQUIRE_ONE(have_D_compound, "D_compound");
    REQUIRE_ONE(have_v_A, "v_A");
    REQUIRE_ONE(have_v_B, "v_B");
    REQUIRE_ONE(have_Vm_compound, "Vm_compound");
    REQUIRE_ONE(have_Vm_alpha_0, "Vm_alpha_0");
    REQUIRE_ONE(have_dVm_alpha_dxB, "dVm_alpha_dxB");
    REQUIRE_ONE(have_ic_vf_init_phi, "ic_vf_init_phi");
    REQUIRE_ONE(have_ic_vf_target_phi, "ic_vf_target_phi");
    REQUIRE_ONE(have_ic_phi_iface_w, "ic_phi_iface_w");
    REQUIRE_ONE(have_gamma_Jm2, "gamma_Jm2");
    REQUIRE_ONE(have_lambda_sm_m, "lambda_sm_m");
    REQUIRE_ONE(have_Vm_alpha_0_phys_m3mol, "Vm_alpha_0_phys_m3mol");
    REQUIRE_ONE(have_elastic_shift_dimless, "elastic_shift_dimless");
    REQUIRE_ONE(have_eps_iso_over_vB, "eps_iso_over_vB");

#undef REQUIRE_ONE

    if (presence->eigen_mask != PF_EIGEN_FULL_MASK) {
        fprintf(stderr, "[fatal] %s 缺少完整的本征应变输入 eps_xx00..eps_xy00\n",
                (path ? path : "<pf-param-file>"));
        ok = 0;
    }
    if (presence->s_mask != PF_VOIGT21_FULL_MASK) {
        fprintf(stderr, "[fatal] %s 缺少完整的基体弹性矩阵输入 S_*\n",
                (path ? path : "<pf-param-file>"));
        ok = 0;
    }
    if (presence->sp_mask != PF_VOIGT21_FULL_MASK) {
        fprintf(stderr, "[fatal] %s 缺少完整的弹性差值矩阵输入 S_p_*\n",
                (path ? path : "<pf-param-file>"));
        ok = 0;
    }
    return ok;
}

static int validate_physical_params_ready(const PFParams *P) {
    int ok = 1;
#define REQUIRE_POSITIVE(field, label) \
    do { \
        if (!((P)->field > 0.0)) { \
            fprintf(stderr, "[fatal] 无效物理参数 %s = %.6e，必须 > 0。\n", (label), (double)(P)->field); \
            ok = 0; \
        } \
    } while (0)
#define REQUIRE_NONNEG(field, label) \
    do { \
        if (!((P)->field >= 0.0)) { \
            fprintf(stderr, "[fatal] 无效物理参数 %s = %.6e，必须 >= 0。\n", (label), (double)(P)->field); \
            ok = 0; \
        } \
    } while (0)

    REQUIRE_POSITIVE(dx, "dx");
    REQUIRE_POSITIVE(dy, "dy");
    REQUIRE_POSITIVE(dz, "dz");
    REQUIRE_POSITIVE(W, "W");
    REQUIRE_POSITIVE(kappa_phi, "kappa_phi");
    REQUIRE_POSITIVE(L_phi, "L_phi");
    REQUIRE_POSITIVE(D_alpha, "D_alpha");
    REQUIRE_NONNEG(D_compound, "D_compound");
    if (!(P->temperature_C > -273.15)) {
        fprintf(stderr, "[fatal] 无效物理参数 temperature_C = %.6e，必须 > -273.15。\n", P->temperature_C);
        ok = 0;
    }
    if (fabs(P->mu_reference_scale) <= 1e-30) {
        fprintf(stderr, "[fatal] 无效物理参数 mu_reference_scale = %.6e，不能为 0。\n", P->mu_reference_scale);
        ok = 0;
    }
    REQUIRE_POSITIVE(v_B, "v_B");
    REQUIRE_POSITIVE(Vm_compound, "Vm_compound");
    REQUIRE_POSITIVE(Vm_alpha_0, "Vm_alpha_0");
    REQUIRE_NONNEG(ic_vf_init_phi, "ic_vf_init_phi");
    REQUIRE_NONNEG(ic_vf_target_phi, "ic_vf_target_phi");
    REQUIRE_POSITIVE(ic_phi_iface_w, "ic_phi_iface_w");
    REQUIRE_POSITIVE(gamma_Jm2, "gamma_Jm2");
    REQUIRE_POSITIVE(lambda_sm_m, "lambda_sm_m");
    REQUIRE_POSITIVE(Vm_alpha_0_phys_m3mol, "Vm_alpha_0_phys_m3mol");
    REQUIRE_NONNEG(eps_iso_over_vB, "eps_iso_over_vB");

#undef REQUIRE_POSITIVE
#undef REQUIRE_NONNEG
    return ok;
}

// 辅助函数：周期边界距离（GPU/CPU两用）
__host__ __device__ static inline double periodic_delta(double coord, double center, double period) {
    double d = coord - center;
    d -= nearbyint(d / period) * period;
    return d;
}

// 辅助函数：限制值在[0,1]（GPU/CPU两用）
__host__ __device__ static inline double clamp01_local(double v) { 
    return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); 
}

// 辅助函数：限制xB值（GPU/CPU两用）
__host__ __device__ static inline double clamp_eps(double v, double eps) { 
    if (v < eps) return eps; 
    if (v > 1.0 - eps) return 1.0 - eps; 
    return v; 
}

// 在 cuda_kernels.cu 中实现：将场按 h(phi) 加权
void launch_scale_field_by_h_kernel(const double *phi_r,
                                    double *field_r,
                                    int total_size);

// Optimization: 显存账本 - 启动时打印常驻数组、scratch 容量、总 bytes
static void print_memory_ledger(
    size_t size_r, size_t size_k, size_t size_r_float, size_t size_k_float,
    size_t scratch_k_bytes, size_t scratch_r_bytes,
    int elastic_enabled, int total_r, int total_k) {
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
    // Optimization: scratch 数组通过 LEDGER 宏记录一次，不再单独打印（避免重复计算）
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
    printf("  Total resident (est.)            %12zu bytes (%.2f MB)\n", resident, resident / (1024.0*1024.0));
    printf("  Peak estimate (resident+transient) ~ same (scratch reused)\n");
    printf("=== End Memory Ledger ===\n\n");
#undef LEDGER
}

// 打印GPU显存信息（便于定位 OOM/泄漏）
static inline void print_gpu_meminfo(const char *tag) {
    size_t free_b = 0, total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        printf("[gpu mem] %s: cudaMemGetInfo failed: %s\n", tag, cudaGetErrorString(err));
        return;
    }
    double free_gb = (double)free_b / (1024.0 * 1024.0 * 1024.0);
    double total_gb = (double)total_b / (1024.0 * 1024.0 * 1024.0);
    double used_gb = total_gb - free_gb;
    printf("[gpu mem] %s: free=%.2f GB / total=%.2f GB (used=%.2f GB)\n", tag, free_gb, total_gb, used_gb);
}

__global__ void square_values_kernel(const double *in, double *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double v = in[idx];
    out[idx] = v * v;
}

// (phi_new - phi_old)^2 for rms_dphi = ||φ^{n+1}-φ^n||_rms / dt
__global__ void diff_sq_kernel(const double *phi_r, const double *phi_old_r, double *out_r, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double d = phi_r[idx] - phi_old_r[idx];
    out_r[idx] = d * d;
}

// GPU kernel：根据种子中心生成初始 phi（支持椭球形/旋转椭球形种子）
__global__ void init_phi_kernel(double *phi_r,
                                const double *centers,
                                const double *rot_mats, // 行优先 3x3 矩阵数组，每个种子一个；可为 NULL
                                int use_rotation,       // =0: 不旋转（退化为轴对齐椭球）；=1: 使用 rot_mats
                                int N_seeds,
                                int Nx, int Ny, int Nz,
                                double dx, double dy, double dz,
                                double Lx, double Ly, double Lz,
                                double w, 
                                double Rx, double Ry, double Rz,
                                int is3D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;
    if (idx >= total) return;

    int i = idx / (Ny * Nz);
    int rem = idx - i * Ny * Nz;
    int j = rem / Nz;
    int k = rem - j * Nz;

    double x = i * dx;
    double y = j * dy;
    double z = k * dz;

    // 寻找最小轴长以保持界面宽度物理一致性
    double min_axis = Rx;
    if (is3D && Ry < min_axis) min_axis = Ry;
    if (Rz < min_axis) min_axis = Rz;

    double max_phi = 0.0;
    for (int s = 0; s < N_seeds; s++) {
        double cx = centers[3 * s + 0];
        double cy = centers[3 * s + 1];
        double cz = centers[3 * s + 2];

        double rx = periodic_delta(x, cx, Lx);
        double rz = periodic_delta(z, cz, Lz);

        double ry = 0.0;
        if (is3D) {
            ry = periodic_delta(y, cy, Ly);
        }

        double qx = rx, qy = ry, qz = rz;
        if (use_rotation && rot_mats != NULL) {
            const double *R = &rot_mats[9 * s];
            // R 为行优先，行向量为 e1,e2,e3；此处使用 q = R^T * r
            // 即 q_i = e_i · r，其中 e_i 是第 i 行
            qx = R[0] * rx + R[1] * ry + R[2] * rz;
            qy = R[3] * rx + R[4] * ry + R[5] * rz;
            qz = R[6] * rx + R[7] * ry + R[8] * rz;
        }

        double r_eff_sq;
        if (is3D) {
            r_eff_sq = (qx*qx)/(Rx*Rx) + (qy*qy)/(Ry*Ry) + (qz*qz)/(Rz*Rz);
        } else {
            // 2D 情形：仅在 x-z 平面内使用旋转（ry 恒为 0）
            r_eff_sq = (qx*qx)/(Rx*Rx) + (qz*qz)/(Rz*Rz);
        }
        double r_eff = sqrt(r_eff_sq);

        // 椭球界面函数：r_eff=1 处 phi=0.5
        double seed = 0.5 * (1.0 + tanh((1.0 - r_eff) * (min_axis / w)));
        if (seed > max_phi) max_phi = seed;
    }
    phi_r[idx] = clamp01_local(max_phi);
}

// GPU kernel：根据 phi 计算 xB 与 Y（可选）
__global__ void init_xB_Y_from_phi_kernel(const double *phi_r,
                                          double *xB_r,
                                          double *Y_r,
                                          double xB_out,
                                          double xB_eq,
                                          double xB_eps,
                                          double Y_clip,
                                          int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double h   = h_of_phi(clamp01_local(phi_r[idx]));
    double xB  = (1.0 - h) * xB_out + h * xB_eq;
    // double xB  = (1.0 - h) * xB_out + h * xB_out;
    xB         = clamp_eps(xB, xB_eps);
    xB_r[idx]  = xB;

    if (Y_r) {
        double Y = log(xB / (1.0 - xB));
        if (Y >  Y_clip) Y =  Y_clip;
        if (Y < -Y_clip) Y = -Y_clip;
        Y_r[idx] = Y;
    }
}

// 生成随机种子中心（与CPU版本一致的逻辑）
static void generate_seed_centers_uniform(const PFParams *P, double *buffer) {
    const int n = P->ic_phi_num_seeds;
    if (n <= 0) return;
    
    const double Lx = P->Nx * P->dx;
    const double Ly = P->Ny * P->dy;
    const double Lz = P->Nz * P->dz;
    
    // 计算边界缓冲区（半径/界面宽度统一为物理长度，随 dx 缩放）
    double w_phys = (P->ic_phi_iface_w > 0.0) ? P->ic_phi_iface_w * P->dx : P->dx;
    double R_estimate = (P->ic_phi_seed_radius > 0.0) ? P->ic_phi_seed_radius : (10.0 * P->dx);
    double margin = R_estimate + 2.0 * w_phys;
    
    double margin_x = fmin(margin, 0.1 * Lx);
    double margin_y = fmin(margin, 0.1 * Ly);
    double margin_z = fmin(margin, 0.1 * Lz);
    
    // 使用种子值初始化随机数生成器
    srand((unsigned int)P->seed);
    
    // 单个种子：放在几何中心
    if (n == 1) {
        buffer[0] = 0.5 * Lx;
        buffer[1] = 0.5 * Ly;
        buffer[2] = 0.5 * Lz;
    } else {
        // 多个种子：随机分布，避开边界
        if (P->Ny > 2) {
            // 3D: 在 [margin, L-margin] 范围内均匀随机布点
            for (int s = 0; s < n; ++s) {
                double rx = (double)rand() / (double)RAND_MAX;
                double ry = (double)rand() / (double)RAND_MAX;
                double rz = (double)rand() / (double)RAND_MAX;
                buffer[3*s + 0] = margin_x + rx * (Lx - 2.0 * margin_x);
                buffer[3*s + 1] = margin_y + ry * (Ly - 2.0 * margin_y);
                buffer[3*s + 2] = margin_z + rz * (Lz - 2.0 * margin_z);
            }
        } else {
            // 2D: 仅在 X-Z 平面上随机布点，Y 固定在中间层
            double cy = 0.5 * Ly;
            for (int s = 0; s < n; ++s) {
                double rx = (double)rand() / (double)RAND_MAX;
                double rz = (double)rand() / (double)RAND_MAX;
                buffer[3*s + 0] = margin_x + rx * (Lx - 2.0 * margin_x);
                buffer[3*s + 1] = cy;
                buffer[3*s + 2] = margin_z + rz * (Lz - 2.0 * margin_z);
            }
        }
    }
}

// 对 phi 场叠加均匀随机噪声：phi <- clamp01(phi + eta), eta∈[-amp,amp]
__global__ void apply_phi_noise_kernel(double *phi_r,
                                       int total_size,
                                       double amp,
                                       unsigned long long seed_base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    if (amp <= 0.0) return;

    // 简单可重复的哈希生成器：基于 seed_base 和 idx
    unsigned long long x = seed_base ^ (unsigned long long)(idx + 1) * 2862933555777941757ULL;
    x ^= (x >> 33);
    x *= 0xff51afd7ed558ccdULL;
    x ^= (x >> 33);
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= (x >> 33);
    // 映射到 (0,1)
    double u = (double)(x & 0xFFFFFFFFULL) / (double)0xFFFFFFFFULL;
    double eta = (u * 2.0 - 1.0) * amp;

    double v = phi_r[idx] + eta;
    phi_r[idx] = clamp01_local(v);
}

// forward decl: used in initialize_fields_cuda / initialize_phi_only_cuda
static void build_seed_rotation_from_normal(double theta_deg, double phi_deg, double R[9]);

static void compute_seed_axes_from_radius(const PFParams *P, double R, int is3D,
                                          double *Rx, double *Ry, double *Rz) {
    if (P->init_shape_mode < 0 || P->init_shape_mode == 0) {
        *Rx = R;
        *Ry = R;
        *Rz = R;
        return;
    }

    double arx = (P->init_axis_ratio_rx > 0.0) ? P->init_axis_ratio_rx : 1.0;
    double ary = (P->init_axis_ratio_ry > 0.0) ? P->init_axis_ratio_ry : 1.0;
    double arz = (P->init_axis_ratio_rz > 0.0) ? P->init_axis_ratio_rz : 1.0;

    if (is3D) {
        double scale = cbrt(arx * ary * arz);
        if (!(scale > 0.0)) scale = 1.0;
        *Rx = R * arx / scale;
        *Ry = R * ary / scale;
        *Rz = R * arz / scale;
    } else {
        // 2D(x-z) 初始化时仅使用 Rx/Rz；归一化以保持与半径 R 对应的等面积。
        double scale = sqrt(arx * arz);
        if (!(scale > 0.0)) scale = 1.0;
        *Rx = R * arx / scale;
        *Ry = R;
        *Rz = R * arz / scale;
    }
}

// 初始化 phi / xB / Y （弥散界面质量严格守恒版本）
void initialize_fields_cuda(double *phi_r, double *Y_r, double *xB_r, const PFParams *P, int total_size){
    const int Nx = P->Nx, Ny = P->Ny, Nz = P->Nz;
    const double dx = P->dx, dy = P->dy, dz = P->dz;
    const double Lx = Nx * dx, Ly = Ny * dy, Lz = Nz * dz;

    // ========= A) 决定球半径 R（与原来保持一致，用 ic_vf_init_phi 估计） =========
    double vf_init_target = fmin(0.999, fmax(1e-6, P->ic_vf_init_phi));
    int    N_seeds        = (P->ic_phi_num_seeds > 0) ? P->ic_phi_num_seeds : 1;
    int    is3D           = (Ny > 2);

    // w_phys: 物理界面宽度；R 物理半径（米制/与 dx 同单位）
    double w = (P->ic_phi_iface_w > 0.0) ? (P->ic_phi_iface_w * dx) : dx;
    double R;

    if (P->ic_phi_seed_radius > 0.0) {
        // 用户输入的半径视为物理长度；网格占用约为 R/dx
        R = P->ic_phi_seed_radius;
    } else {
        double Rtar;
        if (is3D) {
            // 3D: N*(4/3)πR^3 / (LxLyLz) = vf_init_target
            Rtar = cbrt((vf_init_target * Lx * Ly * Lz) /
                        (N_seeds * (4.0 * M_PI / 3.0)));
        } else {
            // 2D (x-z 面): N*πR^2 / (LxLz) = vf_init_target
            Rtar = sqrt((vf_init_target * Lx * Lz) /
                        (N_seeds * M_PI));
        }
        R = fmax(1.0, Rtar);
    }

    // ========= B) 决定椭球轴长 Rx, Ry, Rz =========
    // 约定：命令行/脚本里的 radius 始终表示等体积(3D)或等面积(2D)球半径。
    double Rx, Ry, Rz;
    compute_seed_axes_from_radius(P, R, is3D, &Rx, &Ry, &Rz);

    // ========= C) 生成种子中心（host），后续拷贝到 GPU =========
    double stack_centers[1024];
    double *centers = NULL;
    if (3 * N_seeds <= (int)(sizeof(stack_centers) / sizeof(double))) {
        centers = stack_centers;
    } else {
        centers = (double*)malloc((size_t)3 * N_seeds * sizeof(double));
    }
    generate_seed_centers_uniform(P, centers); // 你原来的函数

    // 对所有种子中心施加用户指定的小偏移（物理单位，与 dx 同）
    if (P->init_center_shift_x != 0.0 ||
        P->init_center_shift_y != 0.0 ||
        P->init_center_shift_z != 0.0) {
        for (int s = 0; s < N_seeds; ++s) {
            centers[3*s + 0] += P->init_center_shift_x;
            centers[3*s + 1] += P->init_center_shift_y;
            centers[3*s + 2] += P->init_center_shift_z;
            // 简单周期包裹，保证仍在 [0,L) 内
            if (centers[3*s + 0] < 0.0) centers[3*s + 0] += Lx;
            if (centers[3*s + 0] >= Lx) centers[3*s + 0] -= Lx;
            if (centers[3*s + 1] < 0.0) centers[3*s + 1] += Ly;
            if (centers[3*s + 1] >= Ly) centers[3*s + 1] -= Ly;
            if (centers[3*s + 2] < 0.0) centers[3*s + 2] += Lz;
            if (centers[3*s + 2] >= Lz) centers[3*s + 2] -= Lz;
        }
    }

    // 为每个种子构造旋转矩阵（若启用旋转，否则使用单位阵）
    double stack_rot[9 * 16]; // 足够容纳 N_seeds<=16 的常见情况
    double *rot_mats = NULL;
    if (P->init_use_rotation && (P->init_tilt_theta_deg != 0.0 || P->init_tilt_phi_deg != 0.0)) {
        if (9 * N_seeds <= (int)(sizeof(stack_rot) / sizeof(double))) {
            rot_mats = stack_rot;
        } else {
            rot_mats = (double*)malloc((size_t)9 * N_seeds * sizeof(double));
        }
        double R_seed[9];
        build_seed_rotation_from_normal(P->init_tilt_theta_deg, P->init_tilt_phi_deg, R_seed);
        for (int s = 0; s < N_seeds; ++s) {
            for (int k = 0; k < 9; ++k) {
                rot_mats[9*s + k] = R_seed[k];
            }
        }
    }

    // ========= C) GPU 分配临时显存并初始化 phi =========
    double *d_centers = NULL;
    double *d_rot_mats = NULL;
    double *d_phi_tmp = NULL;
    double *d_xB_tmp  = NULL;
    double *d_Y_tmp   = NULL;
    size_t centers_bytes = (size_t)3 * N_seeds * sizeof(double);
    size_t size_r = (size_t)total_size * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_centers, centers_bytes));
    CUDA_CHECK(cudaMemcpy(d_centers, centers, centers_bytes, cudaMemcpyHostToDevice));

    if (rot_mats) {
        size_t rot_bytes = (size_t)9 * N_seeds * sizeof(double);
        CUDA_CHECK(cudaMalloc(&d_rot_mats, rot_bytes));
        CUDA_CHECK(cudaMemcpy(d_rot_mats, rot_mats, rot_bytes, cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMalloc(&d_phi_tmp, size_r));
    CUDA_CHECK(cudaMalloc(&d_xB_tmp,  size_r));
    if (Y_r) {
        CUDA_CHECK(cudaMalloc(&d_Y_tmp, size_r));
    }

    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    init_phi_kernel<<<blocks, threads>>>(d_phi_tmp, d_centers,
                                         d_rot_mats, (P->init_use_rotation ? 1 : 0),
                                         N_seeds,
                                         Nx, Ny, Nz, dx, dy, dz,
                                         Lx, Ly, Lz, w,
                                         Rx, Ry, Rz, is3D);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 在 GPU 端对 phi 施加随机噪声（若启用）
    if (P->init_phi_noise_amp > 0.0) {
        unsigned long long seed_base =
            (P->init_phi_noise_seed != 0 ? (unsigned long long)P->init_phi_noise_seed
                                         : (unsigned long long)P->seed);
        apply_phi_noise_kernel<<<blocks, threads>>>(d_phi_tmp, total_size,
                                                    P->init_phi_noise_amp, seed_base);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 将 phi 拷回 CPU，用于质量守恒计算
    CUDA_CHECK(cudaMemcpy(phi_r, d_phi_tmp, size_r, cudaMemcpyDeviceToHost));

    double first_center_log[3] = {0.0, 0.0, 0.0};
    int has_first_center_log = (N_seeds > 0);
    if (has_first_center_log) {
        first_center_log[0] = centers[0];
        first_center_log[1] = centers[1];
        first_center_log[2] = centers[2];
    }
    double R_seed_log[9] = {0.0};
    int has_seed_rotation_log = 0;
    if (P->init_use_rotation && (P->init_tilt_theta_deg != 0.0 || P->init_tilt_phi_deg != 0.0)) {
        build_seed_rotation_from_normal(P->init_tilt_theta_deg, P->init_tilt_phi_deg, R_seed_log);
        has_seed_rotation_log = 1;
    }

    if (centers != stack_centers && centers != NULL) {
        free(centers);
    }
    if (rot_mats && rot_mats != stack_rot) {
        free(rot_mats);
    }

    const double Ntot = (double)Nx * (double)Ny * (double)Nz;

    const char *shape_str = "legacy";
    if (P->init_shape_mode == 0) shape_str = "sphere";
    else if (P->init_shape_mode == 1) shape_str = "ellipsoid";

    // ========= D1) 计算 <h> 和 <h^2>，用 h(φ) 来定义真正的体积分数 =========
    double sum_h  = 0.0;
    double sum_h2 = 0.0;
    for (int i = 0; i < total_size; i++) {
        double h = h_of_phi(phi_r[i]);  // *** 关键：统一用 h(φ) ***
        sum_h  += h;
        sum_h2 += h*h;
    }
    double mean_h   = sum_h  / Ntot;  // = vf_init_eff (真正的初始析出体积分数)
    double mean_h2  = sum_h2 / Ntot;

    // ========= D2) 按 diffuse-interface 质量守恒求 xB_out =========
    double xB_eq     = P->ic_xB_eq_matrix;          // α 相平衡 B 含量
    double vf_target = fmin(0.999, fmax(1e-6, P->ic_vf_target_phi));
    double vB_frac   = P->v_B;                     // 化合物相 B 含量（摩尔分数）
    double xB_out;

    // 目标总 B 含量（lever rule）
    double xBtot_target = vB_frac * vf_target + (1.0 - vf_target) * xB_eq;

    // 由 <xB_tot> = A xB_out + B xB_eq + C 解得 xB_out
    // A = <(1-h)^2> = 1 - 2<h> + <h^2>
    // B = <h(1-h)>  = <h> - <h^2>
    // C = v_B <h>
    double A = 1.0 - 2.0 * mean_h + mean_h2;
    double B = mean_h - mean_h2;
    double C = vB_frac * mean_h;

    if (P->ic_23d_xB_out > 0.0) {
        // 用户强制指定 xB_out：只做 clamp 和打印
        xB_out = clamp_eps(P->ic_23d_xB_out, P->xB_eps);
    } else {
        if (fabs(A) < 1e-12) {
            // 极端情况：几乎全部是 0 相或 1 相，用退化公式兜底
            // 退化：假设 <h^2> ≈ <h>，回到 sharp-interface 近似
            double denom = 1.0 - mean_h;
            if (denom < 1e-12) denom = 1e-12;
            xB_out = ((vf_target - mean_h) * vB_frac +
                      (1.0 - vf_target) * xB_eq) / denom;
        } else {
            xB_out = (xBtot_target - B * xB_eq - C) / A;
        }
        xB_out = clamp_eps(xB_out, P->xB_eps);
    }

    // ========= D3) GPU 上初始化 xB 和 Y，然后拷回 CPU =========
    init_xB_Y_from_phi_kernel<<<blocks, threads>>>(d_phi_tmp,
                                                   d_xB_tmp,
                                                   d_Y_tmp,
                                                   xB_out,
                                                   xB_eq,
                                                   P->xB_eps,
                                                   P->Y_clip,
                                                   total_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(xB_r, d_xB_tmp, size_r, cudaMemcpyDeviceToHost));
    if (Y_r && d_Y_tmp) {
        CUDA_CHECK(cudaMemcpy(Y_r, d_Y_tmp, size_r, cudaMemcpyDeviceToHost));
    }

    // 释放临时 GPU 显存
    if (d_Y_tmp) CUDA_CHECK(cudaFree(d_Y_tmp));
    CUDA_CHECK(cudaFree(d_xB_tmp));
    CUDA_CHECK(cudaFree(d_phi_tmp));
    if (d_rot_mats) CUDA_CHECK(cudaFree(d_rot_mats));
    CUDA_CHECK(cudaFree(d_centers));

    // ========= D4) 再检查一次实际的 <xB> 和 <xB_tot> =========
    double sum_xB    = 0.0;
    double sum_xBtot = 0.0;
    for (int i = 0; i < total_size; i++) {
        double h  = h_of_phi(phi_r[i]);
        double xB = xB_r[i];
        sum_xB    += xB;
        sum_xBtot += (1.0 - h) * xB + vB_frac * h;
    }
    double mean_xB     = sum_xB    / Ntot;
    double mean_xB_tot = sum_xBtot / Ntot;

    double theory_xB_tot;
    if (P->ic_23d_xB_out > 0.0) {
        // 用户指定 xB_out 时，对应的理论总 B 含量（用于打印对比）
        theory_xB_tot = vB_frac * mean_h + (1.0 - mean_h) * xB_out;
    } else {
        // 你原先设定的“设计目标”
        theory_xB_tot = xBtot_target;
    }

    log_section_header("Initialization Geometry");
    log_kv_text("shape", "%s", shape_str);
    log_kv_text("R_input", "%.6f", P->ic_phi_seed_radius);
    log_kv_text("Rx / Ry / Rz", "%.6f / %.6f / %.6f", Rx, Ry, Rz);
    log_kv_text("R_equiv", "%.6f", is3D ? cbrt(Rx * Ry * Rz) : sqrt(Rx * Rz));
    log_kv_text("tilt(theta,phi) [deg]", "%.3f / %.3f", P->init_tilt_theta_deg, P->init_tilt_phi_deg);
    log_kv_text("rotation", "%s", P->init_use_rotation ? "enabled" : "disabled");
    log_kv_text("center_shift", "(%.6f, %.6f, %.6f)", P->init_center_shift_x, P->init_center_shift_y, P->init_center_shift_z);
    log_kv_text("phi_noise_amp", "%.3e", P->init_phi_noise_amp);
    log_kv_text("phi_noise_seed", "%lu", P->init_phi_noise_seed);
    log_kv_text("init_test_id", "%d", P->init_test_id);
    if (has_first_center_log) {
        log_kv_text("first_seed_center", "(%.6f, %.6f, %.6f)",
                    first_center_log[0], first_center_log[1], first_center_log[2]);
    }
    if (has_seed_rotation_log) {
        log_kv_text("rotation_row1", "(%.6f, %.6f, %.6f)", R_seed_log[0], R_seed_log[1], R_seed_log[2]);
        log_kv_text("rotation_row2", "(%.6f, %.6f, %.6f)", R_seed_log[3], R_seed_log[4], R_seed_log[5]);
        log_kv_text("rotation_row3", "(%.6f, %.6f, %.6f)", R_seed_log[6], R_seed_log[7], R_seed_log[8]);
    }

    log_section_header("Initialization Composition");
    log_kv_text("vf_init(set)", "%.6f", P->ic_vf_init_phi);
    log_kv_text("vf_init_eff=<h>", "%.6f", mean_h);
    log_kv_text("<h^2>", "%.6f", mean_h2);
    log_kv_text("vf_target", "%.6f", vf_target);
    log_kv_text("vB_frac", "%.6f", vB_frac);
    log_kv_text("xB_eq", "%.6f", xB_eq);
    log_kv_text("xB_out", "%.6e (ic_23d_xB_out=%.6e)", xB_out, P->ic_23d_xB_out);
    log_kv_text("<xB>", "%.6f", mean_xB);
    log_kv_text("<xB_tot>", "%.6f", mean_xB_tot);
    log_kv_text("theory_target", "%.8f", theory_xB_tot);
    log_kv_text("mass_balance_diff", "%.2e", mean_xB_tot - theory_xB_tot);
}

// minimize 模式专用：仅初始化 phi，不计算/分配 xB/Y（无化学）
void initialize_phi_only_cuda(double *phi_r, const PFParams *P, int total_size) {
    const int Nx = P->Nx, Ny = P->Ny, Nz = P->Nz;
    const double dx = P->dx, dy = P->dy, dz = P->dz;
    const double Lx = Nx * dx, Ly = Ny * dy, Lz = Nz * dz;

    double vf_init_target = fmin(0.999, fmax(1e-6, P->ic_vf_init_phi));
    int N_seeds = (P->ic_phi_num_seeds > 0) ? P->ic_phi_num_seeds : 1;
    int is3D = (Ny > 2);

    double w = (P->ic_phi_iface_w > 0.0) ? (P->ic_phi_iface_w * dx) : dx;
    double R;
    if (P->ic_phi_seed_radius > 0.0) {
        R = P->ic_phi_seed_radius;
    } else {
        double Rtar;
        if (is3D) {
            Rtar = cbrt((vf_init_target * Lx * Ly * Lz) /
                        (N_seeds * (4.0 * M_PI / 3.0)));
        } else {
            Rtar = sqrt((vf_init_target * Lx * Lz) / (N_seeds * M_PI));
        }
        R = fmax(1.0, Rtar);
    }

    double Rx, Ry, Rz;
    // 使用与 initialize_fields_cuda 一致的等体积/等面积形状控制逻辑
    compute_seed_axes_from_radius(P, R, is3D, &Rx, &Ry, &Rz);

    double stack_centers[1024];
    double *centers = NULL;
    if (3 * N_seeds <= (int)(sizeof(stack_centers) / sizeof(double))) {
        centers = stack_centers;
    } else {
        centers = (double*)malloc((size_t)3 * N_seeds * sizeof(double));
    }
    generate_seed_centers_uniform(P, centers);

    double *d_centers = NULL;
    double *d_rot_mats = NULL;
    double *d_phi_tmp = NULL;
    size_t centers_bytes = (size_t)3 * N_seeds * sizeof(double);
    size_t size_r = (size_t)total_size * sizeof(double);

    // 施加中心偏移
    if (P->init_center_shift_x != 0.0 ||
        P->init_center_shift_y != 0.0 ||
        P->init_center_shift_z != 0.0) {
        for (int s = 0; s < N_seeds; ++s) {
            centers[3*s + 0] += P->init_center_shift_x;
            centers[3*s + 1] += P->init_center_shift_y;
            centers[3*s + 2] += P->init_center_shift_z;
            if (centers[3*s + 0] < 0.0) centers[3*s + 0] += Lx;
            if (centers[3*s + 0] >= Lx) centers[3*s + 0] -= Lx;
            if (centers[3*s + 1] < 0.0) centers[3*s + 1] += Ly;
            if (centers[3*s + 1] >= Ly) centers[3*s + 1] -= Ly;
            if (centers[3*s + 2] < 0.0) centers[3*s + 2] += Lz;
            if (centers[3*s + 2] >= Lz) centers[3*s + 2] -= Lz;
        }
    }

    // 构造旋转矩阵（若需要）
    double stack_rot[9 * 16];
    double *rot_mats = NULL;
    if (P->init_use_rotation && (P->init_tilt_theta_deg != 0.0 || P->init_tilt_phi_deg != 0.0)) {
        if (9 * N_seeds <= (int)(sizeof(stack_rot) / sizeof(double))) {
            rot_mats = stack_rot;
        } else {
            rot_mats = (double*)malloc((size_t)9 * N_seeds * sizeof(double));
        }
        double R_seed[9];
        build_seed_rotation_from_normal(P->init_tilt_theta_deg, P->init_tilt_phi_deg, R_seed);
        for (int s = 0; s < N_seeds; ++s) {
            for (int k = 0; k < 9; ++k) {
                rot_mats[9*s + k] = R_seed[k];
            }
        }
    }

    CUDA_CHECK(cudaMalloc(&d_centers, centers_bytes));
    CUDA_CHECK(cudaMemcpy(d_centers, centers, centers_bytes, cudaMemcpyHostToDevice));
    if (rot_mats) {
        size_t rot_bytes = (size_t)9 * N_seeds * sizeof(double);
        CUDA_CHECK(cudaMalloc(&d_rot_mats, rot_bytes));
        CUDA_CHECK(cudaMemcpy(d_rot_mats, rot_mats, rot_bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&d_phi_tmp, size_r));

    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    init_phi_kernel<<<blocks, threads>>>(d_phi_tmp, d_centers,
                                         d_rot_mats, (P->init_use_rotation ? 1 : 0),
                                         N_seeds,
                                         Nx, Ny, Nz, dx, dy, dz,
                                         Lx, Ly, Lz, w,
                                         Rx, Ry, Rz, is3D);
    CUDA_CHECK(cudaDeviceSynchronize());
    // 施加 phi 噪声（若启用）
    if (P->init_phi_noise_amp > 0.0) {
        unsigned long long seed_base =
            (P->init_phi_noise_seed != 0 ? (unsigned long long)P->init_phi_noise_seed
                                         : (unsigned long long)P->seed);
        apply_phi_noise_kernel<<<blocks, threads>>>(d_phi_tmp, total_size,
                                                    P->init_phi_noise_amp, seed_base);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaMemcpy(phi_r, d_phi_tmp, size_r, cudaMemcpyDeviceToHost));

    double first_center_log[3] = {0.0, 0.0, 0.0};
    int has_first_center_log = (N_seeds > 0);
    if (has_first_center_log) {
        first_center_log[0] = centers[0];
        first_center_log[1] = centers[1];
        first_center_log[2] = centers[2];
    }
    double R_seed_log[9] = {0.0};
    int has_seed_rotation_log = 0;
    if (P->init_use_rotation && (P->init_tilt_theta_deg != 0.0 || P->init_tilt_phi_deg != 0.0)) {
        build_seed_rotation_from_normal(P->init_tilt_theta_deg, P->init_tilt_phi_deg, R_seed_log);
        has_seed_rotation_log = 1;
    }

    if (centers != stack_centers && centers != NULL) {
        free(centers);
    }
    if (rot_mats && rot_mats != stack_rot) {
        free(rot_mats);
    }
    CUDA_CHECK(cudaFree(d_phi_tmp));
    if (d_rot_mats) CUDA_CHECK(cudaFree(d_rot_mats));
    CUDA_CHECK(cudaFree(d_centers));

    double sum_h = 0.0;
    for (int i = 0; i < total_size; i++) {
        sum_h += h_of_phi(phi_r[i]);
    }
    double mean_h = sum_h / (double)(Nx * Ny * Nz);
    const char *shape_str = "legacy";
    if (P->init_shape_mode == 0) shape_str = "sphere";
    else if (P->init_shape_mode == 1) shape_str = "ellipsoid";
    log_section_header("Initialization Geometry (phi-only)");
    log_kv_text("mean_h", "%.6f", mean_h);
    log_kv_text("shape", "%s", shape_str);
    log_kv_text("R_input", "%.6f", P->ic_phi_seed_radius);
    log_kv_text("Rx / Ry / Rz", "%.6f / %.6f / %.6f", Rx, Ry, Rz);
    log_kv_text("R_equiv", "%.6f", is3D ? cbrt(Rx * Ry * Rz) : sqrt(Rx * Rz));
    log_kv_text("tilt(theta,phi) [deg]", "%.3f / %.3f", P->init_tilt_theta_deg, P->init_tilt_phi_deg);
    log_kv_text("rotation", "%s", P->init_use_rotation ? "enabled" : "disabled");
    log_kv_text("center_shift", "(%.6f, %.6f, %.6f)", P->init_center_shift_x, P->init_center_shift_y, P->init_center_shift_z);
    log_kv_text("phi_noise_amp", "%.3e", P->init_phi_noise_amp);
    log_kv_text("phi_noise_seed", "%lu", P->init_phi_noise_seed);
    log_kv_text("init_test_id", "%d", P->init_test_id);
    if (has_first_center_log) {
        log_kv_text("first_seed_center", "(%.6f, %.6f, %.6f)",
                    first_center_log[0], first_center_log[1], first_center_log[2]);
    }
    if (has_seed_rotation_log) {
        log_kv_text("rotation_row1", "(%.6f, %.6f, %.6f)", R_seed_log[0], R_seed_log[1], R_seed_log[2]);
        log_kv_text("rotation_row2", "(%.6f, %.6f, %.6f)", R_seed_log[3], R_seed_log[4], R_seed_log[5]);
        log_kv_text("rotation_row3", "(%.6f, %.6f, %.6f)", R_seed_log[6], R_seed_log[7], R_seed_log[8]);
    }
}

// 更新参数中的热力学量
static void pfparams_refresh_thermo(PFParams *P) {
    double temperature_K = P->temperature_C + 273.15;
    if (temperature_K < 1.0) {
        temperature_K = 1.0;
    }
    double xB_eq = xB_eq_from_temperature(temperature_K);
    P->ic_xB_eq_matrix = xB_eq;
    double muA_eq = mu_A_dimless(xB_eq, temperature_K, P->mu_reference_scale);
    double muB_eq = mu_B_dimless(xB_eq, temperature_K, P->mu_reference_scale);
    P->mu0_compound = v_weighted_mu_compound(muA_eq, muB_eq, P->v_A, P->v_B);
}

// 根据给定法向 n(theta,phi) 构造旋转矩阵 R，使椭球短轴对齐到 n
// 约定：输入为角度制，theta,phi 定义为：
//   n = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
// R 采用行优先(row-major)存储，行向量分别为 e1,e2,e3(=n)：
//   q = R * r, 其中 q_x = e1·r, q_y = e2·r, q_z = e3·r
static void build_seed_rotation_from_normal(double theta_deg, double phi_deg, double R[9]) {
    const double deg2rad = M_PI / 180.0;
    double theta = theta_deg * deg2rad;
    double phi   = phi_deg   * deg2rad;

    // 椭球短轴方向 n
    double nx = sin(theta) * cos(phi);
    double ny = sin(theta) * sin(phi);
    double nz = cos(theta);

    // 处理退化情形：若用户给了非常小的倾角，直接退化为单位阵
    double n_norm = sqrt(nx*nx + ny*ny + nz*nz);
    if (n_norm < 1e-8) {
        // 单位阵
        R[0] = 1.0; R[1] = 0.0; R[2] = 0.0;
        R[3] = 0.0; R[4] = 1.0; R[5] = 0.0;
        R[6] = 0.0; R[7] = 0.0; R[8] = 1.0;
        return;
    }
    nx /= n_norm; ny /= n_norm; nz /= n_norm;

    // 选择一个与 n 不平行的参考矢量，用于构造正交基
    double rx = 0.0, ry = 0.0, rz = 1.0;
    if (fabs(nz) > 0.9) { // n 过于接近 z 轴时，改用 y 轴避免病态
        rx = 0.0; ry = 1.0; rz = 0.0;
    }

    // e1 = normalize( r × n )
    double e1x = ry * nz - rz * ny;
    double e1y = rz * nx - rx * nz;
    double e1z = rx * ny - ry * nx;
    double e1_norm = sqrt(e1x*e1x + e1y*e1y + e1z*e1z);
    if (e1_norm < 1e-12) {
        // 退化情况下仍然回退为单位阵
        R[0] = 1.0; R[1] = 0.0; R[2] = 0.0;
        R[3] = 0.0; R[4] = 1.0; R[5] = 0.0;
        R[6] = 0.0; R[7] = 0.0; R[8] = 1.0;
        return;
    }
    e1x /= e1_norm; e1y /= e1_norm; e1z /= e1_norm;

    // e2 = n × e1
    double e2x = ny * e1z - nz * e1y;
    double e2y = nz * e1x - nx * e1z;
    double e2z = nx * e1y - ny * e1x;

    // e3 = n
    double e3x = nx, e3y = ny, e3z = nz;

    // 写入行优先矩阵 R（每一行为一个正交基）
    R[0] = e1x; R[1] = e1y; R[2] = e1z;
    R[3] = e2x; R[4] = e2y; R[5] = e2z;
    R[6] = e3x; R[7] = e3y; R[8] = e3z;
}

// 根据 init_test_id 预设一组初始化参数，仅覆盖初始化相关字段
static void apply_init_test_preset(PFParams *P) {
    if (P->init_test_id < 0) {
        return; // 未启用 preset，保持显式 flag / 默认值
    }

    // 先恢复一套“干净”的初始化默认值，再按 preset 改写
    P->init_shape_mode    = 0;   // sphere
    P->init_axis_ratio_rx = 1.0;
    P->init_axis_ratio_ry = 1.0;
    P->init_axis_ratio_rz = 1.0;
    P->init_tilt_theta_deg = 0.0;
    P->init_tilt_phi_deg   = 0.0;
    P->init_center_shift_x = 0.0;
    P->init_center_shift_y = 0.0;
    P->init_center_shift_z = 0.0;
    P->init_phi_noise_amp  = 0.0;
    P->init_phi_noise_seed = 0;
    P->init_use_rotation   = 0;

    switch (P->init_test_id) {
        case 0:
            // 0: 球形、无扰动、无偏移
            P->init_shape_mode = 0; // sphere
            break;
        case 1:
            // 1: 轴对齐椭球（例如 x 短）
            P->init_shape_mode    = 1;   // ellipsoid
            P->init_axis_ratio_rx = 0.8; // x 方向略短
            P->init_axis_ratio_ry = 1.0;
            P->init_axis_ratio_rz = 1.0;
            break;
        case 2:
            // 2: 轻微 tilt，theta=10°, phi=0°
            P->init_shape_mode     = 1;
            P->init_axis_ratio_rx  = 0.8;
            P->init_axis_ratio_ry  = 1.0;
            P->init_axis_ratio_rz  = 1.0;
            P->init_tilt_theta_deg = 10.0;
            P->init_tilt_phi_deg   = 0.0;
            P->init_use_rotation   = 1;
            break;
        case 3:
            // 3: 轻微 tilt，theta=10°, phi=45°
            P->init_shape_mode     = 1;
            P->init_axis_ratio_rx  = 0.8;
            P->init_axis_ratio_ry  = 1.0;
            P->init_axis_ratio_rz  = 1.0;
            P->init_tilt_theta_deg = 10.0;
            P->init_tilt_phi_deg   = 45.0;
            P->init_use_rotation   = 1;
            break;
        case 4:
            // 4: tilt + 小偏移
            P->init_shape_mode     = 1;
            P->init_axis_ratio_rx  = 0.8;
            P->init_axis_ratio_ry  = 1.0;
            P->init_axis_ratio_rz  = 1.0;
            P->init_tilt_theta_deg = 10.0;
            P->init_tilt_phi_deg   = 30.0;
            P->init_center_shift_x = 2.0;
            P->init_center_shift_y = -1.0;
            P->init_center_shift_z = 0.5;
            P->init_use_rotation   = 1;
            break;
        case 5:
            // 5: 球形 + 小噪声
            P->init_shape_mode    = 0;
            P->init_phi_noise_amp = 1.0e-3;
            P->init_phi_noise_seed = 123UL;
            break;
        case 6:
            // 6: 椭球 + 小噪声
            P->init_shape_mode    = 1;
            P->init_axis_ratio_rx = 0.8;
            P->init_axis_ratio_ry = 1.0;
            P->init_axis_ratio_rz = 1.0;
            P->init_phi_noise_amp = 1.0e-3;
            P->init_phi_noise_seed = 456UL;
            break;
        case 7:
            // 7: tilt + 小噪声
            P->init_shape_mode     = 1;
            P->init_axis_ratio_rx  = 0.8;
            P->init_axis_ratio_ry  = 1.0;
            P->init_axis_ratio_rz  = 1.0;
            P->init_tilt_theta_deg = 10.0;
            P->init_tilt_phi_deg   = 45.0;
            P->init_phi_noise_amp  = 5.0e-4;
            P->init_phi_noise_seed = 2026UL;
            P->init_use_rotation   = 1;
            break;
        default:
            // 未定义的编号：保持“干净默认”但仍打印 test_id 以便诊断
            break;
    }
}

// 默认参数设置
static void params_default(PFParams *P) {
    P->Nx = 400; P->Ny = 400; P->Nz = 400;
    P->dx = P->dy = P->dz = -1.0;
    P->dt = 1.0e-2;
    P->t_real_unit = 1.114792e+00;
    P->nsteps = 10;
    P->out_every = 10;
    P->csv_out_every = 1;  // CSV输出间隔，默认为每个时间步
    P->dimension = 3;
    P->seed = 12345;
    
    // 物理输入必须由 --pf-param-file 提供；这里使用哨兵值以防漏传
    P->W = -1.0;
    P->kappa_phi = -1.0;
    P->L_phi = -1.0;
    P->D_alpha = -1.0;
    P->D_compound = -1.0;
    
    P->temperature_C = -1.0;
    P->mu_reference_scale = -1.0;
    P->v_B = -1.0;
    P->v_A = -1.0;
    
    P->Vm_compound = -1.0;
    P->Vm_alpha_0 = -1.0;
    P->dVm_alpha_dxB = -1.0;
    
    P->Y_clip = 20.0;
    P->xB_eps = 1e-8;
    P->xB_s_floor = 0.0002;
    
    P->ic_vf_init_phi = -1.0;
    // P->ic_23d_xB_out = 4.664952e-03;
    P->ic_23d_xB_out = 0.03;
    // P->ic_23d_xB_out = 3.452500e-02;     /* 2D/3D：外部（φ≈0）区域初始 xB；若 >0，则忽略 ic_vf_target_phi，直接使用该值 */
    P->ic_vf_target_phi = -1.0;      /* 必须由外部物理输入提供 */
    P->ic_phi_num_seeds = 1;               // 种子数量：与CUDA版保持一致
    P->ic_phi_iface_w = -1.0;              // 必须由外部物理输入提供
    P->ic_xB_width_factor = 1.0;           // xB 界面宽度相对于 φ 界面宽度的比例
    P->ic_phi_seed_radius = 10.0;            // 初始化时由 initialize_fields 反求并写回的等效种子半径
    P->ic_xB_eq_matrix = -1.0;
    P->mu0_compound = -1.0;

    // 破对称 / 多初值测试相关初始化默认值（保持旧行为完全不变）
    P->init_shape_mode    = -1;   // legacy: 按旧逻辑决定 Rx,Ry,Rz
    P->init_axis_ratio_rx = 1.0;
    P->init_axis_ratio_ry = 1.0;
    P->init_axis_ratio_rz = 1.0;
    P->init_tilt_theta_deg = 0.0;
    P->init_tilt_phi_deg   = 0.0;
    P->init_center_shift_x = 0.0;
    P->init_center_shift_y = 0.0;
    P->init_center_shift_z = 0.0;
    P->init_phi_noise_amp  = 0.0;
    P->init_phi_noise_seed = 0;
    P->init_test_id        = -1;
    P->init_use_rotation   = 0;

    // 诊断VTK默认关闭（只输出 phi/xB/xBtot）
    P->diag_vtk_enabled = 0;
    
    // 弹性 bulk 惩罚诊断默认关闭
    P->diag_elastic_bulk_penalty_enabled = 0;

    // 物理量标定默认值（请按你的体系修改）
    // 若 elastic_gel_is_dimless=1，则 gel_hat * w_phys -> J/m^3，其中 w_phys=12*gamma/lambda_sm
    P->gamma_Jm2 = -1.0;          // J/m^2
    P->lambda_sm_m = -1.0;        // m
    P->elastic_gel_is_dimless = 1;
    
    // === 摩尔体积设置（用于弹性 bulk 惩罚诊断：Delta_mu_el = E_el_bulk_Jm3 * Vm_alpha_0_phys_m3mol）===
    // 注意：
    //   - 上方第 405 行的 Vm_alpha_0 是无量纲摩尔体积（用于化学势计算等）
    //   - 下方的 Vm_alpha_0_phys_m3mol 是有量纲摩尔体积（单位：m^3/mol），专门用于弹性 bulk 惩罚诊断
    //   - 计算公式：Delta_mu_el (J/mol) = E_el_bulk_Jm3 (J/m^3) * Vm_alpha_0_phys_m3mol (m^3/mol)
    // 默认值（需修改为实际值）：
    P->Vm_alpha_0_phys_m3mol = -1.0;
   
    // =======================================================================================================
    P->oneD_test_mode = 0;       /* =0：关闭 1D slab，使用 2D 圆形种子 IC；=1：启用 1D slab 测试模式 */
    P->ic_1d_half_width_ratio = 0.025;     /* 1D 基准测试：界面半宽比例（保留参数，但在 2D/3D 中默认不用） */
    P->ic_xB_out = 0.058;    /* 仅在 oneD_test_mode=1 时有效：1D 外部区域初始 xB；若 <=0，则自动设为平衡值的 1.05 倍 */
    
    // ============================================================
    // 弹性参数默认值
    // ============================================================
    P->elastic_enabled = 1;      // 默认启用弹性计算（0/1）
    P->elastic_iter_max = 20;     // 默认迭代1次（可根据需要调整）
    // 无量纲弹性 shift 能量密度（加在 delta_mu 上）；仅在 elastic_enabled=1 时生效
    P->elastic_shift_dimless = -1.0;

    P->S_11 = P->S_12 = P->S_13 = P->S_14 = P->S_15 = P->S_16 = -1.0;
    P->S_22 = P->S_23 = P->S_24 = P->S_25 = P->S_26 = -1.0;
    P->S_33 = P->S_34 = P->S_35 = P->S_36 = -1.0;
    P->S_44 = P->S_45 = P->S_46 = -1.0;
    P->S_55 = P->S_56 = -1.0;
    P->S_66 = -1.0;

    P->S_p_11 = P->S_p_12 = P->S_p_13 = P->S_p_14 = P->S_p_15 = P->S_p_16 = -1.0;
    P->S_p_22 = P->S_p_23 = P->S_p_24 = P->S_p_25 = P->S_p_26 = -1.0;
    P->S_p_33 = P->S_p_34 = P->S_p_35 = P->S_p_36 = -1.0;
    P->S_p_44 = P->S_p_45 = P->S_p_46 = -1.0;
    P->S_p_55 = P->S_p_56 = -1.0;
    P->S_p_66 = -1.0;
    
    // E0参数（外部应变，6个分量）- 默认全为0
    P->E0_xx = 0.0;
    P->E0_yy = 0.0;
    P->E0_zz = 0.0;
    P->E0_yz = 0.0;
    P->E0_xz = 0.0;
    P->E0_xy = 0.0;

    // stress-free transformation strain eps^00_ij：默认全0
    P->eps_xx00 = -1.0;
    P->eps_yy00 = -1.0;
    P->eps_zz00 = -1.0;
    P->eps_yz00 = -1.0;
    P->eps_xz00 = -1.0;
    P->eps_xy00 = -1.0;
    
    P->eps_iso_over_vB = -1.0;
    P->xB_ref_for_eps_c = -1.0;

    // ============================================================
    // Energy minimization mode defaults (phi-only / full-model)
    // ============================================================
    P->mode = 0; // 0=dynamics, 1=minimize
    P->minimize_full_model = 0; // 0=phi-only minimize, 1=full-model (chem+diff+volume)
    P->minimize_max_iter = 2000;
    P->minimize_dt = P->dt;
    P->minimize_V0 = 0.0;  // if <=0, use initial mean_h at step 1
    P->minimize_resample_elastic_every = 100;  // N<=0 disables true residual diagnostic
    // 收敛判据默认值（A/B 规则）
    P->minimize_rms_dphi_threshold = 1.0e-6;   // phi 收敛
    P->minimize_rms_dY_threshold   = 5.0e-5;   // Y 收敛（full-model）
    P->minimize_energy_diff_rel_threshold = 1.0e-9; // 能量平台
    P->minimize_rms_res_for_energy_plateau = 1.0e-4; // 仅作诊断打印
    P->minimize_rms_res_threshold = 1.0e-4;  // Euler-Lagrange/KKT 残差 rms_res 硬判停
    P->minimize_vol_err_rel_threshold = 1.0e-4;  // 体积约束相对误差硬判停，V0<=0 时禁用
    P->minimize_convergence_steps = 10;
    P->minimize_dt_safety_limit = 1.0e-6; // 当前不再自动二分 dt，仅作保底/诊断
    P->eta_lambda_vol = 0.2; // lambda_vol under-relaxation 阻尼系数，默认 0.2
    P->minimize_xB_max_safe = 0.07; // minimize: pre-thermo xB 上限（硬裁剪），防止热力学核看到过大的 xB
    P->minimize_post_projection_iters = 1;   // 后投影修正子步数，默认 1

    // 初始化 case 标签默认置空，后续根据 init_test_id / 其它 flag 自动生成
    P->init_case_tag[0] = '\0';

    P->ic_phi_centers_max = 0;
    P->ic_phi_centers = NULL;
}

// 辅助：沿法向在 phi 跨越 0.5 的格点处做线性插值，求边界位置的平均（格点单位）
// 仅对 phi_left 与 phi_right  straddle 0.5 的 (y,z) 或 (x,z) 或 (x,y) 做插值，避免整面平均稀释界面
static inline void boundary_interp_x(const double *h_phi, int Nx, int Ny, int Nz, int x_lo, int x_hi, double dx,
                                     int is_left, double *out_phys) {
    double sum_pos = 0.0;
    int count = 0;
    for (int y = 0; y < Ny; y++)
        for (int z = 0; z < Nz; z++) {
            int idx_lo = (x_lo * Ny + y) * Nz + z;
            int idx_hi = (x_hi * Ny + y) * Nz + z;
            double p_lo = h_phi[idx_lo], p_hi = h_phi[idx_hi];
            if (!((p_lo < 0.5 && p_hi > 0.5) || (p_lo > 0.5 && p_hi < 0.5))) continue;
            double denom = p_hi - p_lo;
            double f = (fabs(denom) < 1e-12) ? 0.5 : (0.5 - p_lo) / denom;
            if (f < 0.0) f = 0.0;
            if (f > 1.0) f = 1.0;
            double pos_grid = is_left ? (x_lo + f) : (x_hi - (1.0 - f));
            sum_pos += pos_grid;
            count++;
        }
    *out_phys = (count > 0) ? (sum_pos / (double)count) * dx : (is_left ? x_lo : x_hi) * dx;
}
static inline void boundary_interp_y(const double *h_phi, int Nx, int Ny, int Nz, int y_lo, int y_hi, double dy,
                                     int is_left, double *out_phys) {
    double sum_pos = 0.0;
    int count = 0;
    for (int x = 0; x < Nx; x++)
        for (int z = 0; z < Nz; z++) {
            int idx_lo = (x * Ny + y_lo) * Nz + z;
            int idx_hi = (x * Ny + y_hi) * Nz + z;
            double p_lo = h_phi[idx_lo], p_hi = h_phi[idx_hi];
            if (!((p_lo < 0.5 && p_hi > 0.5) || (p_lo > 0.5 && p_hi < 0.5))) continue;
            double denom = p_hi - p_lo;
            double f = (fabs(denom) < 1e-12) ? 0.5 : (0.5 - p_lo) / denom;
            if (f < 0.0) f = 0.0;
            if (f > 1.0) f = 1.0;
            double pos_grid = is_left ? (y_lo + f) : (y_hi - (1.0 - f));
            sum_pos += pos_grid;
            count++;
        }
    *out_phys = (count > 0) ? (sum_pos / (double)count) * dy : (is_left ? y_lo : y_hi) * dy;
}
static inline void boundary_interp_z(const double *h_phi, int Nx, int Ny, int Nz, int z_lo, int z_hi, double dz,
                                     int is_left, double *out_phys) {
    double sum_pos = 0.0;
    int count = 0;
    for (int x = 0; x < Nx; x++)
        for (int y = 0; y < Ny; y++) {
            int idx_lo = (x * Ny + y) * Nz + z_lo;
            int idx_hi = (x * Ny + y) * Nz + z_hi;
            double p_lo = h_phi[idx_lo], p_hi = h_phi[idx_hi];
            if (!((p_lo < 0.5 && p_hi > 0.5) || (p_lo > 0.5 && p_hi < 0.5))) continue;
            double denom = p_hi - p_lo;
            double f = (fabs(denom) < 1e-12) ? 0.5 : (0.5 - p_lo) / denom;
            if (f < 0.0) f = 0.0;
            if (f > 1.0) f = 1.0;
            double pos_grid = is_left ? (z_lo + f) : (z_hi - (1.0 - f));
            sum_pos += pos_grid;
            count++;
        }
    *out_phys = (count > 0) ? (sum_pos / (double)count) * dz : (is_left ? z_lo : z_hi) * dz;
}

// ============================================================
// 计算析出相（Nucleus）在 X、Y、Z 三个方向上的物理尺寸
// 在 phi=0.5 跨越处做线性插值，仅对界面穿越点求平均，得到亚网格精度
// ============================================================
__global__ void reduce_h_bbox_kernel(const double *phi_r,
                                    int Nx, int Ny, int Nz,
                                    int *mins, int *maxs,
                                    int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double h = h_of_phi(clamp01_local(phi_r[idx]));
    if (h < 0.5) return;

    int z = idx % Nz;
    int y = (idx / Nz) % Ny;
    int x = idx / (Ny * Nz);
    atomicMin(&mins[0], x);
    atomicMin(&mins[1], y);
    atomicMin(&mins[2], z);
    atomicMax(&maxs[0], x);
    atomicMax(&maxs[1], y);
    atomicMax(&maxs[2], z);
}

__global__ void accumulate_h_boundary_kernel(const double *phi_r,
                                             int Nx, int Ny, int Nz,
                                             int axis,
                                             int lo_idx,
                                             int hi_idx,
                                             double spacing,
                                             double *sum_pos,
                                             unsigned long long *count) {
    int plane_size = 0;
    if (axis == 0) plane_size = Ny * Nz;
    else if (axis == 1) plane_size = Nx * Nz;
    else plane_size = Nx * Ny;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= plane_size) return;

    int x0 = 0, y0 = 0, z0 = 0;
    int x1 = 0, y1 = 0, z1 = 0;

    if (axis == 0) {
        y0 = idx / Nz;
        z0 = idx % Nz;
        x0 = lo_idx;
        x1 = hi_idx;
        y1 = y0;
        z1 = z0;
    } else if (axis == 1) {
        x0 = idx / Nz;
        z0 = idx % Nz;
        y0 = lo_idx;
        y1 = hi_idx;
        x1 = x0;
        z1 = z0;
    } else {
        x0 = idx / Ny;
        y0 = idx % Ny;
        z0 = lo_idx;
        z1 = hi_idx;
        x1 = x0;
        y1 = y0;
    }

    int idx_lo = (x0 * Ny + y0) * Nz + z0;
    int idx_hi = (x1 * Ny + y1) * Nz + z1;
    double h_lo = h_of_phi(clamp01_local(phi_r[idx_lo]));
    double h_hi = h_of_phi(clamp01_local(phi_r[idx_hi]));

    const double iso = 0.5;
    int straddles = ((h_lo <= iso && h_hi >= iso) || (h_lo >= iso && h_hi <= iso));
    if (!straddles) return;

    double denom = h_hi - h_lo;
    double frac = 0.5;
    if (fabs(denom) >= 1e-12) {
        frac = (iso - h_lo) / denom;
        if (frac < 0.0) frac = 0.0;
        if (frac > 1.0) frac = 1.0;
    }

    double pos_grid = (double)lo_idx + frac;
    atomicAdd(sum_pos, pos_grid * spacing);
    atomicAdd(count, 1ULL);
}

static void compute_nucleus_dimensions_gpu(const double *d_phi_r, int Nx, int Ny, int Nz,
                                           double dx, double dy, double dz,
                                           int *d_bbox_mins, int *d_bbox_maxs,
                                           double *d_boundary_sum,
                                           unsigned long long *d_boundary_count,
                                           double *Len_x, double *Len_y, double *Len_z) {
    int h_mins[3] = {Nx, Ny, Nz};
    int h_maxs[3] = {-1, -1, -1};
    const int total_size = Nx * Ny * Nz;
    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;

    CUDA_CHECK(cudaMemcpy(d_bbox_mins, h_mins, sizeof(h_mins), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bbox_maxs, h_maxs, sizeof(h_maxs), cudaMemcpyHostToDevice));
    reduce_h_bbox_kernel<<<blocks, threads>>>(d_phi_r, Nx, Ny, Nz, d_bbox_mins, d_bbox_maxs, total_size);
    CUDA_CHECK(cudaMemcpy(h_mins, d_bbox_mins, sizeof(h_mins), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_maxs, d_bbox_maxs, sizeof(h_maxs), cudaMemcpyDeviceToHost));

    *Len_x = 0.0;
    *Len_y = 0.0;
    *Len_z = 0.0;
    if (!(h_mins[0] <= h_maxs[0] && h_mins[1] <= h_maxs[1] && h_mins[2] <= h_maxs[2])) {
        return;
    }

    double left_x = h_mins[0] * dx;
    double right_x = (h_maxs[0] + 1) * dx;
    double left_y = h_mins[1] * dy;
    double right_y = (h_maxs[1] + 1) * dy;
    double left_z = h_mins[2] * dz;
    double right_z = (h_maxs[2] + 1) * dz;

    auto accumulate_boundary = [&](int axis, int lo_idx, int hi_idx, double spacing, double *out_phys) {
        double zero_sum = 0.0;
        unsigned long long zero_count = 0ULL;
        CUDA_CHECK(cudaMemcpy(d_boundary_sum, &zero_sum, sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_boundary_count, &zero_count, sizeof(unsigned long long), cudaMemcpyHostToDevice));

        int plane_size = (axis == 0) ? (Ny * Nz) : ((axis == 1) ? (Nx * Nz) : (Nx * Ny));
        int plane_blocks = (plane_size + threads - 1) / threads;
        accumulate_h_boundary_kernel<<<plane_blocks, threads>>>(
            d_phi_r, Nx, Ny, Nz, axis, lo_idx, hi_idx, spacing, d_boundary_sum, d_boundary_count);

        double sum_pos = 0.0;
        unsigned long long count = 0ULL;
        CUDA_CHECK(cudaMemcpy(&sum_pos, d_boundary_sum, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&count, d_boundary_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        if (count > 0ULL) {
            *out_phys = sum_pos / (double)count;
        }
    };

    if (h_mins[0] > 0) accumulate_boundary(0, h_mins[0] - 1, h_mins[0], dx, &left_x);
    if (h_maxs[0] < Nx - 1) accumulate_boundary(0, h_maxs[0], h_maxs[0] + 1, dx, &right_x);
    if (h_mins[1] > 0) accumulate_boundary(1, h_mins[1] - 1, h_mins[1], dy, &left_y);
    if (h_maxs[1] < Ny - 1) accumulate_boundary(1, h_maxs[1], h_maxs[1] + 1, dy, &right_y);
    if (h_mins[2] > 0) accumulate_boundary(2, h_mins[2] - 1, h_mins[2], dz, &left_z);
    if (h_maxs[2] < Nz - 1) accumulate_boundary(2, h_maxs[2], h_maxs[2] + 1, dz, &right_z);

    *Len_x = right_x - left_x;
    *Len_y = right_y - left_y;
    *Len_z = right_z - left_z;
}

// ============================================================
// 诊断：弹性 bulk 惩罚(能量密度)与化学势 shift
// ============================================================
typedef struct {
    double sum_h;
    double sum_gel;
    double E_el_bulk_hat;
    double E_el_bulk_Jm3;
    double Delta_mu_el_Jmol;
    int gel_is_dimless;
    int valid;
} ElasticBulkPenaltyDiag;

static ElasticBulkPenaltyDiag compute_elastic_bulk_penalty(
    const double *d_phi_r,
    // strain (total)
    const float *d_uxx_r, const float *d_uyy_r, const float *d_uzz_r,
    const float *d_uxy_r, const float *d_uxz_r, const float *d_uyz_r,
    // Optimization(4): eigenstrain 参数已移除，现场计算
    const double *d_xB_r,  // 需要 xB 来现场计算 eps0（可为 NULL，minimize 模式）
    // stress
    const float *d_sigma_xx_r, const float *d_sigma_yy_r, const float *d_sigma_zz_r,
    const float *d_sigma_xy_r, const float *d_sigma_xz_r, const float *d_sigma_yz_r,
    int total_r,
    const PFParams *P)
{
    ElasticBulkPenaltyDiag out;
    out.sum_h = 0.0;
    out.sum_gel = 0.0;
    out.E_el_bulk_hat = 0.0;
    out.E_el_bulk_Jm3 = 0.0;
    out.Delta_mu_el_Jmol = 0.0;
    out.gel_is_dimless = P->elastic_gel_is_dimless ? 1 : 0;
    out.valid = 0;

    // sum_h = sum h(phi) (GPU reduction via existing helper)
    double mean_h = gpu_compute_vf_from_h(d_phi_r, total_r);
    double sum_h = mean_h * (double)total_r;
    out.sum_h = sum_h;
    if (sum_h < 1e-12) {
        fprintf(stderr, "[warn] elastic bulk penalty: sum_h=%.3e too small, skip.\n", sum_h);
        return out;
    }

    // gel_hat field
    double *d_gel_hat = NULL;
    CUDA_CHECK(cudaMalloc(&d_gel_hat, (size_t)total_r * sizeof(double)));
    launch_compute_gel_density_kernel(
        d_uxx_r, d_uyy_r, d_uzz_r,
        d_uxy_r, d_uxz_r, d_uyz_r,
        d_phi_r, d_xB_r,  // Optimization(4): 需要 phi 和 xB 来现场计算 eps0
        d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
        d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
        (float)P->eps_xx00, (float)P->eps_yy00, (float)P->eps_zz00,
        (float)P->eps_yz00, (float)P->eps_xz00, (float)P->eps_xy00,
        (double)P->eps_iso_over_vB,
        d_gel_hat,
        total_r);

    double sum_gel = gpu_reduce_sum(d_gel_hat, total_r);
    CUDA_CHECK(cudaFree(d_gel_hat));
    out.sum_gel = sum_gel;

    double E_el_bulk_hat = sum_gel / sum_h;
    out.E_el_bulk_hat = E_el_bulk_hat;

    // physical conversion
    double E_el_bulk_Jm3 = E_el_bulk_hat;
    if (P->elastic_gel_is_dimless) {
        if (P->lambda_sm_m <= 0.0) {
            fprintf(stderr, "[warn] elastic bulk penalty: lambda_sm_m<=0, cannot compute w_phys.\n");
            return out;
        }
        double w_phys = 12.0 * P->gamma_Jm2 / P->lambda_sm_m; // J/m^3
        E_el_bulk_Jm3 = w_phys * E_el_bulk_hat;
    }
    out.E_el_bulk_Jm3 = E_el_bulk_Jm3;

    // Delta_mu_el (J/mol): E(J/m^3) * Vm_alpha_0_phys_m3mol (m^3/mol)
    // 注意：使用有量纲摩尔体积 Vm_alpha_0_phys_m3mol，而不是无量纲 Vm_alpha_0
    if (P->Vm_alpha_0_phys_m3mol <= 0.0) {
        fprintf(stderr, "[warn] elastic bulk penalty: Vm_alpha_0_phys_m3mol<=0, cannot compute Delta_mu_el.\n");
        return out;
    }
    out.Delta_mu_el_Jmol = E_el_bulk_Jm3 * P->Vm_alpha_0_phys_m3mol;
    out.valid = 1;
    return out;
}

// ============================================================
// CLI helpers (support both positional args and --flags)
// ============================================================
static inline int starts_with(const char *s, const char *prefix) {
    if (!s || !prefix) return 0;
    while (*prefix) {
        if (*s++ != *prefix++) return 0;
    }
    return 1;
}

static inline const char* get_flag_value(int argc, char **argv, int *i, const char *key) {
    // Supports: --key value  OR  --key=value
    const char *arg = argv[*i];
    size_t keylen = strlen(key);
    if (strncmp(arg, key, keylen) != 0) return NULL;
    if (arg[keylen] == '=' && arg[keylen + 1] != '\0') {
        return arg + keylen + 1;
    }
    if (arg[keylen] == '\0') {
        if (*i + 1 < argc) {
            (*i)++;
            return argv[*i];
        }
    }
    return NULL;
}

static void trim_inplace(char *s) {
    if (!s) return;
    char *start = s;
    while (*start && isspace((unsigned char)*start)) start++;
    if (start != s) {
        memmove(s, start, strlen(start) + 1);
    }
    size_t len = strlen(s);
    while (len > 0 && isspace((unsigned char)s[len - 1])) {
        s[--len] = '\0';
    }
}

static int parse_double_value(const char *text, double *out) {
    if (!text || !out) return 0;
    while (*text && isspace((unsigned char)*text)) text++;
    if (*text == '\0') return 0;
    char *end = NULL;
    double value = strtod(text, &end);
    if (end == text) return 0;
    while (*end && isspace((unsigned char)*end)) end++;
    if (*end != '\0') return 0;
    *out = value;
    return 1;
}

static int parse_int_value(const char *text, int *out) {
    if (!text || !out) return 0;
    while (*text && isspace((unsigned char)*text)) text++;
    if (*text == '\0') return 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (end == text) return 0;
    while (*end && isspace((unsigned char)*end)) end++;
    if (*end != '\0') return 0;
    *out = (int)value;
    return 1;
}

static int apply_pfparams_override_key(PFParams *P, const char *key, const char *value, const char *path, int line_no) {
    if (!P || !key || !value) return -1;

#define TRY_SET_DOUBLE(name, field) \
    if (strcmp((key), (name)) == 0) { \
        double parsed = 0.0; \
        if (!parse_double_value((value), &parsed)) { \
            fprintf(stderr, "[fatal] %s:%d invalid numeric value for %s: %s\n", \
                    (path ? path : "<pf-param-file>"), (line_no), (name), (value)); \
            return -1; \
        } \
        P->field = parsed; \
        return 1; \
    }

#define TRY_SET_INT(name, field) \
    if (strcmp((key), (name)) == 0) { \
        int parsed = 0; \
        if (!parse_int_value((value), &parsed)) { \
            fprintf(stderr, "[fatal] %s:%d invalid integer value for %s: %s\n", \
                    (path ? path : "<pf-param-file>"), (line_no), (name), (value)); \
            return -1; \
        } \
        P->field = parsed; \
        return 1; \
    }

    TRY_SET_DOUBLE("W", W);
    TRY_SET_DOUBLE("dx", dx);
    TRY_SET_DOUBLE("dy", dy);
    TRY_SET_DOUBLE("dz", dz);
    TRY_SET_DOUBLE("kappa_phi", kappa_phi);
    TRY_SET_DOUBLE("L_phi", L_phi);
    TRY_SET_DOUBLE("D_alpha", D_alpha);
    TRY_SET_DOUBLE("D_compound", D_compound);
    TRY_SET_DOUBLE("temperature_C", temperature_C);
    TRY_SET_DOUBLE("mu_reference_scale", mu_reference_scale);
    TRY_SET_DOUBLE("v_A", v_A);
    TRY_SET_DOUBLE("v_B", v_B);
    TRY_SET_DOUBLE("Vm_compound", Vm_compound);
    TRY_SET_DOUBLE("Vm_alpha_0", Vm_alpha_0);
    TRY_SET_DOUBLE("dVm_alpha_dxB", dVm_alpha_dxB);
    TRY_SET_DOUBLE("ic_vf_init_phi", ic_vf_init_phi);
    TRY_SET_DOUBLE("ic_vf_target_phi", ic_vf_target_phi);
    TRY_SET_DOUBLE("ic_phi_iface_w", ic_phi_iface_w);
    TRY_SET_DOUBLE("gamma_Jm2", gamma_Jm2);
    TRY_SET_DOUBLE("lambda_sm_m", lambda_sm_m);
    TRY_SET_DOUBLE("Vm_alpha_0_phys_m3mol", Vm_alpha_0_phys_m3mol);
    TRY_SET_DOUBLE("elastic_shift_dimless", elastic_shift_dimless);
    TRY_SET_DOUBLE("eps_iso_over_vB", eps_iso_over_vB);
    TRY_SET_DOUBLE("eps_xx00", eps_xx00);
    TRY_SET_DOUBLE("eps_yy00", eps_yy00);
    TRY_SET_DOUBLE("eps_zz00", eps_zz00);
    TRY_SET_DOUBLE("eps_yz00", eps_yz00);
    TRY_SET_DOUBLE("eps_xz00", eps_xz00);
    TRY_SET_DOUBLE("eps_xy00", eps_xy00);

    TRY_SET_DOUBLE("S_11", S_11);   TRY_SET_DOUBLE("S_12", S_12);
    TRY_SET_DOUBLE("S_13", S_13);   TRY_SET_DOUBLE("S_14", S_14);
    TRY_SET_DOUBLE("S_15", S_15);   TRY_SET_DOUBLE("S_16", S_16);
    TRY_SET_DOUBLE("S_22", S_22);   TRY_SET_DOUBLE("S_23", S_23);
    TRY_SET_DOUBLE("S_24", S_24);   TRY_SET_DOUBLE("S_25", S_25);
    TRY_SET_DOUBLE("S_26", S_26);   TRY_SET_DOUBLE("S_33", S_33);
    TRY_SET_DOUBLE("S_34", S_34);   TRY_SET_DOUBLE("S_35", S_35);
    TRY_SET_DOUBLE("S_36", S_36);   TRY_SET_DOUBLE("S_44", S_44);
    TRY_SET_DOUBLE("S_45", S_45);   TRY_SET_DOUBLE("S_46", S_46);
    TRY_SET_DOUBLE("S_55", S_55);   TRY_SET_DOUBLE("S_56", S_56);
    TRY_SET_DOUBLE("S_66", S_66);

    TRY_SET_DOUBLE("S_p_11", S_p_11); TRY_SET_DOUBLE("S_p_12", S_p_12);
    TRY_SET_DOUBLE("S_p_13", S_p_13); TRY_SET_DOUBLE("S_p_14", S_p_14);
    TRY_SET_DOUBLE("S_p_15", S_p_15); TRY_SET_DOUBLE("S_p_16", S_p_16);
    TRY_SET_DOUBLE("S_p_22", S_p_22); TRY_SET_DOUBLE("S_p_23", S_p_23);
    TRY_SET_DOUBLE("S_p_24", S_p_24); TRY_SET_DOUBLE("S_p_25", S_p_25);
    TRY_SET_DOUBLE("S_p_26", S_p_26); TRY_SET_DOUBLE("S_p_33", S_p_33);
    TRY_SET_DOUBLE("S_p_34", S_p_34); TRY_SET_DOUBLE("S_p_35", S_p_35);
    TRY_SET_DOUBLE("S_p_36", S_p_36); TRY_SET_DOUBLE("S_p_44", S_p_44);
    TRY_SET_DOUBLE("S_p_45", S_p_45); TRY_SET_DOUBLE("S_p_46", S_p_46);
    TRY_SET_DOUBLE("S_p_55", S_p_55); TRY_SET_DOUBLE("S_p_56", S_p_56);
    TRY_SET_DOUBLE("S_p_66", S_p_66);

    TRY_SET_INT("elastic_enabled", elastic_enabled);
    TRY_SET_INT("elastic_iter_max", elastic_iter_max);
    TRY_SET_INT("diag_vtk_enabled", diag_vtk_enabled);
    TRY_SET_INT("diag_elastic_bulk_penalty_enabled", diag_elastic_bulk_penalty_enabled);
    TRY_SET_INT("mode", mode);
    TRY_SET_INT("minimize_full_model", minimize_full_model);

    if (strcmp(key, "init_case_tag") == 0) {
        strncpy(P->init_case_tag, value, sizeof(P->init_case_tag) - 1);
        P->init_case_tag[sizeof(P->init_case_tag) - 1] = '\0';
        return 1;
    }

#undef TRY_SET_DOUBLE
#undef TRY_SET_INT

    return 0;
}

static int load_pfparams_override_file(PFParams *P, const char *path) {
    if (!P || !path || path[0] == '\0') return 0;
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "[fatal] Could not open pf-param file: %s\n", path);
        return 0;
    }

    char line[2048];
    int line_no = 0;
    int applied = 0;
    PFParamOverridePresence presence;
    memset(&presence, 0, sizeof(presence));
    while (fgets(line, sizeof(line), fp) != NULL) {
        line_no++;
        char *comment = strchr(line, '#');
        if (comment) *comment = '\0';
        trim_inplace(line);
        if (line[0] == '\0') continue;

        char *sep = strchr(line, '=');
        if (!sep) {
            fprintf(stderr, "[fatal] %s:%d expected key=value format\n", path, line_no);
            fclose(fp);
            return 0;
        }
        *sep = '\0';
        char key[512];
        char value[1536];
        strncpy(key, line, sizeof(key) - 1);
        key[sizeof(key) - 1] = '\0';
        strncpy(value, sep + 1, sizeof(value) - 1);
        value[sizeof(value) - 1] = '\0';
        trim_inplace(key);
        trim_inplace(value);

        int rc = apply_pfparams_override_key(P, key, value, path, line_no);
        if (rc < 0) {
            fclose(fp);
            return 0;
        }
        if (rc == 0) {
            fprintf(stderr, "[warn] %s:%d unknown key ignored: %s\n", path, line_no, key);
            continue;
        }
        mark_pfparams_presence(&presence, key);
        applied++;
    }
    fclose(fp);
    if (!validate_required_pfparams_presence(&presence, path)) {
        return 0;
    }
    printf("[pf-param-file] loaded %d override(s) from %s\n", applied, path);
    return 1;
}

int main(int argc, char **argv) {
    // 立即刷新输出，确保能看到调试信息
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    const double wall_t0 = wall_time_sec_monotonic();
    
    // 解析命令行参数
    PFParams P;
    params_default(&P);

    int wants_help = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            wants_help = 1;
            break;
        }
    }

    const char *pf_param_file = NULL;
    for (int i = 1; i < argc; ++i) {
        const char *v = get_flag_value(argc, argv, &i, "--pf-param-file");
        if (v != NULL) {
            pf_param_file = v;
        }
    }
    if (!wants_help && pf_param_file == NULL) {
        fprintf(stderr,
                "[fatal] 物理参数不再使用 main_cuda 内置默认值。请通过 --pf-param-file <path> 提供完整物理输入。\n");
        fprintf(stderr,
                "        可先运行: python3 Unit_Psedobinary.py --input-json physical_inputs.example.json --output-pf-param-file /tmp/generated_pf.params\n");
        return 2;
    }
    if (pf_param_file != NULL) {
        if (!load_pfparams_override_file(&P, pf_param_file)) {
            return 2;
        }
    }

    if (argc >= 2 && argv[1][0] != '-') { P.Nx = atoi(argv[1]); }
    if (argc >= 3 && argv[2][0] != '-') { P.Ny = atoi(argv[2]); }
    if (argc >= 4 && argv[3][0] != '-') { P.Nz = atoi(argv[3]); }
    if (argc >= 5 && argv[4][0] != '-') { P.dt = atof(argv[4]); }
    if (argc >= 6 && argv[5][0] != '-') { P.nsteps = atoi(argv[5]); }
    if (argc >= 7 && argv[6][0] != '-') { P.out_every = atoi(argv[6]); }
    if (argc >= 8 && argv[7][0] != '-') { P.csv_out_every = atoi(argv[7]); }  // CSV输出间隔参数
    if (argc >= 9 && argv[8][0] != '-') { P.elastic_enabled = atoi(argv[8]); }  // 弹性功能开关（0/1）
    // 诊断开关（VTK / 弹性 bulk 惩罚）不再通过命令行控制，统一在 params_default 中设置
    
    // ----------------------------
    // 预扫描 init-test-id 和 init-case-tag（在详细 flag 解析前）
    // ----------------------------
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--init-test-id") == 0 && i + 1 < argc) {
            P.init_test_id = atoi(argv[i + 1]);
        }
        if (strcmp(argv[i], "--init-case-tag") == 0 && i + 1 < argc) {
            strncpy(P.init_case_tag, argv[i + 1], sizeof(P.init_case_tag) - 1);
            P.init_case_tag[sizeof(P.init_case_tag) - 1] = '\0';
        }
    }
    // 先根据 test preset 设置一组初始化参数，后续显式 flags 可以在此基础上覆盖
    apply_init_test_preset(&P);

    // ----------------------------
    // Parse optional flags (do not break positional args)
    // ----------------------------
    for (int i = 1; i < argc; ++i) {
        if (!starts_with(argv[i], "--")) continue;
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage (positional, legacy):\n");
            printf("  ./main_cuda Nx Ny Nz dt nsteps out_every csv_out_every elastic_enabled(0/1)\n");
            printf("\nFlags (optional):\n");
            printf("  --mode=dynamics|minimize\n");
            printf("  --pf-param-file <path>  required: load complete physical PF inputs (key=value)\n");
            printf("  --minimize-max-iter <n>\n");
            printf("  --minimize-dt <dt>\n");
            printf("  --eta-lambda-vol <value>   lambda_vol under-relaxation 阻尼系数 (default: 0.2, range: [0,1])\n");
            printf("  --minimize-rms-dY-threshold <val>    Y 收敛判据 (default: 5e-5)\n");
            printf("  --minimize-rms-res-threshold <val>   Euler-Lagrange/KKT 残差判据 (default: 1e-4)\n");
            printf("  --minimize-vol-err-rel-threshold <val>  体积约束相对误差判据 (default: 1e-4, V0<=0 时以第一步体积为参考)\n");
            printf("  --minimize-post-projection-iters <n>  后投影修正子步数 (default: 1, n>=0)\n");
            printf("  --temperature-C <val>   物理温度（摄氏度），默认 380.0\n");
            printf("  --V0 <val>              target <h(phi)>; if <=0 use initial mean_h\n");
            printf("  --radius <val>          seed radius for initialization (default: 10.0)\n");
            printf("  --minimize-resample-elastic-every N   true residual diagnostic interval; N<=0 disables\n");
            printf("  --elastic 0|1           override elastic (0=off, 1=on)\n");
            printf("\nInitialization (symmetry-breaking tests):\n");
            printf("  --init-shape sphere|ellipsoid\n");
            printf("  --init-axis-ratio-rx <val>\n");
            printf("  --init-axis-ratio-ry <val>\n");
            printf("  --init-axis-ratio-rz <val>\n");
            printf("  --init-tilt-theta-deg <val>\n");
            printf("  --init-tilt-phi-deg <val>\n");
            printf("  --init-center-shift-x <val>\n");
            printf("  --init-center-shift-y <val>\n");
            printf("  --init-center-shift-z <val>\n");
            printf("  --init-phi-noise-amp <val>\n");
            printf("  --init-phi-noise-seed <int>\n");
            printf("  --init-test-id <int>    preset index for batch tests (0..7, <0 to disable)\n");
            printf("\nMinimize 最小可运行示例（小网格，无弹性）：\n");
            printf("  ./main_cuda 32 32 32 0.05 100 10 10 0 --mode=minimize --minimize-max-iter 200 --minimize-dt 0.05\n");
            printf("Minimize 带弹性、指定 V0：\n");
            printf("  ./main_cuda 64 64 64 0.05 500 50 10 1 --mode=minimize --minimize-max-iter 500 --minimize-dt 0.05 --V0 0.1 --elastic 1\n");
            printf("\nMinimize full-model（带化学+扩散+体积约束）的示例：\n");
            printf("  ./main_cuda 64 64 64 0.05 500 50 10 1 --mode=minimize --minimize-full-model --minimize-max-iter 500 --minimize-dt 0.05 --V0 0.1 --elastic 1\n");
            return 0;
        }
        const char *v = NULL;
        if ((v = get_flag_value(argc, argv, &i, "--mode")) != NULL) {
            if (strcmp(v, "minimize") == 0) P.mode = 1;
            else P.mode = 0;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--pf-param-file")) != NULL) {
            (void)v;
            continue;
        }
        // 注意：--minimize-full-model / --minimize-full 是“布尔开关”而非带数值参数的选项，
        // 不能使用 get_flag_value，否则会错误吞掉下一个 flag（例如 --minimize-max-iter）。
        if (strcmp(argv[i], "--minimize-full-model") == 0 ||
            strcmp(argv[i], "--minimize-full") == 0) {
            // Minimization(full-model): enable chem+diffusion with volume Lagrange constraint
            P.minimize_full_model = 1;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--minimize-max-iter")) != NULL) {
            P.minimize_max_iter = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--minimize-dt")) != NULL) {
            P.minimize_dt = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--V0")) != NULL) {
            P.minimize_V0 = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--minimize-resample-elastic-every")) != NULL) {
            P.minimize_resample_elastic_every = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--minimize-rms-dphi-threshold")) != NULL) {
            P.minimize_rms_dphi_threshold = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--minimize-rms-dY-threshold")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--minimize-rms-dy-threshold")) != NULL) {
            P.minimize_rms_dY_threshold = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--minimize-energy-diff-rel-threshold")) != NULL) {
            P.minimize_energy_diff_rel_threshold = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--minimize-rms-res-for-energy-plateau")) != NULL) {
            P.minimize_rms_res_for_energy_plateau = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--minimize-rms-res-threshold")) != NULL) {
            P.minimize_rms_res_threshold = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--minimize-vol-err-rel-threshold")) != NULL) {
            P.minimize_vol_err_rel_threshold = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--minimize-convergence-steps")) != NULL) {
            P.minimize_convergence_steps = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--minimize-dt-safety-limit")) != NULL) {
            P.minimize_dt_safety_limit = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--eta-lambda-vol")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--eta_lambda_vol")) != NULL) {
            P.eta_lambda_vol = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--minimize-xB-max-safe")) != NULL) {
            P.minimize_xB_max_safe = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--minimize-post-projection-iters")) != NULL) {
            int iters = atoi(v);
            if (iters < 0) iters = 0;
            P.minimize_post_projection_iters = iters;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--radius")) != NULL) {
            P.ic_phi_seed_radius = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--elastic")) != NULL) {
            P.elastic_enabled = atoi(v) ? 1 : 0;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--init-case-tag")) != NULL) {
            strncpy(P.init_case_tag, v, sizeof(P.init_case_tag) - 1);
            P.init_case_tag[sizeof(P.init_case_tag) - 1] = '\0';
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--init-shape")) != NULL) {
            if (strcmp(v, "sphere") == 0) {
                P.init_shape_mode = 0;
            } else if (strcmp(v, "ellipsoid") == 0) {
                P.init_shape_mode = 1;
            }
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--init-axis-ratio-rx")) != NULL) {
            P.init_axis_ratio_rx = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--init-axis-ratio-ry")) != NULL) {
            P.init_axis_ratio_ry = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--init-axis-ratio-rz")) != NULL) {
            P.init_axis_ratio_rz = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--init-tilt-theta-deg")) != NULL) {
            P.init_tilt_theta_deg = atof(v);
            if (P.init_tilt_theta_deg != 0.0 || P.init_tilt_phi_deg != 0.0) {
                P.init_use_rotation = 1;
            }
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--init-tilt-phi-deg")) != NULL) {
            P.init_tilt_phi_deg = atof(v);
            if (P.init_tilt_theta_deg != 0.0 || P.init_tilt_phi_deg != 0.0) {
                P.init_use_rotation = 1;
            }
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--init-center-shift-x")) != NULL) {
            P.init_center_shift_x = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--init-center-shift-y")) != NULL) {
            P.init_center_shift_y = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--init-center-shift-z")) != NULL) {
            P.init_center_shift_z = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--init-phi-noise-amp")) != NULL) {
            P.init_phi_noise_amp = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--init-phi-noise-seed")) != NULL) {
            P.init_phi_noise_seed = (unsigned long)strtoul(v, NULL, 10);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--ic-23d-xB-out")) != NULL) {
            // 外部（φ≈0）区域初始 xB；若 >0 则忽略 ic_vf_target_phi，直接使用该值
            P.ic_23d_xB_out = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--temperature-C")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--temperature_C")) != NULL) {
            P.temperature_C = atof(v);
            continue;
        }
        // TODO: 若后续增加 --mu-reference-scale / --v-A / --v-B 等热力学相关 flag，
        // 也应在此处解析并在解析完成后调用 pfparams_refresh_thermo(&P)。
    }

    // 在解析完所有可选参数（包括温度等热力学参数）之后，刷新热力学量。
    pfparams_refresh_thermo(&P);
    P.xB_ref_for_eps_c = P.ic_xB_eq_matrix;
    if (!validate_physical_params_ready(&P)) {
        return 2;
    }

    if (P.mode == 1) {
        P.nsteps = P.minimize_max_iter;
        P.dt = P.minimize_dt;
    }

    if (P.Nx <= 0 || P.Ny <= 0 || P.Nz <= 0) {
        fprintf(stderr, "[fatal] Invalid grid size: Nx=%d Ny=%d Nz=%d. All dimensions must be > 0.\n",
                P.Nx, P.Ny, P.Nz);
        return 2;
    }
    if (P.dx <= 0.0 || P.dy <= 0.0 || P.dz <= 0.0) {
        fprintf(stderr, "[fatal] Invalid spatial spacing: dx=%.6e dy=%.6e dz=%.6e. All must be > 0.\n",
                P.dx, P.dy, P.dz);
        return 2;
    }
    if (P.dt <= 0.0) {
        fprintf(stderr, "[fatal] Invalid dt=%.6e. Time step must be > 0.\n", P.dt);
        return 2;
    }
    if (P.nsteps <= 0) {
        fprintf(stderr, "[fatal] Invalid nsteps=%d. Total step count must be > 0.\n", P.nsteps);
        return 2;
    }
    if (P.out_every <= 0) {
        fprintf(stderr, "[fatal] Invalid out_every=%d. VTK output interval must be > 0.\n", P.out_every);
        return 2;
    }
    if (P.csv_out_every <= 0) {
        fprintf(stderr, "[fatal] Invalid csv_out_every=%d. CSV output interval must be > 0.\n", P.csv_out_every);
        return 2;
    }
    if (P.mode == 1) {
        if (P.minimize_max_iter <= 0) {
            fprintf(stderr, "[fatal] Invalid minimize_max_iter=%d. It must be > 0.\n", P.minimize_max_iter);
            return 2;
        }
        if (P.minimize_dt <= 0.0) {
            fprintf(stderr, "[fatal] Invalid minimize_dt=%.6e. It must be > 0.\n", P.minimize_dt);
            return 2;
        }
    }

    P.elastic_enabled = P.elastic_enabled ? 1 : 0;

    if (P.out_every > P.nsteps) {
        fprintf(stderr, "[warn] 输出间隔 (%d) 大于总时间步数 (%d)，将调整为最后一步输出\n", P.out_every, P.nsteps);
        P.out_every = P.nsteps;
    }
    if (P.csv_out_every > P.nsteps) {
        fprintf(stderr, "[warn] CSV输出间隔 (%d) 大于总时间步数 (%d)，将调整为最后一步输出\n", P.csv_out_every, P.nsteps);
        P.csv_out_every = P.nsteps;
    }

    {
        double temperature_K = P.temperature_C + 273.15;
        char grid_buf[64];
        const double dx_phys_m = P.dx * 1.0e-9;
        const double dy_phys_m = P.dy * 1.0e-9;
        const double dz_phys_m = P.dz * 1.0e-9;
        const double Lx_phys_m = P.Nx * dx_phys_m;
        const double Ly_phys_m = P.Ny * dy_phys_m;
        const double Lz_phys_m = P.Nz * dz_phys_m;
        const double lambda_over_dx = (dx_phys_m > 0.0) ? (P.lambda_sm_m / dx_phys_m) : 0.0;
        const char *mode_label =
            (P.mode == 0) ? "dynamics"
                          : (P.minimize_full_model
                                ? "minimize (full-model: chem+diffusion+volume)"
                                : "minimize (phi-only)");
        snprintf(grid_buf, sizeof(grid_buf), "%dx%dx%d", P.Nx, P.Ny, P.Nz);

        log_section_header("Run Configuration");
        log_kv_text("mode", "%s", mode_label);
        if (pf_param_file != NULL) {
            log_kv_text("pf_param_file", "%s", pf_param_file);
        }
        log_kv_text("grid", "%s", grid_buf);
        log_kv_text("dx / dy / dz", "%.6f / %.6f / %.6f", P.dx, P.dy, P.dz);
        log_kv_text("cell_size_phys_nm", "%.6f / %.6f / %.6f",
                    dx_phys_m * 1.0e9, dy_phys_m * 1.0e9, dz_phys_m * 1.0e9);
        log_kv_text("system_size_phys_nm", "%.6f / %.6f / %.6f",
                    Lx_phys_m * 1.0e9, Ly_phys_m * 1.0e9, Lz_phys_m * 1.0e9);
        log_kv_text("system_size_phys_m", "%.6e / %.6e / %.6e",
                    Lx_phys_m, Ly_phys_m, Lz_phys_m);
        log_kv_text("lambda_sm_nm", "%.6f", P.lambda_sm_m * 1.0e9);
        log_kv_text("lambda_sm/dx", "%.6f", lambda_over_dx);
        if (lambda_over_dx <= 4.0) {
            fprintf(stderr,
                    "[warn] 界面分辨率不足: lambda_sm/dx = %.6f <= 4.000000，建议减小 dx 或增大 lambda_sm。\n",
                    lambda_over_dx);
        }
        log_kv_text("steps", "%d", P.nsteps);
        log_kv_text("dt", "%.6e", P.dt);
        log_kv_text("out_every", "%d", P.out_every);
        log_kv_text("csv_out_every", "%d", P.csv_out_every);
        log_kv_text("elastic", "%s", log_enabled_cn(P.elastic_enabled));
        log_kv_text("diag_vtk", "%s", log_enabled_cn(P.diag_vtk_enabled));
        log_kv_text("diag_elastic_bulk", "%s", log_enabled_cn(P.diag_elastic_bulk_penalty_enabled));

        log_section_header("Thermodynamics");
        log_kv_text("T_C", "%.6f", P.temperature_C);
        log_kv_text("T_K", "%.6f", temperature_K);
        log_kv_text("xB_eq", "%.8e", P.ic_xB_eq_matrix);
        log_kv_text("mu0_compound_hat", "%.8e", P.mu0_compound);
        log_kv_text("mu_reference_scale", "%.8e", P.mu_reference_scale);

        if (P.mode == 1) {
            log_section_header("Minimize Controls");
            log_kv_text("max_iter", "%d", P.minimize_max_iter);
            log_kv_text("minimize_dt", "%.3e", P.minimize_dt);
            log_kv_text("target_V0", "%.6e (0 means use initial <h>)", P.minimize_V0);
            log_kv_text("rms_dphi_threshold", "%.3e", P.minimize_rms_dphi_threshold);
            log_kv_text("rms_dY_threshold", "%.3e", P.minimize_rms_dY_threshold);
            log_kv_text("energy_diff_rel_threshold", "%.3e", P.minimize_energy_diff_rel_threshold);
            log_kv_text("rms_res_threshold", "%.3e", P.minimize_rms_res_threshold);
            log_kv_text("vol_err_rel_threshold", "%.3e", P.minimize_vol_err_rel_threshold);
            log_kv_text("convergence_steps", "%d", P.minimize_convergence_steps);
            log_kv_text("eta_lambda_vol", "%.3f", P.eta_lambda_vol);
            log_kv_text("post_projection_iters", "%d", P.minimize_post_projection_iters);
            log_kv_text("xB safety cap", "soft cap enabled, xB_max_safe=%.6f", P.minimize_xB_max_safe);
            log_kv_text("xB hard clamp", "%s", P.minimize_full_model ? "enabled in full-model minimize" : "disabled in phi-only minimize");
        } else {
            log_section_header("Dynamics Controls");
            log_kv_text("xB safety cap", "not applied in dynamics mode");
        }
        fflush(stdout);
    }
    
    // 创建输出目录
    // 约定：弹性关闭 -> ch 开头；弹性开启 -> chel 开头
    const char *out_prefix = P.elastic_enabled ? "chel" : "ch";
    const char *results_root = "Results";
    char run_dir_name[256];
    char output_dir[4096];
    if (P.diag_elastic_bulk_penalty_enabled || P.mode == 1) {
        snprintf(run_dir_name, sizeof(run_dir_name),
                 "%s_T%.0f_cuda_%dx%dx%d_dt%.3g_steps%d_r%.2f_xB%.3f",
                 out_prefix, P.temperature_C,
                 P.Nx, P.Ny, P.Nz, P.dt, P.nsteps,
                 P.ic_phi_seed_radius, P.ic_23d_xB_out);
    } else {
        snprintf(run_dir_name, sizeof(run_dir_name),
                 "%s_T%.0f_cuda_%dx%dx%d_dt%.3g_steps%d_xB%.3f",
                 out_prefix, P.temperature_C,
                 P.Nx, P.Ny, P.Nz, P.dt, P.nsteps,
                 P.ic_23d_xB_out);
    }
    mkdir(results_root, 0755);
    snprintf(output_dir, sizeof(output_dir), "%s/%s", results_root, run_dir_name);
    mkdir(output_dir, 0755);

    // 若用户未指定 init_case_tag，则根据初始化参数自动生成一个简洁标签
    if (P.init_case_tag[0] == '\0') {
        const char *shape_str = "legacy";
        if (P.init_shape_mode == 0) shape_str = "sphere";
        else if (P.init_shape_mode == 1) shape_str = "ellipsoid";
        snprintf(P.init_case_tag, sizeof(P.init_case_tag),
                 "case_test%d_%s_th%.0f_ph%.0f_noise%.0e",
                 P.init_test_id,
                 shape_str,
                 P.init_tilt_theta_deg,
                 P.init_tilt_phi_deg,
                 P.init_phi_noise_amp);
    }

    // 为当前 case 创建子目录：output_dir/init_case_tag
    // 路径/文件名缓冲区统一放大，避免 snprintf 潜在截断导致的 -Wformat-truncation 警告
    char case_output_dir[4096];
    snprintf(case_output_dir, sizeof(case_output_dir), "%s/%s", output_dir, P.init_case_tag);
    mkdir(case_output_dir, 0755);

    // 统一用于 VTK 文件名后缀（优先 init_case_tag；若为空则回退 case_<init_test_id>）
    char vtk_case_tag_buf[128];
    const char *vtk_case_tag = P.init_case_tag;
    if (vtk_case_tag[0] == '\0') {
        snprintf(vtk_case_tag_buf, sizeof(vtk_case_tag_buf), "case_%d", P.init_test_id);
        vtk_case_tag = vtk_case_tag_buf;
    }

    log_section_header("Output Layout");
    log_kv_text("output_root", "%s", output_dir);
    log_kv_text("init_case_tag", "%s", P.init_case_tag);
    log_kv_text("case_output_dir", "%s", case_output_dir);
    log_kv_text("vtk_mode", "%s", (P.mode == 0) ? "dynamic (no case suffix)" : "minimize (with case suffix)");
    if (P.mode != 0) {
        log_kv_text("vtk_filename_suffix", "%s", vtk_case_tag);
    }
    
    // 创建CSV文件：dynamics 用 vf_precip_vs_time.csv，minimize 用 energy_minimize.csv
    FILE *csv_fp = NULL;
    FILE *energy_fp = NULL; // minimize mode energy log
    if (P.mode == 0 || (P.mode == 1 && P.minimize_full_model == 1)) {
        char csv_path[4096];
        snprintf(csv_path, sizeof(csv_path), "%s/vf_precip_vs_time_%s.csv", case_output_dir, vtk_case_tag);
        csv_fp = fopen(csv_path, "w");
        if (!csv_fp) {
            fprintf(stderr, "[warn] 无法创建CSV文件 %s\n", csv_path);
        } else {
            if (P.diag_elastic_bulk_penalty_enabled) {
                fprintf(csv_fp, "step,t_code,t_real_s,vf_precip,R_avg,sum_h,sum_gel,E_el_bulk_hat,E_el_bulk_Jm3,Delta_mu_el_Jmol\n");
            } else {
                fprintf(csv_fp, "step,t_code,t_real_s,vf_precip,R_avg\n");
            }
            fflush(csv_fp);
        }
    }
    
    int NzC = P.Nz / 2 + 1;
    int total_r = P.Nx * P.Ny * P.Nz;
    int total_k = P.Nx * P.Ny * NzC;
    size_t size_r = total_r * sizeof(double);
    size_t size_k = total_k * sizeof(cufftDoubleComplex);
    
    double temperature_K = P.temperature_C + 273.15;
    double invN = 1.0 / (double)total_r;
    // baseline for bulk chemical free-energy (stoichiometric compound PF)
    double xB_ref = NAN;
    double g_bulk0_hat = NAN;
    double F_el0_hat = 0.0;  // elastic baseline: 默认 0，可选后续测一次 phi=0,xB=xB_ref
    double muB_farfield_hat = NAN; // CNT far-field chemical potential (hat)
    
    // 分配CPU内存（用于初始化和输出）
    double *h_phi_r = (double*)malloc(size_r);
    double *h_Y_r = (double*)malloc(size_r);
    double *h_xB_r = (double*)malloc(size_r);
    double *h_xBtot_r = (double*)malloc(size_r);
    
    // 优化：不再分配d_diag_stats，使用直接归约函数节省显存
    
    // 初始化场
    if (P.mode == 1 && P.minimize_full_model == 0) {
        // 旧行为：phi-only minimize，不初始化xB/Y
        log_section_header("Initialization");
        log_kv_text("path", "phi-only minimize");
        fflush(stdout);
        initialize_phi_only_cuda(h_phi_r, &P, total_r);
        memset(h_Y_r, 0, size_r);
        memset(h_xB_r, 0, size_r);
        memset(h_xBtot_r, 0, size_r);
    } else {
        // dynamics 与 full-model minimize 共用：初始化 phi/Y/xB
        log_section_header("Initialization");
        log_kv_text("path", "mass-conserving phi/xB/Y");
        fflush(stdout);
        initialize_fields_cuda(h_phi_r, h_Y_r, h_xB_r, &P, total_r);
        for (int i = 0; i < total_r; i++) {
            double phi = h_phi_r[i];
            double h = h_of_phi(phi);
            double xB = clamp01(h_xB_r[i]);
            h_xBtot_r[i] = (1.0 - h) * xB + P.v_B * h;
        }
    }

    // 选择 bulk chemical baseline 组成 xB_ref，并计算 g_bulk0_hat
    if (P.mode == 1 && P.minimize_full_model == 1) {
        if (P.ic_23d_xB_out > 0.0) {
            xB_ref = P.ic_23d_xB_out;
            log_section_header("Reference State");
            log_kv_text("xB_ref source", "ic_23d_xB_out");
            log_kv_text("xB_ref", "%.8e", xB_ref);
        } else {
            // 退化情况：未指定 far-field 组成，使用初始场的 <xB> 作为 baseline
            double sum_xB0 = 0.0;
            for (int i = 0; i < total_r; ++i) {
                sum_xB0 += h_xB_r[i];
            }
            xB_ref = sum_xB0 / (double)total_r;
            log_section_header("Reference State");
            log_kv_text("xB_ref source", "<xB>_init");
            log_kv_text("xB_ref", "%.8e", xB_ref);
        }
        // host 侧计算 mu_mix_hat(xB_ref)，并据此给出 g_bulk0_hat
        double muA_ref = mu_A_dimless(xB_ref, temperature_K, P.mu_reference_scale);
        double muB_ref = mu_B_dimless(xB_ref, temperature_K, P.mu_reference_scale);
        double mu_mix_ref = (1.0 - xB_ref) * muA_ref + xB_ref * muB_ref;
        // c_tot_hat 当前取 1.0，与其他 hat 量保持一致
        g_bulk0_hat = mu_mix_ref;
        log_kv_text("g_bulk0_hat", "%.8e", g_bulk0_hat);

        if (P.ic_23d_xB_out > 0.0) {
            muB_farfield_hat = mu_B_dimless(P.ic_23d_xB_out, temperature_K, P.mu_reference_scale);
            log_kv_text("CNT far-field xB_out", "%.8e", P.ic_23d_xB_out);
            log_kv_text("CNT muB_farfield_hat", "%.8e", muB_farfield_hat);
        } else {
            muB_farfield_hat = NAN;
            log_kv_text("CNT far-field", "skipped (ic_23d_xB_out <= 0)");
        }
    }

    // minimize 模式能量 CSV（放在 baseline 计算之后，便于在 header 中记录 xB_ref）
    if (P.mode == 1) {
        char epath[4096];
        snprintf(epath, sizeof(epath), "%s/energy_minimize_T%.0f_r%.2f_xB%.3f_%s.csv",
                 case_output_dir, P.temperature_C, P.ic_phi_seed_radius, P.ic_23d_xB_out, vtk_case_tag);
        energy_fp = fopen(epath, "w");
        if (!energy_fp) {
            fprintf(stderr, "[warn] cannot open energy csv %s\n", epath);
        } else {
            // Minimization energy log (excess-free-energy + CNT-based summaries):
            // iter,dt,mean_h,V0,lambda,
            // F_surf_hat,F_el_hat,F_chem_excess_hat,F_total_excess_hat,
            // F_chem_CNT_hat,F_total_CNT_hat,
            // total_interface_sum,total_el_core_sum,
            // rms_res,rms_dphi,rms_dY,rel_dF,vol_err_rel,post_proj_iters_used,
            // Len_x,Len_y,Len_z,Len_x_Len_z
            fprintf(energy_fp,
                    "iter,dt,mean_h,V0,lambda,"
                    "F_surf_hat,F_el_hat,F_chem_excess_hat,F_total_excess_hat,"
                    "F_chem_CNT_hat,F_total_CNT_hat,"
                    "total_interface_sum,total_el_core_sum,"
                    "rms_res,rms_dphi,rms_dY,rel_dF,vol_err_rel,post_proj_iters_used,"
                    "Len_x,Len_y,Len_z,Len_x_Len_z\n");
            fflush(energy_fp);
        }
    }

    // 输出t=0的VTK文件（先将CPU数据复制到临时GPU数组）
    char filename[4096];  // 用于VTK输出文件名
    double *d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, size_r));

    if (P.mode == 0) {
        CUDA_CHECK(cudaMemcpy(d_temp, h_xBtot_r, size_r, cudaMemcpyHostToDevice));
        build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "xBtot", VTK_NAME_STEP, 0, vtk_case_tag, 0);
        int ret1 = write_vtk_cuda(d_temp, P.Nx, P.Ny, P.Nz, "xB_tot", 0, filename);
        if (!ret1) {
            fprintf(stderr, "ERROR: Failed to write initial xBtot VTK file: %s\n", filename);
        }
    }

    // 初始 phi VTK：dynamic 输出到 output_dir（无 case）；minimize 输出到 case_output_dir（带 case）
    CUDA_CHECK(cudaMemcpy(d_temp, h_phi_r, size_r, cudaMemcpyHostToDevice));
    build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "phi", VTK_NAME_INIT, 0, vtk_case_tag, P.mode);
    int ret_phi_init = write_vtk_cuda(d_temp, P.Nx, P.Ny, P.Nz, "phi", 0, filename);
    if (!ret_phi_init) {
        fprintf(stderr, "ERROR: Failed to write initial phi VTK file: %s\n", filename);
    } else {
        printf("写出初始化 vtk: %s\n", filename);
    }

    if (P.mode == 0) {
        CUDA_CHECK(cudaMemcpy(d_temp, h_xB_r, size_r, cudaMemcpyHostToDevice));
        build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "xB", VTK_NAME_STEP, 0, vtk_case_tag, 0);
        int ret3 = write_vtk_cuda(d_temp, P.Nx, P.Ny, P.Nz, "xB", 0, filename);
        if (!ret3) {
            fprintf(stderr, "ERROR: Failed to write initial xB VTK file: %s\n", filename);
        }
    }

    // 诊断VTK输出：仅 dynamics 模式；minimize 模式不输出 xB 相关
    if (P.diag_vtk_enabled && P.mode == 0) {
        // 计算并输出初始delta_mu_r VTK文件
        // 需要同时使用phi和xB数据，所以需要两个临时数组
        double *d_phi_temp = NULL;
        double *d_xB_temp = NULL;
        double *d_delta_mu_r_init = NULL;
        CUDA_CHECK(cudaMalloc(&d_phi_temp, size_r));
        CUDA_CHECK(cudaMalloc(&d_xB_temp, size_r));
        CUDA_CHECK(cudaMalloc(&d_delta_mu_r_init, size_r));
        
        CUDA_CHECK(cudaMemcpy(d_phi_temp, h_phi_r, size_r, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_xB_temp, h_xB_r, size_r, cudaMemcpyHostToDevice));
        launch_compute_delta_mu_r_kernel(d_phi_temp, d_xB_temp, d_delta_mu_r_init,
                                        temperature_K, P.mu_reference_scale,
                                        P.v_A, P.v_B, P.mu0_compound,
                                        (P.elastic_enabled ? P.elastic_shift_dimless : 0.0),
                                        total_r);
        build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "delta_mu_r", VTK_NAME_STEP, 0, vtk_case_tag, 0);
        int ret4 = write_vtk_cuda(d_delta_mu_r_init, P.Nx, P.Ny, P.Nz, "delta_mu_r", 0, filename);
        if (!ret4) {
            fprintf(stderr, "ERROR: Failed to write initial delta_mu_r VTK file: %s\n", filename);
        }
        CUDA_CHECK(cudaFree(d_phi_temp));
        CUDA_CHECK(cudaFree(d_xB_temp));
        CUDA_CHECK(cudaFree(d_delta_mu_r_init));
        
        // 计算并输出初始f_phi_chem和dgel_dphi VTK文件
        // 注意：初始时可能还没有应力场，所以dgel_dphi在未启用弹性时将为0
        double *d_f_phi_chem_init = NULL;
        double *d_dgel_dphi_init = NULL;
        CUDA_CHECK(cudaMalloc(&d_f_phi_chem_init, size_r));
        CUDA_CHECK(cudaMalloc(&d_dgel_dphi_init, size_r));
        
        // 如果启用弹性，需要临时分配应力场和应变数组（初始时均为0）
        float *d_sigma_init_xx = NULL, *d_sigma_init_yy = NULL, *d_sigma_init_zz = NULL;
        float *d_sigma_init_xy = NULL, *d_sigma_init_xz = NULL, *d_sigma_init_yz = NULL;
        float *d_uxx_init_r = NULL, *d_uyy_init_r = NULL, *d_uzz_init_r = NULL;
        float *d_uxy_init_r = NULL, *d_uxz_init_r = NULL, *d_uyz_init_r = NULL;
        // Optimization(4): d_uxx0_init_r..d_uyz0_init_r 已移除，eps0 现场计算
        if (P.elastic_enabled) {
            size_t size_r_float = total_r * sizeof(float);
            CUDA_CHECK(cudaMalloc(&d_sigma_init_xx, size_r_float));
            CUDA_CHECK(cudaMalloc(&d_sigma_init_yy, size_r_float));
            CUDA_CHECK(cudaMalloc(&d_sigma_init_zz, size_r_float));
            CUDA_CHECK(cudaMalloc(&d_sigma_init_xy, size_r_float));
            CUDA_CHECK(cudaMalloc(&d_sigma_init_xz, size_r_float));
            CUDA_CHECK(cudaMalloc(&d_sigma_init_yz, size_r_float));
            CUDA_CHECK(cudaMalloc(&d_uxx_init_r, size_r_float));
            CUDA_CHECK(cudaMalloc(&d_uyy_init_r, size_r_float));
            CUDA_CHECK(cudaMalloc(&d_uzz_init_r, size_r_float));
            CUDA_CHECK(cudaMalloc(&d_uxy_init_r, size_r_float));
            CUDA_CHECK(cudaMalloc(&d_uxz_init_r, size_r_float));
            CUDA_CHECK(cudaMalloc(&d_uyz_init_r, size_r_float));
            // Optimization(4): d_uxx0_init_r..d_uyz0_init_r 的分配已移除
            CUDA_CHECK(cudaMemset(d_sigma_init_xx, 0, size_r_float));
            CUDA_CHECK(cudaMemset(d_sigma_init_yy, 0, size_r_float));
            CUDA_CHECK(cudaMemset(d_sigma_init_zz, 0, size_r_float));
            CUDA_CHECK(cudaMemset(d_sigma_init_xy, 0, size_r_float));
            CUDA_CHECK(cudaMemset(d_sigma_init_xz, 0, size_r_float));
            CUDA_CHECK(cudaMemset(d_sigma_init_yz, 0, size_r_float));
            CUDA_CHECK(cudaMemset(d_uxx_init_r, 0, size_r_float));
            CUDA_CHECK(cudaMemset(d_uyy_init_r, 0, size_r_float));
            CUDA_CHECK(cudaMemset(d_uzz_init_r, 0, size_r_float));
            CUDA_CHECK(cudaMemset(d_uxy_init_r, 0, size_r_float));
            CUDA_CHECK(cudaMemset(d_uxz_init_r, 0, size_r_float));
            CUDA_CHECK(cudaMemset(d_uyz_init_r, 0, size_r_float));
            // Optimization(4): d_uxx0_init_r..d_uyz0_init_r 的 memset 已移除
        }
        
        // 重新分配临时数组用于初始输出
        CUDA_CHECK(cudaMalloc(&d_phi_temp, size_r));
        CUDA_CHECK(cudaMalloc(&d_xB_temp, size_r));
        CUDA_CHECK(cudaMemcpy(d_phi_temp, h_phi_r, size_r, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_xB_temp, h_xB_r, size_r, cudaMemcpyHostToDevice));
        
        launch_compute_f_phi_chem_dgel_dphi_kernel(
            d_phi_temp, d_xB_temp, d_f_phi_chem_init, NULL, NULL, d_dgel_dphi_init,
            temperature_K, P.mu_reference_scale,
            P.v_A, P.v_B, P.mu0_compound,
            P.Vm_compound, P.Vm_alpha_0, P.dVm_alpha_dxB, P.W,
            d_sigma_init_xx, d_sigma_init_yy, d_sigma_init_zz,
            d_sigma_init_xy, d_sigma_init_xz, d_sigma_init_yz,
            d_uxx_init_r, d_uyy_init_r, d_uzz_init_r, d_uxy_init_r, d_uxz_init_r, d_uyz_init_r,
            // Optimization(4): d_uxx0_init_r..d_uyz0_init_r 参数已移除
            P.S_p_11, P.S_p_12, P.S_p_13, P.S_p_14, P.S_p_15, P.S_p_16,
            P.S_p_22, P.S_p_23, P.S_p_24, P.S_p_25, P.S_p_26,
            P.S_p_33, P.S_p_34, P.S_p_35, P.S_p_36,
            P.S_p_44, P.S_p_45, P.S_p_46,
            P.S_p_55, P.S_p_56, P.S_p_66,
            P.eps_xx00, P.eps_yy00, P.eps_zz00,
            P.eps_yz00, P.eps_xz00, P.eps_xy00,
            P.eps_iso_over_vB,
            (P.elastic_enabled ? P.elastic_shift_dimless : 0.0),
            total_r,
            P.elastic_enabled);
        build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "f_phi_chem", VTK_NAME_STEP, 0, vtk_case_tag, 0);
        int ret5 = write_vtk_cuda(d_f_phi_chem_init, P.Nx, P.Ny, P.Nz, "f_phi_chem", 0, filename);
        if (!ret5) {
            fprintf(stderr, "ERROR: Failed to write initial f_phi_chem VTK file: %s\n", filename);
        }
        build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "dgel_dphi", VTK_NAME_STEP, 0, vtk_case_tag, 0);
        int ret6 = write_vtk_cuda(d_dgel_dphi_init, P.Nx, P.Ny, P.Nz, "dgel_dphi", 0, filename);
        if (!ret6) {
            fprintf(stderr, "ERROR: Failed to write initial dgel_dphi VTK file: %s\n", filename);
        }
        CUDA_CHECK(cudaFree(d_f_phi_chem_init));
        CUDA_CHECK(cudaFree(d_dgel_dphi_init));
        CUDA_CHECK(cudaFree(d_phi_temp));
        CUDA_CHECK(cudaFree(d_xB_temp));
        // 释放临时应力场和应变数组
        if (P.elastic_enabled) {
            CUDA_CHECK(cudaFree(d_sigma_init_xx));
            CUDA_CHECK(cudaFree(d_sigma_init_yy));
            CUDA_CHECK(cudaFree(d_sigma_init_zz));
            CUDA_CHECK(cudaFree(d_sigma_init_xy));
            CUDA_CHECK(cudaFree(d_sigma_init_xz));
            CUDA_CHECK(cudaFree(d_sigma_init_yz));
            CUDA_CHECK(cudaFree(d_uxx_init_r));
            CUDA_CHECK(cudaFree(d_uyy_init_r));
            CUDA_CHECK(cudaFree(d_uzz_init_r));
            CUDA_CHECK(cudaFree(d_uxy_init_r));
            CUDA_CHECK(cudaFree(d_uxz_init_r));
            CUDA_CHECK(cudaFree(d_uyz_init_r));
            // Optimization(4): d_uxx0_init_r..d_uyz0_init_r 的释放已移除
        }
    
        // 如果启用弹性，输出初始本征应变 eigenstrain_xx_0.vtk
        if (P.elastic_enabled) {
            // 此时d_uxx0_r尚未计算，我们需要先根据初始phi计算并输出
            float *h_uxx0_r_init = (float*)malloc(total_r * sizeof(float));
            double eps_xx00 = P.eps_xx00;
            double eps_iso_over_vB = P.eps_iso_over_vB;
            double v_B = P.v_B;
    
            for (int i = 0; i < total_r; i++) {
                double phi = h_phi_r[i];
                double xB = clamp01(h_xB_r[i]);
                double h = h_of_phi(phi);
                h_uxx0_r_init[i] = (float)(h * eps_xx00 + eps_iso_over_vB * (h * xB - h * v_B));
            }
            
            // 转换为double用于VTK输出
            double *h_tmp_double = (double*)malloc(size_r);
            for (int i = 0; i < total_r; i++) h_tmp_double[i] = (double)h_uxx0_r_init[i];
            CUDA_CHECK(cudaMemcpy(d_temp, h_tmp_double, size_r, cudaMemcpyHostToDevice));
    
            build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "eigenstrain_xx", VTK_NAME_STEP, 0, vtk_case_tag, 0);
            write_vtk_cuda(d_temp, P.Nx, P.Ny, P.Nz, "eigenstrain_xx", 0, filename);
            
            free(h_uxx0_r_init);
            free(h_tmp_double);
            printf("已输出初始本征应变文件: %s\n", filename);
        }
    }
    
    CUDA_CHECK(cudaFree(d_temp));
    
    // 输出t=0的诊断统计（需要在GPU内存分配后进行）
    int *d_bbox_mins = NULL, *d_bbox_maxs = NULL;
    double *d_boundary_sum = NULL;
    unsigned long long *d_boundary_count = NULL;
    CUDA_CHECK(cudaMalloc(&d_bbox_mins, 3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bbox_maxs, 3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_boundary_sum, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_boundary_count, sizeof(unsigned long long)));
    
    /*
     * Phase 生命周期表（每时间步）
     * P0: 输入/边界/初始化 - phi_n_saved, Y_n_saved
     * P1: phi 相关 - phi_r→phi_k→phi_rhs→semi_implicit→phi_r; d_phi_rhs_k 可复用
     * P2: Y/xB 相关 - mu_x, divJ 串行累加(scratch), lapY, Y_rhs, semi_implicit; d_divJ_r 与 d_xB_prev_r 复用
     * P3: 弹性(若启用) - eigenstrain, displacement, strain, stress; 应变由 k-space 位移直接计算
     * P4: 诊断/VTK - scratch_r_double 2 slot 串行复用
     * 详见 PHASE_LIFECYCLE.md
     */
    // 分配GPU内存
    printf("分配GPU内存...\n");
    print_gpu_meminfo("before main cudaMallocs");
    double *d_phi_r, *d_Y_r, *d_xB_r;
    double *d_phi_rhs_r, *d_phi_n_saved, *d_Y_n_saved, *d_dY_dt_prev_r;
    // 优化：移除d_xBtot_r，仅在需要输出VTK时临时计算
    
    // 工作空间（Y方程相关）
    double *d_mu_x_r;
    // Optimization: d_xB_prev_r 和 d_divJ_r 复用同一内存
    double *d_divJ_r, *d_xB_prev_r;
    // Optimization: d_phi_rhs_r 在步骤1完成后可复用为 d_lapY_r
    double *d_lapY_r, *d_Y_rhs_r;

    // k空间数组
    // Optimization: phi_k 与 Y_k 使用 in-place 更新，不再常驻 phi_k_new、Y_k_new
    cufftDoubleComplex *d_phi_k, *d_phi_rhs_k;
    cufftDoubleComplex *d_Y_k;
    // Optimization: d_phi_rhs_k 在步骤1完成后可复用为 d_mu_x_k，然后复用为 d_Y_rhs_k
    cufftDoubleComplex *d_mu_x_k, *d_Y_rhs_k;
    // Optimization: 仅保留 divJ_k 常驻，grad/J 的 y/z 分量改为串行累加使用 scratch
    cufftDoubleComplex *d_divJ_k;

    // Optimization: 统一 scratch 池，用于 divJ 串行累加及 VTK 输出等
    void *d_scratch_k_double = NULL;
    void *d_scratch_r_double = NULL;
    size_t scratch_k_double_bytes = size_k;
    // Optimization: 输出步 2-slot 串行复用，scratch 从 6×size_r 降至 2×size_r
    size_t scratch_r_double_bytes = (size_t)2 * size_r;

    CUDA_CHECK(cudaMalloc(&d_phi_r, size_r));
    CUDA_CHECK(cudaMalloc(&d_Y_r, size_r));
    CUDA_CHECK(cudaMalloc(&d_xB_r, size_r));
    CUDA_CHECK(cudaMalloc(&d_phi_rhs_r, size_r));
    CUDA_CHECK(cudaMalloc(&d_phi_n_saved, size_r));
    CUDA_CHECK(cudaMalloc(&d_Y_n_saved, size_r));
    CUDA_CHECK(cudaMalloc(&d_dY_dt_prev_r, size_r));
    // 优化：不再分配d_xBtot_r，节省1GB显存
    // 优化：d_phi_rhs_r 在步骤1完成后可复用为 d_lapY_r（节省1GB）
    d_lapY_r = d_phi_rhs_r;  // 复用指针
    
    CUDA_CHECK(cudaMalloc(&d_mu_x_r, size_r));
    // Optimization: d_Y_rhs_r 复用 d_mu_x_r（Y_rhs 在 mu_x 完全使用后写入）
    d_Y_rhs_r = d_mu_x_r;
    // Optimization: 移除 grad_mu_x/y/z_r 与 Jx/Jy/Jz_r/k 常驻，改用 scratch + divJ 串行累加
    CUDA_CHECK(cudaMalloc(&d_divJ_r, size_r));
    d_xB_prev_r = d_divJ_r;
    // 优化：不再分配d_DY_values，节省1GB显存
    
    CUDA_CHECK(cudaMalloc(&d_phi_k, size_k));
    CUDA_CHECK(cudaMalloc(&d_phi_rhs_k, size_k));
    CUDA_CHECK(cudaMalloc(&d_Y_k, size_k));
    // 优化：d_phi_rhs_k 在步骤1完成后可复用为 d_mu_x_k，然后复用为 d_Y_rhs_k
    // 节省2GB显存（k空间）
    d_mu_x_k = d_phi_rhs_k;
    d_Y_rhs_k = d_phi_rhs_k;
    // Optimization: 仅常驻 divJ_k，grad/J 用 scratch 串行累加
    CUDA_CHECK(cudaMalloc(&d_divJ_k, size_k));
    CUDA_CHECK(cudaMalloc(&d_scratch_k_double, scratch_k_double_bytes));
    CUDA_CHECK(cudaMalloc(&d_scratch_r_double, scratch_r_double_bytes));
    // Optimization: Scratch Arena 初始化（用于 divJ 串行累加等，phase 内复用）
    ScratchArena arena_k = {0}, arena_r = {0};
    scratch_arena_init(&arena_k, d_scratch_k_double, scratch_k_double_bytes, "scratch_k");
    scratch_arena_init(&arena_r, d_scratch_r_double, scratch_r_double_bytes, "scratch_r");
    
    // ============================================================
    // 弹性计算相关数组分配（如果启用弹性计算）
    // ============================================================
    // Optimization(4): d_uxx0_r..d_uyz0_r 已移除，eps0 现场计算；临时 eps0_r 写入 d_uxx_r..d_uyz_r
    // Optimization: 删除 d_ux_r/d_uy_r/d_uz_r 常驻；应变由 k-space 位移直接计算，r-space 位移不参与后续计算
    float *d_uxx_r = NULL, *d_uyy_r = NULL, *d_uzz_r = NULL;
    float *d_uxy_r = NULL, *d_uxz_r = NULL, *d_uyz_r = NULL;
    // 实空间应力（用于弹性能量变分）
    float *d_sigma_xx_r = NULL, *d_sigma_yy_r = NULL, *d_sigma_zz_r = NULL;
    float *d_sigma_xy_r = NULL, *d_sigma_xz_r = NULL, *d_sigma_yz_r = NULL;
    // 注意：不再需要d_S_pert_*数组，perturbation直接在kernel内计算
    // Optimization(3): 移除 d_uxx0_k..d_uyz0_k，复用 d_uxx_k..d_uyz_k 作为临时 eigenstrain_k 容器
    
    cufftComplex *d_ux_k = NULL, *d_uy_k = NULL, *d_uz_k = NULL;
    cufftComplex *d_uxx_k = NULL, *d_uyy_k = NULL, *d_uzz_k = NULL;
    cufftComplex *d_uxy_k = NULL, *d_uxz_k = NULL, *d_uyz_k = NULL;
    
    // Optimization: 移除 d_elastic_tmp_r/d_elastic_tmp_k，弹性 FFT 直接 out-of-place
    
    size_t size_r_float = total_r * sizeof(float);
    size_t size_k_float = total_k * sizeof(cufftComplex);
    
    if (P.elastic_enabled) {
        printf("分配弹性计算GPU内存（float精度）...\n");
        print_gpu_meminfo("before elastic cudaMallocs");
        
        // 实空间数组
        // Optimization(4): d_uxx0_r..d_uyz0_r 已移除，eps0 现场计算；d_uxx_r..d_uyz_r 用于临时 eps0_r 和真实 strain_r
        CUDA_CHECK(cudaMalloc(&d_uxx_r, size_r_float));
        CUDA_CHECK(cudaMalloc(&d_uyy_r, size_r_float));
        CUDA_CHECK(cudaMalloc(&d_uzz_r, size_r_float));
        CUDA_CHECK(cudaMalloc(&d_uxy_r, size_r_float));
        CUDA_CHECK(cudaMalloc(&d_uxz_r, size_r_float));
        CUDA_CHECK(cudaMalloc(&d_uyz_r, size_r_float));
        // 应力场
        CUDA_CHECK(cudaMalloc(&d_sigma_xx_r, size_r_float));
        CUDA_CHECK(cudaMalloc(&d_sigma_yy_r, size_r_float));
        CUDA_CHECK(cudaMalloc(&d_sigma_zz_r, size_r_float));
        CUDA_CHECK(cudaMalloc(&d_sigma_xy_r, size_r_float));
        CUDA_CHECK(cudaMalloc(&d_sigma_xz_r, size_r_float));
        CUDA_CHECK(cudaMalloc(&d_sigma_yz_r, size_r_float));
        
        // 注意：不再分配perturbation数组，直接在kernel内计算以节省显存
        
        // k空间数组
        // Optimization(3): d_uxx0_k..d_uyz0_k 已移除，复用 d_uxx_k..d_uyz_k 作为临时 eigenstrain_k
        CUDA_CHECK(cudaMalloc(&d_ux_k, size_k_float));
        CUDA_CHECK(cudaMalloc(&d_uy_k, size_k_float));
        CUDA_CHECK(cudaMalloc(&d_uz_k, size_k_float));
        CUDA_CHECK(cudaMalloc(&d_uxx_k, size_k_float));
        CUDA_CHECK(cudaMalloc(&d_uyy_k, size_k_float));
        CUDA_CHECK(cudaMalloc(&d_uzz_k, size_k_float));
        CUDA_CHECK(cudaMalloc(&d_uxy_k, size_k_float));
        CUDA_CHECK(cudaMalloc(&d_uxz_k, size_k_float));
        CUDA_CHECK(cudaMalloc(&d_uyz_k, size_k_float));
        
        // FFT临时缓冲区（复用）
        
        printf("弹性计算GPU内存分配完成\n");
        print_gpu_meminfo("after elastic cudaMallocs");
    }

    print_gpu_meminfo("after main cudaMallocs");
    // Optimization: 显存账本 - 启动时打印一次
    print_memory_ledger(size_r, size_k, size_r_float, size_k_float,
                       scratch_k_double_bytes, scratch_r_double_bytes,
                       P.elastic_enabled ? 1 : 0, (int)total_r, (int)total_k);
    
    // 复制初始数据到GPU
    CUDA_CHECK(cudaMemcpy(d_phi_r, h_phi_r, size_r, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y_r, h_Y_r, size_r, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xB_r, h_xB_r, size_r, cudaMemcpyHostToDevice));

    // minimize mode: phi-only, no xB rebuild
    // 初始化工作数组
    CUDA_CHECK(cudaMemset(d_dY_dt_prev_r, 0, size_r));
    
    // 创建cuFFT计划（优化：复用计划以减少内存使用）
    printf("创建cuFFT计划...\n");
    printf("  网格: %dx%dx%d\n", P.Nx, P.Ny, P.Nz);
    
    // 复用计划：只创建2个计划（R2C和C2R），所有FFT操作共享
    cufftHandle plan_r2c_base, plan_c2r_base;
    
    CUFFT_CHECK(cufftPlan3d(&plan_r2c_base, P.Nx, P.Ny, P.Nz, CUFFT_D2Z));
    CUFFT_CHECK(cufftPlan3d(&plan_c2r_base, P.Nx, P.Ny, P.Nz, CUFFT_Z2D));
    
    // 弹性计算的float FFT计划（如果启用弹性计算）
    cufftHandle plan_r2c_elastic = 0;
    cufftHandle plan_c2r_elastic = 0;
    if (P.elastic_enabled) {
        CUFFT_CHECK(cufftPlan3d(&plan_r2c_elastic, P.Nx, P.Ny, P.Nz, CUFFT_R2C));
        CUFFT_CHECK(cufftPlan3d(&plan_c2r_elastic, P.Nx, P.Ny, P.Nz, CUFFT_C2R));
        printf("弹性FFT计划创建成功（float精度）\n");
    }
    
    printf("cuFFT计划创建成功（复用计划，减少内存使用）\n");
    
    // 构建k空间
    printf("构建k空间...\n");
    KSpace_CUDA KS;
    kspace_build_cuda(&KS, P.Nx, P.Ny, P.Nz, P.dx, P.dy, P.dz);
    print_gpu_meminfo("before minimize work buffers");

    // minimize mode: reusable energy work buffers
    // Minimization(full-model): reuse scratch pool instead of stealing Y/xB
    double *d_energy_tmp = NULL;   // e_bulk / e_grad / rhs^2 / reductions
    double *d_gel_hat_tmp = NULL;  // optional elastic energy density (gel_hat)
    double *d_lap_phi_r = NULL;    // Lap(phi) 实空间缓冲
    double *d_res_r = NULL;        // rms_res 诊断专用
    if (P.mode == 1) {
        if (P.minimize_full_model) {
            // Minimization(full-model): 使用 scratch pool，避免占用 Y/xB
            double *scratch_r = (double *)d_scratch_r_double; // 长度 ≥ 2*total_r
            d_energy_tmp = scratch_r;
            d_lap_phi_r = scratch_r + total_r;
            d_res_r = scratch_r;  // 诊断阶段复用 energy 缓冲
            d_gel_hat_tmp = P.elastic_enabled ? (scratch_r + total_r) : NULL;
        } else {
            // 旧 phi-only minimize 策略：复用 Y/xB 相关数组作为工作区
            d_energy_tmp = d_Y_r;
            d_lap_phi_r = d_xB_r;
            d_res_r = d_Y_n_saved;
            d_gel_hat_tmp = P.elastic_enabled ? d_dY_dt_prev_r : NULL;
        }
    }
    
    cufftHandle plan_r2c_phi = plan_r2c_base;
    cufftHandle plan_c2r_phi = plan_c2r_base;
    cufftHandle plan_r2c_phi_rhs = plan_r2c_base;
    cufftHandle plan_r2c_Y = plan_r2c_base;
    cufftHandle plan_c2r_Y = plan_c2r_base;
    cufftHandle plan_r2c_xB = plan_r2c_base;
    cufftHandle plan_c2r_xB = plan_c2r_base;
    
    
    // 输出初始化后的显存使用情况
    {
        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        size_t used_mem = total_mem - free_mem;
        
        printf("\n========================================\n");
        printf("初始化完成后的显存使用情况:\n");
        printf("  网格大小: %dx%dx%d\n", P.Nx, P.Ny, P.Nz);
        printf("  已使用: %.2f GB (%.2f%%)\n", 
               used_mem / (1024.0*1024.0*1024.0),
               (double)used_mem / (double)total_mem * 100.0);
        printf("  空闲: %.2f GB\n", free_mem / (1024.0*1024.0*1024.0));
        printf("  总计: %.2f GB\n", total_mem / (1024.0*1024.0*1024.0));
        if (P.elastic_enabled) {
            printf("  弹性功能: 已启用\n");
        } else {
            printf("  弹性功能: 未启用\n");
        }
        printf("========================================\n\n");
    }
    
    // 优化：使用直接归约计算N_in和N_if，不需要中间数组
    double global_N_in, global_N_if;
    gpu_reduce_sum_N_in_N_if(d_phi_r, total_r, &global_N_in, &global_N_if);
    double vf_precip = gpu_compute_vf_from_h(d_phi_r, total_r); // <h(phi)>
    
    double R_avg = 0.0;
    if (global_N_if > 0.0) {
        // N_if 除以 lambda = ic_phi_iface_w * 2.0
        double lambda = P.ic_phi_iface_w * 2.0;
        double global_N_if_normalized = global_N_if / lambda;
        double ratio = global_N_in / global_N_if_normalized;
        if (P.Ny == 2) {
            R_avg = 2.0 * P.dx * ratio;
        } else {
            R_avg = 3.0 * P.dx * ratio;
        }
    }
    
    double mean_xB_tot = NAN;
    double min_xB = NAN, max_xB = NAN;
    double min_Meff = NAN, max_Meff = NAN;
    if (P.mode == 0 || (P.mode == 1 && P.minimize_full_model == 1)) {
        mean_xB_tot = gpu_reduce_sum_xBtot(d_phi_r, d_xB_r, P.v_B, total_r) / (double)total_r;
        gpu_reduce_min_max(d_xB_r, total_r, &min_xB, &max_xB);
        gpu_reduce_min_max_meff(d_phi_r, d_xB_r, total_r,
                                P.D_alpha, P.D_compound, P.Vm_alpha_0, P.dVm_alpha_dxB, P.Vm_compound,
                                temperature_K, P.mu_reference_scale, &min_Meff, &max_Meff);
    }
    const char *dim_label = (P.Ny == 2) ? "[2D]" : "[3D]";
    time_t now_wall = time(NULL);
    char now_str[64] = {0};
    struct tm *lt_now = localtime(&now_wall);
    if (lt_now) {
        strftime(now_str, sizeof(now_str), "%Y-%m-%d %H:%M:%S", lt_now);
    } else {
        snprintf(now_str, sizeof(now_str), "unknown");
    }
    if (P.mode == 1 && P.minimize_full_model == 0) {
        // 旧 phi-only minimize 只打印 phi 相关
        printf("%s step=%05d t=%.6f vf_precip=%.6e R_avg=%.6e t_wall=%s\n",
               dim_label, 0, 0.0, vf_precip, R_avg, now_str);
    } else {
        printf("%s step=%05d t=%.6f t_real=%.6e s "
               "<x_B_tot>=%.6e vf_precip=%.6e R_avg=%.6e xB_range=[%.4f, %.4f] "
               "Meff_range=[%.4e, %.4e] t_wall=%s\n",
               dim_label, 0, 0.0, 0.0,
               mean_xB_tot, vf_precip, R_avg,
               min_xB, max_xB, min_Meff, max_Meff, now_str);
    }
    
    // 弹性 bulk 惩罚诊断（只在独立诊断参数+弹性启用时，且需在机械平衡求解完成后）
    // 为了满足"t=0 已求解机械平衡"这一条件：这里先暂存 step=0 基础统计，
    // 并在 step=1 开头（求解弹性后、更新phi之前）计算并写入 CSV 的 step=0 行。
    const int need_elastic_bulk_diag = (P.diag_elastic_bulk_penalty_enabled && P.elastic_enabled);
    const int need_delay_step0_csv = (need_elastic_bulk_diag && csv_fp != NULL);
    const double step0_vf_precip = vf_precip;
    const double step0_R_avg = R_avg;

    ElasticBulkPenaltyDiag el_bulk_diag;
    memset(&el_bulk_diag, 0, sizeof(el_bulk_diag));
    el_bulk_diag.valid = 0;
    int el_bulk_diag_ran = 0;

    if (csv_fp && !need_delay_step0_csv) {
        if (P.diag_elastic_bulk_penalty_enabled) {
            fprintf(csv_fp, "%d,%.8f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
                    0, 0.0, 0.0, step0_vf_precip, step0_R_avg,
                    NAN, NAN, NAN, NAN, NAN);
        } else {
            fprintf(csv_fp, "%d,%.8f,%.8e,%.8e,%.8e\n",
                    0, 0.0, 0.0, step0_vf_precip, step0_R_avg);
        }
        fflush(csv_fp);
    }
    
    // 性能计时
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    
    printf("\n开始时间推进...\n");
    printf("========================================\n");
    
    // 时间推进主循环
    const int nsteps_run = (P.mode == 1) ? P.minimize_max_iter : P.nsteps;
    // Minimization: 使用固定 dt_phi = minimize_dt，不再因能量上升二分 dt
    double dt_phi = (P.mode == 1) ? P.minimize_dt : P.dt;
    double V0_target = P.minimize_V0; // in <h> units
    double prev_F_total = NAN;
    // minimize 收敛判据：连续满足 A/B 条件的步数
    int convergence_count = 0;
    // 上一时间步的总能量，用于能量相对变化率判据
    double F_total_prev_step = NAN;
    // lambda_vol: 跨迭代的状态量，用于抗过冲方案
    double lambda_vol = 0.0;
    int steps_completed = 0;

    for (int step = 1; step <= nsteps_run; step++) {
        steps_completed = step;
        // 保存上一时间步
        CUDA_CHECK(cudaMemcpy(d_phi_n_saved, d_phi_r, size_r, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_Y_n_saved, d_Y_r, size_r, cudaMemcpyDeviceToDevice));

        // ============================================================
        // 步骤0：弹性计算（如果启用，迭代弛豫）
        // ============================================================
        if (P.elastic_enabled) {
            double invN_float = (float)invN;
            // stress-free transformation strain eps^00_ij（Voigt顺序）
            float eps_xx00_f = (float)P.eps_xx00;
            float eps_yy00_f = (float)P.eps_yy00;
            float eps_zz00_f = (float)P.eps_zz00;
            float eps_yz00_f = (float)P.eps_yz00;
            float eps_xz00_f = (float)P.eps_xz00;
            float eps_xy00_f = (float)P.eps_xy00;

            float S_11_f = (float)P.S_11, S_12_f = (float)P.S_12, S_13_f = (float)P.S_13;
            float S_14_f = (float)P.S_14, S_15_f = (float)P.S_15, S_16_f = (float)P.S_16;
            float S_22_f = (float)P.S_22, S_23_f = (float)P.S_23, S_24_f = (float)P.S_24;
            float S_25_f = (float)P.S_25, S_26_f = (float)P.S_26;
            float S_33_f = (float)P.S_33, S_34_f = (float)P.S_34, S_35_f = (float)P.S_35, S_36_f = (float)P.S_36;
            float S_44_f = (float)P.S_44, S_45_f = (float)P.S_45, S_46_f = (float)P.S_46;
            float S_55_f = (float)P.S_55, S_56_f = (float)P.S_56;
            float S_66_f = (float)P.S_66;
            
            float S_p_11_f = (float)P.S_p_11, S_p_12_f = (float)P.S_p_12, S_p_13_f = (float)P.S_p_13;
            float S_p_14_f = (float)P.S_p_14, S_p_15_f = (float)P.S_p_15, S_p_16_f = (float)P.S_p_16;
            float S_p_22_f = (float)P.S_p_22, S_p_23_f = (float)P.S_p_23, S_p_24_f = (float)P.S_p_24;
            float S_p_25_f = (float)P.S_p_25, S_p_26_f = (float)P.S_p_26;
            float S_p_33_f = (float)P.S_p_33, S_p_34_f = (float)P.S_p_34, S_p_35_f = (float)P.S_p_35, S_p_36_f = (float)P.S_p_36;
            float S_p_44_f = (float)P.S_p_44, S_p_45_f = (float)P.S_p_45, S_p_46_f = (float)P.S_p_46;
            float S_p_55_f = (float)P.S_p_55, S_p_56_f = (float)P.S_p_56;
            float S_p_66_f = (float)P.S_p_66;
            
            float E0_xx_f = (float)P.E0_xx, E0_yy_f = (float)P.E0_yy, E0_zz_f = (float)P.E0_zz;
            float E0_yz_f = (float)P.E0_yz, E0_xz_f = (float)P.E0_xz, E0_xy_f = (float)P.E0_xy;
            
            // ============================================================
            // 在do循环外：计算eigenstrain和homogeneous displacement field
            // 因为phi在这个时间步是固定的，所以这些只需要计算一次
            // ============================================================
            
            // === 步骤1：eigenstrain（实空间，临时写入 d_uxx_r..d_uyz_r）===
            // Optimization(4): 不再分配持久 eps0_r 数组；临时写入 d_uxx_r..d_uyz_r（此时它们还不是真实应变）
            // minimize(phi-only)：仅用 phi，eigenstrain = h(phi)*eps^00；其余情况：phi + xB
            if (P.mode == 1 && P.minimize_full_model == 0) {
                launch_compute_eigenstrain_from_phi_only_kernel(
                    d_phi_r,
                    d_uxx_r, d_uyy_r, d_uzz_r,
                    d_uxy_r, d_uxz_r, d_uyz_r,
                    eps_xx00_f, eps_yy00_f, eps_zz00_f,
                    eps_yz00_f, eps_xz00_f, eps_xy00_f,
                    total_r);
            } else {
                launch_compute_eigenstrain_from_phi_kernel(
                    d_phi_r,
                    d_xB_r,
                    d_uxx_r, d_uyy_r, d_uzz_r,
                    d_uxy_r, d_uxz_r, d_uyz_r,
                    eps_xx00_f, eps_yy00_f, eps_zz00_f,
                    eps_yz00_f, eps_xz00_f, eps_xy00_f,
                    (double)P.eps_iso_over_vB,
                    total_r);
            }
            
            // === 步骤2：Eigenstrain变换到k空间（临时存储在 d_uxx_k..d_uyz_k）===
            // Optimization(3): 复用 d_uxx_k..d_uyz_k 作为临时 eigenstrain_k 容器
            float *src_r_array[6] = {d_uxx_r, d_uyy_r, d_uzz_r, d_uxy_r, d_uxz_r, d_uyz_r};
            cufftComplex *dst_k_array[6] = {d_uxx_k, d_uyy_k, d_uzz_k, d_uxy_k, d_uxz_k, d_uyz_k};
            
            for (int comp = 0; comp < 6; comp++) {
                CUFFT_CHECK(cufftExecR2C(plan_r2c_elastic, src_r_array[comp], dst_k_array[comp]));
                launch_dealias_float_kernel(dst_k_array[comp], P.Nx, P.Ny, P.Nz, NzC,
                                           P.dx, P.dy, P.dz, total_k);
            }
            
            // === 步骤3：在k空间计算homogeneous elasticity displacement field ===
            // 使用均匀弹性常数，从eigenstrain计算初始位移场
            // Optimization(3): 此时 d_uxx_k..d_uyz_k 中存放的是临时 eigenstrain_k
            launch_compute_displacement_from_eigenstrain_k_kernel(
                d_uxx_k, d_uyy_k, d_uzz_k,
                d_uxy_k, d_uxz_k, d_uyz_k,
                d_ux_k, d_uy_k, d_uz_k,
                S_11_f, S_12_f, S_13_f, S_14_f, S_15_f, S_16_f,
                S_22_f, S_23_f, S_24_f, S_25_f, S_26_f,
                S_33_f, S_34_f, S_35_f, S_36_f,
                S_44_f, S_45_f, S_46_f,
                S_55_f, S_56_f,
                S_66_f,
                P.Nx, P.Ny, P.Nz, NzC,
                P.dx, P.dy, P.dz, total_k);
            launch_dealias_float_kernel(d_ux_k, P.Nx, P.Ny, P.Nz, NzC, P.dx, P.dy, P.dz, total_k);
            launch_dealias_float_kernel(d_uy_k, P.Nx, P.Ny, P.Nz, NzC, P.dx, P.dy, P.dz, total_k);
            launch_dealias_float_kernel(d_uz_k, P.Nx, P.Ny, P.Nz, NzC, P.dx, P.dy, P.dz, total_k);
            
            // 现在d_ux_k, d_uy_k, d_uz_k包含了homogeneous elasticity displacement field
            // Optimization(3)+(4) Lifetime note:
            // - 临时 eps0_r 已写入 d_uxx_r..d_uyz_r，临时 eps0_k 已写入 d_uxx_k..d_uyz_k
            // - compute_displacement_from_eigenstrain_k_kernel 已消费临时 eps0_k
            // - do-while 循环的第一步 launch_compute_strain_from_displacement_k_kernel 会完全覆盖 d_uxx_k..d_uyz_k 为真实 strain_k
            // - 随后 C2R 会覆盖 d_uxx_r..d_uyz_r 为真实 strain_r
            // - 因此临时 eps0 内容在此之后不再被访问，覆盖是安全的
            
            // ============================================================
            // 进入do循环：迭代细化（对非均匀弹性系数进行修正）
            // ============================================================
            int elastic_iter = 1;
            do {
                // === 循环开始：对当前k空间的k_ux, k_uy, k_uz求导得到应变场 ===
                // 第一次迭代：使用的是homogeneous displacement field
                // 第二次及之后迭代：使用的是上一次迭代通过Green function更新的displacement field
                launch_compute_strain_from_displacement_k_kernel(
                    d_ux_k, d_uy_k, d_uz_k,  // 使用当前循环的k空间位移
                    d_uxx_k, d_uyy_k, d_uzz_k,
                    d_uxy_k, d_uxz_k, d_uyz_k,
                    P.Nx, P.Ny, P.Nz, NzC,
                    P.dx, P.dy, P.dz, total_k);
                {
                    cufftComplex *strain_k[6] = {d_uxx_k, d_uyy_k, d_uzz_k, d_uxy_k, d_uxz_k, d_uyz_k};
                    for (int c = 0; c < 6; c++) {
                        launch_dealias_float_kernel(strain_k[c], P.Nx, P.Ny, P.Nz, NzC, P.dx, P.dy, P.dz, total_k);
                    }
                }
                
                // === 应变场C2R到实空间并归一化 ===
                cufftComplex *strain_src_k[6] = {d_uxx_k, d_uyy_k, d_uzz_k, d_uxy_k, d_uxz_k, d_uyz_k};
                float *strain_dst_r[6] = {d_uxx_r, d_uyy_r, d_uzz_r, d_uxy_r, d_uxz_r, d_uyz_r};
                
                for (int comp = 0; comp < 6; comp++) {
                    CUFFT_CHECK(cufftExecC2R(plan_c2r_elastic, strain_src_k[comp], strain_dst_r[comp]));
                }
                // C2R后归一化（除以N），因为cuFFT C2R不归一化，结果 = N * 标准IFFT
                launch_normalize_strain_kernel(d_uxx_r, d_uyy_r, d_uzz_r, d_uxy_r, d_uxz_r, d_uyz_r, invN_float, total_r);
                
                // === 用应变场和eigenstrain组装hij（修正后的local strain）===
                // Optimization(4): eigenstrain 参数已移除，现场计算
                launch_compute_strain_with_perturbation_kernel(
                    d_uxx_r, d_uyy_r, d_uzz_r, d_uxy_r, d_uxz_r, d_uyz_r,
                    d_phi_r,
                    (P.mode == 1 && P.minimize_full_model == 0) ? NULL : d_xB_r,
                    d_uxx_r, d_uyy_r, d_uzz_r, d_uxy_r, d_uxz_r, d_uyz_r,
                    S_11_f, S_12_f, S_13_f, S_14_f, S_15_f, S_16_f,
                    S_22_f, S_23_f, S_24_f, S_25_f, S_26_f,
                    S_33_f, S_34_f, S_35_f, S_36_f,
                    S_44_f, S_45_f, S_46_f,
                    S_55_f, S_56_f,
                    S_66_f,
                    S_p_11_f, S_p_12_f, S_p_13_f, S_p_14_f, S_p_15_f, S_p_16_f,
                    S_p_22_f, S_p_23_f, S_p_24_f, S_p_25_f, S_p_26_f,
                    S_p_33_f, S_p_34_f, S_p_35_f, S_p_36_f,
                    S_p_44_f, S_p_45_f, S_p_46_f,
                    S_p_55_f, S_p_56_f,
                    S_p_66_f,
                    eps_xx00_f, eps_yy00_f, eps_zz00_f,
                    eps_yz00_f, eps_xz00_f, eps_xy00_f,
                    (double)P.eps_iso_over_vB,
                    E0_xx_f, E0_yy_f, E0_zz_f, E0_yz_f, E0_xz_f, E0_xy_f,
                    total_r);
                
                // === 用Green function和hij组装新的k_ux, k_uy, k_uz ===
                // 将hij R2C到k空间
                float *hij_src_r[6] = {d_uxx_r, d_uyy_r, d_uzz_r, d_uxy_r, d_uxz_r, d_uyz_r};
                cufftComplex *hij_dst_k[6] = {d_uxx_k, d_uyy_k, d_uzz_k, d_uxy_k, d_uxz_k, d_uyz_k};
                
                for (int comp = 0; comp < 6; comp++) {
                    CUFFT_CHECK(cufftExecR2C(plan_r2c_elastic, hij_src_r[comp], hij_dst_k[comp]));
                    launch_dealias_float_kernel(hij_dst_k[comp], P.Nx, P.Ny, P.Nz, NzC, P.dx, P.dy, P.dz, total_k);
                }
                
                // 使用Green函数计算新的k空间位移
                launch_compute_displacement_from_hij_green_kernel(
                    d_uxx_k, d_uyy_k, d_uzz_k, d_uxy_k, d_uxz_k, d_uyz_k,
                    d_ux_k, d_uy_k, d_uz_k,
                    S_11_f, S_12_f, S_13_f, S_14_f, S_15_f, S_16_f,
                    S_22_f, S_23_f, S_24_f, S_25_f, S_26_f,
                    S_33_f, S_34_f, S_35_f, S_36_f,
                    S_44_f, S_45_f, S_46_f,
                    S_55_f, S_56_f,
                    S_66_f,
                    P.Nx, P.Ny, P.Nz, NzC,
                    P.dx, P.dy, P.dz, total_k);
                launch_dealias_float_kernel(d_ux_k, P.Nx, P.Ny, P.Nz, NzC, P.dx, P.dy, P.dz, total_k);
                launch_dealias_float_kernel(d_uy_k, P.Nx, P.Ny, P.Nz, NzC, P.dx, P.dy, P.dz, total_k);
                launch_dealias_float_kernel(d_uz_k, P.Nx, P.Ny, P.Nz, NzC, P.dx, P.dy, P.dz, total_k);
                
                // 现在d_ux_k, d_uy_k, d_uz_k包含了更新后的displacement field
                // 下一次循环开始时会使用这些值进行求导
                
                elastic_iter++;
            } while (elastic_iter < P.elastic_iter_max);
            
            // ============================================================
            // 迭代循环结束后：按照SDV_Poly.c:1237-1330的逻辑
            // 1. C2R位移场并归一化（1237-1260行）
            // 2. 对位移场求导得到应变（1262-1288行）
            // 3. 应变C2R到实空间（1290-1313行）
            // 4. 应用外部应变E0（1316-1330行，但全局添加，不乘以Is(r_ps)）
            // ============================================================
            
            // Optimization: 不再将位移 C2R 到 r-space 并归一化；应变直接从 k-space 位移计算
            // === 对位移场求导得到应变（SDV_Poly.c:1262-1288）===
            // 使用当前k空间的k_ux, k_uy, k_uz（已经在循环中更新）
            launch_compute_strain_from_displacement_k_kernel(
                d_ux_k, d_uy_k, d_uz_k,  // 使用循环结束后的k空间位移
                d_uxx_k, d_uyy_k, d_uzz_k,
                d_uxy_k, d_uxz_k, d_uyz_k,
                P.Nx, P.Ny, P.Nz, NzC,
                P.dx, P.dy, P.dz, total_k);
            {
                cufftComplex *fin_strain_k[6] = {d_uxx_k, d_uyy_k, d_uzz_k, d_uxy_k, d_uxz_k, d_uyz_k};
                for (int c = 0; c < 6; c++) {
                    launch_dealias_float_kernel(fin_strain_k[c], P.Nx, P.Ny, P.Nz, NzC, P.dx, P.dy, P.dz, total_k);
                }
            }
            
            // === 步骤3：应变C2R到实空间并归一化（SDV_Poly.c:1290-1313）===
            cufftComplex *final_strain_src_k[6] = {d_uxx_k, d_uyy_k, d_uzz_k, d_uxy_k, d_uxz_k, d_uyz_k};
            float *final_strain_dst_r[6] = {d_uxx_r, d_uyy_r, d_uzz_r, d_uxy_r, d_uxz_r, d_uyz_r};
            
            for (int comp = 0; comp < 6; comp++) {
                CUFFT_CHECK(cufftExecC2R(plan_c2r_elastic, final_strain_src_k[comp], final_strain_dst_r[comp]));
            }
            launch_normalize_strain_kernel(d_uxx_r, d_uyy_r, d_uzz_r, d_uxy_r, d_uxz_r, d_uyz_r, invN_float, total_r);
            
            // === 步骤4：应用外部应变E0（SDV_Poly.c:1316-1330，但全局添加，不乘以Is(r_ps)）===
            launch_add_external_strain_kernel(
                d_uxx_r, d_uyy_r, d_uzz_r,
                d_uxy_r, d_uxz_r, d_uyz_r,
                E0_xx_f, E0_yy_f, E0_zz_f,
                E0_yz_f, E0_xz_f, E0_xy_f,
                total_r);

            // === 额外：由最终应变场计算应力场 sigma_ij(r)，供变分使用 ===
            // 注意：应力计算使用弹性应变 = 总应变 - eigenstrain
            launch_compute_stress_from_strain_with_effective_stiffness_kernel(
                d_uxx_r, d_uyy_r, d_uzz_r,
                d_uxy_r, d_uxz_r, d_uyz_r,
                d_phi_r,
                (P.mode == 1 && P.minimize_full_model == 0) ? NULL : d_xB_r,
                eps_xx00_f, eps_yy00_f, eps_zz00_f,
                eps_yz00_f, eps_xz00_f, eps_xy00_f,
                (double)P.eps_iso_over_vB,
                d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                S_11_f, S_12_f, S_13_f, S_14_f, S_15_f, S_16_f,
                S_22_f, S_23_f, S_24_f, S_25_f, S_26_f,
                S_33_f, S_34_f, S_35_f, S_36_f,
                S_44_f, S_45_f, S_46_f,
                S_55_f, S_56_f,
                S_66_f,
                S_p_11_f, S_p_12_f, S_p_13_f, S_p_14_f, S_p_15_f, S_p_16_f,
                S_p_22_f, S_p_23_f, S_p_24_f, S_p_25_f, S_p_26_f,
                S_p_33_f, S_p_34_f, S_p_35_f, S_p_36_f,
                S_p_44_f, S_p_45_f, S_p_46_f,
                S_p_55_f, S_p_56_f,
                S_p_66_f,
                total_r);

            // === 诊断：弹性 bulk 惩罚(能量密度)（只需在t=0力学平衡求解后输出一次）===
            // 注意：这里在 step=1 且更新phi之前执行，此时的 phi/sigma 对应 t=0。
            if (need_elastic_bulk_diag && !el_bulk_diag_ran) {
                el_bulk_diag = compute_elastic_bulk_penalty(
                    d_phi_r,
                    d_uxx_r, d_uyy_r, d_uzz_r,
                    d_uxy_r, d_uxz_r, d_uyz_r,
                    (P.mode == 1 && P.minimize_full_model == 0) ? NULL : d_xB_r,
                    d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                    d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                    total_r,
                    &P);

                const char *gel_unit = el_bulk_diag.gel_is_dimless ? "dimensionless (gel_hat)" : "J/m^3";
                if (el_bulk_diag.valid) {
                    double w_phys = 0.0;
                    if (el_bulk_diag.gel_is_dimless && P.lambda_sm_m > 0.0) {
                        w_phys = 12.0 * P.gamma_Jm2 / P.lambda_sm_m;
                    }
                    printf("[diag][elastic_bulk_penalty] gel_unit=%s\n", gel_unit);
                    if (el_bulk_diag.gel_is_dimless) {
                        printf("[diag][elastic_bulk_penalty] w_phys=12*gamma/lambda_sm = %.6e J/m^3 (gamma=%.6e J/m^2, lambda_sm=%.6e m)\n",
                               w_phys, P.gamma_Jm2, P.lambda_sm_m);
                    }
                    printf("[diag][elastic_bulk_penalty] sum_h=%.8e, sum_gel=%.8e, E_el_bulk_hat=%.8e, E_el_bulk_Jm3=%.8e, Delta_mu_el(J/mol)=%.8e\n",
                           el_bulk_diag.sum_h, el_bulk_diag.sum_gel, el_bulk_diag.E_el_bulk_hat,
                           el_bulk_diag.E_el_bulk_Jm3, el_bulk_diag.Delta_mu_el_Jmol);
                } else {
                    printf("[diag][elastic_bulk_penalty] gel_unit=%s (invalid/skip)\n", gel_unit);
                }

                // 若之前延迟写入 step=0 CSV，则此时补写（包含诊断列）
                if (need_delay_step0_csv && csv_fp) {
                    double sum_h_csv = el_bulk_diag.valid ? el_bulk_diag.sum_h : NAN;
                    double sum_gel_csv = el_bulk_diag.valid ? el_bulk_diag.sum_gel : NAN;
                    double E_hat_csv = el_bulk_diag.valid ? el_bulk_diag.E_el_bulk_hat : NAN;
                    double E_Jm3_csv = el_bulk_diag.valid ? el_bulk_diag.E_el_bulk_Jm3 : NAN;
                    double dmu_csv = el_bulk_diag.valid ? el_bulk_diag.Delta_mu_el_Jmol : NAN;
                    fprintf(csv_fp, "%d,%.8f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
                            0, 0.0, 0.0, step0_vf_precip, step0_R_avg,
                            sum_h_csv, sum_gel_csv, E_hat_csv, E_Jm3_csv, dmu_csv);
                    fflush(csv_fp);
                }
                el_bulk_diag_ran = 1;
            }
        }

        // ============================================================
        // minimize mode: 能量与体积诊断（全部基于 excess 自由能）
        // ============================================================
        double mean_h_now = NAN;
        double F_surf_hat = NAN, F_el_hat = NAN;
        double F_chem_excess_hat = NAN;      // ΔF_chem_hat = <g_bulk_hat(r) - g_bulk0_hat>
        double F_total_excess_hat = NAN;     // F_surf + F_chem_excess + (F_el - F_el0_hat)
        double F_chem_CNT_hat = NAN;         // CNT 化学体积驱动力: <h> * (mu0 - muB_farfield_hat)
        double F_total_CNT_hat = NAN;        // CNT 总自由能: F_surf + F_el + F_chem_CNT
        double total_interface_sum = NAN;
        double total_el_core_sum = NAN;
        if (P.mode == 1) {
            // 记录上一时间步的总能量，用于能量相对变化率判据
            F_total_prev_step = prev_F_total;
            mean_h_now = gpu_compute_vf_from_h(d_phi_r, total_r); // <h(phi)>
            if (step == 1 && V0_target <= 0.0) {
                V0_target = mean_h_now; // lock to initial volume fraction
            }

            // 表面能 = 梯度能 + 双井能（合并输出）
            launch_compute_grad_energy_density_kernel(
                d_phi_r, d_energy_tmp,
                P.Nx, P.Ny, P.Nz,
                P.dx, P.dy, P.dz,
                P.kappa_phi,
                total_r);
            double sum_grad = gpu_reduce_sum(d_energy_tmp, total_r);
            launch_compute_dw_energy_density_kernel(d_phi_r, d_energy_tmp, P.W, total_r);
            double sum_dw = gpu_reduce_sum(d_energy_tmp, total_r);
            total_interface_sum = sum_grad + sum_dw;
            F_surf_hat = total_interface_sum / (double)total_r;

            // elastic energy density (hat)
            double sum_el = 0.0;
            if (P.elastic_enabled && d_gel_hat_tmp) {
                launch_compute_gel_density_kernel(
                    d_uxx_r, d_uyy_r, d_uzz_r,
                    d_uxy_r, d_uxz_r, d_uyz_r,
                    d_phi_r,
                    (P.mode == 1 && P.minimize_full_model == 0) ? NULL : d_xB_r,
                    d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                    d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                    (float)P.eps_xx00, (float)P.eps_yy00, (float)P.eps_zz00,
                    (float)P.eps_yz00, (float)P.eps_xz00, (float)P.eps_xy00,
                    (double)P.eps_iso_over_vB,
                    d_gel_hat_tmp,
                    total_r);
                sum_el = gpu_reduce_sum(d_gel_hat_tmp, total_r);
                // 核区弹性能总和：gel_hat * h(phi) 的和
                launch_scale_field_by_h_kernel(d_phi_r, d_gel_hat_tmp, total_r);
                total_el_core_sum = gpu_reduce_sum(d_gel_hat_tmp, total_r);
            }
            F_el_hat = sum_el / (double)total_r;

            // 计算 bulk chemical 自由能 excess（DeltaF_chem_hat）
            if (P.minimize_full_model && isfinite(xB_ref) && isfinite(g_bulk0_hat)) {
                launch_compute_gbulk_excess_hat_kernel(
                    d_phi_r, d_xB_r, d_energy_tmp,
                    temperature_K, P.mu_reference_scale,
                    P.mu0_compound,
                    xB_ref, g_bulk0_hat,
                    total_r);
                double sum_gbulk_excess = gpu_reduce_sum(d_energy_tmp, total_r);
                F_chem_excess_hat = sum_gbulk_excess / (double)total_r;
            }

            // 总 excess 自由能（不重复计数双井项，且相对于基体 baseline）
            if (isfinite(F_chem_excess_hat)) {
                F_total_excess_hat = F_chem_excess_hat + F_surf_hat + (F_el_hat - F_el0_hat);
            }

            // CNT 化学体积驱动力与总自由能（用于后处理/诊断，不参与收敛判据）
            if (P.minimize_full_model && isfinite(muB_farfield_hat) && mean_h_now > 0.0) {
                F_chem_CNT_hat = mean_h_now * (P.mu0_compound - muB_farfield_hat);
                F_total_CNT_hat = F_surf_hat + F_el_hat + F_chem_CNT_hat;
            }

            // 记录上一时间步的总 excess 能量，用于能量相对变化率判据（不再用于 dt 二分）
            prev_F_total = F_total_excess_hat;
        }

        // ============================================================
        // 步骤1：相场φ的更新
        // ============================================================

        // minimize mode: Lagrange volume, no linesearch
        double rms_res = NAN;   // res = g_full + λ*h', Euler-Lagrange 残差
        double rms_dphi = NAN;  // ||φ^{n+1}-φ^n||_rms / dt
        if (P.mode == 1) {
            // 1) phi -> k-space，去混叠（与弹性一致）
            CUFFT_CHECK(cufftExecD2Z(plan_r2c_phi, d_phi_r, d_phi_k));
            launch_dealias_kernel(d_phi_k, P.Nx, P.Ny, P.Nz, NzC,
                                 P.dx, P.dy, P.dz, total_k);

            // 2) 计算 g_explicit
            // Minimization(full-model): 使用 compute_phi_rhs_kernel，开启化学驱动力
            // 旧 phi-only minimize：使用 compute_phi_rhs_minimize_kernel（仅双井+弹性）
            if (P.minimize_full_model) {
                // 在计算 phi RHS 之前，先确保 xB 是由当前 (Y,phi) 更新的
                // Minimization(full-model): enable chem+diffusion with volume Lagrange constraint
                launch_compute_mu_x_kernel(d_Y_r, d_phi_r, d_xB_r, d_mu_x_r,
                                          temperature_K, P.mu_reference_scale,
                                          P.v_A, P.v_B, P.mu0_compound,
                                          P.Vm_compound, P.Vm_alpha_0,
                                          P.dVm_alpha_dxB, P.Y_clip, P.xB_eps,
                                          d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                                          P.eps_iso_over_vB,
                                          total_r,
                                          P.elastic_enabled);

                // g_explicit = chem + W*g'(phi) + elastic（不含梯度项）
                launch_compute_phi_rhs_kernel(
                    d_phi_r, d_xB_r, d_phi_rhs_r,
                    P.Nx, P.Ny, P.Nz,
                    temperature_K, P.mu_reference_scale,
                    P.v_A, P.v_B, P.mu0_compound,
                    P.Vm_compound, P.Vm_alpha_0,
                    P.dVm_alpha_dxB, P.W,
                    d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                    d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                    d_uxx_r, d_uyy_r, d_uzz_r, d_uxy_r, d_uxz_r, d_uyz_r,
                    P.S_p_11, P.S_p_12, P.S_p_13, P.S_p_14, P.S_p_15, P.S_p_16,
                    P.S_p_22, P.S_p_23, P.S_p_24, P.S_p_25, P.S_p_26,
                    P.S_p_33, P.S_p_34, P.S_p_35, P.S_p_36,
                    P.S_p_44, P.S_p_45, P.S_p_46,
                    P.S_p_55, P.S_p_56, P.S_p_66,
                    P.eps_xx00, P.eps_yy00, P.eps_zz00,
                    P.eps_yz00, P.eps_xz00, P.eps_xy00,
                    P.eps_iso_over_vB,
                    (P.elastic_enabled ? P.elastic_shift_dimless : 0.0),
                    /* disable_chem = */ 0,
                    total_r,
                    P.elastic_enabled);
            } else {
                // 旧 phi-only minimize：g_explicit = W*g'(φ) + dgel_dphi（双井 + 弹性，不含梯度项）
                launch_compute_phi_rhs_minimize_kernel(
                    d_phi_r, d_phi_rhs_r, P.W,
                    d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                    d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                    d_uxx_r, d_uyy_r, d_uzz_r, d_uxy_r, d_uxz_r, d_uyz_r,
                    // Optimization(4): d_uxx0_r..d_uyz0_r 参数已移除
                    P.S_p_11, P.S_p_12, P.S_p_13, P.S_p_14, P.S_p_15, P.S_p_16,
                    P.S_p_22, P.S_p_23, P.S_p_24, P.S_p_25, P.S_p_26,
                    P.S_p_33, P.S_p_34, P.S_p_35, P.S_p_36,
                    P.S_p_44, P.S_p_45, P.S_p_46,
                    P.S_p_55, P.S_p_56, P.S_p_66,
                    P.eps_xx00, P.eps_yy00, P.eps_zz00,
                    P.eps_yz00, P.eps_xz00, P.eps_xy00,
                    total_r, P.elastic_enabled);
            }

            // 3) 计算 Lap(phi) 到 d_lap_phi_r，Lap 输出去混叠后 C2R
            launch_compute_laplacian_k_kernel(d_phi_k, KS.d_k2, d_phi_rhs_k, total_k);
            launch_dealias_kernel(d_phi_rhs_k, P.Nx, P.Ny, P.Nz, NzC,
                                 P.dx, P.dy, P.dz, total_k);
            CUFFT_CHECK(cufftExecZ2D(plan_c2r_phi, d_phi_rhs_k, d_lap_phi_r));
            launch_normalize_only_kernel(d_lap_phi_r, invN, total_r);
            // g_full = g_explicit - κ*Lap(phi)
            launch_subtract_scaled_kernel(d_phi_rhs_r, d_lap_phi_r, P.kappa_phi, total_r);
            // 现在 d_phi_rhs_r = g_full

            // 4) Lagrange 乘子：λ = -<h'*g_full>/<(h')²>
            launch_compute_hprime_times_rhs_kernel(d_phi_r, d_phi_rhs_r, d_energy_tmp, total_r);
            double sum_hpg = gpu_reduce_sum(d_energy_tmp, total_r);
            launch_compute_hprime_sq_values_kernel(d_phi_r, d_energy_tmp, total_r);
            double sum_hp2 = gpu_reduce_sum(d_energy_tmp, total_r);
            // 任务A：lambda under-relaxation
            double lambda_raw = (fabs(sum_hp2) > 1e-30) ? (-sum_hpg / sum_hp2) : 0.0;
            // under-relaxation（使用可配置参数）
            double eta = P.eta_lambda_vol;
            // 保护：夹到 [0,1]
            if (eta < 0.0) eta = 0.0;
            if (eta > 1.0) eta = 1.0;
            lambda_vol = (1.0 - eta) * lambda_vol + eta * lambda_raw;

            // 5) 更新 RHS = g_explicit + λ*h'（用 d_lap_phi_r 正确加回 κ*Lap(φ)）
            launch_subtract_scaled_kernel(d_phi_rhs_r, d_lap_phi_r, -P.kappa_phi, total_r);
            // d_phi_rhs_r = g_full + κ*Lap = g_explicit
            launch_add_volume_constraint_kernel(d_phi_r, d_phi_rhs_r, lambda_vol, total_r);
            // d_phi_rhs_r = g_explicit + λ*h'（更新方向）
            CUFFT_CHECK(cufftExecD2Z(plan_r2c_phi_rhs, d_phi_rhs_r, d_phi_rhs_k));
            launch_dealias_kernel(d_phi_rhs_k, P.Nx, P.Ny, P.Nz, NzC,
                                 P.dx, P.dy, P.dz, total_k);
            // Optimization: in-place 更新 phi_k
            launch_phi_semi_implicit_update_kernel(d_phi_k, d_phi_rhs_k, KS.d_k2,
                                                   d_phi_k, P.L_phi, P.kappa_phi,
                                                   dt_phi, total_k);
            launch_dealias_kernel(d_phi_k, P.Nx, P.Ny, P.Nz, NzC,
                                 P.dx, P.dy, P.dz, total_k);
            CUFFT_CHECK(cufftExecZ2D(plan_c2r_phi, d_phi_k, d_phi_r));
            launch_phi_normalize_and_clamp_kernel(d_phi_r, invN, total_r);

            // 6) rms_res 诊断：res = g_full(φ^{n+1}) + λ*h'(φ^{n+1})
            // 关键：使用专用 d_res_r，绝不复用 d_phi_rhs_r，避免破坏 step5 的更新 RHS
            CUDA_CHECK(cudaDeviceSynchronize());  // 确保 step5 的 FFT/更新全部完成后再写诊断
            if (P.minimize_full_model) {
                // Full-model minimize: 使用当前更新后的 (phi, Y) 计算 g_full
                // 1) 先用 (Y_r, phi_r) 计算与 φ^{n+1} 匹配的 xB_tmp（复用 d_xB_prev_r）
                launch_compute_mu_x_kernel(d_Y_r, d_phi_r, d_xB_prev_r, d_mu_x_r,
                                          temperature_K, P.mu_reference_scale,
                                          P.v_A, P.v_B, P.mu0_compound,
                                          P.Vm_compound, P.Vm_alpha_0,
                                          P.dVm_alpha_dxB, P.Y_clip, P.xB_eps,
                                          d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                                          P.eps_iso_over_vB,
                                          total_r,
                                          P.elastic_enabled);

                // 2) 使用 (phi_r, xB_tmp) 计算 g_explicit，再后续减 Lap/加 λh'
                launch_compute_phi_rhs_kernel(
                    d_phi_r, d_xB_prev_r, d_res_r,
                    P.Nx, P.Ny, P.Nz,
                    temperature_K, P.mu_reference_scale,
                    P.v_A, P.v_B, P.mu0_compound,
                    P.Vm_compound, P.Vm_alpha_0,
                    P.dVm_alpha_dxB, P.W,
                    d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                    d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                    d_uxx_r, d_uyy_r, d_uzz_r, d_uxy_r, d_uxz_r, d_uyz_r,
                    P.S_p_11, P.S_p_12, P.S_p_13, P.S_p_14, P.S_p_15, P.S_p_16,
                    P.S_p_22, P.S_p_23, P.S_p_24, P.S_p_25, P.S_p_26,
                    P.S_p_33, P.S_p_34, P.S_p_35, P.S_p_36,
                    P.S_p_44, P.S_p_45, P.S_p_46,
                    P.S_p_55, P.S_p_56, P.S_p_66,
                    P.eps_xx00, P.eps_yy00, P.eps_zz00,
                    P.eps_yz00, P.eps_xz00, P.eps_xy00,
                    P.eps_iso_over_vB,
                    (P.elastic_enabled ? P.elastic_shift_dimless : 0.0),
                    /* disable_chem = */ 0,
                    total_r,
                    P.elastic_enabled);
            } else {
                // phi-only minimize：使用当前 φ^{n+1} 计算 g_explicit
                launch_compute_phi_rhs_minimize_kernel(
                    d_phi_r, d_res_r, P.W,
                    d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                    d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                    d_uxx_r, d_uyy_r, d_uzz_r, d_uxy_r, d_uxz_r, d_uyz_r,
                    P.S_p_11, P.S_p_12, P.S_p_13, P.S_p_14, P.S_p_15, P.S_p_16,
                    P.S_p_22, P.S_p_23, P.S_p_24, P.S_p_25, P.S_p_26,
                    P.S_p_33, P.S_p_34, P.S_p_35, P.S_p_36,
                    P.S_p_44, P.S_p_45, P.S_p_46,
                    P.S_p_55, P.S_p_56, P.S_p_66,
                    P.eps_xx00, P.eps_yy00, P.eps_zz00,
                    P.eps_yz00, P.eps_xz00, P.eps_xy00,
                    total_r, P.elastic_enabled);
            }
            // 基于当前 φ^{n+1} 计算 Lap(phi) 与 g_full
            CUFFT_CHECK(cufftExecD2Z(plan_r2c_phi, d_phi_r, d_phi_k));
            launch_dealias_kernel(d_phi_k, P.Nx, P.Ny, P.Nz, NzC,
                                 P.dx, P.dy, P.dz, total_k);
            launch_compute_laplacian_k_kernel(d_phi_k, KS.d_k2, d_phi_rhs_k, total_k);
            launch_dealias_kernel(d_phi_rhs_k, P.Nx, P.Ny, P.Nz, NzC,
                                 P.dx, P.dy, P.dz, total_k);
            CUFFT_CHECK(cufftExecZ2D(plan_c2r_phi, d_phi_rhs_k, d_lap_phi_r));
            launch_normalize_only_kernel(d_lap_phi_r, invN, total_r);
            // d_res_r 此时为 g_explicit(φ^{n+1})，减去 κ*Lap(φ^{n+1}) 得到 g_full
            launch_subtract_scaled_kernel(d_res_r, d_lap_phi_r, P.kappa_phi, total_r);
            // d_res_r = g_full(φ^{n+1})
            launch_add_volume_constraint_kernel(d_phi_r, d_res_r, lambda_vol, total_r);
            // d_res_r = res = g_full(φ^{n+1}) + λ*h'(φ^{n+1})
            square_values_kernel<<<(total_r + 255) / 256, 256>>>(d_res_r, d_energy_tmp, total_r);
            CUDA_CHECK(cudaDeviceSynchronize());
            rms_res = sqrt(gpu_reduce_sum(d_energy_tmp, total_r) / (double)total_r);

            // // 8) Optional volume projection: <h> -> V0
            // if (d_energy_tmp) {
            //     double mean_h_after = gpu_compute_vf_from_h(d_phi_r, total_r);
            //     double vol_err = V0_target - mean_h_after;
            //     if (fabs(vol_err) > 1e-12) {
            //         launch_compute_hprime_sq_values_kernel(d_phi_r, d_energy_tmp, total_r);
            //         double sum_hp2 = gpu_reduce_sum(d_energy_tmp, total_r);
            //         double mean_hp2 = sum_hp2 / (double)total_r;
            //         double lambda_correct = (mean_hp2 > 1e-30) ? (vol_err / mean_hp2) : 0.0;
            //         launch_apply_volume_projection_kernel(d_phi_r, lambda_correct, total_r);
            //     }
            // }

            // 9) rms_dphi
            if (d_energy_tmp && isfinite(dt_phi) && dt_phi > 0.0) {
                diff_sq_kernel<<<(total_r + 255) / 256, 256>>>(d_phi_r, d_phi_n_saved, d_energy_tmp, total_r);
                CUDA_CHECK(cudaDeviceSynchronize());
                rms_dphi = sqrt(gpu_reduce_sum(d_energy_tmp, total_r) / (double)total_r) / dt_phi;
            }

        } else {
            // dynamics mode (original path)
            CUFFT_CHECK(cufftExecD2Z(plan_r2c_phi, d_phi_r, d_phi_k));
            launch_dealias_kernel(d_phi_k, P.Nx, P.Ny, P.Nz, NzC,
                                 P.dx, P.dy, P.dz, total_k);
            launch_compute_phi_rhs_kernel(d_phi_r, d_xB_r, d_phi_rhs_r,
                                         P.Nx, P.Ny, P.Nz,
                                         temperature_K, P.mu_reference_scale,
                                         P.v_A, P.v_B, P.mu0_compound,
                                         P.Vm_compound, P.Vm_alpha_0,
                                         P.dVm_alpha_dxB, P.W,
                                         d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                                         d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                                         d_uxx_r, d_uyy_r, d_uzz_r, d_uxy_r, d_uxz_r, d_uyz_r,
                                         // Optimization(4): d_uxx0_r..d_uyz0_r 参数已移除
                                         P.S_p_11, P.S_p_12, P.S_p_13, P.S_p_14, P.S_p_15, P.S_p_16,
                                         P.S_p_22, P.S_p_23, P.S_p_24, P.S_p_25, P.S_p_26,
                                         P.S_p_33, P.S_p_34, P.S_p_35, P.S_p_36,
                                         P.S_p_44, P.S_p_45, P.S_p_46,
                                         P.S_p_55, P.S_p_56, P.S_p_66,
                                         P.eps_xx00, P.eps_yy00, P.eps_zz00,
                                         P.eps_yz00, P.eps_xz00, P.eps_xy00,
                                         P.eps_iso_over_vB,
                                         (P.elastic_enabled ? P.elastic_shift_dimless : 0.0),
                                         0,
                                         total_r,
                                         P.elastic_enabled);
            CUFFT_CHECK(cufftExecD2Z(plan_r2c_phi_rhs, d_phi_rhs_r, d_phi_rhs_k));
            launch_dealias_kernel(d_phi_rhs_k, P.Nx, P.Ny, P.Nz, NzC,
                                 P.dx, P.dy, P.dz, total_k);
            // Optimization: in-place 更新 phi_k
            launch_phi_semi_implicit_update_kernel(d_phi_k, d_phi_rhs_k, KS.d_k2,
                                                   d_phi_k, P.L_phi, P.kappa_phi,
                                                   dt_phi, total_k);
            launch_dealias_kernel(d_phi_k, P.Nx, P.Ny, P.Nz, NzC,
                                 P.dx, P.dy, P.dz, total_k);
            CUFFT_CHECK(cufftExecZ2D(plan_c2r_phi, d_phi_k, d_phi_r));
            launch_phi_normalize_and_clamp_kernel(d_phi_r, invN, total_r);
        }
        
        // Optimization: 步骤1完成，d_phi_rhs_k 可复用为 d_mu_x_k
        d_mu_x_k = d_phi_rhs_k;
        
        // ============================================================
        // 步骤2：Y方程的更新
        // ============================================================
        int minimize_should_stop = 0;

        if (P.mode == 0 || (P.mode == 1 && P.minimize_full_model == 1)) {
        
        // 2.1 计算mu_x（full-model minimize 与 dynamics 共用）
        launch_compute_mu_x_kernel(d_Y_r, d_phi_r, d_xB_r, d_mu_x_r,
                                  temperature_K, P.mu_reference_scale,
                                  P.v_A, P.v_B, P.mu0_compound,
                                  P.Vm_compound, P.Vm_alpha_0,
                                  P.dVm_alpha_dxB, P.Y_clip, P.xB_eps,
                                  d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                                  P.eps_iso_over_vB,
                                  total_r,
                                  P.elastic_enabled);
        
        // 2.2 mu_x变换到k空间
        CUFFT_CHECK(cufftExecD2Z(plan_r2c_xB, d_mu_x_r, d_mu_x_k));
        launch_dealias_kernel(d_mu_x_k, P.Nx, P.Ny, P.Nz, NzC,
                             P.dx, P.dy, P.dz, total_k);
        d_Y_rhs_k = d_mu_x_k;

        // Optimization: divJ 按方向串行累加，删除 grad_mu_y/z、Jy/z 常驻数组
        // Optimization: P2 phase 开始，reset scratch arena 供本 phase 复用
        scratch_arena_reset(&arena_k);
        scratch_arena_reset(&arena_r);
        double *scratch_r = (double *)d_scratch_r_double;
        cufftDoubleComplex *scratch_k = (cufftDoubleComplex *)d_scratch_k_double;
        launch_zero_divJ_k_kernel(d_divJ_k, total_k);
        CUDA_CHECK(cudaMemcpy(d_xB_prev_r, d_xB_r, size_r, cudaMemcpyDeviceToDevice));
        for (int alpha = 0; alpha < 3; alpha++) {
            // Optimization: use scratch_k_z for grad_mu_k then J_k sequentially
            launch_compute_gradient_single_component_k_kernel(
                d_mu_x_k, scratch_k, alpha,
                P.Nx, P.Ny, P.Nz, NzC, P.dx, P.dy, P.dz, total_k);
            launch_dealias_kernel(scratch_k, P.Nx, P.Ny, P.Nz, NzC,
                                 P.dx, P.dy, P.dz, total_k);
            CUFFT_CHECK(cufftExecZ2D(plan_c2r_xB, scratch_k, scratch_r));
            launch_normalize_only_kernel(scratch_r, invN, total_r);
            // Optimization: reuse scratch_r_d for grad_mu_r and J_r (same buffer)
            launch_compute_flux_single_component_kernel(
                scratch_r, d_phi_r, d_xB_prev_r, scratch_r,
                P.D_alpha, P.D_compound, P.Vm_alpha_0, P.dVm_alpha_dxB,
                P.Vm_compound, temperature_K, P.mu_reference_scale, total_r);
            CUFFT_CHECK(cufftExecD2Z(plan_r2c_xB, scratch_r, scratch_k));
            launch_dealias_kernel(scratch_k, P.Nx, P.Ny, P.Nz, NzC,
                                 P.dx, P.dy, P.dz, total_k);
            launch_divJ_accumulate_kernel(scratch_k, d_divJ_k, alpha,
                                         P.Nx, P.Ny, P.Nz, NzC,
                                         P.dx, P.dy, P.dz, total_k);
        }
        launch_dealias_kernel(d_divJ_k, P.Nx, P.Ny, P.Nz, NzC,
                             P.dx, P.dy, P.dz, total_k);
        CUFFT_CHECK(cufftExecZ2D(plan_c2r_xB, d_divJ_k, d_divJ_r));
        launch_normalize_only_kernel(d_divJ_r, invN, total_r);
        
        // 2.8 计算Y的Laplacian
        CUFFT_CHECK(cufftExecD2Z(plan_r2c_Y, d_Y_r, d_Y_k));
        launch_dealias_kernel(d_Y_k, P.Nx, P.Ny, P.Nz, NzC,
                             P.dx, P.dy, P.dz, total_k);
        launch_compute_laplacian_k_kernel(d_Y_k, KS.d_k2, d_Y_k, total_k);
        launch_dealias_kernel(d_Y_k, P.Nx, P.Ny, P.Nz, NzC,
                             P.dx, P.dy, P.dz, total_k);
        CUFFT_CHECK(cufftExecZ2D(plan_c2r_Y, d_Y_k, d_lapY_r));
        launch_normalize_only_kernel(d_lapY_r, invN, total_r);
        
        // 2.9 计算平均DY（使用GPU归约计算全局平均值，优化版本）
        // mean_DY = mean(D_mix * logistic_deriv) = sum(DY_values) / Ntot
        // 优化：使用直接归约，不需要存储中间数组
        double global_sum_DY = gpu_reduce_sum_DY(d_Y_r, d_phi_r,
                                                  P.D_alpha, P.D_compound, total_r);
        double mean_DY = global_sum_DY / (double)total_r;
        
        // 2.10 计算Y的RHS
        launch_compute_Y_rhs_kernel(d_divJ_r, d_phi_r, d_phi_n_saved,
                                   d_lapY_r, d_Y_r, d_dY_dt_prev_r,
                                   d_Y_rhs_r, P.dt, P.v_B, mean_DY, total_r);
        
        // 2.11 变换RHS到k空间
        CUFFT_CHECK(cufftExecD2Z(plan_r2c_Y, d_Y_rhs_r, d_Y_rhs_k));
        launch_dealias_kernel(d_Y_rhs_k, P.Nx, P.Ny, P.Nz, NzC,
                             P.dx, P.dy, P.dz, total_k);
        
        // 2.12 变换Y到k空间
        CUFFT_CHECK(cufftExecD2Z(plan_r2c_Y, d_Y_r, d_Y_k));
        launch_dealias_kernel(d_Y_k, P.Nx, P.Ny, P.Nz, NzC,
                             P.dx, P.dy, P.dz, total_k);
        
        // 2.13 半隐式更新Y
        // Optimization: in-place 更新 Y_k，删除 Y_k_new 常驻与 pointer swap
        launch_Y_semi_implicit_update_kernel(d_Y_k, d_Y_rhs_k, KS.d_k2,
                                            d_Y_k, mean_DY, P.dt, total_k);
        launch_dealias_kernel(d_Y_k, P.Nx, P.Ny, P.Nz, NzC,
                             P.dx, P.dy, P.dz, total_k);
        
        // 2.14 反变换并归一化
        CUFFT_CHECK(cufftExecZ2D(plan_c2r_Y, d_Y_k, d_Y_r));
        
        // minimize 模式：把 Y 的上限卡到 xB_max_safe 对应的 logit，避免 Y->xB 后再出现 xB>safe
        double Y_upper_cap = P.Y_clip;
        if (P.mode == 1) {
            double xB_cap = P.minimize_xB_max_safe;
            if (xB_cap < P.xB_eps) xB_cap = P.xB_eps;
            if (xB_cap > 1.0 - P.xB_eps) xB_cap = 1.0 - P.xB_eps;
            double y_cap_from_xB = log(xB_cap / (1.0 - xB_cap));
            if (y_cap_from_xB < Y_upper_cap) Y_upper_cap = y_cap_from_xB;
        }
        launch_Y_normalize_and_clamp_kernel(d_Y_r, d_xB_r, invN,
                                           P.Y_clip, Y_upper_cap, P.xB_eps, total_r);

        // 2.15 更新dY_dt_prev（必须在Y更新后立即更新，供下一步使用）
        launch_update_dY_dt_prev_kernel(d_Y_r, d_Y_n_saved, d_dY_dt_prev_r, P.dt, total_r);
        } // end if dynamics mode (Y update)
        
        // -----------------------------------------------------------------
        // 6.5) 后投影修正（minimize 模式）：用当前 φ 重算 λ 并再更新 φ 一次，抑制 vol 慢漂
        // 弹性在后投影子步中冻结，不重复求解
        // -----------------------------------------------------------------
        if (P.mode == 1 && P.minimize_post_projection_iters > 0) {
            for (int post_k = 0; post_k < P.minimize_post_projection_iters; post_k++) {
                // a) 用当前 φ (及 Y/xB)，弹性冻结，重新计算 g_full
                CUFFT_CHECK(cufftExecD2Z(plan_r2c_phi, d_phi_r, d_phi_k));
                launch_dealias_kernel(d_phi_k, P.Nx, P.Ny, P.Nz, NzC,
                                     P.dx, P.dy, P.dz, total_k);
                if (P.minimize_full_model) {
                    launch_compute_phi_rhs_kernel(
                        d_phi_r, d_xB_r, d_phi_rhs_r,
                        P.Nx, P.Ny, P.Nz,
                        temperature_K, P.mu_reference_scale,
                        P.v_A, P.v_B, P.mu0_compound,
                        P.Vm_compound, P.Vm_alpha_0,
                        P.dVm_alpha_dxB, P.W,
                        d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                        d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                        d_uxx_r, d_uyy_r, d_uzz_r, d_uxy_r, d_uxz_r, d_uyz_r,
                        P.S_p_11, P.S_p_12, P.S_p_13, P.S_p_14, P.S_p_15, P.S_p_16,
                        P.S_p_22, P.S_p_23, P.S_p_24, P.S_p_25, P.S_p_26,
                        P.S_p_33, P.S_p_34, P.S_p_35, P.S_p_36,
                        P.S_p_44, P.S_p_45, P.S_p_46,
                        P.S_p_55, P.S_p_56, P.S_p_66,
                        P.eps_xx00, P.eps_yy00, P.eps_zz00,
                        P.eps_yz00, P.eps_xz00, P.eps_xy00,
                        P.eps_iso_over_vB,
                        (P.elastic_enabled ? P.elastic_shift_dimless : 0.0),
                        /* disable_chem = */ 0,
                        total_r,
                        P.elastic_enabled);
                } else {
                    launch_compute_phi_rhs_minimize_kernel(
                        d_phi_r, d_phi_rhs_r, P.W,
                        d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                        d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                        d_uxx_r, d_uyy_r, d_uzz_r, d_uxy_r, d_uxz_r, d_uyz_r,
                        P.S_p_11, P.S_p_12, P.S_p_13, P.S_p_14, P.S_p_15, P.S_p_16,
                        P.S_p_22, P.S_p_23, P.S_p_24, P.S_p_25, P.S_p_26,
                        P.S_p_33, P.S_p_34, P.S_p_35, P.S_p_36,
                        P.S_p_44, P.S_p_45, P.S_p_46,
                        P.S_p_55, P.S_p_56, P.S_p_66,
                        P.eps_xx00, P.eps_yy00, P.eps_zz00,
                        P.eps_yz00, P.eps_xz00, P.eps_xy00,
                        total_r, P.elastic_enabled);
                }
                launch_compute_laplacian_k_kernel(d_phi_k, KS.d_k2, d_phi_rhs_k, total_k);
                launch_dealias_kernel(d_phi_rhs_k, P.Nx, P.Ny, P.Nz, NzC,
                                     P.dx, P.dy, P.dz, total_k);
                CUFFT_CHECK(cufftExecZ2D(plan_c2r_phi, d_phi_rhs_k, d_lap_phi_r));
                launch_normalize_only_kernel(d_lap_phi_r, invN, total_r);
                launch_subtract_scaled_kernel(d_phi_rhs_r, d_lap_phi_r, P.kappa_phi, total_r);
                // b) 重新计算 lambda_raw_tmp（同公式）
                launch_compute_hprime_times_rhs_kernel(d_phi_r, d_phi_rhs_r, d_energy_tmp, total_r);
                double sum_hpg_pp = gpu_reduce_sum(d_energy_tmp, total_r);
                launch_compute_hprime_sq_values_kernel(d_phi_r, d_energy_tmp, total_r);
                double sum_hp2_pp = gpu_reduce_sum(d_energy_tmp, total_r);
                double lambda_raw_tmp = (fabs(sum_hp2_pp) > 1e-30) ? (-sum_hpg_pp / sum_hp2_pp) : 0.0;
                lambda_vol = lambda_raw_tmp;  // c) 后投影强制 eta=1，不做混合
                // d) RHS = g_explicit + λ*h'，再做一次 φ 更新（dt 不变）
                launch_subtract_scaled_kernel(d_phi_rhs_r, d_lap_phi_r, -P.kappa_phi, total_r);
                launch_add_volume_constraint_kernel(d_phi_r, d_phi_rhs_r, lambda_vol, total_r);
                CUFFT_CHECK(cufftExecD2Z(plan_r2c_phi_rhs, d_phi_rhs_r, d_phi_rhs_k));
                launch_dealias_kernel(d_phi_rhs_k, P.Nx, P.Ny, P.Nz, NzC,
                                     P.dx, P.dy, P.dz, total_k);
                launch_phi_semi_implicit_update_kernel(d_phi_k, d_phi_rhs_k, KS.d_k2,
                                                       d_phi_k, P.L_phi, P.kappa_phi,
                                                       dt_phi, total_k);
                launch_dealias_kernel(d_phi_k, P.Nx, P.Ny, P.Nz, NzC,
                                     P.dx, P.dy, P.dz, total_k);
                CUFFT_CHECK(cufftExecZ2D(plan_c2r_phi, d_phi_k, d_phi_r));
                launch_phi_normalize_and_clamp_kernel(d_phi_r, invN, total_r);
                // e) 后投影子步只更新 φ，不额外更新 Y
            }
            // 后投影后 φ 已变，重算 mean_h 供收敛/CSV 使用
            mean_h_now = gpu_compute_vf_from_h(d_phi_r, total_r);
        }
        // minimize 模式：用当前 φ（可能经后投影）重算 mean_h，保证 vol_err_rel/CSV 为最终体积
        if (P.mode == 1 && P.minimize_post_projection_iters == 0) {
            mean_h_now = gpu_compute_vf_from_h(d_phi_r, total_r);
        }
        
        // -----------------------------------------------------------------
        // 7) 收敛判据（minimize 模式）：解耦步长控制与停止条件
        //    注意：此处的 rms_dY 使用“本次迭代 Y 更新后的场”计算，
        //    因此必须放在 Y 半隐式更新 (2.13~2.15) 之后。
        // -----------------------------------------------------------------
        if (P.mode == 1) {
            // 计算析出相尺寸（用于CSV输出）：在 GPU 上按 h(phi)=0.5 的等值面做边界诊断，只回传少量标量
            double Len_x = 0.0, Len_y = 0.0, Len_z = 0.0;
            compute_nucleus_dimensions_gpu(d_phi_r, P.Nx, P.Ny, P.Nz,
                                           P.dx, P.dy, P.dz,
                                           d_bbox_mins, d_bbox_maxs,
                                           d_boundary_sum, d_boundary_count,
                                           &Len_x, &Len_y, &Len_z);
            
            // 计算相对能量变化率（基于 excess 自由能，用于平台判据）
            double energy_diff_rel = NAN;
            if (isfinite(F_total_prev_step) && isfinite(F_total_excess_hat)) {
                double denom = fmax(fabs(F_total_prev_step), 1.0e-20);
                energy_diff_rel = fabs(F_total_excess_hat - F_total_prev_step) / denom;
            }
            // 体积约束相对误差：vol_err_rel = |mean_h - V0_target| / max(V0_target, tiny)
            // 当 P.minimize_V0 <= 0（用户未指定体积目标）时，此判据禁用
            const double vol_tiny = 1e-30;
            double vol_err_rel = NAN;
            if (V0_target > vol_tiny) {
                vol_err_rel = fabs(mean_h_now - V0_target) / (V0_target + vol_tiny);
            } else {
                vol_err_rel = 0.0;  // 无有效目标时视为满足
            }

            // full-model 需要额外的 Y 收敛判据：
            // rms_dY = ||Y^{n+1} - Y^{n}||_rms / dt_Y，其中 dt_Y = P.dt 是 Y 方程使用的时间步。
            double rms_dY = NAN;
            if (P.minimize_full_model && d_energy_tmp) {
                double dt_Y = P.dt;
                if (dt_Y > 0.0) {
                    diff_sq_kernel<<<(total_r + 255) / 256, 256>>>(d_Y_r, d_Y_n_saved, d_energy_tmp, total_r);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    double sum_sq = gpu_reduce_sum(d_energy_tmp, total_r);
                    rms_dY = sqrt(sum_sq / (double)total_r) / dt_Y;
                } else {
                    // 理论上 minimize 模式下 dt_Y 应为正，这里仅作防护
                    rms_dY = NAN;
                }
            }
            // 若 P.minimize_full_model==0（phi-only minimize），则本轮没有对 Y 做更新，
            // 此时保持 rms_dY=NaN，并在 CSV 中写出 NaN，仅作为占位诊断列。

            // 记录 energy_minimize.csv（基于 excess 自由能 + CNT 汇总）
            if (energy_fp) {
                double Len_x_Len_z = (Len_z > 1e-12) ? (Len_x / Len_z) : NAN;
                fprintf(energy_fp,
                        "%d,%.8e,%.8e,%.8e,%.8e,"
                        "%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,"
                        "%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%d,"
                        "%.8e,%.8e,%.8e,%.8e\n",
                        step,
                        dt_phi,
                        (isfinite(mean_h_now) ? mean_h_now : NAN),
                        (isfinite(V0_target) ? V0_target : NAN),
                        (isfinite(lambda_vol) ? lambda_vol : NAN),
                        F_surf_hat, F_el_hat,
                        (isfinite(F_chem_excess_hat) ? F_chem_excess_hat : NAN),
                        (isfinite(F_total_excess_hat) ? F_total_excess_hat : NAN),
                        (isfinite(F_chem_CNT_hat) ? F_chem_CNT_hat : NAN),
                        (isfinite(F_total_CNT_hat) ? F_total_CNT_hat : NAN),
                        (isfinite(total_interface_sum) ? total_interface_sum : NAN),
                        (isfinite(total_el_core_sum) ? total_el_core_sum : NAN),
                        (isfinite(rms_res)  ? rms_res  : NAN),
                        (isfinite(rms_dphi) ? rms_dphi : NAN),
                        (isfinite(rms_dY)   ? rms_dY   : NAN),
                        (isfinite(energy_diff_rel) ? energy_diff_rel : NAN),
                        (isfinite(vol_err_rel) ? vol_err_rel : NAN),
                        P.minimize_post_projection_iters,
                        (isfinite(Len_x) ? Len_x : NAN),
                        (isfinite(Len_y) ? Len_y : NAN),
                        (isfinite(Len_z) ? Len_z : NAN),
                        (isfinite(Len_x_Len_z) ? Len_x_Len_z : NAN));
                fflush(energy_fp);
            }

            // --- 收敛规则 ---
            // 体积约束：vol_err_rel 以 V0_target 为参考
            // V0<=0 时 V0_target 已在 step1 设为 mean_h，即按第一步体积计算误差并判停
            int vol_ok = (isfinite(vol_err_rel) && vol_err_rel < P.minimize_vol_err_rel_threshold);
            // rms_res 硬条件：Euler-Lagrange/KKT 残差
            int rms_res_ok = isfinite(rms_res) && rms_res < P.minimize_rms_res_threshold;

            int this_step_satisfies = 0;
            int condA = 0, condB = 0;

            if (!P.minimize_full_model) {
                // (1) phi-only minimize：φ 收敛 & 能量平台，且均需满足 rms_res、vol_err_rel
                condA = (isfinite(rms_dphi) &&
                         rms_dphi < P.minimize_rms_dphi_threshold &&
                         rms_res_ok && vol_ok);
                condB = (isfinite(energy_diff_rel) &&
                         energy_diff_rel < P.minimize_energy_diff_rel_threshold &&
                         isfinite(rms_dphi) && rms_dphi < 5.0 * P.minimize_rms_dphi_threshold &&
                         rms_res_ok && vol_ok);
                if (condA || condB) this_step_satisfies = 1;
            } else {
                // (2) full-model minimize：双场收敛 + 能量平台，且均需满足 rms_res、vol_err_rel
                condA = (isfinite(rms_dphi) && isfinite(rms_dY) &&
                         rms_dphi < P.minimize_rms_dphi_threshold &&
                         rms_dY   < P.minimize_rms_dY_threshold &&
                         rms_res_ok && vol_ok);
                condB = (isfinite(energy_diff_rel) &&
                         energy_diff_rel < P.minimize_energy_diff_rel_threshold &&
                         isfinite(rms_dphi) && isfinite(rms_dY) &&
                         rms_dphi < 5.0 * P.minimize_rms_dphi_threshold &&
                         rms_dY   < 5.0 * P.minimize_rms_dY_threshold &&
                         rms_res_ok && vol_ok);
                if (condA || condB) this_step_satisfies = 1;
            }

            if (this_step_satisfies) {
                convergence_count++;
            } else {
                convergence_count = 0;
            }

            if (convergence_count >= P.minimize_convergence_steps) {
                minimize_should_stop = 1;
                const char *trigger = condA ? "condA" : "condB";
                log_section_header("Minimize Converged");
                log_kv_text("iter", "%d", step);
                log_kv_text("trigger", "%s", trigger);
                log_kv_text("F_total_excess_hat", "%.6e", (isfinite(F_total_excess_hat) ? F_total_excess_hat : -1.0));
                log_kv_text("rms_res", "%.3e", rms_res);
                log_kv_text("rms_dphi", "%.3e", rms_dphi);
                if (P.minimize_full_model) {
                    log_kv_text("rms_dY", "%.3e", (isfinite(rms_dY) ? rms_dY : -1.0));
                }
                log_kv_text("vol_err_rel", "%.3e", (isfinite(vol_err_rel) ? vol_err_rel : -1.0));
                log_kv_text("energy_diff_rel", "%.3e", (isfinite(energy_diff_rel) ? energy_diff_rel : -1.0));
            }
        }
        
        // 优化：不再计算和存储xBtot，仅在需要输出时临时计算
        
        // 性能计时（精确计时每个时间步）
        if (step == 1) {
            CUDA_CHECK(cudaDeviceSynchronize());  // 确保初始化完成
            CUDA_CHECK(cudaEventRecord(start_event));
        }
        
        if (step == P.nsteps) {
            CUDA_CHECK(cudaDeviceSynchronize());  // 确保最后一步完成
            CUDA_CHECK(cudaEventRecord(stop_event));
            CUDA_CHECK(cudaEventSynchronize(stop_event));
            
            float elapsed_ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
            const int timed_step_begin = 1;
            const int timed_step_end = P.nsteps;
            const int timed_steps = (timed_step_end >= timed_step_begin)
                ? (timed_step_end - timed_step_begin + 1)
                : 0;
            float avg_time_per_step = (timed_steps > 0)
                ? (elapsed_ms / (float)timed_steps)
                : 0.0f;
            double throughput_mpts = (avg_time_per_step > 0.0f)
                ? ((double)total_r / (double)avg_time_per_step * 1e-3)
                : 0.0;
            
            // 获取显存使用情况
            size_t free_mem, total_mem;
            CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
            size_t used_mem = total_mem - free_mem;
            
            printf("\n========================================\n");
            printf("性能统计（精确计时）:\n");
            printf("  总计算时间: %.6f ms (steps %d-%d)\n", elapsed_ms, timed_step_begin, timed_step_end);
            printf("  平均每步: %.6f ms\n", avg_time_per_step);
            printf("  网格大小: %dx%dx%d = %d points\n", P.Nx, P.Ny, P.Nz, total_r);
            printf("  吞吐量: %.2f M points/s\n", throughput_mpts);
            printf("\n显存使用情况:\n");
            printf("  已使用: %.2f GB (%.2f%%)\n", 
                   used_mem / (1024.0*1024.0*1024.0),
                   (double)used_mem / (double)total_mem * 100.0);
            printf("  空闲: %.2f GB\n", free_mem / (1024.0*1024.0*1024.0));
            printf("  总计: %.2f GB\n", total_mem / (1024.0*1024.0*1024.0));
            printf("========================================\n");
        }
        
        // ===== 计算诊断统计量并输出CSV（如果满足CSV输出间隔） =====
        if (step % P.csv_out_every == 0) {
            double current_time = step * P.dt;
            
            // 暂时关闭调试输出
            /*
            if (P.elastic_enabled && d_sigma_xx_r != NULL) {
                // ... (调试代码)
            }
            */
            
            // 优化：使用直接归约计算N_in和N_if，不需要中间数组
            double global_N_in, global_N_if;
            gpu_reduce_sum_N_in_N_if(d_phi_r, total_r, &global_N_in, &global_N_if);
            
            // 调试：检查phi的范围（仅在N_if为0时）
            if (global_N_if < 1.0 && (step % P.csv_out_every == 0)) {
                // 优化：使用GPU归约计算phi范围
                double phi_min, phi_max;
                gpu_reduce_min_max(d_phi_r, total_r, &phi_min, &phi_max);
                printf("[diagnostics] step=%d N_in=%.2f N_if=%.2f phi_range=[%.6f, %.6f]\n",
                       step, global_N_in, global_N_if, phi_min, phi_max);
            }
            
            // 3. 计算体积分数
            double vf_precip = (P.mode == 1 && isfinite(mean_h_now)) ? mean_h_now : gpu_compute_vf_from_h(d_phi_r, total_r); // <h(phi)>
            
            // 4. 计算平均半径 R_avg
            double R_avg = 0.0;
            if (global_N_if > 0.0) {
                // N_if 除以 lambda = ic_phi_iface_w * 2.0
                double lambda = P.ic_phi_iface_w * 2.0;
                double global_N_if_normalized = global_N_if / lambda;
                double ratio = global_N_in / global_N_if_normalized;
                if (P.Ny == 2) {
                    // 2D模式
                    R_avg = 2.0 * P.dx * ratio;
                } else {
                    // 3D模式
                    R_avg = 3.0 * P.dx * ratio;
                }
            }
            
            double t_real = current_time * P.t_real_unit;
            const char *dim_label = (P.Ny == 2) ? "[2D]" : "[3D]";
            time_t now_wall = time(NULL);
            char now_str[64] = {0};
            struct tm *lt_now = localtime(&now_wall);
            if (lt_now) {
                strftime(now_str, sizeof(now_str), "%Y-%m-%d %H:%M:%S", lt_now);
            } else {
                snprintf(now_str, sizeof(now_str), "unknown");
            }
            if (P.mode == 0 || (P.mode == 1 && P.minimize_full_model == 1)) {
                double min_xB = 0.0, max_xB = 0.0;
                gpu_reduce_min_max(d_xB_r, total_r, &min_xB, &max_xB);
                if (P.mode == 1 && P.minimize_full_model == 1 &&
                    max_xB > P.minimize_xB_max_safe + 1e-12) {
                    printf("[minimize step=%d] safety clamp after Y->xB monitor: "
                           "max_before=%.6e xB_max_safe=%.6e\n",
                           step, max_xB, P.minimize_xB_max_safe);
                    launch_clamp_xB_max_kernel(d_xB_r, total_r, P.minimize_xB_max_safe);
                    gpu_reduce_min_max(d_xB_r, total_r, &min_xB, &max_xB);
                }
                double mean_xB_tot = gpu_reduce_sum_xBtot(d_phi_r, d_xB_r, P.v_B, total_r) / (double)total_r;
                double min_Meff, max_Meff;
                gpu_reduce_min_max_meff(d_phi_r, d_xB_r, total_r,
                                        P.D_alpha, P.D_compound, P.Vm_alpha_0, P.dVm_alpha_dxB, P.Vm_compound,
                                        temperature_K, P.mu_reference_scale, &min_Meff, &max_Meff);
                printf("%s step=%05d t=%.6f t_real=%.6e s "
                       "<x_B_tot>=%.6e vf_precip=%.6e R_avg=%.6e xB_range=[%.4f, %.4f] "
                       "Meff_range=[%.4e, %.4e] t_wall=%s\n",
                       dim_label, step, current_time, t_real,
                       mean_xB_tot, vf_precip, R_avg,
                       min_xB, max_xB, min_Meff, max_Meff, now_str);
            }
            // minimize(phi-only) 模式不打印 xB 相关诊断
            
            if (step % P.out_every == 0) {
                // 只在VTK输出时打印
                // printf("%s step=%05d t=%.6f t_real=%.6e s ...", ...); // 已经移出
            }
            
            // 写入CSV文件（每个CSV输出步都写入）
            if (csv_fp) {
                if (P.diag_elastic_bulk_penalty_enabled) {
                    double sum_h_csv = (el_bulk_diag_ran && el_bulk_diag.valid) ? el_bulk_diag.sum_h : NAN;
                    double sum_gel_csv = (el_bulk_diag_ran && el_bulk_diag.valid) ? el_bulk_diag.sum_gel : NAN;
                    double E_hat_csv = (el_bulk_diag_ran && el_bulk_diag.valid) ? el_bulk_diag.E_el_bulk_hat : NAN;
                    double E_Jm3_csv = (el_bulk_diag_ran && el_bulk_diag.valid) ? el_bulk_diag.E_el_bulk_Jm3 : NAN;
                    double dmu_csv = (el_bulk_diag_ran && el_bulk_diag.valid) ? el_bulk_diag.Delta_mu_el_Jmol : NAN;
                    fprintf(csv_fp, "%d,%.8f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
                            step, current_time, t_real, vf_precip, R_avg,
                            sum_h_csv, sum_gel_csv, E_hat_csv, E_Jm3_csv, dmu_csv);
                } else {
                    fprintf(csv_fp, "%d,%.8f,%.8e,%.8e,%.8e\n",
                            step, current_time, t_real, vf_precip, R_avg);
                }
                fflush(csv_fp);
            }
        }
        
        // ===== 输出VTK文件（如果满足VTK输出间隔；minimize 模式不输出任何VTK） =====
        if (P.mode == 0 && (step % P.out_every == 0)) {
            // 统一使用真实步数作为文件编号，避免同一步重复输出多个 phi_* 文件
            int output_step = step;
            
            int ret1 = 1, ret3 = 1;
            if (P.mode == 0) {
                // dynamics: 输出 xBtot, phi, xB
                // Optimization: 使用 scratch 池，避免循环内 malloc/free
                double *d_xBtot_temp = (double *)d_scratch_r_double;
                launch_compute_xBtot_kernel(d_phi_r, d_xB_r, d_xBtot_temp, P.v_B, total_r);
                build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "xBtot", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                ret1 = write_vtk_cuda(d_xBtot_temp, P.Nx, P.Ny, P.Nz, "xB_tot", output_step, filename);
                if (!ret1) fprintf(stderr, "ERROR: Failed to write xBtot VTK file: %s\n", filename);
                build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "xB", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                ret3 = write_vtk_cuda(d_xB_r, P.Nx, P.Ny, P.Nz, "xB", output_step, filename);
                if (!ret3) fprintf(stderr, "ERROR: Failed to write xB VTK file: %s\n", filename);
            }
            // dynamics 输出 phi
            build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "phi", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
            int ret2 = write_vtk_cuda(d_phi_r, P.Nx, P.Ny, P.Nz, "phi", output_step, filename);
            if (!ret2) {
                fprintf(stderr, "ERROR: Failed to write phi VTK file: %s\n", filename);
            }
            if (P.diag_vtk_enabled && P.mode == 0) {
                // dynamics only: delta_mu, f_phi_chem, xB-dependent VTK
                // Optimization: 2-slot scratch 串行复用；PHASE P4 各场写出后 slot 可安全覆盖
                // 安全证明：每场计算→write_vtk→立即复用 slot；后续不再读，见 PHASE_LIFECYCLE.md
                double *scratch0 = (double *)d_scratch_r_double;
                double *scratch1 = (double *)((char *)d_scratch_r_double + size_r);
                launch_compute_delta_mu_r_kernel(d_phi_r, d_xB_r, scratch1,
                                                temperature_K, P.mu_reference_scale,
                                                P.v_A, P.v_B, P.mu0_compound,
                                                (P.elastic_enabled ? P.elastic_shift_dimless : 0.0),
                                                total_r);
                build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "delta_mu_r", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                int ret4 = write_vtk_cuda(scratch1, P.Nx, P.Ny, P.Nz, "delta_mu_r", output_step, filename);
                if (!ret4) fprintf(stderr, "ERROR: Failed to write delta_mu_r VTK file: %s\n", filename);
                // OutA: f_phi_chem, f_phi_dw → slot0, slot1
                launch_compute_f_phi_chem_f_phi_dw_kernel(d_phi_r, d_xB_r, scratch0, scratch1,
                    temperature_K, P.mu_reference_scale, P.v_A, P.v_B, P.mu0_compound,
                    P.Vm_compound, P.Vm_alpha_0, P.dVm_alpha_dxB, P.W,
                    (P.elastic_enabled ? P.elastic_shift_dimless : 0.0), total_r);
                build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "f_phi_chem", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                int ret5 = write_vtk_cuda(scratch0, P.Nx, P.Ny, P.Nz, "f_phi_chem", output_step, filename);
                if (!ret5) fprintf(stderr, "ERROR: Failed to write f_phi_chem VTK file: %s\n", filename);
                build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "f_phi_dw", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                int ret7 = write_vtk_cuda(scratch1, P.Nx, P.Ny, P.Nz, "f_phi_dw", output_step, filename);
                if (!ret7) fprintf(stderr, "ERROR: Failed to write f_phi_dw VTK file: %s\n", filename);
                // OutB: f_phi_bulk, dgel_dphi → slot0, slot1（覆盖，写出后不再读）
                launch_compute_f_phi_bulk_dgel_dphi_kernel(d_phi_r, d_xB_r, scratch0, scratch1,
                    temperature_K, P.mu_reference_scale, P.v_A, P.v_B, P.mu0_compound,
                    P.Vm_compound, P.Vm_alpha_0, P.dVm_alpha_dxB,
                    d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r, d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                    d_uxx_r, d_uyy_r, d_uzz_r, d_uxy_r, d_uxz_r, d_uyz_r,
                    // Optimization(4): d_uxx0_r..d_uyz0_r 参数已移除
                    P.S_p_11, P.S_p_12, P.S_p_13, P.S_p_14, P.S_p_15, P.S_p_16,
                    P.S_p_22, P.S_p_23, P.S_p_24, P.S_p_25, P.S_p_26,
                    P.S_p_33, P.S_p_34, P.S_p_35, P.S_p_36,
                    P.S_p_44, P.S_p_45, P.S_p_46, P.S_p_55, P.S_p_56, P.S_p_66,
                    P.eps_xx00, P.eps_yy00, P.eps_zz00, P.eps_yz00, P.eps_xz00, P.eps_xy00,
                    P.eps_iso_over_vB, (P.elastic_enabled ? P.elastic_shift_dimless : 0.0),
                    total_r, P.elastic_enabled);
                build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "f_phi_bulk", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                int ret8 = write_vtk_cuda(scratch0, P.Nx, P.Ny, P.Nz, "f_phi_bulk", output_step, filename);
                if (!ret8) fprintf(stderr, "ERROR: Failed to write f_phi_bulk VTK file: %s\n", filename);
                build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "dgel_dphi", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                int ret6 = write_vtk_cuda(scratch1, P.Nx, P.Ny, P.Nz, "dgel_dphi", output_step, filename);
                if (!ret6) fprintf(stderr, "ERROR: Failed to write dgel_dphi VTK file: %s\n", filename);
                // f_phi_grad → slot0
                CUFFT_CHECK(cufftExecD2Z(plan_r2c_phi, d_phi_r, d_phi_rhs_k));
                launch_dealias_kernel(d_phi_rhs_k, P.Nx, P.Ny, P.Nz, NzC,
                                     P.dx, P.dy, P.dz, total_k);
                launch_compute_laplacian_k_kernel(d_phi_rhs_k, KS.d_k2, d_phi_rhs_k, total_k);
                launch_dealias_kernel(d_phi_rhs_k, P.Nx, P.Ny, P.Nz, NzC,
                                     P.dx, P.dy, P.dz, total_k);
                CUFFT_CHECK(cufftExecZ2D(plan_c2r_phi, d_phi_rhs_k, scratch0));
                launch_normalize_only_kernel(scratch0, invN, total_r);
                launch_scale_array_kernel(scratch0, -P.kappa_phi, total_r);
                build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "f_phi_grad", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                int ret9 = write_vtk_cuda(scratch0, P.Nx, P.Ny, P.Nz, "f_phi_grad", output_step, filename);
                if (!ret9) fprintf(stderr, "ERROR: Failed to write f_phi_grad VTK file: %s\n", filename);
                // driving_force 直接计算到 slot1（读 slot0=f_phi_grad）
                launch_compute_driving_force_direct_kernel(d_phi_r, d_xB_r, scratch0, scratch1,
                    temperature_K, P.mu_reference_scale, P.v_A, P.v_B, P.mu0_compound,
                    P.Vm_compound, P.Vm_alpha_0, P.dVm_alpha_dxB, P.W,
                    (P.elastic_enabled ? P.elastic_shift_dimless : 0.0),
                    d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                    d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                    d_uxx_r, d_uyy_r, d_uzz_r, d_uxy_r, d_uxz_r, d_uyz_r,
                    // Optimization(4): d_uxx0_r..d_uyz0_r 参数已移除
                    P.S_p_11, P.S_p_12, P.S_p_13, P.S_p_14, P.S_p_15, P.S_p_16,
                    P.S_p_22, P.S_p_23, P.S_p_24, P.S_p_25, P.S_p_26,
                    P.S_p_33, P.S_p_34, P.S_p_35, P.S_p_36,
                    P.S_p_44, P.S_p_45, P.S_p_46, P.S_p_55, P.S_p_56, P.S_p_66,
                    P.eps_xx00, P.eps_yy00, P.eps_zz00, P.eps_yz00, P.eps_xz00, P.eps_xy00,
                    P.eps_iso_over_vB, total_r, P.elastic_enabled);
                build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "driving_force", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                int ret10 = write_vtk_cuda(scratch1, P.Nx, P.Ny, P.Nz, "driving_force", output_step, filename);
                if (!ret10) fprintf(stderr, "ERROR: Failed to write driving_force VTK file: %s\n", filename);

                // === 弹性诊断模式：输出弹性能密度 gel 的VTK分布 ===
                if (P.diag_elastic_bulk_penalty_enabled && P.elastic_enabled) {
                    // Optimization: gel 复用 scratch slot 0
                    double *d_gel_temp = (double *)d_scratch_r_double;
                    launch_compute_gel_density_kernel(
                        d_uxx_r, d_uyy_r, d_uzz_r,
                        d_uxy_r, d_uxz_r, d_uyz_r,
                        d_phi_r,
                        (P.mode == 1) ? NULL : d_xB_r,  // Optimization(4): minimize 模式 xB_r 为 NULL
                        d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                        d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                        (float)P.eps_xx00, (float)P.eps_yy00, (float)P.eps_zz00,
                        (float)P.eps_yz00, (float)P.eps_xz00, (float)P.eps_xy00,
                        (double)P.eps_iso_over_vB,
                        d_gel_temp,
                        total_r);
                    build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "gel", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                    int ret11 = write_vtk_cuda(d_gel_temp, P.Nx, P.Ny, P.Nz, "gel", output_step, filename);
                    if (!ret11) fprintf(stderr, "ERROR: Failed to write gel VTK file: %s\n", filename);
                    
                    if (ret1 && ret2 && ret3 && ret4 && ret5 && ret6 && ret7 && ret8 && ret9 && ret10 && ret11) {
                        printf("step=%05d 完成，已输出VTK文件（phi_%d.vtk, xB_%d.vtk, xBtot_%d.vtk, delta_mu_r_%d.vtk, f_phi_chem_%d.vtk, dgel_dphi_%d.vtk, f_phi_dw_%d.vtk, f_phi_bulk_%d.vtk, f_phi_grad_%d.vtk, driving_force_%d.vtk, gel_%d.vtk）\n", 
                               step, output_step, output_step, output_step, output_step, output_step, output_step,
                               output_step, output_step, output_step, output_step, output_step);
                    } else {
                        fprintf(stderr, "step=%05d 警告：诊断VTK文件输出失败！\n", step);
                    }
                } else {
                    // 诊断模式但未启用弹性bulk诊断，或弹性未启用
                    if (ret1 && ret2 && ret3 && ret4 && ret5 && ret6 && ret7 && ret8 && ret9 && ret10) {
                        printf("step=%05d 完成，已输出VTK文件（phi_%d.vtk, xB_%d.vtk, xBtot_%d.vtk, delta_mu_r_%d.vtk, f_phi_chem_%d.vtk, dgel_dphi_%d.vtk, f_phi_dw_%d.vtk, f_phi_bulk_%d.vtk, f_phi_grad_%d.vtk, driving_force_%d.vtk）\n", 
                               step, output_step, output_step, output_step, output_step, output_step, output_step,
                               output_step, output_step, output_step, output_step);
                    } else {
                        fprintf(stderr, "step=%05d 警告：诊断VTK文件输出失败！\n", step);
                    }
                }
            } else {
                if (P.mode == 1) {
                    if (ret2) printf("step=%05d 完成，已输出VTK文件（phi_%d.vtk）\n", step, output_step);
                    else fprintf(stderr, "step=%05d 警告：VTK文件输出失败！\n", step);
                } else {
                    if (ret1 && ret2 && ret3) {
                        printf("step=%05d 完成，已输出VTK文件（phi_%d.vtk, xB_%d.vtk, xBtot_%d.vtk）\n",
                               step, output_step, output_step, output_step);
                    } else {
                        fprintf(stderr, "step=%05d 警告：VTK文件输出失败！\n", step);
                    }
                }
            }
        }

        if (P.mode == 1 && minimize_should_stop) {
            // shrink "effective nsteps" so the final-output logic matches actual stop iter
            P.nsteps = step;
            break;
        }
    }
    
    // minimize 模式：在收敛/退出后额外输出一次 final VTK（最小改动：只输出 phi）
    if (P.mode == 1) {
        build_case_vtk_path(filename, sizeof(filename),
                            output_dir, case_output_dir,
                            "phi", VTK_NAME_FINAL, 0, vtk_case_tag, 1);
        int ret_phi_final = write_vtk_cuda(d_phi_r, P.Nx, P.Ny, P.Nz, "phi", P.nsteps, filename);
        if (!ret_phi_final) {
            fprintf(stderr, "ERROR: Failed to write phi VTK file (final): %s\n", filename);
        } else {
            printf("写出 final vtk: %s\n", filename);
        }
    }

    // 循环结束后，检查最后一步是否需要输出VTK（仅 dynamics 模式保留 final output）
    if (P.mode == 0 && (P.nsteps % P.out_every != 0)) {
        int final_step = P.nsteps;
        int output_step = (final_step / P.out_every) + 1;
        int ret1 = 1, ret3 = 1;
        if (P.mode == 0) {
            // Optimization: 使用 scratch 池
            double *d_xBtot_temp = (double *)d_scratch_r_double;
            launch_compute_xBtot_kernel(d_phi_r, d_xB_r, d_xBtot_temp, P.v_B, total_r);
            build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "xBtot", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
            ret1 = write_vtk_cuda(d_xBtot_temp, P.Nx, P.Ny, P.Nz, "xB_tot", output_step, filename);
            if (!ret1) fprintf(stderr, "ERROR: Failed to write xBtot VTK file (final step): %s\n", filename);
            build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "xB", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
            ret3 = write_vtk_cuda(d_xB_r, P.Nx, P.Ny, P.Nz, "xB", output_step, filename);
            if (!ret3) fprintf(stderr, "ERROR: Failed to write xB VTK file (final step): %s\n", filename);
        }
        build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "phi", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
        int ret2 = write_vtk_cuda(d_phi_r, P.Nx, P.Ny, P.Nz, "phi", output_step, filename);
        if (!ret2) fprintf(stderr, "ERROR: Failed to write phi VTK file (final step): %s\n", filename);
        if (P.diag_vtk_enabled && P.mode == 0) {
            // Optimization: 2-slot scratch 串行复用（与主循环一致）
            double *scratch0 = (double *)d_scratch_r_double;
            double *scratch1 = (double *)((char *)d_scratch_r_double + size_r);
            launch_compute_delta_mu_r_kernel(d_phi_r, d_xB_r, scratch1,
                                            temperature_K, P.mu_reference_scale,
                                            P.v_A, P.v_B, P.mu0_compound,
                                            (P.elastic_enabled ? P.elastic_shift_dimless : 0.0),
                                            total_r);
            build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "delta_mu_r", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
            int ret4 = write_vtk_cuda(scratch1, P.Nx, P.Ny, P.Nz, "delta_mu_r", output_step, filename);
            if (!ret4) fprintf(stderr, "ERROR: Failed to write delta_mu_r VTK file (final step): %s\n", filename);
            launch_compute_f_phi_chem_f_phi_dw_kernel(d_phi_r, d_xB_r, scratch0, scratch1,
                temperature_K, P.mu_reference_scale, P.v_A, P.v_B, P.mu0_compound,
                P.Vm_compound, P.Vm_alpha_0, P.dVm_alpha_dxB, P.W,
                (P.elastic_enabled ? P.elastic_shift_dimless : 0.0), total_r);
            build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "f_phi_chem", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
            int ret5 = write_vtk_cuda(scratch0, P.Nx, P.Ny, P.Nz, "f_phi_chem", output_step, filename);
            if (!ret5) fprintf(stderr, "ERROR: Failed to write f_phi_chem VTK file (final step): %s\n", filename);
            build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "f_phi_dw", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
            int ret7 = write_vtk_cuda(scratch1, P.Nx, P.Ny, P.Nz, "f_phi_dw", output_step, filename);
            if (!ret7) fprintf(stderr, "ERROR: Failed to write f_phi_dw VTK file (final step): %s\n", filename);
            launch_compute_f_phi_bulk_dgel_dphi_kernel(d_phi_r, d_xB_r, scratch0, scratch1,
                temperature_K, P.mu_reference_scale, P.v_A, P.v_B, P.mu0_compound,
                P.Vm_compound, P.Vm_alpha_0, P.dVm_alpha_dxB,
                d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r, d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                d_uxx_r, d_uyy_r, d_uzz_r, d_uxy_r, d_uxz_r, d_uyz_r,
                // Optimization(4): d_uxx0_r..d_uyz0_r 参数已移除
                P.S_p_11, P.S_p_12, P.S_p_13, P.S_p_14, P.S_p_15, P.S_p_16,
                P.S_p_22, P.S_p_23, P.S_p_24, P.S_p_25, P.S_p_26,
                P.S_p_33, P.S_p_34, P.S_p_35, P.S_p_36,
                P.S_p_44, P.S_p_45, P.S_p_46, P.S_p_55, P.S_p_56, P.S_p_66,
                P.eps_xx00, P.eps_yy00, P.eps_zz00, P.eps_yz00, P.eps_xz00, P.eps_xy00,
                P.eps_iso_over_vB, (P.elastic_enabled ? P.elastic_shift_dimless : 0.0),
                total_r, P.elastic_enabled);
            build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "f_phi_bulk", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
            int ret8 = write_vtk_cuda(scratch0, P.Nx, P.Ny, P.Nz, "f_phi_bulk", output_step, filename);
            if (!ret8) fprintf(stderr, "ERROR: Failed to write f_phi_bulk VTK file (final step): %s\n", filename);
            build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "dgel_dphi", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
            int ret6 = write_vtk_cuda(scratch1, P.Nx, P.Ny, P.Nz, "dgel_dphi", output_step, filename);
            if (!ret6) fprintf(stderr, "ERROR: Failed to write dgel_dphi VTK file (final step): %s\n", filename);
            CUFFT_CHECK(cufftExecD2Z(plan_r2c_phi, d_phi_r, d_phi_rhs_k));
            launch_dealias_kernel(d_phi_rhs_k, P.Nx, P.Ny, P.Nz, NzC,
                                 P.dx, P.dy, P.dz, total_k);
            launch_compute_laplacian_k_kernel(d_phi_rhs_k, KS.d_k2, d_phi_rhs_k, total_k);
            launch_dealias_kernel(d_phi_rhs_k, P.Nx, P.Ny, P.Nz, NzC,
                                 P.dx, P.dy, P.dz, total_k);
            CUFFT_CHECK(cufftExecZ2D(plan_c2r_phi, d_phi_rhs_k, scratch0));
            launch_normalize_only_kernel(scratch0, invN, total_r);
            launch_scale_array_kernel(scratch0, -P.kappa_phi, total_r);
            build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "f_phi_grad", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
            int ret9 = write_vtk_cuda(scratch0, P.Nx, P.Ny, P.Nz, "f_phi_grad", output_step, filename);
            if (!ret9) fprintf(stderr, "ERROR: Failed to write f_phi_grad VTK file (final step): %s\n", filename);
            launch_compute_driving_force_direct_kernel(d_phi_r, d_xB_r, scratch0, scratch1,
                temperature_K, P.mu_reference_scale, P.v_A, P.v_B, P.mu0_compound,
                P.Vm_compound, P.Vm_alpha_0, P.dVm_alpha_dxB, P.W,
                (P.elastic_enabled ? P.elastic_shift_dimless : 0.0),
                d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                d_uxx_r, d_uyy_r, d_uzz_r, d_uxy_r, d_uxz_r, d_uyz_r,
                // Optimization(4): d_uxx0_r..d_uyz0_r 参数已移除
                P.S_p_11, P.S_p_12, P.S_p_13, P.S_p_14, P.S_p_15, P.S_p_16,
                P.S_p_22, P.S_p_23, P.S_p_24, P.S_p_25, P.S_p_26,
                P.S_p_33, P.S_p_34, P.S_p_35, P.S_p_36,
                P.S_p_44, P.S_p_45, P.S_p_46, P.S_p_55, P.S_p_56, P.S_p_66,
                P.eps_xx00, P.eps_yy00, P.eps_zz00, P.eps_yz00, P.eps_xz00, P.eps_xy00,
                P.eps_iso_over_vB, total_r, P.elastic_enabled);
            build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "driving_force", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
            int ret10 = write_vtk_cuda(scratch1, P.Nx, P.Ny, P.Nz, "driving_force", output_step, filename);
            if (!ret10) fprintf(stderr, "ERROR: Failed to write driving_force VTK file (final step): %s\n", filename);
            
            if (ret1 && ret2 && ret3 && ret4 && ret5 && ret6 && ret7 && ret8 && ret9 && ret10) {
                printf("step=%05d（最终步）完成，已输出VTK文件（phi_%d.vtk, xB_%d.vtk, xBtot_%d.vtk, delta_mu_r_%d.vtk, ...）\n",
                       final_step, output_step, output_step, output_step, output_step);
            } else {
                fprintf(stderr, "step=%05d（最终步）警告：诊断VTK文件输出失败！\n", final_step);
            }
        } else {
            if (P.mode == 1) {
                if (ret2) printf("step=%05d（最终步）完成，已输出VTK文件（phi_%d.vtk）\n", final_step, output_step);
                else fprintf(stderr, "step=%05d（最终步）警告：VTK文件输出失败！\n", final_step);
            } else {
                if (ret1 && ret2 && ret3) {
                    printf("step=%05d（最终步）完成，已输出VTK文件（phi_%d.vtk, xB_%d.vtk, xBtot_%d.vtk）\n",
                           final_step, output_step, output_step, output_step);
                } else {
                    fprintf(stderr, "step=%05d（最终步）警告：VTK文件输出失败！\n", final_step);
                }
            }
        }
        
        // 也输出CSV（如果最后一步没有按csv_out_every输出）
        if (final_step % P.csv_out_every != 0 && csv_fp) {
            double current_time = final_step * P.dt;
            
            // 优化：使用直接归约计算统计量
            double global_N_in, global_N_if;
            gpu_reduce_sum_N_in_N_if(d_phi_r, total_r, &global_N_in, &global_N_if);
            
            double vf_precip = gpu_compute_vf_from_h(d_phi_r, total_r);
            double R_avg = 0.0;
            if (global_N_if > 0.0) {
                // N_if 除以 lambda = ic_phi_iface_w * 2.0
                double lambda = P.ic_phi_iface_w * 2.0;
                double global_N_if_normalized = global_N_if / lambda;
                double ratio = global_N_in / global_N_if_normalized;
                if (P.Ny == 2) {
                    R_avg = 2.0 * P.dx * ratio;
                } else {
                    R_avg = 3.0 * P.dx * ratio;
                }
            }
            double t_real = current_time * P.t_real_unit;
            if (P.diag_elastic_bulk_penalty_enabled && P.mode == 0) {
                double sum_h_csv = (el_bulk_diag_ran && el_bulk_diag.valid) ? el_bulk_diag.sum_h : NAN;
                double sum_gel_csv = (el_bulk_diag_ran && el_bulk_diag.valid) ? el_bulk_diag.sum_gel : NAN;
                double E_hat_csv = (el_bulk_diag_ran && el_bulk_diag.valid) ? el_bulk_diag.E_el_bulk_hat : NAN;
                double E_Jm3_csv = (el_bulk_diag_ran && el_bulk_diag.valid) ? el_bulk_diag.E_el_bulk_Jm3 : NAN;
                double dmu_csv = (el_bulk_diag_ran && el_bulk_diag.valid) ? el_bulk_diag.Delta_mu_el_Jmol : NAN;
                fprintf(csv_fp, "%d,%.8f,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
                        final_step, current_time, t_real, vf_precip, R_avg,
                        sum_h_csv, sum_gel_csv, E_hat_csv, E_Jm3_csv, dmu_csv);
            } else {
                fprintf(csv_fp, "%d,%.8f,%.8e,%.8e,%.8e\n",
                        final_step, current_time, t_real, vf_precip, R_avg);
            }
            fflush(csv_fp);
        }
    }
    
    // 清理（只销毁实际创建的计划）
    CUFFT_CHECK(cufftDestroy(plan_r2c_base));
    CUFFT_CHECK(cufftDestroy(plan_c2r_base));
    
    if (P.elastic_enabled) {
        CUFFT_CHECK(cufftDestroy(plan_r2c_elastic));
        CUFFT_CHECK(cufftDestroy(plan_c2r_elastic));
        
        // 释放弹性数组
        // Optimization(4): d_uxx0_r..d_uyz0_r 已移除，不再需要释放
        CUDA_CHECK(cudaFree(d_uxx_r));
        CUDA_CHECK(cudaFree(d_uyy_r));
        CUDA_CHECK(cudaFree(d_uzz_r));
        CUDA_CHECK(cudaFree(d_uxy_r));
        CUDA_CHECK(cudaFree(d_uxz_r));
        CUDA_CHECK(cudaFree(d_uyz_r));
        // 应力场
        CUDA_CHECK(cudaFree(d_sigma_xx_r));
        CUDA_CHECK(cudaFree(d_sigma_yy_r));
        CUDA_CHECK(cudaFree(d_sigma_zz_r));
        CUDA_CHECK(cudaFree(d_sigma_xy_r));
        CUDA_CHECK(cudaFree(d_sigma_xz_r));
        CUDA_CHECK(cudaFree(d_sigma_yz_r));
        // 注意：不再释放perturbation数组，因为不再分配
        // Optimization(3): d_uxx0_k..d_uyz0_k 已移除，不再需要释放
        CUDA_CHECK(cudaFree(d_ux_k));
        CUDA_CHECK(cudaFree(d_uy_k));
        CUDA_CHECK(cudaFree(d_uz_k));
        CUDA_CHECK(cudaFree(d_uxx_k));
        CUDA_CHECK(cudaFree(d_uyy_k));
        CUDA_CHECK(cudaFree(d_uzz_k));
        CUDA_CHECK(cudaFree(d_uxy_k));
        CUDA_CHECK(cudaFree(d_uxz_k));
        CUDA_CHECK(cudaFree(d_uyz_k));
    }
    
    kspace_free_cuda(&KS);

    // minimize work buffers 复用 d_Y_r 等，不需要单独释放
    
    CUDA_CHECK(cudaFree(d_phi_r));
    CUDA_CHECK(cudaFree(d_Y_r));
    CUDA_CHECK(cudaFree(d_xB_r));
    CUDA_CHECK(cudaFree(d_phi_rhs_r));
    // 优化：d_phi_rhs_r被d_lapY_r复用，只需要释放一次
    CUDA_CHECK(cudaFree(d_phi_n_saved));
    CUDA_CHECK(cudaFree(d_Y_n_saved));
    CUDA_CHECK(cudaFree(d_dY_dt_prev_r));
    // 优化：不再需要释放d_xBtot_r
    CUDA_CHECK(cudaFree(d_mu_x_r));
    // Optimization: d_Y_rhs_r 复用 d_mu_x_r，不需要单独释放
    // Optimization: 移除 grad_mu/J 常驻，d_xB_prev_r 与 d_divJ_r 共享内存
    CUDA_CHECK(cudaFree(d_divJ_r));
    // 优化：不再需要释放d_DY_values
    CUDA_CHECK(cudaFree(d_phi_k));
    CUDA_CHECK(cudaFree(d_phi_rhs_k));
    CUDA_CHECK(cudaFree(d_Y_k));
    CUDA_CHECK(cudaFree(d_divJ_k));
    // Optimization: 释放 scratch 池
    CUDA_CHECK(cudaFree(d_scratch_k_double));
    CUDA_CHECK(cudaFree(d_scratch_r_double));
    CUDA_CHECK(cudaFree(d_bbox_mins));
    CUDA_CHECK(cudaFree(d_bbox_maxs));
    CUDA_CHECK(cudaFree(d_boundary_sum));
    CUDA_CHECK(cudaFree(d_boundary_count));
    // 优化：不再需要释放d_Y_rhs_k（复用d_phi_rhs_k）
    // 优化：不再需要释放d_diag_stats
    
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    
    free(h_phi_r);
    free(h_Y_r);
    free(h_xB_r);
    free(h_xBtot_r);
    
    if (csv_fp) {
        fclose(csv_fp);
    }
    if (energy_fp) {
        fclose(energy_fp);
    }
    
    const double wall_t1 = wall_time_sec_monotonic();
    const double wall_elapsed = wall_t1 - wall_t0;
    int hh = 0, mm = 0, ss = 0;
    format_hms(wall_elapsed, &hh, &mm, &ss);
    log_section_header("Run Summary");
    log_kv_text("mode", "%s", (P.mode == 0 ? "dynamic" : "minimize"));
    log_kv_text("steps_completed", "%d", steps_completed);
    log_kv_text("wall_time_s", "%.3f", wall_elapsed);
    log_kv_text("wall_time_hms", "%02d:%02d:%02d", hh, mm, ss);
    return 0;
}
