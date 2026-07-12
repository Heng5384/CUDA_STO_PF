/**
 * 第二阶段：CUDA Kernel实现
 * 
 * 实现的核心kernel：
 * 1. 去混叠kernel（2/3规则）
 * 2. 相场方程RHS计算kernel
 * 3. 相场方程半隐式更新kernel
 * 4. phi归一化和截断kernel
 * 5. Y方程相关kernel
 */

#include "cuda_common.h"
#include "cuda_kernels.h"
#include "pf_params.h"
#include "phase_functions.h"
#include "thermo_utils.h"
#include <cufft.h>
#include <cuComplex.h>

// 与 main_cuda.cu / CPU 版本保持一致的辅助函数：限制 xB 在 [eps, 1-eps] 区间
__host__ __device__ static inline double clamp_eps(double v, double eps) {
    if (v < eps) return eps;
    if (v > 1.0 - eps) return 1.0 - eps;
    return v;
}

// The transport update and its diagnostic must share one positive, bounded
// mobility.  Without this guard, a clipped xB cell can make Gamma approach
// zero or change sign in the flux kernel while the reported Meff is capped.
__device__ static inline double stabilized_meff(double Dm, double gamma) {
    const double gamma_floor = 1.0e-4;
    const double D_safe = fmax(Dm, 0.0);
    const double gamma_safe = (isfinite(gamma) && gamma > gamma_floor)
        ? gamma : gamma_floor;
    const double cap = D_safe * 1000.0;
    const double value = D_safe / gamma_safe;
    return (isfinite(value) && value < cap) ? value : cap;
}

// 前置声明：gpu_reduce_sum 在后部实现，这里先声明以供前面函数使用
double gpu_reduce_sum(const double *d_array, int n);

__device__ static inline double atomicMaxAbsDouble(double *address, double value_abs) {
    unsigned long long int *address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    while (__longlong_as_double(old) < value_abs) {
        unsigned long long int assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(value_abs));
        if (old == assumed) break;
    }
    return __longlong_as_double(old);
}

__device__ static inline double atomicMinDouble(double *address, double value) {
    unsigned long long int *address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    while (__longlong_as_double(old) > value) {
        unsigned long long int assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(value));
        if (old == assumed) break;
    }
    return __longlong_as_double(old);
}

__device__ static inline double atomicMaxDouble(double *address, double value) {
    unsigned long long int *address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    while (__longlong_as_double(old) < value) {
        unsigned long long int assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(value));
        if (old == assumed) break;
    }
    return __longlong_as_double(old);
}

__device__ __host__ static inline double sigmoid_from_logit_unclipped_local(double Y) {
    if (Y >= 0.0) {
        double e_negY = exp(-Y);
        return 1.0 / (1.0 + e_negY);
    }
    double e_Y = exp(Y);
    return e_Y / (1.0 + e_Y);
}

__device__ static inline double h_inverse_bisection_device(double target) {
    if (target <= 0.0) return 0.0;
    if (target >= 1.0) return 1.0;
    double lo = 0.0;
    double hi = 1.0;
    for (int iter = 0; iter < 60; ++iter) {
        double mid = 0.5 * (lo + hi);
        double h_mid = h_of_phi(mid);
        if (h_mid < target) lo = mid;
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// ============================================================================
// 1. 去混叠kernel（2/3规则）
// ============================================================================

__global__ void dealias_23_kernel(
    cuDoubleComplex *f_k,
    int Nx, int Ny, int Nz, int NzC,
    double dx, double dy, double dz,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    // 计算三维索引
    int k = idx % NzC;
    int remainder = idx / NzC;
    int j = remainder % Ny;
    int i = remainder / Ny;
    
    const double k_max_factor = 2.0 / 3.0;
    const double kx_max = M_PI * k_max_factor / dx;
    const double ky_max = M_PI * k_max_factor / dy;
    const double kz_max = M_PI * k_max_factor / dz;
    
    // 计算k向量
    double kx = kx_wrap(i, Nx, dx);
    double ky = ky_wrap(j, Ny, dy);
    double kz = kz_wrap(k, Nz, dz);
    
    // 如果超出2/3截断范围，置零
    if (fabs(kx) >= kx_max || fabs(ky) >= ky_max || fabs(kz) >= kz_max) {
        f_k[idx] = make_cuDoubleComplex(0.0, 0.0);
    }
}

// 弹性用 float complex（cufftComplex）的去混叠
__global__ void dealias_23_float_kernel(
    cufftComplex *f_k,
    int Nx, int Ny, int Nz, int NzC,
    double dx, double dy, double dz,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    int k = idx % NzC;
    int remainder = idx / NzC;
    int j = remainder % Ny;
    int i = remainder / Ny;

    const double k_max_factor = 2.0 / 3.0;
    const double kx_max = M_PI * k_max_factor / dx;
    const double ky_max = M_PI * k_max_factor / dy;
    const double kz_max = M_PI * k_max_factor / dz;

    double kx = kx_wrap(i, Nx, dx);
    double ky = ky_wrap(j, Ny, dy);
    double kz = kz_wrap(k, Nz, dz);

    if (fabs(kx) >= kx_max || fabs(ky) >= ky_max || fabs(kz) >= kz_max) {
        f_k[idx] = make_cuFloatComplex(0.0f, 0.0f);
    }
}

// ============================================================================
// 辅助：计算 Q = e^T * deltaC_voigt * e（工程应变 e = [exx, eyy, ezz, 2*eyz, 2*exz, 2*exy]）
// deltaC 即 S_p_ij（21 分量），与应力 kernel 的工程应变约定一致
// ============================================================================
__device__ static inline double compute_Q_voigt(
    double exx, double eyy, double ezz, double eyz, double exz, double exy,
    double S_p_11, double S_p_12, double S_p_13, double S_p_14, double S_p_15, double S_p_16,
    double S_p_22, double S_p_23, double S_p_24, double S_p_25, double S_p_26,
    double S_p_33, double S_p_34, double S_p_35, double S_p_36,
    double S_p_44, double S_p_45, double S_p_46,
    double S_p_55, double S_p_56,
    double S_p_66)
{
    double e4 = 2.0 * eyz, e5 = 2.0 * exz, e6 = 2.0 * exy;
    double Q = S_p_11*exx*exx + S_p_22*eyy*eyy + S_p_33*ezz*ezz
             + S_p_44*e4*e4 + S_p_55*e5*e5 + S_p_66*e6*e6
             + 2.0*(S_p_12*exx*eyy + S_p_13*exx*ezz + S_p_14*exx*e4 + S_p_15*exx*e5 + S_p_16*exx*e6)
             + 2.0*(S_p_23*eyy*ezz + S_p_24*eyy*e4 + S_p_25*eyy*e5 + S_p_26*eyy*e6)
             + 2.0*(S_p_34*ezz*e4 + S_p_35*ezz*e5 + S_p_36*ezz*e6)
             + 2.0*(S_p_45*e4*e5 + S_p_46*e4*e6)
             + 2.0*S_p_56*e5*e6;
    return Q;
}

// ============================================================================
// Optimization(4): 现场计算 eigenstrain 的 helper 函数
// ============================================================================

// 现场计算 eigenstrain(phi,xB)，float 精度输出
// 与 compute_eigenstrain_from_phi_kernel 使用完全相同的公式和精度
__device__ __forceinline__
void eigenstrain_phi_xB_point(
    double phi, double xB,
    float eps_xx00, float eps_yy00, float eps_zz00,
    float eps_yz00, float eps_xz00, float eps_xy00,
    double eps_iso_over_vB,
    float &eps_xx0_f, float &eps_yy0_f, float &eps_zz0_f,
    float &eps_xy0_f, float &eps_xz0_f, float &eps_yz0_f)
{
    double h   = h_of_phi(phi);
    double xB_c = clamp01(xB);
    
    // 化学体积膨胀：ε^c(x_B) = x_B * (ε_iso / v_B)
    double eps_c = xB_c * eps_iso_over_vB;
    
    // stress-free transformation strain：只在析出相（φ→1）启动
    double eps_xx00_d = (double)eps_xx00;
    double eps_yy00_d = (double)eps_yy00;
    double eps_zz00_d = (double)eps_zz00;
    double eps_yz00_d = (double)eps_yz00;
    double eps_xz00_d = (double)eps_xz00;
    double eps_xy00_d = (double)eps_xy00;
    
    // 总 eigenstrain: ε^0_ij = (1-h) ε^c δ_ij + h ε^00_ij
    double one_minus_h = 1.0 - h;
    
    double eps_xx0 = one_minus_h * eps_c + h * eps_xx00_d;
    double eps_yy0 = one_minus_h * eps_c + h * eps_yy00_d;
    double eps_zz0 = one_minus_h * eps_c + h * eps_zz00_d;
    double eps_yz0 =                       h * eps_yz00_d;
    double eps_xz0 =                       h * eps_xz00_d;
    double eps_xy0 =                       h * eps_xy00_d;
    
    // 写回 float（与原先 kernel 一致）
    eps_xx0_f = (float)eps_xx0;
    eps_yy0_f = (float)eps_yy0;
    eps_zz0_f = (float)eps_zz0;
    eps_yz0_f = (float)eps_yz0;
    eps_xz0_f = (float)eps_xz0;
    eps_xy0_f = (float)eps_xy0;
}

__device__ __forceinline__
double stiffness_beta_weight_point(
    double phi, double eta,
    int gp_mode_enabled,
    int gp_elastic_enabled)
{
    if (gp_mode_enabled && gp_elastic_enabled) {
        return stiffness_beta_weight_gp(phi, eta);
    }
    return h_of_phi(phi);
}

__device__ __forceinline__
void eigenstrain_phi_eta_xB_point(
    double phi, double eta, double xB,
    float eps_xx00, float eps_yy00, float eps_zz00,
    float eps_yz00, float eps_xz00, float eps_xy00,
    double eps_iso_over_vB,
    int gp_mode_enabled,
    int gp_elastic_enabled,
    double gp_eps_iso,
    float &eps_xx0_f, float &eps_yy0_f, float &eps_zz0_f,
    float &eps_xy0_f, float &eps_xz0_f, float &eps_yz0_f)
{
    if (!(gp_mode_enabled && gp_elastic_enabled)) {
        eigenstrain_phi_xB_point(
            phi, xB,
            eps_xx00, eps_yy00, eps_zz00,
            eps_yz00, eps_xz00, eps_xy00,
            eps_iso_over_vB,
            eps_xx0_f, eps_yy0_f, eps_zz0_f,
            eps_xy0_f, eps_xz0_f, eps_yz0_f);
        return;
    }

    double h_alpha = 0.0;
    double h_GP = 0.0;
    double h_beta = 0.0;
    phase_fractions_gp(phi, eta, &h_alpha, &h_GP, &h_beta);
    double xB_c = clamp01(xB);
    double eps_alpha = xB_c * eps_iso_over_vB;

    double eps_xx0 = h_alpha * eps_alpha + h_GP * gp_eps_iso + h_beta * (double)eps_xx00;
    double eps_yy0 = h_alpha * eps_alpha + h_GP * gp_eps_iso + h_beta * (double)eps_yy00;
    double eps_zz0 = h_alpha * eps_alpha + h_GP * gp_eps_iso + h_beta * (double)eps_zz00;
    double eps_yz0 = h_beta * (double)eps_yz00;
    double eps_xz0 = h_beta * (double)eps_xz00;
    double eps_xy0 = h_beta * (double)eps_xy00;

    eps_xx0_f = (float)eps_xx0;
    eps_yy0_f = (float)eps_yy0;
    eps_zz0_f = (float)eps_zz0;
    eps_yz0_f = (float)eps_yz0;
    eps_xz0_f = (float)eps_xz0;
    eps_xy0_f = (float)eps_xy0;
}

__device__ __forceinline__
double compute_dgel_deta_gp_point(
    double phi, double eta, double xB,
    double sigma_xx, double sigma_yy, double sigma_zz,
    double sigma_xy, double sigma_xz, double sigma_yz,
    double eps_iso_over_vB,
    double gp_eps_iso)
{
    (void)sigma_xy;
    (void)sigma_xz;
    (void)sigma_yz;
    double xB_c = clamp01(xB);
    double eps_alpha = xB_c * eps_iso_over_vB;
    double prefactor = (1.0 - h_of_phi(phi)) * h_prime_of_eta(eta);
    double delta_diag = prefactor * (gp_eps_iso - eps_alpha);
    return -(sigma_xx + sigma_yy + sigma_zz) * delta_diag;
}

__device__ __forceinline__
double compute_dgel_dphi_gp_point(
    double phi, double eta, double xB,
    double sigma_xx, double sigma_yy, double sigma_zz,
    double sigma_xy, double sigma_xz, double sigma_yz,
    double exx_el, double eyy_el, double ezz_el,
    double eyz_el, double exz_el, double exy_el,
    double v_B,
    double eps_iso_over_vB,
    double gp_eps_iso,
    double S_p_11, double S_p_12, double S_p_13, double S_p_14, double S_p_15, double S_p_16,
    double S_p_22, double S_p_23, double S_p_24, double S_p_25, double S_p_26,
    double S_p_33, double S_p_34, double S_p_35, double S_p_36,
    double S_p_44, double S_p_45, double S_p_46,
    double S_p_55, double S_p_56,
    double S_p_66,
    double eps_xx00, double eps_yy00, double eps_zz00,
    double eps_yz00, double eps_xz00, double eps_xy00)
{
    double hp = h_prime_of_phi(phi);
    double h_eta = h_of_eta(eta);
    double xB_c = clamp01(xB);
    double eps_alpha = xB_c * eps_iso_over_vB;
    double diag_iso_term = -eps_alpha + eps_iso_over_vB * (xB_c - v_B);

    double d_eps_xx0_dphi = hp * (eps_xx00 + (1.0 - h_eta) * diag_iso_term - h_eta * gp_eps_iso);
    double d_eps_yy0_dphi = hp * (eps_yy00 + (1.0 - h_eta) * diag_iso_term - h_eta * gp_eps_iso);
    double d_eps_zz0_dphi = hp * (eps_zz00 + (1.0 - h_eta) * diag_iso_term - h_eta * gp_eps_iso);
    double d_eps_xy0_dphi = hp * eps_xy00;
    double d_eps_xz0_dphi = hp * eps_xz00;
    double d_eps_yz0_dphi = hp * eps_yz00;

    double dgel_eigen =
        -( sigma_xx * d_eps_xx0_dphi
         + sigma_yy * d_eps_yy0_dphi
         + sigma_zz * d_eps_zz0_dphi
         + 2.0 * sigma_xy * d_eps_xy0_dphi
         + 2.0 * sigma_xz * d_eps_xz0_dphi
         + 2.0 * sigma_yz * d_eps_yz0_dphi );

    double Q = compute_Q_voigt(exx_el, eyy_el, ezz_el, eyz_el, exz_el, exy_el,
        S_p_11, S_p_12, S_p_13, S_p_14, S_p_15, S_p_16,
        S_p_22, S_p_23, S_p_24, S_p_25, S_p_26,
        S_p_33, S_p_34, S_p_35, S_p_36,
        S_p_44, S_p_45, S_p_46,
        S_p_55, S_p_56, S_p_66);
    double dgel_C = 0.5 * hp * Q;
    return dgel_eigen + dgel_C;
}

// Minimize 模式（仅 phi）：eigenstrain = h(phi)*eps^00
// 与 compute_eigenstrain_from_phi_only_kernel 使用完全相同的公式
__device__ __forceinline__
void eigenstrain_phi_only_point(
    double phi,
    float eps_xx00, float eps_yy00, float eps_zz00,
    float eps_yz00, float eps_xz00, float eps_xy00,
    float &eps_xx0_f, float &eps_yy0_f, float &eps_zz0_f,
    float &eps_xy0_f, float &eps_xz0_f, float &eps_yz0_f)
{
    double h   = h_of_phi(phi);
    double eps_xx00_d = (double)eps_xx00;
    double eps_yy00_d = (double)eps_yy00;
    double eps_zz00_d = (double)eps_zz00;
    double eps_yz00_d = (double)eps_yz00;
    double eps_xz00_d = (double)eps_xz00;
    double eps_xy00_d = (double)eps_xy00;
    
    eps_xx0_f = (float)(h * eps_xx00_d);
    eps_yy0_f = (float)(h * eps_yy00_d);
    eps_zz0_f = (float)(h * eps_zz00_d);
    eps_yz0_f = (float)(h * eps_yz00_d);
    eps_xz0_f = (float)(h * eps_xz00_d);
    eps_xy0_f = (float)(h * eps_xy00_d);
}

// ============================================================================
// 2. 相场方程RHS计算kernel（在实空间）
// ============================================================================

__global__ void compute_phi_rhs_kernel(
    const double *phi_r,
    const double *eta_r,
    const double *xB_r,
    double *rhs_r,
    int Nx, int Ny, int Nz,
    double temperature_K,
    double mu_reference_scale,
    double v_A, double v_B,
    double mu0_compound,
    double Vm_compound,
    double Vm_alpha_0,
    double dVm_alpha_dxB,
    double W,
    // 弹性：应力、应变、eigenstrain、S_p
    const float *sigma_xx_r,
    const float *sigma_yy_r,
    const float *sigma_zz_r,
    const float *sigma_xy_r,
    const float *sigma_xz_r,
    const float *sigma_yz_r,
    const float *uxx_r, const float *uyy_r, const float *uzz_r,
    const float *uxy_r, const float *uxz_r, const float *uyz_r,
    // Optimization(4): eigenstrain 参数已移除，现场计算
    double S_p_11, double S_p_12, double S_p_13, double S_p_14, double S_p_15, double S_p_16,
    double S_p_22, double S_p_23, double S_p_24, double S_p_25, double S_p_26,
    double S_p_33, double S_p_34, double S_p_35, double S_p_36,
    double S_p_44, double S_p_45, double S_p_46,
    double S_p_55, double S_p_56,
    double S_p_66,
    double eps_xx00, double eps_yy00, double eps_zz00,
    double eps_yz00, double eps_xz00, double eps_xy00,
    double eps_iso_over_vB,
    int gp_mode_enabled,
    int gp_elastic_enabled,
    int gp_elastic_active_phi,
    double gp_eps_iso,
    double gp_elastic_derivative_scale,
    double elastic_shift_dimless,
    int disable_chem,
    int total_size,
    int elastic_enabled,
    double *diag_stats)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    double phi = phi_r[idx];
    double hp  = h_prime_of_phi(phi);
    double gp  = g_prime_of_phi(phi);
    double h   = h_of_phi(phi);
    double eta = eta_r ? clamp01(eta_r[idx]) : 0.0;
    
    double xB  = clamp01(xB_r[idx]);

    // double-well driving is always kept
    double f_phi = W * gp;

    // chemical driving (bulk thermodynamics) can be disabled in minimize mode
    if (!disable_chem) {
        double muA = mu_A_dimless(xB, temperature_K, mu_reference_scale);
        double muB = mu_B_dimless(xB, temperature_K, mu_reference_scale);

        // delta_mu = mu0_compound - v_A * muA - v_B * muB - elastic_shift_dimless
        // 统一约定：输入的 elastic_shift_dimless 作为要减去的弹性 shift
        double delta_mu = mu0_compound - v_A * muA - v_B * muB - elastic_shift_dimless;

        double c_bulk = c_xB_phi(xB, Vm_alpha_0, dVm_alpha_dxB, Vm_compound, h);
        double mu_total = mu_tot_mix(muA, muB, xB, mu0_compound, h);
        double Vm_alpha = Vm_alpha_of_xB(xB, Vm_alpha_0, dVm_alpha_dxB);
        double volume_term = Vm_compound - Vm_alpha + dVm_alpha_dxB * (xB - v_B);

        double partial_g_bulk = c_bulk * hp * (delta_mu - c_bulk * mu_total * volume_term);
        f_phi += partial_g_bulk;
    }

    if (elastic_enabled) {
        // -------- 弹性部分: dgel/dphi = -sigma:deps0_dphi + 0.5*h'(phi)*Q --------
        // 1) dgel_eigen = - sigma_ij * d eps0_ij / d phi
        double sigma_xx = (double)sigma_xx_r[idx];
        double sigma_yy = (double)sigma_yy_r[idx];
        double sigma_zz = (double)sigma_zz_r[idx];
        double sigma_xy = (double)sigma_xy_r[idx];
        double sigma_xz = (double)sigma_xz_r[idx];
        double sigma_yz = (double)sigma_yz_r[idx];
    
        double eps_c = xB * eps_iso_over_vB;
        double eps_c_prime = eps_iso_over_vB;
        double diag_iso_term = -eps_c + eps_c_prime * (xB - v_B);

        double d_eps_xx0_dphi = hp * (eps_xx00 + diag_iso_term);
        double d_eps_yy0_dphi = hp * (eps_yy00 + diag_iso_term);
        double d_eps_zz0_dphi = hp * (eps_zz00 + diag_iso_term);
        double d_eps_xy0_dphi = hp * eps_xy00;
        double d_eps_xz0_dphi = hp * eps_xz00;
        double d_eps_yz0_dphi = hp * eps_yz00;
    
        double dgel_eigen =
            -( sigma_xx * d_eps_xx0_dphi
             + sigma_yy * d_eps_yy0_dphi
             + sigma_zz * d_eps_zz0_dphi
             + 2.0 * sigma_xy * d_eps_xy0_dphi
             + 2.0 * sigma_xz * d_eps_xz0_dphi
             + 2.0 * sigma_yz * d_eps_yz0_dphi );

        // 2) dgel_C = 0.5 * h'(phi) * Q, Q = e^T * deltaC_voigt * e
        //    eps_el = eps_total - eps_eigen, e = [exx,eyy,ezz,2*eyz,2*exz,2*exy]
        // Optimization(4): 现场计算 eps0
        float eps_xx0_f, eps_yy0_f, eps_zz0_f;
        float eps_xy0_f, eps_xz0_f, eps_yz0_f;
        eigenstrain_phi_xB_point(
            phi, xB,
            (float)eps_xx00, (float)eps_yy00, (float)eps_zz00,
            (float)eps_yz00, (float)eps_xz00, (float)eps_xy00,
            eps_iso_over_vB,
            eps_xx0_f, eps_yy0_f, eps_zz0_f,
            eps_xy0_f, eps_xz0_f, eps_yz0_f);
        
        double exx_el = (double)uxx_r[idx] - (double)eps_xx0_f;
        double eyy_el = (double)uyy_r[idx] - (double)eps_yy0_f;
        double ezz_el = (double)uzz_r[idx] - (double)eps_zz0_f;
        double eyz_el = (double)uyz_r[idx] - (double)eps_yz0_f;
        double exz_el = (double)uxz_r[idx] - (double)eps_xz0_f;
        double exy_el = (double)uxy_r[idx] - (double)eps_xy0_f;
        double elastic_part = 0.0;
        if (gp_mode_enabled && gp_elastic_enabled && gp_elastic_active_phi) {
            elastic_part = gp_elastic_derivative_scale *
                compute_dgel_dphi_gp_point(
                    phi, eta, xB,
                    sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz,
                    exx_el, eyy_el, ezz_el, eyz_el, exz_el, exy_el,
                    v_B, eps_iso_over_vB, gp_eps_iso,
                    S_p_11, S_p_12, S_p_13, S_p_14, S_p_15, S_p_16,
                    S_p_22, S_p_23, S_p_24, S_p_25, S_p_26,
                    S_p_33, S_p_34, S_p_35, S_p_36,
                    S_p_44, S_p_45, S_p_46,
                    S_p_55, S_p_56, S_p_66,
                    eps_xx00, eps_yy00, eps_zz00,
                    eps_yz00, eps_xz00, eps_xy00);
        } else {
            double Q = compute_Q_voigt(exx_el, eyy_el, ezz_el, eyz_el, exz_el, exy_el,
                S_p_11, S_p_12, S_p_13, S_p_14, S_p_15, S_p_16,
                S_p_22, S_p_23, S_p_24, S_p_25, S_p_26,
                S_p_33, S_p_34, S_p_35, S_p_36,
                S_p_44, S_p_45, S_p_46,
                S_p_55, S_p_56, S_p_66);
            double dgel_C = 0.5 * hp * Q;
            elastic_part = dgel_eigen + dgel_C;
        }
        f_phi += elastic_part;
        if (diag_stats && gp_mode_enabled && gp_elastic_enabled && gp_elastic_active_phi) {
            double dgel_raw = (fabs(gp_elastic_derivative_scale) > 1.0e-300)
                                  ? (elastic_part / gp_elastic_derivative_scale)
                                  : 0.0;
            atomicAdd(&diag_stats[GP_ELASTIC_STATS_SUM_DGEL_DPHI], dgel_raw);
            atomicMaxAbsDouble(&diag_stats[GP_ELASTIC_STATS_MAXABS_DGEL_DPHI],
                               fabs(dgel_raw));
            atomicAdd(&diag_stats[GP_ELASTIC_STATS_SUM_PHI_RHS_ELASTIC], elastic_part);
            atomicMaxAbsDouble(&diag_stats[GP_ELASTIC_STATS_MAXABS_PHI_RHS_ELASTIC], fabs(elastic_part));
        }
    }
    rhs_r[idx] = f_phi;
}

// ============================================================================
// Minimize mode helper kernels
// ============================================================================

__global__ void reconstruct_xB_from_phi_kernel(
    const double *phi_r,
    double *xB_r,
    double xB_out,
    double xB_eq,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    double phi = clamp01(phi_r[idx]);
    double h = h_of_phi(phi);
    xB_r[idx] = (1.0 - h) * xB_out + h * xB_eq;
}

__global__ void add_volume_constraint_kernel(
    const double *phi_r,
    double *rhs_r,
    double lambda,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    double phi = phi_r[idx];
    double hp = h_prime_of_phi(phi);
    rhs_r[idx] += lambda * hp;
}

// Minimize: g = rhs_r - kappa*lap_r (in-place: rhs_r -= kappa*lap_r)
__global__ void subtract_scaled_kernel(double *rhs_r, const double *lap_r, double scale, int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    rhs_r[idx] -= scale * lap_r[idx];
}

// Post-update volume projection: phi[i] += lambda_correct * h'(phi[i]), then clamp to [0,1].
// Used to enforce <h(phi)> = V0 after each discrete step (first-order correction).
__global__ void apply_volume_projection_kernel(double *phi_r, double lambda_correct, int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    double phi = phi_r[idx];
    double hp = h_prime_of_phi(phi);
    phi_r[idx] = clamp01(phi + lambda_correct * hp);
}

// ============================================================================
// volume constraint helpers (lagrange mode)
// ============================================================================
__global__ void compute_hprime_values_kernel(
    const double *phi_r,
    double *out_r,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    out_r[idx] = h_prime_of_phi(phi_r[idx]);
}

__global__ void compute_hprime_sq_values_kernel(
    const double *phi_r,
    double *out_r,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    double hp = h_prime_of_phi(phi_r[idx]);
    out_r[idx] = hp * hp;
}

__global__ void compute_hprime_times_rhs_kernel(
    const double *phi_r,
    const double *rhs_r,
    double *out_r,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    double hp = h_prime_of_phi(phi_r[idx]);
    out_r[idx] = hp * rhs_r[idx];
}

// Full Euler-Lagrange residual: res = δF/δφ - λ h' = (f - κ lap φ) - λ h' = rhs - κ*lap_phi - λ h'
// (rhs = f 不含 λ h' 时；或 res = g_full + λ h' 时由 add_volume_constraint 单独加。此处按 res = g_full - λ h' 约定)
// residual_r may alias lap_phi_r (overwrites).
__global__ void compute_euler_lagrange_residual_kernel(
    const double *rhs_r,
    const double *lap_phi_r,
    const double *phi_r,
    double kappa,
    double lambda_vol,
    double *residual_r,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    double lap = lap_phi_r[idx];
    double hp = h_prime_of_phi(phi_r[idx]);
    residual_r[idx] = rhs_r[idx] - kappa * lap - lambda_vol * hp;
}

// bulk chemical free-energy excess density kernel:
// g_bulk_hat(r) = c_tot_hat * [ (1-h(phi)) * mu_mix_hat(xB(r)) + h(phi) * mu0_compound_hat ]
// g_bulk0_hat   = c_tot_hat * mu_mix_hat(xB_ref)
// out[idx]      = g_bulk_hat(r) - g_bulk0_hat
__global__ void compute_gbulk_excess_hat_kernel(
    const double *phi_r,
    const double *xB_r,
    double *gbulk_excess_r,
    double temperature_K,
    double mu_reference_scale,
    double mu0_compound,
    double xB_ref,
    double g_bulk0_hat,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double phi = clamp01(phi_r[idx]);
    double h   = h_of_phi(phi);
    double xB  = clamp_eps(xB_r[idx], 1e-12);

    // mu_mix_hat(xB) = (1 - xB) * muA_hat(xB) + xB * muB_hat(xB)
    double muA = mu_A_dimless(xB, temperature_K, mu_reference_scale);
    double muB = mu_B_dimless(xB, temperature_K, mu_reference_scale);
    double mu_mix_hat = (1.0 - xB) * muA + xB * muB;

    // c_tot_hat 目前取 1.0，与现有 hat 量一致
    double g_bulk_hat = (1.0 - h) * mu_mix_hat + h * mu0_compound;

    // baseline: g_bulk0_hat = c_tot_hat * mu_mix_hat(xB_ref)，已在 host 侧给出
    gbulk_excess_r[idx] = g_bulk_hat - g_bulk0_hat;
}

// 将任意场按 h(phi) 加权：field_r[i] <- field_r[i] * h(phi[i])
__global__ void scale_field_by_h_kernel(
    const double *phi_r,
    double *field_r,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    double phi = clamp01(phi_r[idx]);
    double h = h_of_phi(phi);
    field_r[idx] *= h;
}

__global__ void compute_grad_energy_density_kernel(
    const double *phi_r,
    double *e_grad_r,
    int Nx, int Ny, int Nz,
    double dx, double dy, double dz,
    double kappa_phi,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    int k = idx % Nz;
    int rem = idx / Nz;
    int j = rem % Ny;
    int i = rem / Ny;

    // periodic neighbors
    int ip = (i + 1) % Nx;
    int jp = (j + 1) % Ny;
    int kp = (k + 1) % Nz;
    int im = (i - 1 + Nx) % Nx;
    int jm = (j - 1 + Ny) % Ny;
    int km = (k - 1 + Nz) % Nz;

    // NOTE: avoid device lambda (requires --extended-lambda)
    int idx_xp = (ip * Ny + j ) * Nz + k;
    int idx_xm = (im * Ny + j ) * Nz + k;
    int idx_yp = (i  * Ny + jp) * Nz + k;
    int idx_ym = (i  * Ny + jm) * Nz + k;
    int idx_zp = (i  * Ny + j ) * Nz + kp;
    int idx_zm = (i  * Ny + j ) * Nz + km;

    double dphidx = (phi_r[idx_xp] - phi_r[idx_xm]) / (2.0 * dx);
    double dphidy = (phi_r[idx_yp] - phi_r[idx_ym]) / (2.0 * dy);
    double dphidz = (phi_r[idx_zp] - phi_r[idx_zm]) / (2.0 * dz);
    e_grad_r[idx] = 0.5 * kappa_phi * (dphidx*dphidx + dphidy*dphidy + dphidz*dphidz);
}

// Minimize mode: double-well energy density e_dw = W*g(φ)
__global__ void compute_dw_energy_density_kernel(const double *phi_r, double *e_dw_r, double W, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    e_dw_r[idx] = W * g_of_phi(phi_r[idx]);
}

// ============================================================================
// 3. 相场方程半隐式更新kernel（在k空间）
// ============================================================================

__global__ void phi_semi_implicit_update_kernel(
    const cuDoubleComplex *phi_k_old,
    const cuDoubleComplex *rhs_k,
    const double *k2,
    cuDoubleComplex *phi_k_new,
    double L_phi,
    double kappa_phi,
    double dt,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    double L_dt = L_phi * dt;
    double denom = 1.0 + L_dt * kappa_phi * k2[idx];
    
    cuDoubleComplex phi_old = phi_k_old[idx];
    cuDoubleComplex rhs = rhs_k[idx];
    
    // phi_k_new = (phi_k_old - L_dt * rhs_k) / denom
    cuDoubleComplex numerator = cuCsub(phi_old, cuCmul(make_cuDoubleComplex(L_dt, 0.0), rhs));
    phi_k_new[idx] = cuCdiv(numerator, make_cuDoubleComplex(denom, 0.0));
}

__global__ void compute_eta_rhs_kernel(
    const double *eta_r,
    const double *phi_r,
    const double *xB_alpha_r,
    const float *sigma_xx_r,
    const float *sigma_yy_r,
    const float *sigma_zz_r,
    const float *sigma_xy_r,
    const float *sigma_xz_r,
    const float *sigma_yz_r,
    double *rhs_r,
    double temperature_K,
    double mu_reference_scale,
    double Vm_alpha_0,
    double dVm_alpha_dxB,
    double gp_xB_fixed,
    double gp_delta_g0,
    int gp_raw_reaction_drive_only,
    int gp_raw_reaction_drive_use_raw_units_debug,
    double gp_reaction_nu_A,
    double gp_reaction_nu_B,
    double gp_mu_reference_raw,
    double gp_W_eta,
    double eps_iso_over_vB,
    int gp_elastic_enabled,
    int gp_elastic_active_eta,
    double gp_eps_iso,
    double gp_elastic_derivative_scale,
    int total_size,
    double *diag_stats)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double eta = clamp01(eta_r[idx]);
    double phi = clamp01(phi_r[idx]);
    double xB_alpha = clamp_eps(xB_alpha_r[idx], 1.0e-12);
    double h_phi = h_of_phi(phi);
    double hp_eta = h_prime_of_phi(eta);
    double gp_eta = g_prime_of_phi(eta);

    double c_ref = 1.0 / Vm_alpha_of_xB(xB_alpha, Vm_alpha_0, dVm_alpha_dxB);
    double dgbulk_deta = 0.0;
    if (gp_raw_reaction_drive_only) {
        double muA_alpha_raw = mu_PbTe_calphad(temperature_K, xB_alpha);
        double muB_alpha_raw = mu_Ag2Te_calphad(temperature_K, xB_alpha);
        double minus_delta_mu_r_gp = NAN;
        if (gp_raw_reaction_drive_use_raw_units_debug) {
            minus_delta_mu_r_gp =
                gp_reaction_nu_A * muA_alpha_raw +
                gp_reaction_nu_B * muB_alpha_raw -
                gp_mu_reference_raw;
        } else {
            const double mu_scale =
                (fabs(mu_reference_scale) > 1.0e-300) ? mu_reference_scale : 1.0;
            const double muA_alpha_dimless = muA_alpha_raw / mu_scale;
            const double muB_alpha_dimless = muB_alpha_raw / mu_scale;
            const double gp_mu_reference_dimless = gp_mu_reference_raw / mu_scale;
            minus_delta_mu_r_gp =
                gp_reaction_nu_A * muA_alpha_dimless +
                gp_reaction_nu_B * muB_alpha_dimless -
                gp_mu_reference_dimless;
        }
        dgbulk_deta = c_ref * (1.0 - h_phi) * hp_eta * (-minus_delta_mu_r_gp);
        if (diag_stats) {
            atomicAdd(&diag_stats[GP_ELASTIC_STATS_SUM_MINUS_DELTA_MU_R_GP], minus_delta_mu_r_gp);
            atomicMinDouble(&diag_stats[GP_ELASTIC_STATS_MIN_MINUS_DELTA_MU_R_GP], minus_delta_mu_r_gp);
            atomicMaxDouble(&diag_stats[GP_ELASTIC_STATS_MAX_MINUS_DELTA_MU_R_GP], minus_delta_mu_r_gp);
        }
    } else {
        double muA_alpha = mu_A_dimless(xB_alpha, temperature_K, mu_reference_scale);
        double muB_alpha = mu_B_dimless(xB_alpha, temperature_K, mu_reference_scale);
        double g_alpha = (1.0 - xB_alpha) * muA_alpha + xB_alpha * muB_alpha;

        double xB_gp = clamp_eps(gp_xB_fixed, 1.0e-12);
        double muA_gp = mu_A_dimless(xB_gp, temperature_K, mu_reference_scale);
        double muB_gp = mu_B_dimless(xB_gp, temperature_K, mu_reference_scale);
        double g_gp = (1.0 - xB_gp) * muA_gp + xB_gp * muB_gp - gp_delta_g0;
        dgbulk_deta = c_ref * (1.0 - h_phi) * hp_eta * (g_gp - g_alpha);
    }
    double elastic_part = 0.0;
    if (gp_elastic_enabled && gp_elastic_active_eta) {
        elastic_part = gp_elastic_derivative_scale *
            compute_dgel_deta_gp_point(
                phi, eta, xB_alpha,
                (double)sigma_xx_r[idx], (double)sigma_yy_r[idx], (double)sigma_zz_r[idx],
                (double)sigma_xy_r[idx], (double)sigma_xz_r[idx], (double)sigma_yz_r[idx],
                eps_iso_over_vB, gp_eps_iso);
        if (diag_stats) {
            double dgel_raw = (fabs(gp_elastic_derivative_scale) > 1.0e-300)
                                  ? (elastic_part / gp_elastic_derivative_scale)
                                  : 0.0;
            atomicAdd(&diag_stats[GP_ELASTIC_STATS_SUM_DGEL_DETA], dgel_raw);
            atomicMaxAbsDouble(&diag_stats[GP_ELASTIC_STATS_MAXABS_DGEL_DETA],
                               fabs(dgel_raw));
            atomicAdd(&diag_stats[GP_ELASTIC_STATS_SUM_ETA_RHS_ELASTIC], elastic_part);
            atomicMaxAbsDouble(&diag_stats[GP_ELASTIC_STATS_MAXABS_ETA_RHS_ELASTIC], fabs(elastic_part));
        }
    }

    rhs_r[idx] = dgbulk_deta + gp_W_eta * gp_eta + elastic_part;
}

__global__ void compute_eta_rhs_components_kernel(
    const double *eta_r,
    const double *phi_r,
    const double *xB_alpha_r,
    const float *sigma_xx_r,
    const float *sigma_yy_r,
    const float *sigma_zz_r,
    const float *sigma_xy_r,
    const float *sigma_xz_r,
    const float *sigma_yz_r,
    double *chem_r,
    double *dw_r,
    double *elastic_r,
    double *net_explicit_r,
    double *minus_delta_mu_r_gp_r,
    double temperature_K,
    double mu_reference_scale,
    double Vm_alpha_0,
    double dVm_alpha_dxB,
    double gp_xB_fixed,
    double gp_delta_g0,
    int gp_raw_reaction_drive_only,
    int gp_raw_reaction_drive_use_raw_units_debug,
    double gp_reaction_nu_A,
    double gp_reaction_nu_B,
    double gp_mu_reference_raw,
    double gp_W_eta,
    double eps_iso_over_vB,
    int gp_elastic_enabled,
    int gp_elastic_active_eta,
    double gp_eps_iso,
    double gp_elastic_derivative_scale,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double eta = clamp01(eta_r[idx]);
    double phi = clamp01(phi_r[idx]);
    double xB_alpha = clamp_eps(xB_alpha_r[idx], 1.0e-12);
    double h_phi = h_of_phi(phi);
    double hp_eta = h_prime_of_phi(eta);
    double gp_eta = g_prime_of_phi(eta);
    double c_ref = 1.0 / Vm_alpha_of_xB(xB_alpha, Vm_alpha_0, dVm_alpha_dxB);

    double minus_delta_mu_r_gp = NAN;
    double chem_part = 0.0;
    if (gp_raw_reaction_drive_only) {
        double muA_alpha_raw = mu_PbTe_calphad(temperature_K, xB_alpha);
        double muB_alpha_raw = mu_Ag2Te_calphad(temperature_K, xB_alpha);
        if (gp_raw_reaction_drive_use_raw_units_debug) {
            minus_delta_mu_r_gp =
                gp_reaction_nu_A * muA_alpha_raw +
                gp_reaction_nu_B * muB_alpha_raw -
                gp_mu_reference_raw;
        } else {
            const double mu_scale =
                (fabs(mu_reference_scale) > 1.0e-300) ? mu_reference_scale : 1.0;
            const double muA_alpha_dimless = muA_alpha_raw / mu_scale;
            const double muB_alpha_dimless = muB_alpha_raw / mu_scale;
            const double gp_mu_reference_dimless = gp_mu_reference_raw / mu_scale;
            minus_delta_mu_r_gp =
                gp_reaction_nu_A * muA_alpha_dimless +
                gp_reaction_nu_B * muB_alpha_dimless -
                gp_mu_reference_dimless;
        }
        chem_part = c_ref * (1.0 - h_phi) * hp_eta * (-minus_delta_mu_r_gp);
    } else {
        double muA_alpha = mu_A_dimless(xB_alpha, temperature_K, mu_reference_scale);
        double muB_alpha = mu_B_dimless(xB_alpha, temperature_K, mu_reference_scale);
        double g_alpha = (1.0 - xB_alpha) * muA_alpha + xB_alpha * muB_alpha;

        double xB_gp = clamp_eps(gp_xB_fixed, 1.0e-12);
        double muA_gp = mu_A_dimless(xB_gp, temperature_K, mu_reference_scale);
        double muB_gp = mu_B_dimless(xB_gp, temperature_K, mu_reference_scale);
        double g_gp = (1.0 - xB_gp) * muA_gp + xB_gp * muB_gp - gp_delta_g0;
        chem_part = c_ref * (1.0 - h_phi) * hp_eta * (g_gp - g_alpha);
    }

    double dw_part = gp_W_eta * gp_eta;
    double elastic_part = 0.0;
    if (gp_elastic_enabled && gp_elastic_active_eta) {
        elastic_part = gp_elastic_derivative_scale *
            compute_dgel_deta_gp_point(
                phi, eta, xB_alpha,
                (double)sigma_xx_r[idx], (double)sigma_yy_r[idx], (double)sigma_zz_r[idx],
                (double)sigma_xy_r[idx], (double)sigma_xz_r[idx], (double)sigma_yz_r[idx],
                eps_iso_over_vB, gp_eps_iso);
    }

    if (chem_r) chem_r[idx] = chem_part;
    if (dw_r) dw_r[idx] = dw_part;
    if (elastic_r) elastic_r[idx] = elastic_part;
    if (net_explicit_r) net_explicit_r[idx] = chem_part + dw_part + elastic_part;
    if (minus_delta_mu_r_gp_r) minus_delta_mu_r_gp_r[idx] = minus_delta_mu_r_gp;
}

__global__ void eta_semi_implicit_update_kernel(
    const cuDoubleComplex *eta_k_old,
    const cuDoubleComplex *rhs_k,
    const double *k2,
    cuDoubleComplex *eta_k_new,
    double L_eta,
    double kappa_eta,
    double dt,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double L_dt = L_eta * dt;
    double denom = 1.0 + L_dt * kappa_eta * k2[idx];

    cuDoubleComplex eta_old = eta_k_old[idx];
    cuDoubleComplex rhs = rhs_k[idx];
    cuDoubleComplex numerator = cuCsub(eta_old, cuCmul(make_cuDoubleComplex(L_dt, 0.0), rhs));
    eta_k_new[idx] = cuCdiv(numerator, make_cuDoubleComplex(denom, 0.0));
}

// ============================================================================
// 4. phi归一化和截断kernel
// ============================================================================

__global__ void phi_normalize_and_clamp_kernel(
    double *phi_r,
    double invN,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    phi_r[idx] *= invN;
    
    // 截断到合理范围
    if (phi_r[idx] < -1e-6) phi_r[idx] = -1e-6;
    if (phi_r[idx] > 1.0 + 1e-6) phi_r[idx] = 1.0 + 1e-6;
}

__global__ void eta_normalize_and_clamp_kernel(
    double *eta_r,
    double invN,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double eta = eta_r[idx] * invN;
    if (eta < 0.0) eta = 0.0;
    if (eta > 1.0) eta = 1.0;
    eta_r[idx] = eta;
}

__device__ static inline int gp_recovered_xB_valid_device(double xBtot_available,
                                                           double h_beta,
                                                           double A,
                                                           double h_GP,
                                                           double xB_GP,
                                                           double xB_min,
                                                           double xB_max,
                                                           double eps_h,
                                                           double *xB_rec_out) {
    if (!isfinite(xBtot_available) || !isfinite(h_beta) || !isfinite(A) || !isfinite(h_GP)) return 0;
    if (A < 0.0) return 0;
    if (h_GP < 0.0 || h_GP > A) return 0;
    double h_alpha = A - h_GP;
    if (!(h_alpha >= eps_h)) return 0;
    double numerator = xBtot_available - h_GP * xB_GP - h_beta;
    if (!isfinite(numerator)) return 0;
    double xB_rec = numerator / h_alpha;
    if (!isfinite(xB_rec)) return 0;
    if (xB_rec_out) *xB_rec_out = xB_rec;
    return (xB_rec >= xB_min && xB_rec <= xB_max) ? 1 : 0;
}

__device__ static inline double gp_limited_hGP_device(double xBtot_available,
                                                       double h_beta,
                                                       double A,
                                                       double h_GP_tent,
                                                       double xB_GP,
                                                       double xB_min,
                                                       double xB_max,
                                                       double eps_h) {
    if (!isfinite(xBtot_available) || !isfinite(h_beta) || !isfinite(A) || !isfinite(h_GP_tent)) return 0.0;
    double h_GP_hi = fmin(fmax(h_GP_tent, 0.0), fmax(A, 0.0));
    double xB_tmp = 0.0;
    if (gp_recovered_xB_valid_device(xBtot_available, h_beta, A, h_GP_hi, xB_GP,
                                     xB_min, xB_max, eps_h, &xB_tmp)) {
        return h_GP_hi;
    }
    double lo = 0.0;
    double hi = h_GP_hi;
    for (int iter = 0; iter < 80; ++iter) {
        double mid = 0.5 * (lo + hi);
        if (gp_recovered_xB_valid_device(xBtot_available, h_beta, A, mid, xB_GP,
                                         xB_min, xB_max, eps_h, &xB_tmp)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__global__ void gp_eta_feasibility_and_limiter_kernel(
    double *eta_r,
    const double *eta_old_r,
    const double *phi_new_r,
    const double *phi_old_r,
    const double *xB_old_r,
    const double *divJ_r,
    double dt,
    double xB_GP,
    double xB_min,
    double xB_max,
    double gp_h_alpha_eps,
    int enable_limiter,
    double *stats,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double eta_tent = clamp01(eta_r[idx]);
    double phi_new = clamp01(phi_new_r[idx]);
    double phi_old = clamp01(phi_old_r[idx]);
    double xB_old = clamp_eps(xB_old_r[idx], xB_min);
    double h_alpha_old = 0.0;
    double h_GP_old = 0.0;
    double h_beta_old = 0.0;
    phase_fractions_gp(phi_old, clamp01(eta_old_r[idx]), &h_alpha_old, &h_GP_old, &h_beta_old);
    double h_phi_new = h_of_phi(phi_new);
    double one_minus_hphi_new = fmax(1.0 - h_phi_new, 0.0);
    double h_beta_new = h_phi_new;
    double h_eta_tent = h_of_eta(eta_tent);
    double h_GP_tent = one_minus_hphi_new * h_eta_tent;
    double h_alpha_tent = fmax(one_minus_hphi_new - h_GP_tent, 0.0);
    double xBtot_old = xB_old + h_GP_old * (xB_GP - xB_old) + h_beta_old * (1.0 - xB_old);
    double xBtot_target = xBtot_old + dt * (divJ_r ? divJ_r[idx] : 0.0);
    const double A = one_minus_hphi_new;
    double recovered_before = 0.0;
    int valid_before = gp_recovered_xB_valid_device(xBtot_target, h_beta_new, A, h_GP_tent,
                                                    xB_GP, xB_min, xB_max, gp_h_alpha_eps,
                                                    &recovered_before);
    if (!isfinite(recovered_before)) recovered_before = 0.0;

    double eta_final = eta_tent;
    double h_GP_final = h_GP_tent;
    double limiter_delta = 0.0;
    double limiter_mass_prevented = 0.0;
    int limited = 0;

    if (enable_limiter && A > gp_h_alpha_eps && h_GP_tent > 0.0 && !valid_before) {
        h_GP_final = gp_limited_hGP_device(xBtot_target, h_beta_new, A, h_GP_tent,
                                           xB_GP, xB_min, xB_max, gp_h_alpha_eps);
        double h_eta_allowed = h_GP_final / fmax(A, gp_h_alpha_eps);
        h_eta_allowed = clamp01(h_eta_allowed);
        eta_final = h_inverse_bisection_device(h_eta_allowed);
        if (!isfinite(eta_final)) eta_final = 0.0;
        eta_final = clamp01(eta_final);
        limiter_delta = fabs(eta_tent - eta_final);
        limiter_mass_prevented = fmax(h_GP_tent - h_GP_final, 0.0) * fmax(xB_GP - xB_old, 0.0);
        limited = (limiter_delta > 0.0) ? 1 : 0;
    }

    // This kernel now runs after eta has been normalized back to [0,1], so keep
    // the accepted eta directly in normalized storage.
    eta_r[idx] = eta_final;
    double h_eta_final = h_of_eta(eta_final);
    if (!isfinite(h_eta_final)) h_eta_final = 0.0;
    h_GP_final = one_minus_hphi_new * h_eta_final;
    double h_alpha_final = fmax(one_minus_hphi_new - h_GP_final, 0.0);
    double recovered_final = 0.0;
    int valid_after = gp_recovered_xB_valid_device(xBtot_target, h_beta_new, A, h_GP_final,
                                                   xB_GP, xB_min, xB_max, gp_h_alpha_eps,
                                                   &recovered_final);
    if (!isfinite(recovered_final)) recovered_final = 0.0;
    double recovered_final_mass = h_alpha_final * recovered_final + h_GP_final * xB_GP + h_beta_new;
    double invalid_final_mass = recovered_final_mass - xBtot_target;
    if (!isfinite(invalid_final_mass)) invalid_final_mass = 0.0;

    if (stats) {
        atomicMaxDouble(&stats[GP_ETA_FEAS_MAX_ETA_TENT], eta_tent);
        atomicAdd(&stats[GP_ETA_FEAS_SUM_RECOVERED_XB_BEFORE], recovered_before);
        atomicMinDouble(&stats[GP_ETA_FEAS_MIN_RECOVERED_XB_BEFORE], recovered_before);
        atomicMaxDouble(&stats[GP_ETA_FEAS_MAX_RECOVERED_XB_BEFORE], recovered_before);
        atomicAdd(&stats[GP_ETA_FEAS_GT_XBMAX_COUNT_BEFORE], (recovered_before > xB_max) ? 1.0 : 0.0);
        atomicAdd(&stats[GP_ETA_FEAS_LT_XBMIN_COUNT_BEFORE], (recovered_before < xB_min) ? 1.0 : 0.0);
        atomicAdd(&stats[GP_ETA_FEAS_INVALID_COUNT_BEFORE], valid_before ? 0.0 : 1.0);
        atomicAdd(&stats[GP_ETA_FEAS_SUM_RECOVERED_XB_AFTER], recovered_final);
        atomicMinDouble(&stats[GP_ETA_FEAS_MIN_RECOVERED_XB_AFTER], recovered_final);
        atomicMaxDouble(&stats[GP_ETA_FEAS_MAX_RECOVERED_XB_AFTER], recovered_final);
        atomicAdd(&stats[GP_ETA_FEAS_GT_XBMAX_COUNT_AFTER], (recovered_final > xB_max) ? 1.0 : 0.0);
        atomicAdd(&stats[GP_ETA_FEAS_LT_XBMIN_COUNT_AFTER], (recovered_final < xB_min) ? 1.0 : 0.0);
        atomicAdd(&stats[GP_ETA_FEAS_INVALID_COUNT_AFTER], valid_after ? 0.0 : 1.0);
        atomicAdd(&stats[GP_ETA_FEAS_INVALID_MASS], invalid_final_mass);
        atomicMaxDouble(&stats[GP_ETA_FEAS_MAX_HGP_NEW], h_GP_final);
        atomicMaxDouble(&stats[GP_ETA_FEAS_MAX_ETA_NEW], eta_final);
        atomicAdd(&stats[GP_ETA_FEAS_LIMITER_COUNT], (double)limited);
        atomicMaxDouble(&stats[GP_ETA_FEAS_LIMITER_MAX_DELTA], limiter_delta);
        atomicAdd(&stats[GP_ETA_FEAS_LIMITER_MASS_PREVENTED], limiter_mass_prevented);
    }
}

// 仅归一化（无截断），供非phi/Y数组使用
__global__ void normalize_only_kernel(
    double *arr,
    double invN,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    arr[idx] *= invN;
}

// ============================================================================
// 5. Y方程相关kernel：计算mu_x（化学势差）
// ============================================================================

__global__ void compute_mu_x_kernel(
    const double *Y_r,
    const double *phi_r,
    double *xB_r,
    double *mu_x_r,
    double temperature_K,
    double mu_reference_scale,
    double v_A, double v_B,
    double mu0_compound,
    double Vm_compound,
    double Vm_alpha_0,
    double dVm_alpha_dxB,
    double Y_clip,
    double xB_eps,
    const float *sigma_xx_r,
    const float *sigma_yy_r,
    const float *sigma_zz_r,
    double eps_iso_over_vB,
    int total_size,
    int elastic_enabled)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    // 从logit转换为xB
    double Y = Y_r[idx];
    double xB = sigmoid_from_logit(Y, Y_clip, xB_eps);
    xB_r[idx] = xB;

    double phi = phi_r[idx];
    double h   = h_of_phi(phi);
    
    double muA = mu_A_dimless(xB, temperature_K, mu_reference_scale);
    double muB = mu_B_dimless(xB, temperature_K, mu_reference_scale);
    double c_bulk = c_xB_phi(xB, Vm_alpha_0, dVm_alpha_dxB, Vm_compound, h);
    double mu_total = mu_tot_mix(muA, muB, xB, mu0_compound, h);
    
    double mu_x_val = c_bulk * (muB - muA - c_bulk * mu_total * dVm_alpha_dxB);


    // 如启用弹性，再加上弹性对 μ_x 的修正: d g_el / d xB_tot
    if (elastic_enabled) {
        double sigma_xx = (double)sigma_xx_r[idx];
        double sigma_yy = (double)sigma_yy_r[idx];
        double sigma_zz = (double)sigma_zz_r[idx];
        double sigma_hydro = sigma_xx + sigma_yy + sigma_zz;

        // eps_c(xB) = xB * eps_iso_over_vB  =>  eps_c'(xB) = eps_iso_over_vB
        double eps_c_prime = eps_iso_over_vB;

        // d g_el / d xB_tot = - eps_c'(xB) * (sigma_xx + sigma_yy + sigma_zz)
        double dgel_dxBtot = -eps_c_prime * sigma_hydro;

        // 假设弹性能量已经按与 mu_A_dimless / mu_B_dimless 相同的尺度无量纲化，
        // 则这项可以直接加到 mu_x 中
        mu_x_val += dgel_dxBtot;
    }
    mu_x_r[idx] = mu_x_val;
}

__global__ void compute_mu_C_gp_kernel(
    const double *Y_r,
    double *xB_alpha_r,
    double *mu_C_r,
    double temperature_K,
    double mu_reference_scale,
    double Vm_alpha_0,
    double dVm_alpha_dxB,
    double Y_clip,
    double xB_eps,
    const float *sigma_xx_r,
    const float *sigma_yy_r,
    const float *sigma_zz_r,
    double eps_iso_over_vB,
    int total_size,
    int elastic_enabled)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double xB_alpha = sigmoid_from_logit(Y_r[idx], Y_clip, xB_eps);
    xB_alpha_r[idx] = xB_alpha;

    double mu_C_val = 0.0;
    double muA = mu_A_dimless(xB_alpha, temperature_K, mu_reference_scale);
    double muB = mu_B_dimless(xB_alpha, temperature_K, mu_reference_scale);
    double c_ref = 1.0 / Vm_alpha_of_xB(xB_alpha, Vm_alpha_0, dVm_alpha_dxB);
    mu_C_val = c_ref * (muB - muA);

    if (elastic_enabled) {
        double sigma_hydro = (double)sigma_xx_r[idx] +
                             (double)sigma_yy_r[idx] +
                             (double)sigma_zz_r[idx];
        mu_C_val += -eps_iso_over_vB * sigma_hydro;
    }

    mu_C_r[idx] = mu_C_val;
}

// ============================================================================
// 5.0b Minimize mode helper: compute mu_x from (phi, xB) directly (no Y/logit)
// ============================================================================
__global__ void compute_mu_x_from_xB_kernel(
    const double *phi_r,
    const double *xB_r,
    double *mu_x_r,
    double temperature_K,
    double mu_reference_scale,
    double v_A, double v_B,
    double mu0_compound,
    double Vm_compound,
    double Vm_alpha_0,
    double dVm_alpha_dxB,
    const float *sigma_xx_r,
    const float *sigma_yy_r,
    const float *sigma_zz_r,
    double eps_iso_over_vB,
    int total_size,
    int elastic_enabled)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double xB = clamp01(xB_r[idx]);
    double phi = phi_r[idx];
    double h   = h_of_phi(phi);

    double muA = mu_A_dimless(xB, temperature_K, mu_reference_scale);
    double muB = mu_B_dimless(xB, temperature_K, mu_reference_scale);
    double c_bulk = c_xB_phi(xB, Vm_alpha_0, dVm_alpha_dxB, Vm_compound, h);
    double mu_total = mu_tot_mix(muA, muB, xB, mu0_compound, h);

    double mu_x_val = c_bulk * (muB - muA - c_bulk * mu_total * dVm_alpha_dxB);

    if (elastic_enabled) {
        double sigma_xx = (double)sigma_xx_r[idx];
        double sigma_yy = (double)sigma_yy_r[idx];
        double sigma_zz = (double)sigma_zz_r[idx];
        double sigma_hydro = sigma_xx + sigma_yy + sigma_zz;

        // d g_el / d xB_tot = - eps_c'(xB) * tr(sigma), with eps_c'(xB)=eps_iso_over_vB
        double dgel_dxBtot = -eps_iso_over_vB * sigma_hydro;
        mu_x_val += dgel_dxBtot;
    }

    mu_x_r[idx] = mu_x_val;
}

// ============================================================================
// 5.0c Minimize mode chain rule: rhs_phi += (∂f/∂xB) * (dxB/dphi)
// ============================================================================
__global__ void add_xB_chain_rule_to_phi_rhs_kernel(
    const double *phi_r,
    const double *mu_x_r,
    double *rhs_phi_r,
    double xB_out,
    double xB_eq,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double phi = phi_r[idx];
    double hp  = h_prime_of_phi(phi);
    double dxB_dphi = (xB_eq - xB_out) * hp;

    rhs_phi_r[idx] += mu_x_r[idx] * dxB_dphi;
}

// ============================================================================
// 5.1 计算delta_mu_r（用于VTK输出）
// ============================================================================

__global__ void compute_delta_mu_r_kernel(
    const double *phi_r,
    const double *xB_r,
    double *delta_mu_r,
    double temperature_K,
    double mu_reference_scale,
    double v_A, double v_B,
    double mu0_compound,
    double elastic_shift_dimless,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    double xB = clamp01(xB_r[idx]);
    double muA = mu_A_dimless(xB, temperature_K, mu_reference_scale);
    double muB = mu_B_dimless(xB, temperature_K, mu_reference_scale);
    
    // delta_mu = mu0_compound - v_A * muA - v_B * muB - elastic_shift_dimless
    // 统一约定：输入的 elastic_shift_dimless 作为要减去的弹性 shift
    double delta_mu = mu0_compound - v_A * muA - v_B * muB - elastic_shift_dimless;
    
    delta_mu_r[idx] = delta_mu;
}

// ============================================================================
// 5.2 计算f_phi_chem和dgel_dphi（用于VTK输出）
// ============================================================================

__global__ void compute_f_phi_chem_dgel_dphi_kernel(
    const double *phi_r,
    const double *xB_r,
    double *f_phi_chem_r,
    double *f_phi_dw_r,
    double *f_phi_bulk_r,
    double *dgel_dphi_r,
    double temperature_K,
    double mu_reference_scale,
    double v_A, double v_B,
    double mu0_compound,
    double Vm_compound,
    double Vm_alpha_0,
    double dVm_alpha_dxB,
    double W,
    // 弹性：应力、应变、S_p（Optimization(4): eigenstrain 参数已移除，现场计算）
    const float *sigma_xx_r,
    const float *sigma_yy_r,
    const float *sigma_zz_r,
    const float *sigma_xy_r,
    const float *sigma_xz_r,
    const float *sigma_yz_r,
    const float *uxx_r, const float *uyy_r, const float *uzz_r,
    const float *uxy_r, const float *uxz_r, const float *uyz_r,
    double S_p_11, double S_p_12, double S_p_13, double S_p_14, double S_p_15, double S_p_16,
    double S_p_22, double S_p_23, double S_p_24, double S_p_25, double S_p_26,
    double S_p_33, double S_p_34, double S_p_35, double S_p_36,
    double S_p_44, double S_p_45, double S_p_46,
    double S_p_55, double S_p_56,
    double S_p_66,
    double eps_xx00, double eps_yy00, double eps_zz00,
    double eps_yz00, double eps_xz00, double eps_xy00,
    double eps_iso_over_vB,
    double elastic_shift_dimless,
    int total_size,
    int elastic_enabled)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    double phi = phi_r[idx];
    double hp  = h_prime_of_phi(phi);
    double gp  = g_prime_of_phi(phi);
    double h   = h_of_phi(phi);
    
    double xB  = clamp01(xB_r[idx]);
    double muA = mu_A_dimless(xB, temperature_K, mu_reference_scale);
    double muB = mu_B_dimless(xB, temperature_K, mu_reference_scale);
    
    // delta_mu = mu0_compound - v_A * muA - v_B * muB - elastic_shift_dimless
    // 统一约定：输入的 elastic_shift_dimless 作为要减去的弹性 shift
    double delta_mu = mu0_compound - v_A * muA - v_B * muB - elastic_shift_dimless;
    
    double c_bulk = c_xB_phi(xB, Vm_alpha_0, dVm_alpha_dxB, Vm_compound, h);
    double mu_total = mu_tot_mix(muA, muB, xB, mu0_compound, h);
    double Vm_alpha = Vm_alpha_of_xB(xB, Vm_alpha_0, dVm_alpha_dxB);
    double volume_term = Vm_compound - Vm_alpha + dVm_alpha_dxB * (xB - v_B);
    
    double partial_g_bulk = c_bulk * hp * (delta_mu - c_bulk * mu_total * volume_term);
    double f_phi_dw = W * gp;
    double f_phi_chem = partial_g_bulk + f_phi_dw;
    
    f_phi_chem_r[idx] = f_phi_chem;
    if (f_phi_dw_r)   f_phi_dw_r[idx] = f_phi_dw;
    if (f_phi_bulk_r) f_phi_bulk_r[idx] = partial_g_bulk;
    
    // 计算弹性驱动力 dgel/dphi = dgel_eigen + dgel_C
    double dgel_dphi = 0.0;
    if (elastic_enabled) {
        double sigma_xx = (double)sigma_xx_r[idx];
        double sigma_yy = (double)sigma_yy_r[idx];
        double sigma_zz = (double)sigma_zz_r[idx];
        double sigma_xy = (double)sigma_xy_r[idx];
        double sigma_xz = (double)sigma_xz_r[idx];
        double sigma_yz = (double)sigma_yz_r[idx];
    
        double eps_c = xB * eps_iso_over_vB;
        double eps_c_prime = eps_iso_over_vB;
        double diag_iso_term = -eps_c + eps_c_prime * (xB - v_B);

        double d_eps_xx0_dphi = hp * (eps_xx00 + diag_iso_term);
        double d_eps_yy0_dphi = hp * (eps_yy00 + diag_iso_term);
        double d_eps_zz0_dphi = hp * (eps_zz00 + diag_iso_term);
        double d_eps_xy0_dphi = hp * eps_xy00;
        double d_eps_xz0_dphi = hp * eps_xz00;
        double d_eps_yz0_dphi = hp * eps_yz00;
    
        double dgel_eigen =
            -( sigma_xx * d_eps_xx0_dphi
             + sigma_yy * d_eps_yy0_dphi
             + sigma_zz * d_eps_zz0_dphi
             + 2.0 * sigma_xy * d_eps_xy0_dphi
             + 2.0 * sigma_xz * d_eps_xz0_dphi
             + 2.0 * sigma_yz * d_eps_yz0_dphi );

        // Optimization(4): 现场计算 eps0
        float eps_xx0_f, eps_yy0_f, eps_zz0_f;
        float eps_xy0_f, eps_xz0_f, eps_yz0_f;
        eigenstrain_phi_xB_point(phi, xB, (float)eps_xx00, (float)eps_yy00, (float)eps_zz00,
                                  (float)eps_yz00, (float)eps_xz00, (float)eps_xy00,
                                  eps_iso_over_vB, eps_xx0_f, eps_yy0_f, eps_zz0_f,
                                  eps_xy0_f, eps_xz0_f, eps_yz0_f);
        double exx_el = (double)uxx_r[idx] - (double)eps_xx0_f;
        double eyy_el = (double)uyy_r[idx] - (double)eps_yy0_f;
        double ezz_el = (double)uzz_r[idx] - (double)eps_zz0_f;
        double eyz_el = (double)uyz_r[idx] - (double)eps_yz0_f;
        double exz_el = (double)uxz_r[idx] - (double)eps_xz0_f;
        double exy_el = (double)uxy_r[idx] - (double)eps_xy0_f;
        double Q = compute_Q_voigt(exx_el, eyy_el, ezz_el, eyz_el, exz_el, exy_el,
            S_p_11, S_p_12, S_p_13, S_p_14, S_p_15, S_p_16,
            S_p_22, S_p_23, S_p_24, S_p_25, S_p_26,
            S_p_33, S_p_34, S_p_35, S_p_36,
            S_p_44, S_p_45, S_p_46,
            S_p_55, S_p_56, S_p_66);
        double dgel_C = 0.5 * hp * Q;
        dgel_dphi = dgel_eigen + dgel_C;
    }
    dgel_dphi_r[idx] = dgel_dphi;
}

// ============================================================================
// 5.3 辅助：缩放数组（用于梯度项等）
// ============================================================================
__global__ void scale_array_kernel(double *arr, double factor, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    arr[idx] *= factor;
}

__global__ void add_arrays_kernel(const double *a, const double *b, double *out, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    out[idx] = a[idx] + b[idx];
}

// Optimization: 拆分为 OutA(chem,dw) 与 OutB(bulk,dgel)，便于 2-slot scratch 串行复用
__global__ void compute_f_phi_chem_f_phi_dw_kernel(
    const double *phi_r, const double *xB_r,
    double *f_phi_chem_r, double *f_phi_dw_r,
    double temperature_K, double mu_reference_scale,
    double v_A, double v_B, double mu0_compound,
    double Vm_compound, double Vm_alpha_0, double dVm_alpha_dxB,
    double W, double elastic_shift_dimless,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    double phi = phi_r[idx];
    double hp  = h_prime_of_phi(phi);
    double gp  = g_prime_of_phi(phi);
    double h   = h_of_phi(phi);
    double xB  = clamp01(xB_r[idx]);
    double muA = mu_A_dimless(xB, temperature_K, mu_reference_scale);
    double muB = mu_B_dimless(xB, temperature_K, mu_reference_scale);
    double delta_mu = mu0_compound - v_A * muA - v_B * muB - elastic_shift_dimless;
    double c_bulk = c_xB_phi(xB, Vm_alpha_0, dVm_alpha_dxB, Vm_compound, h);
    double mu_total = mu_tot_mix(muA, muB, xB, mu0_compound, h);
    double Vm_alpha = Vm_alpha_of_xB(xB, Vm_alpha_0, dVm_alpha_dxB);
    double volume_term = Vm_compound - Vm_alpha + dVm_alpha_dxB * (xB - v_B);
    double partial_g_bulk = c_bulk * hp * (delta_mu - c_bulk * mu_total * volume_term);
    double f_phi_dw = W * gp;
    f_phi_chem_r[idx] = partial_g_bulk + f_phi_dw;
    f_phi_dw_r[idx] = f_phi_dw;
}

__global__ void compute_f_phi_bulk_dgel_dphi_kernel(
    const double *phi_r, const double *xB_r,
    double *f_phi_bulk_r, double *dgel_dphi_r,
    double temperature_K, double mu_reference_scale,
    double v_A, double v_B, double mu0_compound,
    double Vm_compound, double Vm_alpha_0, double dVm_alpha_dxB,
    const float *sigma_xx_r, const float *sigma_yy_r, const float *sigma_zz_r,
    const float *sigma_xy_r, const float *sigma_xz_r, const float *sigma_yz_r,
    const float *uxx_r, const float *uyy_r, const float *uzz_r,
    const float *uxy_r, const float *uxz_r, const float *uyz_r,
    // Optimization(4): eigenstrain 参数已移除，现场计算
    double S_p_11, double S_p_12, double S_p_13, double S_p_14, double S_p_15, double S_p_16,
    double S_p_22, double S_p_23, double S_p_24, double S_p_25, double S_p_26,
    double S_p_33, double S_p_34, double S_p_35, double S_p_36,
    double S_p_44, double S_p_45, double S_p_46, double S_p_55, double S_p_56, double S_p_66,
    double eps_xx00, double eps_yy00, double eps_zz00,
    double eps_yz00, double eps_xz00, double eps_xy00,
    double eps_iso_over_vB, double elastic_shift_dimless,
    int total_size, int elastic_enabled)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    double phi = phi_r[idx];
    double hp  = h_prime_of_phi(phi);
    double h   = h_of_phi(phi);
    double xB  = clamp01(xB_r[idx]);
    double muA = mu_A_dimless(xB, temperature_K, mu_reference_scale);
    double muB = mu_B_dimless(xB, temperature_K, mu_reference_scale);
    double delta_mu = mu0_compound - v_A * muA - v_B * muB - elastic_shift_dimless;
    double c_bulk = c_xB_phi(xB, Vm_alpha_0, dVm_alpha_dxB, Vm_compound, h);
    double mu_total = mu_tot_mix(muA, muB, xB, mu0_compound, h);
    double Vm_alpha = Vm_alpha_of_xB(xB, Vm_alpha_0, dVm_alpha_dxB);
    double volume_term = Vm_compound - Vm_alpha + dVm_alpha_dxB * (xB - v_B);
    double partial_g_bulk = c_bulk * hp * (delta_mu - c_bulk * mu_total * volume_term);
    f_phi_bulk_r[idx] = partial_g_bulk;
    double dgel_dphi = 0.0;
    if (elastic_enabled) {
        double sigma_xx = (double)sigma_xx_r[idx], sigma_yy = (double)sigma_yy_r[idx], sigma_zz = (double)sigma_zz_r[idx];
        double sigma_xy = (double)sigma_xy_r[idx], sigma_xz = (double)sigma_xz_r[idx], sigma_yz = (double)sigma_yz_r[idx];
        double eps_c = xB * eps_iso_over_vB, eps_c_prime = eps_iso_over_vB;
        double diag_iso_term = -eps_c + eps_c_prime * (xB - v_B);
        double d_eps_xx0_dphi = hp * (eps_xx00 + diag_iso_term);
        double d_eps_yy0_dphi = hp * (eps_yy00 + diag_iso_term);
        double d_eps_zz0_dphi = hp * (eps_zz00 + diag_iso_term);
        double d_eps_xy0_dphi = hp * eps_xy00;
        double d_eps_xz0_dphi = hp * eps_xz00;
        double d_eps_yz0_dphi = hp * eps_yz00;
        double dgel_eigen = -(sigma_xx * d_eps_xx0_dphi + sigma_yy * d_eps_yy0_dphi + sigma_zz * d_eps_zz0_dphi
            + 2.0 * sigma_xy * d_eps_xy0_dphi + 2.0 * sigma_xz * d_eps_xz0_dphi + 2.0 * sigma_yz * d_eps_yz0_dphi);
        // Optimization(4): 现场计算 eps0
        float eps_xx0_f, eps_yy0_f, eps_zz0_f;
        float eps_xy0_f, eps_xz0_f, eps_yz0_f;
        eigenstrain_phi_xB_point(phi, xB, (float)eps_xx00, (float)eps_yy00, (float)eps_zz00,
                                  (float)eps_yz00, (float)eps_xz00, (float)eps_xy00,
                                  eps_iso_over_vB, eps_xx0_f, eps_yy0_f, eps_zz0_f,
                                  eps_xy0_f, eps_xz0_f, eps_yz0_f);
        double exx_el = (double)uxx_r[idx] - (double)eps_xx0_f, eyy_el = (double)uyy_r[idx] - (double)eps_yy0_f;
        double ezz_el = (double)uzz_r[idx] - (double)eps_zz0_f, eyz_el = (double)uyz_r[idx] - (double)eps_yz0_f;
        double exz_el = (double)uxz_r[idx] - (double)eps_xz0_f, exy_el = (double)uxy_r[idx] - (double)eps_xy0_f;
        double Q = compute_Q_voigt(exx_el, eyy_el, ezz_el, eyz_el, exz_el, exy_el,
            S_p_11, S_p_12, S_p_13, S_p_14, S_p_15, S_p_16,
            S_p_22, S_p_23, S_p_24, S_p_25, S_p_26,
            S_p_33, S_p_34, S_p_35, S_p_36,
            S_p_44, S_p_45, S_p_46, S_p_55, S_p_56, S_p_66);
        dgel_dphi = dgel_eigen + 0.5 * hp * Q;
    }
    dgel_dphi_r[idx] = dgel_dphi;
}

// Optimization: 直接计算 driving_force = dw + bulk + dgel + f_phi_grad，避免同时持有 4 个 size_r buffer
__global__ void compute_driving_force_direct_kernel(
    const double *phi_r, const double *xB_r,
    const double *f_phi_grad_r,
    double *driving_force_r,
    double temperature_K, double mu_reference_scale,
    double v_A, double v_B, double mu0_compound,
    double Vm_compound, double Vm_alpha_0, double dVm_alpha_dxB,
    double W, double elastic_shift_dimless,
    const float *sigma_xx_r, const float *sigma_yy_r, const float *sigma_zz_r,
    const float *sigma_xy_r, const float *sigma_xz_r, const float *sigma_yz_r,
    const float *uxx_r, const float *uyy_r, const float *uzz_r,
    const float *uxy_r, const float *uxz_r, const float *uyz_r,
    // Optimization(4): eigenstrain 参数已移除，现场计算
    double S_p_11, double S_p_12, double S_p_13, double S_p_14, double S_p_15, double S_p_16,
    double S_p_22, double S_p_23, double S_p_24, double S_p_25, double S_p_26,
    double S_p_33, double S_p_34, double S_p_35, double S_p_36,
    double S_p_44, double S_p_45, double S_p_46, double S_p_55, double S_p_56, double S_p_66,
    double eps_xx00, double eps_yy00, double eps_zz00,
    double eps_yz00, double eps_xz00, double eps_xy00,
    double eps_iso_over_vB,
    int total_size, int elastic_enabled)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    double phi = phi_r[idx], hp = h_prime_of_phi(phi), gp = g_prime_of_phi(phi), h = h_of_phi(phi);
    double xB = clamp01(xB_r[idx]);
    double muA = mu_A_dimless(xB, temperature_K, mu_reference_scale);
    double muB = mu_B_dimless(xB, temperature_K, mu_reference_scale);
    double delta_mu = mu0_compound - v_A * muA - v_B * muB - elastic_shift_dimless;
    double c_bulk = c_xB_phi(xB, Vm_alpha_0, dVm_alpha_dxB, Vm_compound, h);
    double mu_total = mu_tot_mix(muA, muB, xB, mu0_compound, h);
    double Vm_alpha = Vm_alpha_of_xB(xB, Vm_alpha_0, dVm_alpha_dxB);
    double volume_term = Vm_compound - Vm_alpha + dVm_alpha_dxB * (xB - v_B);
    double partial_g_bulk = c_bulk * hp * (delta_mu - c_bulk * mu_total * volume_term);
    double f_phi_dw = W * gp;
    double dgel_dphi = 0.0;
    if (elastic_enabled) {
        double sigma_xx = (double)sigma_xx_r[idx], sigma_yy = (double)sigma_yy_r[idx], sigma_zz = (double)sigma_zz_r[idx];
        double sigma_xy = (double)sigma_xy_r[idx], sigma_xz = (double)sigma_xz_r[idx], sigma_yz = (double)sigma_yz_r[idx];
        double eps_c = xB * eps_iso_over_vB, eps_c_prime = eps_iso_over_vB;
        double diag_iso_term = -eps_c + eps_c_prime * (xB - v_B);
        double d_eps_xx0_dphi = hp * (eps_xx00 + diag_iso_term);
        double d_eps_yy0_dphi = hp * (eps_yy00 + diag_iso_term);
        double d_eps_zz0_dphi = hp * (eps_zz00 + diag_iso_term);
        double d_eps_xy0_dphi = hp * eps_xy00;
        double d_eps_xz0_dphi = hp * eps_xz00;
        double d_eps_yz0_dphi = hp * eps_yz00;
        double dgel_eigen = -(sigma_xx * d_eps_xx0_dphi + sigma_yy * d_eps_yy0_dphi + sigma_zz * d_eps_zz0_dphi
            + 2.0 * sigma_xy * d_eps_xy0_dphi + 2.0 * sigma_xz * d_eps_xz0_dphi + 2.0 * sigma_yz * d_eps_yz0_dphi);
        // Optimization(4): 现场计算 eps0
        float eps_xx0_f, eps_yy0_f, eps_zz0_f;
        float eps_xy0_f, eps_xz0_f, eps_yz0_f;
        eigenstrain_phi_xB_point(phi, xB, (float)eps_xx00, (float)eps_yy00, (float)eps_zz00,
                                  (float)eps_yz00, (float)eps_xz00, (float)eps_xy00,
                                  eps_iso_over_vB, eps_xx0_f, eps_yy0_f, eps_zz0_f,
                                  eps_xy0_f, eps_xz0_f, eps_yz0_f);
        double exx_el = (double)uxx_r[idx] - (double)eps_xx0_f, eyy_el = (double)uyy_r[idx] - (double)eps_yy0_f;
        double ezz_el = (double)uzz_r[idx] - (double)eps_zz0_f, eyz_el = (double)uyz_r[idx] - (double)eps_yz0_f;
        double exz_el = (double)uxz_r[idx] - (double)eps_xz0_f, exy_el = (double)uxy_r[idx] - (double)eps_xy0_f;
        double Q = compute_Q_voigt(exx_el, eyy_el, ezz_el, eyz_el, exz_el, exy_el,
            S_p_11, S_p_12, S_p_13, S_p_14, S_p_15, S_p_16,
            S_p_22, S_p_23, S_p_24, S_p_25, S_p_26,
            S_p_33, S_p_34, S_p_35, S_p_36,
            S_p_44, S_p_45, S_p_46, S_p_55, S_p_56, S_p_66);
        dgel_dphi = dgel_eigen + 0.5 * hp * Q;
    }
    driving_force_r[idx] = f_phi_dw + partial_g_bulk + dgel_dphi + f_phi_grad_r[idx];
}

__global__ void sum_driving_force_kernel(const double *dw, const double *bulk,
                                         const double *grad, const double *el,
                                         double *out, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    double v = 0.0;
    if (dw)   v += dw[idx];
    if (bulk) v += bulk[idx];
    if (grad) v += grad[idx];
    if (el)   v += el[idx];
    out[idx] = v;
}

// ============================================================================
// 5.5 计算弹性能密度 gel_hat：0.5 * sigma_ij * eps_elastic_ij
// eps_elastic = eps_total - eps_eigen，剪切项按张量收缩需乘 2
// ============================================================================
__global__ void compute_gel_density_kernel(
    const float *uxx_r, const float *uyy_r, const float *uzz_r,
    const float *uxy_r, const float *uxz_r, const float *uyz_r,
    // Optimization(4): eigenstrain 参数已移除，现场计算；需要 phi 和 xB
    const double *phi_r,
    const double *eta_r,
    const double *xB_r,  // 可为 NULL（minimize 模式）
    const float *sigma_xx_r, const float *sigma_yy_r, const float *sigma_zz_r,
    const float *sigma_xy_r, const float *sigma_xz_r, const float *sigma_yz_r,
    // Optimization(4): 需要 eps0 相关参数
    float eps_xx00, float eps_yy00, float eps_zz00,
    float eps_yz00, float eps_xz00, float eps_xy00,
    double eps_iso_over_vB,
    int gp_mode_enabled,
    int gp_elastic_enabled,
    double gp_eps_iso,
    double *gel_hat_r,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    // Optimization(4): 现场计算 eigenstrain
    double phi = phi_r[idx];
    double eta = eta_r ? eta_r[idx] : 0.0;
    double xB = (xB_r != NULL) ? clamp01(xB_r[idx]) : 0.0;
    float eps_xx0_f, eps_yy0_f, eps_zz0_f;
    float eps_xy0_f, eps_xz0_f, eps_yz0_f;
    if (xB_r != NULL) {
        eigenstrain_phi_eta_xB_point(phi, eta, xB, eps_xx00, eps_yy00, eps_zz00,
                                     eps_yz00, eps_xz00, eps_xy00, eps_iso_over_vB,
                                     gp_mode_enabled, gp_elastic_enabled, gp_eps_iso,
                                     eps_xx0_f, eps_yy0_f, eps_zz0_f,
                                     eps_xy0_f, eps_xz0_f, eps_yz0_f);
    } else {
        eigenstrain_phi_only_point(phi, eps_xx00, eps_yy00, eps_zz00, eps_yz00, eps_xz00, eps_xy00,
                                    eps_xx0_f, eps_yy0_f, eps_zz0_f, eps_xy0_f, eps_xz0_f, eps_yz0_f);
    }

    // elastic strain
    double exx = (double)uxx_r[idx] - (double)eps_xx0_f;
    double eyy = (double)uyy_r[idx] - (double)eps_yy0_f;
    double ezz = (double)uzz_r[idx] - (double)eps_zz0_f;
    double exy = (double)uxy_r[idx] - (double)eps_xy0_f;
    double exz = (double)uxz_r[idx] - (double)eps_xz0_f;
    double eyz = (double)uyz_r[idx] - (double)eps_yz0_f;

    // stress
    double sxx = (double)sigma_xx_r[idx];
    double syy = (double)sigma_yy_r[idx];
    double szz = (double)sigma_zz_r[idx];
    double sxy = (double)sigma_xy_r[idx];
    double sxz = (double)sigma_xz_r[idx];
    double syz = (double)sigma_yz_r[idx];

    // sigma:epsilon = sxx*exx + syy*eyy + szz*ezz + 2*(sxy*exy + sxz*exz + syz*eyz)
    double sig_dot_eps =
        sxx * exx + syy * eyy + szz * ezz
        + 2.0 * (sxy * exy + sxz * exz + syz * eyz);

    gel_hat_r[idx] = 0.5 * sig_dot_eps;
}

__global__ void compute_gp_elastic_stats_kernel(
    const double *phi_r,
    const double *eta_r,
    const double *gel_hat_r,
    const float *sigma_xx_r,
    const float *sigma_yy_r,
    const float *sigma_zz_r,
    double gp_eps_iso,
    double *stats,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double gel = gel_hat_r[idx];
    double sigma_hydro = (double)sigma_xx_r[idx] + (double)sigma_yy_r[idx] + (double)sigma_zz_r[idx];
    double h_alpha = 0.0;
    double h_GP = 0.0;
    double h_beta = 0.0;
    phase_fractions_gp(phi_r[idx], eta_r[idx], &h_alpha, &h_GP, &h_beta);
    double eps0_gp_diag = h_GP * gp_eps_iso;

    atomicAdd(&stats[GP_ELASTIC_STATS_SUM_GEL], gel);
    atomicMaxAbsDouble(&stats[GP_ELASTIC_STATS_MAX_GEL], fabs(gel));
    atomicMinDouble(&stats[GP_ELASTIC_STATS_MIN_SIGMA_HYDRO], sigma_hydro);
    atomicMaxDouble(&stats[GP_ELASTIC_STATS_MAX_SIGMA_HYDRO], sigma_hydro);
    atomicMinDouble(&stats[GP_ELASTIC_STATS_MIN_EPS0_GP_DIAG], eps0_gp_diag);
    atomicMaxDouble(&stats[GP_ELASTIC_STATS_MAX_EPS0_GP_DIAG], eps0_gp_diag);
}

// ============================================================================
// 6. Y方程：计算梯度（在k空间）
// ============================================================================

// Optimization: 单分量梯度 kernel，用于 divJ 按方向串行累加，仅计算 grad_alpha = i*k_alpha*f_k
__global__ void compute_gradient_single_component_k_kernel(
    const cuDoubleComplex *f_k,
    cuDoubleComplex *grad_alpha_k,
    int alpha,  // 0=x, 1=y, 2=z
    int Nx, int Ny, int Nz, int NzC,
    double dx, double dy, double dz,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    int k = idx % NzC;
    int remainder = idx / NzC;
    int j = remainder % Ny;
    int i = remainder / Ny;

    double kx = kx_wrap(i, Nx, dx);
    double ky = ky_wrap(j, Ny, dy);
    double kz = kz_wrap(k, Nz, dz);

    double k_alpha = (alpha == 0) ? kx : ((alpha == 1) ? ky : kz);
    cuDoubleComplex f = f_k[idx];
    cuDoubleComplex I_k = make_cuDoubleComplex(0.0, 1.0);
    grad_alpha_k[idx] = cuCmul(I_k, cuCmul(make_cuDoubleComplex(k_alpha, 0.0), f));
}

__global__ void compute_gradient_k_kernel(
    const cuDoubleComplex *f_k,
    cuDoubleComplex *grad_x_k,
    cuDoubleComplex *grad_y_k,
    cuDoubleComplex *grad_z_k,
    int Nx, int Ny, int Nz, int NzC,
    double dx, double dy, double dz,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    // 计算三维索引
    int k = idx % NzC;
    int remainder = idx / NzC;
    int j = remainder % Ny;
    int i = remainder / Ny;
    
    double kx = kx_wrap(i, Nx, dx);
    double ky = ky_wrap(j, Ny, dy);
    double kz = kz_wrap(k, Nz, dz);
    
    cuDoubleComplex f = f_k[idx];
    cuDoubleComplex I_k = make_cuDoubleComplex(0.0, 1.0);
    
    grad_x_k[idx] = cuCmul(I_k, cuCmul(make_cuDoubleComplex(kx, 0.0), f));
    grad_y_k[idx] = cuCmul(I_k, cuCmul(make_cuDoubleComplex(ky, 0.0), f));
    grad_z_k[idx] = cuCmul(I_k, cuCmul(make_cuDoubleComplex(kz, 0.0), f));
}

// ============================================================================
// 7. Y方程：计算通量（在实空间）
// ============================================================================

__global__ void compute_flux_kernel(
    const double *grad_mu_x_r,
    const double *grad_mu_y_r,
    const double *grad_mu_z_r,
    const double *phi_r,
    const double *xB_prev_r,
    double *Jx_r,
    double *Jy_r,
    double *Jz_r,
    double D_alpha,
    double D_compound,
    double Vm_alpha_0,
    double dVm_alpha_dxB,
    double Vm_compound,
    double temperature_K,
    double mu_reference_scale,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    double phi = phi_r[idx];
    double h = h_of_phi(phi);
    double xB0 = xB_prev_r[idx];
    
    // 计算混合扩散系数和热力学因子
    double Dm = D_mix(h, D_alpha, D_compound);
    double G = gamma_thermo_nonlinear(xB0, h, Vm_alpha_0, dVm_alpha_dxB, 
                                      Vm_compound, temperature_K, mu_reference_scale);
    
    double Meff = stabilized_meff(Dm, G);
    
    Jx_r[idx] = Meff * grad_mu_x_r[idx];
    Jy_r[idx] = Meff * grad_mu_y_r[idx];
    Jz_r[idx] = Meff * grad_mu_z_r[idx];
}

// Optimization: 单分量通量 kernel，用于 divJ 按方向串行累加，仅计算 J_alpha = Meff * grad_mu_alpha
__global__ void compute_flux_single_component_kernel(
    const double *grad_mu_alpha_r,
    const double *phi_r,
    const double *xB_prev_r,
    double *J_alpha_r,
    double D_alpha,
    double D_compound,
    double Vm_alpha_0,
    double dVm_alpha_dxB,
    double Vm_compound,
    double temperature_K,
    double mu_reference_scale,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double phi = phi_r[idx];
    double h = h_of_phi(phi);
    double xB0 = xB_prev_r[idx];

    double Dm = D_mix(h, D_alpha, D_compound);
    double G = gamma_thermo_nonlinear(xB0, h, Vm_alpha_0, dVm_alpha_dxB,
                                      Vm_compound, temperature_K, mu_reference_scale);
    double Meff = stabilized_meff(Dm, G);
    
    J_alpha_r[idx] = Meff * grad_mu_alpha_r[idx];
}

__global__ void compute_flux_single_component_q_kernel(
    const double *grad_mu_alpha_r,
    const double *phi_r,
    const double *xB_prev_r,
    double *J_alpha_r,
    double D_alpha,
    double Vm_alpha_0,
    double dVm_alpha_dxB,
    double Vm_compound,
    double temperature_K,
    double mu_reference_scale,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    const double h = h_of_phi(clamp01(phi_r[idx]));
    const double alpha = 1.0 - h;
    const double xB0 = xB_prev_r[idx];
    const double G_alpha = gamma_thermo_nonlinear(
        xB0, 0.0, Vm_alpha_0, dVm_alpha_dxB,
        Vm_compound, temperature_K, mu_reference_scale);
    const double M_alpha = stabilized_meff(D_alpha, G_alpha);
    J_alpha_r[idx] = alpha * M_alpha * grad_mu_alpha_r[idx];
}

__global__ void compute_flux_single_component_gp_kernel(
    const double *grad_mu_alpha_r,
    const double *phi_r,
    const double *eta_r,
    const double *xB_alpha_prev_r,
    double *J_alpha_r,
    double D_alpha,
    double gp_M_GP,
    double gp_M_beta,
    double Vm_alpha_0,
    double dVm_alpha_dxB,
    double Vm_compound,
    double temperature_K,
    double mu_reference_scale,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double h_alpha = 0.0;
    double h_GP = 0.0;
    double h_beta = 0.0;
    phase_fractions_gp(phi_r[idx], eta_r[idx], &h_alpha, &h_GP, &h_beta);

    double xB0 = xB_alpha_prev_r[idx];
    double G_alpha = gamma_thermo_nonlinear(xB0, 0.0, Vm_alpha_0, dVm_alpha_dxB,
                                            Vm_compound, temperature_K, mu_reference_scale);
    double M_alpha = stabilized_meff(D_alpha, G_alpha);
    double M_eff = h_alpha * M_alpha + h_GP * gp_M_GP + h_beta * gp_M_beta;

    J_alpha_r[idx] = M_eff * grad_mu_alpha_r[idx];
}

__global__ void compute_gp_transport_stats_kernel(
    const double *phi_r,
    const double *eta_r,
    const double *xB_alpha_r,
    double xB_ref,
    double D_alpha,
    double gp_M_GP,
    double gp_M_beta,
    double Vm_alpha_0,
    double dVm_alpha_dxB,
    double Vm_compound,
    double temperature_K,
    double mu_reference_scale,
    double *stats,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double h_alpha = 0.0;
    double h_GP = 0.0;
    double h_beta = 0.0;
    phase_fractions_gp(phi_r[idx], eta_r[idx], &h_alpha, &h_GP, &h_beta);

    double xB = xB_alpha_r[idx];
    double G_alpha = gamma_thermo_nonlinear(xB, 0.0, Vm_alpha_0, dVm_alpha_dxB,
                                            Vm_compound, temperature_K, mu_reference_scale);
    double M_alpha = stabilized_meff(D_alpha, G_alpha);
    double M_eff = h_alpha * M_alpha + h_GP * gp_M_GP + h_beta * gp_M_beta;

    atomicMaxAbsDouble(&stats[GP_TRANSPORT_MAXABS_XB_PERTURB], fabs(xB - xB_ref));
    atomicMinDouble(&stats[GP_TRANSPORT_MIN_HALPHA], h_alpha);
    atomicMaxDouble(&stats[GP_TRANSPORT_MAX_HALPHA], h_alpha);
    atomicMinDouble(&stats[GP_TRANSPORT_MIN_M_ALPHA], M_alpha);
    atomicMaxDouble(&stats[GP_TRANSPORT_MAX_M_ALPHA], M_alpha);
    atomicMinDouble(&stats[GP_TRANSPORT_MIN_M_EFF], M_eff);
    atomicMaxDouble(&stats[GP_TRANSPORT_MAX_M_EFF], M_eff);
}

// Optimization: divJ 累加 kernel，divJ_k += i*k_alpha*J_alpha_k
__global__ void divJ_accumulate_kernel(
    const cuDoubleComplex *J_alpha_k,
    cuDoubleComplex *divJ_k,
    int alpha,
    int Nx, int Ny, int Nz, int NzC,
    double dx, double dy, double dz,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    int k = idx % NzC;
    int remainder = idx / NzC;
    int j = remainder % Ny;
    int i = remainder / Ny;

    double kx = kx_wrap(i, Nx, dx);
    double ky = ky_wrap(j, Ny, dy);
    double kz = kz_wrap(k, Nz, dz);

    double k_alpha = (alpha == 0) ? kx : ((alpha == 1) ? ky : kz);
    cuDoubleComplex I_k = make_cuDoubleComplex(0.0, 1.0);
    cuDoubleComplex term = cuCmul(I_k, cuCmul(make_cuDoubleComplex(k_alpha, 0.0), J_alpha_k[idx]));
    divJ_k[idx] = cuCadd(divJ_k[idx], term);
}

// Optimization: 将 divJ_k 清零
__global__ void zero_divJ_k_kernel(cuDoubleComplex *divJ_k, int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    divJ_k[idx] = make_cuDoubleComplex(0.0, 0.0);
}

// ============================================================================
// 8. Y方程：计算散度（在k空间）
// ============================================================================

__global__ void compute_divergence_k_kernel(
    const cuDoubleComplex *Jx_k,
    const cuDoubleComplex *Jy_k,
    const cuDoubleComplex *Jz_k,
    cuDoubleComplex *divJ_k,
    int Nx, int Ny, int Nz, int NzC,
    double dx, double dy, double dz,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    // 计算三维索引
    int k = idx % NzC;
    int remainder = idx / NzC;
    int j = remainder % Ny;
    int i = remainder / Ny;
    
    double kx = kx_wrap(i, Nx, dx);
    double ky = ky_wrap(j, Ny, dy);
    double kz = kz_wrap(k, Nz, dz);
    
    cuDoubleComplex I_k = make_cuDoubleComplex(0.0, 1.0);
    
    cuDoubleComplex term_x = cuCmul(I_k, cuCmul(make_cuDoubleComplex(kx, 0.0), Jx_k[idx]));
    cuDoubleComplex term_y = cuCmul(I_k, cuCmul(make_cuDoubleComplex(ky, 0.0), Jy_k[idx]));
    cuDoubleComplex term_z = cuCmul(I_k, cuCmul(make_cuDoubleComplex(kz, 0.0), Jz_k[idx]));
    
    divJ_k[idx] = cuCadd(cuCadd(term_x, term_y), term_z);
}

// ============================================================================
// 9. Y方程：计算RHS（在实空间）
// ============================================================================

__global__ void compute_Y_rhs_kernel(
    const double *divJ_r,
    const double *phi_r,
    const double *phi_prev,
    const double *lapY_r,
    const double *Y_r,
    const double *dY_dt_prev,
    double *rhs_r,
    double dt,
    double v_B,
    double mean_DY,
    int total_size,
    int use_previous_time_level,
    int disable_gamma_term,
    double term_h_scale,
    double *diag_stats)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    double phi_new = phi_r[idx];
    double phi_old = phi_prev[idx];
    double h_new = h_of_phi(phi_new);
    double h_old = h_of_phi(phi_old);
    double dh_dt = (h_new - h_old) / dt;
    
    // 从 Y 稳定地计算 xB 与 logistic 导数，避免直接 exp(±Y) 带来的溢出
    double Y_val = Y_r[idx];
    double xB;
    if (Y_val >= 0.0) {
        double e_negY = exp(-Y_val);
        xB = 1.0 / (1.0 + e_negY);
    } else {
        double e_Y = exp(Y_val);
        xB = e_Y / (1.0 + e_Y);
    }
    if (xB < 1e-12) xB = 1e-12;
    if (xB > 1.0 - 1e-12) xB = 1.0 - 1e-12;
    double logistic_deriv = xB * (1.0 - xB);
    double term_h = term_h_scale * dh_dt * (v_B - xB);

    double h_for_explicit = use_previous_time_level ? h_old : h_new;
    double gamma_local = ((1.0 - h_for_explicit) * logistic_deriv - 1.0);
    double dY_dt = 0.0;
    if (dY_dt_prev) {
        dY_dt = dY_dt_prev[idx];
    }
    double term_gamma = disable_gamma_term ? 0.0 : (gamma_local * dY_dt);
    
    double lapY = lapY_r[idx];
    double term_lap = mean_DY * lapY;
    
    double S_term = -term_h - term_lap - term_gamma;
    rhs_r[idx] = divJ_r[idx] + S_term;

    if (diag_stats) {
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_H_OLD], h_old);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_H_NEW], h_new);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_GAMMA_LOCAL], gamma_local);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_TERM_H], term_h);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_TERM_LAP], term_lap);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_TERM_GAMMA], term_gamma);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_DIVJ], divJ_r[idx]);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_LAPY], lapY);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_TOTAL], rhs_r[idx]);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMABS_DIVJ], fabs(divJ_r[idx]));
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMABS_TERM_H], fabs(term_h));
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMABS_TERM_LAP], fabs(term_lap));
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMABS_TERM_GAMMA], fabs(term_gamma));
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMABS_TOTAL], fabs(rhs_r[idx]));
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMSQ_DIVJ], divJ_r[idx] * divJ_r[idx]);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMSQ_TERM_H], term_h * term_h);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMSQ_TERM_LAP], term_lap * term_lap);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMSQ_TERM_GAMMA], term_gamma * term_gamma);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMSQ_TOTAL], rhs_r[idx] * rhs_r[idx]);
        atomicMaxAbsDouble(&diag_stats[MASS_DIAG_Y_RHS_MAXABS_DIVJ], fabs(divJ_r[idx]));
        atomicMaxAbsDouble(&diag_stats[MASS_DIAG_Y_RHS_MAXABS_TERM_H], fabs(term_h));
        atomicMaxAbsDouble(&diag_stats[MASS_DIAG_Y_RHS_MAXABS_TERM_LAP], fabs(term_lap));
        atomicMaxAbsDouble(&diag_stats[MASS_DIAG_Y_RHS_MAXABS_TERM_GAMMA], fabs(term_gamma));
        atomicMaxAbsDouble(&diag_stats[MASS_DIAG_Y_RHS_MAXABS_TOTAL], fabs(rhs_r[idx]));
    }
}

__global__ void compute_Y_rhs_gp_kernel(
    const double *divJ_r,
    const double *phi_r,
    const double *phi_prev,
    const double *eta_r,
    const double *eta_prev_r,
    const double *lapY_r,
    const double *Y_r,
    const double *dY_dt_prev,
    double *rhs_r,
    double dt,
    double xB_GP,
    double mean_DY,
    double gp_h_alpha_eps,
    int total_size,
    int use_previous_time_level,
    int disable_gamma_term,
    double *diag_stats)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double h_alpha_new = 0.0;
    double h_GP_new = 0.0;
    double h_beta_new = 0.0;
    phase_fractions_gp(phi_r[idx], eta_r[idx], &h_alpha_new, &h_GP_new, &h_beta_new);

    double h_alpha_old = 0.0;
    double h_GP_old = 0.0;
    double h_beta_old = 0.0;
    phase_fractions_gp(phi_prev[idx], eta_prev_r[idx], &h_alpha_old, &h_GP_old, &h_beta_old);

    double dh_GP_dt = (h_GP_new - h_GP_old) / dt;
    double dh_beta_dt = (h_beta_new - h_beta_old) / dt;

    double xB = sigmoid_from_logit(Y_r[idx], 20.0, 1e-12);
    double logistic_deriv = xB * (1.0 - xB);
    double storage_source = dh_GP_dt * (xB_GP - xB) + dh_beta_dt * (1.0 - xB);

    double h_alpha_for_explicit = use_previous_time_level ? h_alpha_old : h_alpha_new;
    double h_alpha_eff = fmax(h_alpha_for_explicit, gp_h_alpha_eps);
    double gamma_local = (h_alpha_eff * logistic_deriv - 1.0);
    double dY_dt = 0.0;
    if (dY_dt_prev) {
        dY_dt = dY_dt_prev[idx];
    }
    double term_gamma = disable_gamma_term ? 0.0 : (gamma_local * dY_dt);
    double lapY = lapY_r[idx];
    double term_lap = mean_DY * lapY;

    rhs_r[idx] = divJ_r[idx] - storage_source - term_lap - term_gamma;

    if (diag_stats) {
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_H_OLD], h_alpha_old);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_H_NEW], h_alpha_new);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_GAMMA_LOCAL], gamma_local);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_TERM_H], storage_source);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_TERM_LAP], term_lap);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_TERM_GAMMA], term_gamma);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_DIVJ], divJ_r[idx]);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_LAPY], lapY);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_TOTAL], rhs_r[idx]);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMABS_DIVJ], fabs(divJ_r[idx]));
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMABS_TERM_H], fabs(storage_source));
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMABS_TERM_LAP], fabs(term_lap));
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMABS_TERM_GAMMA], fabs(term_gamma));
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMABS_TOTAL], fabs(rhs_r[idx]));
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMSQ_DIVJ], divJ_r[idx] * divJ_r[idx]);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMSQ_TERM_H], storage_source * storage_source);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMSQ_TERM_LAP], term_lap * term_lap);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMSQ_TERM_GAMMA], term_gamma * term_gamma);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMSQ_TOTAL], rhs_r[idx] * rhs_r[idx]);
        atomicMaxAbsDouble(&diag_stats[MASS_DIAG_Y_RHS_MAXABS_DIVJ], fabs(divJ_r[idx]));
        atomicMaxAbsDouble(&diag_stats[MASS_DIAG_Y_RHS_MAXABS_TERM_H], fabs(storage_source));
        atomicMaxAbsDouble(&diag_stats[MASS_DIAG_Y_RHS_MAXABS_TERM_LAP], fabs(term_lap));
        atomicMaxAbsDouble(&diag_stats[MASS_DIAG_Y_RHS_MAXABS_TERM_GAMMA], fabs(term_gamma));
        atomicMaxAbsDouble(&diag_stats[MASS_DIAG_Y_RHS_MAXABS_TOTAL], fabs(rhs_r[idx]));
    }
}

__global__ void compute_Y_rhs_gp_conservative_kernel(
    const double *divJ_r,
    const double *phi_r,
    const double *phi_prev,
    const double *eta_r,
    const double *eta_prev_r,
    const double *lapY_r,
    const double *Y_r,
    const double *dY_dt_prev,
    double *rhs_r,
    double dt,
    double xB_GP,
    double mean_DY,
    int total_size,
    int use_previous_time_level,
    int disable_gamma_term,
    double *diag_stats)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double h_alpha_new = 0.0;
    double h_GP_new = 0.0;
    double h_beta_new = 0.0;
    phase_fractions_gp(phi_r[idx], eta_r[idx], &h_alpha_new, &h_GP_new, &h_beta_new);

    double h_alpha_old = 0.0;
    double h_GP_old = 0.0;
    double h_beta_old = 0.0;
    phase_fractions_gp(phi_prev[idx], eta_prev_r[idx], &h_alpha_old, &h_GP_old, &h_beta_old);

    double dh_GP_dt = (h_GP_new - h_GP_old) / dt;
    double dh_beta_dt = (h_beta_new - h_beta_old) / dt;

    double xB = sigmoid_from_logit(Y_r[idx], 20.0, 1.0e-12);
    double q = xB * (1.0 - xB);
    double storage_source = dh_GP_dt * (xB_GP - xB) + dh_beta_dt * (1.0 - xB);

    const double h_alpha_explicit = use_previous_time_level ? h_alpha_old : h_alpha_new;
    const double hq = h_alpha_explicit * q;
    const double gamma_local = hq - 1.0;
    double lagged_dYdt = 0.0;
    if (dY_dt_prev) lagged_dYdt = dY_dt_prev[idx];
    const double term_gamma = disable_gamma_term ? 0.0 : (gamma_local * lagged_dYdt);
    const double lapY = lapY_r[idx];
    const double term_lap = mean_DY * lapY;
    const double fY = divJ_r[idx] - storage_source - term_lap - term_gamma;

    rhs_r[idx] = fY;

    if (diag_stats) {
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_H_OLD], h_alpha_old);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_H_NEW], h_alpha_new);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_HQ], hq);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_GAMMA_LOCAL], gamma_local);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_TERM_H], storage_source);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_TERM_LAP], term_lap);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_TERM_GAMMA], term_gamma);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_DIVJ], divJ_r[idx]);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_LAPY], lapY);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUM_TOTAL], fY);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMABS_DIVJ], fabs(divJ_r[idx]));
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMABS_TERM_H], fabs(storage_source));
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMABS_TERM_LAP], fabs(term_lap));
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMABS_TERM_GAMMA], fabs(term_gamma));
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMABS_TOTAL], fabs(fY));
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMSQ_DIVJ], divJ_r[idx] * divJ_r[idx]);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMSQ_TERM_H], storage_source * storage_source);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMSQ_TERM_LAP], term_lap * term_lap);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMSQ_TERM_GAMMA], term_gamma * term_gamma);
        atomicAdd(&diag_stats[MASS_DIAG_Y_RHS_SUMSQ_TOTAL], fY * fY);
        atomicMinDouble(&diag_stats[MASS_DIAG_Y_RHS_MIN_H], h_alpha_explicit);
        atomicMinDouble(&diag_stats[MASS_DIAG_Y_RHS_MIN_HQ], hq);
        atomicMaxDouble(&diag_stats[MASS_DIAG_Y_RHS_MAX_HQ], hq);
        atomicMaxAbsDouble(&diag_stats[MASS_DIAG_Y_RHS_MAXABS_DIVJ], fabs(divJ_r[idx]));
        atomicMaxAbsDouble(&diag_stats[MASS_DIAG_Y_RHS_MAXABS_TERM_H], fabs(storage_source));
        atomicMaxAbsDouble(&diag_stats[MASS_DIAG_Y_RHS_MAXABS_TERM_LAP], fabs(term_lap));
        atomicMaxAbsDouble(&diag_stats[MASS_DIAG_Y_RHS_MAXABS_TERM_GAMMA], fabs(term_gamma));
        atomicMaxAbsDouble(&diag_stats[MASS_DIAG_Y_RHS_MAXABS_TOTAL], fabs(fY));
        atomicMaxAbsDouble(&diag_stats[MASS_DIAG_Y_RHS_MAXABS_LAGGED_DYDT], fabs(lagged_dYdt));
    }
}

// ============================================================================
// 10. Y方程：半隐式更新（在k空间）
// ============================================================================

__global__ void Y_semi_implicit_update_kernel(
    const cuDoubleComplex *Y_k_old,
    const cuDoubleComplex *rhs_k,
    const double *k2,
    cuDoubleComplex *Y_k_new,
    double mean_DY,
    double dt,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    double denom = 1.0 + dt * mean_DY * k2[idx];
    
    cuDoubleComplex Y_old = Y_k_old[idx];
    cuDoubleComplex rhs = rhs_k[idx];
    
    // Y_k_new = (Y_k_old + dt * rhs_k) / denom
    cuDoubleComplex numerator = cuCadd(Y_old, cuCmul(make_cuDoubleComplex(dt, 0.0), rhs));
    Y_k_new[idx] = cuCdiv(numerator, make_cuDoubleComplex(denom, 0.0));
}

// ============================================================================
// 11. Y归一化和截断kernel
// ============================================================================

__global__ void Y_normalize_and_clamp_kernel(
    double *Y_r,
    double *xB_r,
    double invN,
    double Y_clip,
    double Y_upper_cap,
    double xB_eps,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    double Y_new = Y_r[idx] * invN;
    
    if (Y_new > Y_upper_cap) Y_new = Y_upper_cap;
    if (Y_new < -Y_clip) Y_new = -Y_clip;
    
    Y_r[idx] = Y_new;
    xB_r[idx] = sigmoid_from_logit(Y_new, Y_clip, xB_eps);
}

__global__ void phi_mass_diagnostics_kernel(
    const double *phi_ifft_r,
    const double *phi_before_r,
    const double *xB_before_r,
    double *stats,
    double invN,
    double v_B,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double phi_before = phi_before_r[idx];
    double xB_before = xB_before_r[idx];
    double h_before = h_of_phi(phi_before);
    double phi_raw = phi_ifft_r[idx] * invN;
    double phi_after = phi_raw;
    double clip_low = 0.0;
    double clip_high = 0.0;
    if (phi_after < -1.0e-6) {
        phi_after = -1.0e-6;
        clip_low = 1.0;
    }
    if (phi_after > 1.0 + 1.0e-6) {
        phi_after = 1.0 + 1.0e-6;
        clip_high = 1.0;
    }
    double h_raw = h_of_phi(phi_raw);
    double h_after = h_of_phi(phi_after);
    double xBtot_raw = (1.0 - h_raw) * xB_before + v_B * h_raw;
    double xBtot_after = (1.0 - h_after) * xB_before + v_B * h_after;
    double pred = (h_after - h_before) * (v_B - xB_before);

    atomicAdd(&stats[MASS_DIAG_PHI_SUM_XBTOT_RAW], xBtot_raw);
    atomicAdd(&stats[MASS_DIAG_PHI_SUM_XBTOT_CLAMPED], xBtot_after);
    atomicAdd(&stats[MASS_DIAG_PHI_SUM_H_BEFORE], h_before);
    atomicAdd(&stats[MASS_DIAG_PHI_SUM_H_AFTER], h_after);
    atomicAdd(&stats[MASS_DIAG_PHI_SUM_XB_BEFORE], xB_before);
    atomicAdd(&stats[MASS_DIAG_PHI_SUM_PHI_BEFORE], phi_before);
    atomicAdd(&stats[MASS_DIAG_PHI_SUM_PHI_AFTER], phi_after);
    atomicAdd(&stats[MASS_DIAG_PHI_SUM_PRED_DELTA_XBTOT], pred);
    atomicAdd(&stats[MASS_DIAG_PHI_CLIP_LOW_COUNT], clip_low);
    atomicAdd(&stats[MASS_DIAG_PHI_CLIP_HIGH_COUNT], clip_high);
}

__global__ void Y_mass_diagnostics_kernel(
    const double *Y_ifft_r,
    const double *phi_r,
    const double *xB_before_r,
    double *stats,
    double invN,
    double Y_clip,
    double Y_upper_cap,
    double xB_eps,
    double v_B,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double phi = phi_r[idx];
    double h = h_of_phi(phi);
    double xB_before = xB_before_r[idx];
    // Y_ifft_r is already normalized back to real-space before this diagnostic kernel.
    // Do not apply invN again here, otherwise xB_raw collapses toward 0.5 and
    // all Y-update mass diagnostics become meaningless.
    double Y_raw = Y_ifft_r[idx];
    double xB_raw = sigmoid_from_logit_unclipped_local(Y_raw);

    double y_low = 0.0;
    double y_high = 0.0;
    double Y_clamped = Y_raw;
    if (Y_clamped > Y_upper_cap) {
        Y_clamped = Y_upper_cap;
        y_high = 1.0;
    }
    if (Y_clamped < -Y_clip) {
        Y_clamped = -Y_clip;
        y_low = 1.0;
    }

    double xB_preclip = sigmoid_from_logit_unclipped_local(Y_clamped);
    double xb_low = 0.0;
    double xb_high = 0.0;
    double xB_final = xB_preclip;
    if (xB_final < xB_eps) {
        xB_final = xB_eps;
        xb_low = 1.0;
    }
    if (xB_final > 1.0 - xB_eps) {
        xB_final = 1.0 - xB_eps;
        xb_high = 1.0;
    }

    atomicAdd(&stats[MASS_DIAG_Y_SUM_XBTOT_RAWY], (1.0 - h) * xB_raw + v_B * h);
    atomicAdd(&stats[MASS_DIAG_Y_SUM_XBTOT_YCLAMP_PRE_XBCLIP], (1.0 - h) * xB_preclip + v_B * h);
    atomicAdd(&stats[MASS_DIAG_Y_SUM_XBTOT_FINAL], (1.0 - h) * xB_final + v_B * h);
    atomicAdd(&stats[MASS_DIAG_Y_SUM_XB_BEFORE], xB_before);
    atomicAdd(&stats[MASS_DIAG_Y_SUM_XB_RAWY], xB_raw);
    atomicAdd(&stats[MASS_DIAG_Y_SUM_XB_YCLAMP_PRE_XBCLIP], xB_preclip);
    atomicAdd(&stats[MASS_DIAG_Y_SUM_XB_FINAL], xB_final);
    atomicAdd(&stats[MASS_DIAG_Y_CLIP_LOW_COUNT], y_low);
    atomicAdd(&stats[MASS_DIAG_Y_CLIP_HIGH_COUNT], y_high);
    atomicAdd(&stats[MASS_DIAG_Y_XB_CLIP_LOW_COUNT], xb_low);
    atomicAdd(&stats[MASS_DIAG_Y_XB_CLIP_HIGH_COUNT], xb_high);
}

__global__ void Y_mass_diagnostics_gp_kernel(
    const double *Y_ifft_r,
    const double *phi_r,
    const double *eta_r,
    const double *xB_before_r,
    double *stats,
    double invN,
    double Y_clip,
    double Y_upper_cap,
    double xB_eps,
    double xB_GP,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double h_alpha = 0.0;
    double h_GP = 0.0;
    double h_beta = 0.0;
    phase_fractions_gp(phi_r[idx], eta_r[idx], &h_alpha, &h_GP, &h_beta);

    double xB_before = xB_before_r[idx];
    (void)invN;
    double Y_raw = Y_ifft_r[idx];
    double xB_raw = sigmoid_from_logit_unclipped_local(Y_raw);

    double y_low = 0.0;
    double y_high = 0.0;
    double Y_clamped = Y_raw;
    if (Y_clamped > Y_upper_cap) {
        Y_clamped = Y_upper_cap;
        y_high = 1.0;
    }
    if (Y_clamped < -Y_clip) {
        Y_clamped = -Y_clip;
        y_low = 1.0;
    }

    double xB_preclip = sigmoid_from_logit_unclipped_local(Y_clamped);
    double xb_low = 0.0;
    double xb_high = 0.0;
    double xB_final = xB_preclip;
    if (xB_final < xB_eps) {
        xB_final = xB_eps;
        xb_low = 1.0;
    }
    if (xB_final > 1.0 - xB_eps) {
        xB_final = 1.0 - xB_eps;
        xb_high = 1.0;
    }

    atomicAdd(&stats[MASS_DIAG_Y_SUM_XBTOT_RAWY], h_alpha * xB_raw + h_GP * xB_GP + h_beta);
    atomicAdd(&stats[MASS_DIAG_Y_SUM_XBTOT_YCLAMP_PRE_XBCLIP], h_alpha * xB_preclip + h_GP * xB_GP + h_beta);
    atomicAdd(&stats[MASS_DIAG_Y_SUM_XBTOT_FINAL], h_alpha * xB_final + h_GP * xB_GP + h_beta);
    atomicAdd(&stats[MASS_DIAG_Y_SUM_XB_BEFORE], xB_before);
    atomicAdd(&stats[MASS_DIAG_Y_SUM_XB_RAWY], xB_raw);
    atomicAdd(&stats[MASS_DIAG_Y_SUM_XB_YCLAMP_PRE_XBCLIP], xB_preclip);
    atomicAdd(&stats[MASS_DIAG_Y_SUM_XB_FINAL], xB_final);
    atomicAdd(&stats[MASS_DIAG_Y_CLIP_LOW_COUNT], y_low);
    atomicAdd(&stats[MASS_DIAG_Y_CLIP_HIGH_COUNT], y_high);
    atomicAdd(&stats[MASS_DIAG_Y_XB_CLIP_LOW_COUNT], xb_low);
    atomicAdd(&stats[MASS_DIAG_Y_XB_CLIP_HIGH_COUNT], xb_high);
}

__global__ void gp_storage_exact_Y_update_kernel(
    const double *divJ_r,
    const double *phi_new_r,
    const double *phi_old_r,
    const double *eta_new_r,
    const double *eta_old_r,
    const double *Y_old_r,
    double *Y_r,
    double *xB_r,
    double dt,
    double xB_GP,
    double Y_clip,
    double Y_upper_cap,
    double xB_eps,
    double gp_h_alpha_eps,
    int disable_internal_clip,
    double *update_stats,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double h_alpha_old = 0.0, h_GP_old = 0.0, h_beta_old = 0.0;
    double h_alpha_new = 0.0, h_GP_new = 0.0, h_beta_new = 0.0;
    phase_fractions_gp(phi_old_r[idx], eta_old_r[idx], &h_alpha_old, &h_GP_old, &h_beta_old);
    phase_fractions_gp(phi_new_r[idx], eta_new_r[idx], &h_alpha_new, &h_GP_new, &h_beta_new);

    double xB_old = sigmoid_from_logit(Y_old_r[idx], Y_clip, xB_eps);
    double S_old = h_GP_old * (xB_GP - xB_old) + h_beta_old * (1.0 - xB_old);
    double xBtot_old = xB_old + S_old;
    double xBtot_target = xBtot_old + dt * divJ_r[idx];
    if (!isfinite(xBtot_target)) {
        Y_r[idx] = Y_old_r[idx];
        xB_r[idx] = xB_old;
        if (update_stats) {
            atomicAdd(&update_stats[GP_Y_UPDATE_SMALL_HALPHA_COUNT], 1.0);
        }
        return;
    }

    double denom = h_alpha_new;
    const int small_h_alpha = (denom < gp_h_alpha_eps) ? 1 : 0;
    double xB_unclipped = xB_old;
    if (!small_h_alpha) {
        double numerator = xBtot_target - h_GP_new * xB_GP - h_beta_new;
        xB_unclipped = numerator / denom;
    }
    if (!isfinite(xB_unclipped)) xB_unclipped = xB_old;
    double xBtot_after_unclipped = h_alpha_new * xB_unclipped + h_GP_new * xB_GP + h_beta_new;
    double small_h_alpha_mass_error = xBtot_after_unclipped - xBtot_target;

    double xb_low = 0.0;
    double xb_high = 0.0;
    double y_low = 0.0;
    double y_high = 0.0;
    double xB_final = xB_unclipped;
    double Y_final = Y_old_r[idx];
    int bypass_clip = 0;
    if (disable_internal_clip &&
        isfinite(xB_unclipped) &&
        xB_unclipped >= xB_eps &&
        xB_unclipped <= 1.0 - xB_eps) {
        double Y_no_clip = log(xB_unclipped / (1.0 - xB_unclipped));
        if (isfinite(Y_no_clip)) {
            xB_final = xB_unclipped;
            Y_final = Y_no_clip;
            bypass_clip = 1;
        }
    }
    if (!bypass_clip) {
        if (xB_final < xB_eps) {
            xB_final = xB_eps;
            xb_low = 1.0;
        }
        if (xB_final > 1.0 - xB_eps) {
            xB_final = 1.0 - xB_eps;
            xb_high = 1.0;
        }

        double Y_unclipped = logit_from_fraction(xB_final, xB_eps, Y_clip);
        Y_final = Y_unclipped;
        if (Y_final > Y_upper_cap) {
            Y_final = Y_upper_cap;
            y_high = 1.0;
        }
        if (Y_final < -Y_clip) {
            Y_final = -Y_clip;
            y_low = 1.0;
        }
        xB_final = sigmoid_from_logit(Y_final, Y_clip, xB_eps);
    }
    double xBtot_after_clipped = h_alpha_new * xB_final + h_GP_new * xB_GP + h_beta_new;
    double clipping_mass_error = xBtot_after_clipped - xBtot_after_unclipped;

    Y_r[idx] = Y_final;
    xB_r[idx] = xB_final;

    if (update_stats) {
        atomicAdd(&update_stats[GP_Y_UPDATE_SUM_CLIPPING_MASS_ERROR], clipping_mass_error);
        atomicAdd(&update_stats[GP_Y_UPDATE_SUM_SMALL_HALPHA_MASS_ERROR], small_h_alpha_mass_error);
        atomicAdd(&update_stats[GP_Y_UPDATE_Y_CLIP_COUNT], y_low + y_high);
        atomicAdd(&update_stats[GP_Y_UPDATE_XB_CLIP_COUNT], xb_low + xb_high);
        if (small_h_alpha) {
            atomicAdd(&update_stats[GP_Y_UPDATE_SMALL_HALPHA_COUNT], 1.0);
        }
    }
}

// Same-step conservative update for the two-phase storage
// C=(1-h_beta)*xB_alpha+h_beta*v_B.  This avoids using a derivative from the
// previous physical timestep as an algebraic Jacobian iterate.  Local bound or
// small-storage residuals remain explicit in update_stats and are closed by the
// existing global mass projection; they are never silently discarded.
__global__ void two_phase_storage_exact_Y_update_kernel(
    const double *divJ_r,
    const double *phi_new_r,
    const double *phi_old_r,
    const double *Y_old_r,
    double *Y_r,
    double *xB_r,
    double dt,
    double v_B,
    double Y_clip,
    double Y_upper_cap,
    double xB_eps,
    double h_alpha_eps,
    double *update_stats,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    const double h_beta_old = h_of_phi(clamp01(phi_old_r[idx]));
    const double h_beta_new = h_of_phi(clamp01(phi_new_r[idx]));
    const double h_alpha_old = 1.0 - h_beta_old;
    const double h_alpha_new = 1.0 - h_beta_new;
    const double xB_old = sigmoid_from_logit(Y_old_r[idx], Y_clip, xB_eps);
    const double C_old = h_alpha_old * xB_old + h_beta_old * v_B;
    const double C_target = C_old + dt * divJ_r[idx];

    double xB_trial = xB_old;
    const int small_h_alpha = !(h_alpha_new >= h_alpha_eps);
    if (!small_h_alpha && isfinite(C_target)) {
        xB_trial = (C_target - h_beta_new * v_B) / h_alpha_new;
    }
    const int infeasible_reconstruction =
        small_h_alpha || !isfinite(xB_trial) ||
        xB_trial < xB_eps || xB_trial > 1.0 - xB_eps;
    // x_alpha is an unobservable continuation when matrix storage vanishes.
    // Likewise, an out-of-bounds trial means the split phi step requested more
    // local storage than exists. Preserve the previous bounded continuation and
    // expose the residual to the conservative global projection.
    double xB_final = infeasible_reconstruction ? xB_old : xB_trial;
    double Y_final = logit_from_fraction(xB_final, xB_eps, Y_clip);
    Y_final = fmin(fmax(Y_final, -Y_clip), Y_upper_cap);
    xB_final = sigmoid_from_logit(Y_final, Y_clip, xB_eps);
    const double C_final = h_alpha_new * xB_final + h_beta_new * v_B;
    const double protected_reconstruction_mass_error = C_final - C_target;

    Y_r[idx] = Y_final;
    xB_r[idx] = xB_final;
    if (update_stats) {
        atomicAdd(&update_stats[GP_Y_UPDATE_SUM_CLIPPING_MASS_ERROR],
                  protected_reconstruction_mass_error);
        atomicAdd(&update_stats[GP_Y_UPDATE_SUM_SMALL_HALPHA_MASS_ERROR],
                  small_h_alpha ? protected_reconstruction_mass_error : 0.0);
        atomicAdd(&update_stats[GP_Y_UPDATE_Y_CLIP_COUNT],
                  (Y_final <= -Y_clip || Y_final >= Y_upper_cap) ? 1.0 : 0.0);
        atomicAdd(&update_stats[GP_Y_UPDATE_XB_CLIP_COUNT],
                  infeasible_reconstruction ? 1.0 : 0.0);
        if (small_h_alpha) {
            atomicAdd(&update_stats[GP_Y_UPDATE_SMALL_HALPHA_COUNT], 1.0);
        }
    }
}

__global__ void compute_xB_storage_rhs_kernel(
    const double *divJ_r,
    const double *phi_new_r,
    const double *phi_old_r,
    const double *lap_xB_r,
    const double *xB_old_r,
    double *rhs_xB_r,
    double dt,
    double v_B,
    double mean_Dx,
    double h_alpha_eps,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    const double h_new = h_of_phi(clamp01(phi_new_r[idx]));
    const double h_alpha = 1.0 - h_new;
    const double lap_old = lap_xB_r[idx];
    (void)phi_old_r;
    (void)xB_old_r;
    (void)dt;
    (void)v_B;
    // With phi held at the accepted split level, C_t=divJ gives
    // (1-h)*x_alpha,t=divJ. The phi-storage residual is closed separately by
    // the full-storage projection; beta support below the numerical storage
    // floor is protected from singular local inversion.
    const double physical_rate =
        (h_alpha >= h_alpha_eps) ? (divJ_r[idx] / h_alpha) : 0.0;
    rhs_xB_r[idx] = physical_rate - mean_Dx * lap_old;
}

__global__ void xB_normalize_clamp_and_logit_kernel(
    double *xB_ifft_r,
    double *xB_r,
    double *Y_r,
    double invN,
    double xB_eps,
    double Y_clip,
    double Y_upper_cap,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    double x = xB_ifft_r[idx] * invN;
    if (!isfinite(x)) x = xB_r[idx];
    x = fmin(fmax(x, xB_eps), 1.0 - xB_eps);
    double y = logit_from_fraction(x, xB_eps, Y_clip);
    y = fmin(fmax(y, -Y_clip), Y_upper_cap);
    x = sigmoid_from_logit(y, Y_clip, xB_eps);
    xB_r[idx] = x;
    Y_r[idx] = y;
}

__global__ void initialize_q_alpha_kernel(const double *phi_r,
                                          const double *xB_r,
                                          double *q_alpha_r,
                                          int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    const double h = h_of_phi(clamp01(phi_r[idx]));
    q_alpha_r[idx] = (1.0 - h) * xB_r[idx];
}

// Pure phase substep at zero transport: q_new=q_old-Delta(h)*v_B.
// An infeasible phase request is reported to the host and the run rejects the
// step; this kernel never clips q and silently discards conserved storage.
__global__ void apply_local_phase_storage_transfer_q_kernel(
    const double *phi_new_r,
    const double *phi_old_r,
    double *q_alpha_r,
    double v_B,
    double feasibility_tol,
    double *stats,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    const double h_old = h_of_phi(clamp01(phi_old_r[idx]));
    const double h_new = h_of_phi(clamp01(phi_new_r[idx]));
    const double alpha_new = 1.0 - h_new;
    const double delta_h = h_new - h_old;
    const double q_old = q_alpha_r[idx];
    const double q_new = q_old - delta_h * v_B;
    const double residual = (q_new + h_new * v_B) -
                            (q_old + h_old * v_B);
    atomicAdd(&stats[PF_Q_SUM_DELTA_H], delta_h * v_B);
    atomicAdd(&stats[PF_Q_SUM_LOCAL_DELTA_Q], q_new - q_old);
    atomicAdd(&stats[PF_Q_SUM_LOCAL_RESIDUAL], residual);
    atomicMaxAbsDouble(&stats[PF_Q_MAXABS_LOCAL_RESIDUAL], fabs(residual));
    atomicMinDouble(&stats[PF_Q_MIN_ALPHA], alpha_new);
    if (!isfinite(q_new) || q_new < -feasibility_tol ||
        q_new > alpha_new + feasibility_tol) {
        atomicAdd(&stats[PF_Q_INFEASIBLE_PHASE_COUNT], 1.0);
        return;
    }
    q_alpha_r[idx] = fmin(fmax(q_new, 0.0), alpha_new);
}

__global__ void compute_q_transport_rhs_kernel(const double *divJ_r,
                                               const double *lap_q_r,
                                               double *rhs_q_r,
                                               double mean_Dq,
                                               int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    rhs_q_r[idx] = divJ_r[idx] - mean_Dq * lap_q_r[idx];
}

__global__ void q_normalize_validate_and_reconstruct_kernel(
    double *q_ifft_r,
    double *q_alpha_r,
    const double *phi_r,
    double *xB_r,
    double *Y_r,
    double invN,
    double xB_eps,
    double Y_clip,
    double Y_upper_cap,
    double bound_tol,
    double *stats,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    const double h = h_of_phi(clamp01(phi_r[idx]));
    const double alpha = 1.0 - h;
    double q = q_ifft_r[idx] * invN;
    const double q_unbounded = q;
    if (!isfinite(q) || q < -bound_tol || q > alpha + bound_tol) {
        atomicAdd(&stats[PF_Q_TRANSPORT_BOUND_VIOLATION_COUNT], 1.0);
    }
    q = isfinite(q) ? fmin(fmax(q, 0.0), alpha) : 0.0;
    atomicAdd(&stats[PF_Q_TRANSPORT_BOUND_MASS], q - q_unbounded);
    q_alpha_r[idx] = q;

    // q and alpha co-vanish in beta support.  Their bounded ratio is the
    // declared matrix-channel context; at alpha=0 the previous bounded context
    // is retained because it carries zero conserved storage.
    double x = xB_r[idx];
    if (alpha > 1.0e-14) x = q / alpha;
    x = fmin(fmax(x, xB_eps), 1.0 - xB_eps);
    double y = logit_from_fraction(x, xB_eps, Y_clip);
    y = fmin(fmax(y, -Y_clip), Y_upper_cap);
    xB_r[idx] = sigmoid_from_logit(y, Y_clip, xB_eps);
    Y_r[idx] = y;
}

__global__ void q_explicit_transport_and_reconstruct_kernel(
    double *q_alpha_r,
    const double *divJ_r,
    const double *phi_r,
    double *xB_r,
    double *Y_r,
    double dt,
    double xB_eps,
    double Y_clip,
    double Y_upper_cap,
    double bound_tol,
    double *stats,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    const double h = h_of_phi(clamp01(phi_r[idx]));
    const double alpha = 1.0 - h;
    const double q_trial = q_alpha_r[idx] + dt * divJ_r[idx];
    if (!isfinite(q_trial) || q_trial < -bound_tol ||
        q_trial > alpha + bound_tol) {
        atomicAdd(&stats[PF_Q_TRANSPORT_BOUND_VIOLATION_COUNT], 1.0);
    }
    const double q = isfinite(q_trial)
                         ? fmin(fmax(q_trial, 0.0), alpha) : 0.0;
    atomicAdd(&stats[PF_Q_TRANSPORT_BOUND_MASS], q - q_trial);
    q_alpha_r[idx] = q;
    double x = xB_r[idx];
    if (alpha > 1.0e-14) x = q / alpha;
    x = fmin(fmax(x, xB_eps), 1.0 - xB_eps);
    double y = logit_from_fraction(x, xB_eps, Y_clip);
    y = fmin(fmax(y, -Y_clip), Y_upper_cap);
    xB_r[idx] = sigmoid_from_logit(y, Y_clip, xB_eps);
    Y_r[idx] = y;
}

__global__ void sync_q_from_phi_xB_kernel(const double *phi_r,
                                          const double *xB_r,
                                          double *q_alpha_r,
                                          int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    const double h = h_of_phi(clamp01(phi_r[idx]));
    q_alpha_r[idx] = (1.0 - h) * xB_r[idx];
}

// ============================================================================
// PF-only conservative Ctot/q_alpha architectures.
// Each checkerboard sweep owns one periodic face exactly once.  The two cells
// receive opposite increments, so limiting cannot change the global sum.
// ============================================================================
__global__ void initialize_conservative_storage_kernel(
    const double *phi_r, const double *xB_r, double *storage_r,
    double v_B, int primary_is_ctot, int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    const double h = h_of_phi(clamp01(phi_r[idx]));
    const double q = (1.0 - h) * clamp01(xB_r[idx]);
    storage_r[idx] = primary_is_ctot ? (q + h * v_B) : q;
}

__global__ void reconstruct_conservative_context_kernel(
    const double *storage_r, const double *phi_r, double *xB_r, double *Y_r,
    double temperature_K, double v_B, int primary_is_ctot,
    double beta_support_eps, double xB_eps, double Y_clip,
    double *stats, int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    const double h = h_of_phi(clamp01(phi_r[idx]));
    double alpha = fmax(1.0 - h, 0.0);
    const double q = primary_is_ctot ? (storage_r[idx] - h * v_B)
                                     : storage_r[idx];
    double x = solve_x_eq_device(temperature_K);
    if (alpha > beta_support_eps) {
        x = q / alpha;
    } else if (stats) {
        atomicAdd(&stats[PF_CONS_BETA_CONTEXT_COUNT], 1.0);
    }
    if (!isfinite(x)) {
        x = solve_x_eq_device(temperature_K);
        if (stats) atomicAdd(&stats[PF_CONS_NONFINITE_COUNT], 1.0);
    }
    x = fmin(fmax(x, xB_eps), 1.0 - xB_eps);
    xB_r[idx] = x;
    Y_r[idx] = logit_from_fraction(x, xB_eps, Y_clip);
}

__global__ void pairwise_conservative_face_sweep_kernel(
    double *storage_r, const double *mu_r, const double *phi_r,
    const double *xB_r, int Nx, int Ny, int Nz, int axis, int parity,
    double spacing, double dt, double D_alpha,
    double Vm_alpha_0, double dVm_alpha_dxB, double Vm_compound,
    double temperature_K, double mu_reference_scale,
    double v_B, int primary_is_ctot, int backward_euler,
    double bound_tol, double *stats, int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    const int k = idx % Nz;
    const int t = idx / Nz;
    const int j = t % Ny;
    const int i = t / Ny;
    const int coord = (axis == 0) ? i : ((axis == 1) ? j : k);
    if ((coord & 1) != parity) return;
    int ni = i, nj = j, nk = k;
    if (axis == 0) ni = (i + 1 == Nx) ? 0 : i + 1;
    else if (axis == 1) nj = (j + 1 == Ny) ? 0 : j + 1;
    else nk = (k + 1 == Nz) ? 0 : k + 1;
    const int nidx = (ni * Ny + nj) * Nz + nk;

    const double hi = h_of_phi(clamp01(phi_r[idx]));
    const double hj = h_of_phi(clamp01(phi_r[nidx]));
    const double ai = fmax(1.0 - hi, 0.0);
    const double aj = fmax(1.0 - hj, 0.0);
    const double xi = clamp_eps(xB_r[idx], 1.0e-12);
    const double xj = clamp_eps(xB_r[nidx], 1.0e-12);
    const double Gi = gamma_thermo_nonlinear(
        xi, 0.0, Vm_alpha_0, dVm_alpha_dxB, Vm_compound,
        temperature_K, mu_reference_scale);
    const double Gj = gamma_thermo_nonlinear(
        xj, 0.0, Vm_alpha_0, dVm_alpha_dxB, Vm_compound,
        temperature_K, mu_reference_scale);
    const double Mi = ai * stabilized_meff(D_alpha, Gi);
    const double Mj = aj * stabilized_meff(D_alpha, Gj);
    const double Mface = (Mi > 0.0 && Mj > 0.0)
        ? (2.0 * Mi * Mj / (Mi + Mj)) : 0.0;
    const double inv_dx2 = 1.0 / (spacing * spacing);
    double transfer = dt * Mface * (mu_r[idx] - mu_r[nidx]) * inv_dx2;
    if (backward_euler && Mface > 0.0) {
        const double dmu_dCi = Gi / fmax(ai, 1.0e-14);
        const double dmu_dCj = Gj / fmax(aj, 1.0e-14);
        const double denom = 1.0 + dt * Mface * inv_dx2 *
                                   fmax(dmu_dCi + dmu_dCj, 0.0);
        transfer /= denom;
    }

    const double si = storage_r[idx];
    const double sj = storage_r[nidx];
    const double loi = primary_is_ctot ? hi * v_B : 0.0;
    const double loj = primary_is_ctot ? hj * v_B : 0.0;
    const double hii = primary_is_ctot ? (hi * v_B + ai) : ai;
    const double hij = primary_is_ctot ? (hj * v_B + aj) : aj;
    const double raw = transfer;
    if (transfer >= 0.0) {
        transfer = fmin(transfer, fmax(si - loi, 0.0));
        transfer = fmin(transfer, fmax(hij - sj, 0.0));
    } else {
        const double mag = fmin(-transfer, fmax(sj - loj, 0.0));
        transfer = -fmin(mag, fmax(hii - si, 0.0));
    }
    if (stats) {
        atomicAdd(&stats[PF_CONS_SUM_SIGNED_TRANSFER], transfer - transfer);
        atomicAdd(&stats[PF_CONS_SUM_ABS_TRANSFER], fabs(transfer));
        atomicMaxAbsDouble(&stats[PF_CONS_MAXABS_FACE_TRANSFER], fabs(transfer));
        if (fabs(transfer - raw) > bound_tol)
            atomicAdd(&stats[PF_CONS_LIMITED_FACE_COUNT], 1.0);
    }
    storage_r[idx] = si - transfer;
    storage_r[nidx] = sj + transfer;
    if (stats && (storage_r[idx] < loi - bound_tol || storage_r[idx] > hii + bound_tol ||
                  storage_r[nidx] < loj - bound_tol || storage_r[nidx] > hij + bound_tol)) {
        atomicAdd(&stats[PF_CONS_BOUND_VIOLATION_COUNT], 1.0);
    }
}

__global__ void constrain_phase_and_reconstruct_conservative_kernel(
    double *phi_new_r, const double *phi_old_r, double *storage_r,
    double *xB_r, double *Y_r, double temperature_K, double v_B,
    int primary_is_ctot, double beta_support_eps, double xB_eps,
    double Y_clip, double bound_tol, double *stats, int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    const double phi_old = clamp01(phi_old_r[idx]);
    const double h_old = h_of_phi(phi_old);
    const double C = primary_is_ctot ? storage_r[idx]
                                     : storage_r[idx] + h_old * v_B;
    double phi = clamp01(phi_new_r[idx]);
    double h = h_of_phi(phi);
    const double hmax = (v_B > 0.0) ? fmin(fmax(C / v_B, 0.0), 1.0) : 1.0;
    if (h > hmax + bound_tol) {
        const double unconstrained = phi;
        phi = h_inverse_bisection_device(hmax);
        h = h_of_phi(phi);
        if (stats) {
            atomicAdd(&stats[PF_CONS_PHASE_CONSTRAINT_COUNT], 1.0);
            atomicMaxAbsDouble(&stats[PF_CONS_MAXABS_PHASE_CORRECTION],
                               fabs(phi - unconstrained));
        }
    }
    double alpha = fmax(1.0 - h, 0.0);
    double q = C - h * v_B;
    if (!isfinite(q) || q < -bound_tol || q > alpha + bound_tol) {
        if (stats) atomicAdd(&stats[PF_CONS_BOUND_VIOLATION_COUNT], 1.0);
        // Do not clip conserved storage.  Mark failure and retain the accepted
        // phase value; the host audit rejects/rolls back this step.
        phi = phi_old;
        h = h_old;
        alpha = fmax(1.0 - h, 0.0);
        q = C - h * v_B;
    }
    phi_new_r[idx] = phi;
    if (!primary_is_ctot) storage_r[idx] = q;
    double x = solve_x_eq_device(temperature_K);
    if (alpha > beta_support_eps) x = q / alpha;
    else if (stats) atomicAdd(&stats[PF_CONS_BETA_CONTEXT_COUNT], 1.0);
    if (!isfinite(x)) {
        x = solve_x_eq_device(temperature_K);
        if (stats) atomicAdd(&stats[PF_CONS_NONFINITE_COUNT], 1.0);
    }
    x = fmin(fmax(x, xB_eps), 1.0 - xB_eps);
    xB_r[idx] = x;
    Y_r[idx] = logit_from_fraction(x, xB_eps, Y_clip);
}


__global__ void gp_picard_storage_Y_update_kernel(
    const double *divJ_r,
    const double *phi_new_r,
    const double *phi_old_r,
    const double *eta_new_r,
    const double *eta_old_r,
    const double *Y_old_r,
    double *Y_guess_r,
    double *xB_r,
    double dt,
    double xB_GP,
    double Y_clip,
    double Y_upper_cap,
    double xB_eps,
    double *update_stats,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double h_alpha_old = 0.0, h_GP_old = 0.0, h_beta_old = 0.0;
    double h_alpha_new = 0.0, h_GP_new = 0.0, h_beta_new = 0.0;
    phase_fractions_gp(phi_old_r[idx], eta_old_r[idx], &h_alpha_old, &h_GP_old, &h_beta_old);
    phase_fractions_gp(phi_new_r[idx], eta_new_r[idx], &h_alpha_new, &h_GP_new, &h_beta_new);

    double xB_old = sigmoid_from_logit(Y_old_r[idx], Y_clip, xB_eps);
    double xB_guess = sigmoid_from_logit(Y_guess_r[idx], Y_clip, xB_eps);
    double S_old = h_GP_old * (xB_GP - xB_old) + h_beta_old * (1.0 - xB_old);
    double xBtot_target = xB_old + S_old + dt * divJ_r[idx];
    const int small_h_alpha = (h_alpha_new < fmax(xB_eps, 1.0e-8)) ? 1 : 0;

    double xB_unclipped = xB_old;
    if (!small_h_alpha) {
        double S_new_guess = h_GP_new * (xB_GP - xB_guess) + h_beta_new * (1.0 - xB_guess);
        double storage_source_discrete = (S_new_guess - S_old) / dt;
        xB_unclipped = xB_old + dt * (divJ_r[idx] - storage_source_discrete);
    }
    double xBtot_after_unclipped = h_alpha_new * xB_unclipped + h_GP_new * xB_GP + h_beta_new;

    double xb_low = 0.0;
    double xb_high = 0.0;
    double xB_final = xB_unclipped;
    if (xB_final < xB_eps) {
        xB_final = xB_eps;
        xb_low = 1.0;
    }
    if (xB_final > 1.0 - xB_eps) {
        xB_final = 1.0 - xB_eps;
        xb_high = 1.0;
    }

    double Y_unclipped = logit_from_fraction(xB_final, xB_eps, Y_clip);
    double y_low = 0.0;
    double y_high = 0.0;
    double Y_final = Y_unclipped;
    if (Y_final > Y_upper_cap) {
        Y_final = Y_upper_cap;
        y_high = 1.0;
    }
    if (Y_final < -Y_clip) {
        Y_final = -Y_clip;
        y_low = 1.0;
    }
    xB_final = sigmoid_from_logit(Y_final, Y_clip, xB_eps);
    double xBtot_after_clipped = h_alpha_new * xB_final + h_GP_new * xB_GP + h_beta_new;
    double clipping_mass_error = xBtot_after_clipped - xBtot_after_unclipped;
    double small_h_alpha_mass_error = xBtot_after_unclipped - xBtot_target;

    Y_guess_r[idx] = Y_final;
    xB_r[idx] = xB_final;

    if (update_stats) {
        atomicAdd(&update_stats[GP_Y_UPDATE_SUM_CLIPPING_MASS_ERROR], clipping_mass_error);
        atomicAdd(&update_stats[GP_Y_UPDATE_SUM_SMALL_HALPHA_MASS_ERROR], small_h_alpha_mass_error);
        atomicAdd(&update_stats[GP_Y_UPDATE_Y_CLIP_COUNT], y_low + y_high);
        atomicAdd(&update_stats[GP_Y_UPDATE_XB_CLIP_COUNT], xb_low + xb_high);
        if (small_h_alpha) {
            atomicAdd(&update_stats[GP_Y_UPDATE_SMALL_HALPHA_COUNT], 1.0);
        }
    }
}

__global__ void apply_Y_shift_recompute_xB_kernel(
    const double *Y_base_r,
    double *Y_r,
    double *xB_r,
    double lambda_shift,
    double Y_clip,
    double Y_upper_cap,
    double xB_eps,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    double Y_shifted = Y_base_r[idx] + lambda_shift;
    if (Y_shifted > Y_upper_cap) Y_shifted = Y_upper_cap;
    if (Y_shifted < -Y_clip) Y_shifted = -Y_clip;
    Y_r[idx] = Y_shifted;
    xB_r[idx] = sigmoid_from_logit(Y_shifted, Y_clip, xB_eps);
}

__global__ void gp_storage_diagnostics_kernel(
    const double *phi_old_r,
    const double *eta_old_r,
    const double *xB_old_r,
    const double *phi_new_r,
    const double *eta_new_r,
    const double *xB_new_r,
    const double *divJ_r,
    double dt,
    double xB_GP,
    double gp_h_alpha_eps,
    double *stats,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double h_alpha_old = 0.0, h_GP_old = 0.0, h_beta_old = 0.0;
    double h_alpha_new = 0.0, h_GP_new = 0.0, h_beta_new = 0.0;
    phase_fractions_gp(phi_old_r[idx], eta_old_r[idx], &h_alpha_old, &h_GP_old, &h_beta_old);
    phase_fractions_gp(phi_new_r[idx], eta_new_r[idx], &h_alpha_new, &h_GP_new, &h_beta_new);

    double xB_before = xB_old_r[idx];
    double xB_after = xB_new_r[idx];
    double storage_gp_before = h_GP_old * (xB_GP - xB_before);
    double storage_beta_before = h_beta_old * (1.0 - xB_before);
    double storage_gp_after = h_GP_new * (xB_GP - xB_after);
    double storage_beta_after = h_beta_new * (1.0 - xB_after);
    double xBtot_before = xB_before + storage_gp_before + storage_beta_before;
    double xBtot_after = xB_after + storage_gp_after + storage_beta_after;
    double delta_xB = xB_after - xB_before;
    double delta_storage_gp = storage_gp_after - storage_gp_before;
    double delta_storage_beta = storage_beta_after - storage_beta_before;
    double dt_divJ = dt * divJ_r[idx];
    double closure_error = (delta_xB + delta_storage_gp + delta_storage_beta) - dt_divJ;

    atomicAdd(&stats[MASS_DIAG_GP_STORAGE_SUM_XBTOT_BEFORE], xBtot_before);
    atomicAdd(&stats[MASS_DIAG_GP_STORAGE_SUM_XB_ALPHA_BEFORE], xB_before);
    atomicAdd(&stats[MASS_DIAG_GP_STORAGE_SUM_STORAGE_GP_BEFORE], storage_gp_before);
    atomicAdd(&stats[MASS_DIAG_GP_STORAGE_SUM_STORAGE_BETA_BEFORE], storage_beta_before);
    atomicAdd(&stats[MASS_DIAG_GP_STORAGE_SUM_XBTOT_AFTER], xBtot_after);
    atomicAdd(&stats[MASS_DIAG_GP_STORAGE_SUM_XB_ALPHA_AFTER], xB_after);
    atomicAdd(&stats[MASS_DIAG_GP_STORAGE_SUM_STORAGE_GP_AFTER], storage_gp_after);
    atomicAdd(&stats[MASS_DIAG_GP_STORAGE_SUM_STORAGE_BETA_AFTER], storage_beta_after);
    atomicAdd(&stats[MASS_DIAG_GP_STORAGE_SUM_DELTA_XB_ALPHA], delta_xB);
    atomicAdd(&stats[MASS_DIAG_GP_STORAGE_SUM_DELTA_STORAGE_GP], delta_storage_gp);
    atomicAdd(&stats[MASS_DIAG_GP_STORAGE_SUM_DELTA_STORAGE_BETA], delta_storage_beta);
    atomicAdd(&stats[MASS_DIAG_GP_STORAGE_SUM_DT_DIVJ], dt_divJ);
    atomicAdd(&stats[MASS_DIAG_GP_STORAGE_SUM_CLOSURE_ERROR], closure_error);
    atomicAdd(&stats[MASS_DIAG_GP_STORAGE_SUM_DIVJ], divJ_r[idx]);
    if (h_alpha_new < gp_h_alpha_eps) {
        atomicAdd(&stats[MASS_DIAG_GP_STORAGE_SMALL_HALPHA_COUNT], 1.0);
    }
    atomicMinDouble(&stats[MASS_DIAG_GP_STORAGE_MIN_HALPHA], h_alpha_new);
    atomicMaxDouble(&stats[MASS_DIAG_GP_STORAGE_MAX_HALPHA], h_alpha_new);
    atomicMinDouble(&stats[MASS_DIAG_GP_STORAGE_MIN_HGP], h_GP_new);
    atomicMaxDouble(&stats[MASS_DIAG_GP_STORAGE_MAX_HGP], h_GP_new);
    atomicMinDouble(&stats[MASS_DIAG_GP_STORAGE_MIN_ETA], eta_new_r[idx]);
    atomicMaxDouble(&stats[MASS_DIAG_GP_STORAGE_MAX_ETA], eta_new_r[idx]);
    atomicMinDouble(&stats[MASS_DIAG_GP_STORAGE_MIN_XB_ALPHA], xB_after);
    atomicMaxDouble(&stats[MASS_DIAG_GP_STORAGE_MAX_XB_ALPHA], xB_after);
}

// ============================================================================
// 11.5 xB 峰值限幅 kernel
// ============================================================================

__global__ void clamp_xB_max_kernel(
    double *xB,
    int N,
    double xB_max)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    if (xB[idx] > xB_max) xB[idx] = xB_max;
    if (xB[idx] < 0.0) xB[idx] = 0.0;  // 避免负值
}

// ============================================================================
// 12. 计算Laplacian kernel（在k空间）
// ============================================================================

__global__ void compute_laplacian_k_kernel(
    const cuDoubleComplex *f_k,
    const double *k2,
    cuDoubleComplex *lap_k,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    // lap_k = -k^2 * f_k
    lap_k[idx] = cuCmul(f_k[idx], make_cuDoubleComplex(-k2[idx], 0.0));
}

// ============================================================================
// 13. 更新dY_dt_prev kernel（在Y更新后调用）
// ============================================================================

__global__ void update_dY_dt_prev_kernel(
    const double *Y_r,
    const double *Y_n_saved,
    double *dY_dt_prev_r,
    double dt,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    dY_dt_prev_r[idx] = (Y_r[idx] - Y_n_saved[idx]) / dt;
}

__global__ void relax_dY_dt_guess_kernel(
    const double *dY_dt_new_r,
    const double *dY_dt_old_guess_r,
    double *dY_dt_guess_r,
    double omega,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    dY_dt_guess_r[idx] =
        omega * dY_dt_new_r[idx] + (1.0 - omega) * dY_dt_old_guess_r[idx];
}

__global__ void compute_diff_sumsq_max_kernel(
    const double *a_r,
    const double *b_r,
    double *stats,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    double d = a_r[idx] - b_r[idx];
    atomicAdd(&stats[0], d * d);
    atomicMaxAbsDouble(&stats[1], fabs(d));
}

// ============================================================================
// 14. 辅助kernel：计算DY值（用于归约计算mean_DY）
// ============================================================================

__global__ void compute_DY_values_kernel(
    const double *Y_r,
    const double *phi_r,
    double *DY_values,  // 输出每个点的DY值
    double D_alpha,
    double D_compound,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    double phi = phi_r[idx];
    double h = h_of_phi(phi);
    double Dm = D_mix(h, D_alpha, D_compound);
    
    double Y_val = Y_r[idx];
    double xB;
    if (Y_val >= 0.0) {
        double e_negY = exp(-Y_val);
        xB = 1.0 / (1.0 + e_negY);
    } else {
        double e_Y = exp(Y_val);
        xB = e_Y / (1.0 + e_Y);
    }
    if (xB < 1e-12) xB = 1e-12;
    if (xB > 1.0 - 1e-12) xB = 1.0 - 1e-12;
    double logistic_deriv = xB * (1.0 - xB);
    
    DY_values[idx] = Dm * logistic_deriv;
}

// ============================================================================
// 归约kernel：计算数组的和（使用共享内存优化）
// ============================================================================

__global__ void reduce_sum_kernel(
    const double *input,
    double *output,
    int n)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载数据到共享内存
    sdata[tid] = (i < n) ? input[i] : 0.0;
    __syncthreads();
    
    // 归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 写入结果
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// 14. 辅助kernel：计算xBtot（保留用于VTK输出时的临时计算）
// ============================================================================

__global__ void compute_xBtot_kernel(
    const double *phi_r,
    const double *xB_r,
    double *xBtot_r,
    double v_B,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    double phi = phi_r[idx];
    double h = h_of_phi(phi);
    double xB = xB_r[idx];
    
    xBtot_r[idx] = (1.0 - h) * xB + v_B * h;
}

__global__ void compute_xBtot_gp_kernel(
    const double *phi_r,
    const double *eta_r,
    const double *xB_alpha_r,
    double *xBtot_gp_r,
    double xB_GP,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double h_alpha = 0.0;
    double h_GP = 0.0;
    double h_beta = 0.0;
    phase_fractions_gp(phi_r[idx], eta_r[idx], &h_alpha, &h_GP, &h_beta);
    xBtot_gp_r[idx] = h_alpha * xB_alpha_r[idx] + h_GP * xB_GP + h_beta;
}

// ============================================================================
// 优化kernel：直接归约计算N_in和N_if（不需要中间存储数组）
// ============================================================================

__global__ void reduce_sum_N_in_N_if_kernel(
    const double *phi_r,
    double *block_sums_N_in,
    double *block_sums_N_if,
    int total_size)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 计算当前点的N_in和N_if贡献
    double N_in_val = 0.0;
    double N_if_val = 0.0;
    
    if (i < total_size) {
        double phi = phi_r[i];
        N_in_val = (phi > 0.5) ? 1.0 : 0.0;
        // 界面判定范围：0.12 < phi < 0.88
        N_if_val = ((phi > 0.12) && (phi < 0.88)) ? 1.0 : 0.0;
    }
    
    // 使用共享内存进行归约（存储两个值：N_in和N_if）
    sdata[2*tid] = N_in_val;
    sdata[2*tid+1] = N_if_val;
    __syncthreads();
    
    // 归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[2*tid] += sdata[2*(tid+s)];
            sdata[2*tid+1] += sdata[2*(tid+s)+1];
        }
        __syncthreads();
    }
    
    // 写入结果
    if (tid == 0) {
        block_sums_N_in[blockIdx.x] = sdata[0];
        block_sums_N_if[blockIdx.x] = sdata[1];
    }
}

// ============================================================================
// 优化kernel：直接归约计算xBtot的和（不需要存储完整数组）
// ============================================================================

__global__ void reduce_sum_xBtot_kernel(
    const double *phi_r,
    const double *xB_r,
    double *block_sums,
    double v_B,
    int total_size)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 计算xBtot
    double val = 0.0;
    if (i < total_size) {
        double phi = phi_r[i];
        double h = h_of_phi(phi);
        double xB = xB_r[i];
        val = (1.0 - h) * xB + v_B * h;
    }
    
    // 加载到共享内存
    sdata[tid] = val;
    __syncthreads();
    
    // 归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 写入结果
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_sum_xBtot_gp_kernel(
    const double *phi_r,
    const double *eta_r,
    const double *xB_alpha_r,
    double *block_sums,
    double xB_GP,
    int total_size)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    double val = 0.0;
    if (i < total_size) {
        double h_alpha = 0.0;
        double h_GP = 0.0;
        double h_beta = 0.0;
        phase_fractions_gp(phi_r[i], eta_r[i], &h_alpha, &h_GP, &h_beta);
        val = h_alpha * xB_alpha_r[i] + h_GP * xB_GP + h_beta;
    }

    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// 优化kernel：直接归约计算DY的和（不需要存储完整数组）
// ============================================================================

__global__ void reduce_sum_DY_kernel(
    const double *Y_r,
    const double *phi_r,
    double *block_sums,
    double D_alpha,
    double D_compound,
    int total_size)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 计算DY值
    double val = 0.0;
    if (i < total_size) {
        double phi = phi_r[i];
        double h = h_of_phi(phi);
        double Dm = D_mix(h, D_alpha, D_compound);
        
        double Y_val = Y_r[i];
        double xB;
        if (Y_val >= 0.0) {
            double e_negY = exp(-Y_val);
            xB = 1.0 / (1.0 + e_negY);
        } else {
            double e_Y = exp(Y_val);
            xB = e_Y / (1.0 + e_Y);
        }
        if (xB < 1e-12) xB = 1e-12;
        if (xB > 1.0 - 1e-12) xB = 1.0 - 1e-12;
        double logistic_deriv = xB * (1.0 - xB);
        
        val = Dm * logistic_deriv;
    }
    
    // 加载到共享内存
    sdata[tid] = val;
    __syncthreads();
    
    // 归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 写入结果
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_sum_DY_gp_kernel(
    const double *Y_r,
    const double *phi_r,
    const double *eta_r,
    double *block_sums,
    double D_alpha,
    int total_size)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    double val = 0.0;
    if (i < total_size) {
        double h_alpha = 0.0;
        phase_fractions_gp(phi_r[i], eta_r[i], &h_alpha, NULL, NULL);

        double xB = sigmoid_from_logit(Y_r[i], 20.0, 1e-12);
        double logistic_deriv = xB * (1.0 - xB);
        val = h_alpha * D_alpha * logistic_deriv;
    }

    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// 优化kernel：计算析出相内部的平均hydrostatic stress
// ============================================================================

__global__ void reduce_sum_sigma_hydro_in_precipitate_kernel(
    const double *phi_r,
    const float *sigma_xx_r,
    const float *sigma_yy_r,
    const float *sigma_zz_r,
    double *block_sums_sigma,
    double *block_sums_count,
    int total_size)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 计算当前点的贡献
    double sigma_sum = 0.0;
    double count = 0.0;
    
    if (i < total_size) {
        double phi = phi_r[i];
        // 只计算析出相内部（phi > 0.5）的点
        if (phi > 0.5) {
            double sigma_hydro = (double)sigma_xx_r[i] + (double)sigma_yy_r[i] + (double)sigma_zz_r[i];
            sigma_sum = sigma_hydro;
            count = 1.0;
        }
    }
    
    // 使用共享内存进行归约（存储两个值：sigma_sum和count）
    sdata[2*tid] = sigma_sum;
    sdata[2*tid+1] = count;
    __syncthreads();
    
    // 归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[2*tid] += sdata[2*(tid+s)];
            sdata[2*tid+1] += sdata[2*(tid+s)+1];
        }
        __syncthreads();
    }
    
    // 写入结果
    if (tid == 0) {
        block_sums_sigma[blockIdx.x] = sdata[0];
        block_sums_count[blockIdx.x] = sdata[1];
    }
}

double gpu_reduce_avg_sigma_hydro_in_precipitate(
    const double *phi_r,
    const float *sigma_xx_r,
    const float *sigma_yy_r,
    const float *sigma_zz_r,
    int total_size)
{
    if (total_size <= 0) return 0.0;
    
    int threads_per_block = 256;
    int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;
    
    // 分配临时数组存储每个block的归约结果
    double *d_block_sums_sigma, *d_block_sums_count;
    CUDA_CHECK(cudaMalloc(&d_block_sums_sigma, num_blocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_sums_count, num_blocks * sizeof(double)));
    
    // 第一次归约：每个block归约
    size_t shared_mem_size = 2 * threads_per_block * sizeof(double);  // 存储sigma_sum和count
    reduce_sum_sigma_hydro_in_precipitate_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        phi_r, sigma_xx_r, sigma_yy_r, sigma_zz_r,
        d_block_sums_sigma, d_block_sums_count, total_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 递归归约：对block结果再次归约
    double sum_sigma = gpu_reduce_sum(d_block_sums_sigma, num_blocks);
    double sum_count = gpu_reduce_sum(d_block_sums_count, num_blocks);
    
    CUDA_CHECK(cudaFree(d_block_sums_sigma));
    CUDA_CHECK(cudaFree(d_block_sums_count));
    
    // 计算平均值
    if (sum_count > 0.0) {
        return sum_sigma / sum_count;
    } else {
        return 0.0;
    }
}

// ============================================================================
// 优化kernel：直接归约计算min和max（不需要复制到CPU）
// ============================================================================

__global__ void reduce_min_max_kernel(
    const double *input,
    double *block_mins,
    double *block_maxs,
    int total_size)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载数据：对于超出范围的值，min使用很大的数，max使用很小的数
    // 这样在归约时它们不会影响有效数据的min/max
    double min_val = (i < total_size) ? input[i] : 1e30;
    double max_val = (i < total_size) ? input[i] : -1e30;
    
    // 存储到共享内存（交错存储min和max）
    sdata[2*tid] = min_val;
    sdata[2*tid+1] = max_val;
    __syncthreads();
    
    // 归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[2*tid] = fmin(sdata[2*tid], sdata[2*(tid+s)]);
            sdata[2*tid+1] = fmax(sdata[2*tid+1], sdata[2*(tid+s)+1]);
        }
        __syncthreads();
    }
    
    // 写入结果
    if (tid == 0) {
        block_mins[blockIdx.x] = sdata[0];
        block_maxs[blockIdx.x] = sdata[1];
    }
}

// ============================================================================
// 优化kernel：直接归约计算Meff的min和max
// ============================================================================

__global__ void reduce_min_max_meff_kernel(
    const double *phi_r,
    const double *xB_r,
    double *block_mins,
    double *block_maxs,
    double D_alpha,
    double D_compound,
    double Vm_alpha_0,
    double dVm_alpha_dxB,
    double Vm_compound,
    double temperature_K,
    double mu_reference_scale,
    int total_size)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    double min_val = 1e30;
    double max_val = -1e30;
    
    if (i < total_size) {
        double phi = phi_r[i];
        double h = h_of_phi(phi);
        double xB = xB_r[i];
        
        // 计算混合扩散系数和热力学因子 (与 compute_flux_kernel 逻辑一致)
        double Dm = D_mix(h, D_alpha, D_compound);
        double G = gamma_thermo_nonlinear(xB, h, Vm_alpha_0, dVm_alpha_dxB, 
                                          Vm_compound, temperature_K, mu_reference_scale);
        
        double Meff = stabilized_meff(Dm, G);
        
        min_val = Meff;
        max_val = Meff;
    }
    
    // 使用共享内存进行归约
    sdata[2*tid] = min_val;
    sdata[2*tid+1] = max_val;
    __syncthreads();
    
    // 归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[2*tid] = fmin(sdata[2*tid], sdata[2*(tid+s)]);
            sdata[2*tid+1] = fmax(sdata[2*tid+1], sdata[2*(tid+s)+1]);
        }
        __syncthreads();
    }
    
    // 写入结果
    if (tid == 0) {
        block_mins[blockIdx.x] = sdata[0];
        block_maxs[blockIdx.x] = sdata[1];
    }
}

// ============================================================================
// Launch函数（kernel包装器，配置启动参数）
// ============================================================================

static inline void configure_launch(int total_size, int &threads_per_block, int &blocks) {
    threads_per_block = 256;
    blocks = (total_size + threads_per_block - 1) / threads_per_block;
}

void launch_dealias_kernel(cuDoubleComplex *f_k, int Nx, int Ny, int Nz, int NzC,
                           double dx, double dy, double dz, int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    dealias_23_kernel<<<blocks, threads>>>(f_k, Nx, Ny, Nz, NzC, dx, dy, dz, total_size);
}

void launch_dealias_float_kernel(cufftComplex *f_k, int Nx, int Ny, int Nz, int NzC,
                                 double dx, double dy, double dz, int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    dealias_23_float_kernel<<<blocks, threads>>>(f_k, Nx, Ny, Nz, NzC, dx, dy, dz, total_size);
}

void launch_compute_phi_rhs_kernel(const double *phi_r, const double *eta_r, const double *xB_r,
                                   double *rhs_r, int Nx, int Ny, int Nz,
                                   double temperature_K, double mu_reference_scale,
                                   double v_A, double v_B, double mu0_compound,
                                   double Vm_compound, double Vm_alpha_0,
                                   double dVm_alpha_dxB, double W,
                                   const float *sigma_xx_r,
                                   const float *sigma_yy_r,
                                   const float *sigma_zz_r,
                                   const float *sigma_xy_r,
                                   const float *sigma_xz_r,
                                   const float *sigma_yz_r,
                                   const float *uxx_r, const float *uyy_r, const float *uzz_r,
                                   const float *uxy_r, const float *uxz_r, const float *uyz_r,
                                   // Optimization(4): uxx0_r..uyz0_r 参数已移除
                                   double S_p_11, double S_p_12, double S_p_13, double S_p_14, double S_p_15, double S_p_16,
                                   double S_p_22, double S_p_23, double S_p_24, double S_p_25, double S_p_26,
                                   double S_p_33, double S_p_34, double S_p_35, double S_p_36,
                                   double S_p_44, double S_p_45, double S_p_46,
                                   double S_p_55, double S_p_56,
                                   double S_p_66,
                                   double eps_xx00, double eps_yy00, double eps_zz00,
                                   double eps_yz00, double eps_xz00, double eps_xy00,
                                   double eps_iso_over_vB,
                                   int gp_mode_enabled,
                                   int gp_elastic_enabled,
                                   int gp_elastic_active_phi,
                                   double gp_eps_iso,
                                   double gp_elastic_derivative_scale,
                                   double elastic_shift_dimless,
                                   int disable_chem,
                                   int total_size,
                                   int elastic_enabled,
                                   double *diag_stats) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_phi_rhs_kernel<<<blocks, threads>>>(
        phi_r, eta_r, xB_r, rhs_r,
        Nx, Ny, Nz,
        temperature_K, mu_reference_scale,
        v_A, v_B, mu0_compound,
        Vm_compound, Vm_alpha_0,
        dVm_alpha_dxB, W,
        sigma_xx_r, sigma_yy_r, sigma_zz_r,
        sigma_xy_r, sigma_xz_r, sigma_yz_r,
        uxx_r, uyy_r, uzz_r, uxy_r, uxz_r, uyz_r,
        // Optimization(4): uxx0_r..uyz0_r 参数已移除
        S_p_11, S_p_12, S_p_13, S_p_14, S_p_15, S_p_16,
        S_p_22, S_p_23, S_p_24, S_p_25, S_p_26,
        S_p_33, S_p_34, S_p_35, S_p_36,
        S_p_44, S_p_45, S_p_46,
        S_p_55, S_p_56, S_p_66,
        eps_xx00, eps_yy00, eps_zz00,
        eps_yz00, eps_xz00, eps_xy00,
        eps_iso_over_vB,
        gp_mode_enabled, gp_elastic_enabled, gp_elastic_active_phi,
        gp_eps_iso, gp_elastic_derivative_scale,
        elastic_shift_dimless,
        disable_chem,
        total_size,
        elastic_enabled,
        diag_stats);
}

void launch_reconstruct_xB_from_phi_kernel(const double *phi_r, double *xB_r,
                                           double xB_out, double xB_eq,
                                           int total_size)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    reconstruct_xB_from_phi_kernel<<<blocks, threads>>>(phi_r, xB_r, xB_out, xB_eq, total_size);
}

void launch_add_volume_constraint_kernel(const double *phi_r, double *rhs_r,
                                         double lambda,
                                         int total_size)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    add_volume_constraint_kernel<<<blocks, threads>>>(phi_r, rhs_r, lambda, total_size);
}

void launch_apply_volume_projection_kernel(double *phi_r, double lambda_correct, int total_size)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    apply_volume_projection_kernel<<<blocks, threads>>>(phi_r, lambda_correct, total_size);
}

void launch_compute_hprime_values_kernel(const double *phi_r, double *out_r, int total_size)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_hprime_values_kernel<<<blocks, threads>>>(phi_r, out_r, total_size);
}

void launch_compute_hprime_sq_values_kernel(const double *phi_r, double *out_r, int total_size)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_hprime_sq_values_kernel<<<blocks, threads>>>(phi_r, out_r, total_size);
}

void launch_compute_hprime_times_rhs_kernel(const double *phi_r, const double *rhs_r,
                                            double *out_r, int total_size)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_hprime_times_rhs_kernel<<<blocks, threads>>>(phi_r, rhs_r, out_r, total_size);
}

void launch_compute_euler_lagrange_residual_kernel(const double *rhs_r,
                                                   const double *lap_phi_r,
                                                   const double *phi_r,
                                                   double kappa,
                                                   double lambda_vol,
                                                   double *residual_r,
                                                   int total_size)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_euler_lagrange_residual_kernel<<<blocks, threads>>>(
        rhs_r, lap_phi_r, phi_r, kappa, lambda_vol, residual_r, total_size);
}

void launch_compute_gbulk_excess_hat_kernel(const double *phi_r, const double *xB_r,
                                            double *gbulk_excess_r,
                                            double temperature_K, double mu_reference_scale,
                                            double mu0_compound,
                                            double xB_ref, double g_bulk0_hat,
                                            int total_size)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_gbulk_excess_hat_kernel<<<blocks, threads>>>(
        phi_r, xB_r, gbulk_excess_r,
        temperature_K, mu_reference_scale,
        mu0_compound,
        xB_ref, g_bulk0_hat,
        total_size);
}

void launch_scale_field_by_h_kernel(const double *phi_r,
                                    double *field_r,
                                    int total_size)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    scale_field_by_h_kernel<<<blocks, threads>>>(
        phi_r, field_r, total_size);
}

void launch_compute_grad_energy_density_kernel(const double *phi_r, double *e_grad_r,
                                               int Nx, int Ny, int Nz,
                                               double dx, double dy, double dz,
                                               double kappa_phi,
                                               int total_size)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_grad_energy_density_kernel<<<blocks, threads>>>(
        phi_r, e_grad_r,
        Nx, Ny, Nz,
        dx, dy, dz,
        kappa_phi,
        total_size);
}

void launch_compute_dw_energy_density_kernel(const double *phi_r, double *e_dw_r, double W, int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_dw_energy_density_kernel<<<blocks, threads>>>(phi_r, e_dw_r, W, total_size);
}

void launch_phi_semi_implicit_update_kernel(const cuDoubleComplex *phi_k_old,
                                             const cuDoubleComplex *rhs_k,
                                             const double *k2,
                                             cuDoubleComplex *phi_k_new,
                                             double L_phi, double kappa_phi,
                                             double dt, int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    phi_semi_implicit_update_kernel<<<blocks, threads>>>(phi_k_old, rhs_k, k2,
                                                          phi_k_new, L_phi, kappa_phi,
                                                          dt, total_size);
}

void launch_compute_eta_rhs_kernel(const double *eta_r,
                                   const double *phi_r,
                                   const double *xB_alpha_r,
                                   const float *sigma_xx_r,
                                   const float *sigma_yy_r,
                                   const float *sigma_zz_r,
                                   const float *sigma_xy_r,
                                   const float *sigma_xz_r,
                                   const float *sigma_yz_r,
                                   double *rhs_r,
                                   double temperature_K,
                                   double mu_reference_scale,
                                   double Vm_alpha_0,
                                   double dVm_alpha_dxB,
                                   double gp_xB_fixed,
                                   double gp_delta_g0,
                                   int gp_raw_reaction_drive_only,
                                   int gp_raw_reaction_drive_use_raw_units_debug,
                                   double gp_reaction_nu_A,
                                   double gp_reaction_nu_B,
                                   double gp_mu_reference_raw,
                                   double gp_W_eta,
                                   double eps_iso_over_vB,
                                   int gp_elastic_enabled,
                                   int gp_elastic_active_eta,
                                   double gp_eps_iso,
                                   double gp_elastic_derivative_scale,
                                   int total_size,
                                   double *diag_stats)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_eta_rhs_kernel<<<blocks, threads>>>(
        eta_r, phi_r, xB_alpha_r,
        sigma_xx_r, sigma_yy_r, sigma_zz_r,
        sigma_xy_r, sigma_xz_r, sigma_yz_r,
        rhs_r,
        temperature_K, mu_reference_scale,
        Vm_alpha_0, dVm_alpha_dxB,
        gp_xB_fixed, gp_delta_g0,
        gp_raw_reaction_drive_only, gp_raw_reaction_drive_use_raw_units_debug,
        gp_reaction_nu_A, gp_reaction_nu_B, gp_mu_reference_raw,
        gp_W_eta,
        eps_iso_over_vB,
        gp_elastic_enabled, gp_elastic_active_eta,
        gp_eps_iso, gp_elastic_derivative_scale,
        total_size, diag_stats);
}

void launch_compute_eta_rhs_components_kernel(const double *eta_r,
                                              const double *phi_r,
                                              const double *xB_alpha_r,
                                              const float *sigma_xx_r,
                                              const float *sigma_yy_r,
                                              const float *sigma_zz_r,
                                              const float *sigma_xy_r,
                                              const float *sigma_xz_r,
                                              const float *sigma_yz_r,
                                              double *chem_r,
                                              double *dw_r,
                                              double *elastic_r,
                                              double *net_explicit_r,
                                              double *minus_delta_mu_r_gp_r,
                                              double temperature_K,
                                              double mu_reference_scale,
                                              double Vm_alpha_0,
                                              double dVm_alpha_dxB,
                                              double gp_xB_fixed,
                                              double gp_delta_g0,
                                              int gp_raw_reaction_drive_only,
                                              int gp_raw_reaction_drive_use_raw_units_debug,
                                              double gp_reaction_nu_A,
                                              double gp_reaction_nu_B,
                                              double gp_mu_reference_raw,
                                              double gp_W_eta,
                                              double eps_iso_over_vB,
                                              int gp_elastic_enabled,
                                              int gp_elastic_active_eta,
                                              double gp_eps_iso,
                                              double gp_elastic_derivative_scale,
                                              int total_size)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_eta_rhs_components_kernel<<<blocks, threads>>>(
        eta_r, phi_r, xB_alpha_r,
        sigma_xx_r, sigma_yy_r, sigma_zz_r,
        sigma_xy_r, sigma_xz_r, sigma_yz_r,
        chem_r, dw_r, elastic_r, net_explicit_r, minus_delta_mu_r_gp_r,
        temperature_K, mu_reference_scale,
        Vm_alpha_0, dVm_alpha_dxB,
        gp_xB_fixed, gp_delta_g0,
        gp_raw_reaction_drive_only, gp_raw_reaction_drive_use_raw_units_debug,
        gp_reaction_nu_A, gp_reaction_nu_B, gp_mu_reference_raw,
        gp_W_eta,
        eps_iso_over_vB,
        gp_elastic_enabled, gp_elastic_active_eta,
        gp_eps_iso, gp_elastic_derivative_scale,
        total_size);
}

void launch_eta_semi_implicit_update_kernel(const cuDoubleComplex *eta_k_old,
                                            const cuDoubleComplex *rhs_k,
                                            const double *k2,
                                            cuDoubleComplex *eta_k_new,
                                            double L_eta, double kappa_eta,
                                            double dt, int total_size)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    eta_semi_implicit_update_kernel<<<blocks, threads>>>(
        eta_k_old, rhs_k, k2, eta_k_new, L_eta, kappa_eta, dt, total_size);
}

void launch_phi_normalize_and_clamp_kernel(double *phi_r, double invN, int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    phi_normalize_and_clamp_kernel<<<blocks, threads>>>(phi_r, invN, total_size);
}

void launch_eta_normalize_and_clamp_kernel(double *eta_r, double invN, int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    eta_normalize_and_clamp_kernel<<<blocks, threads>>>(eta_r, invN, total_size);
}

void launch_gp_eta_feasibility_and_limiter_kernel(double *eta_r,
                                                  const double *eta_old_r,
                                                  const double *phi_new_r,
                                                  const double *phi_old_r,
                                                  const double *xB_old_r,
                                                  const double *divJ_r,
                                                  double dt,
                                                  double xB_GP,
                                                  double xB_min,
                                                  double xB_max,
                                                  double gp_h_alpha_eps,
                                                  int enable_limiter,
                                                  double *stats,
                                                  int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    gp_eta_feasibility_and_limiter_kernel<<<blocks, threads>>>(
        eta_r, eta_old_r, phi_new_r, phi_old_r, xB_old_r, divJ_r, dt,
        xB_GP, xB_min, xB_max,
        gp_h_alpha_eps, enable_limiter, stats, total_size);
}

// Backward-compatible overload kept so any stale TU that still expects the
// earlier pre-divJ signature links cleanly. It falls back to zero divJ and
// phi_old = phi_new, which matches the original diagnostic-only behavior.
void launch_gp_eta_feasibility_and_limiter_kernel(double *eta_r,
                                                  const double *eta_old_r,
                                                  const double *phi_r,
                                                  const double *xB_r,
                                                  double invN,
                                                  double xB_GP,
                                                  double xB_min,
                                                  double xB_max,
                                                  double gp_h_alpha_eps,
                                                  int enable_limiter,
                                                  double *stats,
                                                  int total_size) {
    (void)invN;
    launch_gp_eta_feasibility_and_limiter_kernel(
        eta_r, eta_old_r, phi_r, phi_r, xB_r, NULL, 0.0,
        xB_GP, xB_min, xB_max, gp_h_alpha_eps, enable_limiter, stats, total_size);
}

void launch_phi_mass_diagnostics_kernel(const double *phi_ifft_r,
                                        const double *phi_before_r,
                                        const double *xB_before_r,
                                        double *stats,
                                        double invN,
                                        double v_B,
                                        int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    phi_mass_diagnostics_kernel<<<blocks, threads>>>(
        phi_ifft_r, phi_before_r, xB_before_r, stats, invN, v_B, total_size);
}

void launch_normalize_only_kernel(double *arr, double invN, int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    normalize_only_kernel<<<blocks, threads>>>(arr, invN, total_size);
}

void launch_compute_mu_x_kernel(const double *Y_r, const double *phi_r,
                                double *xB_r, double *mu_x_r,
                                double temperature_K, double mu_reference_scale,
                                double v_A, double v_B, double mu0_compound,
                                double Vm_compound, double Vm_alpha_0,
                                double dVm_alpha_dxB, double Y_clip, double xB_eps,
                                const float *sigma_xx_r,
                                const float *sigma_yy_r,
                                const float *sigma_zz_r,
                                double eps_iso_over_vB,
                                int total_size,
                                int elastic_enabled) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_mu_x_kernel<<<blocks, threads>>>(
        Y_r, phi_r, xB_r, mu_x_r,
        temperature_K, mu_reference_scale,
        v_A, v_B, mu0_compound,
        Vm_compound, Vm_alpha_0,
        dVm_alpha_dxB, Y_clip, xB_eps,
        sigma_xx_r, sigma_yy_r, sigma_zz_r,
        eps_iso_over_vB,
        total_size,
        elastic_enabled);
}

void launch_compute_mu_C_gp_kernel(const double *Y_r,
                                   double *xB_alpha_r, double *mu_C_r,
                                   double temperature_K, double mu_reference_scale,
                                   double Vm_alpha_0, double dVm_alpha_dxB,
                                   double Y_clip, double xB_eps,
                                   const float *sigma_xx_r,
                                   const float *sigma_yy_r,
                                   const float *sigma_zz_r,
                                   double eps_iso_over_vB,
                                   int total_size,
                                   int elastic_enabled) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_mu_C_gp_kernel<<<blocks, threads>>>(
        Y_r, xB_alpha_r, mu_C_r,
        temperature_K, mu_reference_scale,
        Vm_alpha_0, dVm_alpha_dxB,
        Y_clip, xB_eps,
        sigma_xx_r, sigma_yy_r, sigma_zz_r,
        eps_iso_over_vB,
        total_size,
        elastic_enabled);
}

void launch_compute_mu_x_from_xB_kernel(const double *phi_r,
                                        const double *xB_r,
                                        double *mu_x_r,
                                        double temperature_K, double mu_reference_scale,
                                        double v_A, double v_B, double mu0_compound,
                                        double Vm_compound, double Vm_alpha_0,
                                        double dVm_alpha_dxB,
                                        const float *sigma_xx_r,
                                        const float *sigma_yy_r,
                                        const float *sigma_zz_r,
                                        double eps_iso_over_vB,
                                        int total_size,
                                        int elastic_enabled)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_mu_x_from_xB_kernel<<<blocks, threads>>>(
        phi_r, xB_r, mu_x_r,
        temperature_K, mu_reference_scale,
        v_A, v_B, mu0_compound,
        Vm_compound, Vm_alpha_0,
        dVm_alpha_dxB,
        sigma_xx_r, sigma_yy_r, sigma_zz_r,
        eps_iso_over_vB,
        total_size,
        elastic_enabled);
}

void launch_add_xB_chain_rule_to_phi_rhs_kernel(const double *phi_r,
                                                const double *mu_x_r,
                                                double *rhs_phi_r,
                                                double xB_out, double xB_eq,
                                                int total_size)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    add_xB_chain_rule_to_phi_rhs_kernel<<<blocks, threads>>>(
        phi_r, mu_x_r, rhs_phi_r,
        xB_out, xB_eq,
        total_size);
}

void launch_compute_delta_mu_r_kernel(const double *phi_r, const double *xB_r,
                                     double *delta_mu_r,
                                     double temperature_K, double mu_reference_scale,
                                     double v_A, double v_B, double mu0_compound,
                                     double elastic_shift_dimless,
                                     int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_delta_mu_r_kernel<<<blocks, threads>>>(
        phi_r, xB_r, delta_mu_r,
        temperature_K, mu_reference_scale,
        v_A, v_B, mu0_compound,
        elastic_shift_dimless,
        total_size);
}

void launch_compute_f_phi_chem_dgel_dphi_kernel(
    const double *phi_r,
    const double *xB_r,
    double *f_phi_chem_r,
    double *f_phi_dw_r,
    double *f_phi_bulk_r,
    double *dgel_dphi_r,
    double temperature_K,
    double mu_reference_scale,
    double v_A, double v_B,
    double mu0_compound,
    double Vm_compound,
    double Vm_alpha_0,
    double dVm_alpha_dxB,
    double W,
    const float *sigma_xx_r,
    const float *sigma_yy_r,
    const float *sigma_zz_r,
    const float *sigma_xy_r,
    const float *sigma_xz_r,
    const float *sigma_yz_r,
    const float *uxx_r, const float *uyy_r, const float *uzz_r,
    const float *uxy_r, const float *uxz_r, const float *uyz_r,
    // Optimization(4): eigenstrain 参数已移除
    double S_p_11, double S_p_12, double S_p_13, double S_p_14, double S_p_15, double S_p_16,
    double S_p_22, double S_p_23, double S_p_24, double S_p_25, double S_p_26,
    double S_p_33, double S_p_34, double S_p_35, double S_p_36,
    double S_p_44, double S_p_45, double S_p_46,
    double S_p_55, double S_p_56,
    double S_p_66,
    double eps_xx00, double eps_yy00, double eps_zz00,
    double eps_yz00, double eps_xz00, double eps_xy00,
    double eps_iso_over_vB,
    double elastic_shift_dimless,
    int total_size,
    int elastic_enabled) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_f_phi_chem_dgel_dphi_kernel<<<blocks, threads>>>(
        phi_r, xB_r, f_phi_chem_r, f_phi_dw_r, f_phi_bulk_r, dgel_dphi_r,
        temperature_K, mu_reference_scale,
        v_A, v_B, mu0_compound,
        Vm_compound, Vm_alpha_0, dVm_alpha_dxB, W,
        sigma_xx_r, sigma_yy_r, sigma_zz_r,
        sigma_xy_r, sigma_xz_r, sigma_yz_r,
        uxx_r, uyy_r, uzz_r, uxy_r, uxz_r, uyz_r,
        // Optimization(4): eigenstrain 参数已移除
        S_p_11, S_p_12, S_p_13, S_p_14, S_p_15, S_p_16,
        S_p_22, S_p_23, S_p_24, S_p_25, S_p_26,
        S_p_33, S_p_34, S_p_35, S_p_36,
        S_p_44, S_p_45, S_p_46,
        S_p_55, S_p_56, S_p_66,
        eps_xx00, eps_yy00, eps_zz00,
        eps_yz00, eps_xz00, eps_xy00,
        eps_iso_over_vB,
        elastic_shift_dimless,
        total_size,
        elastic_enabled);
}

// Optimization: 2-slot scratch 用：OutA 输出 f_phi_chem, f_phi_dw
void launch_compute_f_phi_chem_f_phi_dw_kernel(
    const double *phi_r, const double *xB_r,
    double *f_phi_chem_r, double *f_phi_dw_r,
    double temperature_K, double mu_reference_scale,
    double v_A, double v_B, double mu0_compound,
    double Vm_compound, double Vm_alpha_0, double dVm_alpha_dxB,
    double W, double elastic_shift_dimless, int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_f_phi_chem_f_phi_dw_kernel<<<blocks, threads>>>(
        phi_r, xB_r, f_phi_chem_r, f_phi_dw_r,
        temperature_K, mu_reference_scale, v_A, v_B, mu0_compound,
        Vm_compound, Vm_alpha_0, dVm_alpha_dxB, W, elastic_shift_dimless, total_size);
}

// Optimization: 2-slot scratch 用：OutB 输出 f_phi_bulk, dgel_dphi
void launch_compute_f_phi_bulk_dgel_dphi_kernel(
    const double *phi_r, const double *xB_r,
    double *f_phi_bulk_r, double *dgel_dphi_r,
    double temperature_K, double mu_reference_scale,
    double v_A, double v_B, double mu0_compound,
    double Vm_compound, double Vm_alpha_0, double dVm_alpha_dxB,
    const float *sigma_xx_r, const float *sigma_yy_r, const float *sigma_zz_r,
    const float *sigma_xy_r, const float *sigma_xz_r, const float *sigma_yz_r,
    const float *uxx_r, const float *uyy_r, const float *uzz_r,
    const float *uxy_r, const float *uxz_r, const float *uyz_r,
    // Optimization(4): eigenstrain 参数已移除
    double S_p_11, double S_p_12, double S_p_13, double S_p_14, double S_p_15, double S_p_16,
    double S_p_22, double S_p_23, double S_p_24, double S_p_25, double S_p_26,
    double S_p_33, double S_p_34, double S_p_35, double S_p_36,
    double S_p_44, double S_p_45, double S_p_46, double S_p_55, double S_p_56, double S_p_66,
    double eps_xx00, double eps_yy00, double eps_zz00,
    double eps_yz00, double eps_xz00, double eps_xy00,
    double eps_iso_over_vB, double elastic_shift_dimless,
    int total_size, int elastic_enabled) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_f_phi_bulk_dgel_dphi_kernel<<<blocks, threads>>>(
        phi_r, xB_r, f_phi_bulk_r, dgel_dphi_r,
        temperature_K, mu_reference_scale, v_A, v_B, mu0_compound,
        Vm_compound, Vm_alpha_0, dVm_alpha_dxB,
        sigma_xx_r, sigma_yy_r, sigma_zz_r, sigma_xy_r, sigma_xz_r, sigma_yz_r,
        uxx_r, uyy_r, uzz_r, uxy_r, uxz_r, uyz_r,
        // Optimization(4): eigenstrain 参数已移除
        S_p_11, S_p_12, S_p_13, S_p_14, S_p_15, S_p_16,
        S_p_22, S_p_23, S_p_24, S_p_25, S_p_26,
        S_p_33, S_p_34, S_p_35, S_p_36,
        S_p_44, S_p_45, S_p_46, S_p_55, S_p_56, S_p_66,
        eps_xx00, eps_yy00, eps_zz00, eps_yz00, eps_xz00, eps_xy00,
        eps_iso_over_vB, elastic_shift_dimless, total_size, elastic_enabled);
}

// Optimization: 直接计算 driving_force，避免 4 场同时驻留
void launch_compute_driving_force_direct_kernel(
    const double *phi_r, const double *xB_r, const double *f_phi_grad_r,
    double *driving_force_r,
    double temperature_K, double mu_reference_scale,
    double v_A, double v_B, double mu0_compound,
    double Vm_compound, double Vm_alpha_0, double dVm_alpha_dxB,
    double W, double elastic_shift_dimless,
    const float *sigma_xx_r, const float *sigma_yy_r, const float *sigma_zz_r,
    const float *sigma_xy_r, const float *sigma_xz_r, const float *sigma_yz_r,
    const float *uxx_r, const float *uyy_r, const float *uzz_r,
    const float *uxy_r, const float *uxz_r, const float *uyz_r,
    // Optimization(4): eigenstrain 参数已移除
    double S_p_11, double S_p_12, double S_p_13, double S_p_14, double S_p_15, double S_p_16,
    double S_p_22, double S_p_23, double S_p_24, double S_p_25, double S_p_26,
    double S_p_33, double S_p_34, double S_p_35, double S_p_36,
    double S_p_44, double S_p_45, double S_p_46, double S_p_55, double S_p_56, double S_p_66,
    double eps_xx00, double eps_yy00, double eps_zz00,
    double eps_yz00, double eps_xz00, double eps_xy00,
    double eps_iso_over_vB,
    int total_size, int elastic_enabled) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_driving_force_direct_kernel<<<blocks, threads>>>(
        phi_r, xB_r, f_phi_grad_r, driving_force_r,
        temperature_K, mu_reference_scale, v_A, v_B, mu0_compound,
        Vm_compound, Vm_alpha_0, dVm_alpha_dxB, W, elastic_shift_dimless,
        sigma_xx_r, sigma_yy_r, sigma_zz_r, sigma_xy_r, sigma_xz_r, sigma_yz_r,
        uxx_r, uyy_r, uzz_r, uxy_r, uxz_r, uyz_r,
        // Optimization(4): eigenstrain 参数已移除
        S_p_11, S_p_12, S_p_13, S_p_14, S_p_15, S_p_16,
        S_p_22, S_p_23, S_p_24, S_p_25, S_p_26,
        S_p_33, S_p_34, S_p_35, S_p_36,
        S_p_44, S_p_45, S_p_46, S_p_55, S_p_56, S_p_66,
        eps_xx00, eps_yy00, eps_zz00, eps_yz00, eps_xz00, eps_xy00,
        eps_iso_over_vB, total_size, elastic_enabled);
}

void launch_scale_array_kernel(double *arr, double factor, int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    scale_array_kernel<<<blocks, threads>>>(arr, factor, total_size);
}

void launch_add_arrays_kernel(const double *a, const double *b, double *out, int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    add_arrays_kernel<<<blocks, threads>>>(a, b, out, total_size);
}

void launch_subtract_scaled_kernel(double *rhs_r, const double *lap_r, double scale, int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    subtract_scaled_kernel<<<blocks, threads>>>(rhs_r, lap_r, scale, total_size);
}

void launch_sum_driving_force_kernel(const double *dw, const double *bulk,
                                     const double *grad, const double *el,
                                     double *out, int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    sum_driving_force_kernel<<<blocks, threads>>>(dw, bulk, grad, el, out, total_size);
}

void launch_compute_gel_density_kernel(
    const float *uxx_r, const float *uyy_r, const float *uzz_r,
    const float *uxy_r, const float *uxz_r, const float *uyz_r,
    // Optimization(4): eigenstrain 参数已移除，现场计算；需要 phi 和 xB
    const double *phi_r,
    const double *eta_r,
    const double *xB_r,  // 可为 NULL（minimize 模式）
    const float *sigma_xx_r, const float *sigma_yy_r, const float *sigma_zz_r,
    const float *sigma_xy_r, const float *sigma_xz_r, const float *sigma_yz_r,
    // Optimization(4): 需要 eps0 相关参数
    float eps_xx00, float eps_yy00, float eps_zz00,
    float eps_yz00, float eps_xz00, float eps_xy00,
    double eps_iso_over_vB,
    int gp_mode_enabled,
    int gp_elastic_enabled,
    double gp_eps_iso,
    double *gel_hat_r,
    int total_size)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_gel_density_kernel<<<blocks, threads>>>(
        uxx_r, uyy_r, uzz_r, uxy_r, uxz_r, uyz_r,
        phi_r, eta_r, xB_r,
        sigma_xx_r, sigma_yy_r, sigma_zz_r, sigma_xy_r, sigma_xz_r, sigma_yz_r,
        eps_xx00, eps_yy00, eps_zz00, eps_yz00, eps_xz00, eps_xy00, eps_iso_over_vB,
        gp_mode_enabled, gp_elastic_enabled, gp_eps_iso,
        gel_hat_r,
        total_size);
}

void launch_compute_gp_elastic_stats_kernel(
    const double *phi_r,
    const double *eta_r,
    const double *gel_hat_r,
    const float *sigma_xx_r,
    const float *sigma_yy_r,
    const float *sigma_zz_r,
    double gp_eps_iso,
    double *stats,
    int total_size)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_gp_elastic_stats_kernel<<<blocks, threads>>>(
        phi_r, eta_r, gel_hat_r,
        sigma_xx_r, sigma_yy_r, sigma_zz_r,
        gp_eps_iso, stats, total_size);
}

void launch_compute_gradient_k_kernel(const cuDoubleComplex *f_k,
                                       cuDoubleComplex *grad_x_k,
                                       cuDoubleComplex *grad_y_k,
                                       cuDoubleComplex *grad_z_k,
                                       int Nx, int Ny, int Nz, int NzC,
                                       double dx, double dy, double dz, int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_gradient_k_kernel<<<blocks, threads>>>(f_k, grad_x_k, grad_y_k, grad_z_k,
                                                    Nx, Ny, Nz, NzC, dx, dy, dz, total_size);
}

void launch_compute_flux_kernel(const double *grad_mu_x_r, const double *grad_mu_y_r,
                                 const double *grad_mu_z_r, const double *phi_r,
                                 const double *xB_prev_r, double *Jx_r, double *Jy_r,
                                 double *Jz_r, double D_alpha, double D_compound,
                                 double Vm_alpha_0, double dVm_alpha_dxB,
                                 double Vm_compound, double temperature_K,
                                 double mu_reference_scale, int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_flux_kernel<<<blocks, threads>>>(grad_mu_x_r, grad_mu_y_r, grad_mu_z_r,
                                              phi_r, xB_prev_r, Jx_r, Jy_r, Jz_r,
                                              D_alpha, D_compound, Vm_alpha_0,
                                              dVm_alpha_dxB, Vm_compound,
                                              temperature_K, mu_reference_scale,
                                              total_size);
}

void launch_compute_divergence_k_kernel(const cuDoubleComplex *Jx_k,
                                         const cuDoubleComplex *Jy_k,
                                         const cuDoubleComplex *Jz_k,
                                         cuDoubleComplex *divJ_k,
                                         int Nx, int Ny, int Nz, int NzC,
                                         double dx, double dy, double dz, int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_divergence_k_kernel<<<blocks, threads>>>(Jx_k, Jy_k, Jz_k, divJ_k,
                                                      Nx, Ny, Nz, NzC, dx, dy, dz, total_size);
}

// Optimization: launch wrappers for divJ serial accumulation
void launch_compute_gradient_single_component_k_kernel(
    const cuDoubleComplex *f_k,
    cuDoubleComplex *grad_alpha_k,
    int alpha,
    int Nx, int Ny, int Nz, int NzC,
    double dx, double dy, double dz,
    int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_gradient_single_component_k_kernel<<<blocks, threads>>>(
        f_k, grad_alpha_k, alpha, Nx, Ny, Nz, NzC, dx, dy, dz, total_size);
}
void launch_compute_flux_single_component_kernel(
    const double *grad_mu_alpha_r,
    const double *phi_r,
    const double *xB_prev_r,
    double *J_alpha_r,
    double D_alpha, double D_compound,
    double Vm_alpha_0, double dVm_alpha_dxB,
    double Vm_compound, double temperature_K,
    double mu_reference_scale,
    int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_flux_single_component_kernel<<<blocks, threads>>>(
        grad_mu_alpha_r, phi_r, xB_prev_r, J_alpha_r,
        D_alpha, D_compound, Vm_alpha_0, dVm_alpha_dxB,
        Vm_compound, temperature_K, mu_reference_scale, total_size);
}
void launch_compute_flux_single_component_q_kernel(
    const double *grad_mu_alpha_r, const double *phi_r,
    const double *xB_prev_r, double *J_alpha_r,
    double D_alpha, double Vm_alpha_0, double dVm_alpha_dxB,
    double Vm_compound, double temperature_K,
    double mu_reference_scale, int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_flux_single_component_q_kernel<<<blocks, threads>>>(
        grad_mu_alpha_r, phi_r, xB_prev_r, J_alpha_r,
        D_alpha, Vm_alpha_0, dVm_alpha_dxB, Vm_compound,
        temperature_K, mu_reference_scale, total_size);
}
void launch_compute_flux_single_component_gp_kernel(
    const double *grad_mu_alpha_r,
    const double *phi_r,
    const double *eta_r,
    const double *xB_alpha_prev_r,
    double *J_alpha_r,
    double D_alpha, double gp_M_GP, double gp_M_beta,
    double Vm_alpha_0, double dVm_alpha_dxB,
    double Vm_compound, double temperature_K,
    double mu_reference_scale,
    int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_flux_single_component_gp_kernel<<<blocks, threads>>>(
        grad_mu_alpha_r, phi_r, eta_r, xB_alpha_prev_r, J_alpha_r,
        D_alpha, gp_M_GP, gp_M_beta,
        Vm_alpha_0, dVm_alpha_dxB,
        Vm_compound, temperature_K, mu_reference_scale,
        total_size);
}
void launch_compute_gp_transport_stats_kernel(const double *phi_r,
                                              const double *eta_r,
                                              const double *xB_alpha_r,
                                              double xB_ref,
                                              double D_alpha,
                                              double gp_M_GP,
                                              double gp_M_beta,
                                              double Vm_alpha_0,
                                              double dVm_alpha_dxB,
                                              double Vm_compound,
                                              double temperature_K,
                                              double mu_reference_scale,
                                              double *stats,
                                              int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_gp_transport_stats_kernel<<<blocks, threads>>>(
        phi_r, eta_r, xB_alpha_r, xB_ref, D_alpha, gp_M_GP, gp_M_beta,
        Vm_alpha_0, dVm_alpha_dxB, Vm_compound, temperature_K, mu_reference_scale,
        stats, total_size);
}
void launch_divJ_accumulate_kernel(
    const cuDoubleComplex *J_alpha_k,
    cuDoubleComplex *divJ_k,
    int alpha,
    int Nx, int Ny, int Nz, int NzC,
    double dx, double dy, double dz,
    int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    divJ_accumulate_kernel<<<blocks, threads>>>(
        J_alpha_k, divJ_k, alpha, Nx, Ny, Nz, NzC, dx, dy, dz, total_size);
}
void launch_zero_divJ_k_kernel(cuDoubleComplex *divJ_k, int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    zero_divJ_k_kernel<<<blocks, threads>>>(divJ_k, total_size);
}

void launch_compute_Y_rhs_kernel(const double *divJ_r, const double *phi_r,
                                  const double *phi_prev, const double *lapY_r,
                                  const double *Y_r, const double *dY_dt_prev,
                                  double *rhs_r, double dt, double v_B,
                                  double mean_DY, int total_size,
                                  int use_previous_time_level,
                                  int disable_gamma_term,
                                  double term_h_scale,
                                  double *diag_stats) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_Y_rhs_kernel<<<blocks, threads>>>(divJ_r, phi_r, phi_prev, lapY_r,
                                               Y_r, dY_dt_prev, rhs_r, dt, v_B,
                                               mean_DY, total_size,
                                               use_previous_time_level,
                                               disable_gamma_term,
                                               term_h_scale,
                                               diag_stats);
}

void launch_compute_Y_rhs_gp_kernel(const double *divJ_r, const double *phi_r,
                                     const double *phi_prev, const double *eta_r,
                                     const double *eta_prev_r, const double *lapY_r,
                                     const double *Y_r, const double *dY_dt_prev,
                                     double *rhs_r, double dt, double xB_GP,
                                     double mean_DY, double gp_h_alpha_eps,
                                     int total_size,
                                     int use_previous_time_level,
                                     int disable_gamma_term,
                                     double *diag_stats)
{
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    compute_Y_rhs_gp_kernel<<<blocks, threads>>>(divJ_r, phi_r, phi_prev,
                                                  eta_r, eta_prev_r, lapY_r,
                                                  Y_r, dY_dt_prev, rhs_r, dt,
                                                  xB_GP, mean_DY, gp_h_alpha_eps,
                                                  total_size,
                                                  use_previous_time_level,
                                                  disable_gamma_term,
                                                  diag_stats);
}

void launch_compute_Y_rhs_gp_conservative_kernel(const double *divJ_r, const double *phi_r,
                                                 const double *phi_prev, const double *eta_r,
                                                 const double *eta_prev_r, const double *lapY_r,
                                                 const double *Y_r, const double *dY_dt_prev,
                                                 double *rhs_r, double dt, double xB_GP,
                                                 double mean_DY, int total_size,
                                                 int use_previous_time_level,
                                                 int disable_gamma_term,
                                                 double *diag_stats)
{
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    compute_Y_rhs_gp_conservative_kernel<<<blocks, threads>>>(
        divJ_r, phi_r, phi_prev, eta_r, eta_prev_r, lapY_r, Y_r, dY_dt_prev,
        rhs_r, dt, xB_GP, mean_DY, total_size, use_previous_time_level,
        disable_gamma_term, diag_stats);
}

void launch_Y_semi_implicit_update_kernel(const cuDoubleComplex *Y_k_old,
                                           const cuDoubleComplex *rhs_k,
                                           const double *k2,
                                           cuDoubleComplex *Y_k_new,
                                           double mean_DY, double dt, int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    Y_semi_implicit_update_kernel<<<blocks, threads>>>(Y_k_old, rhs_k, k2,
                                                        Y_k_new, mean_DY, dt, total_size);
}

void launch_Y_normalize_and_clamp_kernel(double *Y_r, double *xB_r,
                                          double invN, double Y_clip, double Y_upper_cap, double xB_eps,
                                          int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    Y_normalize_and_clamp_kernel<<<blocks, threads>>>(Y_r, xB_r, invN, Y_clip, Y_upper_cap, xB_eps, total_size);
}

void launch_relax_dY_dt_guess_kernel(const double *dY_dt_new_r,
                                     const double *dY_dt_old_guess_r,
                                     double *dY_dt_guess_r,
                                     double omega,
                                     int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    relax_dY_dt_guess_kernel<<<blocks, threads>>>(
        dY_dt_new_r, dY_dt_old_guess_r, dY_dt_guess_r, omega, total_size);
}

void launch_compute_diff_sumsq_max_kernel(const double *a_r,
                                          const double *b_r,
                                          double *stats,
                                          int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_diff_sumsq_max_kernel<<<blocks, threads>>>(a_r, b_r, stats, total_size);
}

void launch_Y_mass_diagnostics_kernel(const double *Y_ifft_r,
                                      const double *phi_r,
                                      const double *xB_before_r,
                                      double *stats,
                                      double invN,
                                      double Y_clip,
                                      double Y_upper_cap,
                                      double xB_eps,
                                      double v_B,
                                      int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    Y_mass_diagnostics_kernel<<<blocks, threads>>>(
        Y_ifft_r, phi_r, xB_before_r, stats, invN, Y_clip, Y_upper_cap, xB_eps, v_B, total_size);
}

void launch_Y_mass_diagnostics_gp_kernel(const double *Y_ifft_r,
                                         const double *phi_r,
                                         const double *eta_r,
                                         const double *xB_before_r,
                                         double *stats,
                                         double invN,
                                         double Y_clip,
                                         double Y_upper_cap,
                                         double xB_eps,
                                         double xB_GP,
                                         int total_size)
{
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    Y_mass_diagnostics_gp_kernel<<<blocks, threads>>>(
        Y_ifft_r, phi_r, eta_r, xB_before_r, stats, invN,
        Y_clip, Y_upper_cap, xB_eps, xB_GP, total_size);
}

void launch_gp_storage_exact_Y_update_kernel(const double *divJ_r,
                                             const double *phi_new_r,
                                             const double *phi_old_r,
                                             const double *eta_new_r,
                                             const double *eta_old_r,
                                             const double *Y_old_r,
                                             double *Y_r,
                                             double *xB_r,
                                             double dt,
                                             double xB_GP,
                                             double Y_clip,
                                             double Y_upper_cap,
                                             double xB_eps,
                                             double gp_h_alpha_eps,
                                             int disable_internal_clip,
                                             double *update_stats,
                                             int total_size)
{
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    gp_storage_exact_Y_update_kernel<<<blocks, threads>>>(
        divJ_r, phi_new_r, phi_old_r, eta_new_r, eta_old_r, Y_old_r,
        Y_r, xB_r, dt, xB_GP, Y_clip, Y_upper_cap, xB_eps, gp_h_alpha_eps,
        disable_internal_clip,
        update_stats, total_size);
}

void launch_two_phase_storage_exact_Y_update_kernel(const double *divJ_r,
                                                    const double *phi_new_r,
                                                    const double *phi_old_r,
                                                    const double *Y_old_r,
                                                    double *Y_r,
                                                    double *xB_r,
                                                    double dt,
                                                    double v_B,
                                                    double Y_clip,
                                                    double Y_upper_cap,
                                                    double xB_eps,
                                                    double h_alpha_eps,
                                                    double *update_stats,
                                                    int total_size)
{
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    two_phase_storage_exact_Y_update_kernel<<<blocks, threads>>>(
        divJ_r, phi_new_r, phi_old_r, Y_old_r, Y_r, xB_r, dt, v_B,
        Y_clip, Y_upper_cap, xB_eps, h_alpha_eps, update_stats, total_size);
}

void launch_compute_xB_storage_rhs_kernel(const double *divJ_r,
                                          const double *phi_new_r,
                                          const double *phi_old_r,
                                          const double *lap_xB_r,
                                          const double *xB_old_r,
                                          double *rhs_xB_r,
                                          double dt,
                                          double v_B,
                                          double mean_Dx,
                                          double h_alpha_eps,
                                          int total_size)
{
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    compute_xB_storage_rhs_kernel<<<blocks, threads>>>(
        divJ_r, phi_new_r, phi_old_r, lap_xB_r, xB_old_r, rhs_xB_r,
        dt, v_B, mean_Dx, h_alpha_eps, total_size);
}

void launch_xB_normalize_clamp_and_logit_kernel(double *xB_ifft_r,
                                                double *xB_r,
                                                double *Y_r,
                                                double invN,
                                                double xB_eps,
                                                double Y_clip,
                                                double Y_upper_cap,
                                                int total_size)
{
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    xB_normalize_clamp_and_logit_kernel<<<blocks, threads>>>(
        xB_ifft_r, xB_r, Y_r, invN, xB_eps, Y_clip, Y_upper_cap, total_size);
}

void launch_initialize_q_alpha_kernel(const double *phi_r,
                                      const double *xB_r,
                                      double *q_alpha_r,
                                      int total_size)
{
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    initialize_q_alpha_kernel<<<blocks, threads>>>(phi_r, xB_r, q_alpha_r, total_size);
}

void launch_apply_local_phase_storage_transfer_q_kernel(
    const double *phi_new_r, const double *phi_old_r,
    double *q_alpha_r, double v_B, double feasibility_tol,
    double *stats, int total_size)
{
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    apply_local_phase_storage_transfer_q_kernel<<<blocks, threads>>>(
        phi_new_r, phi_old_r, q_alpha_r, v_B, feasibility_tol, stats, total_size);
}

void launch_compute_q_transport_rhs_kernel(const double *divJ_r,
                                           const double *lap_q_r,
                                           double *rhs_q_r,
                                           double mean_Dq,
                                           int total_size)
{
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    compute_q_transport_rhs_kernel<<<blocks, threads>>>(
        divJ_r, lap_q_r, rhs_q_r, mean_Dq, total_size);
}

void launch_q_normalize_validate_and_reconstruct_kernel(
    double *q_ifft_r, double *q_alpha_r, const double *phi_r,
    double *xB_r, double *Y_r, double invN, double xB_eps,
    double Y_clip, double Y_upper_cap, double bound_tol,
    double *stats, int total_size)
{
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    q_normalize_validate_and_reconstruct_kernel<<<blocks, threads>>>(
        q_ifft_r, q_alpha_r, phi_r, xB_r, Y_r, invN, xB_eps,
        Y_clip, Y_upper_cap, bound_tol, stats, total_size);
}

void launch_q_explicit_transport_and_reconstruct_kernel(
    double *q_alpha_r, const double *divJ_r, const double *phi_r,
    double *xB_r, double *Y_r, double dt, double xB_eps,
    double Y_clip, double Y_upper_cap, double bound_tol,
    double *stats, int total_size)
{
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    q_explicit_transport_and_reconstruct_kernel<<<blocks, threads>>>(
        q_alpha_r, divJ_r, phi_r, xB_r, Y_r, dt, xB_eps,
        Y_clip, Y_upper_cap, bound_tol, stats, total_size);
}

void launch_sync_q_from_phi_xB_kernel(const double *phi_r,
                                      const double *xB_r,
                                      double *q_alpha_r,
                                      int total_size)
{
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    sync_q_from_phi_xB_kernel<<<blocks, threads>>>(
        phi_r, xB_r, q_alpha_r, total_size);
}

void launch_initialize_conservative_storage_kernel(
    const double *phi_r, const double *xB_r, double *storage_r,
    double v_B, int primary_is_ctot, int total_size)
{
    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;
    initialize_conservative_storage_kernel<<<blocks, threads>>>(
        phi_r, xB_r, storage_r, v_B, primary_is_ctot, total_size);
}

void launch_reconstruct_conservative_context_kernel(
    const double *storage_r, const double *phi_r, double *xB_r, double *Y_r,
    double temperature_K, double v_B, int primary_is_ctot,
    double beta_support_eps, double xB_eps, double Y_clip,
    double *stats, int total_size)
{
    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;
    reconstruct_conservative_context_kernel<<<blocks, threads>>>(
        storage_r, phi_r, xB_r, Y_r, temperature_K, v_B,
        primary_is_ctot, beta_support_eps, xB_eps, Y_clip, stats, total_size);
}

void launch_pairwise_conservative_face_sweep_kernel(
    double *storage_r, const double *mu_r, const double *phi_r,
    const double *xB_r, int Nx, int Ny, int Nz, int axis, int parity,
    double spacing, double dt, double D_alpha,
    double Vm_alpha_0, double dVm_alpha_dxB, double Vm_compound,
    double temperature_K, double mu_reference_scale,
    double v_B, int primary_is_ctot, int backward_euler,
    double bound_tol, double *stats, int total_size)
{
    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;
    pairwise_conservative_face_sweep_kernel<<<blocks, threads>>>(
        storage_r, mu_r, phi_r, xB_r, Nx, Ny, Nz, axis, parity,
        spacing, dt, D_alpha, Vm_alpha_0, dVm_alpha_dxB, Vm_compound,
        temperature_K, mu_reference_scale, v_B, primary_is_ctot,
        backward_euler, bound_tol, stats, total_size);
}

void launch_constrain_phase_and_reconstruct_conservative_kernel(
    double *phi_new_r, const double *phi_old_r, double *storage_r,
    double *xB_r, double *Y_r, double temperature_K, double v_B,
    int primary_is_ctot, double beta_support_eps, double xB_eps,
    double Y_clip, double bound_tol, double *stats, int total_size)
{
    const int threads = 256;
    const int blocks = (total_size + threads - 1) / threads;
    constrain_phase_and_reconstruct_conservative_kernel<<<blocks, threads>>>(
        phi_new_r, phi_old_r, storage_r, xB_r, Y_r, temperature_K, v_B,
        primary_is_ctot, beta_support_eps, xB_eps, Y_clip, bound_tol,
        stats, total_size);
}

void launch_apply_Y_shift_recompute_xB_kernel(const double *Y_base_r,
                                              double *Y_r,
                                              double *xB_r,
                                              double lambda_shift,
                                              double Y_clip,
                                              double Y_upper_cap,
                                              double xB_eps,
                                              int total_size)
{
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    apply_Y_shift_recompute_xB_kernel<<<blocks, threads>>>(
        Y_base_r, Y_r, xB_r, lambda_shift, Y_clip, Y_upper_cap, xB_eps, total_size);
}

void launch_gp_picard_storage_Y_update_kernel(const double *divJ_r,
                                              const double *phi_new_r,
                                              const double *phi_old_r,
                                              const double *eta_new_r,
                                              const double *eta_old_r,
                                              const double *Y_old_r,
                                              double *Y_guess_r,
                                              double *xB_r,
                                              double dt,
                                              double xB_GP,
                                              double Y_clip,
                                              double Y_upper_cap,
                                              double xB_eps,
                                              double *update_stats,
                                              int total_size)
{
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    gp_picard_storage_Y_update_kernel<<<blocks, threads>>>(
        divJ_r, phi_new_r, phi_old_r, eta_new_r, eta_old_r, Y_old_r,
        Y_guess_r, xB_r, dt, xB_GP, Y_clip, Y_upper_cap, xB_eps,
        update_stats, total_size);
}

void launch_gp_storage_diagnostics_kernel(const double *phi_old_r,
                                          const double *eta_old_r,
                                          const double *xB_old_r,
                                          const double *phi_new_r,
                                          const double *eta_new_r,
                                          const double *xB_new_r,
                                          const double *divJ_r,
                                          double dt,
                                          double xB_GP,
                                          double gp_h_alpha_eps,
                                          double *stats,
                                          int total_size)
{
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    gp_storage_diagnostics_kernel<<<blocks, threads>>>(
        phi_old_r, eta_old_r, xB_old_r, phi_new_r, eta_new_r, xB_new_r,
        divJ_r, dt, xB_GP, gp_h_alpha_eps, stats, total_size);
}

void launch_clamp_xB_max_kernel(double *xB_r, int total_size, double xB_max) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    clamp_xB_max_kernel<<<blocks, threads>>>(xB_r, total_size, xB_max);
}

void launch_compute_laplacian_k_kernel(const cuDoubleComplex *f_k, const double *k2,
                                        cuDoubleComplex *lap_k, int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_laplacian_k_kernel<<<blocks, threads>>>(f_k, k2, lap_k, total_size);
}

void launch_compute_xBtot_kernel(const double *phi_r, const double *xB_r,
                                  double *xBtot_r, double v_B, int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_xBtot_kernel<<<blocks, threads>>>(phi_r, xB_r, xBtot_r, v_B, total_size);
}

void launch_compute_xBtot_gp_kernel(const double *phi_r, const double *eta_r,
                                    const double *xB_alpha_r, double *xBtot_gp_r,
                                    double xB_GP, int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_xBtot_gp_kernel<<<blocks, threads>>>(phi_r, eta_r, xB_alpha_r,
                                                 xBtot_gp_r, xB_GP, total_size);
}

void launch_compute_DY_values_kernel(const double *Y_r, const double *phi_r,
                                     double *DY_values, double D_alpha,
                                     double D_compound, int total_size) {
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_DY_values_kernel<<<blocks, threads>>>(Y_r, phi_r, DY_values,
                                                   D_alpha, D_compound, total_size);
}

// ============================================================================
// 额外统计：直接归约 h(phi)，避免为 h 场单独分配整块临时数组
// ============================================================================
__global__ void reduce_sum_h_kernel(
    const double *phi_r,
    double *block_sums,
    int total_size)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double val = 0.0;
    if (idx < total_size) {
        double phi = clamp01(phi_r[idx]);
        val = h_of_phi(phi);
    }
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

double gpu_compute_vf_from_h(const double *d_phi_r, int total_size) {
    if (total_size <= 0) return 0.0;

    int threads_per_block = 256;
    int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;
    double *d_block_sums = NULL;
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(double)));

    size_t shared_mem_size = threads_per_block * sizeof(double);
    reduce_sum_h_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        d_phi_r, d_block_sums, total_size);

    double sum_h = gpu_reduce_sum(d_block_sums, num_blocks);
    CUDA_CHECK(cudaFree(d_block_sums));
    return sum_h / (double)total_size;
}

// GPU归约求和函数（优化版本，适合大数组）
double gpu_reduce_sum(const double *d_array, int n) {
    if (n <= 0) return 0.0;
    
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    // 分配临时数组存储每个block的归约结果
    double *d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(double)));
    
    // 第一次归约：每个block归约
    size_t shared_mem_size = threads_per_block * sizeof(double);
    reduce_sum_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        d_array, d_block_sums, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 如果只有一个block，直接返回结果
    double result = 0.0;
    if (num_blocks == 1) {
        CUDA_CHECK(cudaMemcpy(&result, d_block_sums, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_block_sums));
        return result;
    }
    
    // 递归归约：对block结果再次归约
    int remaining = num_blocks;
    double *d_input = d_block_sums;
    
    while (remaining > 1) {
        int new_blocks = (remaining + threads_per_block - 1) / threads_per_block;
        
        double *d_output;
        CUDA_CHECK(cudaMalloc(&d_output, new_blocks * sizeof(double)));
        
        // 直接对 remaining 个元素做归约（现代 CUDA 的 1D gridDim.x 支持远超 65535）
        reduce_sum_kernel<<<new_blocks, threads_per_block, shared_mem_size>>>(
            d_input, d_output, remaining);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        if (d_input != d_block_sums) {
            CUDA_CHECK(cudaFree(d_input));
        }
        d_input = d_output;
        remaining = new_blocks;
    }
    
    CUDA_CHECK(cudaMemcpy(&result, d_input, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_block_sums));
    
    return result;
}

// ============================================================================
// 统计kernel：计算体积分数和平均半径相关统计量
// ============================================================================

__global__ void compute_diagnostics_stats_kernel(
    const double *phi_r,
    double *stats,  // 输出：前total_size个是N_in，后total_size个是N_if
                    // stats[0...total_size-1] = N_in (phi>0.5的点数)
                    // stats[total_size...2*total_size-1] = N_if (0.12<phi<0.88的点数)
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    double phi = phi_r[idx];
    
    // N_in: phi > 0.5的点数 (存储在stats的前半部分)
    double count_in = (phi > 0.5) ? 1.0 : 0.0;
    stats[idx] = count_in;
    
    // N_if: 0.12 < phi < 0.88的点数 (存储在stats的后半部分)
    double count_if = ((phi > 0.12) && (phi < 0.88)) ? 1.0 : 0.0;
    stats[idx + total_size] = count_if;
}

void launch_compute_diagnostics_stats_kernel(
    const double *phi_r,
    double *d_stats,
    int total_size)
{
    int threads_per_block = 256;
    int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;
    
    compute_diagnostics_stats_kernel<<<num_blocks, threads_per_block>>>(
        phi_r, d_stats, total_size);
}

// ============================================================================
// 优化函数：直接归约计算N_in和N_if（节省显存）
// ============================================================================

void gpu_reduce_sum_N_in_N_if(const double *phi_r, int total_size,
                               double *N_in_sum, double *N_if_sum)
{
    if (total_size <= 0) {
        *N_in_sum = 0.0;
        *N_if_sum = 0.0;
        return;
    }
    
    int threads_per_block = 256;
    int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;
    
    // 分配临时数组存储每个block的归约结果
    double *d_block_sums_N_in, *d_block_sums_N_if;
    CUDA_CHECK(cudaMalloc(&d_block_sums_N_in, num_blocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_sums_N_if, num_blocks * sizeof(double)));
    
    // 第一次归约：每个block归约
    size_t shared_mem_size = 2 * threads_per_block * sizeof(double);  // 存储N_in和N_if
    reduce_sum_N_in_N_if_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        phi_r, d_block_sums_N_in, d_block_sums_N_if, total_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 递归归约：对block结果再次归约（对N_in和N_if分别归约）
    double sum_N_in = gpu_reduce_sum(d_block_sums_N_in, num_blocks);
    double sum_N_if = gpu_reduce_sum(d_block_sums_N_if, num_blocks);
    
    CUDA_CHECK(cudaFree(d_block_sums_N_in));
    CUDA_CHECK(cudaFree(d_block_sums_N_if));
    
    *N_in_sum = sum_N_in;
    *N_if_sum = sum_N_if;
}

// ============================================================================
// 优化函数：直接归约计算xBtot的和（节省显存）
// ============================================================================

double gpu_reduce_sum_xBtot(const double *phi_r, const double *xB_r,
                            double v_B, int total_size)
{
    if (total_size <= 0) return 0.0;
    
    int threads_per_block = 256;
    int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;
    
    // 分配临时数组存储每个block的归约结果
    double *d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(double)));
    
    // 第一次归约：每个block归约
    size_t shared_mem_size = threads_per_block * sizeof(double);
    reduce_sum_xBtot_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        phi_r, xB_r, d_block_sums, v_B, total_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 递归归约：对block结果再次归约
    double result = gpu_reduce_sum(d_block_sums, num_blocks);
    
    CUDA_CHECK(cudaFree(d_block_sums));

    return result;
}

double gpu_reduce_sum_xBtot_gp(const double *phi_r, const double *eta_r,
                               const double *xB_alpha_r, double xB_GP,
                               int total_size)
{
    if (total_size <= 0) return 0.0;

    int threads_per_block = 256;
    int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;

    double *d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(double)));

    size_t shared_mem_size = threads_per_block * sizeof(double);
    reduce_sum_xBtot_gp_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        phi_r, eta_r, xB_alpha_r, d_block_sums, xB_GP, total_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    double result = gpu_reduce_sum(d_block_sums, num_blocks);
    CUDA_CHECK(cudaFree(d_block_sums));

    return result;
}

// ============================================================================
// 优化函数：直接归约计算DY的和（节省显存）
// ============================================================================

double gpu_reduce_sum_DY(const double *Y_r, const double *phi_r,
                        double D_alpha, double D_compound, int total_size)
{
    if (total_size <= 0) return 0.0;
    
    int threads_per_block = 256;
    int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;
    
    // 分配临时数组存储每个block的归约结果
    double *d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(double)));
    
    // 第一次归约：每个block归约
    size_t shared_mem_size = threads_per_block * sizeof(double);
    reduce_sum_DY_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        Y_r, phi_r, d_block_sums, D_alpha, D_compound, total_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 递归归约：对block结果再次归约
    double result = gpu_reduce_sum(d_block_sums, num_blocks);
    
    CUDA_CHECK(cudaFree(d_block_sums));
    
    return result;
}

double gpu_reduce_sum_DY_gp(const double *Y_r, const double *phi_r,
                            const double *eta_r, double D_alpha,
                            int total_size)
{
    if (total_size <= 0) {
        return 0.0;
    }

    int threads_per_block = 256;
    int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;
    size_t shared_mem_size = threads_per_block * sizeof(double);

    double *d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(double)));

    reduce_sum_DY_gp_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        Y_r, phi_r, eta_r, d_block_sums, D_alpha, total_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    double result = 0.0;
    if (num_blocks == 1) {
        CUDA_CHECK(cudaMemcpy(&result, d_block_sums, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_block_sums));
        return result;
    }

    int remaining = num_blocks;
    double *d_input = d_block_sums;

    while (remaining > 1) {
        int new_blocks = (remaining + threads_per_block - 1) / threads_per_block;

        double *d_output;
        CUDA_CHECK(cudaMalloc(&d_output, new_blocks * sizeof(double)));

        reduce_sum_kernel<<<new_blocks, threads_per_block, shared_mem_size>>>(
            d_input, d_output, remaining);
        CUDA_CHECK(cudaDeviceSynchronize());

        if (d_input != d_block_sums) {
            CUDA_CHECK(cudaFree(d_input));
        }
        d_input = d_output;
        remaining = new_blocks;
    }

    CUDA_CHECK(cudaMemcpy(&result, d_input, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_block_sums));

    return result;
}

// ============================================================================
// 优化函数：直接归约计算min和max（节省CPU内存和传输时间）
// ============================================================================

void gpu_reduce_min_max(const double *d_array, int n,
                       double *min_val, double *max_val)
{
    if (n <= 0) {
        *min_val = 0.0;
        *max_val = 0.0;
        return;
    }
    
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    // 分配临时数组存储每个block的归约结果
    double *d_block_mins, *d_block_maxs;
    CUDA_CHECK(cudaMalloc(&d_block_mins, num_blocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_maxs, num_blocks * sizeof(double)));
    
    // 第一次归约：每个block归约
    size_t shared_mem_size = 2 * threads_per_block * sizeof(double);  // 存储min和max
    reduce_min_max_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        d_array, d_block_mins, d_block_maxs, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 递归归约：对block结果再次归约
    // 对于min，需要使用特殊处理（找到最小值）
    // 对于max，需要使用特殊处理（找到最大值）
    // 由于block数量通常不多，可以直接复制到CPU处理
    if (num_blocks == 1) {
        CUDA_CHECK(cudaMemcpy(min_val, d_block_mins, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(max_val, d_block_maxs, sizeof(double), cudaMemcpyDeviceToHost));
    } else {
        // 创建临时数组用于CPU端归约
        double *h_block_mins = (double*)malloc(num_blocks * sizeof(double));
        double *h_block_maxs = (double*)malloc(num_blocks * sizeof(double));
        CUDA_CHECK(cudaMemcpy(h_block_mins, d_block_mins, num_blocks * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_block_maxs, d_block_maxs, num_blocks * sizeof(double), cudaMemcpyDeviceToHost));
        
        *min_val = h_block_mins[0];
        *max_val = h_block_maxs[0];
        for (int i = 1; i < num_blocks; i++) {
            if (h_block_mins[i] < *min_val) *min_val = h_block_mins[i];
            if (h_block_maxs[i] > *max_val) *max_val = h_block_maxs[i];
        }
        
        free(h_block_mins);
        free(h_block_maxs);
    }
    
    CUDA_CHECK(cudaFree(d_block_mins));
    CUDA_CHECK(cudaFree(d_block_maxs));
}

// ============================================================================
// Launch函数：更新dY_dt_prev
// ============================================================================

void launch_update_dY_dt_prev_kernel(
    const double *Y_r,
    const double *Y_n_saved,
    double *dY_dt_prev_r,
    double dt,
    int total_size)
{
    int threads_per_block = 256;
    int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;
    
    update_dY_dt_prev_kernel<<<num_blocks, threads_per_block>>>(
        Y_r, Y_n_saved, dY_dt_prev_r, dt, total_size);
}

// ============================================================================
// 优化函数：从grad_mu直接计算divJ，内部使用临时数组，节省显存
// ============================================================================

void compute_divJ_from_grad_mu_optimized(
    const double *grad_mu_x_r,
    const double *grad_mu_y_r,
    const double *grad_mu_z_r,
    const double *phi_r,
    const double *xB_prev_r,
    double *divJ_r,
    cufftHandle plan_r2c_xB,
    cufftHandle plan_c2r_xB,
    int Nx, int Ny, int Nz, int NzC,
    double dx, double dy, double dz,
    double invN,
    double D_alpha,
    double D_compound,
    double Vm_alpha_0,
    double dVm_alpha_dxB,
    double Vm_compound,
    double temperature_K,
    double mu_reference_scale,
    int total_r,
    int total_k)
{
    // 内部分配临时数组（计算完成后自动释放）
    double *d_Jx_temp = NULL;
    double *d_Jy_temp = NULL;
    double *d_Jz_temp = NULL;
    cufftDoubleComplex *d_Jx_k_temp = NULL;
    cufftDoubleComplex *d_Jy_k_temp = NULL;
    cufftDoubleComplex *d_Jz_k_temp = NULL;
    cufftDoubleComplex *d_divJ_k_temp = NULL;
    
    size_t size_r = total_r * sizeof(double);
    size_t size_k = total_k * sizeof(cufftDoubleComplex);
    
    CUDA_CHECK(cudaMalloc(&d_Jx_temp, size_r));
    CUDA_CHECK(cudaMalloc(&d_Jy_temp, size_r));
    CUDA_CHECK(cudaMalloc(&d_Jz_temp, size_r));
    CUDA_CHECK(cudaMalloc(&d_Jx_k_temp, size_k));
    CUDA_CHECK(cudaMalloc(&d_Jy_k_temp, size_k));
    CUDA_CHECK(cudaMalloc(&d_Jz_k_temp, size_k));
    CUDA_CHECK(cudaMalloc(&d_divJ_k_temp, size_k));
    
    // 2.5 计算通量
    launch_compute_flux_kernel(grad_mu_x_r, grad_mu_y_r, grad_mu_z_r,
                              phi_r, xB_prev_r,
                              d_Jx_temp, d_Jy_temp, d_Jz_temp,
                              D_alpha, D_compound,
                              Vm_alpha_0, dVm_alpha_dxB,
                              Vm_compound, temperature_K,
                              mu_reference_scale, total_r);
    
    // 2.6 通量变换到k空间并计算散度
    CUFFT_CHECK(cufftExecD2Z(plan_r2c_xB, d_Jx_temp, d_Jx_k_temp));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c_xB, d_Jy_temp, d_Jy_k_temp));
    CUFFT_CHECK(cufftExecD2Z(plan_r2c_xB, d_Jz_temp, d_Jz_k_temp));
    
    launch_dealias_kernel(d_Jx_k_temp, Nx, Ny, Nz, NzC, dx, dy, dz, total_k);
    launch_dealias_kernel(d_Jy_k_temp, Nx, Ny, Nz, NzC, dx, dy, dz, total_k);
    launch_dealias_kernel(d_Jz_k_temp, Nx, Ny, Nz, NzC, dx, dy, dz, total_k);
    
    launch_compute_divergence_k_kernel(d_Jx_k_temp, d_Jy_k_temp, d_Jz_k_temp, d_divJ_k_temp,
                                      Nx, Ny, Nz, NzC, dx, dy, dz, total_k);
    
    // 2.7 反变换散度
    CUFFT_CHECK(cufftExecZ2D(plan_c2r_xB, d_divJ_k_temp, divJ_r));
    launch_normalize_only_kernel(divJ_r, invN, total_r);
    
    // 释放临时数组
    CUDA_CHECK(cudaFree(d_Jx_temp));
    CUDA_CHECK(cudaFree(d_Jy_temp));
    CUDA_CHECK(cudaFree(d_Jz_temp));
    CUDA_CHECK(cudaFree(d_Jx_k_temp));
    CUDA_CHECK(cudaFree(d_Jy_k_temp));
    CUDA_CHECK(cudaFree(d_Jz_k_temp));
    CUDA_CHECK(cudaFree(d_divJ_k_temp));
}

// ============================================================================
// 初始化kernel：并行初始化phi场
// ============================================================================

// Device版本的periodic_delta函数
__device__ static inline double periodic_delta_device(double coord, double center, double period) {
    double d = coord - center;
    d -= rint(d / period) * period;
    return d;
}

// 3D初始化kernel
__global__ void initialize_phi_3d_kernel(
    double *phi_r,
    const double *centers,  // 种子中心数组 [cx, cy, cz, ...] (3*N_seeds个元素)
    int N_seeds,
    int Nx, int Ny, int Nz,
    double dx, double dy, double dz,
    double Lx, double Ly, double Lz,
    double R, double w,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    // 计算当前点的3D坐标
    int k = idx % Nz;
    int j = (idx / Nz) % Ny;
    int i = idx / (Ny * Nz);
    
    double x = i * dx;
    double y = j * dy;
    double z = k * dz;
    
    // 计算所有种子的最大值
    double max_phi = 0.0;
    for (int s = 0; s < N_seeds; s++) {
        double cx = centers[3*s + 0];
        double cy = centers[3*s + 1];
        double cz = centers[3*s + 2];
        
        double rx = periodic_delta_device(x, cx, Lx);
        double ry = periodic_delta_device(y, cy, Ly);
        double rz = periodic_delta_device(z, cz, Lz);
        
        double r = sqrt(rx*rx + ry*ry + rz*rz);
        double seed = 0.5 * (1.0 + tanh((R - r) / w));
        if (seed > max_phi) max_phi = seed;
    }
    
    phi_r[idx] = clamp01(max_phi);
}

// 2D初始化kernel（Ny=2的情况）
__global__ void initialize_phi_2d_kernel(
    double *phi_r,
    const double *centers,  // 种子中心数组 [cx, cz, ...] (只使用x和z坐标)
    int N_seeds,
    int Nx, int Ny, int Nz,
    double dx, double dz,
    double Lx, double Lz,
    double R, double w,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    // 计算当前点的2D坐标（忽略y）
    int k = idx % Nz;
    int j = (idx / Nz) % Ny;
    int i = idx / (Ny * Nz);
    
    // 只对j=0层计算，然后复制到所有y层
    if (j == 0) {
        double x = i * dx;
        double z = k * dz;
        
        double max_phi = 0.0;
        for (int s = 0; s < N_seeds; s++) {
            double cx = centers[3*s + 0];
            double cz = centers[3*s + 2];
            
            double rx = periodic_delta_device(x, cx, Lx);
            double rz = periodic_delta_device(z, cz, Lz);
            
            double r = sqrt(rx*rx + rz*rz);
            double seed = 0.5 * (1.0 + tanh((R - r) / w));
            if (seed > max_phi) max_phi = seed;
        }
        
        max_phi = clamp01(max_phi);
        
        // 写入所有y层
        for (int jj = 0; jj < Ny; jj++) {
            int idx_all = (i * Ny + jj) * Nz + k;
            phi_r[idx_all] = max_phi;
        }
    }
}

void launch_initialize_phi_kernel(
    double *d_phi_r,
    const double *d_centers,
    int N_seeds,
    int Nx, int Ny, int Nz,
    double dx, double dy, double dz,
    double Lx, double Ly, double Lz,
    double R, double w,
    int total_size)
{
    int threads_per_block = 256;
    int num_blocks = (total_size + threads_per_block - 1) / threads_per_block;
    
    if (Ny > 2) {
        // 3D模式
        initialize_phi_3d_kernel<<<num_blocks, threads_per_block>>>(
            d_phi_r, d_centers, N_seeds,
            Nx, Ny, Nz, dx, dy, dz, Lx, Ly, Lz,
            R, w, total_size);
    } else {
        // 2D模式
        initialize_phi_2d_kernel<<<num_blocks, threads_per_block>>>(
            d_phi_r, d_centers, N_seeds,
            Nx, Ny, Nz, dx, dz, Lx, Lz,
            R, w, total_size);
    }
}

// ============================================================================
// 弹性相关kernel实现
// ============================================================================

// ============================================================================
// 1. Eigenstrain计算kernel（从phi, xB 计算 Wu & Ji 形式的 eigenstrain）
// 注意：这些 kernel 现在仅用于临时写入 d_uxx_r..d_uyz_r（Optimization(4)）
// ============================================================================

__global__ void compute_eigenstrain_from_phi_kernel(
    const double *phi_r,
    const double *eta_r,
    const double *xB_r,
    float *uxx0_r, float *uyy0_r, float *uzz0_r,
    float *uxy0_r, float *uxz0_r, float *uyz0_r,
    // stress-free transformation strain eps^00_ij（Voigt顺序）
    float eps_xx00, float eps_yy00, float eps_zz00,
    float eps_yz00, float eps_xz00, float eps_xy00,
    // 各向同性化学膨胀参数：eps_iso_over_vB = ε_iso / v_B（常数）
    double eps_iso_over_vB,
    int gp_mode_enabled,
    int gp_elastic_enabled,
    double gp_eps_iso,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    double phi = phi_r[idx];
    double eta = eta_r ? eta_r[idx] : 0.0;
    double xB  = clamp01(xB_r[idx]);
    float eps_xx0, eps_yy0, eps_zz0;
    float eps_xy0, eps_xz0, eps_yz0;
    eigenstrain_phi_eta_xB_point(phi, eta, xB,
                                 eps_xx00, eps_yy00, eps_zz00,
                                 eps_yz00, eps_xz00, eps_xy00,
                                 eps_iso_over_vB,
                                 gp_mode_enabled, gp_elastic_enabled, gp_eps_iso,
                                 eps_xx0, eps_yy0, eps_zz0,
                                 eps_xy0, eps_xz0, eps_yz0);

    uxx0_r[idx] = eps_xx0;
    uyy0_r[idx] = eps_yy0;
    uzz0_r[idx] = eps_zz0;
    uyz0_r[idx] = eps_yz0;
    uxz0_r[idx] = eps_xz0;
    uxy0_r[idx] = eps_xy0;
}

// Minimize mode only: eigenstrain from phi only, no xB.
// eigenstrain(phi) = h(phi) * eps^00_ij (matrix = 0).
__global__ void compute_eigenstrain_from_phi_only_kernel(
    const double *phi_r,
    float *uxx0_r, float *uyy0_r, float *uzz0_r,
    float *uxy0_r, float *uxz0_r, float *uyz0_r,
    float eps_xx00, float eps_yy00, float eps_zz00,
    float eps_yz00, float eps_xz00, float eps_xy00,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    double phi = phi_r[idx];
    double h   = h_of_phi(phi);
    double eps_xx00_d = (double)eps_xx00;
    double eps_yy00_d = (double)eps_yy00;
    double eps_zz00_d = (double)eps_zz00;
    double eps_yz00_d = (double)eps_yz00;
    double eps_xz00_d = (double)eps_xz00;
    double eps_xy00_d = (double)eps_xy00;
    uxx0_r[idx] = (float)(h * eps_xx00_d);
    uyy0_r[idx] = (float)(h * eps_yy00_d);
    uzz0_r[idx] = (float)(h * eps_zz00_d);
    uyz0_r[idx] = (float)(h * eps_yz00_d);
    uxz0_r[idx] = (float)(h * eps_xz00_d);
    uxy0_r[idx] = (float)(h * eps_xy00_d);
}

// Minimize mode: g(r) = δF/δphi = double-well + elastic (no chemical).
// elastic: dgel/dphi = -sigma:deps0_dphi + 0.5*h'(phi)*Q; eigenstrain = h*eps00, stiffness C0+h*deltaC.
__global__ void compute_phi_rhs_minimize_kernel(
    const double *phi_r,
    double *rhs_r,
    double W,
    const float *sigma_xx_r,
    const float *sigma_yy_r,
    const float *sigma_zz_r,
    const float *sigma_xy_r,
    const float *sigma_xz_r,
    const float *sigma_yz_r,
    const float *uxx_r, const float *uyy_r, const float *uzz_r,
    const float *uxy_r, const float *uxz_r, const float *uyz_r,
    // Optimization(4): uxx0_r..uyz0_r 参数已移除，eps0 现场计算
    double S_p_11, double S_p_12, double S_p_13, double S_p_14, double S_p_15, double S_p_16,
    double S_p_22, double S_p_23, double S_p_24, double S_p_25, double S_p_26,
    double S_p_33, double S_p_34, double S_p_35, double S_p_36,
    double S_p_44, double S_p_45, double S_p_46,
    double S_p_55, double S_p_56,
    double S_p_66,
    double eps_xx00, double eps_yy00, double eps_zz00,
    double eps_yz00, double eps_xz00, double eps_xy00,
    int total_size,
    int elastic_enabled)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    double phi = phi_r[idx];
    double f_phi = W * g_prime_of_phi(phi);
    if (elastic_enabled) {
        double hp  = h_prime_of_phi(phi);
        double sigma_xx = (double)sigma_xx_r[idx];
        double sigma_yy = (double)sigma_yy_r[idx];
        double sigma_zz = (double)sigma_zz_r[idx];
        double sigma_xy = (double)sigma_xy_r[idx];
        double sigma_xz = (double)sigma_xz_r[idx];
        double sigma_yz = (double)sigma_yz_r[idx];
        double d_eps_xx0_dphi = hp * eps_xx00;
        double d_eps_yy0_dphi = hp * eps_yy00;
        double d_eps_zz0_dphi = hp * eps_zz00;
        double d_eps_xy0_dphi = hp * eps_xy00;
        double d_eps_xz0_dphi = hp * eps_xz00;
        double d_eps_yz0_dphi = hp * eps_yz00;
        double dgel_eigen = -( sigma_xx * d_eps_xx0_dphi + sigma_yy * d_eps_yy0_dphi + sigma_zz * d_eps_zz0_dphi
                            + 2.0 * ( sigma_xy * d_eps_xy0_dphi + sigma_xz * d_eps_xz0_dphi + sigma_yz * d_eps_yz0_dphi ) );
        // Optimization(4): 现场计算 eps0（minimize 模式仅用 phi）
        float eps_xx0_f, eps_yy0_f, eps_zz0_f;
        float eps_xy0_f, eps_xz0_f, eps_yz0_f;
        eigenstrain_phi_only_point(
            phi,
            (float)eps_xx00, (float)eps_yy00, (float)eps_zz00,
            (float)eps_yz00, (float)eps_xz00, (float)eps_xy00,
            eps_xx0_f, eps_yy0_f, eps_zz0_f,
            eps_xy0_f, eps_xz0_f, eps_yz0_f);
        
        double exx_el = (double)uxx_r[idx] - (double)eps_xx0_f;
        double eyy_el = (double)uyy_r[idx] - (double)eps_yy0_f;
        double ezz_el = (double)uzz_r[idx] - (double)eps_zz0_f;
        double eyz_el = (double)uyz_r[idx] - (double)eps_yz0_f;
        double exz_el = (double)uxz_r[idx] - (double)eps_xz0_f;
        double exy_el = (double)uxy_r[idx] - (double)eps_xy0_f;
        double Q = compute_Q_voigt(exx_el, eyy_el, ezz_el, eyz_el, exz_el, exy_el,
            S_p_11, S_p_12, S_p_13, S_p_14, S_p_15, S_p_16,
            S_p_22, S_p_23, S_p_24, S_p_25, S_p_26,
            S_p_33, S_p_34, S_p_35, S_p_36,
            S_p_44, S_p_45, S_p_46,
            S_p_55, S_p_56, S_p_66);
        double dgel_C = 0.5 * hp * Q;
        f_phi += dgel_eigen + dgel_C;
    }
    rhs_r[idx] = f_phi;
}

// ============================================================================
// 2. 位移归一化kernel
// ============================================================================

__global__ void normalize_displacement_kernel(
    float *ux_r, float *uy_r, float *uz_r,
    float invN,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    ux_r[idx] *= invN;
    uy_r[idx] *= invN;
    uz_r[idx] *= invN;
}

// 应变场归一化kernel（cuFFT C2R后需要归一化）
__global__ void normalize_strain_kernel(
    float *uxx_r, float *uyy_r, float *uzz_r,
    float *uxy_r, float *uxz_r, float *uyz_r,
    float invN,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    uxx_r[idx] *= invN;
    uyy_r[idx] *= invN;
    uzz_r[idx] *= invN;
    uxy_r[idx] *= invN;
    uxz_r[idx] *= invN;
    uyz_r[idx] *= invN;
}

// ============================================================================
// 添加外部应变kernel（实空间，全局添加，不乘以插值函数）
// 按照SDV_Poly.c:1316-1330行的逻辑，但不需要乘以Is(r_ps)
// ============================================================================

__global__ void add_external_strain_kernel(
    float *uxx_r, float *uyy_r, float *uzz_r,
    float *uxy_r, float *uxz_r, float *uyz_r,
    float E0_xx, float E0_yy, float E0_zz,
    float E0_yz, float E0_xz, float E0_xy,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    // 全局添加外部应变（SDV_Poly.c:1322-1327，但不需要乘以Is(r_ps)）
    uxx_r[idx] += E0_xx;
    uyy_r[idx] += E0_yy;
    uzz_r[idx] += E0_zz;
    uxy_r[idx] += E0_xy;
    uxz_r[idx] += E0_xz;
    uyz_r[idx] += E0_yz;
}

// ============================================================================
// 6. Green函数方法：从hij计算新的k空间位移（第二次迭代及之后）
// 按照SDV_Poly.c:1217-1226行的公式
// ============================================================================

__global__ void compute_displacement_from_hij_green_kernel(
    const cufftComplex *k_uxx, const cufftComplex *k_uyy, const cufftComplex *k_uzz,
    const cufftComplex *k_uxy, const cufftComplex *k_uxz, const cufftComplex *k_uyz,
    cufftComplex *k_ux, cufftComplex *k_uy, cufftComplex *k_uz,
    // 基体弹性刚度矩阵参数（21个S_ij）
    float S_11, float S_12, float S_13, float S_14, float S_15, float S_16,
    float S_22, float S_23, float S_24, float S_25, float S_26,
    float S_33, float S_34, float S_35, float S_36,
    float S_44, float S_45, float S_46,
    float S_55, float S_56,
    float S_66,
    int Nx, int Ny, int Nz, int NzC,
    double dx, double dy, double dz,
    int total_k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_k) return;
    
    // 计算三维索引
    int k = idx % NzC;
    int remainder = idx / NzC;
    int j = remainder % Ny;
    int i = remainder / Ny;
    
    // k=0特殊处理（设置为0）
    if (i == 0 && j == 0 && k == 0) {
        k_ux[0] = make_cuFloatComplex(0.0f, 0.0f);
        k_uy[0] = make_cuFloatComplex(0.0f, 0.0f);
        k_uz[0] = make_cuFloatComplex(0.0f, 0.0f);
        return;
    }
    
    // 计算kx, ky, kz（考虑周期性边界条件）
    double kx, ky, kz;
    if (i <= Nx/2) {
        kx = 2.0 * M_PI * i / (dx * Nx);
    } else {
        kx = 2.0 * M_PI * (i - Nx) / (dx * Nx);
    }
    
    if (j <= Ny/2) {
        ky = 2.0 * M_PI * j / (dy * Ny);
    } else {
        ky = 2.0 * M_PI * (j - Ny) / (dy * Ny);
    }
    
    kz = 2.0 * M_PI * k / (dz * Nz);
    
    double kx2 = kx * kx;
    double ky2 = ky * ky;
    double kz2 = kz * kz;
    
    cufftComplex I = make_cuFloatComplex(0.0f, -1.0f);  // -I (注意SDV_Poly.c中是-I)
    
    // 按照SDV_Poly.c:1227-1245行的新方法：使用公共中间量
    /* 公共中间量 */
    double gA = kx2*S_15 + kz2*S_35 + ky*kz*(S_36 + S_45)
                + ky2*S_46 + kx*kz*(S_13 + S_55) + kx*ky*(S_14 + S_56);
    
    double gB = ky2*S_24 + kz2*S_34 + ky*kz*(S_23 + S_44)
                + kx*kz*(S_36 + S_45) + kx*ky*(S_25 + S_46) + kx2*S_56;
    
    double gC = kz2*S_33 + 2.0*kz*(ky*S_34 + kx*S_35)
                + ky2*S_44 + 2.0*kx*ky*S_45 + kx2*S_55;
    
    double gD = ky2*S_22 + 2.0*ky*(kz*S_24 + kx*S_26)
                + kz2*S_44 + 2.0*kx*kz*S_46 + kx2*S_66;
    
    double gE = kx2*S_11 + 2.0*kx*(kz*S_15 + ky*S_16)
                + kz2*S_55 + 2.0*ky*kz*S_56 + ky2*S_66;
    
    double gF = kx2*S_16 + ky2*S_26 + kz2*S_45
                + ky*kz*(S_25 + S_46) + kx*kz*(S_14 + S_56)
                + kx*ky*(S_12 + S_66);
    
    /* det(G^{-1}) - 按照SDV_Poly.c:1248-1250行 */
    double tmp1 = gA * (-gA * gD + gB * gF)
                  - gB * (gB * gE - gA * gF)
                  + gC * (gD * gE - gF * gF);
    
    // 避免除零（如果tmp1接近0，则跳过该点）
    if (fabs(tmp1) < 1e-30) {
        k_ux[idx] = make_cuFloatComplex(0.0f, 0.0f);
        k_uy[idx] = make_cuFloatComplex(0.0f, 0.0f);
        k_uz[idx] = make_cuFloatComplex(0.0f, 0.0f);
        return;
    }
    
    /* Gij - 按照SDV_Poly.c:1253-1258行 */
    double tmp2 = (-gB * gB + gC * gD) / tmp1;           // G11
    double tmp3 = (gB * gA - gC * gF) / tmp1;           // G12
    double tmp4 = (-gA * gD + gB * gF) / tmp1;           // G13
    double tmp5 = (-gA * gA + gC * gE) / tmp1;           // G22
    double tmp6 = (-gB * gE + gA * gF) / tmp1;           // G23
    double tmp7 = (gD * gE - gF * gF) / tmp1;  // 正确的 G33
    // double tmp7 = (gD * gE - gF * gD * gE - gF) / tmp1;  // G33 (按照SDV_Poly.c:1258行)
    
    // 转换为float
    float tmp2_f = (float)tmp2;
    float tmp3_f = (float)tmp3;
    float tmp4_f = (float)tmp4;
    float tmp5_f = (float)tmp5;
    float tmp6_f = (float)tmp6;
    float tmp7_f = (float)tmp7;
    
    float kx_f = (float)kx;
    float ky_f = (float)ky;
    float kz_f = (float)kz;
    
    cufftComplex kx_c = make_cuFloatComplex(kx_f, 0.0f);
    cufftComplex ky_c = make_cuFloatComplex(ky_f, 0.0f);
    cufftComplex kz_c = make_cuFloatComplex(kz_f, 0.0f);
    
    // 获取hij的k空间值
    cufftComplex hij_xx = k_uxx[idx];
    cufftComplex hij_yy = k_uyy[idx];
    cufftComplex hij_zz = k_uzz[idx];
    cufftComplex hij_xy = k_uxy[idx];
    cufftComplex hij_xz = k_uxz[idx];
    cufftComplex hij_yz = k_uyz[idx];
    
    // 按照SDV_Poly.c:1224-1226计算新的位移场
    // term1 = kx*hij_xx + ky*hij_xy + kz*hij_xz
    // term2 = kx*hij_xy + ky*hij_yy + kz*hij_yz
    // term3 = kx*hij_xz + ky*hij_yz + kz*hij_zz
    cufftComplex term1 = cuCaddf(cuCaddf(cuCmulf(kx_c, hij_xx), cuCmulf(ky_c, hij_xy)), cuCmulf(kz_c, hij_xz));
    cufftComplex term2 = cuCaddf(cuCaddf(cuCmulf(kx_c, hij_xy), cuCmulf(ky_c, hij_yy)), cuCmulf(kz_c, hij_yz));
    cufftComplex term3 = cuCaddf(cuCaddf(cuCmulf(kx_c, hij_xz), cuCmulf(ky_c, hij_yz)), cuCmulf(kz_c, hij_zz));
    
    // k_ux = -I*(tmp2*term1 + tmp3*term2 + tmp4*term3)
    cufftComplex sum_ux = cuCaddf(cuCaddf(
        cuCmulf(make_cuFloatComplex(tmp2_f, 0.0f), term1),
        cuCmulf(make_cuFloatComplex(tmp3_f, 0.0f), term2)),
        cuCmulf(make_cuFloatComplex(tmp4_f, 0.0f), term3));
    k_ux[idx] = cuCmulf(I, sum_ux);  // I = -I，所以这里已经是-I*sum
    
    // k_uy = -I*(tmp3*term1 + tmp5*term2 + tmp6*term3)
    cufftComplex sum_uy = cuCaddf(cuCaddf(
        cuCmulf(make_cuFloatComplex(tmp3_f, 0.0f), term1),
        cuCmulf(make_cuFloatComplex(tmp5_f, 0.0f), term2)),
        cuCmulf(make_cuFloatComplex(tmp6_f, 0.0f), term3));
    k_uy[idx] = cuCmulf(I, sum_uy);
    
    // k_uz = -I*(tmp4*term1 + tmp6*term2 + tmp7*term3)
    cufftComplex sum_uz = cuCaddf(cuCaddf(
        cuCmulf(make_cuFloatComplex(tmp4_f, 0.0f), term1),
        cuCmulf(make_cuFloatComplex(tmp6_f, 0.0f), term2)),
        cuCmulf(make_cuFloatComplex(tmp7_f, 0.0f), term3));
    k_uz[idx] = cuCmulf(I, sum_uz);
}

// Launch function for Green function kernel
void launch_compute_displacement_from_hij_green_kernel(
    const cufftComplex *k_uxx, const cufftComplex *k_uyy, const cufftComplex *k_uzz,
    const cufftComplex *k_uxy, const cufftComplex *k_uxz, const cufftComplex *k_uyz,
    cufftComplex *k_ux, cufftComplex *k_uy, cufftComplex *k_uz,
    float S_11, float S_12, float S_13, float S_14, float S_15, float S_16,
    float S_22, float S_23, float S_24, float S_25, float S_26,
    float S_33, float S_34, float S_35, float S_36,
    float S_44, float S_45, float S_46,
    float S_55, float S_56,
    float S_66,
    int Nx, int Ny, int Nz, int NzC,
    double dx, double dy, double dz,
    int total_k)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_k + threadsPerBlock - 1) / threadsPerBlock;
    
    compute_displacement_from_hij_green_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        k_uxx, k_uyy, k_uzz, k_uxy, k_uxz, k_uyz,
        k_ux, k_uy, k_uz,
        S_11, S_12, S_13, S_14, S_15, S_16,
        S_22, S_23, S_24, S_25, S_26,
        S_33, S_34, S_35, S_36,
        S_44, S_45, S_46,
        S_55, S_56,
        S_66,
        Nx, Ny, Nz, NzC,
        dx, dy, dz,
        total_k);
    
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 3. 应变场计算kernel（k空间，从位移计算应变）
// ============================================================================

__global__ void compute_strain_from_displacement_k_kernel(
    const cufftComplex *ux_k, const cufftComplex *uy_k, const cufftComplex *uz_k,
    cufftComplex *uxx_k, cufftComplex *uyy_k, cufftComplex *uzz_k,
    cufftComplex *uxy_k, cufftComplex *uxz_k, cufftComplex *uyz_k,
    int Nx, int Ny, int Nz, int NzC,
    double dx, double dy, double dz,
    int total_k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_k) return;
    
    // 计算三维索引
    int k = idx % NzC;
    int remainder = idx / NzC;
    int j = remainder % Ny;
    int i = remainder / Ny;
    
    double kx = kx_wrap(i, Nx, dx);
    double ky = ky_wrap(j, Ny, dy);
    double kz = kz_wrap(k, Nz, dz);
    
    cufftComplex I = make_cuFloatComplex(0.0f, 1.0f);
    cufftComplex ux = ux_k[idx];
    cufftComplex uy = uy_k[idx];
    cufftComplex uz = uz_k[idx];
    
    // 不除以N，在C2R后再归一化（与主程序保持一致）
    // 这样可以保持与主程序弹性迭代流程的兼容性
    cufftComplex kx_c = make_cuFloatComplex((float)kx, 0.0f);
    cufftComplex ky_c = make_cuFloatComplex((float)ky, 0.0f);
    cufftComplex kz_c = make_cuFloatComplex((float)kz, 0.0f);
    
    // uxx = I*kx*ux （不除以N，在C2R后归一化）
    uxx_k[idx] = cuCmulf(cuCmulf(I, kx_c), ux);
    
    // uyy = I*ky*uy （不除以N）
    uyy_k[idx] = cuCmulf(cuCmulf(I, ky_c), uy);
    
    // uzz = I*kz*uz （不除以N）
    uzz_k[idx] = cuCmulf(cuCmulf(I, kz_c), uz);
    
    // uxy = I*(ky*ux + kx*uy)/2 （不除以N）
    cufftComplex term_xy = cuCaddf(cuCmulf(ky_c, ux), cuCmulf(kx_c, uy));
    cufftComplex half_c = make_cuFloatComplex(0.5f, 0.0f);
    uxy_k[idx] = cuCmulf(cuCmulf(I, term_xy), half_c);
    
    // uxz = I*(kz*ux + kx*uz)/2 （不除以N）
    cufftComplex term_xz = cuCaddf(cuCmulf(kz_c, ux), cuCmulf(kx_c, uz));
    uxz_k[idx] = cuCmulf(cuCmulf(I, term_xz), half_c);
    
    // uyz = I*(ky*uz + kz*uy)/2 （不除以N）
    cufftComplex term_yz = cuCaddf(cuCmulf(ky_c, uz), cuCmulf(kz_c, uy));
    uyz_k[idx] = cuCmulf(cuCmulf(I, term_yz), half_c);
}

// 不除以N的版本（用于归一化后的位移场）
// ============================================================================
// 4. 位移场计算kernel（k空间，使用均匀弹性常数）- 最复杂
// 按照SDV_Poly.c:949-952行的公式
// ============================================================================

__global__ void compute_displacement_from_eigenstrain_k_kernel(
    const cufftComplex *uxx0_k, const cufftComplex *uyy0_k, const cufftComplex *uzz0_k,
    const cufftComplex *uxy0_k, const cufftComplex *uxz0_k, const cufftComplex *uyz0_k,
    cufftComplex *ux_k, cufftComplex *uy_k, cufftComplex *uz_k,
    // 基体弹性刚度矩阵参数（21个S_ij）
    float S_11, float S_12, float S_13, float S_14, float S_15, float S_16,
    float S_22, float S_23, float S_24, float S_25, float S_26,
    float S_33, float S_34, float S_35, float S_36,
    float S_44, float S_45, float S_46,
    float S_55, float S_56,
    float S_66,
    int Nx, int Ny, int Nz, int NzC,
    double dx, double dy, double dz,
    int total_k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_k) return;

    // 三维索引
    int k = idx % NzC;
    int remainder = idx / NzC;
    int j = remainder % Ny;
    int i = remainder / Ny;

    // k=0 模式置零
    if (i == 0 && j == 0 && k == 0) {
        ux_k[0] = make_cuFloatComplex(0.0f, 0.0f);
        uy_k[0] = make_cuFloatComplex(0.0f, 0.0f);
        uz_k[0] = make_cuFloatComplex(0.0f, 0.0f);
        return;
    }

    // 波矢 kx, ky, kz（周期边界）
    double kx, ky, kz;
    if (i <= Nx / 2) {
        kx = 2.0 * M_PI * i / (dx * Nx);
    } else {
        kx = 2.0 * M_PI * (i - Nx) / (dx * Nx);
    }

    if (j <= Ny / 2) {
        ky = 2.0 * M_PI * j / (dy * Ny);
    } else {
        ky = 2.0 * M_PI * (j - Ny) / (dy * Ny);
    }

    kz = 2.0 * M_PI * k / (dz * Nz);

    double kx2 = kx * kx;
    double ky2 = ky * ky;
    double kz2 = kz * kz;

    // 读取 eigenstrain(k)
    cufftComplex uxx0 = uxx0_k[idx];
    cufftComplex uyy0 = uyy0_k[idx];
    cufftComplex uzz0 = uzz0_k[idx];
    cufftComplex uxy0 = uxy0_k[idx];
    cufftComplex uxz0 = uxz0_k[idx];
    cufftComplex uyz0 = uyz0_k[idx];

    double uxx0_r = (double)cuCrealf(uxx0);
    double uxx0_i = (double)cuCimagf(uxx0);
    double uyy0_r = (double)cuCrealf(uyy0);
    double uyy0_i = (double)cuCimagf(uyy0);
    double uzz0_r = (double)cuCrealf(uzz0);
    double uzz0_i = (double)cuCimagf(uzz0);
    double uxy0_r = (double)cuCrealf(uxy0);
    double uxy0_i = (double)cuCimagf(uxy0);
    double uxz0_r = (double)cuCrealf(uxz0);
    double uxz0_i = (double)cuCimagf(uxz0);
    double uyz0_r = (double)cuCrealf(uyz0);
    double uyz0_i = (double)cuCimagf(uyz0);

    // 转成 double 方便后续运算
    double S11 = (double)S_11, S12 = (double)S_12, S13 = (double)S_13;
    double S14 = (double)S_14, S15 = (double)S_15, S16 = (double)S_16;
    double S22 = (double)S_22, S23 = (double)S_23, S24 = (double)S_24;
    double S25 = (double)S_25, S26 = (double)S_26;
    double S33 = (double)S_33, S34 = (double)S_34, S35 = (double)S_35, S36 = (double)S_36;
    double S44 = (double)S_44, S45 = (double)S_45, S46 = (double)S_46;
    double S55 = (double)S_55, S56 = (double)S_56;
    double S66 = (double)S_66;

    // ==========
    // 构造 K 矩阵分量（与 Mathematica 中的 k^T S k 完全对应）
    // K11 = k_i k_j S_i1j1
    double K11 = kx2 * S11
               + 2.0 * kx * (kz * S15 + ky * S16)
               + kz2 * S55
               + 2.0 * ky * kz * S56
               + ky2 * S66;

    // K22 = k_i k_j S_i2j2
    double K22 = ky2 * S22
               + 2.0 * ky * (kz * S24 + kx * S26)
               + kz2 * S44
               + 2.0 * kx * kz * S46
               + kx2 * S66;

    // K33 = k_i k_j S_i3j3
    double K33 = kz2 * S33
               + 2.0 * kz * (ky * S34 + kx * S35)
               + ky2 * S44
               + 2.0 * kx * ky * S45
               + kx2 * S55;

    // K13 = k_i k_j S_i1j3，对应 Mathematica 里的那一大串 kx^2 S15 + ...
    double K13 = kx2 * S15
               + kz2 * S35
               + ky * kz * (S36 + S45)
               + ky2 * S46
               + kx * kz * (S13 + S55)
               + kx * ky * (S14 + S56);

    // K23 = k_i k_j S_i2j3
    double K23 = ky2 * S24
               + kz2 * S34
               + ky * kz * (S23 + S44)
               + kx * kz * (S36 + S45)
               + kx * ky * (S25 + S46)
               + kx2 * S56;

    // K12 = k_i k_j S_i1j2
    double K12 = kx2 * S16
               + ky2 * S26
               + kz2 * S45
               + ky * kz * (S25 + S46)
               + kx * kz * (S14 + S56)
               + kx * ky * (S12 + S66);

    // ==========
    // 分母：det(K) = det[[K11 K12 K13],[K12 K22 K23],[K13 K23 K33]]
    double tmp1 = K13 * (-K13 * K22 + K23 * K12)
                - K23 * ( K23 * K11 - K13 * K12)
                + K33 * ( K22 * K11 - K12 * K12);

    if (fabs(tmp1) < 1e-30) {
        // 避免除以零（或非常接近零）
        ux_k[idx] = make_cuFloatComplex(0.0f, 0.0f);
        uy_k[idx] = make_cuFloatComplex(0.0f, 0.0f);
        uz_k[idx] = make_cuFloatComplex(0.0f, 0.0f);
        return;
    }

    double inv_tmp1 = 1.0 / tmp1;

    // ==========
    // 构造与 Mathematica 中 Subscript[Γ,ij], Subscript[ε,ii] 对应的组合
    // V1 对应 Mathematica 第一组括号里的 eigenstrain 组合
    // 这里修正了物理逻辑：对角项系数为 1，剪切项系数为 2
    double V1_r = (
        2.0 * (kz * S36 + ky * S46 + kx * S56) * uxy0_r +
        2.0 * (kz * S35 + ky * S45 + kx * S55) * uxz0_r +
        2.0 * (kz * S34 + ky * S44 + kx * S45) * uyz0_r +
        (kz * S13 + ky * S14 + kx * S15) * uxx0_r +
        (kz * S23 + ky * S24 + kx * S25) * uyy0_r +
        (kz * S33 + ky * S34 + kx * S35) * uzz0_r
    );

    double V1_i = (
        2.0 * (kz * S36 + ky * S46 + kx * S56) * uxy0_i +
        2.0 * (kz * S35 + ky * S45 + kx * S55) * uxz0_i +
        2.0 * (kz * S34 + ky * S44 + kx * S45) * uyz0_i +
        (kz * S13 + ky * S14 + kx * S15) * uxx0_i +
        (kz * S23 + ky * S24 + kx * S25) * uyy0_i +
        (kz * S33 + ky * S34 + kx * S35) * uzz0_i
    );

    // V2 对应 Mathematica 第二组括号里的 eigenstrain 组合
    double V2_r = (
        2.0 * (ky * S26 + kz * S46 + kx * S66) * uxy0_r +
        2.0 * (ky * S25 + kz * S45 + kx * S56) * uxz0_r +
        2.0 * (ky * S24 + kz * S44 + kx * S46) * uyz0_r +
        (ky * S12 + kz * S14 + kx * S16) * uxx0_r +
        (ky * S22 + kz * S24 + kx * S26) * uyy0_r +
        (ky * S23 + kz * S34 + kx * S36) * uzz0_r
    );

    double V2_i = (
        2.0 * (ky * S26 + kz * S46 + kx * S66) * uxy0_i +
        2.0 * (ky * S25 + kz * S45 + kx * S56) * uxz0_i +
        2.0 * (ky * S24 + kz * S44 + kx * S46) * uyz0_i +
        (ky * S12 + kz * S14 + kx * S16) * uxx0_i +
        (ky * S22 + kz * S24 + kx * S26) * uyy0_i +
        (ky * S23 + kz * S34 + kx * S36) * uzz0_i
    );

    // V3 对应 Mathematica 第三组括号里的 eigenstrain 组合
    double V3_r = (
        2.0 * (kx * S16 + kz * S56 + ky * S66) * uxy0_r +
        2.0 * (kx * S15 + kz * S55 + ky * S56) * uxz0_r +
        2.0 * (kx * S14 + kz * S45 + ky * S46) * uyz0_r +
        (kx * S11 + kz * S15 + ky * S16) * uxx0_r +
        (kx * S12 + kz * S25 + ky * S26) * uyy0_r +
        (kx * S13 + kz * S35 + ky * S36) * uzz0_r
    );

    double V3_i = (
        2.0 * (kx * S16 + kz * S56 + ky * S66) * uxy0_i +
        2.0 * (kx * S15 + kz * S55 + ky * S56) * uxz0_i +
        2.0 * (kx * S14 + kz * S45 + ky * S46) * uyz0_i +
        (kx * S11 + kz * S15 + ky * S16) * uxx0_i +
        (kx * S12 + kz * S25 + ky * S26) * uyy0_i +
        (kx * S13 + kz * S35 + ky * S36) * uzz0_i
    );

    // ==========
    // 对应 Mathematica 中 ux, uy, uz 的三个“系数块”
    // 记：
    //  Ax, Bx, Cx  对应 kux 的三个系数
    //  Ay, By, Cy  对应 kuy 的三个系数
    //  Az, Bz, Cz  对应 kuz 的三个系数

    // --- x 方向 ---
    double Ax = -K13 * K22 + K23 * K12;
    double Bx =  K23 * K13 - K33 * K12;
    double Cx = -K23 * K23 + K33 * K22;

    // --- y 方向 ---
    double Ay = -K23 * K11 + K13 * K12;
    double By = -K13 * K13 + K33 * K11;
    double Cy =  K23 * K13 - K33 * K12;

    // --- z 方向 ---
    double Az =  K22 * K11 - K12 * K12;
    double Bz = -K23 * K11 + K13 * K12;
    double Cz = -K13 * K22 + K23 * K12;

    // ==========
    // 计算 k 空间位移 ux(k), uy(k), uz(k)
    // 修正：位移场与力向量有 90 度相位差 (u = -i * K^-1 * F)
    // 因此：u_real = (K^-1 * F)_imag, u_imag = -(K^-1 * F)_real
    
    // 先计算 (K^-1 * F) 的实部和虚部
    double sum_x_r = (Ax * V1_r + Bx * V2_r + Cx * V3_r) * inv_tmp1;
    double sum_x_i = (Ax * V1_i + Bx * V2_i + Cx * V3_i) * inv_tmp1;

    double sum_y_r = (Ay * V1_r + By * V2_r + Cy * V3_r) * inv_tmp1;
    double sum_y_i = (Ay * V1_i + By * V2_i + Cy * V3_i) * inv_tmp1;

    double sum_z_r = (Az * V1_r + Bz * V2_r + Cz * V3_r) * inv_tmp1;
    double sum_z_i = (Az * V1_i + Bz * V2_i + Cz * V3_i) * inv_tmp1;

    // 应用相位变换 u = -i * (sum_r + i*sum_i) = sum_i - i*sum_r
    double ux_r =  sum_x_i;
    double ux_i = -sum_x_r;

    double uy_r =  sum_y_i;
    double uy_i = -sum_y_r;

    double uz_r =  sum_z_i;
    double uz_i = -sum_z_r;

    // 写回（float 复数）
    ux_k[idx] = make_cuFloatComplex((float)ux_r, (float)ux_i);
    uy_k[idx] = make_cuFloatComplex((float)uy_r, (float)uy_i);
    uz_k[idx] = make_cuFloatComplex((float)uz_r, (float)uz_i);
}

// ============================================================================
// 5. 弹性常数perturbation计算hij kernel（实空间，使用h(phi)插值）
// ============================================================================

// ============================================================================
// 6. 应变场计算kernel（实空间，考虑perturbation和外部应变） 这里计算的是hij！！！！！用来用格林函数组装在弹性不均一和外部应变场下新的displacement
// 按照SDV_Poly.c:1126-1131行的公式
// ============================================================================

__global__ void compute_strain_with_perturbation_kernel(
    // 输入：从位移计算得到的初始应变场
    const float *uxx_init, const float *uyy_init, const float *uzz_init,
    const float *uxy_init, const float *uxz_init, const float *uyz_init,
        // Optimization(4): eigenstrain 参数已移除，现场计算
    // 输入：相场和 xB（用于现场计算 eigenstrain）
    const double *phi_r,
    const double *eta_r,
    const double *xB_r,  // 可为 NULL（minimize 模式）
    // 输出：修正后的local strain
    float *uxx, float *uyy, float *uzz,
    float *uxy, float *uxz, float *uyz,
    // 基体弹性常数（21个S_ij）
    float S_11, float S_12, float S_13, float S_14, float S_15, float S_16,
    float S_22, float S_23, float S_24, float S_25, float S_26,
    float S_33, float S_34, float S_35, float S_36,
    float S_44, float S_45, float S_46,
    float S_55, float S_56,
    float S_66,
    // Perturbation参数（21个S_p_ij）- 直接传入，不再需要数组
    float S_p_11, float S_p_12, float S_p_13, float S_p_14, float S_p_15, float S_p_16,
    float S_p_22, float S_p_23, float S_p_24, float S_p_25, float S_p_26,
    float S_p_33, float S_p_34, float S_p_35, float S_p_36,
    float S_p_44, float S_p_45, float S_p_46,
    float S_p_55, float S_p_56,
    float S_p_66,
    // Optimization(4): 需要 eps0 相关参数来现场计算
    float eps_xx00, float eps_yy00, float eps_zz00,
    float eps_yz00, float eps_xz00, float eps_xy00,
    double eps_iso_over_vB,
    int gp_mode_enabled,
    int gp_elastic_enabled,
    double gp_eps_iso,
    // 外部应变（6个分量）
    float E0_xx, float E0_yy, float E0_zz, float E0_yz, float E0_xz, float E0_xy,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    // 读取初始应变
    float tmp1 = uxx_init[idx];
    float tmp2 = uyy_init[idx];
    float tmp3 = uzz_init[idx];
    float tmp4 = uxy_init[idx];
    float tmp5 = uxz_init[idx];
    float tmp6 = uyz_init[idx];
    
    // Optimization(4): 现场计算 eigenstrain
    double phi = phi_r[idx];
    double eta = eta_r ? eta_r[idx] : 0.0;
    double xB = (xB_r != NULL) ? clamp01(xB_r[idx]) : 0.0;
    float r_uxx0, r_uyy0, r_uzz0, r_uxy0, r_uxz0, r_uyz0;
    if (xB_r != NULL) {
        eigenstrain_phi_eta_xB_point(phi, eta, xB, eps_xx00, eps_yy00, eps_zz00,
                                     eps_yz00, eps_xz00, eps_xy00, eps_iso_over_vB,
                                     gp_mode_enabled, gp_elastic_enabled, gp_eps_iso,
                                     r_uxx0, r_uyy0, r_uzz0, r_uxy0, r_uxz0, r_uyz0);
    } else {
        eigenstrain_phi_only_point(phi, eps_xx00, eps_yy00, eps_zz00, eps_yz00, eps_xz00, eps_xy00,
                                   r_uxx0, r_uyy0, r_uzz0, r_uxy0, r_uxz0, r_uyz0);
    }
    
    // 直接在kernel内计算perturbation（使用h(phi)插值，按照SDV_Poly.c:1101-1121行的逻辑）
    // 用户要求忽略substrate和Air phase，只考虑单个phase field
    // phi 已在上面声明，无需重复声明
    double h_val = stiffness_beta_weight_point(phi, eta, gp_mode_enabled, gp_elastic_enabled);
    
    // 计算perturbation项（直接在strain计算中使用，不存储）
    float tmpp1 = (float)(h_val * S_p_11);
    float tmpp2 = (float)(h_val * S_p_22);
    float tmpp3 = (float)(h_val * S_p_33);
    float tmpp4 = (float)(h_val * S_p_44);
    float tmpp5 = (float)(h_val * S_p_55);
    float tmpp6 = (float)(h_val * S_p_66);
    float tmpp7 = (float)(h_val * S_p_12);
    float tmpp8 = (float)(h_val * S_p_13);
    float tmpp9 = (float)(h_val * S_p_23);
    float tmpp10 = (float)(h_val * S_p_14);
    float tmpp11 = (float)(h_val * S_p_15);
    float tmpp12 = (float)(h_val * S_p_16);
    float tmpp13 = (float)(h_val * S_p_24);
    float tmpp14 = (float)(h_val * S_p_25);
    float tmpp15 = (float)(h_val * S_p_26);
    float tmpp16 = (float)(h_val * S_p_34);
    float tmpp17 = (float)(h_val * S_p_35);
    float tmpp18 = (float)(h_val * S_p_36);
    float tmpp19 = (float)(h_val * S_p_45);
    float tmpp20 = (float)(h_val * S_p_46);
    float tmpp21 = (float)(h_val * S_p_56);
    
    // 注意：用户要求忽略Is(r_ps)部分，所以tmp8=0（外部应变应用简化）
    // 按照SDV_Poly.c:1126-1131行的公式计算（忽略substrate相关项）
    // 简化的外部应变应用（不考虑Is插值）
    
    // 按照SDV_Poly.c:1126-1131行的公式计算所有应变分量
    // 注意：用户要求忽略Is(r_ps)部分，所以tmp8=0（外部应变直接应用）
    
    // r_uxx (SDV_Poly.c:1126行)
    uxx[idx] = -(S_11+tmpp1)*tmp1 - 2.0f*(S_16+tmpp12)*tmp4 - 2.0f*(S_15+tmpp11)*tmp5 
                - (S_12+tmpp7)*tmp2 - 2.0f*(S_14+tmpp10)*tmp6 - (S_13+tmpp8)*tmp3
                - (S_11+tmpp1)*E0_xx - 2.0f*(S_16+tmpp12)*E0_xy - 2.0f*(S_15+tmpp11)*E0_xz
                - (S_12+tmpp7)*E0_yy - 2.0f*(S_14+tmpp10)*E0_yz - (S_13+tmpp8)*E0_zz
                + tmp1*S_11 + tmp2*S_12 + tmp3*S_13 + 2.0f*tmp6*S_14 + 2.0f*tmp5*S_15 + 2.0f*tmp4*S_16
                + 2.0f*(S_16+tmpp12)*r_uxy0 + 2.0f*(S_15+tmpp11)*r_uxz0 + 2.0f*(S_14+tmpp10)*r_uyz0
                + (S_11+tmpp1)*r_uxx0 + (S_12+tmpp7)*r_uyy0 + (S_13+tmpp8)*r_uzz0;
    
    // r_uxy (SDV_Poly.c:1127行)
    uxy[idx] = -(S_16+tmpp12)*tmp1 - 2.0f*(S_66+tmpp6)*tmp4 - 2.0f*(S_56+tmpp21)*tmp5
                - (S_26+tmpp15)*tmp2 - 2.0f*(S_46+tmpp20)*tmp6 - (S_36+tmpp18)*tmp3
                - (S_16+tmpp12)*E0_xx - 2.0f*(S_66+tmpp6)*E0_xy - 2.0f*(S_56+tmpp21)*E0_xz
                - (S_26+tmpp15)*E0_yy - 2.0f*(S_46+tmpp20)*E0_yz - (S_36+tmpp18)*E0_zz
                + tmp1*S_16 + tmp2*S_26 + tmp3*S_36 + 2.0f*tmp6*S_46 + 2.0f*tmp5*S_56 + 2.0f*tmp4*S_66
                + 2.0f*(S_66+tmpp6)*r_uxy0 + 2.0f*(S_56+tmpp21)*r_uxz0 + 2.0f*(S_46+tmpp20)*r_uyz0
                + (S_16+tmpp12)*r_uxx0 + (S_26+tmpp15)*r_uyy0 + (S_36+tmpp18)*r_uzz0;
    
    // r_uxz (SDV_Poly.c:1128行)
    uxz[idx] = -(S_15+tmpp11)*tmp1 - 2.0f*(S_56+tmpp21)*tmp4 - 2.0f*(S_55+tmpp5)*tmp5
                - (S_25+tmpp14)*tmp2 - 2.0f*(S_45+tmpp19)*tmp6 - (S_35+tmpp17)*tmp3
                - (S_15+tmpp11)*E0_xx - 2.0f*(S_56+tmpp21)*E0_xy - 2.0f*(S_55+tmpp5)*E0_xz
                - (S_25+tmpp14)*E0_yy - 2.0f*(S_45+tmpp19)*E0_yz - (S_35+tmpp17)*E0_zz
                + tmp1*S_15 + tmp2*S_25 + tmp3*S_35 + 2.0f*tmp6*S_45 + 2.0f*tmp5*S_55 + 2.0f*tmp4*S_56
                + 2.0f*(S_56+tmpp21)*r_uxy0 + 2.0f*(S_55+tmpp5)*r_uxz0 + 2.0f*(S_45+tmpp19)*r_uyz0
                + (S_15+tmpp11)*r_uxx0 + (S_25+tmpp14)*r_uyy0 + (S_35+tmpp17)*r_uzz0;
    
    // r_uyy (SDV_Poly.c:1129行)
    uyy[idx] = -(S_12+tmpp7)*tmp1 - 2.0f*(S_26+tmpp15)*tmp4 - 2.0f*(S_25+tmpp14)*tmp5
                - (S_22+tmpp2)*tmp2 - 2.0f*(S_24+tmpp13)*tmp6 - (S_23+tmpp9)*tmp3
                - (S_12+tmpp7)*E0_xx - 2.0f*(S_26+tmpp15)*E0_xy - 2.0f*(S_25+tmpp14)*E0_xz
                - (S_22+tmpp2)*E0_yy - 2.0f*(S_24+tmpp13)*E0_yz - (S_23+tmpp9)*E0_zz
                + tmp1*S_12 + tmp2*S_22 + tmp3*S_23 + 2.0f*tmp6*S_24 + 2.0f*tmp5*S_25 + 2.0f*tmp4*S_26
                + 2.0f*(S_26+tmpp15)*r_uxy0 + 2.0f*(S_25+tmpp14)*r_uxz0 + 2.0f*(S_24+tmpp13)*r_uyz0
                + (S_12+tmpp7)*r_uxx0 + (S_22+tmpp2)*r_uyy0 + (S_23+tmpp9)*r_uzz0;
    
    // r_uyz (SDV_Poly.c:1130行)
    uyz[idx] = -(S_14+tmpp10)*tmp1 - 2.0f*(S_46+tmpp20)*tmp4 - 2.0f*(S_45+tmpp19)*tmp5
                - (S_24+tmpp13)*tmp2 - 2.0f*(S_44+tmpp4)*tmp6 - (S_34+tmpp16)*tmp3
                - (S_14+tmpp10)*E0_xx - 2.0f*(S_46+tmpp20)*E0_xy - 2.0f*(S_45+tmpp19)*E0_xz
                - (S_24+tmpp13)*E0_yy - 2.0f*(S_44+tmpp4)*E0_yz - (S_34+tmpp16)*E0_zz
                + tmp1*S_14 + tmp2*S_24 + tmp3*S_34 + 2.0f*tmp6*S_44 + 2.0f*tmp5*S_45 + 2.0f*tmp4*S_46
                + 2.0f*(S_46+tmpp20)*r_uxy0 + 2.0f*(S_45+tmpp19)*r_uxz0 + 2.0f*(S_44+tmpp4)*r_uyz0
                + (S_14+tmpp10)*r_uxx0 + (S_24+tmpp13)*r_uyy0 + (S_34+tmpp16)*r_uzz0;
    
    // r_uzz (SDV_Poly.c:1131行)
    uzz[idx] = -(S_13+tmpp8)*tmp1 - 2.0f*(S_36+tmpp18)*tmp4 - 2.0f*(S_35+tmpp17)*tmp5
                - (S_23+tmpp9)*tmp2 - 2.0f*(S_34+tmpp16)*tmp6 - (S_33+tmpp3)*tmp3
                - (S_13+tmpp8)*E0_xx - 2.0f*(S_36+tmpp18)*E0_xy - 2.0f*(S_35+tmpp17)*E0_xz
                - (S_23+tmpp9)*E0_yy - 2.0f*(S_34+tmpp16)*E0_yz - (S_33+tmpp3)*E0_zz
                + tmp1*S_13 + tmp2*S_23 + tmp3*S_33 + 2.0f*tmp6*S_34 + 2.0f*tmp5*S_35 + 2.0f*tmp4*S_36
                + 2.0f*(S_36+tmpp18)*r_uxy0 + 2.0f*(S_35+tmpp17)*r_uxz0 + 2.0f*(S_34+tmpp16)*r_uyz0
                + (S_13+tmpp8)*r_uxx0 + (S_23+tmpp9)*r_uyy0 + (S_33+tmpp3)*r_uzz0;
}

// ============================================================================
// Launch函数（kernel包装器）
// ============================================================================

void launch_compute_eigenstrain_from_phi_kernel(
    const double *phi_r,
    const double *eta_r,
    const double *xB_r,
    float *uxx0_r, float *uyy0_r, float *uzz0_r,
    float *uxy0_r, float *uxz0_r, float *uyz0_r,
    float eps_xx00, float eps_yy00, float eps_zz00,
    float eps_yz00, float eps_xz00, float eps_xy00,
    double eps_iso_over_vB,
    int gp_mode_enabled,
    int gp_elastic_enabled,
    double gp_eps_iso,
    int total_size)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_eigenstrain_from_phi_kernel<<<blocks, threads>>>(
        phi_r, eta_r, xB_r,
        uxx0_r, uyy0_r, uzz0_r, uxy0_r, uxz0_r, uyz0_r,
        eps_xx00, eps_yy00, eps_zz00,
        eps_yz00, eps_xz00, eps_xy00, eps_iso_over_vB,
        gp_mode_enabled, gp_elastic_enabled, gp_eps_iso,
        total_size);
}

void launch_compute_eigenstrain_from_phi_only_kernel(
    const double *phi_r,
    float *uxx0_r, float *uyy0_r, float *uzz0_r,
    float *uxy0_r, float *uxz0_r, float *uyz0_r,
    float eps_xx00, float eps_yy00, float eps_zz00,
    float eps_yz00, float eps_xz00, float eps_xy00,
    int total_size)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_eigenstrain_from_phi_only_kernel<<<blocks, threads>>>(
        phi_r,
        uxx0_r, uyy0_r, uzz0_r, uxy0_r, uxz0_r, uyz0_r,
        eps_xx00, eps_yy00, eps_zz00, eps_yz00, eps_xz00, eps_xy00,
        total_size);
}

void launch_compute_phi_rhs_minimize_kernel(
    const double *phi_r,
    double *rhs_r,
    double W,
    const float *sigma_xx_r,
    const float *sigma_yy_r,
    const float *sigma_zz_r,
    const float *sigma_xy_r,
    const float *sigma_xz_r,
    const float *sigma_yz_r,
    const float *uxx_r, const float *uyy_r, const float *uzz_r,
    const float *uxy_r, const float *uxz_r, const float *uyz_r,
    // Optimization(4): uxx0_r..uyz0_r 参数已移除，eps0 现场计算
    double S_p_11, double S_p_12, double S_p_13, double S_p_14, double S_p_15, double S_p_16,
    double S_p_22, double S_p_23, double S_p_24, double S_p_25, double S_p_26,
    double S_p_33, double S_p_34, double S_p_35, double S_p_36,
    double S_p_44, double S_p_45, double S_p_46,
    double S_p_55, double S_p_56,
    double S_p_66,
    double eps_xx00, double eps_yy00, double eps_zz00,
    double eps_yz00, double eps_xz00, double eps_xy00,
    int total_size,
    int elastic_enabled)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_phi_rhs_minimize_kernel<<<blocks, threads>>>(
        phi_r, rhs_r, W,
        sigma_xx_r, sigma_yy_r, sigma_zz_r,
        sigma_xy_r, sigma_xz_r, sigma_yz_r,
        uxx_r, uyy_r, uzz_r, uxy_r, uxz_r, uyz_r,
        // Optimization(4): uxx0_r..uyz0_r 参数已移除
        S_p_11, S_p_12, S_p_13, S_p_14, S_p_15, S_p_16,
        S_p_22, S_p_23, S_p_24, S_p_25, S_p_26,
        S_p_33, S_p_34, S_p_35, S_p_36,
        S_p_44, S_p_45, S_p_46,
        S_p_55, S_p_56, S_p_66,
        eps_xx00, eps_yy00, eps_zz00, eps_yz00, eps_xz00, eps_xy00,
        total_size, elastic_enabled);
}

void launch_normalize_displacement_kernel(
    float *ux_r, float *uy_r, float *uz_r,
    float invN,
    int total_size)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    normalize_displacement_kernel<<<blocks, threads>>>(ux_r, uy_r, uz_r, invN, total_size);
}

void launch_normalize_strain_kernel(
    float *uxx_r, float *uyy_r, float *uzz_r,
    float *uxy_r, float *uxz_r, float *uyz_r,
    float invN,
    int total_size)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    normalize_strain_kernel<<<blocks, threads>>>(uxx_r, uyy_r, uzz_r, uxy_r, uxz_r, uyz_r, invN, total_size);
}

// 添加外部应变
void launch_add_external_strain_kernel(
    float *uxx_r, float *uyy_r, float *uzz_r,
    float *uxy_r, float *uxz_r, float *uyz_r,
    float E0_xx, float E0_yy, float E0_zz,
    float E0_yz, float E0_xz, float E0_xy,
    int total_size)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    add_external_strain_kernel<<<blocks, threads>>>(
        uxx_r, uyy_r, uzz_r,
        uxy_r, uxz_r, uyz_r,
        E0_xx, E0_yy, E0_zz,
        E0_yz, E0_xz, E0_xy,
        total_size);
}

void launch_compute_strain_from_displacement_k_kernel(
    const cufftComplex *ux_k, const cufftComplex *uy_k, const cufftComplex *uz_k,
    cufftComplex *uxx_k, cufftComplex *uyy_k, cufftComplex *uzz_k,
    cufftComplex *uxy_k, cufftComplex *uxz_k, cufftComplex *uyz_k,
    int Nx, int Ny, int Nz, int NzC,
    double dx, double dy, double dz,
    int total_k)
{
    int threads, blocks;
    configure_launch(total_k, threads, blocks);
    compute_strain_from_displacement_k_kernel<<<blocks, threads>>>(
        ux_k, uy_k, uz_k, uxx_k, uyy_k, uzz_k, uxy_k, uxz_k, uyz_k,
        Nx, Ny, Nz, NzC, dx, dy, dz, total_k);
}

void launch_compute_displacement_from_eigenstrain_k_kernel(
    const cufftComplex *uxx0_k, const cufftComplex *uyy0_k, const cufftComplex *uzz0_k,
    const cufftComplex *uxy0_k, const cufftComplex *uxz0_k, const cufftComplex *uyz0_k,
    cufftComplex *ux_k, cufftComplex *uy_k, cufftComplex *uz_k,
    float S_11, float S_12, float S_13, float S_14, float S_15, float S_16,
    float S_22, float S_23, float S_24, float S_25, float S_26,
    float S_33, float S_34, float S_35, float S_36,
    float S_44, float S_45, float S_46,
    float S_55, float S_56,
    float S_66,
    int Nx, int Ny, int Nz, int NzC,
    double dx, double dy, double dz,
    int total_k)
{
    int threads, blocks;
    configure_launch(total_k, threads, blocks);
    compute_displacement_from_eigenstrain_k_kernel<<<blocks, threads>>>(
        uxx0_k, uyy0_k, uzz0_k, uxy0_k, uxz0_k, uyz0_k,
        ux_k, uy_k, uz_k,
        S_11, S_12, S_13, S_14, S_15, S_16,
        S_22, S_23, S_24, S_25, S_26,
        S_33, S_34, S_35, S_36,
        S_44, S_45, S_46,
        S_55, S_56,
        S_66,
        Nx, Ny, Nz, NzC, dx, dy, dz, total_k);
}

void launch_compute_strain_with_perturbation_kernel(
    const float *uxx_init, const float *uyy_init, const float *uzz_init,
    const float *uxy_init, const float *uxz_init, const float *uyz_init,
    // Optimization(4): eigenstrain 参数已移除，现场计算
    const double *phi_r,
    const double *eta_r,
    const double *xB_r,  // 可为 NULL（minimize 模式）
    float *uxx, float *uyy, float *uzz,
    float *uxy, float *uxz, float *uyz,
    float S_11, float S_12, float S_13, float S_14, float S_15, float S_16,
    float S_22, float S_23, float S_24, float S_25, float S_26,
    float S_33, float S_34, float S_35, float S_36,
    float S_44, float S_45, float S_46,
    float S_55, float S_56,
    float S_66,
    float S_p_11, float S_p_12, float S_p_13, float S_p_14, float S_p_15, float S_p_16,
    float S_p_22, float S_p_23, float S_p_24, float S_p_25, float S_p_26,
    float S_p_33, float S_p_34, float S_p_35, float S_p_36,
    float S_p_44, float S_p_45, float S_p_46,
    float S_p_55, float S_p_56,
    float S_p_66,
    // Optimization(4): 需要 eps0 相关参数
    float eps_xx00, float eps_yy00, float eps_zz00,
    float eps_yz00, float eps_xz00, float eps_xy00,
    double eps_iso_over_vB,
    int gp_mode_enabled,
    int gp_elastic_enabled,
    double gp_eps_iso,
    float E0_xx, float E0_yy, float E0_zz, float E0_yz, float E0_xz, float E0_xy,
    int total_size)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_strain_with_perturbation_kernel<<<blocks, threads>>>(
        uxx_init, uyy_init, uzz_init, uxy_init, uxz_init, uyz_init,
        // Optimization(4): eigenstrain 参数已移除
        phi_r, eta_r, xB_r,
        uxx, uyy, uzz, uxy, uxz, uyz,
        S_11, S_12, S_13, S_14, S_15, S_16,
        S_22, S_23, S_24, S_25, S_26,
        S_33, S_34, S_35, S_36,
        S_44, S_45, S_46,
        S_55, S_56,
        S_66,
        S_p_11, S_p_12, S_p_13, S_p_14, S_p_15, S_p_16,
        S_p_22, S_p_23, S_p_24, S_p_25, S_p_26,
        S_p_33, S_p_34, S_p_35, S_p_36,
        S_p_44, S_p_45, S_p_46,
        S_p_55, S_p_56,
        S_p_66,
        eps_xx00, eps_yy00, eps_zz00, eps_yz00, eps_xz00, eps_xy00, eps_iso_over_vB,
        gp_mode_enabled, gp_elastic_enabled, gp_eps_iso,
        E0_xx, E0_yy, E0_zz, E0_yz, E0_xz, E0_xy,
        total_size);
}

// 使用有效弹性系数计算应力的kernel（考虑phi和S_p）
__global__ void compute_stress_from_strain_with_effective_stiffness_kernel(
    const float *uxx_r, const float *uyy_r, const float *uzz_r,
    const float *uxy_r, const float *uxz_r, const float *uyz_r,
    const double *phi_r,  // 相场，用于计算有效弹性系数
    const double *eta_r,  // gp_zone: 参与 eigenstrain / stiffness helper
    const double *xB_r,   // Optimization(4): 需要 xB 来现场计算 eps0（可为 NULL，minimize 模式）
    // Optimization(4): eigenstrain 参数已移除，现场计算；需要 eps0 相关参数
    float eps_xx00, float eps_yy00, float eps_zz00,
    float eps_yz00, float eps_xz00, float eps_xy00,
    double eps_iso_over_vB,
    int gp_mode_enabled,
    int gp_elastic_enabled,
    double gp_eps_iso,
    float *sigma_xx_r, float *sigma_yy_r, float *sigma_zz_r,
    float *sigma_xy_r, float *sigma_xz_r, float *sigma_yz_r,
    // 基体弹性刚度矩阵（Voigt记号）
    float S_11, float S_12, float S_13, float S_14, float S_15, float S_16,
    float S_22, float S_23, float S_24, float S_25, float S_26,
    float S_33, float S_34, float S_35, float S_36,
    float S_44, float S_45, float S_46,
    float S_55, float S_56,
    float S_66,
    // Perturbation参数（21个S_p_ij）
    float S_p_11, float S_p_12, float S_p_13, float S_p_14, float S_p_15, float S_p_16,
    float S_p_22, float S_p_23, float S_p_24, float S_p_25, float S_p_26,
    float S_p_33, float S_p_34, float S_p_35, float S_p_36,
    float S_p_44, float S_p_45, float S_p_46,
    float S_p_55, float S_p_56,
    float S_p_66,
    int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    // 读取总应变（Voigt记号：xx, yy, zz, yz, xz, xy）
    float eps_xx_total = uxx_r[idx];
    float eps_yy_total = uyy_r[idx];
    float eps_zz_total = uzz_r[idx];
    float eps_yz_total = uyz_r[idx];
    float eps_xz_total = uxz_r[idx];
    float eps_xy_total = uxy_r[idx];
    
    // Optimization(4): 现场计算 eigenstrain
    double phi = phi_r[idx];
    double eta = eta_r ? eta_r[idx] : 0.0;
    double xB = (xB_r != NULL) ? clamp01(xB_r[idx]) : 0.0;  // 如果 xB_r 为 NULL（minimize 模式），使用 0
    float eps_xx0_f, eps_yy0_f, eps_zz0_f;
    float eps_xy0_f, eps_xz0_f, eps_yz0_f;
    if (xB_r != NULL) {
        eigenstrain_phi_eta_xB_point(phi, eta, xB, eps_xx00, eps_yy00, eps_zz00,
                                     eps_yz00, eps_xz00, eps_xy00, eps_iso_over_vB,
                                     gp_mode_enabled, gp_elastic_enabled, gp_eps_iso,
                                     eps_xx0_f, eps_yy0_f, eps_zz0_f,
                                     eps_xy0_f, eps_xz0_f, eps_yz0_f);
    } else {
        eigenstrain_phi_only_point(phi, eps_xx00, eps_yy00, eps_zz00, eps_yz00, eps_xz00, eps_xy00,
                                    eps_xx0_f, eps_yy0_f, eps_zz0_f, eps_xy0_f, eps_xz0_f, eps_yz0_f);
    }
    
    // 计算弹性应变：eps_elastic = eps_total - eps_eigen
    float eps_xx = eps_xx_total - eps_xx0_f;
    float eps_yy = eps_yy_total - eps_yy0_f;
    float eps_zz = eps_zz_total - eps_zz0_f;
    float eps_yz = eps_yz_total - eps_yz0_f;
    float eps_xz = eps_xz_total - eps_xz0_f;
    float eps_xy = eps_xy_total - eps_xy0_f;
    
    // 计算有效弹性系数：S_eff = S_ij + phi * S_p_ij（用h(phi)控制）
    double h = stiffness_beta_weight_point(phi, eta, gp_mode_enabled, gp_elastic_enabled);
    float S_eff_11 = S_11 + (float)(h * S_p_11);
    float S_eff_12 = S_12 + (float)(h * S_p_12);
    float S_eff_13 = S_13 + (float)(h * S_p_13);
    float S_eff_14 = S_14 + (float)(h * S_p_14);
    float S_eff_15 = S_15 + (float)(h * S_p_15);
    float S_eff_16 = S_16 + (float)(h * S_p_16);
    float S_eff_22 = S_22 + (float)(h * S_p_22);
    float S_eff_23 = S_23 + (float)(h * S_p_23);
    float S_eff_24 = S_24 + (float)(h * S_p_24);
    float S_eff_25 = S_25 + (float)(h * S_p_25);
    float S_eff_26 = S_26 + (float)(h * S_p_26); 
    float S_eff_33 = S_33 + (float)(h * S_p_33);
    float S_eff_34 = S_34 + (float)(h * S_p_34);
    float S_eff_35 = S_35 + (float)(h * S_p_35);
    float S_eff_36 = S_36 + (float)(h * S_p_36);
    float S_eff_44 = S_44 + (float)(h * S_p_44);
    float S_eff_45 = S_45 + (float)(h * S_p_45);
    float S_eff_46 = S_46 + (float)(h * S_p_46);
    float S_eff_55 = S_55 + (float)(h * S_p_55);
    float S_eff_56 = S_56 + (float)(h * S_p_56);
    float S_eff_66 = S_66 + (float)(h * S_p_66);
    
    // Hooke定律：sigma = C_eff : epsilon，使用有效弹性系数
    // 在Voigt标记下，张量剪切应变项必须乘以 2.0 转换为工程应变参与乘法
    sigma_xx_r[idx] = S_eff_11*eps_xx + S_eff_12*eps_yy + S_eff_13*eps_zz + 2.0f*S_eff_14*eps_yz + 2.0f*S_eff_15*eps_xz + 2.0f*S_eff_16*eps_xy;
    sigma_yy_r[idx] = S_eff_12*eps_xx + S_eff_22*eps_yy + S_eff_23*eps_zz + 2.0f*S_eff_24*eps_yz + 2.0f*S_eff_25*eps_xz + 2.0f*S_eff_26*eps_xy;
    sigma_zz_r[idx] = S_eff_13*eps_xx + S_eff_23*eps_yy + S_eff_33*eps_zz + 2.0f*S_eff_34*eps_yz + 2.0f*S_eff_35*eps_xz + 2.0f*S_eff_36*eps_xy;
    sigma_yz_r[idx] = S_eff_14*eps_xx + S_eff_24*eps_yy + S_eff_34*eps_zz + 2.0f*S_eff_44*eps_yz + 2.0f*S_eff_45*eps_xz + 2.0f*S_eff_46*eps_xy;
    sigma_xz_r[idx] = S_eff_15*eps_xx + S_eff_25*eps_yy + S_eff_35*eps_zz + 2.0f*S_eff_45*eps_yz + 2.0f*S_eff_55*eps_xz + 2.0f*S_eff_56*eps_xy;
    sigma_xy_r[idx] = S_eff_16*eps_xx + S_eff_26*eps_yy + S_eff_36*eps_zz + 2.0f*S_eff_46*eps_yz + 2.0f*S_eff_56*eps_xz + 2.0f*S_eff_66*eps_xy;
}

void launch_compute_stress_from_strain_with_effective_stiffness_kernel(
    const float *uxx_r, const float *uyy_r, const float *uzz_r,
    const float *uxy_r, const float *uxz_r, const float *uyz_r,
    const double *phi_r,
    const double *eta_r,
    const double *xB_r,  // Optimization(4): 需要 xB 来现场计算 eps0（可为 NULL）
    // Optimization(4): eigenstrain 参数已移除，需要 eps0 相关参数
    float eps_xx00, float eps_yy00, float eps_zz00,
    float eps_yz00, float eps_xz00, float eps_xy00,
    double eps_iso_over_vB,
    int gp_mode_enabled,
    int gp_elastic_enabled,
    double gp_eps_iso,
    float *sigma_xx_r, float *sigma_yy_r, float *sigma_zz_r,
    float *sigma_xy_r, float *sigma_xz_r, float *sigma_yz_r,
    float S_11, float S_12, float S_13, float S_14, float S_15, float S_16,
    float S_22, float S_23, float S_24, float S_25, float S_26,
    float S_33, float S_34, float S_35, float S_36,
    float S_44, float S_45, float S_46,
    float S_55, float S_56,
    float S_66,
    float S_p_11, float S_p_12, float S_p_13, float S_p_14, float S_p_15, float S_p_16,
    float S_p_22, float S_p_23, float S_p_24, float S_p_25, float S_p_26,
    float S_p_33, float S_p_34, float S_p_35, float S_p_36,
    float S_p_44, float S_p_45, float S_p_46,
    float S_p_55, float S_p_56,
    float S_p_66,
    int total_size)
{
    int threads, blocks;
    configure_launch(total_size, threads, blocks);
    compute_stress_from_strain_with_effective_stiffness_kernel<<<blocks, threads>>>(
        uxx_r, uyy_r, uzz_r, uxy_r, uxz_r, uyz_r,
        phi_r, eta_r, xB_r,
        eps_xx00, eps_yy00, eps_zz00, eps_yz00, eps_xz00, eps_xy00, eps_iso_over_vB,
        gp_mode_enabled, gp_elastic_enabled, gp_eps_iso,
        sigma_xx_r, sigma_yy_r, sigma_zz_r, sigma_xy_r, sigma_xz_r, sigma_yz_r,
        S_11, S_12, S_13, S_14, S_15, S_16,
        S_22, S_23, S_24, S_25, S_26,
        S_33, S_34, S_35, S_36,
        S_44, S_45, S_46,
        S_55, S_56,
        S_66,
        S_p_11, S_p_12, S_p_13, S_p_14, S_p_15, S_p_16,
        S_p_22, S_p_23, S_p_24, S_p_25, S_p_26,
        S_p_33, S_p_34, S_p_35, S_p_36,
        S_p_44, S_p_45, S_p_46,
        S_p_55, S_p_56,
        S_p_66,
        total_size);
}

void gpu_reduce_min_max_meff(
    const double *phi_r,
    const double *xB_r,
    int total_size,
    double D_alpha,
    double D_compound,
    double Vm_alpha_0,
    double dVm_alpha_dxB,
    double Vm_compound,
    double temperature_K,
    double mu_reference_scale,
    double *min_val,
    double *max_val)
{
    if (total_size <= 0) {
        *min_val = 0.0;
        *max_val = 0.0;
        return;
    }
    
    int threads_per_block = 256;
    int blocks = (total_size + threads_per_block - 1) / threads_per_block;
    
    double *d_mins, *d_maxs;
    CUDA_CHECK(cudaMalloc(&d_mins, blocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_maxs, blocks * sizeof(double)));
    
    size_t shared_mem_size = 2 * threads_per_block * sizeof(double);
    reduce_min_max_meff_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        phi_r, xB_r, d_mins, d_maxs,
        D_alpha, D_compound, Vm_alpha_0, dVm_alpha_dxB, Vm_compound,
        temperature_K, mu_reference_scale, total_size);
    
    double *h_mins = (double*)malloc(blocks * sizeof(double));
    double *h_maxs = (double*)malloc(blocks * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_mins, d_mins, blocks * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_maxs, d_maxs, blocks * sizeof(double), cudaMemcpyDeviceToHost));
    
    double global_min = h_mins[0];
    double global_max = h_maxs[0];
    for (int i = 1; i < blocks; i++) {
        if (h_mins[i] < global_min) global_min = h_mins[i];
        if (h_maxs[i] > global_max) global_max = h_maxs[i];
    }
    
    *min_val = global_min;
    *max_val = global_max;
    
    free(h_mins);
    free(h_maxs);
    CUDA_CHECK(cudaFree(d_mins));
    CUDA_CHECK(cudaFree(d_maxs));
}
