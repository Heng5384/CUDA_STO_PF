#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cufft.h>
#include <cuComplex.h>

enum {
    MASS_DIAG_PHI_SUM_XBTOT_RAW = 0,
    MASS_DIAG_PHI_SUM_XBTOT_CLAMPED,
    MASS_DIAG_PHI_SUM_H_BEFORE,
    MASS_DIAG_PHI_SUM_H_AFTER,
    MASS_DIAG_PHI_SUM_XB_BEFORE,
    MASS_DIAG_PHI_SUM_PHI_BEFORE,
    MASS_DIAG_PHI_SUM_PHI_AFTER,
    MASS_DIAG_PHI_SUM_PRED_DELTA_XBTOT,
    MASS_DIAG_PHI_CLIP_LOW_COUNT,
    MASS_DIAG_PHI_CLIP_HIGH_COUNT,
    MASS_DIAG_PHI_STATS_COUNT
};

enum {
    MASS_DIAG_Y_SUM_XBTOT_RAWY = 0,
    MASS_DIAG_Y_SUM_XBTOT_YCLAMP_PRE_XBCLIP,
    MASS_DIAG_Y_SUM_XBTOT_FINAL,
    MASS_DIAG_Y_SUM_XB_BEFORE,
    MASS_DIAG_Y_SUM_XB_RAWY,
    MASS_DIAG_Y_SUM_XB_YCLAMP_PRE_XBCLIP,
    MASS_DIAG_Y_SUM_XB_FINAL,
    MASS_DIAG_Y_CLIP_LOW_COUNT,
    MASS_DIAG_Y_CLIP_HIGH_COUNT,
    MASS_DIAG_Y_XB_CLIP_LOW_COUNT,
    MASS_DIAG_Y_XB_CLIP_HIGH_COUNT,
    MASS_DIAG_Y_STATS_COUNT
};

enum {
    MASS_DIAG_Y_RHS_SUM_H_OLD = 0,
    MASS_DIAG_Y_RHS_SUM_H_NEW,
    MASS_DIAG_Y_RHS_SUM_HQ,
    MASS_DIAG_Y_RHS_SUM_GAMMA_LOCAL,
    MASS_DIAG_Y_RHS_SUM_TERM_H,
    MASS_DIAG_Y_RHS_SUM_TERM_LAP,
    MASS_DIAG_Y_RHS_SUM_TERM_GAMMA,
    MASS_DIAG_Y_RHS_SUM_DIVJ,
    MASS_DIAG_Y_RHS_SUM_LAPY,
    MASS_DIAG_Y_RHS_SUM_TOTAL,
    MASS_DIAG_Y_RHS_SUMABS_DIVJ,
    MASS_DIAG_Y_RHS_SUMABS_TERM_H,
    MASS_DIAG_Y_RHS_SUMABS_TERM_LAP,
    MASS_DIAG_Y_RHS_SUMABS_TERM_GAMMA,
    MASS_DIAG_Y_RHS_SUMABS_TOTAL,
    MASS_DIAG_Y_RHS_SUMSQ_DIVJ,
    MASS_DIAG_Y_RHS_SUMSQ_TERM_H,
    MASS_DIAG_Y_RHS_SUMSQ_TERM_LAP,
    MASS_DIAG_Y_RHS_SUMSQ_TERM_GAMMA,
    MASS_DIAG_Y_RHS_SUMSQ_TOTAL,
    MASS_DIAG_Y_RHS_MIN_H,
    MASS_DIAG_Y_RHS_MIN_HQ,
    MASS_DIAG_Y_RHS_MAX_HQ,
    MASS_DIAG_Y_RHS_MAXABS_DIVJ,
    MASS_DIAG_Y_RHS_MAXABS_TERM_H,
    MASS_DIAG_Y_RHS_MAXABS_TERM_LAP,
    MASS_DIAG_Y_RHS_MAXABS_TERM_GAMMA,
    MASS_DIAG_Y_RHS_MAXABS_TOTAL,
    MASS_DIAG_Y_RHS_MAXABS_LAGGED_DYDT,
    MASS_DIAG_Y_RHS_STATS_COUNT
};

enum {
    MASS_DIAG_GP_STORAGE_SUM_XBTOT_BEFORE = 0,
    MASS_DIAG_GP_STORAGE_SUM_XB_ALPHA_BEFORE,
    MASS_DIAG_GP_STORAGE_SUM_STORAGE_GP_BEFORE,
    MASS_DIAG_GP_STORAGE_SUM_STORAGE_BETA_BEFORE,
    MASS_DIAG_GP_STORAGE_SUM_XBTOT_AFTER,
    MASS_DIAG_GP_STORAGE_SUM_XB_ALPHA_AFTER,
    MASS_DIAG_GP_STORAGE_SUM_STORAGE_GP_AFTER,
    MASS_DIAG_GP_STORAGE_SUM_STORAGE_BETA_AFTER,
    MASS_DIAG_GP_STORAGE_SUM_DELTA_XB_ALPHA,
    MASS_DIAG_GP_STORAGE_SUM_DELTA_STORAGE_GP,
    MASS_DIAG_GP_STORAGE_SUM_DELTA_STORAGE_BETA,
    MASS_DIAG_GP_STORAGE_SUM_DT_DIVJ,
    MASS_DIAG_GP_STORAGE_SUM_CLOSURE_ERROR,
    MASS_DIAG_GP_STORAGE_SUM_DIVJ,
    MASS_DIAG_GP_STORAGE_SMALL_HALPHA_COUNT,
    MASS_DIAG_GP_STORAGE_MIN_HALPHA,
    MASS_DIAG_GP_STORAGE_MAX_HALPHA,
    MASS_DIAG_GP_STORAGE_MIN_HGP,
    MASS_DIAG_GP_STORAGE_MAX_HGP,
    MASS_DIAG_GP_STORAGE_MIN_ETA,
    MASS_DIAG_GP_STORAGE_MAX_ETA,
    MASS_DIAG_GP_STORAGE_MIN_XB_ALPHA,
    MASS_DIAG_GP_STORAGE_MAX_XB_ALPHA,
    MASS_DIAG_GP_STORAGE_STATS_COUNT
};

enum {
    GP_Y_UPDATE_SUM_CLIPPING_MASS_ERROR = 0,
    GP_Y_UPDATE_SUM_SMALL_HALPHA_MASS_ERROR,
    GP_Y_UPDATE_Y_CLIP_COUNT,
    GP_Y_UPDATE_XB_CLIP_COUNT,
    GP_Y_UPDATE_SMALL_HALPHA_COUNT,
    GP_Y_UPDATE_STATS_COUNT
};

enum {
    GP_TRANSPORT_MAXABS_XB_PERTURB = 0,
    GP_TRANSPORT_MIN_HALPHA,
    GP_TRANSPORT_MAX_HALPHA,
    GP_TRANSPORT_MIN_M_ALPHA,
    GP_TRANSPORT_MAX_M_ALPHA,
    GP_TRANSPORT_MIN_M_EFF,
    GP_TRANSPORT_MAX_M_EFF,
    GP_TRANSPORT_STATS_COUNT
};

enum {
    GP_ETA_FEAS_SUM_RECOVERED_XB_BEFORE = 0,
    GP_ETA_FEAS_MIN_RECOVERED_XB_BEFORE,
    GP_ETA_FEAS_MAX_RECOVERED_XB_BEFORE,
    GP_ETA_FEAS_GT_XBMAX_COUNT_BEFORE,
    GP_ETA_FEAS_LT_XBMIN_COUNT_BEFORE,
    GP_ETA_FEAS_INVALID_COUNT_BEFORE,
    GP_ETA_FEAS_SUM_RECOVERED_XB_AFTER,
    GP_ETA_FEAS_MIN_RECOVERED_XB_AFTER,
    GP_ETA_FEAS_MAX_RECOVERED_XB_AFTER,
    GP_ETA_FEAS_GT_XBMAX_COUNT_AFTER,
    GP_ETA_FEAS_LT_XBMIN_COUNT_AFTER,
    GP_ETA_FEAS_INVALID_COUNT_AFTER,
    GP_ETA_FEAS_INVALID_MASS,
    GP_ETA_FEAS_MAX_ETA_TENT,
    GP_ETA_FEAS_MAX_HGP_NEW,
    GP_ETA_FEAS_MAX_ETA_NEW,
    GP_ETA_FEAS_LIMITER_COUNT,
    GP_ETA_FEAS_LIMITER_MAX_DELTA,
    GP_ETA_FEAS_LIMITER_MASS_PREVENTED,
    GP_ETA_FEAS_STATS_COUNT
};

enum {
    GP_ELASTIC_STATS_SUM_GEL = 0,
    GP_ELASTIC_STATS_MAX_GEL,
    GP_ELASTIC_STATS_MIN_SIGMA_HYDRO,
    GP_ELASTIC_STATS_MAX_SIGMA_HYDRO,
    GP_ELASTIC_STATS_MIN_EPS0_GP_DIAG,
    GP_ELASTIC_STATS_MAX_EPS0_GP_DIAG,
    GP_ELASTIC_STATS_SUM_DGEL_DETA,
    GP_ELASTIC_STATS_MAXABS_DGEL_DETA,
    GP_ELASTIC_STATS_SUM_DGEL_DPHI,
    GP_ELASTIC_STATS_MAXABS_DGEL_DPHI,
    GP_ELASTIC_STATS_SUM_ETA_RHS_ELASTIC,
    GP_ELASTIC_STATS_MAXABS_ETA_RHS_ELASTIC,
    GP_ELASTIC_STATS_SUM_PHI_RHS_ELASTIC,
    GP_ELASTIC_STATS_MAXABS_PHI_RHS_ELASTIC,
    GP_ELASTIC_STATS_SUM_MINUS_DELTA_MU_R_GP,
    GP_ELASTIC_STATS_MIN_MINUS_DELTA_MU_R_GP,
    GP_ELASTIC_STATS_MAX_MINUS_DELTA_MU_R_GP,
    GP_ELASTIC_STATS_COUNT
};

// 注意：这些函数在cuda_kernels.cu中实现
// 它们负责配置并启动对应的CUDA kernel

// Kernel函数声明（在.cu文件中实现）
// 这些函数可以在主机端调用，会自动处理kernel启动

// 去混叠kernel（double complex 用于 phi 等；float complex 用于弹性）
void launch_dealias_kernel(cuDoubleComplex *f_k, int Nx, int Ny, int Nz, int NzC,
                           double dx, double dy, double dz, int total_size);
void launch_dealias_float_kernel(cufftComplex *f_k, int Nx, int Ny, int Nz, int NzC,
                                 double dx, double dy, double dz, int total_size);

// 相场方程RHS计算（含弹性项）
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
                                   int gp_mode_enabled,
                                   int gp_elastic_enabled,
                                   int gp_elastic_active_phi,
                                   double gp_eps_iso,
                                   double gp_elastic_derivative_scale,
                                   double elastic_shift_dimless,
                                   int disable_chem,
                                   int total_size,
                                   int elastic_enabled,
                                   double *diag_stats);

// 相场方程半隐式更新
void launch_phi_semi_implicit_update_kernel(const cuDoubleComplex *phi_k_old,
                                             const cuDoubleComplex *rhs_k,
                                             const double *k2,
                                             cuDoubleComplex *phi_k_new,
                                             double L_phi, double kappa_phi,
                                             double dt, int total_size);

// GP-zone eta 方程：RHS、半隐式更新与截断
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
                                   double *diag_stats);
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
                                              int total_size);
void launch_eta_semi_implicit_update_kernel(const cuDoubleComplex *eta_k_old,
                                            const cuDoubleComplex *rhs_k,
                                            const double *k2,
                                            cuDoubleComplex *eta_k_new,
                                            double L_eta, double kappa_eta,
                                            double dt, int total_size);
void launch_eta_normalize_and_clamp_kernel(double *eta_r, double invN, int total_size);
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
                                                  int total_size);

// phi归一化和截断
void launch_phi_normalize_and_clamp_kernel(double *phi_r, double invN, int total_size);
// 仅归一化（无截断）供非phi/Y数组使用
void launch_normalize_only_kernel(double *arr, double invN, int total_size);
void launch_add_arrays_kernel(const double *a, const double *b, double *out, int total_size);
// 计算h(phi)并做体积分数归约
__global__ void compute_h_values_kernel(const double *phi_r, double *h_values, int total_size);
double gpu_compute_vf_from_h(const double *d_phi_r, int total_size);

// minimize mode: reconstruct xB from phi via xB = (1-h(phi))*xBout + h(phi)*xBeq
void launch_reconstruct_xB_from_phi_kernel(const double *phi_r, double *xB_r,
                                           double xB_out, double xB_eq,
                                           int total_size);

// volume constraint: rhs += lambda * h'(phi)
void launch_add_volume_constraint_kernel(const double *phi_r, double *rhs_r,
                                         double lambda,
                                         int total_size);

// volume constraint (lagrange mode helpers):
// out[idx] = h'(phi)
void launch_compute_hprime_values_kernel(const double *phi_r, double *out_r, int total_size);
// out[idx] = (h'(phi))^2
void launch_compute_hprime_sq_values_kernel(const double *phi_r, double *out_r, int total_size);
// out[idx] = h'(phi) * rhs(phi)
void launch_compute_hprime_times_rhs_kernel(const double *phi_r, const double *rhs_r,
                                            double *out_r, int total_size);

// Full Euler-Lagrange residual: res = δF/δφ - λ h' (residual_r may alias lap_phi_r)
void launch_compute_euler_lagrange_residual_kernel(const double *rhs_r,
                                                   const double *lap_phi_r,
                                                   const double *phi_r,
                                                   double kappa,
                                                   double lambda_vol,
                                                   double *residual_r,
                                                   int total_size);

// Post-update volume projection: phi_corrected[i] = phi[i] + lambda_correct * h_prime_of_phi(phi[i])
void launch_apply_volume_projection_kernel(double *phi_r, double lambda_correct, int total_size);

// bulk chemical free-energy excess density:
// g_bulk_hat(r) = c_tot_hat * [ (1-h(phi)) * mu_mix_hat(xB(r)) + h(phi) * mu0_compound_hat ]
// g_bulk0_hat   = c_tot_hat * mu_mix_hat(xB_ref)
// out[idx]      = g_bulk_hat(r) - g_bulk0_hat
void launch_compute_gbulk_excess_hat_kernel(const double *phi_r, const double *xB_r,
                                            double *gbulk_excess_r,
                                            double temperature_K, double mu_reference_scale,
                                            double mu0_compound,
                                            double xB_ref, double g_bulk0_hat,
                                            int total_size);

void launch_compute_grad_energy_density_kernel(const double *phi_r, double *e_grad_r,
                                               int Nx, int Ny, int Nz,
                                               double dx, double dy, double dz,
                                               double kappa_phi,
                                               int total_size);

// Y方程：计算mu_x（含弹性修正）
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
                                int elastic_enabled);

// gp_zone: 计算 matrix composition 的 mu_C，不把 h_alpha 乘进 chemical potential
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
                                   int elastic_enabled);

// minimize mode helper: compute mu_x directly from (phi, xB), without using Y/logit.
// This is useful when xB is reconstructed from phi each iteration.
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
                                        int elastic_enabled);

// minimize mode chain rule: rhs_phi += (∂f/∂xB) * (dxB/dphi)
// with xB(phi) = (1-h)*xBout + h*xBeq  =>  dxB/dphi = (xBeq-xBout)*h'(phi)
void launch_add_xB_chain_rule_to_phi_rhs_kernel(const double *phi_r,
                                                const double *mu_x_r,
                                                double *rhs_phi_r,
                                                double xB_out, double xB_eq,
                                                int total_size);

// 计算delta_mu_r（用于VTK输出）
void launch_compute_delta_mu_r_kernel(const double *phi_r, const double *xB_r,
                                     double *delta_mu_r,
                                     double temperature_K, double mu_reference_scale,
                                     double v_A, double v_B, double mu0_compound,
                                     double elastic_shift_dimless,
                                     int total_size);

// 计算f_phi_chem和dgel_dphi（用于VTK输出）
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
    int elastic_enabled);

// 缩放数组：arr[i] *= factor
void launch_scale_array_kernel(double *arr, double factor, int total_size);
void launch_subtract_scaled_kernel(double *rhs_r, const double *lap_r, double scale, int total_size);

// Optimization: 2-slot scratch 用 - 拆分 4 场为 OutA(chem,dw) + OutB(bulk,dgel)
void launch_compute_f_phi_chem_f_phi_dw_kernel(const double *phi_r, const double *xB_r,
    double *f_phi_chem_r, double *f_phi_dw_r,
    double temperature_K, double mu_reference_scale, double v_A, double v_B,
    double mu0_compound, double Vm_compound, double Vm_alpha_0, double dVm_alpha_dxB,
    double W, double elastic_shift_dimless, int total_size);
void launch_compute_f_phi_bulk_dgel_dphi_kernel(const double *phi_r, const double *xB_r,
    double *f_phi_bulk_r, double *dgel_dphi_r,
    double temperature_K, double mu_reference_scale, double v_A, double v_B,
    double mu0_compound, double Vm_compound, double Vm_alpha_0, double dVm_alpha_dxB,
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
    double eps_iso_over_vB, double elastic_shift_dimless, int total_size, int elastic_enabled);
void launch_compute_driving_force_direct_kernel(const double *phi_r, const double *xB_r,
    const double *f_phi_grad_r, double *driving_force_r,
    double temperature_K, double mu_reference_scale, double v_A, double v_B,
    double mu0_compound, double Vm_compound, double Vm_alpha_0, double dVm_alpha_dxB,
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
    double eps_iso_over_vB, int total_size, int elastic_enabled);

// 四项求和：out = (dw + bulk + grad + el)；其中 el 可为 NULL（视为 0）
void launch_sum_driving_force_kernel(const double *dw, const double *bulk,
                                     const double *grad, const double *el,
                                     double *out, int total_size);

// 计算弹性能密度 gel_hat（默认按 0.5*sigma:epsilon_elastic）
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
    int total_size);

void launch_compute_gp_elastic_stats_kernel(
    const double *phi_r,
    const double *eta_r,
    const double *gel_hat_r,
    const float *sigma_xx_r,
    const float *sigma_yy_r,
    const float *sigma_zz_r,
    double gp_eps_iso,
    double *stats,
    int total_size);

// Y方程：计算梯度
void launch_compute_gradient_k_kernel(const cuDoubleComplex *f_k,
                                       cuDoubleComplex *grad_x_k,
                                       cuDoubleComplex *grad_y_k,
                                       cuDoubleComplex *grad_z_k,
                                       int Nx, int Ny, int Nz, int NzC,
                                       double dx, double dy, double dz, int total_size);

// Y方程：计算通量
void launch_compute_flux_kernel(const double *grad_mu_x_r, const double *grad_mu_y_r,
                                 const double *grad_mu_z_r, const double *phi_r,
                                 const double *xB_prev_r, double *Jx_r, double *Jy_r,
                                 double *Jz_r, double D_alpha, double D_compound,
                                 double Vm_alpha_0, double dVm_alpha_dxB,
                                 double Vm_compound, double temperature_K,
                                 double mu_reference_scale, int total_size);

// Y方程：计算散度
void launch_compute_divergence_k_kernel(const cuDoubleComplex *Jx_k,
                                         const cuDoubleComplex *Jy_k,
                                         const cuDoubleComplex *Jz_k,
                                         cuDoubleComplex *divJ_k,
                                         int Nx, int Ny, int Nz, int NzC,
                                         double dx, double dy, double dz, int total_size);

// Optimization: divJ 按方向串行累加 - 单分量梯度/通量/累加
void launch_compute_gradient_single_component_k_kernel(
    const cuDoubleComplex *f_k,
    cuDoubleComplex *grad_alpha_k,
    int alpha,
    int Nx, int Ny, int Nz, int NzC,
    double dx, double dy, double dz,
    int total_size);
void launch_compute_flux_single_component_kernel(
    const double *grad_mu_alpha_r,
    const double *phi_r,
    const double *xB_prev_r,
    double *J_alpha_r,
    double D_alpha, double D_compound,
    double Vm_alpha_0, double dVm_alpha_dxB,
    double Vm_compound, double temperature_K,
    double mu_reference_scale,
    int total_size);
void launch_compute_flux_single_component_q_kernel(
    const double *grad_mu_alpha_r, const double *phi_r,
    const double *xB_prev_r, double *J_alpha_r,
    double D_alpha, double Vm_alpha_0, double dVm_alpha_dxB,
    double Vm_compound, double temperature_K,
    double mu_reference_scale, int total_size);
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
    int total_size);
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
                                              int total_size);
void launch_divJ_accumulate_kernel(
    const cuDoubleComplex *J_alpha_k,
    cuDoubleComplex *divJ_k,
    int alpha,
    int Nx, int Ny, int Nz, int NzC,
    double dx, double dy, double dz,
    int total_size);
void launch_zero_divJ_k_kernel(cuDoubleComplex *divJ_k, int total_size);

// Y方程：计算RHS
void launch_compute_Y_rhs_kernel(const double *divJ_r, const double *phi_r,
                                  const double *phi_prev, const double *lapY_r,
                                  const double *Y_r, const double *dY_dt_prev,
                                  double *rhs_r, double dt, double v_B,
                                  double mean_DY, int total_size,
                                  int use_previous_time_level,
                                  int disable_gamma_term,
                                  double term_h_scale,
                                  double *diag_stats);
void launch_compute_Y_rhs_gp_kernel(const double *divJ_r, const double *phi_r,
                                     const double *phi_prev, const double *eta_r,
                                     const double *eta_prev_r, const double *lapY_r,
                                     const double *Y_r, const double *dY_dt_prev,
                                     double *rhs_r, double dt, double xB_GP,
                                     double mean_DY, double gp_h_alpha_eps,
                                     int total_size,
                                     int use_previous_time_level,
                                     int disable_gamma_term,
                                     double *diag_stats);
void launch_compute_Y_rhs_gp_conservative_kernel(const double *divJ_r, const double *phi_r,
                                                 const double *phi_prev, const double *eta_r,
                                                 const double *eta_prev_r, const double *lapY_r,
                                                 const double *Y_r, const double *dY_dt_prev,
                                                 double *rhs_r, double dt, double xB_GP,
                                                 double mean_DY, int total_size,
                                                 int use_previous_time_level,
                                                 int disable_gamma_term,
                                                 double *diag_stats);

// Y方程：半隐式更新
void launch_Y_semi_implicit_update_kernel(const cuDoubleComplex *Y_k_old,
                                           const cuDoubleComplex *rhs_k,
                                           const double *k2,
                                           cuDoubleComplex *Y_k_new,
                                           double mean_DY, double dt, int total_size);

// Y归一化和截断
void launch_Y_normalize_and_clamp_kernel(double *Y_r, double *xB_r,
                                          double invN, double Y_clip, double Y_upper_cap, double xB_eps,
                                          int total_size);
void launch_relax_dY_dt_guess_kernel(const double *dY_dt_new_r,
                                     const double *dY_dt_old_guess_r,
                                     double *dY_dt_guess_r,
                                     double omega,
                                     int total_size);
void launch_compute_diff_sumsq_max_kernel(const double *a_r,
                                          const double *b_r,
                                          double *stats,
                                          int total_size);

void launch_phi_mass_diagnostics_kernel(const double *phi_ifft_r,
                                        const double *phi_before_r,
                                        const double *xB_before_r,
                                        double *stats,
                                        double invN,
                                        double v_B,
                                        int total_size);

void launch_Y_mass_diagnostics_kernel(const double *Y_ifft_r,
                                      const double *phi_r,
                                      const double *xB_before_r,
                                      double *stats,
                                      double invN,
                                      double Y_clip,
                                      double Y_upper_cap,
                                      double xB_eps,
                                      double v_B,
                                      int total_size);
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
                                         int total_size);
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
                                             int total_size);
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
                                                    int total_size);
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
                                          int total_size);
void launch_xB_normalize_clamp_and_logit_kernel(double *xB_ifft_r,
                                                double *xB_r,
                                                double *Y_r,
                                                double invN,
                                                double xB_eps,
                                                double Y_clip,
                                                double Y_upper_cap,
                                                int total_size);

enum PFQTransportStatsIndex {
    PF_Q_SUM_DELTA_H = 0,
    PF_Q_SUM_LOCAL_DELTA_Q = 1,
    PF_Q_SUM_LOCAL_RESIDUAL = 2,
    PF_Q_MAXABS_LOCAL_RESIDUAL = 3,
    PF_Q_INFEASIBLE_PHASE_COUNT = 4,
    PF_Q_TRANSPORT_BOUND_VIOLATION_COUNT = 5,
    PF_Q_TRANSPORT_BOUND_MASS = 6,
    PF_Q_MIN_ALPHA = 7,
    PF_Q_STATS_COUNT = 8
};

enum PFConservativeStatsIndex {
    PF_CONS_SUM_SIGNED_TRANSFER = 0,
    PF_CONS_SUM_ABS_TRANSFER = 1,
    PF_CONS_MAXABS_FACE_TRANSFER = 2,
    PF_CONS_LIMITED_FACE_COUNT = 3,
    PF_CONS_BOUND_VIOLATION_COUNT = 4,
    PF_CONS_PHASE_CONSTRAINT_COUNT = 5,
    PF_CONS_MAXABS_PHASE_CORRECTION = 6,
    PF_CONS_BETA_CONTEXT_COUNT = 7,
    PF_CONS_NONFINITE_COUNT = 8,
    PF_CONS_STATS_COUNT = 9
};

void launch_initialize_conservative_storage_kernel(
    const double *phi_r, const double *xB_r, double *storage_r,
    double v_B, int primary_is_ctot, int total_size);
void launch_reconstruct_conservative_context_kernel(
    const double *storage_r, const double *phi_r, double *xB_r, double *Y_r,
    double temperature_K, double v_B, int primary_is_ctot,
    double beta_support_eps, double xB_eps, double Y_clip,
    double *stats, int total_size);
void launch_pairwise_conservative_face_sweep_kernel(
    double *storage_r, const double *mu_r, const double *phi_r,
    const double *xB_r, int Nx, int Ny, int Nz, int axis, int parity,
    double spacing, double dt, double D_alpha,
    double Vm_alpha_0, double dVm_alpha_dxB, double Vm_compound,
    double temperature_K, double mu_reference_scale,
    double v_B, int primary_is_ctot, int backward_euler,
    double bound_tol, double *stats, int total_size);
void launch_constrain_phase_and_reconstruct_conservative_kernel(
    double *phi_new_r, const double *phi_old_r, double *storage_r,
    double *xB_r, double *Y_r, double temperature_K, double v_B,
    int primary_is_ctot, double beta_support_eps, double xB_eps,
    double Y_clip, double bound_tol, double *stats, int total_size);

void launch_initialize_q_alpha_kernel(const double *phi_r,
                                      const double *xB_r,
                                      double *q_alpha_r,
                                      int total_size);
void launch_apply_local_phase_storage_transfer_q_kernel(
    const double *phi_new_r, const double *phi_old_r,
    double *q_alpha_r, double v_B, double feasibility_tol,
    double *stats, int total_size);
void launch_compute_q_transport_rhs_kernel(const double *divJ_r,
                                           const double *lap_q_r,
                                           double *rhs_q_r,
                                           double mean_Dq,
                                           int total_size);
void launch_q_normalize_validate_and_reconstruct_kernel(
    double *q_ifft_r, double *q_alpha_r, const double *phi_r,
    double *xB_r, double *Y_r, double invN, double xB_eps,
    double Y_clip, double Y_upper_cap, double bound_tol,
    double *stats, int total_size);
void launch_q_explicit_transport_and_reconstruct_kernel(
    double *q_alpha_r, const double *divJ_r, const double *phi_r,
    double *xB_r, double *Y_r, double dt, double xB_eps,
    double Y_clip, double Y_upper_cap, double bound_tol,
    double *stats, int total_size);
void launch_sync_q_from_phi_xB_kernel(const double *phi_r,
                                      const double *xB_r,
                                      double *q_alpha_r,
                                      int total_size);
void launch_apply_Y_shift_recompute_xB_kernel(const double *Y_base_r,
                                              double *Y_r,
                                              double *xB_r,
                                              double lambda_shift,
                                              double Y_clip,
                                              double Y_upper_cap,
                                              double xB_eps,
                                              int total_size);
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
                                              int total_size);
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
                                          int total_size);

// xB 峰值限幅 kernel
void launch_clamp_xB_max_kernel(double *xB_r, int total_size, double xB_max);

// 计算Laplacian
void launch_compute_laplacian_k_kernel(const cuDoubleComplex *f_k, const double *k2,
                                        cuDoubleComplex *lap_k, int total_size);

// 计算xBtot
void launch_compute_xBtot_kernel(const double *phi_r, const double *xB_r,
                                  double *xBtot_r, double v_B, int total_size);

// 计算 gp_zone 模式的 xBtot_gp
void launch_compute_xBtot_gp_kernel(const double *phi_r, const double *eta_r,
                                    const double *xB_alpha_r, double *xBtot_gp_r,
                                    double xB_GP, int total_size);

// 计算DY值（用于归约）
void launch_compute_DY_values_kernel(const double *Y_r, const double *phi_r,
                                     double *DY_values, double D_alpha,
                                     double D_compound, int total_size);

// 更新dY_dt_prev（在Y更新后调用）
void launch_update_dY_dt_prev_kernel(const double *Y_r, const double *Y_n_saved,
                                     double *dY_dt_prev_r, double dt, int total_size);

// 归约求和（辅助函数）
double gpu_reduce_sum(const double *d_array, int n);

// 诊断统计：计算N_in和N_if（旧版本，需要中间存储数组）
void launch_compute_diagnostics_stats_kernel(const double *phi_r,
                                             double *d_stats,
                                             int total_size);

// 优化版本：直接归约计算N_in和N_if（节省显存，不需要中间数组）
void gpu_reduce_sum_N_in_N_if(const double *phi_r, int total_size,
                               double *N_in_sum, double *N_if_sum);

// 优化版本：直接归约计算xBtot的和（节省显存）
double gpu_reduce_sum_xBtot(const double *phi_r, const double *xB_r,
                            double v_B, int total_size);

// 优化版本：直接归约计算xBtot_gp的和（节省显存）
double gpu_reduce_sum_xBtot_gp(const double *phi_r, const double *eta_r,
                               const double *xB_alpha_r, double xB_GP,
                               int total_size);

// 优化版本：直接归约计算DY的和（节省显存）
double gpu_reduce_sum_DY(const double *Y_r, const double *phi_r,
                        double D_alpha, double D_compound, int total_size);
double gpu_reduce_sum_DY_gp(const double *Y_r, const double *phi_r,
                            const double *eta_r, double D_alpha,
                            int total_size);

// 优化版本：直接归约计算min和max（节省CPU内存和传输时间）
void gpu_reduce_min_max(const double *d_array, int n,
                       double *min_val, double *max_val);

// 优化版本：直接归约计算Meff的min和max
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
    double *max_val);

// 优化版本：计算析出相内部（phi>0.5）的平均hydrostatic stress
double gpu_reduce_avg_sigma_hydro_in_precipitate(
    const double *phi_r,
    const float *sigma_xx_r,
    const float *sigma_yy_r,
    const float *sigma_zz_r,
    int total_size);

// 优化版本：从grad_mu直接计算divJ，内部分配临时数组，节省显存
// 这个函数内部会分配临时的J数组（实空间和k空间），计算完成后自动释放
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
    int total_k);

// 初始化：GPU并行初始化phi场
void launch_initialize_phi_kernel(
    double *d_phi_r,
    const double *d_centers,
    int N_seeds,
    int Nx, int Ny, int Nz,
    double dx, double dy, double dz,
    double Lx, double Ly, double Lz,
    double R, double w,
    int total_size);

// 弹性相关kernel（新增）
// ============================================

// 1. Eigenstrain计算（从phi, xB 计算 Wu & Ji 形式的 eigenstrain）
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
    int total_size);

// Minimize mode only: eigenstrain from phi only (no xB)
void launch_compute_eigenstrain_from_phi_only_kernel(
    const double *phi_r,
    float *uxx0_r, float *uyy0_r, float *uzz0_r,
    float *uxy0_r, float *uxz0_r, float *uyz0_r,
    float eps_xx00, float eps_yy00, float eps_zz00,
    float eps_yz00, float eps_xz00, float eps_xy00,
    int total_size);

// Minimize mode: rhs = δF/δphi = double-well (W*g') + elastic (no chemical)
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
    int elastic_enabled);

// Minimize mode: double-well energy density e_dw = W*g(φ)
void launch_compute_dw_energy_density_kernel(const double *phi_r, double *e_dw_r, double W, int total_size);

// 位移场计算（k空间，使用均匀弹性常数）
void launch_compute_displacement_from_eigenstrain_k_kernel(
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
    int total_k);

// 应变场计算（k空间，从位移计算应变）
void launch_compute_strain_from_displacement_k_kernel(
    const cufftComplex *ux_k, const cufftComplex *uy_k, const cufftComplex *uz_k,
    cufftComplex *uxx_k, cufftComplex *uyy_k, cufftComplex *uzz_k,
    cufftComplex *uxy_k, cufftComplex *uxz_k, cufftComplex *uyz_k,
    int Nx, int Ny, int Nz, int NzC,
    double dx, double dy, double dz,
    int total_k);

// 应变场计算（实空间，考虑perturbation和外部应变）
void launch_compute_strain_with_perturbation_kernel(
    // 输入：从位移计算得到的初始应变场
    const float *uxx_init, const float *uyy_init, const float *uzz_init,
    const float *uxy_init, const float *uxz_init, const float *uyz_init,
    // Optimization(4): eigenstrain 参数已移除，现场计算
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
    // Perturbation参数（21个S_p_ij，直接传入而不是数组）
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
    // 外部应变（6个分量）
    float E0_xx, float E0_yy, float E0_zz, float E0_yz, float E0_xz, float E0_xy,
    int total_size);

// 位移归一化kernel
void launch_normalize_displacement_kernel(
    float *ux_r, float *uy_r, float *uz_r,
    float invN,  // 1/(Nx*Ny*Nz)
    int total_size);

void launch_normalize_strain_kernel(
    float *uxx_r, float *uyy_r, float *uzz_r,
    float *uxy_r, float *uxz_r, float *uyz_r,
    float invN,  // 1/(Nx*Ny*Nz)
    int total_size);

// 添加外部应变（全局添加，不乘以插值函数）
void launch_add_external_strain_kernel(
    float *uxx_r, float *uyy_r, float *uzz_r,
    float *uxy_r, float *uxz_r, float *uyz_r,
    float E0_xx, float E0_yy, float E0_zz,
    float E0_yz, float E0_xz, float E0_xy,
    int total_size);

// Green函数方法：从hij计算新的k空间位移（第二次迭代及之后）
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
    int total_k);

// 从应变计算应力（使用phi插值得到有效刚度S_eff = S + h(phi) * S_p）
// 注意：应力计算使用弹性应变 = 总应变 - eigenstrain
void launch_compute_stress_from_strain_with_effective_stiffness_kernel(
    const float *uxx_r, const float *uyy_r, const float *uzz_r,
    const float *uxy_r, const float *uxz_r, const float *uyz_r,
    const double *phi_r,
    const double *eta_r,
    // Optimization(4): eigenstrain 参数已移除，现场计算；需要 xB（可为 NULL，minimize 模式）
    const double *xB_r,
    // Optimization(4): 需要 eps0 相关参数
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
    int total_size);

#endif // CUDA_KERNELS_H
