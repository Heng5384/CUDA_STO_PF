// #ifndef THERMO_UTILS_H
// #define THERMO_UTILS_H

// #include <math.h>

// #ifndef M_PI
// #define M_PI 3.14159265358979323846
// #endif

// // 气体常数
// #define R_GAS 8.31446261815324

// // CUDA设备函数标记
// #define DEVICE_FUNC __device__ __host__

// // 将数值限制在 [0,1] 内
// DEVICE_FUNC static inline double clamp01(double value){
//     if(value < 0.0) return 0.0;
//     if(value > 1.0) return 1.0;
//     return value;
// }

// // 浓度限制，防止 log(0)
// DEVICE_FUNC static inline double clamp_fraction_eps(double x){
//     if(x < 1e-12) return 1e-12;
//     if(x > 1.0 - 1e-12) return 1.0 - 1e-12;
//     return x;
// }

// // =========================================================
// // 1. SGTE 基础热力学数据 (Gierlotka 2010)
// // =========================================================

// DEVICE_FUNC static inline double GHSER_Pb(double T) {
//     if (T < 600.61) {
//         return -7650.085 + 101.700244*T - 24.5242231*T*log(T) - 0.00365895*T*T - 2.4395e-7*pow(T, 3);
//     } else {
//         return -10531.095 + 154.243182*T - 32.4913959*T*log(T) + 0.00154613*T*T + 8.05448e25*pow(T, -9);
//     }
// }

// DEVICE_FUNC static inline double GHSER_Ag(double T) {
//     if (T < 1234.93) {
//         return -7209.512 + 118.202013*T - 23.8463314*T*log(T) - 0.001790585*T*T - 3.98587e-7*pow(T, 3) - 12011.0/T;
//     } else {
//         return -15095.252 + 190.266404*T - 33.472*T*log(T) + 1.411773e29*pow(T, -9);
//     }
// }

// DEVICE_FUNC static inline double GHSER_Te(double T) {
//     if (T < 722.66) {
//         return -10544.679 + 183.372894*T - 35.6687*T*log(T) + 0.01583435*T*T - 5.240417e-6*pow(T, 3) + 155015.0/T;
//     } else {
//         return 9160.595 - 129.265373*T + 13.004*T*log(T) - 0.0362361*T*T + 5.006367e-6*pow(T, 3) - 1.28681e30*pow(T, -9);
//     }
// }

// // 1 mol PbTe 分子单元的标准吉布斯能
// DEVICE_FUNC static inline double G_PbTe_Solid(double T) {
//     double base = -76063.2138 + 9.67716633 * T;
//     return base + GHSER_Pb(T) + GHSER_Te(T);
// }

// // 1 mol Ag2Te 分子单元的标准吉布斯能 (3倍原子能量)
// DEVICE_FUNC static inline double G_Ag2Te_Solid(double T) {
//     double base_per_atom = -10128.93 - 12.645115 * T;
//     double G_atom = base_per_atom + (2.0/3.0)*GHSER_Ag(T) + (1.0/3.0)*GHSER_Te(T);
//     return 3.0 * G_atom;
// }

// // =========================================================
// // 2. 伪二元规则溶液模型 (Pseudo-Binary Regular Solution)
// // =========================================================

// // 相互作用参数 L(T)
// DEVICE_FUNC static inline double get_L_param(double T) {
//     return 41504.29119633958 - 18.469276826409214 * T;
// }

// // 溶剂 PbTe (Matrix) 的化学势
// // mu_PbTe = G0_PbTe + RT * ln(1-x) + L * x^2
// DEVICE_FUNC static inline double mu_PbTe_raw(double T, double xB) {
//     double x = clamp_fraction_eps(xB);
//     double G0 = G_PbTe_Solid(T);
//     double L = get_L_param(T);
//     return G0 + R_GAS * T * log(1.0 - x) + L * x * x;
// }

// // 溶质 Ag2Te (Precipitate) 的化学势
// // mu_Ag2Te = G0_Ag2Te + RT * ln(x) + L * (1-x)^2
// DEVICE_FUNC static inline double mu_Ag2Te_raw(double T, double xB) {
//     double x = clamp_fraction_eps(xB);
//     double G0 = G_Ag2Te_Solid(T);
//     double L = get_L_param(T);
//     return G0 + R_GAS * T * log(x) + L * (1.0 - x) * (1.0 - x);
// }

// // =========================================================
// // 3. 辅助接口函数 (无量纲化 & 混合)
// // =========================================================

// // 对应 mu_A (Matrix/Solvent) -> PbTe
// DEVICE_FUNC static inline double mu_A_dimless(double xB, double temperature_K, double energy_scale){
//     double scale = (fabs(energy_scale) < 1e-30) ? ((energy_scale >= 0.0) ? 1e-30 : -1e-30) : energy_scale;
//     return mu_PbTe_raw(temperature_K, xB) / scale;
// }

// // 对应 mu_B (Precipitate/Solute) -> Ag2Te
// DEVICE_FUNC static inline double mu_B_dimless(double xB, double temperature_K, double energy_scale){
//     double scale = (fabs(energy_scale) < 1e-30) ? ((energy_scale >= 0.0) ? 1e-30 : -1e-30) : energy_scale;
//     return mu_Ag2Te_raw(temperature_K, xB) / scale;
// }

// // =========================================================
// // 4. 平衡浓度求解器 (Newton-Raphson on Device)
// // =========================================================

// // 在 GPU 上实时计算平衡浓度 x_eq
// // 求解方程: RT * ln(x) + L * (1-x)^2 = 0
// // 这是基于规则溶液模型的固溶度极限 (假设与纯 Ag2Te 平衡)
// DEVICE_FUNC static inline double solve_x_eq_device(double T) {
//     double L = get_L_param(T);
//     double RT = R_GAS * T;
    
//     // 初始猜测: 稀溶液近似 x ~ exp(-L/RT)
//     double x = exp(-L / RT);
    
//     // 边界保护初始值
//     if (x < 1e-9) x = 1e-9;
//     if (x > 0.99) x = 0.5; // 如果 L 很小或负值，避免初始值过大

//     // Newton-Raphson 迭代
//     // f(x) = RT*ln(x) + L*(1-x)^2
//     // f'(x) = RT/x - 2*L*(1-x)
//     for (int i = 0; i < 20; ++i) {
//         double f = RT * log(x) + L * (1.0 - x) * (1.0 - x);
//         double df = RT / x - 2.0 * L * (1.0 - x);
        
//         double delta = f / df;
//         x -= delta;
        
//         // 强制边界约束，防止 x <= 0 导致 log 崩溃
//         if (x < 1e-10) x = 1e-10;
//         if (x > 0.999) x = 0.999;
        
//         if (fabs(delta) < 1e-8) break; // 收敛
//     }
    
//     return x;
// }

// // 替换旧的 xB_eq_from_temperature，直接调用求解器
// DEVICE_FUNC static inline double xB_eq_from_temperature(double temperature_K){
//     return solve_x_eq_device(temperature_K);
// }

// // =========================================================
// // 5. 其他工具函数 (保持不变)
// // =========================================================

// // 基体体积的线性近似
// DEVICE_FUNC static inline double Vm_alpha_of_xB(double xB, double Vm_alpha_0, double dVm_alpha_dxB){
//     return Vm_alpha_0 + dVm_alpha_dxB * xB;
// }

// // 由体积加权得到的浓度系数 c(xB, φ)
// DEVICE_FUNC static inline double c_xB_phi(double xB, double Vm_alpha_0, double dVm_alpha_dxB, double Vm_compound, double h){
//     double Vm_alpha = Vm_alpha_of_xB(xB, Vm_alpha_0, dVm_alpha_dxB);
//     double denom = Vm_alpha * (1.0 - h) + Vm_compound * h;
//     if(fabs(denom) < 1e-12){
//         denom = (denom >= 0.0) ? 1e-12 : -1e-12;
//     }
//     return 1.0 / denom;
// }

// // 计算混合化学势
// DEVICE_FUNC static inline double mu_tot_mix(double mu_A, double mu_B, double xB,
//                                 double mu0_compound, double h){
//     double mu_mix = mu_A * (1.0 - xB) + mu_B * xB;
//     return (1.0 - h) * mu_mix + h * mu0_compound;
// }

// // 以化学计量系数加权得到化合物参考化学势
// DEVICE_FUNC static inline double v_weighted_mu_compound(double muA_eq, double muB_eq,
//                                             double v_A, double v_B){
//     double v_sum = v_A + v_B;
//     if(fabs(v_sum) < 1e-30) v_sum = (v_sum >= 0.0) ? 1e-30 : -1e-30;
//     return (v_A * muA_eq + v_B * muB_eq) / v_sum;
// }

// // logit 转换
// DEVICE_FUNC static inline double sigmoid_from_logit(double Y, double Y_clip, double eps){
//     if(Y > Y_clip) Y = Y_clip;
//     if(Y < -Y_clip) Y = -Y_clip;
//     // 使用更稳定的 sigmoid 计算方式
//     double xB;
//     if (Y >= 0) {
//         double e_negY = exp(-Y);
//         xB = 1.0 / (1.0 + e_negY);
//     } else {
//         double e_Y = exp(Y);
//         xB = e_Y / (1.0 + e_Y);
//     }
//     if(xB < eps) xB = eps;
//     if(xB > 1.0 - eps) xB = 1.0 - eps;
//     return xB;
// }

// DEVICE_FUNC static inline double logit_from_fraction(double xB, double eps, double Y_clip){
//     if(xB < eps) xB = eps;
//     if(xB > 1.0 - eps) xB = 1.0 - eps;
//     double Y = log(xB / (1.0 - xB));
//     if(Y > Y_clip) return Y_clip;
//     if(Y < -Y_clip) return -Y_clip;
//     return Y;
// }

// // 扩散系数 D(φ)
// DEVICE_FUNC static inline double D_mix(double h, double D_alpha, double D_compound){
//     return (1.0 - h) * D_alpha + h * D_compound;
// }

// // 热力学因子 Γ (更新为使用新的化学势函数)
// DEVICE_FUNC static inline double gamma_thermo_nonlinear(double xB, double h,
//                                             double Vm_alpha_0, double dVm_alpha_dxB,
//                                             double Vm_compound,
//                                             double temperature_K, double energy_scale){
//     // 稳定性增强：当 xB 超过限制时，固定热力学因子在 0.08 处的值，防止进入 spinodal 区域导致 Meff 发散
//     double xB_limit = 0.08;
//     double xB_calc = (xB > xB_limit) ? xB_limit : xB;
    
//     double c_bulk = c_xB_phi(xB_calc, Vm_alpha_0, dVm_alpha_dxB, Vm_compound, h);
//     double dx = 1e-5; // 稍微减小差分步长
//     double x_plus = clamp_fraction_eps(xB_calc + dx);
//     double x_minus = clamp_fraction_eps(xB_calc - dx);
    
//     // 注意：这里使用的是新的 mu_A_dimless (PbTe) 和 mu_B_dimless (Ag2Te)
//     double dmuA_dx = (mu_A_dimless(x_plus, temperature_K, energy_scale) -
//                       mu_A_dimless(x_minus, temperature_K, energy_scale)) / (x_plus - x_minus);
//     double dmuB_dx = (mu_B_dimless(x_plus, temperature_K, energy_scale) -
//                       mu_B_dimless(x_minus, temperature_K, energy_scale)) / (x_plus - x_minus);
                      
//     return c_bulk * (dmuB_dx - dmuA_dx);
// }

// #endif // THERMO_UTILS_H

#ifndef THERMO_UTILS_H
#define THERMO_UTILS_H

#include <math.h>
#include "generated/pf_kwn_validation_contract_v1.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// The validation control consumes the generated view of the canonical JSON.
// These compatibility names deliberately reference generated constants so the
// PF path has no independently hand-maintained thermodynamic literals.
#define R_GAS PF_KWN_R_GAS
#define THERMO_DELTA_H_J_PER_MOL PF_KWN_DELTA_H_J_PER_MOL
#define THERMO_DELTA_S_J_PER_MOL_K PF_KWN_DELTA_S_J_PER_MOL_K
#define THERMO_CONTRACT_VERSION PF_KWN_VALIDATION_CONTRACT_SCHEMA

// CUDA设备函数标记. The host-side probe also includes this header.
#if defined(__CUDACC__)
#define DEVICE_FUNC __device__ __host__
#else
#define DEVICE_FUNC
#endif

#ifdef THERMO_UTILS_DEFINE_GLOBALS
#ifdef __CUDACC__
__device__ __constant__ int d_thermo_convex_extrapolation_enabled =
    PF_KWN_CONVEX_EXTRAPOLATION_ENABLED;
#endif
int h_thermo_convex_extrapolation_enabled =
    PF_KWN_CONVEX_EXTRAPOLATION_ENABLED;
#else
#ifdef __CUDACC__
extern __device__ __constant__ int d_thermo_convex_extrapolation_enabled;
#endif
extern int h_thermo_convex_extrapolation_enabled;
#endif

DEVICE_FUNC static inline int thermo_convex_extrapolation_enabled_runtime(void) {
#if defined(__CUDA_ARCH__)
    return d_thermo_convex_extrapolation_enabled;
#else
    return h_thermo_convex_extrapolation_enabled;
#endif
}

// D(Ag in PbTe), m^2/s; mirrors Unit_Psedobinary.py:D_Ag_in_PbTe_m2_per_s
DEVICE_FUNC static inline double D_Ag_in_PbTe_m2_per_s(double T_K) {
    return pf_kwn_D_alpha_m2_s(T_K);
}

// The validation contract owns this optional stabilization branch.  It is
// currently disabled, but its latent threshold and penalty remain generated
// rather than hand-maintained PF-side literals.
#define X_LIMIT_CONVEX PF_KWN_CONVEX_EXTRAPOLATION_XB_LIMIT
#define THERMO_CONVEX_PENALTY_J_PER_MOL \
    PF_KWN_CONVEX_EXTRAPOLATION_PENALTY_J_PER_MOL

// 将数值限制在 [0,1] 内
DEVICE_FUNC static inline double clamp01(double value){
    if(value < 0.0) return 0.0;
    if(value > 1.0) return 1.0;
    return value;
}

// 浓度限制，防止 log(0)
DEVICE_FUNC static inline double clamp_fraction_eps(double x){
    if(x < 1e-12) return 1e-12;
    if(x > 1.0 - 1e-12) return 1.0 - 1e-12;
    return x;
}

// =========================================================
// 1. SGTE 基础热力学数据 (Gierlotka 2010)
// =========================================================

DEVICE_FUNC static inline double GHSER_Pb(double T) {
    return pf_kwn_GHSER_Pb(T);
}

DEVICE_FUNC static inline double GHSER_Ag(double T) {
    return pf_kwn_GHSER_Ag(T);
}

DEVICE_FUNC static inline double GHSER_Te(double T) {
    return pf_kwn_GHSER_Te(T);
}

// 1 mol PbTe 分子单元的标准吉布斯能
DEVICE_FUNC static inline double G_PbTe_Solid(double T) {
    return pf_kwn_G_PbTe(T);
}

// 1 mol Ag2Te 分子单元的标准吉布斯能 (3倍原子能量)
DEVICE_FUNC static inline double G_Ag2Te_Solid(double T) {
    return pf_kwn_G_Ag2Te(T);
}

// =========================================================
// 2. 伪二元规则溶液模型 (Pseudo-Binary Regular Solution)
//    [核心修改区域：引入 Convex Extrapolation]
// =========================================================

// 相互作用参数 L(T)
DEVICE_FUNC static inline double get_L_param(double T) {
    return pf_kwn_L(T);
}

// 辅助函数：计算未修改的 CALPHAD 化学势 (PbTe)
DEVICE_FUNC static inline double mu_PbTe_calphad(double T, double xB) {
    return pf_kwn_mu_A(T, xB);
}

// 辅助函数：计算未修改的 CALPHAD 化学势 (Ag2Te)
DEVICE_FUNC static inline double mu_Ag2Te_calphad(double T, double xB) {
    return pf_kwn_mu_B(T, xB);
}

// 辅助函数：计算未修改的 d(mu_PbTe)/dx (解析导数)
DEVICE_FUNC static inline double dmu_PbTe_calphad_dx(double T, double xB) {
    double x = clamp_fraction_eps(xB);
    // mu = G0 + RT*ln(1-x) + L*x^2
    // dmu/dx = -RT/(1-x) + 2Lx
    return -PF_KWN_R_GAS * T / (1.0 - x) + 2.0 * pf_kwn_L(T) * x;
}

// 辅助函数：计算未修改的 d(mu_Ag2Te)/dx (解析导数)
DEVICE_FUNC static inline double dmu_Ag2Te_calphad_dx(double T, double xB) {
    double x = clamp_fraction_eps(xB);
    // mu = G0 + RT*ln(x) + L*(1-x)^2
    // dmu/dx = RT/x - 2L(1-x)
    return PF_KWN_R_GAS * T / x - 2.0 * pf_kwn_L(T) * (1.0 - x);
}

DEVICE_FUNC static inline double dmu_PbTe_raw_dx(double T, double xB) {
    if (!thermo_convex_extrapolation_enabled_runtime() ||
        xB <= X_LIMIT_CONVEX) {
        return dmu_PbTe_calphad_dx(T, xB);
    }
    return dmu_PbTe_calphad_dx(T, X_LIMIT_CONVEX) +
           2.0 * THERMO_CONVEX_PENALTY_J_PER_MOL * (xB - X_LIMIT_CONVEX);
}

DEVICE_FUNC static inline double dmu_Ag2Te_raw_dx(double T, double xB) {
    if (!thermo_convex_extrapolation_enabled_runtime() ||
        xB <= X_LIMIT_CONVEX) {
        return dmu_Ag2Te_calphad_dx(T, xB);
    }
    return dmu_Ag2Te_calphad_dx(T, X_LIMIT_CONVEX) +
           2.0 * THERMO_CONVEX_PENALTY_J_PER_MOL * (xB - X_LIMIT_CONVEX);
}

// --- [核心修改] 安全的化学势函数 (带凸化外推) ---

// 溶剂 PbTe (Matrix) 的化学势
DEVICE_FUNC static inline double mu_PbTe_raw(double T, double xB) {
    if (!thermo_convex_extrapolation_enabled_runtime() || xB <= X_LIMIT_CONVEX) {
        // 正常区域：使用物理模型
        return mu_PbTe_calphad(T, xB);
    } else {
        // 虚构区域：凸化外推
        // mu(x) = mu(xc) + slope * (x - xc) + K * (x - xc)^2
        double xc = X_LIMIT_CONVEX;
        double mu_c = mu_PbTe_calphad(T, xc);
        double slope_c = dmu_PbTe_calphad_dx(T, xc);
        double delta_x = xB - xc;
        return mu_c + slope_c * delta_x +
               THERMO_CONVEX_PENALTY_J_PER_MOL * delta_x * delta_x;
    }
}

// 溶质 Ag2Te (Precipitate) 的化学势
DEVICE_FUNC static inline double mu_Ag2Te_raw(double T, double xB) {
    if (!thermo_convex_extrapolation_enabled_runtime() || xB <= X_LIMIT_CONVEX) {
        // 正常区域：使用物理模型
        return mu_Ag2Te_calphad(T, xB);
    } else {
        // 虚构区域：凸化外推
        double xc = X_LIMIT_CONVEX;
        double mu_c = mu_Ag2Te_calphad(T, xc);
        double slope_c = dmu_Ag2Te_calphad_dx(T, xc);
        double delta_x = xB - xc;
        return mu_c + slope_c * delta_x +
               THERMO_CONVEX_PENALTY_J_PER_MOL * delta_x * delta_x;
    }
}

// =========================================================
// 3. 辅助接口函数 (无量纲化 & 混合)
// =========================================================

// 对应 mu_A (Matrix/Solvent) -> PbTe
DEVICE_FUNC static inline double mu_A_dimless(double xB, double temperature_K, double energy_scale){
    double scale = (fabs(energy_scale) < 1e-30) ? ((energy_scale >= 0.0) ? 1e-30 : -1e-30) : energy_scale;
    return mu_PbTe_raw(temperature_K, xB) / scale;
}

// 对应 mu_B (Precipitate/Solute) -> Ag2Te
DEVICE_FUNC static inline double mu_B_dimless(double xB, double temperature_K, double energy_scale){
    double scale = (fabs(energy_scale) < 1e-30) ? ((energy_scale >= 0.0) ? 1e-30 : -1e-30) : energy_scale;
    return mu_Ag2Te_raw(temperature_K, xB) / scale;
}

// =========================================================
// 4. 平衡浓度求解器 (Newton-Raphson on Device)
// =========================================================

// 在 GPU 上实时计算平衡浓度 x_eq
DEVICE_FUNC static inline double solve_x_eq_device(double T) {
    return pf_kwn_planar_solvus(T);
}

// 替换旧的 xB_eq_from_temperature
DEVICE_FUNC static inline double xB_eq_from_temperature(double temperature_K){
    return solve_x_eq_device(temperature_K);
}

// =========================================================
// 5. 其他工具函数
// =========================================================

// 基体体积的线性近似
DEVICE_FUNC static inline double Vm_alpha_of_xB(double xB, double Vm_alpha_0, double dVm_alpha_dxB){
    return Vm_alpha_0 + dVm_alpha_dxB * xB;
}

// 由体积加权得到的浓度系数 c(xB, φ)
DEVICE_FUNC static inline double c_xB_phi(double xB, double Vm_alpha_0, double dVm_alpha_dxB, double Vm_compound, double h){
    double Vm_alpha = Vm_alpha_of_xB(xB, Vm_alpha_0, dVm_alpha_dxB);
    double denom = Vm_alpha * (1.0 - h) + Vm_compound * h;
    if(fabs(denom) < 1e-12){
        denom = (denom >= 0.0) ? 1e-12 : -1e-12;
    }
    return 1.0 / denom;
}

// 计算混合化学势
DEVICE_FUNC static inline double mu_tot_mix(double mu_A, double mu_B, double xB,
                                double mu0_compound, double h){
    double mu_mix = mu_A * (1.0 - xB) + mu_B * xB;
    return (1.0 - h) * mu_mix + h * mu0_compound;
}

// 以化学计量系数加权得到化合物参考化学势
DEVICE_FUNC static inline double v_weighted_mu_compound(double muA_eq, double muB_eq,
                                            double v_A, double v_B){
    double v_sum = v_A + v_B;
    if(fabs(v_sum) < 1e-30) v_sum = (v_sum >= 0.0) ? 1e-30 : -1e-30;
    return (v_A * muA_eq + v_B * muB_eq) / v_sum;
}

// logit 转换
DEVICE_FUNC static inline double sigmoid_from_logit(double Y, double Y_clip, double eps){
    if(Y > Y_clip) Y = Y_clip;
    if(Y < -Y_clip) Y = -Y_clip;
    double xB;
    if (Y >= 0) {
        double e_negY = exp(-Y);
        xB = 1.0 / (1.0 + e_negY);
    } else {
        double e_Y = exp(Y);
        xB = e_Y / (1.0 + e_Y);
    }
    if(xB < eps) xB = eps;
    if(xB > 1.0 - eps) xB = 1.0 - eps;
    return xB;
}

DEVICE_FUNC static inline double logit_from_fraction(double xB, double eps, double Y_clip){
    if(xB < eps) xB = eps;
    if(xB > 1.0 - eps) xB = 1.0 - eps;
    double Y = log(xB / (1.0 - xB));
    if(Y > Y_clip) return Y_clip;
    if(Y < -Y_clip) return -Y_clip;
    return Y;
}

// 扩散系数 D(φ)
DEVICE_FUNC static inline double D_mix(double h, double D_alpha, double D_compound){
    return (1.0 - h) * D_alpha + h * D_compound;
}

// 热力学因子 Γ (使用差分法计算二阶导)
DEVICE_FUNC static inline double gamma_thermo_nonlinear(double xB, double h,
                                            double Vm_alpha_0, double dVm_alpha_dxB,
                                            double Vm_compound,
                                            double temperature_K, double energy_scale){
    // Any high-composition limiter is owned by the same generated validation
    // contract as the chemical-potential branch.
    double xB_calc = xB;
    if (thermo_convex_extrapolation_enabled_runtime() &&
        xB_calc > X_LIMIT_CONVEX) {
        xB_calc = X_LIMIT_CONVEX;
    }

    double c_bulk = c_xB_phi(xB_calc, Vm_alpha_0, dVm_alpha_dxB, Vm_compound, h);
    double dx = 1e-5; 
    double x_plus = clamp_fraction_eps(xB_calc + dx);
    double x_minus = clamp_fraction_eps(xB_calc - dx);
    
    double dmuA_dx = (mu_A_dimless(x_plus, temperature_K, energy_scale) -
                      mu_A_dimless(x_minus, temperature_K, energy_scale)) / (x_plus - x_minus);
    double dmuB_dx = (mu_B_dimless(x_plus, temperature_K, energy_scale) -
                      mu_B_dimless(x_minus, temperature_K, energy_scale)) / (x_plus - x_minus);
                      
    // 返回 thermo_factor = c_bulk * d(mu_B - mu_A)/dx
    // 由于凸化，d(mu_B - mu_A)/dx 在高浓度区也是正的且很大的
    return c_bulk * (dmuB_dx - dmuA_dx);
}

#endif // THERMO_UTILS_H
