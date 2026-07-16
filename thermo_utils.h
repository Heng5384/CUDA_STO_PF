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
//     return 41212.9 - 18.05 * T;
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 气体常数
#define R_GAS 8.31446261815324

// CUDA设备函数标记
#define DEVICE_FUNC __device__ __host__

#ifdef __CUDACC__
static __device__ __constant__ int d_thermo_convex_extrapolation_enabled = 1;
#endif
#ifdef THERMO_UTILS_DEFINE_GLOBALS
int h_thermo_convex_extrapolation_enabled = 1;
#else
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
    const double D0_cm2_s = 4.251e-11;
    const double Q_J_mol = 3.403e+04;
    const double D_cm2_s = D0_cm2_s * exp(-Q_J_mol / (R_GAS * T_K));
    return D_cm2_s * 1.0e-4;
}

// 定义凸化外推的临界浓度
// 当 xB 超过此值时，启用二次惩罚以保证热力学稳定性
// 0.15 通常是一个安全的选择 (大于溶解度 ~0.016，且小于 Spinodal点)
#define X_LIMIT_CONVEX 0.09

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
    if (T < 600.61) {
        return -7650.085 + 101.700244*T - 24.5242231*T*log(T) - 0.00365895*T*T - 2.4395e-7*pow(T, 3);
    } else {
        return -10531.095 + 154.243182*T - 32.4913959*T*log(T) + 0.00154613*T*T + 8.05448e25*pow(T, -9);
    }
}

DEVICE_FUNC static inline double GHSER_Ag(double T) {
    if (T < 1234.93) {
        return -7209.512 + 118.202013*T - 23.8463314*T*log(T) - 0.001790585*T*T - 3.98587e-7*pow(T, 3) - 12011.0/T;
    } else {
        return -15095.252 + 190.266404*T - 33.472*T*log(T) + 1.411773e29*pow(T, -9);
    }
}

DEVICE_FUNC static inline double GHSER_Te(double T) {
    if (T < 722.66) {
        return -10544.679 + 183.372894*T - 35.6687*T*log(T) + 0.01583435*T*T - 5.240417e-6*pow(T, 3) + 155015.0/T;
    } else {
        return 9160.595 - 129.265373*T + 13.004*T*log(T) - 0.0362361*T*T + 5.006367e-6*pow(T, 3) - 1.28681e30*pow(T, -9);
    }
}

// 1 mol PbTe 分子单元的标准吉布斯能
DEVICE_FUNC static inline double G_PbTe_Solid(double T) {
    double base = -76063.2138 + 9.67716633 * T;
    return base + GHSER_Pb(T) + GHSER_Te(T);
}

// 1 mol Ag2Te 分子单元的标准吉布斯能 (3倍原子能量)
DEVICE_FUNC static inline double G_Ag2Te_Solid(double T) {
    double base_per_atom = -10128.93 - 12.645115 * T;
    double G_atom = base_per_atom + (2.0/3.0)*GHSER_Ag(T) + (1.0/3.0)*GHSER_Te(T);
    return 3.0 * G_atom;
}

// =========================================================
// 2. 伪二元规则溶液模型 (Pseudo-Binary Regular Solution)
//    [核心修改区域：引入 Convex Extrapolation]
// =========================================================

// 相互作用参数 L(T)
DEVICE_FUNC static inline double get_L_param(double T) {
    return 41212.9 - 18.05 * T;
}

// 辅助函数：计算未修改的 CALPHAD 化学势 (PbTe)
DEVICE_FUNC static inline double mu_PbTe_calphad(double T, double xB) {
    double x = clamp_fraction_eps(xB);
    double G0 = G_PbTe_Solid(T);
    double L = get_L_param(T);
    return G0 + R_GAS * T * log(1.0 - x) + L * x * x;
}

// 辅助函数：计算未修改的 CALPHAD 化学势 (Ag2Te)
DEVICE_FUNC static inline double mu_Ag2Te_calphad(double T, double xB) {
    double x = clamp_fraction_eps(xB);
    double G0 = G_Ag2Te_Solid(T);
    double L = get_L_param(T);
    return G0 + R_GAS * T * log(x) + L * (1.0 - x) * (1.0 - x);
}

// 辅助函数：计算未修改的 d(mu_PbTe)/dx (解析导数)
DEVICE_FUNC static inline double dmu_PbTe_calphad_dx(double T, double xB) {
    double x = clamp_fraction_eps(xB);
    double L = get_L_param(T);
    // mu = G0 + RT*ln(1-x) + L*x^2
    // dmu/dx = -RT/(1-x) + 2Lx
    return -R_GAS * T / (1.0 - x) + 2.0 * L * x;
}

// 辅助函数：计算未修改的 d(mu_Ag2Te)/dx (解析导数)
DEVICE_FUNC static inline double dmu_Ag2Te_calphad_dx(double T, double xB) {
    double x = clamp_fraction_eps(xB);
    double L = get_L_param(T);
    // mu = G0 + RT*ln(x) + L*(1-x)^2
    // dmu/dx = RT/x - 2L(1-x)
    return R_GAS * T / x - 2.0 * L * (1.0 - x);
}

// Unified matrix molar free energy.  The optional high-composition branch is a
// numerical convex extension, not a calibrated high-x thermodynamic model.
DEVICE_FUNC static inline double g_alpha_calphad(double T, double xB) {
    const double x = clamp_fraction_eps(xB);
    const double GA = G_PbTe_Solid(T);
    const double GB = G_Ag2Te_Solid(T);
    const double L = get_L_param(T);
    return (1.0 - x) * GA + x * GB
         + R_GAS * T * ((1.0 - x) * log(1.0 - x) + x * log(x))
         + L * x * (1.0 - x);
}

DEVICE_FUNC static inline double g_alpha_prime_calphad(double T, double xB) {
    const double x = clamp_fraction_eps(xB);
    const double L = get_L_param(T);
    return G_Ag2Te_Solid(T) - G_PbTe_Solid(T)
         + R_GAS * T * log(x / (1.0 - x))
         + L * (1.0 - 2.0 * x);
}

DEVICE_FUNC static inline double g_alpha_second_calphad(double T, double xB) {
    const double x = clamp_fraction_eps(xB);
    return R_GAS * T / (x * (1.0 - x)) - 2.0 * get_L_param(T);
}

DEVICE_FUNC static inline double g_alpha_extension_K(double T) {
    const double xc = X_LIMIT_CONVEX;
    const double g_second_c = g_alpha_second_calphad(T, xc);
    const double ideal_curvature = R_GAS * T / (xc * (1.0 - xc));
    return (g_second_c > ideal_curvature) ? g_second_c : ideal_curvature;
}

DEVICE_FUNC static inline double g_alpha_raw(double T, double xB) {
    const double x = clamp_fraction_eps(xB);
    if (!thermo_convex_extrapolation_enabled_runtime() || x <= X_LIMIT_CONVEX) {
        return g_alpha_calphad(T, x);
    }
    const double xc = X_LIMIT_CONVEX;
    const double dx = x - xc;
    return g_alpha_calphad(T, xc)
         + g_alpha_prime_calphad(T, xc) * dx
         + 0.5 * g_alpha_extension_K(T) * dx * dx;
}

DEVICE_FUNC static inline double g_alpha_prime_raw(double T, double xB) {
    const double x = clamp_fraction_eps(xB);
    if (!thermo_convex_extrapolation_enabled_runtime() || x <= X_LIMIT_CONVEX) {
        return g_alpha_prime_calphad(T, x);
    }
    return g_alpha_prime_calphad(T, X_LIMIT_CONVEX)
         + g_alpha_extension_K(T) * (x - X_LIMIT_CONVEX);
}

DEVICE_FUNC static inline double g_alpha_second_raw(double T, double xB) {
    const double x = clamp_fraction_eps(xB);
    if (!thermo_convex_extrapolation_enabled_runtime() || x <= X_LIMIT_CONVEX) {
        return g_alpha_second_calphad(T, x);
    }
    return g_alpha_extension_K(T);
}

DEVICE_FUNC static inline double mu_PbTe_raw(double T, double xB) {
    const double x = clamp_fraction_eps(xB);
    const double g = g_alpha_raw(T, x);
    const double gp = g_alpha_prime_raw(T, x);
    return g - x * gp;
}

DEVICE_FUNC static inline double mu_Ag2Te_raw(double T, double xB) {
    const double x = clamp_fraction_eps(xB);
    const double g = g_alpha_raw(T, x);
    const double gp = g_alpha_prime_raw(T, x);
    return g + (1.0 - x) * gp;
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
    double L = get_L_param(T);
    double RT = R_GAS * T;

    // 初始猜测: 稀溶液近似 x ~ exp(-L/RT)
    double x = exp(-L / RT);

    // 边界保护
    if (x < 1e-9) x = 1e-9;
    if (x > 0.99) x = 0.5;

    // Newton-Raphson 迭代 (使用未修改的物理方程，因为平衡点肯定在正常区)
    for (int i = 0; i < 20; ++i) {
        double f = RT * log(x) + L * (1.0 - x) * (1.0 - x);
        double df = RT / x - 2.0 * L * (1.0 - x);

        double delta = f / df;
        x -= delta;

        if (x < 1e-10) x = 1e-10;
        if (x > 0.999) x = 0.999;

        if (fabs(delta) < 1e-8) break;
    }

    return x;
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

// Matrix thermodynamic factor from the same scalar free-energy backend.
DEVICE_FUNC static inline double gamma_thermo_nonlinear(double xB, double h,
                                            double Vm_alpha_0, double dVm_alpha_dxB,
                                            double Vm_compound,
                                            double temperature_K, double energy_scale){
    const double x = clamp_fraction_eps(xB);
    const double scale = (fabs(energy_scale) < 1e-30)
                           ? ((energy_scale >= 0.0) ? 1e-30 : -1e-30)
                           : energy_scale;
    const double c_bulk = c_xB_phi(x, Vm_alpha_0, dVm_alpha_dxB,
                                    Vm_compound, h);
    return c_bulk * g_alpha_second_raw(temperature_K, x) / scale;
}

// Candidate-only constitutive mobility.  No floor, absolute value, or sign
// correction is permitted: nonpositive/nonfinite Gamma is a gate failure.
DEVICE_FUNC static inline double matrix_mobility_alpha_candidate(
    double xB, double D_alpha,
    double Vm_alpha_0, double dVm_alpha_dxB, double Vm_compound,
    double temperature_K, double energy_scale) {
    const double gamma_alpha = gamma_thermo_nonlinear(
        xB, 0.0, Vm_alpha_0, dVm_alpha_dxB, Vm_compound,
        temperature_K, energy_scale);
    if (!isfinite(D_alpha) || D_alpha < 0.0 ||
        !isfinite(gamma_alpha) || gamma_alpha <= 0.0) {
        return NAN;
    }
    return D_alpha / gamma_alpha;
}

DEVICE_FUNC static inline double matrix_capacity_mobility_candidate(
    double h, double xB, double D_alpha,
    double Vm_alpha_0, double dVm_alpha_dxB, double Vm_compound,
    double temperature_K, double energy_scale,
    double matrix_support_eps) {
    const double alpha = 1.0 - h;
    if (!isfinite(alpha) || !isfinite(matrix_support_eps) ||
        matrix_support_eps < 0.0 || alpha > 1.0) return NAN;
    // One-sided closed beta support. The quintic h evaluation can exceed one
    // by roundoff near phi=1; only the same inactive-support tolerance band is
    // interpreted as the exact alpha=0 removable limit. A materially negative
    // capacity remains a constitutive failure and is not clipped or hidden.
    if (alpha < -matrix_support_eps) return NAN;
    if (alpha <= matrix_support_eps) return 0.0;
    const double M_alpha = matrix_mobility_alpha_candidate(
        xB, D_alpha, Vm_alpha_0, dVm_alpha_dxB, Vm_compound,
        temperature_K, energy_scale);
    return alpha * M_alpha;
}

// Coarse4-only numerical closure.  The exact legacy expression is returned
// when a_M=0 so every default path retains its prior floating-point operation
// sequence.  b=4h(1-h) vanishes in both bulks and is nonnegative on the
// admissible phase interval.
DEVICE_FUNC static inline double matrix_capacity_mobility_coarse_candidate(
    double h, double xB, double D_alpha,
    double Vm_alpha_0, double dVm_alpha_dxB, double Vm_compound,
    double temperature_K, double energy_scale,
    double matrix_support_eps, double a_M) {
    const double base = matrix_capacity_mobility_candidate(
        h, xB, D_alpha, Vm_alpha_0, dVm_alpha_dxB, Vm_compound,
        temperature_K, energy_scale, matrix_support_eps);
    if (a_M == 0.0 || !isfinite(base) || base == 0.0) return base;
    if (!isfinite(a_M) || a_M < 0.0 || !isfinite(h) ||
        h < 0.0 || h > 1.0) return NAN;
    const double b = 4.0 * h * (1.0 - h);
    const double factor = 1.0 + a_M * b;
    return isfinite(factor) && factor >= 1.0 ? base * factor : NAN;
}

#endif // THERMO_UTILS_H
