// GENERATED FILE. DO NOT EDIT.
// Source: contracts/pf_kwn_validation_contract_v1.json
// Generator: tools/generate_pf_contract_header.py
#ifndef PF_KWN_VALIDATION_CONTRACT_V1_H
#define PF_KWN_VALIDATION_CONTRACT_V1_H

#include <math.h>

#if defined(__CUDACC__)
#define PF_KWN_HD __host__ __device__
#else
#define PF_KWN_HD
#endif

#define PF_KWN_VALIDATION_CONTRACT_HASH "71359665955cf45531e2b1b22fec8a4fd47cdcf3f207e0336f75a6b02e7405e5"
#define PF_KWN_VALIDATION_CONTRACT_SCHEMA "PF_KWN_VALIDATION_CONTRACT_V1"

static constexpr double PF_KWN_R_GAS = 8.3144626181532395;
static constexpr double PF_KWN_DELTA_H_J_PER_MOL = 41504.291196339582;
static constexpr double PF_KWN_DELTA_S_J_PER_MOL_K = 18.469276826409214;
static constexpr double PF_KWN_D0_CM2_PER_S = 4.2509999999999997e-11;
static constexpr double PF_KWN_ACTIVATION_ENERGY_J_PER_MOL = 34030;
static constexpr double PF_KWN_GAMMA_J_PER_M2 = 0.16800000000000001;
static constexpr double PF_KWN_VM_ALPHA_M3_PER_MOL = 4.1009e-05;
static constexpr double PF_KWN_VM_BETA_M3_PER_MOL = 4.1009e-05;
static constexpr double PF_KWN_V_B = 1;

PF_KWN_HD static inline double pf_kwn_clamp_fraction(double x) {
    return x < 1.0e-12 ? 1.0e-12 : (x > 1.0 - 1.0e-12 ? 1.0 - 1.0e-12 : x);
}

PF_KWN_HD static inline double pf_kwn_standard_state_piecewise(
    double temperature_K, double transition_K,
    double a0_low, double a1_low, double alog_low, double a2_low, double a3_low, double ainv_low, double pinv_low,
    double a0_high, double a1_high, double alog_high, double a2_high, double a3_high, double ainv_high, double pinv_high) {
    const bool low = temperature_K < transition_K;
    const double a0 = low ? a0_low : a0_high;
    const double a1 = low ? a1_low : a1_high;
    const double alog = low ? alog_low : alog_high;
    const double a2 = low ? a2_low : a2_high;
    const double a3 = low ? a3_low : a3_high;
    const double ainv = low ? ainv_low : ainv_high;
    const double pinv = low ? pinv_low : pinv_high;
    return a0 + a1 * temperature_K + alog * temperature_K * log(temperature_K) +
           a2 * temperature_K * temperature_K + a3 * temperature_K * temperature_K * temperature_K +
           (ainv == 0.0 ? 0.0 : ainv * pow(temperature_K, pinv));
}

PF_KWN_HD static inline double pf_kwn_GHSER_Pb(double T) {
    return pf_kwn_standard_state_piecewise(T, 600.61000000000001, -7650.085, 101.700244, -24.5242231, -0.0036589499999999998, -2.4395000000000001e-07, 0, 0, -10531.094999999999, 154.24318199999999, -32.491395900000001, 0.0015461299999999999, 0, 8.0544800000000007e+25, -9);
}
PF_KWN_HD static inline double pf_kwn_GHSER_Ag(double T) {
    return pf_kwn_standard_state_piecewise(T, 1234.9300000000001, -7209.5119999999997, 118.20201299999999, -23.8463314, -0.0017905849999999999, -3.9858699999999999e-07, -12011, -1, -15095.252, 190.26640399999999, -33.472000000000001, 0, 0, 1.4117729999999999e+29, -9);
}
PF_KWN_HD static inline double pf_kwn_GHSER_Te(double T) {
    return pf_kwn_standard_state_piecewise(T, 722.65999999999997, -10544.679, 183.372894, -35.668700000000001, 0.015834350000000001, -5.2404169999999997e-06, 155015, -1, 9160.5949999999993, -129.26537300000001, 13.004, -0.0362361, 5.0063669999999998e-06, -1.2868100000000001e+30, -9);
}
PF_KWN_HD static inline double pf_kwn_G_PbTe(double T) {
    return -76063.213799999998 + 9.6771663300000004 * T + pf_kwn_GHSER_Pb(T) + pf_kwn_GHSER_Te(T);
}
PF_KWN_HD static inline double pf_kwn_G_Ag2Te(double T) {
    const double atom = -10128.93 + -12.645115000000001 * T +
        0.66666666666666663 * pf_kwn_GHSER_Ag(T) +
        0.33333333333333331 * pf_kwn_GHSER_Te(T);
    return 3 * atom;
}
PF_KWN_HD static inline double pf_kwn_L(double T) {
    return PF_KWN_DELTA_H_J_PER_MOL - PF_KWN_DELTA_S_J_PER_MOL_K * T;
}
PF_KWN_HD static inline double pf_kwn_mu_A(double T, double xB) {
    const double x = pf_kwn_clamp_fraction(xB);
    return pf_kwn_G_PbTe(T) + PF_KWN_R_GAS * T * log(1.0 - x) + pf_kwn_L(T) * x * x;
}
PF_KWN_HD static inline double pf_kwn_mu_B(double T, double xB) {
    const double x = pf_kwn_clamp_fraction(xB);
    return pf_kwn_G_Ag2Te(T) + PF_KWN_R_GAS * T * log(x) + pf_kwn_L(T) * (1.0 - x) * (1.0 - x);
}
PF_KWN_HD static inline double pf_kwn_G_alpha(double T, double xB) {
    const double x = pf_kwn_clamp_fraction(xB);
    return (1.0 - x) * pf_kwn_G_PbTe(T) + x * pf_kwn_G_Ag2Te(T) +
        PF_KWN_R_GAS * T * ((1.0 - x) * log(1.0 - x) + x * log(x)) +
        pf_kwn_L(T) * x * (1.0 - x);
}
PF_KWN_HD static inline double pf_kwn_dGdx(double T, double xB) {
    return pf_kwn_mu_B(T, xB) - pf_kwn_mu_A(T, xB);
}
PF_KWN_HD static inline double pf_kwn_d2Gdx2(double T, double xB) {
    const double x = pf_kwn_clamp_fraction(xB);
    return PF_KWN_R_GAS * T / (x * (1.0 - x)) - 2.0 * pf_kwn_L(T);
}
PF_KWN_HD static inline double pf_kwn_beta_driving_force(double T, double xB) {
    const double x = pf_kwn_clamp_fraction(xB);
    return PF_KWN_R_GAS * T * log(x) + pf_kwn_L(T) * (1.0 - x) * (1.0 - x);
}
PF_KWN_HD static inline double pf_kwn_D_alpha_m2_s(double T) {
    return PF_KWN_D0_CM2_PER_S * exp(-PF_KWN_ACTIVATION_ENERGY_J_PER_MOL / (PF_KWN_R_GAS * T)) * 1.0e-4;
}
PF_KWN_HD static inline double pf_kwn_planar_solvus(double T) {
    double lo = 1.0e-12;
    double hi = 0.5;
    for (int iter = 0; iter < 200; ++iter) {
        const double mid = 0.5 * (lo + hi);
        const double f = pf_kwn_beta_driving_force(T, mid);
        if (f > 0.0) hi = mid; else lo = mid;
    }
    return 0.5 * (lo + hi);
}
PF_KWN_HD static inline double pf_kwn_curvature_equilibrium(double T, double radius_m, double elastic_penalty_J_m3) {
    const double xeq = pf_kwn_planar_solvus(T);
    return xeq * exp(((2.0 * PF_KWN_GAMMA_J_PER_M2 / radius_m) + elastic_penalty_J_m3) *
                      PF_KWN_VM_BETA_M3_PER_MOL / (PF_KWN_R_GAS * T));
}
PF_KWN_HD static inline double pf_kwn_h_of_phi(double phi) {
    const double p = phi;
    return 6.0 * p * p * p * p * p - 15.0 * p * p * p * p + 10.0 * p * p * p;
}

#endif  // PF_KWN_VALIDATION_CONTRACT_V1_H
