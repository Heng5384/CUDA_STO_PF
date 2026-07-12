#ifndef CUDA_NUCLEUS_EVAL_CU
#define CUDA_NUCLEUS_EVAL_CU

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

struct ParametricNucleusSpec {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    double dx_nm = 0.1;
    double dy_nm = 0.1;
    double dz_nm = 0.1;
    double center[3] = {0.0, 0.0, 0.0};
    double axes[3][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    };
    double semiaxes_nm[3] = {1.0, 1.0, 1.0};
    double r_eff_nm = 1.0;
    double matrix_xB = 0.03;
    double vB_fraction = 1.0;
    double xB_min = 1.0e-9;
    double xB_max = 0.999999;
    bool anisotropic = false;
};

struct ParametricNucleusStats {
    double amplitude = 1.0;
    double target_phi_volume_nm3 = 0.0;
    double phi_integral_amp1_nm3 = 0.0;
    double phi_integral_nm3 = 0.0;
    double mean_hphi = 0.0;
    double mean_xBtot_before = 0.0;
    double mean_xBtot_after = 0.0;
    double mass_error = 0.0;
    double mass_error_abs = 0.0;
    double phi_min = 0.0;
    double phi_max = 0.0;
    double xB_min = 0.0;
    double xB_max = 0.0;
    double grad_phi_max = 0.0;
    int finite = 1;
    int connected_components = 0;
    int largest_component_voxels = 0;
    int accepted = 0;
};

static inline double h_phi_parametric(double phi) {
    const double p = std::max(0.0, std::min(1.0, phi));
    const double p2 = p * p;
    const double p3 = p2 * p;
    return p3 * (10.0 + p * (-15.0 + 6.0 * p));
}

static inline std::size_t flat_index_parametric(const ParametricNucleusSpec &s, int i, int j, int k) {
    return (static_cast<std::size_t>(i) * static_cast<std::size_t>(s.ny) + static_cast<std::size_t>(j)) *
           static_cast<std::size_t>(s.nz) + static_cast<std::size_t>(k);
}

static inline double analytic_q_parametric(const ParametricNucleusSpec &s, double x, double y, double z) {
    const double rel[3] = {x - s.center[0], y - s.center[1], z - s.center[2]};
    if (!s.anisotropic) {
        const double r2 = rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2];
        return r2 / std::max(s.r_eff_nm * s.r_eff_nm, 1.0e-30);
    }
    double q = 0.0;
    for (int a = 0; a < 3; ++a) {
        const double u = rel[0] * s.axes[a][0] + rel[1] * s.axes[a][1] + rel[2] * s.axes[a][2];
        const double sigma = std::max(s.semiaxes_nm[a], 1.0e-12);
        q += (u * u) / (sigma * sigma);
    }
    return q;
}

static inline double target_volume_parametric(const ParametricNucleusSpec &s) {
    if (s.anisotropic) {
        return 4.0 * M_PI * s.semiaxes_nm[0] * s.semiaxes_nm[1] * s.semiaxes_nm[2] / 3.0;
    }
    return 4.0 * M_PI * std::pow(s.r_eff_nm, 3.0) / 3.0;
}

static int count_components_parametric(const std::vector<float> &phi, const ParametricNucleusSpec &s, double threshold, int *largest_out) {
    const std::size_t n = phi.size();
    std::vector<unsigned char> seen(n, 0);
    int components = 0;
    int largest = 0;
    const int di[6] = {1, -1, 0, 0, 0, 0};
    const int dj[6] = {0, 0, 1, -1, 0, 0};
    const int dk[6] = {0, 0, 0, 0, 1, -1};
    std::vector<std::size_t> stack;
    for (int i = 0; i < s.nx; ++i) {
        for (int j = 0; j < s.ny; ++j) {
            for (int k = 0; k < s.nz; ++k) {
                const std::size_t start = flat_index_parametric(s, i, j, k);
                if (seen[start] || phi[start] <= threshold) continue;
                components += 1;
                int count = 0;
                stack.clear();
                stack.push_back(start);
                seen[start] = 1;
                while (!stack.empty()) {
                    const std::size_t idx = stack.back();
                    stack.pop_back();
                    count += 1;
                    const int ck = static_cast<int>(idx % static_cast<std::size_t>(s.nz));
                    const int cj = static_cast<int>((idx / static_cast<std::size_t>(s.nz)) % static_cast<std::size_t>(s.ny));
                    const int ci = static_cast<int>(idx / (static_cast<std::size_t>(s.ny) * static_cast<std::size_t>(s.nz)));
                    for (int nb = 0; nb < 6; ++nb) {
                        const int ni = ci + di[nb];
                        const int nj = cj + dj[nb];
                        const int nk = ck + dk[nb];
                        if (ni < 0 || nj < 0 || nk < 0 || ni >= s.nx || nj >= s.ny || nk >= s.nz) continue;
                        const std::size_t nidx = flat_index_parametric(s, ni, nj, nk);
                        if (seen[nidx] || phi[nidx] <= threshold) continue;
                        seen[nidx] = 1;
                        stack.push_back(nidx);
                    }
                }
                largest = std::max(largest, count);
            }
        }
    }
    if (largest_out) *largest_out = largest;
    return components;
}

static double grad_phi_max_parametric(const std::vector<float> &phi, const ParametricNucleusSpec &s) {
    double out = 0.0;
    for (int i = 1; i < s.nx - 1; ++i) {
        for (int j = 1; j < s.ny - 1; ++j) {
            for (int k = 1; k < s.nz - 1; ++k) {
                const double gx = (phi[flat_index_parametric(s, i + 1, j, k)] - phi[flat_index_parametric(s, i - 1, j, k)]) / (2.0 * s.dx_nm);
                const double gy = (phi[flat_index_parametric(s, i, j + 1, k)] - phi[flat_index_parametric(s, i, j - 1, k)]) / (2.0 * s.dy_nm);
                const double gz = (phi[flat_index_parametric(s, i, j, k + 1)] - phi[flat_index_parametric(s, i, j, k - 1)]) / (2.0 * s.dz_nm);
                out = std::max(out, std::sqrt(gx * gx + gy * gy + gz * gz));
            }
        }
    }
    return out;
}

static double analytic_grad_bound_parametric(const ParametricNucleusSpec &s, double amplitude) {
    double sigma_min = s.r_eff_nm;
    if (s.anisotropic) {
        sigma_min = std::min(s.semiaxes_nm[0], std::min(s.semiaxes_nm[1], s.semiaxes_nm[2]));
    }
    sigma_min = std::max(sigma_min, 1.0e-12);
    return amplitude * std::sqrt(2.0 / std::exp(1.0)) / sigma_min;
}

static ParametricNucleusStats build_parametric_nucleus(
    const ParametricNucleusSpec &s,
    std::vector<float> *phi_out,
    std::vector<float> *xB_out,
    double mass_tol = 1.0e-10) {
    const std::size_t n = static_cast<std::size_t>(s.nx) * static_cast<std::size_t>(s.ny) * static_cast<std::size_t>(s.nz);
    const double voxel = s.dx_nm * s.dy_nm * s.dz_nm;
    ParametricNucleusStats st;
    st.target_phi_volume_nm3 = target_volume_parametric(s);
    std::vector<double> q_values(n, 0.0);
    double integral_amp1 = 0.0;
    for (int i = 0; i < s.nx; ++i) {
        const double x = (i + 0.5) * s.dx_nm;
        for (int j = 0; j < s.ny; ++j) {
            const double y = (j + 0.5) * s.dy_nm;
            for (int k = 0; k < s.nz; ++k) {
                const double z = (k + 0.5) * s.dz_nm;
                const std::size_t idx = flat_index_parametric(s, i, j, k);
                const double q = analytic_q_parametric(s, x, y, z);
                q_values[idx] = q;
                integral_amp1 += std::exp(-q) * voxel;
            }
        }
    }
    st.phi_integral_amp1_nm3 = integral_amp1;
    st.amplitude = st.target_phi_volume_nm3 / std::max(integral_amp1, 1.0e-300);
    phi_out->assign(n, 0.0f);
    xB_out->assign(n, static_cast<float>(s.matrix_xB));
    st.phi_min = 1.0e300;
    st.phi_max = -1.0e300;
    double sum_h = 0.0;
    double sum_xbtot_before = 0.0;
    for (std::size_t idx = 0; idx < n; ++idx) {
        const double phi = st.amplitude * std::exp(-q_values[idx]);
        (*phi_out)[idx] = static_cast<float>(phi);
        st.phi_integral_nm3 += phi * voxel;
        st.phi_min = std::min(st.phi_min, phi);
        st.phi_max = std::max(st.phi_max, phi);
        if (!std::isfinite(phi)) st.finite = 0;
        const double h = h_phi_parametric(phi);
        sum_h += h;
        sum_xbtot_before += (1.0 - h) * s.matrix_xB + s.vB_fraction * h;
    }
    st.mean_hphi = sum_h / static_cast<double>(n);
    st.mean_xBtot_before = sum_xbtot_before / static_cast<double>(n);

    const double target_mean_xBtot = s.matrix_xB;
    const double target_total = target_mean_xBtot * static_cast<double>(n);
    double current_total = sum_xbtot_before;
    std::vector<double> weights(n, 0.0);
    double denom = 0.0;
    for (std::size_t idx = 0; idx < n; ++idx) {
        const double h = h_phi_parametric((*phi_out)[idx]);
        const double w = std::max(0.0, 1.0 - h);
        if ((*phi_out)[idx] < 0.35 && w > 1.0e-12) {
            weights[idx] = w;
            denom += w * w;
        }
    }
    for (int iter = 0; iter < 64; ++iter) {
        const double err = target_total - current_total;
        if (std::fabs(err / std::max(std::fabs(target_total), 1.0)) <= mass_tol) break;
        if (denom <= 1.0e-300) break;
        for (std::size_t idx = 0; idx < n; ++idx) {
            if (weights[idx] <= 0.0) continue;
            const double h = h_phi_parametric((*phi_out)[idx]);
            double xb = (*xB_out)[idx] + err * weights[idx] / (denom * std::max(1.0 - h, 1.0e-30));
            xb = std::max(s.xB_min, std::min(s.xB_max, xb));
            (*xB_out)[idx] = static_cast<float>(xb);
        }
        current_total = 0.0;
        for (std::size_t idx = 0; idx < n; ++idx) {
            const double h = h_phi_parametric((*phi_out)[idx]);
            current_total += (1.0 - h) * (*xB_out)[idx] + s.vB_fraction * h;
        }
    }
    st.mean_xBtot_after = current_total / static_cast<double>(n);
    st.mass_error = st.mean_xBtot_after - target_mean_xBtot;
    st.mass_error_abs = std::fabs(st.mass_error);
    st.xB_min = 1.0e300;
    st.xB_max = -1.0e300;
    for (std::size_t idx = 0; idx < n; ++idx) {
        st.xB_min = std::min(st.xB_min, static_cast<double>((*xB_out)[idx]));
        st.xB_max = std::max(st.xB_max, static_cast<double>((*xB_out)[idx]));
        if (!std::isfinite((*xB_out)[idx])) st.finite = 0;
    }
    int largest = 0;
    const double threshold = std::min(0.5, 0.5 * st.phi_max);
    if (n > 8000000ULL) {
        st.connected_components = 1;
        st.largest_component_voxels = -1;
        st.grad_phi_max = analytic_grad_bound_parametric(s, st.amplitude);
    } else {
        st.connected_components = count_components_parametric(*phi_out, s, threshold, &largest);
        st.largest_component_voxels = largest;
        st.grad_phi_max = grad_phi_max_parametric(*phi_out, s);
    }
    const bool phi_bounds_ok = st.phi_min >= -1.0e-12 && st.phi_max <= 1.0 + 1.0e-12;
    const bool mass_ok = st.mass_error_abs <= mass_tol;
    const bool connected_ok = st.connected_components == 1;
    const bool volume_ok = std::fabs(st.phi_integral_nm3 - st.target_phi_volume_nm3) /
                           std::max(st.target_phi_volume_nm3, 1.0e-300) < 5.0e-4;
    st.accepted = (st.finite && phi_bounds_ok && mass_ok && connected_ok && volume_ok) ? 1 : 0;
    return st;
}

#endif
