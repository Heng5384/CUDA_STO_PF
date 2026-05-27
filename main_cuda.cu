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
#include <vector>
#include <array>
#include <algorithm>
#include <limits>
#include <string>
#include <map>

#include "cuda_common.h"
#include "pf_params.h"
#include "phase_functions.h"
#define THERMO_UTILS_DEFINE_GLOBALS
#include "thermo_utils.h"
#include "cuda_kernels.h"
#include "io_vtk_cuda.h"

typedef enum {
    VTK_NAME_INIT = 0,
    VTK_NAME_STEP = 1,
    VTK_NAME_FINAL = 2
} VtkNameKind;

static int g_dynamic_outputs_use_case_dir = 0;

static int is_valid_model_mode(const char *mode);
static int is_valid_gp_init_mode(const char *mode);
static int is_valid_gp_init_mass_mode(const char *mode);
static int is_valid_gp_obs_profile_type(const char *mode);
static int is_valid_gp_obs_match_mode(const char *mode);
static int is_valid_gp_obs_compensation_mode(const char *mode);
static int is_valid_gp_eta_mass_limiter(const char *mode);
static int is_valid_gp_y_update_mode(const char *mode);
static int is_valid_gp_nuc_mass_mode(const char *mode);
static int is_valid_gp_to_beta_mass_mode(const char *mode);
static int is_valid_gp_to_beta_eta_deplete_mode(const char *mode);
static int is_valid_gp_to_beta_phi_insert_mode(const char *mode);
static int is_valid_gp_to_beta_barrier_mode(const char *mode);
static int is_valid_gp_to_beta_drive_mode(const char *mode);
static int is_valid_gp_C_mode(const char *mode);
static int is_valid_gp_eps_mode(const char *mode);
static int is_valid_gp_L_eta_mode(const char *mode);
static int is_valid_y_update_mass_projection_target_mode(const char *mode);
static int is_gp_zone_mode(const PFParams *P);
static void sync_thermo_runtime_flags(const PFParams *P);
static double gpu_reduce_sum_model_xBtot(const PFParams *P,
                                         const double *phi_r,
                                         const double *eta_r,
                                         const double *xB_alpha_r,
                                         int total_size);
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

__device__ static inline double atomicAddDoubleCompat(double *address, double val) {
    unsigned long long int *address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    while (1) {
        unsigned long long int assumed = old;
        double updated = __longlong_as_double(assumed) + val;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(updated));
        if (old == assumed) break;
    }
    return __longlong_as_double(old);
}

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
        const int use_case_dir = g_dynamic_outputs_use_case_dir && case_output_dir && case_output_dir[0] != '\0';
        const char *base_dir = use_case_dir ? case_output_dir : output_dir;
        if (!base_dir) {
            out[0] = '\0';
            return;
        }
        if (kind == VTK_NAME_INIT) {
            snprintf(out, out_size, "%s/%s_init.vtk", base_dir, field);
        } else if (kind == VTK_NAME_FINAL) {
            snprintf(out, out_size, "%s/%s_final.vtk", base_dir, field);
        } else {
            snprintf(out, out_size, "%s/%s_%d.vtk", base_dir, field, step);
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

static void build_case_summary_path(char *out, size_t out_size,
                                    const char *output_dir,
                                    const char *case_output_dir,
                                    const char *case_tag,
                                    int mode) {
    if (!out || out_size == 0) return;
    const int use_case_dir = (mode != 0) || (g_dynamic_outputs_use_case_dir && case_output_dir && case_output_dir[0] != '\0');
    const char *base_dir = use_case_dir ? case_output_dir : output_dir;
    if (!base_dir) {
        out[0] = '\0';
        return;
    }
    snprintf(out, out_size, "%s/summary.txt", base_dir);
}

static int copy_text_file(const char *src, const char *dst) {
    FILE *fin = NULL;
    FILE *fout = NULL;
    char buffer[8192];
    size_t nread = 0;

    if (!src || !dst || src[0] == '\0' || dst[0] == '\0') return 0;
    if (strcmp(src, dst) == 0) return 1;
    fin = fopen(src, "rb");
    if (!fin) return 0;
    fout = fopen(dst, "wb");
    if (!fout) {
        fclose(fin);
        return 0;
    }

    while ((nread = fread(buffer, 1, sizeof(buffer), fin)) > 0) {
        if (fwrite(buffer, 1, nread, fout) != nread) {
            fclose(fin);
            fclose(fout);
            return 0;
        }
    }

    fclose(fin);
    fclose(fout);
    return 1;
}

static int build_continue_case_pf_param_path(const char *continue_phi_vtk_path,
                                             char *out,
                                             size_t out_size) {
    char tmp[4096];
    char *slash = NULL;

    if (!continue_phi_vtk_path || continue_phi_vtk_path[0] == '\0' || !out || out_size == 0) {
        return 0;
    }
    snprintf(tmp, sizeof(tmp), "%s", continue_phi_vtk_path);
    slash = strrchr(tmp, '/');
    if (!slash) {
        return 0;
    }
    *slash = '\0';
    snprintf(out, out_size, "%s/pf_input.params", tmp);
    return 1;
}

static int derive_continue_output_root(const char *continue_phi_vtk_path,
                                       char *out,
                                       size_t out_size) {
    char tmp[4096];
    char *slash = NULL;

    if (!continue_phi_vtk_path || continue_phi_vtk_path[0] == '\0' || !out || out_size == 0) {
        return 0;
    }
    snprintf(tmp, sizeof(tmp), "%s", continue_phi_vtk_path);
    slash = strrchr(tmp, '/');
    if (!slash) return 0;
    *slash = '\0'; // case_output_dir
    slash = strrchr(tmp, '/');
    if (!slash) return 0;
    *slash = '\0'; // output_root
    snprintf(out, out_size, "%s", tmp);
    return 1;
}

static int derive_continue_case_output_dir(const char *continue_phi_vtk_path,
                                           char *out,
                                           size_t out_size) {
    char tmp[4096];
    char *slash = NULL;

    if (!continue_phi_vtk_path || continue_phi_vtk_path[0] == '\0' || !out || out_size == 0) {
        return 0;
    }
    snprintf(tmp, sizeof(tmp), "%s", continue_phi_vtk_path);
    slash = strrchr(tmp, '/');
    if (!slash) return 0;
    *slash = '\0';
    snprintf(out, out_size, "%s", tmp);
    return 1;
}

static void path_basename_copy(const char *path, char *out, size_t out_size) {
    const char *base = NULL;
    if (!out || out_size == 0) return;
    out[0] = '\0';
    if (!path || path[0] == '\0') return;
    base = strrchr(path, '/');
    base = base ? (base + 1) : path;
    snprintf(out, out_size, "%s", base);
}

static int file_exists_nonempty(const char *path) {
    struct stat st;
    if (!path || path[0] == '\0') return 0;
    if (stat(path, &st) != 0) return 0;
    return (st.st_size > 0) ? 1 : 0;
}

static int read_last_vf_csv_state(const char *path, int *last_step, double *last_t_code, double *last_t_real) {
    FILE *fp = NULL;
    char line[4096];
    int found = 0;
    int step = 0;
    double t_code = 0.0, t_real = 0.0;

    if (!path) return 0;
    fp = fopen(path, "r");
    if (!fp) return 0;
    while (fgets(line, sizeof(line), fp) != NULL) {
        if (!isdigit((unsigned char)line[0]) && line[0] != '-') continue;
        if (sscanf(line, "%d,%lf,%lf", &step, &t_code, &t_real) == 3) {
            found = 1;
            if (last_step) *last_step = step;
            if (last_t_code) *last_t_code = t_code;
            if (last_t_real) *last_t_real = t_real;
        }
    }
    fclose(fp);
    return found;
}

static int read_last_energy_iter(const char *path, int *last_iter) {
    FILE *fp = NULL;
    char line[4096];
    int found = 0;
    int iter = 0;

    if (!path) return 0;
    fp = fopen(path, "r");
    if (!fp) return 0;
    while (fgets(line, sizeof(line), fp) != NULL) {
        if (!isdigit((unsigned char)line[0]) && line[0] != '-') continue;
        if (sscanf(line, "%d,", &iter) == 1) {
            found = 1;
            if (last_iter) *last_iter = iter;
        }
    }
    fclose(fp);
    return found;
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
    int valid;
    int component_count;
    int chosen_component;
    size_t voxel_count;
    size_t boundary_voxel_count;
    double threshold;
    double center[3];
    double full_axes[3];
    double bbox_lengths[3];
    double principal_axes[3][3];  // [long/mid/short][x/y/z]
    double face_points[6][3];
    double face_normals[6][3];
    int face_valid[6];
} NucleusGeometrySummary;

typedef struct {
    int enabled;
    int write_each_step;
    int print_each_step;
    int calls;
    int valid_calls;
    double total_wall_s;
} GeometrySummaryRuntime;

static void compute_nucleus_dimensions_gpu(const double *d_phi_r,
                                           int Nx, int Ny, int Nz,
                                           double dx, double dy, double dz,
                                           int *d_bbox_mins, int *d_bbox_maxs,
                                           double *d_boundary_sum,
                                           unsigned long long *d_boundary_count,
                                           double *Len_x, double *Len_y, double *Len_z);

static inline int host_index_xyz(int x, int y, int z, int Ny, int Nz) {
    return (x * Ny + y) * Nz + z;
}

static inline void host_decode_index(int idx, int Ny, int Nz, int *x, int *y, int *z) {
    *z = idx % Nz;
    idx /= Nz;
    *y = idx % Ny;
    idx /= Ny;
    *x = idx;
}

static inline double host_clamp01(double v) {
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

static inline double host_dot3(const double a[3], const double b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static inline double host_norm3(const double v[3]) {
    return sqrt(host_dot3(v, v));
}

static inline void host_normalize3(double v[3]) {
    double n = host_norm3(v);
    if (n < 1.0e-30) return;
    v[0] /= n;
    v[1] /= n;
    v[2] /= n;
}

static inline void host_cross3(const double a[3], const double b[3], double out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

static inline double host_angle_deg_abs(const double a[3], const double b[3]) {
    double aa[3] = {a[0], a[1], a[2]};
    double bb[3] = {b[0], b[1], b[2]};
    host_normalize3(aa);
    host_normalize3(bb);
    double c = fabs(host_dot3(aa, bb));
    if (c > 1.0) c = 1.0;
    return acos(c) * (180.0 / M_PI);
}

static void canonicalize_axis_sign(double axis[3]) {
    int max_abs_idx = 0;
    double max_abs_val = fabs(axis[0]);
    for (int i = 1; i < 3; ++i) {
        double v = fabs(axis[i]);
        if (v > max_abs_val) {
            max_abs_val = v;
            max_abs_idx = i;
        }
    }
    if (axis[max_abs_idx] < 0.0) {
        axis[0] = -axis[0];
        axis[1] = -axis[1];
        axis[2] = -axis[2];
    }
}

static void jacobi_eigen_symm3(const double cov_in[3][3], double eigvals[3], double eigvecs[3][3]) {
    double a[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            a[i][j] = cov_in[i][j];
            eigvecs[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int iter = 0; iter < 32; ++iter) {
        int p = 0, q = 1;
        double max_offdiag = fabs(a[0][1]);
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                double v = fabs(a[i][j]);
                if (v > max_offdiag) {
                    max_offdiag = v;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_offdiag < 1.0e-12) break;

        double app = a[p][p];
        double aqq = a[q][q];
        double apq = a[p][q];
        double tau = (aqq - app) / (2.0 * apq);
        double t = ((tau >= 0.0) ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau * tau));
        double c = 1.0 / sqrt(1.0 + t * t);
        double s = t * c;

        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for (int k = 0; k < 3; ++k) {
            if (k == p || k == q) continue;
            double aik = a[k][p];
            double akq = a[k][q];
            a[k][p] = c * aik - s * akq;
            a[p][k] = a[k][p];
            a[k][q] = s * aik + c * akq;
            a[q][k] = a[k][q];
        }

        for (int k = 0; k < 3; ++k) {
            double vip = eigvecs[k][p];
            double viq = eigvecs[k][q];
            eigvecs[k][p] = c * vip - s * viq;
            eigvecs[k][q] = s * vip + c * viq;
        }
    }

    eigvals[0] = a[0][0];
    eigvals[1] = a[1][1];
    eigvals[2] = a[2][2];
}

static int compute_nucleus_geometry_summary_host(const double *h_phi,
                                                 int Nx, int Ny, int Nz,
                                                 double dx, double dy, double dz,
                                                 double threshold,
                                                 double bbox_len_x,
                                                 double bbox_len_y,
                                                 double bbox_len_z,
                                                 NucleusGeometrySummary *summary) {
    if (!h_phi || !summary) return 0;

    const size_t total_size = (size_t)Nx * (size_t)Ny * (size_t)Nz;
    const unsigned char FLAG_MASK = 1u << 0;
    const unsigned char FLAG_VISITED = 1u << 1;
    const unsigned char FLAG_SELECTED = 1u << 2;

    memset(summary, 0, sizeof(*summary));
    summary->threshold = threshold;
    summary->bbox_lengths[0] = bbox_len_x;
    summary->bbox_lengths[1] = bbox_len_y;
    summary->bbox_lengths[2] = bbox_len_z;

    std::vector<unsigned char> state(total_size, 0u);
    size_t masked_count = 0;
    for (size_t idx = 0; idx < total_size; ++idx) {
        if (host_clamp01(h_phi[idx]) > threshold) {
            state[idx] |= FLAG_MASK;
            masked_count++;
        }
    }
    if (masked_count == 0) return 0;

    std::vector<int> largest_component;
    std::vector<int> component;
    largest_component.reserve(4096);
    component.reserve(4096);

    int component_count = 0;
    int chosen_component = 0;
    for (size_t seed = 0; seed < total_size; ++seed) {
        if (!(state[seed] & FLAG_MASK) || (state[seed] & FLAG_VISITED)) continue;
        component_count++;
        component.clear();
        component.push_back((int)seed);
        state[seed] |= FLAG_VISITED;

        size_t head = 0;
        while (head < component.size()) {
            int idx = component[head++];
            int x = 0, y = 0, z = 0;
            host_decode_index(idx, Ny, Nz, &x, &y, &z);
            for (int dx_n = -1; dx_n <= 1; ++dx_n) {
                int xn = x + dx_n;
                if (xn < 0 || xn >= Nx) continue;
                for (int dy_n = -1; dy_n <= 1; ++dy_n) {
                    int yn = y + dy_n;
                    if (yn < 0 || yn >= Ny) continue;
                    for (int dz_n = -1; dz_n <= 1; ++dz_n) {
                        int zn = z + dz_n;
                        if (zn < 0 || zn >= Nz) continue;
                        if (dx_n == 0 && dy_n == 0 && dz_n == 0) continue;
                        int nidx = host_index_xyz(xn, yn, zn, Ny, Nz);
                        if (!(state[nidx] & FLAG_MASK) || (state[nidx] & FLAG_VISITED)) continue;
                        state[nidx] |= FLAG_VISITED;
                        component.push_back(nidx);
                    }
                }
            }
        }

        if (component.size() > largest_component.size()) {
            largest_component = component;
            chosen_component = component_count;
        }
    }

    if (largest_component.empty()) return 0;
    for (int idx : largest_component) state[(size_t)idx] |= FLAG_SELECTED;

    summary->component_count = component_count;
    summary->chosen_component = chosen_component;
    summary->voxel_count = largest_component.size();

    double center[3] = {0.0, 0.0, 0.0};
    for (int idx : largest_component) {
        int x = 0, y = 0, z = 0;
        host_decode_index(idx, Ny, Nz, &x, &y, &z);
        center[0] += (double)x * dx;
        center[1] += (double)y * dy;
        center[2] += (double)z * dz;
    }
    double inv_count = 1.0 / (double)largest_component.size();
    center[0] *= inv_count;
    center[1] *= inv_count;
    center[2] *= inv_count;
    summary->center[0] = center[0];
    summary->center[1] = center[1];
    summary->center[2] = center[2];

    double cov[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    for (int idx : largest_component) {
        int x = 0, y = 0, z = 0;
        host_decode_index(idx, Ny, Nz, &x, &y, &z);
        double r[3] = {(double)x * dx - center[0], (double)y * dy - center[1], (double)z * dz - center[2]};
        cov[0][0] += r[0] * r[0];
        cov[0][1] += r[0] * r[1];
        cov[0][2] += r[0] * r[2];
        cov[1][1] += r[1] * r[1];
        cov[1][2] += r[1] * r[2];
        cov[2][2] += r[2] * r[2];
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = i; j < 3; ++j) {
            cov[i][j] *= inv_count;
            cov[j][i] = cov[i][j];
        }
    }

    double eigvals_raw[3] = {0.0, 0.0, 0.0};
    double eigvecs_raw[3][3];
    jacobi_eigen_symm3(cov, eigvals_raw, eigvecs_raw);

    int order[3] = {0, 1, 2};
    std::sort(order, order + 3, [&](int a, int b) {
        return eigvals_raw[a] > eigvals_raw[b];
    });

    for (int axis = 0; axis < 3; ++axis) {
        int src = order[axis];
        double v[3] = {eigvecs_raw[0][src], eigvecs_raw[1][src], eigvecs_raw[2][src]};
        host_normalize3(v);
        canonicalize_axis_sign(v);
        summary->principal_axes[axis][0] = v[0];
        summary->principal_axes[axis][1] = v[1];
        summary->principal_axes[axis][2] = v[2];
        summary->full_axes[axis] = 2.0 * sqrt(fmax(5.0 * eigvals_raw[src], 0.0));
    }

    std::vector<std::array<double, 3>> boundary_points;
    std::vector<std::array<double, 3>> boundary_normals;
    boundary_points.reserve(largest_component.size() / 4 + 16);
    boundary_normals.reserve(largest_component.size() / 4 + 16);

    auto sample_phi = [&](int x, int y, int z) -> double {
        return h_phi[(size_t)host_index_xyz(x, y, z, Ny, Nz)];
    };

    for (int idx : largest_component) {
        int x = 0, y = 0, z = 0;
        host_decode_index(idx, Ny, Nz, &x, &y, &z);

        int is_boundary = 0;
        const int nb[6][3] = {
            {-1,  0,  0}, {1, 0, 0},
            { 0, -1,  0}, {0, 1, 0},
            { 0,  0, -1}, {0, 0, 1}
        };
        for (int k = 0; k < 6; ++k) {
            int xn = x + nb[k][0];
            int yn = y + nb[k][1];
            int zn = z + nb[k][2];
            if (xn < 0 || xn >= Nx || yn < 0 || yn >= Ny || zn < 0 || zn >= Nz) {
                is_boundary = 1;
                break;
            }
            int nidx = host_index_xyz(xn, yn, zn, Ny, Nz);
            if (!(state[(size_t)nidx] & FLAG_SELECTED)) {
                is_boundary = 1;
                break;
            }
        }
        if (!is_boundary) continue;

        double gx = 0.0, gy = 0.0, gz = 0.0;
        if (Nx > 1) {
            if (x == 0) gx = (sample_phi(x + 1, y, z) - sample_phi(x, y, z)) / dx;
            else if (x == Nx - 1) gx = (sample_phi(x, y, z) - sample_phi(x - 1, y, z)) / dx;
            else gx = (sample_phi(x + 1, y, z) - sample_phi(x - 1, y, z)) / (2.0 * dx);
        }
        if (Ny > 1) {
            if (y == 0) gy = (sample_phi(x, y + 1, z) - sample_phi(x, y, z)) / dy;
            else if (y == Ny - 1) gy = (sample_phi(x, y, z) - sample_phi(x, y - 1, z)) / dy;
            else gy = (sample_phi(x, y + 1, z) - sample_phi(x, y - 1, z)) / (2.0 * dy);
        }
        if (Nz > 1) {
            if (z == 0) gz = (sample_phi(x, y, z + 1) - sample_phi(x, y, z)) / dz;
            else if (z == Nz - 1) gz = (sample_phi(x, y, z) - sample_phi(x, y, z - 1)) / dz;
            else gz = (sample_phi(x, y, z + 1) - sample_phi(x, y, z - 1)) / (2.0 * dz);
        }

        double normal[3] = {-gx, -gy, -gz};
        if (host_norm3(normal) < 1.0e-14) continue;
        host_normalize3(normal);

        boundary_points.push_back({(double)x * dx, (double)y * dy, (double)z * dz});
        boundary_normals.push_back({normal[0], normal[1], normal[2]});
    }

    summary->boundary_voxel_count = boundary_points.size();
    if (boundary_points.empty()) return 0;

    const int face_axis_ids[6] = {0, 0, 1, 1, 2, 2};
    const int face_signs[6] = {+1, -1, +1, -1, +1, -1};
    const double face_thickness_frac = 0.10;
    const double face_radius_frac = 0.35;

    for (int face = 0; face < 6; ++face) {
        const int axis_id = face_axis_ids[face];
        const int face_sign = face_signs[face];
        double q_ext = (face_sign > 0) ? -std::numeric_limits<double>::infinity()
                                       :  std::numeric_limits<double>::infinity();
        std::vector<int> axial_candidates;
        axial_candidates.reserve(boundary_points.size());

        for (size_t i = 0; i < boundary_points.size(); ++i) {
            double rel[3] = {
                boundary_points[i][0] - center[0],
                boundary_points[i][1] - center[1],
                boundary_points[i][2] - center[2]
            };
            double q_axis = host_dot3(rel, summary->principal_axes[axis_id]);
            if (face_sign > 0) q_ext = fmax(q_ext, q_axis);
            else q_ext = fmin(q_ext, q_axis);
        }

        double r_ref = face_radius_frac * fmin(summary->full_axes[(axis_id + 1) % 3],
                                               summary->full_axes[(axis_id + 2) % 3]);
        std::vector<int> selected;
        selected.reserve(boundary_points.size());
        for (size_t i = 0; i < boundary_points.size(); ++i) {
            double rel[3] = {
                boundary_points[i][0] - center[0],
                boundary_points[i][1] - center[1],
                boundary_points[i][2] - center[2]
            };
            double q[3] = {
                host_dot3(rel, summary->principal_axes[0]),
                host_dot3(rel, summary->principal_axes[1]),
                host_dot3(rel, summary->principal_axes[2])
            };
            int sel_axis = (face_sign > 0)
                ? (q[axis_id] > (q_ext - face_thickness_frac * summary->full_axes[axis_id]))
                : (q[axis_id] < (q_ext + face_thickness_frac * summary->full_axes[axis_id]));
            double transverse = sqrt(q[(axis_id + 1) % 3] * q[(axis_id + 1) % 3] +
                                     q[(axis_id + 2) % 3] * q[(axis_id + 2) % 3]);
            if (sel_axis) axial_candidates.push_back((int)i);
            if (sel_axis && transverse < r_ref) selected.push_back((int)i);
        }
        if (selected.size() < 20) selected = axial_candidates;
        if (selected.size() < 5) continue;

        double mean_point[3] = {0.0, 0.0, 0.0};
        double mean_normal[3] = {0.0, 0.0, 0.0};
        for (int idx_sel : selected) {
            mean_point[0] += boundary_points[(size_t)idx_sel][0];
            mean_point[1] += boundary_points[(size_t)idx_sel][1];
            mean_point[2] += boundary_points[(size_t)idx_sel][2];
            mean_normal[0] += boundary_normals[(size_t)idx_sel][0];
            mean_normal[1] += boundary_normals[(size_t)idx_sel][1];
            mean_normal[2] += boundary_normals[(size_t)idx_sel][2];
        }
        double inv_sel = 1.0 / (double)selected.size();
        mean_point[0] *= inv_sel;
        mean_point[1] *= inv_sel;
        mean_point[2] *= inv_sel;
        mean_normal[0] *= inv_sel;
        mean_normal[1] *= inv_sel;
        mean_normal[2] *= inv_sel;
        if (host_norm3(mean_normal) < 1.0e-14) continue;
        host_normalize3(mean_normal);

        summary->face_valid[face] = 1;
        summary->face_points[face][0] = mean_point[0];
        summary->face_points[face][1] = mean_point[1];
        summary->face_points[face][2] = mean_point[2];
        summary->face_normals[face][0] = mean_normal[0];
        summary->face_normals[face][1] = mean_normal[1];
        summary->face_normals[face][2] = mean_normal[2];
    }

    summary->valid = 1;
    return 1;
}

static int write_nucleus_geometry_summary(const char *summary_path,
                                         const NucleusGeometrySummary *summary,
                                         const PFParams *P,
                                         const char *phi_vtk_path) {
    static const char *const face_names[6] = {
        "+long face", "-long face",
        "+mid face", "-mid face",
        "+short face", "-short face"
    };
    static const double ref_axes[3][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };
    static const char *const ref_axis_names[3] = {"x", "y", "z"};

    if (!summary_path || !summary || !summary->valid) return 0;

    FILE *fp = fopen(summary_path, "w");
    if (!fp) {
        fprintf(stderr, "ERROR: Failed to create summary file: %s\n", summary_path);
        return 0;
    }

    fprintf(fp, "==================== Analysis Summary ====================\n");
    fprintf(fp, "summary_file              : %s\n", summary_path);
    if (phi_vtk_path && phi_vtk_path[0] != '\0') {
        fprintf(fp, "phi_vtk_file              : %s\n", phi_vtk_path);
    }
    fprintf(fp, "mode                      : %s\n", (P && P->mode == 0) ? "dynamic" : "minimize");
    fprintf(fp, "grid_dimensions           : (%d, %d, %d)\n", P ? P->Nx : 0, P ? P->Ny : 0, P ? P->Nz : 0);
    fprintf(fp, "spacing_sim_units         : (%.6f, %.6f, %.6f)\n", P ? P->dx : 0.0, P ? P->dy : 0.0, P ? P->dz : 0.0);
    fprintf(fp, "threshold_mode            : phi > %.3f\n", summary->threshold);
    fprintf(fp, "connected_components      : %d\n", summary->component_count);
    fprintf(fp, "chosen_component          : %d\n", summary->chosen_component);
    fprintf(fp, "voxel_count               : %zu\n", summary->voxel_count);
    fprintf(fp, "boundary_voxel_count      : %zu\n", summary->boundary_voxel_count);
    fprintf(fp, "center_of_mass            : [%.6f, %.6f, %.6f]\n",
            summary->center[0], summary->center[1], summary->center[2]);
    fprintf(fp, "bbox_length_xyz           : [%.6f, %.6f, %.6f]\n",
            summary->bbox_lengths[0], summary->bbox_lengths[1], summary->bbox_lengths[2]);
    fprintf(fp, "\n--- Principal axes (unit vectors) ---\n");
    fprintf(fp, "long_axis                 : [%.6f, %.6f, %.6f]\n",
            summary->principal_axes[0][0], summary->principal_axes[0][1], summary->principal_axes[0][2]);
    fprintf(fp, "mid_axis                  : [%.6f, %.6f, %.6f]\n",
            summary->principal_axes[1][0], summary->principal_axes[1][1], summary->principal_axes[1][2]);
    fprintf(fp, "short_axis                : [%.6f, %.6f, %.6f]\n",
            summary->principal_axes[2][0], summary->principal_axes[2][1], summary->principal_axes[2][2]);

    fprintf(fp, "\n--- Principal lengths (covariance estimate) ---\n");
    fprintf(fp, "L1_long                   : %.6e\n", summary->full_axes[0]);
    fprintf(fp, "L2_mid                    : %.6e\n", summary->full_axes[1]);
    fprintf(fp, "L3_short                  : %.6e\n", summary->full_axes[2]);
    fprintf(fp, "L1/L3                     : %.6f\n",
            (summary->full_axes[2] > 0.0) ? (summary->full_axes[0] / summary->full_axes[2]) : 0.0);
    fprintf(fp, "L2/L3                     : %.6f\n",
            (summary->full_axes[2] > 0.0) ? (summary->full_axes[1] / summary->full_axes[2]) : 0.0);
    fprintf(fp, "L1/L2                     : %.6f\n",
            (summary->full_axes[1] > 0.0) ? (summary->full_axes[0] / summary->full_axes[1]) : 0.0);

    fprintf(fp, "\n--- Axis angles (deg, abs dot) ---\n");
    for (int axis = 0; axis < 3; ++axis) {
        const char *axis_name = (axis == 0) ? "long" : ((axis == 1) ? "mid" : "short");
        for (int ref = 0; ref < 3; ++ref) {
            fprintf(fp, "%s_vs_%s                 : %.3f\n",
                    axis_name, ref_axis_names[ref],
                    host_angle_deg_abs(summary->principal_axes[axis], ref_axes[ref]));
        }
    }

    fprintf(fp, "\n==================== Representative Face Normals ====================\n");
    for (int face = 0; face < 6; ++face) {
        fprintf(fp, "\n%s\n", face_names[face]);
        if (!summary->face_valid[face]) {
            fprintf(fp, "  [warning] not enough boundary points selected.\n");
            continue;
        }
        fprintf(fp, "  mean point              : [%.6f, %.6f, %.6f]\n",
                summary->face_points[face][0], summary->face_points[face][1], summary->face_points[face][2]);
        fprintf(fp, "  mean normal             : [%.6f, %.6f, %.6f]\n",
                summary->face_normals[face][0], summary->face_normals[face][1], summary->face_normals[face][2]);
        fprintf(fp, "  angle with x            : %.3f\n",
                host_angle_deg_abs(summary->face_normals[face], ref_axes[0]));
        fprintf(fp, "  angle with y            : %.3f\n",
                host_angle_deg_abs(summary->face_normals[face], ref_axes[1]));
        fprintf(fp, "  angle with z            : %.3f\n",
                host_angle_deg_abs(summary->face_normals[face], ref_axes[2]));
        fprintf(fp, "  angle with long axis    : %.3f\n",
                host_angle_deg_abs(summary->face_normals[face], summary->principal_axes[0]));
        fprintf(fp, "  angle with mid axis     : %.3f\n",
                host_angle_deg_abs(summary->face_normals[face], summary->principal_axes[1]));
        fprintf(fp, "  angle with short axis   : %.3f\n",
                host_angle_deg_abs(summary->face_normals[face], summary->principal_axes[2]));
    }

    fclose(fp);
    return 1;
}

static int env_flag_enabled(const char *name) {
    const char *value = getenv(name);
    if (!value || value[0] == '\0') return 0;
    if (strcmp(value, "0") == 0) return 0;
    if (strcmp(value, "false") == 0 || strcmp(value, "FALSE") == 0) return 0;
    if (strcmp(value, "no") == 0 || strcmp(value, "NO") == 0) return 0;
    return 1;
}

static int compute_geometry_summary_from_device(const PFParams *P,
                                                const double *d_phi_r,
                                                double *h_phi_r,
                                                size_t size_r,
                                                int *d_bbox_mins,
                                                int *d_bbox_maxs,
                                                double *d_boundary_sum,
                                                unsigned long long *d_boundary_count,
                                                const char *summary_path,
                                                const char *phi_vtk_path,
                                                GeometrySummaryRuntime *runtime) {
    if (!P || !d_phi_r || !h_phi_r) return 0;

    const double t0 = wall_time_sec_monotonic();

    CUDA_CHECK(cudaMemcpy(h_phi_r, d_phi_r, size_r, cudaMemcpyDeviceToHost));

    double Len_x = 0.0, Len_y = 0.0, Len_z = 0.0;
    compute_nucleus_dimensions_gpu(d_phi_r, P->Nx, P->Ny, P->Nz,
                                   P->dx, P->dy, P->dz,
                                   d_bbox_mins, d_bbox_maxs,
                                   d_boundary_sum, d_boundary_count,
                                   &Len_x, &Len_y, &Len_z);

    NucleusGeometrySummary geometry_summary;
    int ok = compute_nucleus_geometry_summary_host(h_phi_r, P->Nx, P->Ny, P->Nz,
                                                   P->dx, P->dy, P->dz,
                                                   0.5, Len_x, Len_y, Len_z,
                                                   &geometry_summary);
    if (ok && summary_path && summary_path[0] != '\0') {
        ok = write_nucleus_geometry_summary(summary_path, &geometry_summary, P, phi_vtk_path);
    }

    const double t1 = wall_time_sec_monotonic();
    if (runtime) {
        runtime->calls += 1;
        runtime->total_wall_s += (t1 - t0);
        if (ok) runtime->valid_calls += 1;
    }

    return ok;
}

typedef struct {
    int have_pf_params_schema_version;
    int have_dt;
    int have_dx, have_dy, have_dz;
    int have_t_real_unit;
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
    if (strcmp(key, "pf_params_schema_version") == 0) { presence->have_pf_params_schema_version = 1; return; }
    if (strcmp(key, "dt") == 0) { presence->have_dt = 1; return; }
    if (strcmp(key, "dx") == 0) { presence->have_dx = 1; return; }
    if (strcmp(key, "dy") == 0) { presence->have_dy = 1; return; }
    if (strcmp(key, "dz") == 0) { presence->have_dz = 1; return; }
    if (strcmp(key, "t_real_unit") == 0) { presence->have_t_real_unit = 1; return; }
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

    REQUIRE_ONE(have_pf_params_schema_version, "pf_params_schema_version");
    REQUIRE_ONE(have_dt, "dt");
    REQUIRE_ONE(have_dx, "dx");
    REQUIRE_ONE(have_dy, "dy");
    REQUIRE_ONE(have_dz, "dz");
    REQUIRE_ONE(have_t_real_unit, "t_real_unit");
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

    REQUIRE_POSITIVE(dt, "dt");
    REQUIRE_POSITIVE(dx, "dx");
    REQUIRE_POSITIVE(dy, "dy");
    REQUIRE_POSITIVE(dz, "dz");
    REQUIRE_POSITIVE(t_real_unit, "t_real_unit");
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
    if (!is_valid_model_mode(P->model_mode)) {
        fprintf(stderr,
                "[fatal] invalid model_mode = '%s'，必须是 two_phase 或 gp_zone。\n",
                P->model_mode);
        ok = 0;
    }
    if (is_gp_zone_mode(P)) {
        if (!is_valid_gp_init_mode(P->gp_init_mode)) {
            fprintf(stderr, "[fatal] invalid gp_init_mode = '%s'\n", P->gp_init_mode);
            ok = 0;
        }
        if (!is_valid_gp_init_mass_mode(P->gp_init_mass_mode)) {
            fprintf(stderr, "[fatal] invalid gp_init_mass_mode = '%s'\n", P->gp_init_mass_mode);
            ok = 0;
        }
        if (!is_valid_gp_obs_profile_type(P->gp_obs_profile_type)) {
            fprintf(stderr, "[fatal] invalid gp_obs_profile_type = '%s'\n", P->gp_obs_profile_type);
            ok = 0;
        }
        if (!is_valid_gp_obs_match_mode(P->gp_obs_match_mode)) {
            fprintf(stderr, "[fatal] invalid gp_obs_match_mode = '%s'\n", P->gp_obs_match_mode);
            ok = 0;
        }
        if (!is_valid_gp_obs_compensation_mode(P->gp_obs_compensation_mode)) {
            fprintf(stderr, "[fatal] invalid gp_obs_compensation_mode = '%s'\n", P->gp_obs_compensation_mode);
            ok = 0;
        }
        if (!is_valid_gp_eta_mass_limiter(P->gp_eta_mass_limiter)) {
            fprintf(stderr, "[fatal] invalid gp_eta_mass_limiter = '%s'\n", P->gp_eta_mass_limiter);
            ok = 0;
        }
        if (!is_valid_gp_y_update_mode(P->gp_y_update_mode)) {
            fprintf(stderr, "[fatal] invalid gp_y_update_mode = '%s'\n", P->gp_y_update_mode);
            ok = 0;
        }
        if (!is_valid_gp_nuc_mass_mode(P->gp_nuc_mass_mode)) {
            fprintf(stderr, "[fatal] invalid gp_nuc_mass_mode = '%s'\n", P->gp_nuc_mass_mode);
            ok = 0;
        }
        if (!is_valid_gp_to_beta_mass_mode(P->gp_to_beta_mass_mode)) {
            fprintf(stderr, "[fatal] invalid gp_to_beta_mass_mode = '%s'\n", P->gp_to_beta_mass_mode);
            ok = 0;
        }
        if (!is_valid_gp_to_beta_barrier_mode(P->gp_to_beta_barrier_mode)) {
            fprintf(stderr, "[fatal] invalid gp_to_beta_barrier_mode = '%s'\n", P->gp_to_beta_barrier_mode);
            ok = 0;
        }
        if (!is_valid_gp_to_beta_drive_mode(P->gp_to_beta_drive_mode)) {
            fprintf(stderr, "[fatal] invalid gp_to_beta_drive_mode = '%s'\n", P->gp_to_beta_drive_mode);
            ok = 0;
        }
        if (!is_valid_gp_to_beta_eta_deplete_mode(P->gp_to_beta_eta_deplete_mode)) {
            fprintf(stderr, "[fatal] invalid gp_to_beta_eta_deplete_mode = '%s'\n", P->gp_to_beta_eta_deplete_mode);
            ok = 0;
        }
        if (!is_valid_gp_to_beta_phi_insert_mode(P->gp_to_beta_phi_insert_mode)) {
            fprintf(stderr, "[fatal] invalid gp_to_beta_phi_insert_mode = '%s'\n", P->gp_to_beta_phi_insert_mode);
            ok = 0;
        }
        if (!is_valid_gp_C_mode(P->gp_C_mode)) {
            fprintf(stderr, "[fatal] invalid gp_C_mode = '%s'\n", P->gp_C_mode);
            ok = 0;
        }
        if (!is_valid_gp_eps_mode(P->gp_eps_mode)) {
            fprintf(stderr, "[fatal] invalid gp_eps_mode = '%s'\n", P->gp_eps_mode);
            ok = 0;
        }
        if (!is_valid_gp_L_eta_mode(P->gp_L_eta_mode)) {
            fprintf(stderr, "[fatal] invalid gp_L_eta_mode = '%s'\n", P->gp_L_eta_mode);
            ok = 0;
        }
        if (!is_valid_y_update_mass_projection_target_mode(P->y_update_mass_projection_target_mode)) {
            fprintf(stderr, "[fatal] invalid y_update_mass_projection_target_mode = '%s'\n",
                    P->y_update_mass_projection_target_mode);
            ok = 0;
        }
        if (P->gp_to_beta_check_interval <= 0) {
            fprintf(stderr, "[fatal] gp_to_beta_check_interval must be > 0\n");
            ok = 0;
        }
        if (P->gp_nuc_check_interval <= 0) {
            fprintf(stderr, "[fatal] gp_nuc_check_interval must be > 0\n");
            ok = 0;
        }
        if (P->gp_nuc_patch_radius <= 0.0) {
            fprintf(stderr, "[fatal] gp_nuc_patch_radius must be > 0\n");
            ok = 0;
        }
        if (P->gp_nuc_shell_inner_radius < 0.0 ||
            P->gp_nuc_shell_outer_radius <= P->gp_nuc_shell_inner_radius) {
            fprintf(stderr, "[fatal] invalid gp_nuc shell radii: inner=%.6e outer=%.6e\n",
                    P->gp_nuc_shell_inner_radius, P->gp_nuc_shell_outer_radius);
            ok = 0;
        }
        if (P->gp_nuc_seed_radius <= 0.0 || P->gp_nuc_seed_peak <= 0.0) {
            fprintf(stderr, "[fatal] gp_nuc seed radius/peak must be > 0\n");
            ok = 0;
        }
        if (P->gp_to_beta_patch_radius <= 0.0) {
            fprintf(stderr, "[fatal] gp_to_beta_patch_radius must be > 0\n");
            ok = 0;
        }
        if (P->gp_to_beta_shell_inner_radius < 0.0 ||
            P->gp_to_beta_shell_outer_radius <= P->gp_to_beta_shell_inner_radius) {
            fprintf(stderr, "[fatal] invalid gp_to_beta shell radii: inner=%.6e outer=%.6e\n",
                    P->gp_to_beta_shell_inner_radius, P->gp_to_beta_shell_outer_radius);
            ok = 0;
        }
        if (P->gp_to_beta_seed_radius <= 0.0 || P->gp_to_beta_seed_peak <= 0.0) {
            fprintf(stderr, "[fatal] gp_to_beta seed radius/peak must be > 0\n");
            ok = 0;
        }
        if (P->gp_to_beta_max_events_per_check <= 0) {
            fprintf(stderr, "[fatal] gp_to_beta_max_events_per_check must be > 0\n");
            ok = 0;
        }
        if (!(P->gp_to_beta_min_shell_capacity_factor > 0.0)) {
            fprintf(stderr, "[fatal] gp_to_beta_min_shell_capacity_factor must be > 0\n");
            ok = 0;
        }
        if (!(P->gp_to_beta_min_seed_amplitude >= 0.0 &&
              P->gp_to_beta_min_seed_amplitude <= 1.0)) {
            fprintf(stderr, "[fatal] gp_to_beta_min_seed_amplitude must be in [0,1]\n");
            ok = 0;
        }
        if (P->gp_to_beta_event_cooldown_steps < 0) {
            fprintf(stderr, "[fatal] gp_to_beta_event_cooldown_steps must be >= 0\n");
            ok = 0;
        }
        if (P->gp_to_beta_min_event_spacing < 0.0) {
            fprintf(stderr, "[fatal] gp_to_beta_min_event_spacing must be >= 0\n");
            ok = 0;
        }
        if (P->gp_to_beta_event_exclusion_radius < 0.0) {
            fprintf(stderr, "[fatal] gp_to_beta_event_exclusion_radius must be >= 0\n");
            ok = 0;
        }
        if (P->gp_to_beta_max_events_global <= 0) {
            fprintf(stderr, "[fatal] gp_to_beta_max_events_global must be > 0\n");
            ok = 0;
        }
        if (P->gp_to_beta_max_events_per_window <= 0) {
            fprintf(stderr, "[fatal] gp_to_beta_max_events_per_window must be > 0\n");
            ok = 0;
        }
        if (P->gp_to_beta_event_window_steps < 0) {
            fprintf(stderr, "[fatal] gp_to_beta_event_window_steps must be >= 0\n");
            ok = 0;
        }
        if (P->y_update_k0_audit_steps <= 0) {
            fprintf(stderr, "[fatal] y_update_k0_audit_steps must be > 0\n");
            ok = 0;
        }
        if (P->y_update_mass_projection_report_enabled != 0 &&
            P->y_update_mass_projection_report_enabled != 1) {
            fprintf(stderr, "[fatal] y_update_mass_projection_report_enabled must be 0 or 1\n");
            ok = 0;
        }
        if (P->y_update_mass_projection_max_iter <= 0) {
            fprintf(stderr, "[fatal] y_update_mass_projection_max_iter must be > 0\n");
            ok = 0;
        }
        if (!(P->y_update_mass_projection_tol > 0.0)) {
            fprintf(stderr, "[fatal] y_update_mass_projection_tol must be > 0\n");
            ok = 0;
        }
        if (strcmp(P->gp_init_mode, "observed_gp_diffuse") == 0) {
            if (!(P->gp_obs_target_radius_nm > 0.0)) {
                fprintf(stderr, "[fatal] gp_obs_target_radius_nm must be > 0\n");
                ok = 0;
            }
            if (!(P->gp_obs_eta_peak > 0.0 && P->gp_obs_eta_peak <= 1.0)) {
                fprintf(stderr, "[fatal] gp_obs_eta_peak must be in (0,1]\n");
                ok = 0;
            }
            if (!(P->gp_obs_iface_width_nm > 0.0)) {
                fprintf(stderr, "[fatal] gp_obs_iface_width_nm must be > 0\n");
                ok = 0;
            }
            if (!(P->gp_obs_depletion_radius_factor > 0.0)) {
                fprintf(stderr, "[fatal] gp_obs_depletion_radius_factor must be > 0\n");
                ok = 0;
            }
            if (!(P->gp_obs_depletion_smooth_width_factor > 0.0)) {
                fprintf(stderr, "[fatal] gp_obs_depletion_smooth_width_factor must be > 0\n");
                ok = 0;
            }
            if (!(P->gp_obs_min_xB_alpha >= 0.0 && P->gp_obs_min_xB_alpha < 1.0)) {
                fprintf(stderr, "[fatal] gp_obs_min_xB_alpha must be in [0,1)\n");
                ok = 0;
            }
            if (!(P->gp_obs_max_xB_alpha > 0.0 && P->gp_obs_max_xB_alpha <= 1.0)) {
                fprintf(stderr, "[fatal] gp_obs_max_xB_alpha must be in (0,1]\n");
                ok = 0;
            }
            if (!(P->gp_obs_min_xB_alpha < P->gp_obs_max_xB_alpha)) {
                fprintf(stderr, "[fatal] gp_obs_min_xB_alpha must be < gp_obs_max_xB_alpha\n");
                ok = 0;
            }
        }
    }

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
    LEDGER("d_eta_r", total_r, sizeof(double));
    LEDGER("d_Y_r", total_r, sizeof(double));
    LEDGER("d_xB_r", total_r, sizeof(double));
    LEDGER("d_phi_rhs_r(=d_lapY_r)", total_r, sizeof(double));
    LEDGER("d_eta_prev_r", total_r, sizeof(double));
    LEDGER("d_eta_rhs_r", total_r, sizeof(double));
    LEDGER("d_phi_n_saved", total_r, sizeof(double));
    LEDGER("d_Y_n_saved", total_r, sizeof(double));
    LEDGER("d_dY_dt_prev_r", total_r, sizeof(double));
    LEDGER("d_mu_x_r(=d_Y_rhs_r)", total_r, sizeof(double));
    LEDGER("d_divJ_r(=d_xB_prev_r)", total_r, sizeof(double));
    LEDGER("d_phi_k", total_k, sizeof(cufftDoubleComplex));
    LEDGER("d_eta_k", total_k, sizeof(cufftDoubleComplex));
    LEDGER("d_eta_rhs_k", total_k, sizeof(cufftDoubleComplex));
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
static void derive_next_continue_case_tag(const char *vtk_path, char *out, size_t out_size, int mode);
static int rebuild_full_model_composition_from_phi(double *phi_r, double *Y_r, double *xB_r, double *xBtot_r,
                                                   const PFParams *P, int total_size, int emit_logs);
static int load_continue_fields_from_vtk(double *phi_r, double *Y_r, double *xB_r, double *xBtot_r,
                                         const PFParams *P, int total_size);
static int is_valid_model_mode(const char *mode);
static int is_gp_zone_mode(const PFParams *P);
static double gpu_reduce_sum_model_xBtot(const PFParams *P,
                                         const double *phi_r,
                                         const double *eta_r,
                                         const double *xB_alpha_r,
                                         int total_size);

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

static double compute_effective_vf_target(const PFParams *P) {
    const int N_seeds = (P->ic_phi_num_seeds > 0) ? P->ic_phi_num_seeds : 1;
    const int is3D = (P->Ny > 2);
    const double Lx = P->Nx * P->dx;
    const double Ly = P->Ny * P->dy;
    const double Lz = P->Nz * P->dz;

    double vf_target = fmin(0.999, fmax(1e-6, P->ic_vf_target_phi));
    if (!(P->ic_phi_seed_radius > 0.0)) {
        return vf_target;
    }

    double Rx = 0.0, Ry = 0.0, Rz = 0.0;
    compute_seed_axes_from_radius(P, P->ic_phi_seed_radius, is3D, &Rx, &Ry, &Rz);

    double target_fraction = vf_target;
    if (is3D) {
        const double seed_volume = (4.0 * M_PI / 3.0) * Rx * Ry * Rz;
        const double box_volume = Lx * Ly * Lz;
        if (box_volume > 0.0) {
            target_fraction = ((double)N_seeds * seed_volume) / box_volume;
        }
    } else {
        const double seed_area = M_PI * Rx * Rz;
        const double box_area = Lx * Lz;
        if (box_area > 0.0) {
            target_fraction = ((double)N_seeds * seed_area) / box_area;
        }
    }

    return fmin(0.999, fmax(1e-6, target_fraction));
}

static void derive_next_continue_case_tag(const char *vtk_path, char *out, size_t out_size, int mode) {
    int max_continue_idx = 0;
    const char *p = vtk_path;
    const char *prefix = (mode == 0) ? "continue_dyn_" : "continue_min_";

    if (!out || out_size == 0) return;
    out[0] = '\0';

    if (!vtk_path || vtk_path[0] == '\0') {
        snprintf(out, out_size, "%s1", prefix);
        return;
    }

    while ((p = strstr(p, prefix)) != NULL) {
        const char *digits = p + (int)strlen(prefix);
        if (*digits >= '0' && *digits <= '9') {
            int value = 0;
            while (*digits >= '0' && *digits <= '9') {
                value = value * 10 + (*digits - '0');
                ++digits;
            }
            if (value > max_continue_idx) {
                max_continue_idx = value;
            }
        }
        ++p;
    }

    snprintf(out, out_size, "%s%d", prefix, max_continue_idx + 1);
}

static int rebuild_full_model_composition_from_phi(double *phi_r, double *Y_r, double *xB_r, double *xBtot_r,
                                                   const PFParams *P, int total_size, int emit_logs) {
    const double Ntot = (double)P->Nx * (double)P->Ny * (double)P->Nz;
    const double xB_eq = P->ic_xB_eq_matrix;
    const double vf_target = compute_effective_vf_target(P);
    const double vB_frac = P->v_B;
    double sum_h = 0.0;
    double sum_h2 = 0.0;

    for (int i = 0; i < total_size; ++i) {
        double phi = clamp01(phi_r[i]);
        double h = h_of_phi(phi);
        phi_r[i] = phi;
        sum_h += h;
        sum_h2 += h * h;
    }

    double mean_h = sum_h / Ntot;
    double mean_h2 = sum_h2 / Ntot;
    double xBtot_target = vB_frac * vf_target + (1.0 - vf_target) * xB_eq;
    double A = 1.0 - 2.0 * mean_h + mean_h2;
    double B = mean_h - mean_h2;
    double C = vB_frac * mean_h;
    double xB_out;

    if (P->ic_23d_xB_out > 0.0) {
        xB_out = clamp_eps(P->ic_23d_xB_out, P->xB_eps);
    } else {
        if (fabs(A) < 1e-12) {
            double denom = 1.0 - mean_h;
            if (denom < 1e-12) denom = 1e-12;
            xB_out = ((vf_target - mean_h) * vB_frac +
                      (1.0 - vf_target) * xB_eq) / denom;
        } else {
            xB_out = (xBtot_target - B * xB_eq - C) / A;
        }
        xB_out = clamp_eps(xB_out, P->xB_eps);
    }

    double sum_xB = 0.0;
    double sum_xBtot = 0.0;
    for (int i = 0; i < total_size; ++i) {
        double h = h_of_phi(phi_r[i]);
        double xB = (1.0 - h) * xB_out + h * xB_eq;
        xB = clamp_eps(xB, P->xB_eps);
        xB_r[i] = xB;
        Y_r[i] = logit_from_fraction(xB, P->xB_eps, P->Y_clip);
        xBtot_r[i] = (1.0 - h) * xB + vB_frac * h;
        sum_xB += xB;
        sum_xBtot += xBtot_r[i];
    }

    if (emit_logs) {
        double theory_xB_tot = (P->ic_23d_xB_out > 0.0)
            ? (vB_frac * mean_h + (1.0 - mean_h) * xB_out)
            : xBtot_target;
        log_section_header("Continuation Composition");
        log_kv_text("xB_source", "%s",
                    (P->ic_23d_xB_out > 0.0) ? "rebuild from phi + ic_23d_xB_out"
                                             : ((P->ic_phi_seed_radius > 0.0)
                                                ? "rebuild from phi + seed_radius_target"
                                                : "rebuild from phi + vf_target"));
        log_kv_text("vf_init_eff=<h>", "%.6f", mean_h);
        log_kv_text("<h^2>", "%.6f", mean_h2);
        log_kv_text("vf_target", "%.6f", vf_target);
        if (P->ic_phi_seed_radius > 0.0) {
            log_kv_text("vf_target_source", "seed radius %.6f", P->ic_phi_seed_radius);
        }
        log_kv_text("xB_eq", "%.6f", xB_eq);
        log_kv_text("xB_out", "%.6e (ic_23d_xB_out=%.6e)", xB_out, P->ic_23d_xB_out);
        log_kv_text("<xB>", "%.6f", sum_xB / Ntot);
        log_kv_text("<xB_tot>", "%.6f", sum_xBtot / Ntot);
        log_kv_text("theory_target", "%.8f", theory_xB_tot);
        log_kv_text("mass_balance_diff", "%.2e", (sum_xBtot / Ntot) - theory_xB_tot);
    }

    return 1;
}

static int load_continue_fields_from_vtk(double *phi_r, double *Y_r, double *xB_r, double *xBtot_r,
                                         const PFParams *P, int total_size) {
    (void)total_size;
    if (!read_vtk_ascii_to_host(P->continue_phi_vtk_path, P->Nx, P->Ny, P->Nz, phi_r, "phi")) {
        return 0;
    }

    const int need_composition_fields = (P->mode == 0) || P->minimize_full_model;

    if (need_composition_fields) {
        int xB_loaded_from_vtk = 0;
        if (P->continue_xB_vtk_path[0] != '\0') {
            struct stat st;
            if (stat(P->continue_xB_vtk_path, &st) == 0) {
                if (!read_vtk_ascii_to_host(P->continue_xB_vtk_path, P->Nx, P->Ny, P->Nz, xB_r, "xB")) {
                    return 0;
                }
                xB_loaded_from_vtk = 1;
            } else {
                fprintf(stderr,
                        "[warn] continuation xB VTK 不存在或不可读，将按 init 逻辑从 phi 重建 xB/Y: %s\n",
                        P->continue_xB_vtk_path);
            }
        }

        if (xB_loaded_from_vtk) {
            for (int i = 0; i < total_size; ++i) {
                double phi = clamp01(phi_r[i]);
                double h = h_of_phi(phi);
                double xB = clamp_eps(xB_r[i], P->xB_eps);
                phi_r[i] = phi;
                xB_r[i] = xB;
                Y_r[i] = logit_from_fraction(xB, P->xB_eps, P->Y_clip);
                xBtot_r[i] = (1.0 - h) * xB + P->v_B * h;
            }
        } else {
            rebuild_full_model_composition_from_phi(phi_r, Y_r, xB_r, xBtot_r, P, total_size, 1);
        }
    } else {
        for (int i = 0; i < total_size; ++i) {
            phi_r[i] = clamp01(phi_r[i]);
            Y_r[i] = 0.0;
            xB_r[i] = 0.0;
            xBtot_r[i] = 0.0;
        }
    }

    return 1;
}

typedef struct {
    int valid;
    int Nx, Ny, Nz;
    double dx_nm;
    double interface_width_nm;
    double dt_recommended;
    double mean_xBtot;
    double xB_max_safe;
    char dtype[32];
    char order[16];
} RawInitMeta;

static char *read_text_file_alloc(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long n = ftell(fp);
    if (n < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t)n, fp);
    fclose(fp);
    buf[got] = '\0';
    return buf;
}

static int json_get_number_simple(const char *json, const char *key, double *out) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return 0;
    p = strchr(p, ':');
    if (!p) return 0;
    ++p;
    while (*p && isspace((unsigned char)*p)) ++p;
    char *endp = NULL;
    double v = strtod(p, &endp);
    if (endp == p) return 0;
    *out = v;
    return 1;
}

static int json_get_int_simple(const char *json, const char *key, int *out) {
    double v = 0.0;
    if (!json_get_number_simple(json, key, &v)) return 0;
    *out = (int)llround(v);
    return 1;
}

static int json_get_string_simple(const char *json, const char *key, char *out, size_t out_size) {
    if (!out || out_size == 0) return 0;
    out[0] = '\0';
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return 0;
    p = strchr(p, ':');
    if (!p) return 0;
    ++p;
    while (*p && isspace((unsigned char)*p)) ++p;
    if (*p != '"') return 0;
    ++p;
    const char *q = strchr(p, '"');
    if (!q) return 0;
    size_t n = (size_t)(q - p);
    if (n >= out_size) n = out_size - 1;
    memcpy(out, p, n);
    out[n] = '\0';
    return 1;
}

static int load_raw_init_meta(const char *path, RawInitMeta *meta) {
    if (!path || path[0] == '\0' || !meta) return 0;
    memset(meta, 0, sizeof(*meta));
    meta->dt_recommended = NAN;
    meta->mean_xBtot = NAN;
    meta->xB_max_safe = NAN;
    snprintf(meta->dtype, sizeof(meta->dtype), "float32");
    snprintf(meta->order, sizeof(meta->order), "C");
    char *json = read_text_file_alloc(path);
    if (!json) {
        fprintf(stderr, "[fatal] cannot read raw init meta: %s\n", path);
        return 0;
    }
    int ok = 1;
    ok &= json_get_int_simple(json, "Nx", &meta->Nx);
    ok &= json_get_int_simple(json, "Ny", &meta->Ny);
    ok &= json_get_int_simple(json, "Nz", &meta->Nz);
    ok &= json_get_number_simple(json, "dx_nm", &meta->dx_nm);
    ok &= json_get_number_simple(json, "interface_width_nm", &meta->interface_width_nm);
    json_get_number_simple(json, "dt_recommended", &meta->dt_recommended);
    json_get_number_simple(json, "mean_xBtot", &meta->mean_xBtot);
    json_get_number_simple(json, "xB_max_safe", &meta->xB_max_safe);
    json_get_string_simple(json, "dtype", meta->dtype, sizeof(meta->dtype));
    json_get_string_simple(json, "order", meta->order, sizeof(meta->order));
    free(json);
    meta->valid = ok ? 1 : 0;
    if (!ok) {
        fprintf(stderr, "[fatal] raw init meta missing one of Nx/Ny/Nz/dx_nm/interface_width_nm: %s\n", path);
    }
    return ok;
}

static int read_raw_field_to_double(const char *path, const char *dtype, size_t total, double *out, const char *label) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[fatal] cannot open raw %s field: %s\n", label, path);
        return 0;
    }
    if (strcmp(dtype, "float32") == 0) {
        float *tmp = (float *)malloc(total * sizeof(float));
        if (!tmp) {
            fclose(fp);
            fprintf(stderr, "[fatal] host allocation failed while reading raw %s\n", label);
            return 0;
        }
        size_t got = fread(tmp, sizeof(float), total, fp);
        fclose(fp);
        if (got != total) {
            fprintf(stderr, "[fatal] raw %s size mismatch: got %zu float32 values, expected %zu\n", label, got, total);
            free(tmp);
            return 0;
        }
        for (size_t i = 0; i < total; ++i) out[i] = (double)tmp[i];
        free(tmp);
        return 1;
    } else if (strcmp(dtype, "float64") == 0) {
        size_t got = fread(out, sizeof(double), total, fp);
        fclose(fp);
        if (got != total) {
            fprintf(stderr, "[fatal] raw %s size mismatch: got %zu float64 values, expected %zu\n", label, got, total);
            return 0;
        }
        return 1;
    }
    fclose(fp);
    fprintf(stderr, "[fatal] unsupported raw dtype '%s' for %s; use float32 or float64\n", dtype, label);
    return 0;
}

static int validate_raw_init_meta_against_run(const RawInitMeta *meta, const PFParams *P,
                                              double dx_phys_nm, double interface_width_nm) {
    if (!meta || !meta->valid) return 0;
    int ok = 1;
    if (meta->Nx != P->Nx || meta->Ny != P->Ny || meta->Nz != P->Nz) {
        fprintf(stderr,
                "[fatal] raw init grid mismatch: meta=%dx%dx%d run=%dx%dx%d\n",
                meta->Nx, meta->Ny, meta->Nz, P->Nx, P->Ny, P->Nz);
        ok = 0;
    }
    if (strcmp(meta->order, "C") != 0) {
        fprintf(stderr, "[fatal] raw init order mismatch: meta order='%s', only C is supported\n", meta->order);
        ok = 0;
    }
    if (fabs(meta->dx_nm - dx_phys_nm) > 1.0e-6) {
        fprintf(stderr,
                "[fatal] raw init dx mismatch: meta dx_nm=%.9g run dx_nm=%.9g\n",
                meta->dx_nm, dx_phys_nm);
        ok = 0;
    }
    if (fabs(meta->interface_width_nm - interface_width_nm) > 1.0e-6) {
        fprintf(stderr,
                "[fatal] raw init interface width mismatch: meta=%.9g nm run lambda_sm=%.9g nm\n",
                meta->interface_width_nm, interface_width_nm);
        ok = 0;
    }
    if (isfinite(meta->dt_recommended) && fabs(meta->dt_recommended - P->dt) > 1.0e-12) {
        fprintf(stderr,
                "[warn] raw init dt differs from meta recommendation: meta dt=%.9g run dt=%.9g\n",
                meta->dt_recommended, P->dt);
    }
    return ok;
}

static int load_raw_init_fields(double *phi_r, double *eta_r, double *Y_r, double *xB_r, double *xBtot_r,
                                const PFParams *P, int total_size,
                                const RawInitMeta *meta, int emit_logs) {
    if (!meta || !meta->valid) return 0;
    if (!read_raw_field_to_double(P->init_phi_raw_path, meta->dtype, (size_t)total_size, phi_r, "phi")) {
        return 0;
    }
    if (!read_raw_field_to_double(P->init_xB_raw_path, meta->dtype, (size_t)total_size, xB_r, "xB")) {
        return 0;
    }
    if (P->init_eta_raw_path[0] != '\0') {
        if (!read_raw_field_to_double(P->init_eta_raw_path, meta->dtype, (size_t)total_size, eta_r, "eta")) {
            return 0;
        }
    } else {
        memset(eta_r, 0, (size_t)total_size * sizeof(double));
    }

    double phi_min = 1.0e300, phi_max = -1.0e300, phi_sum = 0.0;
    double eta_min = 1.0e300, eta_max = -1.0e300, eta_sum = 0.0;
    double xb_min = 1.0e300, xb_max = -1.0e300, xb_sum = 0.0;
    double h_sum = 0.0, xbtot_sum = 0.0;
    int phi_low_clamp = 0, phi_high_clamp = 0, eta_low_clamp = 0, eta_high_clamp = 0;
    int xb_low_clamp = 0, xb_high_clamp = 0;
    const double xb_cap = isfinite(meta->xB_max_safe) ? meta->xB_max_safe : P->minimize_xB_max_safe;
    for (int i = 0; i < total_size; ++i) {
        double phi0 = phi_r[i];
        double eta0 = eta_r[i];
        double xb0 = xB_r[i];
        if (!isfinite(phi0) || !isfinite(eta0) || !isfinite(xb0)) {
            fprintf(stderr, "[fatal] raw init contains NaN/Inf at idx=%d (phi=%g, eta=%g, xB=%g)\n",
                    i, phi0, eta0, xb0);
            return 0;
        }
        if (phi0 < 0.0) ++phi_low_clamp;
        if (phi0 > 1.0) ++phi_high_clamp;
        if (eta0 < 0.0) ++eta_low_clamp;
        if (eta0 > 1.0) ++eta_high_clamp;
        if (xb0 < P->xB_eps) ++xb_low_clamp;
        if (isfinite(xb_cap) && xb0 > xb_cap) ++xb_high_clamp;
        double phi = clamp01(phi0);
        double eta = clamp01(eta0);
        double xb = clamp_eps(xb0, P->xB_eps);
        if (isfinite(xb_cap) && xb > xb_cap) xb = xb_cap;
        phi_r[i] = phi;
        eta_r[i] = eta;
        xB_r[i] = xb;
        Y_r[i] = logit_from_fraction(xb, P->xB_eps, P->Y_clip);
        if (is_gp_zone_mode(P)) {
            double h_alpha = 0.0, h_GP = 0.0, h_beta = 0.0;
            phase_fractions_gp(phi, eta, &h_alpha, &h_GP, &h_beta);
            xBtot_r[i] = h_alpha * xb + h_GP * P->gp_xB_fixed + h_beta;
            h_sum += h_GP;
        } else {
            double h = h_of_phi(phi);
            xBtot_r[i] = (1.0 - h) * xb + P->v_B * h;
            h_sum += h;
        }
        phi_min = fmin(phi_min, phi);
        phi_max = fmax(phi_max, phi);
        eta_min = fmin(eta_min, eta);
        eta_max = fmax(eta_max, eta);
        xb_min = fmin(xb_min, xb);
        xb_max = fmax(xb_max, xb);
        phi_sum += phi;
        eta_sum += eta;
        xb_sum += xb;
        xbtot_sum += xBtot_r[i];
    }
    if (emit_logs) {
        const double invN = 1.0 / (double)total_size;
        log_section_header("INIT raw_fields");
        log_kv_text("init_phi_raw", "%s", P->init_phi_raw_path);
        log_kv_text("init_xB_raw", "%s", P->init_xB_raw_path);
        if (P->init_eta_raw_path[0] != '\0') {
            log_kv_text("init_eta_raw", "%s", P->init_eta_raw_path);
        }
        log_kv_text("init_meta", "%s", P->init_meta_path);
        log_kv_text("dtype/order", "%s / %s", meta->dtype, meta->order);
        log_kv_text("phi min/max/mean", "%.8e / %.8e / %.8e", phi_min, phi_max, phi_sum * invN);
        log_kv_text("eta min/max/mean", "%.8e / %.8e / %.8e", eta_min, eta_max, eta_sum * invN);
        log_kv_text("xB min/max/mean", "%.8e / %.8e / %.8e", xb_min, xb_max, xb_sum * invN);
        log_kv_text(is_gp_zone_mode(P) ? "hGP mean" : "hphi mean", "%.8e", h_sum * invN);
        log_kv_text("xBtot mean", "%.8e", xbtot_sum * invN);
        if (isfinite(meta->mean_xBtot)) {
            log_kv_text("python_meta mean_xBtot", "%.8e (diff=%.3e)", meta->mean_xBtot, xbtot_sum * invN - meta->mean_xBtot);
        }
        log_kv_text("phi clamp count", "low=%d high=%d fraction=%.3e",
                    phi_low_clamp, phi_high_clamp, (phi_low_clamp + phi_high_clamp) * invN);
        log_kv_text("eta clamp count", "low=%d high=%d fraction=%.3e",
                    eta_low_clamp, eta_high_clamp, (eta_low_clamp + eta_high_clamp) * invN);
        log_kv_text("xB clamp count", "low=%d high=%d fraction=%.3e",
                    xb_low_clamp, xb_high_clamp, (xb_low_clamp + xb_high_clamp) * invN);
        printf("[INIT] phi min/max/mean = %.8e / %.8e / %.8e, eta min/max/mean = %.8e / %.8e / %.8e, "
               "xB min/max/mean = %.8e / %.8e / %.8e, %s mean = %.8e, xBtot mean = %.8e, "
               "clamp counts phi_low/high=%d/%d eta_low/high=%d/%d xB_low/high=%d/%d\n",
               phi_min, phi_max, phi_sum * invN,
               eta_min, eta_max, eta_sum * invN,
               xb_min, xb_max, xb_sum * invN,
               is_gp_zone_mode(P) ? "hGP" : "hphi", h_sum * invN, xbtot_sum * invN,
               phi_low_clamp, phi_high_clamp, eta_low_clamp, eta_high_clamp, xb_low_clamp, xb_high_clamp);
        if (phi_low_clamp || phi_high_clamp || eta_low_clamp || eta_high_clamp || xb_low_clamp || xb_high_clamp) {
            fprintf(stderr,
                    "[warn] raw init required clamping: phi_low=%d phi_high=%d eta_low=%d eta_high=%d xB_low=%d xB_high=%d\n",
                    phi_low_clamp, phi_high_clamp, eta_low_clamp, eta_high_clamp, xb_low_clamp, xb_high_clamp);
        }
    }
    return 1;
}

static void write_relaxation_diag_row(FILE *fp, int step, double t_code, double t_real,
                                      const double *d_phi_r, const double *d_eta_r, const double *d_xB_r,
                                      const PFParams *P, int total_size,
                                      double initial_mean_xBtot) {
    if (!fp) return;
    double mean_phi = gpu_reduce_sum(d_phi_r, total_size) / (double)total_size;
    double mean_hphi = gpu_compute_vf_from_h(d_phi_r, total_size);
    double mean_xB = gpu_reduce_sum(d_xB_r, total_size) / (double)total_size;
    double mean_xBtot = gpu_reduce_sum_model_xBtot(P, d_phi_r, d_eta_r, d_xB_r, total_size) / (double)total_size;
    double xB_min = 0.0, xB_max = 0.0, phi_min = 0.0, phi_max = 0.0;
    gpu_reduce_min_max(d_xB_r, total_size, &xB_min, &xB_max);
    gpu_reduce_min_max(d_phi_r, total_size, &phi_min, &phi_max);
    double delta = mean_xBtot - initial_mean_xBtot;
    double rel = delta / fmax(fabs(initial_mean_xBtot), 1.0e-30);
    fprintf(fp,
            "%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e\n",
            step, t_code, t_real,
            mean_phi, mean_hphi, mean_xB, mean_xBtot,
            initial_mean_xBtot, delta, rel,
            xB_min, xB_max, phi_min, phi_max);
    fflush(fp);
    double abs_rel = fabs(rel);
    if (abs_rel > 1.0e-2) {
        fprintf(stderr, "[warn] xBtot relative drift exceeded 1e-2 at step %d: %.6e\n", step, rel);
    } else if (abs_rel > 1.0e-3) {
        fprintf(stderr, "[warn] xBtot relative drift exceeded 1e-3 at step %d: %.6e\n", step, rel);
    } else if (abs_rel > 1.0e-4) {
        fprintf(stderr, "[warn] xBtot relative drift exceeded 1e-4 at step %d: %.6e\n", step, rel);
    }
}

typedef struct {
    double initial_mean_xBtot;
    double final_mean_xBtot;
    double total_absolute_drift;
    double total_relative_drift;
    double cumulative_delta_phi_update;
    double cumulative_delta_eta_update;
    double cumulative_delta_gp_to_beta_event;
    double cumulative_delta_Y_update;
    double cumulative_delta_Y_to_xB;
    double cumulative_delta_clipping;
    double cumulative_delta_other;
    double max_abs_step_drift;
    double max_abs_delta_phi_update;
    double max_abs_delta_eta_update;
    double max_abs_delta_gp_to_beta_event;
    double max_abs_delta_Y_update;
    double max_abs_delta_Y_to_xB;
    double max_abs_delta_clipping;
    double total_clip_count_phi;
    double total_clip_count_xB;
    double total_clip_count_Y;
    double max_abs_mass_closure_resid;
    double summary_matches_csv;
    double max_summary_csv_abs_mismatch;
    const char *suspected_primary_source;
} DynamicsMassDiagSummary;

typedef struct {
    char case_name[128];
    double picard_enabled;
    double picard_iters;
    double picard_omega;
    double nsteps;
    double dt;
    double total_walltime_s;
    double avg_walltime_per_step_s;
    double median_walltime_per_step_s;
    double min_walltime_per_step_s;
    double max_walltime_per_step_s;
    double warmup_excluded_avg_walltime_per_step_s;
    double estimated_steps_per_hour;
    double relative_slowdown_vs_baseline;
} PerformanceSummary;

typedef struct {
    int step;
    double time_code;
    double mean_xBtot_before_step;
    double mean_xBtot_after_phi_update;
    double mean_xBtot_after_eta_update;
    double mean_xBtot_after_gp_to_beta_event;
    double mean_xBtot_after_Y_update;
    double mean_xBtot_after_Y_to_xB;
    double mean_xBtot_before_clipping;
    double mean_xBtot_after_clipping;
    double mean_xBtot_end_step;
    double mean_xBtot_gp_before_step;
    double mean_xBtot_gp_after_phi_update;
    double mean_xBtot_gp_after_eta_update;
    double mean_xBtot_gp_after_gp_to_beta_event;
    double mean_xBtot_gp_after_Y_update;
    double mean_xBtot_gp_end_step;
    double mean_xBtot_gp_before_Y;
    double mean_xB_alpha_before_Y;
    double mean_storage_GP_before_Y;
    double mean_storage_beta_before_Y;
    double mean_xB_alpha_after_Y;
    double mean_storage_GP_after_Y;
    double mean_storage_beta_after_Y;
    double mean_delta_xB_alpha;
    double mean_delta_storage_GP;
    double mean_delta_storage_beta;
    double mean_delta_xBtot_reconstructed;
    double mean_dt_divJ;
    double gp_closure_error;
    double gp_clipping_mass_error;
    double gp_small_h_alpha_mass_error;
    double gp_small_h_alpha_count;
    double gp_min_h_alpha;
    double gp_max_h_alpha;
    double gp_min_h_GP;
    double gp_max_h_GP;
    double gp_min_eta;
    double gp_max_eta;
    double gp_min_xB_alpha;
    double gp_max_xB_alpha;
    double xB_alpha_perturb_maxabs;
    double dmuC_dx_ref_numeric;
    double eta_step_max;
    double eta_step_max_delta;
    double eta_integral;
    double gp_y_update_mode_code;
    double mu_C_mean;
    double mu_C_min;
    double mu_C_max;
    double mean_elastic_energy;
    double max_elastic_energy;
    double stress_hydro_min;
    double stress_hydro_max;
    double eps0_GP_contrib_min;
    double eps0_GP_contrib_max;
    double mean_dgel_deta_el;
    double max_abs_dgel_deta_el;
    double mean_dgel_dphi_el;
    double max_abs_dgel_dphi_el;
    double mean_eta_rhs_elastic_part;
    double max_abs_eta_rhs_elastic_part;
    double mean_phi_rhs_elastic_part;
    double max_abs_phi_rhs_elastic_part;
    double gp_mu_reference_raw;
    double gp_minus_delta_mu_r_mean;
    double gp_minus_delta_mu_r_min;
    double gp_minus_delta_mu_r_max;
    double grad_mu_x_min;
    double grad_mu_x_max;
    double grad_mu_y_min;
    double grad_mu_y_max;
    double grad_mu_z_min;
    double grad_mu_z_max;
    double M_alpha_min;
    double M_alpha_max;
    double h_alpha_transport_min;
    double h_alpha_transport_max;
    double M_eff_min;
    double M_eff_max;
    double flux_x_min;
    double flux_x_max;
    double flux_y_min;
    double flux_y_max;
    double flux_z_min;
    double flux_z_max;
    double divJ_min;
    double divJ_max;
    double dt_divJ_min;
    double dt_divJ_max;
    double max_abs_xBtot_target_delta;
    double max_abs_xB_alpha_delta;
    double recovered_xB_min;
    double recovered_xB_max;
    double recovered_xB_gt_xBmax_count;
    double recovered_xB_lt_xBmin_count;
    double recovered_xB_min_after_limiter;
    double recovered_xB_max_after_limiter;
    double recovered_xB_gt_xBmax_count_after_limiter;
    double recovered_xB_lt_xBmin_count_after_limiter;
    double recovered_xB_invalid_count_before_limiter;
    double recovered_xB_invalid_count_after_limiter;
    double recovered_xB_invalid_mass;
    double h_GP_new_max;
    double eta_new_max;
    double eta_tentative_max;
    double eta_limiter_count;
    double eta_limiter_max_delta;
    double eta_limiter_mass_prevented;
    double delta_mass_phi_update;
    double delta_mass_eta_update;
    double delta_mass_gp_to_beta_event;
    double delta_mass_Y_update;
    double delta_mass_Y_to_xB;
    double delta_mass_clipping;
    double total_delta_mass_step;
    double mean_h_before_phi_update;
    double mean_h_after_phi_update;
    double delta_mean_h_phi_update;
    double mean_xB_before_phi_update;
    double mean_xB_after_Y_update;
    double mean_xB_after_clipping;
    double predicted_delta_xBtot_from_phi_change;
    double mean_phi_before;
    double mean_phi_after;
    double Y_k0_before_re;
    double Y_k0_before_im;
    double Y_rhs_k0_re;
    double Y_rhs_k0_im;
    double Y_k0_after_re;
    double Y_k0_after_im;
    double delta_Y_k0_re;
    double delta_Y_k0_im;
    double phi_clip_count_low;
    double phi_clip_count_high;
    double Y_clip_count_low;
    double Y_clip_count_high;
    double xB_clip_count_low;
    double xB_clip_count_high;
    double mean_xBtot_before_phi_clip;
    double mean_xBtot_after_phi_clip;
    double mean_xBtot_before_Y_clip;
    double mean_xBtot_after_Y_clip;
    double mean_xBtot_before_xB_clip;
    double mean_xBtot_after_xB_clip;
    double delta_mass_phi_clip;
    double delta_mass_Y_clip;
    double delta_mass_xB_clip;
    double y_rhs_prev_time_level_enabled;
    double mean_xBtot_old;
    double mean_h_old_for_Y_rhs;
    double mean_h_new_for_Y_rhs;
    double mean_h_alpha_q_for_Y_rhs;
    double min_h_alpha_for_Y_rhs;
    double min_h_alpha_q_for_Y_rhs;
    double max_h_alpha_q_for_Y_rhs;
    double mean_gamma_local;
    double mean_term_h;
    double mean_term_lap;
    double mean_term_gamma;
    double mean_divJ;
    double mean_divJ_effective;
    double mean_lapY;
    double mean_Y_rhs_total;
    double k0_divJ_re;
    double k0_divJ_im;
    double k0_term_h_re;
    double k0_term_h_im;
    double k0_term_lap_re;
    double k0_term_lap_im;
    double k0_term_gamma_re;
    double k0_term_gamma_im;
    double k0_Y_rhs_total_re;
    double k0_Y_rhs_total_im;
    double l1_divJ;
    double l1_term_h;
    double l1_term_lap;
    double l1_term_gamma;
    double l1_Y_rhs_total;
    double l2_divJ;
    double l2_term_h;
    double l2_term_lap;
    double l2_term_gamma;
    double l2_Y_rhs_total;
    double maxabs_divJ;
    double maxabs_term_h;
    double maxabs_term_lap;
    double maxabs_term_gamma;
    double maxabs_Y_rhs_total;
    double max_abs_fY;
    double max_abs_lagged_dYdt;
    double conservative_balance_residual;
    double conservative_residual_plus;
    double conservative_residual_minus;
    double delta_M_phi;
    double delta_M_Y_actual;
    double delta_M_Y_required;
    double Y_compensation_ratio;
    double predicted_delta_M_from_plus_divJ;
    double predicted_delta_M_from_minus_divJ;
    double divJ_k0_real_before_update;
    double divJ_k0_imag_before_update;
    double y_rhs_picard_enabled;
    double y_rhs_picard_iters;
    double y_rhs_picard_omega;
    double picard_rms_dYdt_change_final;
    double picard_maxabs_dYdt_change_final;
    double picard_mean_xBtot_after_iter_0;
    double picard_mean_xBtot_after_final;
    double picard_Y_compensation_ratio_final;
} DynamicsMassDiagRow;

static const char *infer_primary_mass_source(const DynamicsMassDiagSummary *s) {
    if (!s) return "mixed_or_unclear";
    double a_phi = fabs(s->cumulative_delta_phi_update);
    double a_eta = fabs(s->cumulative_delta_eta_update);
    double a_gp_event = fabs(s->cumulative_delta_gp_to_beta_event);
    double a_Y = fabs(s->cumulative_delta_Y_update);
    double a_conv = fabs(s->cumulative_delta_Y_to_xB);
    double a_clip = fabs(s->cumulative_delta_clipping);
    double a_other = fabs(s->cumulative_delta_other);
    double best = a_phi;
    const char *label = "phi_update_splitting";
    if (a_eta > best) { best = a_eta; label = "eta_update_storage"; }
    if (a_gp_event > best) { best = a_gp_event; label = "gp_to_beta_event"; }
    if (a_Y > best) { best = a_Y; label = "Y_update_k0_drift"; }
    if (a_conv > best) { best = a_conv; label = "Y_to_xB_conversion"; }
    if (a_clip > best) { best = a_clip; label = "clipping"; }
    if (a_other > best) { best = a_other; label = "mixed_or_unclear"; }
    return label;
}

static void write_mass_diag_csv_header(FILE *fp, const PFParams *P) {
    if (!fp || !P) return;
    if (is_gp_zone_mode(P)) {
        fprintf(fp,
                "step,time,"
                "y_rhs_prev_time_level_enabled,"
                "mean_xBtot_gp_old,"
                "mean_xBtot_gp_before_step,mean_xBtot_gp_after_phi_update,mean_xBtot_gp_after_eta_update,mean_xBtot_gp_after_gp_to_beta_event,mean_xBtot_gp_after_Y_update,mean_xBtot_gp_after_Y_to_xB,"
                "mean_xBtot_gp_before_clipping,mean_xBtot_gp_after_clipping,mean_xBtot_gp_end_step,"
                "gp_y_update_mode_code,"
                "mean_xBtot_gp_before_Y,mean_xB_alpha_before_Y,mean_storage_GP_before_Y,mean_storage_beta_before_Y,"
                "mean_xB_alpha_after_Y,mean_storage_GP_after_Y,mean_storage_beta_after_Y,"
                "mean_delta_xB_alpha,mean_delta_storage_GP,mean_delta_storage_beta,mean_delta_xBtot_reconstructed,mean_dt_divJ,gp_closure_error,"
                "gp_clipping_mass_error,gp_small_h_alpha_mass_error,gp_small_h_alpha_count,"
                "gp_min_h_alpha,gp_max_h_alpha,gp_min_h_GP,gp_max_h_GP,gp_min_eta,gp_max_eta,gp_min_xB_alpha,gp_max_xB_alpha,"
                "xB_alpha_perturb_maxabs,dmuC_dx_ref_numeric,"
                "eta_step_max,eta_step_max_delta,eta_integral,mu_C_mean,mu_C_min,mu_C_max,"
                "mean_elastic_energy,max_elastic_energy,stress_hydro_min,stress_hydro_max,eps0_GP_contrib_min,eps0_GP_contrib_max,"
                "mean_dgel_deta_el,max_abs_dgel_deta_el,mean_dgel_dphi_el,max_abs_dgel_dphi_el,"
                "mean_eta_rhs_elastic_part,max_abs_eta_rhs_elastic_part,mean_phi_rhs_elastic_part,max_abs_phi_rhs_elastic_part,"
                "gp_mu_reference_raw,gp_minus_delta_mu_r_mean,gp_minus_delta_mu_r_min,gp_minus_delta_mu_r_max,"
                "grad_mu_x_min,grad_mu_x_max,grad_mu_y_min,grad_mu_y_max,grad_mu_z_min,grad_mu_z_max,"
                "M_alpha_min,M_alpha_max,h_alpha_transport_min,h_alpha_transport_max,M_eff_min,M_eff_max,"
                "flux_x_min,flux_x_max,flux_y_min,flux_y_max,flux_z_min,flux_z_max,"
                "divJ_min,divJ_max,dt_divJ_min,dt_divJ_max,max_abs_xBtot_target_delta,max_abs_xB_alpha_delta,"
                "recovered_xB_min,recovered_xB_max,recovered_xB_gt_xBmax_count,recovered_xB_lt_xBmin_count,"
                "recovered_xB_min_after_limiter,recovered_xB_max_after_limiter,recovered_xB_gt_xBmax_count_after_limiter,recovered_xB_lt_xBmin_count_after_limiter,"
                "recovered_xB_invalid_count_before_limiter,recovered_xB_invalid_count_after_limiter,"
                "recovered_xB_invalid_mass,h_GP_new_max,eta_new_max,eta_tentative_max,"
                "eta_limiter_count,eta_limiter_max_delta,eta_limiter_mass_prevented,"
                "delta_mass_phi_update,delta_mass_eta_update,delta_mass_gp_to_beta_event,delta_mass_Y_update,delta_mass_Y_to_xB,delta_mass_clipping,total_delta_mass_step,"
                "delta_M_phi,delta_M_Y_actual,delta_M_Y_required,Y_compensation_ratio,"
                "mean_h_before_phi_update,mean_h_after_phi_update,delta_mean_h_phi_update,"
                "mean_xB_before_phi_update,mean_xB_after_Y_update,mean_xB_after_clipping,"
                "mean_h_old_for_Y_rhs,mean_h_new_for_Y_rhs,mean_h_alpha_q_for_Y_rhs,min_h_alpha_for_Y_rhs,min_h_alpha_q_for_Y_rhs,max_h_alpha_q_for_Y_rhs,mean_gamma_local,mean_term_h,mean_term_lap,mean_term_gamma,mean_divJ,mean_divJ_effective,mean_lapY,mean_Y_rhs_total,"
                "k0_divJ_re,k0_divJ_im,k0_term_h_re,k0_term_h_im,k0_term_lap_re,k0_term_lap_im,k0_term_gamma_re,k0_term_gamma_im,k0_Y_rhs_total_re,k0_Y_rhs_total_im,"
                "l1_divJ,l1_term_h,l1_term_lap,l1_term_gamma,l1_Y_rhs_total,"
                "l2_divJ,l2_term_h,l2_term_lap,l2_term_gamma,l2_Y_rhs_total,"
                "maxabs_divJ,maxabs_term_h,maxabs_term_lap,maxabs_term_gamma,maxabs_Y_rhs_total,max_abs_fY,max_abs_lagged_dYdt,"
                "divJ_k0_real_before_update,divJ_k0_imag_before_update,"
                "Y_rhs_picard_enabled,Y_rhs_picard_iters,Y_rhs_picard_omega,"
                "picard_rms_dYdt_change_final,picard_maxabs_dYdt_change_final,"
                "picard_mean_xBtot_gp_after_iter_0,picard_mean_xBtot_gp_after_final,picard_Y_compensation_ratio_final,"
                "conservative_residual_plus,conservative_residual_minus,conservative_balance_residual,"
                "predicted_delta_M_from_plus_divJ,predicted_delta_M_from_minus_divJ,"
                "predicted_delta_xBtot_gp_from_phi_change,mean_phi_before,mean_phi_after,"
                "Y_k0_before_re,Y_k0_before_im,Y_rhs_k0_re,Y_rhs_k0_im,Y_k0_after_re,Y_k0_after_im,delta_Y_k0_re,delta_Y_k0_im,"
                "phi_clip_count_low,phi_clip_count_high,Y_clip_count_low,Y_clip_count_high,xB_clip_count_low,xB_clip_count_high,"
                "mean_xBtot_gp_before_phi_clip,mean_xBtot_gp_after_phi_clip,"
                "mean_xBtot_gp_before_Y_clip,mean_xBtot_gp_after_Y_clip,"
                "mean_xBtot_gp_before_xB_clip,mean_xBtot_gp_after_xB_clip,"
                "delta_mass_phi_clip,delta_mass_Y_clip,delta_mass_xB_clip\n");
        return;
    }
    fprintf(fp,
            "step,time,"
            "y_rhs_prev_time_level_enabled,"
            "mean_xBtot_old,"
            "mean_xBtot_before_step,mean_xBtot_after_phi_update,mean_xBtot_after_Y_update,mean_xBtot_after_Y_to_xB,"
            "mean_xBtot_before_clipping,mean_xBtot_after_clipping,mean_xBtot_end_step,"
            "delta_mass_phi_update,delta_mass_Y_update,delta_mass_Y_to_xB,delta_mass_clipping,total_delta_mass_step,"
            "delta_M_phi,delta_M_Y_actual,delta_M_Y_required,Y_compensation_ratio,"
            "mean_h_before_phi_update,mean_h_after_phi_update,delta_mean_h_phi_update,"
            "mean_xB_before_phi_update,mean_xB_after_Y_update,mean_xB_after_clipping,"
            "mean_h_old_for_Y_rhs,mean_h_new_for_Y_rhs,mean_h_alpha_q_for_Y_rhs,min_h_alpha_for_Y_rhs,min_h_alpha_q_for_Y_rhs,max_h_alpha_q_for_Y_rhs,mean_gamma_local,mean_term_h,mean_term_lap,mean_term_gamma,mean_divJ,mean_divJ_effective,mean_lapY,mean_Y_rhs_total,"
            "k0_divJ_re,k0_divJ_im,k0_term_h_re,k0_term_h_im,k0_term_lap_re,k0_term_lap_im,k0_term_gamma_re,k0_term_gamma_im,k0_Y_rhs_total_re,k0_Y_rhs_total_im,"
            "l1_divJ,l1_term_h,l1_term_lap,l1_term_gamma,l1_Y_rhs_total,"
            "l2_divJ,l2_term_h,l2_term_lap,l2_term_gamma,l2_Y_rhs_total,"
            "maxabs_divJ,maxabs_term_h,maxabs_term_lap,maxabs_term_gamma,maxabs_Y_rhs_total,max_abs_fY,max_abs_lagged_dYdt,"
            "divJ_k0_real_before_update,divJ_k0_imag_before_update,"
            "Y_rhs_picard_enabled,Y_rhs_picard_iters,Y_rhs_picard_omega,"
            "picard_rms_dYdt_change_final,picard_maxabs_dYdt_change_final,"
            "picard_mean_xBtot_after_iter_0,picard_mean_xBtot_after_final,picard_Y_compensation_ratio_final,"
            "conservative_residual_plus,conservative_residual_minus,conservative_balance_residual,"
            "predicted_delta_M_from_plus_divJ,predicted_delta_M_from_minus_divJ,"
            "predicted_delta_xBtot_from_phi_change,mean_phi_before,mean_phi_after,"
            "Y_k0_before_re,Y_k0_before_im,Y_rhs_k0_re,Y_rhs_k0_im,Y_k0_after_re,Y_k0_after_im,delta_Y_k0_re,delta_Y_k0_im,"
            "phi_clip_count_low,phi_clip_count_high,Y_clip_count_low,Y_clip_count_high,xB_clip_count_low,xB_clip_count_high,"
            "mean_xBtot_before_phi_clip,mean_xBtot_after_phi_clip,"
            "mean_xBtot_before_Y_clip,mean_xBtot_after_Y_clip,"
            "mean_xBtot_before_xB_clip,mean_xBtot_after_xB_clip,"
            "delta_mass_phi_clip,delta_mass_Y_clip,delta_mass_xB_clip\n");
}

static void write_mass_diag_row_csv(FILE *fp, const DynamicsMassDiagRow *r, const PFParams *P) {
    if (!fp || !r) return;
#define CSV_I(v) fprintf(fp, "%d", (int)(v))
#define CSV_D(v) fprintf(fp, "%.10e", (double)(v))
#define CSV_COMMA() fputc(',', fp)
    CSV_I(r->step); CSV_COMMA();
    CSV_D(r->time_code); CSV_COMMA();
    CSV_D(r->y_rhs_prev_time_level_enabled); CSV_COMMA();
    if (P && is_gp_zone_mode(P)) {
        CSV_D(r->mean_xBtot_old); CSV_COMMA();
        CSV_D(r->mean_xBtot_gp_before_step); CSV_COMMA();
        CSV_D(r->mean_xBtot_gp_after_phi_update); CSV_COMMA();
        CSV_D(r->mean_xBtot_gp_after_eta_update); CSV_COMMA();
        CSV_D(r->mean_xBtot_gp_after_gp_to_beta_event); CSV_COMMA();
        CSV_D(r->mean_xBtot_gp_after_Y_update); CSV_COMMA();
        CSV_D(r->mean_xBtot_after_Y_to_xB); CSV_COMMA();
        CSV_D(r->mean_xBtot_before_clipping); CSV_COMMA();
        CSV_D(r->mean_xBtot_after_clipping); CSV_COMMA();
        CSV_D(r->mean_xBtot_gp_end_step); CSV_COMMA();
        CSV_D(r->gp_y_update_mode_code); CSV_COMMA();
        CSV_D(r->mean_xBtot_gp_before_Y); CSV_COMMA();
        CSV_D(r->mean_xB_alpha_before_Y); CSV_COMMA();
        CSV_D(r->mean_storage_GP_before_Y); CSV_COMMA();
        CSV_D(r->mean_storage_beta_before_Y); CSV_COMMA();
        CSV_D(r->mean_xB_alpha_after_Y); CSV_COMMA();
        CSV_D(r->mean_storage_GP_after_Y); CSV_COMMA();
        CSV_D(r->mean_storage_beta_after_Y); CSV_COMMA();
        CSV_D(r->mean_delta_xB_alpha); CSV_COMMA();
        CSV_D(r->mean_delta_storage_GP); CSV_COMMA();
        CSV_D(r->mean_delta_storage_beta); CSV_COMMA();
        CSV_D(r->mean_delta_xBtot_reconstructed); CSV_COMMA();
        CSV_D(r->mean_dt_divJ); CSV_COMMA();
        CSV_D(r->gp_closure_error); CSV_COMMA();
        CSV_D(r->gp_clipping_mass_error); CSV_COMMA();
        CSV_D(r->gp_small_h_alpha_mass_error); CSV_COMMA();
        CSV_D(r->gp_small_h_alpha_count); CSV_COMMA();
        CSV_D(r->gp_min_h_alpha); CSV_COMMA();
        CSV_D(r->gp_max_h_alpha); CSV_COMMA();
        CSV_D(r->gp_min_h_GP); CSV_COMMA();
        CSV_D(r->gp_max_h_GP); CSV_COMMA();
        CSV_D(r->gp_min_eta); CSV_COMMA();
        CSV_D(r->gp_max_eta); CSV_COMMA();
        CSV_D(r->gp_min_xB_alpha); CSV_COMMA();
        CSV_D(r->gp_max_xB_alpha); CSV_COMMA();
        CSV_D(r->xB_alpha_perturb_maxabs); CSV_COMMA();
        CSV_D(r->dmuC_dx_ref_numeric); CSV_COMMA();
        CSV_D(r->eta_step_max); CSV_COMMA();
        CSV_D(r->eta_step_max_delta); CSV_COMMA();
        CSV_D(r->eta_integral); CSV_COMMA();
        CSV_D(r->mu_C_mean); CSV_COMMA();
        CSV_D(r->mu_C_min); CSV_COMMA();
        CSV_D(r->mu_C_max); CSV_COMMA();
        CSV_D(r->mean_elastic_energy); CSV_COMMA();
        CSV_D(r->max_elastic_energy); CSV_COMMA();
        CSV_D(r->stress_hydro_min); CSV_COMMA();
        CSV_D(r->stress_hydro_max); CSV_COMMA();
        CSV_D(r->eps0_GP_contrib_min); CSV_COMMA();
        CSV_D(r->eps0_GP_contrib_max); CSV_COMMA();
        CSV_D(r->mean_dgel_deta_el); CSV_COMMA();
        CSV_D(r->max_abs_dgel_deta_el); CSV_COMMA();
        CSV_D(r->mean_dgel_dphi_el); CSV_COMMA();
        CSV_D(r->max_abs_dgel_dphi_el); CSV_COMMA();
        CSV_D(r->mean_eta_rhs_elastic_part); CSV_COMMA();
        CSV_D(r->max_abs_eta_rhs_elastic_part); CSV_COMMA();
        CSV_D(r->mean_phi_rhs_elastic_part); CSV_COMMA();
        CSV_D(r->max_abs_phi_rhs_elastic_part); CSV_COMMA();
        CSV_D(r->gp_mu_reference_raw); CSV_COMMA();
        CSV_D(r->gp_minus_delta_mu_r_mean); CSV_COMMA();
        CSV_D(r->gp_minus_delta_mu_r_min); CSV_COMMA();
        CSV_D(r->gp_minus_delta_mu_r_max); CSV_COMMA();
        CSV_D(r->grad_mu_x_min); CSV_COMMA();
        CSV_D(r->grad_mu_x_max); CSV_COMMA();
        CSV_D(r->grad_mu_y_min); CSV_COMMA();
        CSV_D(r->grad_mu_y_max); CSV_COMMA();
        CSV_D(r->grad_mu_z_min); CSV_COMMA();
        CSV_D(r->grad_mu_z_max); CSV_COMMA();
        CSV_D(r->M_alpha_min); CSV_COMMA();
        CSV_D(r->M_alpha_max); CSV_COMMA();
        CSV_D(r->h_alpha_transport_min); CSV_COMMA();
        CSV_D(r->h_alpha_transport_max); CSV_COMMA();
        CSV_D(r->M_eff_min); CSV_COMMA();
        CSV_D(r->M_eff_max); CSV_COMMA();
        CSV_D(r->flux_x_min); CSV_COMMA();
        CSV_D(r->flux_x_max); CSV_COMMA();
        CSV_D(r->flux_y_min); CSV_COMMA();
        CSV_D(r->flux_y_max); CSV_COMMA();
        CSV_D(r->flux_z_min); CSV_COMMA();
        CSV_D(r->flux_z_max); CSV_COMMA();
        CSV_D(r->divJ_min); CSV_COMMA();
        CSV_D(r->divJ_max); CSV_COMMA();
        CSV_D(r->dt_divJ_min); CSV_COMMA();
        CSV_D(r->dt_divJ_max); CSV_COMMA();
        CSV_D(r->max_abs_xBtot_target_delta); CSV_COMMA();
        CSV_D(r->max_abs_xB_alpha_delta); CSV_COMMA();
        CSV_D(r->recovered_xB_min); CSV_COMMA();
        CSV_D(r->recovered_xB_max); CSV_COMMA();
        CSV_D(r->recovered_xB_gt_xBmax_count); CSV_COMMA();
        CSV_D(r->recovered_xB_lt_xBmin_count); CSV_COMMA();
        CSV_D(r->recovered_xB_min_after_limiter); CSV_COMMA();
        CSV_D(r->recovered_xB_max_after_limiter); CSV_COMMA();
        CSV_D(r->recovered_xB_gt_xBmax_count_after_limiter); CSV_COMMA();
        CSV_D(r->recovered_xB_lt_xBmin_count_after_limiter); CSV_COMMA();
        CSV_D(r->recovered_xB_invalid_count_before_limiter); CSV_COMMA();
        CSV_D(r->recovered_xB_invalid_count_after_limiter); CSV_COMMA();
        CSV_D(r->recovered_xB_invalid_mass); CSV_COMMA();
        CSV_D(r->h_GP_new_max); CSV_COMMA();
        CSV_D(r->eta_new_max); CSV_COMMA();
        CSV_D(r->eta_tentative_max); CSV_COMMA();
        CSV_D(r->eta_limiter_count); CSV_COMMA();
        CSV_D(r->eta_limiter_max_delta); CSV_COMMA();
        CSV_D(r->eta_limiter_mass_prevented); CSV_COMMA();
        CSV_D(r->delta_mass_phi_update); CSV_COMMA();
        CSV_D(r->delta_mass_eta_update); CSV_COMMA();
        CSV_D(r->delta_mass_gp_to_beta_event); CSV_COMMA();
    } else {
    CSV_D(r->mean_xBtot_old); CSV_COMMA();
    CSV_D(r->mean_xBtot_before_step); CSV_COMMA();
    CSV_D(r->mean_xBtot_after_phi_update); CSV_COMMA();
    CSV_D(r->mean_xBtot_after_Y_update); CSV_COMMA();
    CSV_D(r->mean_xBtot_after_Y_to_xB); CSV_COMMA();
    CSV_D(r->mean_xBtot_before_clipping); CSV_COMMA();
    CSV_D(r->mean_xBtot_after_clipping); CSV_COMMA();
    CSV_D(r->mean_xBtot_end_step); CSV_COMMA();
    CSV_D(r->delta_mass_phi_update); CSV_COMMA();
    }
    CSV_D(r->delta_mass_Y_update); CSV_COMMA();
    CSV_D(r->delta_mass_Y_to_xB); CSV_COMMA();
    CSV_D(r->delta_mass_clipping); CSV_COMMA();
    CSV_D(r->total_delta_mass_step); CSV_COMMA();
    CSV_D(r->delta_M_phi); CSV_COMMA();
    CSV_D(r->delta_M_Y_actual); CSV_COMMA();
    CSV_D(r->delta_M_Y_required); CSV_COMMA();
    CSV_D(r->Y_compensation_ratio); CSV_COMMA();
    CSV_D(r->mean_h_before_phi_update); CSV_COMMA();
    CSV_D(r->mean_h_after_phi_update); CSV_COMMA();
    CSV_D(r->delta_mean_h_phi_update); CSV_COMMA();
    CSV_D(r->mean_xB_before_phi_update); CSV_COMMA();
    CSV_D(r->mean_xB_after_Y_update); CSV_COMMA();
    CSV_D(r->mean_xB_after_clipping); CSV_COMMA();
    CSV_D(r->mean_h_old_for_Y_rhs); CSV_COMMA();
    CSV_D(r->mean_h_new_for_Y_rhs); CSV_COMMA();
    CSV_D(r->mean_h_alpha_q_for_Y_rhs); CSV_COMMA();
    CSV_D(r->min_h_alpha_for_Y_rhs); CSV_COMMA();
    CSV_D(r->min_h_alpha_q_for_Y_rhs); CSV_COMMA();
    CSV_D(r->max_h_alpha_q_for_Y_rhs); CSV_COMMA();
    CSV_D(r->mean_gamma_local); CSV_COMMA();
    CSV_D(r->mean_term_h); CSV_COMMA();
    CSV_D(r->mean_term_lap); CSV_COMMA();
    CSV_D(r->mean_term_gamma); CSV_COMMA();
    CSV_D(r->mean_divJ); CSV_COMMA();
    CSV_D(r->mean_divJ_effective); CSV_COMMA();
    CSV_D(r->mean_lapY); CSV_COMMA();
    CSV_D(r->mean_Y_rhs_total); CSV_COMMA();
    CSV_D(r->k0_divJ_re); CSV_COMMA();
    CSV_D(r->k0_divJ_im); CSV_COMMA();
    CSV_D(r->k0_term_h_re); CSV_COMMA();
    CSV_D(r->k0_term_h_im); CSV_COMMA();
    CSV_D(r->k0_term_lap_re); CSV_COMMA();
    CSV_D(r->k0_term_lap_im); CSV_COMMA();
    CSV_D(r->k0_term_gamma_re); CSV_COMMA();
    CSV_D(r->k0_term_gamma_im); CSV_COMMA();
    CSV_D(r->k0_Y_rhs_total_re); CSV_COMMA();
    CSV_D(r->k0_Y_rhs_total_im); CSV_COMMA();
    CSV_D(r->l1_divJ); CSV_COMMA();
    CSV_D(r->l1_term_h); CSV_COMMA();
    CSV_D(r->l1_term_lap); CSV_COMMA();
    CSV_D(r->l1_term_gamma); CSV_COMMA();
    CSV_D(r->l1_Y_rhs_total); CSV_COMMA();
    CSV_D(r->l2_divJ); CSV_COMMA();
    CSV_D(r->l2_term_h); CSV_COMMA();
    CSV_D(r->l2_term_lap); CSV_COMMA();
    CSV_D(r->l2_term_gamma); CSV_COMMA();
    CSV_D(r->l2_Y_rhs_total); CSV_COMMA();
    CSV_D(r->maxabs_divJ); CSV_COMMA();
    CSV_D(r->maxabs_term_h); CSV_COMMA();
    CSV_D(r->maxabs_term_lap); CSV_COMMA();
    CSV_D(r->maxabs_term_gamma); CSV_COMMA();
    CSV_D(r->maxabs_Y_rhs_total); CSV_COMMA();
    CSV_D(r->max_abs_fY); CSV_COMMA();
    CSV_D(r->max_abs_lagged_dYdt); CSV_COMMA();
    CSV_D(r->divJ_k0_real_before_update); CSV_COMMA();
    CSV_D(r->divJ_k0_imag_before_update); CSV_COMMA();
    CSV_D(r->y_rhs_picard_enabled); CSV_COMMA();
    CSV_D(r->y_rhs_picard_iters); CSV_COMMA();
    CSV_D(r->y_rhs_picard_omega); CSV_COMMA();
    CSV_D(r->picard_rms_dYdt_change_final); CSV_COMMA();
    CSV_D(r->picard_maxabs_dYdt_change_final); CSV_COMMA();
    CSV_D(r->picard_mean_xBtot_after_iter_0); CSV_COMMA();
    CSV_D(r->picard_mean_xBtot_after_final); CSV_COMMA();
    CSV_D(r->picard_Y_compensation_ratio_final); CSV_COMMA();
    CSV_D(r->conservative_residual_plus); CSV_COMMA();
    CSV_D(r->conservative_residual_minus); CSV_COMMA();
    CSV_D(r->conservative_balance_residual); CSV_COMMA();
    CSV_D(r->predicted_delta_M_from_plus_divJ); CSV_COMMA();
    CSV_D(r->predicted_delta_M_from_minus_divJ); CSV_COMMA();
    CSV_D(r->predicted_delta_xBtot_from_phi_change); CSV_COMMA();
    CSV_D(r->mean_phi_before); CSV_COMMA();
    CSV_D(r->mean_phi_after); CSV_COMMA();
    CSV_D(r->Y_k0_before_re); CSV_COMMA();
    CSV_D(r->Y_k0_before_im); CSV_COMMA();
    CSV_D(r->Y_rhs_k0_re); CSV_COMMA();
    CSV_D(r->Y_rhs_k0_im); CSV_COMMA();
    CSV_D(r->Y_k0_after_re); CSV_COMMA();
    CSV_D(r->Y_k0_after_im); CSV_COMMA();
    CSV_D(r->delta_Y_k0_re); CSV_COMMA();
    CSV_D(r->delta_Y_k0_im); CSV_COMMA();
    CSV_D(r->phi_clip_count_low); CSV_COMMA();
    CSV_D(r->phi_clip_count_high); CSV_COMMA();
    CSV_D(r->Y_clip_count_low); CSV_COMMA();
    CSV_D(r->Y_clip_count_high); CSV_COMMA();
    CSV_D(r->xB_clip_count_low); CSV_COMMA();
    CSV_D(r->xB_clip_count_high); CSV_COMMA();
    if (P && is_gp_zone_mode(P)) {
        CSV_D(r->mean_xBtot_before_phi_clip); CSV_COMMA();
        CSV_D(r->mean_xBtot_after_phi_clip); CSV_COMMA();
        CSV_D(r->mean_xBtot_before_Y_clip); CSV_COMMA();
        CSV_D(r->mean_xBtot_after_Y_clip); CSV_COMMA();
        CSV_D(r->mean_xBtot_before_xB_clip); CSV_COMMA();
        CSV_D(r->mean_xBtot_after_xB_clip); CSV_COMMA();
    } else {
    CSV_D(r->mean_xBtot_before_phi_clip); CSV_COMMA();
    CSV_D(r->mean_xBtot_after_phi_clip); CSV_COMMA();
    CSV_D(r->mean_xBtot_before_Y_clip); CSV_COMMA();
    CSV_D(r->mean_xBtot_after_Y_clip); CSV_COMMA();
    CSV_D(r->mean_xBtot_before_xB_clip); CSV_COMMA();
    CSV_D(r->mean_xBtot_after_xB_clip); CSV_COMMA();
    }
    CSV_D(r->delta_mass_phi_clip); CSV_COMMA();
    CSV_D(r->delta_mass_Y_clip); CSV_COMMA();
    CSV_D(r->delta_mass_xB_clip);
    fputc('\n', fp);
#undef CSV_I
#undef CSV_D
#undef CSV_COMMA
}

static void write_mass_drift_summary_json(const char *path, const DynamicsMassDiagSummary *s, const PFParams *P) {
    if (!path || !s || !P) return;
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "[warn] cannot open mass drift summary json %s\n", path);
        return;
    }
    if (is_gp_zone_mode(P)) {
        fprintf(fp,
                "{\n"
                "  \"initial_mean_xBtot_gp\": %.17e,\n"
                "  \"final_mean_xBtot_gp\": %.17e,\n"
                "  \"total_absolute_drift\": %.17e,\n"
                "  \"total_relative_drift\": %.17e,\n"
                "  \"cumulative_delta_phi_update\": %.17e,\n"
                "  \"cumulative_delta_eta_update\": %.17e,\n"
                "  \"cumulative_delta_gp_to_beta_event\": %.17e,\n"
                "  \"cumulative_delta_Y_update\": %.17e,\n"
                "  \"cumulative_delta_Y_to_xB\": %.17e,\n"
                "  \"cumulative_delta_clipping\": %.17e,\n"
                "  \"cumulative_delta_other\": %.17e,\n"
                "  \"max_abs_step_drift\": %.17e,\n"
                "  \"max_abs_delta_phi_update\": %.17e,\n"
                "  \"max_abs_delta_eta_update\": %.17e,\n"
                "  \"max_abs_delta_gp_to_beta_event\": %.17e,\n"
                "  \"max_abs_delta_Y_update\": %.17e,\n"
                "  \"max_abs_delta_Y_to_xB\": %.17e,\n"
                "  \"max_abs_delta_clipping\": %.17e,\n"
                "  \"total_clip_count_phi\": %.17e,\n"
                "  \"total_clip_count_xB\": %.17e,\n"
                "  \"total_clip_count_Y\": %.17e,\n"
                "  \"summary_matches_csv\": %.17e,\n"
                "  \"max_summary_csv_abs_mismatch\": %.17e,\n"
                "  \"max_abs_mass_closure_resid\": %.17e,\n"
                "  \"suspected_primary_source\": \"%s\"\n"
                "}\n",
                s->initial_mean_xBtot,
                s->final_mean_xBtot,
                s->total_absolute_drift,
                s->total_relative_drift,
                s->cumulative_delta_phi_update,
                s->cumulative_delta_eta_update,
                s->cumulative_delta_gp_to_beta_event,
                s->cumulative_delta_Y_update,
                s->cumulative_delta_Y_to_xB,
                s->cumulative_delta_clipping,
                s->cumulative_delta_other,
                s->max_abs_step_drift,
                s->max_abs_delta_phi_update,
                s->max_abs_delta_eta_update,
                s->max_abs_delta_gp_to_beta_event,
                s->max_abs_delta_Y_update,
                s->max_abs_delta_Y_to_xB,
                s->max_abs_delta_clipping,
                s->total_clip_count_phi,
                s->total_clip_count_xB,
                s->total_clip_count_Y,
                s->summary_matches_csv,
                s->max_summary_csv_abs_mismatch,
                s->max_abs_mass_closure_resid,
                s->suspected_primary_source ? s->suspected_primary_source : "mixed_or_unclear");
        fclose(fp);
        return;
    }
    fprintf(fp,
            "{\n"
            "  \"initial_mean_xBtot\": %.17e,\n"
            "  \"final_mean_xBtot\": %.17e,\n"
            "  \"total_absolute_drift\": %.17e,\n"
            "  \"total_relative_drift\": %.17e,\n"
            "  \"cumulative_delta_phi_update\": %.17e,\n"
            "  \"cumulative_delta_Y_update\": %.17e,\n"
            "  \"cumulative_delta_Y_to_xB\": %.17e,\n"
            "  \"cumulative_delta_clipping\": %.17e,\n"
            "  \"cumulative_delta_other\": %.17e,\n"
            "  \"max_abs_step_drift\": %.17e,\n"
            "  \"max_abs_delta_phi_update\": %.17e,\n"
            "  \"max_abs_delta_Y_update\": %.17e,\n"
            "  \"max_abs_delta_Y_to_xB\": %.17e,\n"
            "  \"max_abs_delta_clipping\": %.17e,\n"
            "  \"total_clip_count_phi\": %.17e,\n"
            "  \"total_clip_count_xB\": %.17e,\n"
            "  \"total_clip_count_Y\": %.17e,\n"
            "  \"summary_matches_csv\": %.17e,\n"
            "  \"max_summary_csv_abs_mismatch\": %.17e,\n"
            "  \"max_abs_mass_closure_resid\": %.17e,\n"
            "  \"suspected_primary_source\": \"%s\"\n"
            "}\n",
            s->initial_mean_xBtot,
            s->final_mean_xBtot,
            s->total_absolute_drift,
            s->total_relative_drift,
            s->cumulative_delta_phi_update,
            s->cumulative_delta_Y_update,
            s->cumulative_delta_Y_to_xB,
            s->cumulative_delta_clipping,
            s->cumulative_delta_other,
            s->max_abs_step_drift,
            s->max_abs_delta_phi_update,
            s->max_abs_delta_Y_update,
            s->max_abs_delta_Y_to_xB,
            s->max_abs_delta_clipping,
            s->total_clip_count_phi,
            s->total_clip_count_xB,
            s->total_clip_count_Y,
            s->summary_matches_csv,
            s->max_summary_csv_abs_mismatch,
            s->max_abs_mass_closure_resid,
            s->suspected_primary_source ? s->suspected_primary_source : "mixed_or_unclear");
    fclose(fp);
}

static int compare_double_asc(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da > db) - (da < db);
}

static void write_performance_summary_json(const char *path, const PerformanceSummary *s) {
    if (!path || !s) return;
    FILE *fp = fopen(path, "w");
    if (!fp) return;
    fprintf(fp,
            "{\n"
            "  \"case_name\": \"%s\",\n"
            "  \"picard_enabled\": %.17e,\n"
            "  \"picard_iters\": %.17e,\n"
            "  \"picard_omega\": %.17e,\n"
            "  \"nsteps\": %.17e,\n"
            "  \"dt\": %.17e,\n"
            "  \"total_walltime_s\": %.17e,\n"
            "  \"avg_walltime_per_step_s\": %.17e,\n"
            "  \"median_walltime_per_step_s\": %.17e,\n"
            "  \"min_walltime_per_step_s\": %.17e,\n"
            "  \"max_walltime_per_step_s\": %.17e,\n"
            "  \"warmup_excluded_avg_walltime_per_step_s\": %.17e,\n"
            "  \"estimated_steps_per_hour\": %.17e,\n"
            "  \"relative_slowdown_vs_baseline\": %.17e\n"
            "}\n",
            s->case_name,
            s->picard_enabled,
            s->picard_iters,
            s->picard_omega,
            s->nsteps,
            s->dt,
            s->total_walltime_s,
            s->avg_walltime_per_step_s,
            s->median_walltime_per_step_s,
            s->min_walltime_per_step_s,
            s->max_walltime_per_step_s,
            s->warmup_excluded_avg_walltime_per_step_s,
            s->estimated_steps_per_hour,
            s->relative_slowdown_vs_baseline);
    fclose(fp);
}

static void write_gp_to_beta_conversion_audit_fields(const PFParams *P,
                                                     const char *case_output_dir,
                                                     double *d_phi_r,
                                                     double *d_eta_r,
                                                     double *d_xB_r,
                                                     double *d_scratch_r_double,
                                                     int total_r) {
    if (!P || !case_output_dir || !case_output_dir[0] || !d_phi_r || !d_eta_r || !d_xB_r || !d_scratch_r_double) return;
    char path[4096];
    snprintf(path, sizeof(path), "%s/phi_after_conversion.vtk", case_output_dir);
    write_vtk_cuda(d_phi_r, P->Nx, P->Ny, P->Nz, "phi_after_conversion", 0, path);
    snprintf(path, sizeof(path), "%s/eta_after_conversion.vtk", case_output_dir);
    write_vtk_cuda(d_eta_r, P->Nx, P->Ny, P->Nz, "eta_after_conversion", 0, path);
    snprintf(path, sizeof(path), "%s/xB_after_conversion.vtk", case_output_dir);
    write_vtk_cuda(d_xB_r, P->Nx, P->Ny, P->Nz, "xB_after_conversion", 0, path);
    if (is_gp_zone_mode(P)) {
        launch_compute_xBtot_gp_kernel(d_phi_r, d_eta_r, d_xB_r, d_scratch_r_double, P->gp_xB_fixed, total_r);
    } else {
        launch_compute_xBtot_kernel(d_phi_r, d_xB_r, d_scratch_r_double, P->v_B, total_r);
    }
    snprintf(path, sizeof(path), "%s/xBtot_gp_after_conversion.vtk", case_output_dir);
    write_vtk_cuda(d_scratch_r_double, P->Nx, P->Ny, P->Nz, "xBtot_gp_after_conversion", 0, path);
}

static void capture_post_conversion_device_state(const PFParams *P,
                                                 double *d_phi_r,
                                                 double *d_eta_r,
                                                 double *d_Y_r,
                                                 double *d_xB_r,
                                                 cufftHandle plan_r2c_Y,
                                                 cufftDoubleComplex *d_Y_k,
                                                 int total_r,
                                                 int total_k,
                                                 double *sum_xBtot,
                                                 double *mean_xB,
                                                 double *min_xB,
                                                 double *max_xB,
                                                 double *mean_Y,
                                                 double *Y_k0_re,
                                                 double *Y_k0_im) {
    if (sum_xBtot) *sum_xBtot = gpu_reduce_sum_model_xBtot(P, d_phi_r, d_eta_r, d_xB_r, total_r);
    if (mean_xB) *mean_xB = gpu_reduce_sum(d_xB_r, total_r) / (double)total_r;
    if (mean_Y) *mean_Y = gpu_reduce_sum(d_Y_r, total_r) / (double)total_r;
    if (min_xB && max_xB) {
        gpu_reduce_min_max(d_xB_r, total_r, min_xB, max_xB);
    }
    if (Y_k0_re || Y_k0_im) {
        CUFFT_CHECK(cufftExecD2Z(plan_r2c_Y, d_Y_r, d_Y_k));
        launch_dealias_kernel(d_Y_k, P->Nx, P->Ny, P->Nz, P->Nz / 2 + 1,
                              P->dx, P->dy, P->dz, total_k);
        cuDoubleComplex Y_k0 = make_cuDoubleComplex(0.0, 0.0);
        CUDA_CHECK(cudaMemcpy(&Y_k0, d_Y_k, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        if (Y_k0_re) *Y_k0_re = cuCreal(Y_k0);
        if (Y_k0_im) *Y_k0_im = cuCimag(Y_k0);
    }
}

static void write_post_conversion_y_audit_row(FILE *fp,
                                              int step,
                                              const char *stage,
                                              double sum_xBtot,
                                              double prev_sum_xBtot,
                                              double baseline_sum_xBtot,
                                              double mean_xBtot,
                                              double mean_xB,
                                              double min_xB,
                                              double max_xB,
                                              double mean_Y,
                                              double Y_k0_before,
                                              double Y_k0_after,
                                              double delta_Y_k0,
                                              double Y_update_k0_drift,
                                              double Y_update_nonzero_mode_drift,
                                              double clipped_mass_loss,
                                              double num_clipped_low,
                                              double num_clipped_high,
                                              double storage_residual_sum,
                                              double storage_residual_max_abs) {
    if (!fp || !stage) return;
    fprintf(fp,
            "%d,%s,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n",
            step, stage,
            sum_xBtot,
            sum_xBtot - prev_sum_xBtot,
            sum_xBtot - baseline_sum_xBtot,
            mean_xBtot,
            mean_xB,
            min_xB,
            max_xB,
            mean_Y,
            Y_k0_before,
            Y_k0_after,
            delta_Y_k0,
            Y_update_k0_drift,
            Y_update_nonzero_mode_drift,
            clipped_mass_loss,
            num_clipped_low,
            num_clipped_high,
            storage_residual_sum,
            storage_residual_max_abs);
}

typedef struct {
    FILE *csv;
    int active;
    int projection_armed;
    int trigger_step;
    int rows_remaining_steps;
    int post_conversion_step_index;
    double post_conversion_baseline_sum_xBtot;
    double pre_Y_sum_xBtot;
    double pre_Y_mean_xB;
    double pre_Y_min_xB;
    double pre_Y_max_xB;
    double pre_Y_mean_Y;
    double pre_Y_Y_k0_re;
    double pre_Y_Y_k0_im;
    char prefix[128];
} YUpdateK0AuditRuntime;

static void write_y_update_k0_audit_row(
    FILE *fp,
    int step,
    int audit_step_index,
    const char *stage,
    int post_conversion_step,
    double sum_xBtot_before_Y,
    double sum_xBtot_after_Y,
    double delta_sum_xBtot_Y_update,
    double target_sum_xBtot,
    double Y_k0_before,
    double Y_k0_after,
    double delta_Y_k0,
    double RHS_k0_total,
    double RHS_k0_linear,
    double RHS_k0_nonlinear,
    double RHS_k0_stabilization_add,
    double RHS_k0_stabilization_subtract,
    double RHS_k0_source,
    double RHS_k0_transport,
    double RHS_k0_unknown,
    double mean_Y_before,
    double mean_Y_after,
    double mean_xB_before,
    double mean_xB_after,
    double min_xB_before,
    double max_xB_before,
    double min_xB_after,
    double max_xB_after,
    double mean_phi,
    double mean_eta,
    double mean_h_gp,
    double mean_h_beta,
    double storage_residual_sum_before,
    double storage_residual_sum_after,
    double storage_residual_max_abs_before,
    double storage_residual_max_abs_after,
    const char *notes)
{
    if (!fp || !stage) return;
    fprintf(fp,
            "%d,%d,%s,%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%s\n",
            step, audit_step_index, stage, post_conversion_step,
            sum_xBtot_before_Y, sum_xBtot_after_Y, delta_sum_xBtot_Y_update, target_sum_xBtot,
            Y_k0_before, Y_k0_after, delta_Y_k0,
            RHS_k0_total, RHS_k0_linear, RHS_k0_nonlinear,
            RHS_k0_stabilization_add, RHS_k0_stabilization_subtract,
            RHS_k0_source, RHS_k0_transport, RHS_k0_unknown,
            mean_Y_before, mean_Y_after,
            mean_xB_before, mean_xB_after,
            min_xB_before, max_xB_before,
            min_xB_after, max_xB_after,
            mean_phi, mean_eta, mean_h_gp, mean_h_beta,
            storage_residual_sum_before, storage_residual_sum_after,
            storage_residual_max_abs_before, storage_residual_max_abs_after,
            (notes && notes[0] != '\0') ? notes : "");
}

static void write_y_update_mass_projection_row(
    FILE *fp,
    int step,
    int post_conversion_step,
    const char *target_mode,
    double target_sum_xBtot,
    double sum_xBtot_before_projection,
    double sum_xBtot_after_projection,
    double delta_before_projection,
    double delta_after_projection,
    double lambda_shift,
    int num_iter,
    int converged,
    double projection_residual,
    double min_xB_before_projection,
    double max_xB_before_projection,
    double min_xB_after_projection,
    double max_xB_after_projection,
    double num_clipped_low_after_projection,
    double num_clipped_high_after_projection,
    const char *notes)
{
    if (!fp) return;
    fprintf(fp,
            "%d,%d,%s,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%d,%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%s\n",
            step, post_conversion_step,
            target_mode ? target_mode : "",
            target_sum_xBtot,
            sum_xBtot_before_projection,
            sum_xBtot_after_projection,
            delta_before_projection,
            delta_after_projection,
            lambda_shift,
            num_iter,
            converged,
            projection_residual,
            min_xB_before_projection,
            max_xB_before_projection,
            min_xB_after_projection,
            max_xB_after_projection,
            num_clipped_low_after_projection,
            num_clipped_high_after_projection,
            (notes && notes[0] != '\0') ? notes : "");
}

static int run_y_update_mass_projection(const PFParams *P,
                                        int step,
                                        int post_conversion_step,
                                        const char *target_mode,
                                        double target_sum_xBtot,
                                        double *d_phi_r,
                                        double *d_eta_r,
                                        double *d_Y_base_r,
                                        double *d_Y_r,
                                        double *d_xB_r,
                                        int total_r,
                                        FILE *projection_fp,
                                        double *lambda_shift_out,
                                        double *sum_before_out,
                                        double *sum_after_out,
                                        double *min_xB_before_out,
                                        double *max_xB_before_out,
                                        double *min_xB_after_out,
                                        double *max_xB_after_out,
                                        int *converged_out) {
    if (!P || !d_phi_r || !d_eta_r || !d_Y_base_r || !d_Y_r || !d_xB_r) return 0;
    const double tol_abs = P->y_update_mass_projection_tol * fmax(1.0, fabs(target_sum_xBtot));
    double sum_before = gpu_reduce_sum_model_xBtot(P, d_phi_r, d_eta_r, d_xB_r, total_r);
    double min_xB_before = 0.0, max_xB_before = 0.0;
    gpu_reduce_min_max(d_xB_r, total_r, &min_xB_before, &max_xB_before);
    CUDA_CHECK(cudaMemcpy(d_Y_base_r, d_Y_r, (size_t)total_r * sizeof(double), cudaMemcpyDeviceToDevice));

    double lambda_best = 0.0;
    double sum_best = sum_before;
    int converged = (fabs(sum_before - target_sum_xBtot) <= tol_abs) ? 1 : 0;
    int num_iter = 0;
    const char *notes = converged ? "already_within_tolerance" : "";

    auto eval_lambda = [&](double lambda) -> double {
        launch_apply_Y_shift_recompute_xB_kernel(d_Y_base_r, d_Y_r, d_xB_r, lambda, total_r);
        return gpu_reduce_sum_model_xBtot(P, d_phi_r, d_eta_r, d_xB_r, total_r);
    };

    if (!converged) {
        double lo = -50.0, hi = 50.0;
        double flo = eval_lambda(lo) - target_sum_xBtot;
        double fhi = eval_lambda(hi) - target_sum_xBtot;
        lambda_best = 0.0;
        sum_best = sum_before;
        if (fabs(flo) <= tol_abs) {
            lambda_best = lo;
            sum_best = flo + target_sum_xBtot;
            converged = 1;
        } else if (fabs(fhi) <= tol_abs) {
            lambda_best = hi;
            sum_best = fhi + target_sum_xBtot;
            converged = 1;
        } else {
            if (flo * fhi > 0.0) {
                lo = -100.0;
                hi = 100.0;
                flo = eval_lambda(lo) - target_sum_xBtot;
                fhi = eval_lambda(hi) - target_sum_xBtot;
            }
            if (flo * fhi <= 0.0) {
                for (num_iter = 1; num_iter <= P->y_update_mass_projection_max_iter; ++num_iter) {
                    double mid = 0.5 * (lo + hi);
                    double sum_mid = eval_lambda(mid);
                    double fmid = sum_mid - target_sum_xBtot;
                    lambda_best = mid;
                    sum_best = sum_mid;
                    if (fabs(fmid) <= tol_abs) {
                        converged = 1;
                        break;
                    }
                    if (flo * fmid <= 0.0) {
                        hi = mid;
                        fhi = fmid;
                    } else {
                        lo = mid;
                        flo = fmid;
                    }
                }
                if (!converged) {
                    notes = "max_iter_reached";
                }
            } else {
                notes = "projection_bracket_failed";
                launch_apply_Y_shift_recompute_xB_kernel(d_Y_base_r, d_Y_r, d_xB_r, 0.0, total_r);
                lambda_best = 0.0;
                sum_best = sum_before;
            }
        }
    }

    if (!converged && strcmp(notes, "projection_bracket_failed") != 0) {
        launch_apply_Y_shift_recompute_xB_kernel(d_Y_base_r, d_Y_r, d_xB_r, lambda_best, total_r);
    }

    double sum_after = gpu_reduce_sum_model_xBtot(P, d_phi_r, d_eta_r, d_xB_r, total_r);
    double min_xB_after = 0.0, max_xB_after = 0.0;
    gpu_reduce_min_max(d_xB_r, total_r, &min_xB_after, &max_xB_after);
    if (projection_fp) {
        write_y_update_mass_projection_row(
            projection_fp, step, post_conversion_step, target_mode,
            target_sum_xBtot, sum_before, sum_after,
            sum_before - target_sum_xBtot,
            sum_after - target_sum_xBtot,
            lambda_best, num_iter, converged,
            sum_after - target_sum_xBtot,
            min_xB_before, max_xB_before,
            min_xB_after, max_xB_after,
            0.0, 0.0, notes);
        fflush(projection_fp);
    }
    if (lambda_shift_out) *lambda_shift_out = lambda_best;
    if (sum_before_out) *sum_before_out = sum_before;
    if (sum_after_out) *sum_after_out = sum_after;
    if (min_xB_before_out) *min_xB_before_out = min_xB_before;
    if (max_xB_before_out) *max_xB_before_out = max_xB_before;
    if (min_xB_after_out) *min_xB_after_out = min_xB_after;
    if (max_xB_after_out) *max_xB_after_out = max_xB_after;
    if (converged_out) *converged_out = converged;
    return 1;
}

typedef struct {
    int step;
    double cx, cy, cz;
    int fired;
} ScheduledNucEvent;

typedef struct {
    char label[32];
    std::vector<double> d_nm;
    std::vector<double> phi;
    std::vector<double> xB;
} ScheduledNucFamilyProfile;

typedef struct {
    std::vector<ScheduledNucFamilyProfile> families;
    ScheduledNucFamilyProfile avg_family;
    double semiaxes_nm[3];
    double xB_matrix_reference;
    double xB_matrix_near;
    int using_average_family;
    int loaded;
} ScheduledNucProfileSet;

typedef struct {
    std::vector<ScheduledNucEvent> events;
    ScheduledNucProfileSet profile;
    FILE *events_csv;
    int event_counter;
    std::vector<ScheduledNucEvent> fired_events;
} ScheduledNucRuntime;

typedef struct {
    int step;
    int i;
    int j;
    int k;
} GpToBetaAcceptedEvent;

typedef struct {
    FILE *events_csv;
    FILE *audit_csv;
    int event_counter;
    char audit_prefix[128];
    int count_events_rejected_capacity;
    int count_events_rejected_cooldown;
    int count_events_rejected_spacing;
    int count_events_rejected_window;
    int count_events_rejected_global;
    int count_events_scaled_capacity;
    int count_events_accepted;
    double min_capacity_ratio;
    double sum_capacity_ratio;
    int capacity_ratio_count;
    int last_accepted_step;
    int last_accepted_i;
    int last_accepted_j;
    int last_accepted_k;
    std::vector<GpToBetaAcceptedEvent> accepted_events;
} GpToBetaRuntime;

typedef struct {
    FILE *csv;
    int active;
    int trigger_step;
    int rows_remaining_steps;
    double baseline_sum_xBtot;
    char prefix[128];
    double before_step_sum_xBtot;
    double before_step_mean_xB;
    double before_step_min_xB;
    double before_step_max_xB;
    double before_step_mean_Y;
    double before_step_Y_k0_re;
    double before_step_Y_k0_im;
    double pre_Y_sum_xBtot;
    double pre_Y_mean_xB;
    double pre_Y_min_xB;
    double pre_Y_max_xB;
    double pre_Y_mean_Y;
    double pre_Y_Y_k0_re;
    double pre_Y_Y_k0_im;
} PostConversionYAuditRuntime;

typedef struct {
    FILE *events_csv;
    int event_counter;
} GpNucRuntime;

static std::string trim_copy_cpp(const std::string &s) {
    size_t a = 0;
    while (a < s.size() && isspace((unsigned char)s[a])) ++a;
    size_t b = s.size();
    while (b > a && isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

static std::string lower_copy_cpp(std::string s) {
    for (size_t i = 0; i < s.size(); ++i) s[i] = (char)tolower((unsigned char)s[i]);
    return s;
}

static std::vector<std::string> split_csv_simple_cpp(const std::string &line) {
    std::vector<std::string> out;
    std::string cur;
    int in_quote = 0;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            in_quote = !in_quote;
        } else if (c == ',' && !in_quote) {
            out.push_back(trim_copy_cpp(cur));
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(trim_copy_cpp(cur));
    return out;
}

static int find_header_col(const std::vector<std::string> &headers, const char **names, int n_names) {
    for (size_t i = 0; i < headers.size(); ++i) {
        std::string h = lower_copy_cpp(trim_copy_cpp(headers[i]));
        for (int j = 0; j < n_names; ++j) {
            if (h == names[j]) return (int)i;
        }
    }
    return -1;
}

static int parse_double_list3(const char *text, double *a, double *b, double *c) {
    if (!text || !a || !b || !c) return 0;
    char *endp = NULL;
    *a = strtod(text, &endp);
    if (endp == text) return 0;
    while (*endp == ',' || isspace((unsigned char)*endp)) ++endp;
    char *p2 = endp;
    *b = strtod(p2, &endp);
    if (endp == p2) return 0;
    while (*endp == ',' || isspace((unsigned char)*endp)) ++endp;
    char *p3 = endp;
    *c = strtod(p3, &endp);
    return endp != p3;
}

static int parse_scheduled_events(const PFParams *P, std::vector<ScheduledNucEvent> *events) {
    if (!P || !events) return 0;
    events->clear();
    if (P->scheduled_nuc_steps_csv[0] == '\0' || P->scheduled_nuc_centers_nm[0] == '\0') {
        fprintf(stderr, "[fatal] scheduled nucleation requires --scheduled-nuc-steps and --scheduled-nuc-centers-nm\n");
        return 0;
    }

    std::vector<int> steps;
    {
        char buf[1024];
        snprintf(buf, sizeof(buf), "%s", P->scheduled_nuc_steps_csv);
        char *save = NULL;
        char *tok = strtok_r(buf, ",", &save);
        while (tok) {
            steps.push_back(atoi(tok));
            tok = strtok_r(NULL, ",", &save);
        }
    }

    std::vector<ScheduledNucEvent> tmp;
    {
        char buf[2048];
        snprintf(buf, sizeof(buf), "%s", P->scheduled_nuc_centers_nm);
        char *save = NULL;
        char *tok = strtok_r(buf, ";", &save);
        while (tok) {
            ScheduledNucEvent ev;
            memset(&ev, 0, sizeof(ev));
            if (!parse_double_list3(tok, &ev.cx, &ev.cy, &ev.cz)) {
                fprintf(stderr, "[fatal] cannot parse scheduled nucleus center triple: '%s'\n", tok);
                return 0;
            }
            tmp.push_back(ev);
            tok = strtok_r(NULL, ";", &save);
        }
    }

    if (steps.empty() || steps.size() != tmp.size()) {
        fprintf(stderr, "[fatal] scheduled nucleation steps/centers count mismatch: steps=%zu centers=%zu\n",
                steps.size(), tmp.size());
        return 0;
    }
    for (size_t i = 0; i < tmp.size(); ++i) {
        tmp[i].step = steps[i];
        tmp[i].fired = 0;
        events->push_back(tmp[i]);
    }
    std::sort(events->begin(), events->end(), [](const ScheduledNucEvent &a, const ScheduledNucEvent &b) {
        return a.step < b.step;
    });
    return 1;
}

static void sort_profile_family(ScheduledNucFamilyProfile *fam) {
    if (!fam || fam->d_nm.size() <= 1) return;
    std::vector<size_t> idx(fam->d_nm.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return fam->d_nm[a] < fam->d_nm[b]; });
    std::vector<double> d, p, x;
    d.reserve(idx.size()); p.reserve(idx.size()); x.reserve(idx.size());
    for (size_t k = 0; k < idx.size(); ++k) {
        size_t i = idx[k];
        d.push_back(fam->d_nm[i]);
        p.push_back(fam->phi[i]);
        x.push_back(fam->xB[i]);
    }
    fam->d_nm.swap(d);
    fam->phi.swap(p);
    fam->xB.swap(x);
}

static double interp_profile_clamped(const std::vector<double> &x, const std::vector<double> &y,
                                     double q, int *out_of_range) {
    if (x.empty() || y.empty()) return 0.0;
    if (q <= x.front()) {
        if (out_of_range && q < x.front()) ++(*out_of_range);
        return y.front();
    }
    if (q >= x.back()) {
        if (out_of_range && q > x.back()) ++(*out_of_range);
        return y.back();
    }
    std::vector<double>::const_iterator it = std::lower_bound(x.begin(), x.end(), q);
    size_t hi = (size_t)(it - x.begin());
    size_t lo = hi - 1;
    double t = (q - x[lo]) / fmax(x[hi] - x[lo], 1.0e-30);
    return y[lo] * (1.0 - t) + y[hi] * t;
}

static int source_summary_semiaxes_nm(const char *source_dyn_dir, double *a, double *b, double *c) {
    if (!source_dyn_dir || source_dyn_dir[0] == '\0' || !a || !b || !c) return 0;
    char path[4096];
    snprintf(path, sizeof(path), "%s/summary.txt", source_dyn_dir);
    FILE *fp = fopen(path, "r");
    if (!fp) return 0;
    char line[1024];
    double L1 = NAN, L2 = NAN, L3 = NAN;
    while (fgets(line, sizeof(line), fp)) {
        char *colon = strchr(line, ':');
        if (!colon) continue;
        *colon = '\0';
        char *val = colon + 1;
        double x = strtod(val, NULL);
        if (strstr(line, "L1_long")) L1 = x;
        else if (strstr(line, "L2_mid")) L2 = x;
        else if (strstr(line, "L3_short")) L3 = x;
    }
    fclose(fp);
    if (isfinite(L1) && isfinite(L2) && isfinite(L3) && L1 > 0.0 && L2 > 0.0 && L3 > 0.0) {
        *a = 0.5 * L1;
        *b = 0.5 * L2;
        *c = 0.5 * L3;
        return 1;
    }
    return 0;
}

static int load_scheduled_profile_csv(const PFParams *P, ScheduledNucProfileSet *profile) {
    if (!P || !profile) return 0;
    profile->families.clear();
    profile->avg_family = ScheduledNucFamilyProfile();
    profile->semiaxes_nm[0] = profile->semiaxes_nm[1] = profile->semiaxes_nm[2] = 5.0;
    profile->xB_matrix_reference = 0.03;
    profile->xB_matrix_near = 0.03;
    profile->using_average_family = 0;
    profile->loaded = 0;
    if (P->scheduled_nuc_profile_dir[0] == '\0') {
        fprintf(stderr, "[fatal] scheduled nucleation requires --scheduled-nuc-profile-dir; analytic sphere is not default.\n");
        return 0;
    }
    char csv_path[4096];
    snprintf(csv_path, sizeof(csv_path), "%s/faceted_family_profiles.csv", P->scheduled_nuc_profile_dir);
    FILE *fp = fopen(csv_path, "r");
    if (!fp) {
        fprintf(stderr, "[fatal] cannot open scheduled nucleus faceted profile CSV: %s\n", csv_path);
        fprintf(stderr, "        Run tools/analysis/extract_faceted_rebuild_profiles.py for the no-strain dynamic-continue source first.\n");
        return 0;
    }

    char line[8192];
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        fprintf(stderr, "[fatal] empty profile CSV: %s\n", csv_path);
        return 0;
    }
    std::vector<std::string> headers = split_csv_simple_cpp(line);
    const char *family_names[] = {"family", "face_family"};
    const char *region_names[] = {"region"};
    const char *d_names[] = {"u_nm", "d_nm", "distance_nm"};
    const char *phi_names[] = {"phi_mean", "phi"};
    const char *xb_names[] = {"xb_mean", "xB_mean", "xb", "xB"};
    int col_family = find_header_col(headers, family_names, 2);
    int col_region = find_header_col(headers, region_names, 1);
    int col_d = find_header_col(headers, d_names, 3);
    int col_phi = find_header_col(headers, phi_names, 2);
    int col_xb = find_header_col(headers, xb_names, 4);
    if (col_family < 0 || col_d < 0 || col_phi < 0 || col_xb < 0) {
        fclose(fp);
        fprintf(stderr, "[fatal] profile CSV missing required columns. Need family,u_nm/d_nm,phi_mean,xB_mean: %s\n", csv_path);
        return 0;
    }

    std::map<std::string, int> fam_index;
    while (fgets(line, sizeof(line), fp)) {
        std::vector<std::string> cols = split_csv_simple_cpp(line);
        int need = std::max(std::max(col_family, col_d), std::max(col_phi, col_xb));
        if ((int)cols.size() <= need) continue;
        if (col_region >= 0 && (int)cols.size() > col_region) {
            std::string region = lower_copy_cpp(cols[col_region]);
            if (region != "face") continue;
        }
        std::string label = cols[col_family];
        double d = atof(cols[col_d].c_str());
        double phi = atof(cols[col_phi].c_str());
        double xb = atof(cols[col_xb].c_str());
        if (!isfinite(d) || !isfinite(phi) || !isfinite(xb)) continue;
        int idx = -1;
        if (fam_index.find(label) == fam_index.end()) {
            idx = (int)profile->families.size();
            fam_index[label] = idx;
            ScheduledNucFamilyProfile fam;
            memset(fam.label, 0, sizeof(fam.label));
            snprintf(fam.label, sizeof(fam.label), "%s", label.c_str());
            profile->families.push_back(fam);
        } else {
            idx = fam_index[label];
        }
        profile->families[idx].d_nm.push_back(d);
        profile->families[idx].phi.push_back(phi);
        profile->families[idx].xB.push_back(xb);
    }
    fclose(fp);

    if (profile->families.empty()) {
        fprintf(stderr, "[fatal] no usable face profiles loaded from %s\n", csv_path);
        return 0;
    }
    for (size_t i = 0; i < profile->families.size(); ++i) {
        sort_profile_family(&profile->families[i]);
    }

    profile->avg_family = profile->families[0];
    snprintf(profile->avg_family.label, sizeof(profile->avg_family.label), "average");
    for (size_t k = 0; k < profile->avg_family.d_nm.size(); ++k) {
        double ps = 0.0, xs = 0.0;
        int cnt = 0;
        double d = profile->avg_family.d_nm[k];
        for (size_t f = 0; f < profile->families.size(); ++f) {
            ps += interp_profile_clamped(profile->families[f].d_nm, profile->families[f].phi, d, NULL);
            xs += interp_profile_clamped(profile->families[f].d_nm, profile->families[f].xB, d, NULL);
            cnt++;
        }
        profile->avg_family.phi[k] = ps / fmax((double)cnt, 1.0);
        profile->avg_family.xB[k] = xs / fmax((double)cnt, 1.0);
    }

    if (!source_summary_semiaxes_nm(P->scheduled_nuc_source_dyn_dir,
                                    &profile->semiaxes_nm[0],
                                    &profile->semiaxes_nm[1],
                                    &profile->semiaxes_nm[2])) {
        profile->semiaxes_nm[0] = 5.0;
        profile->semiaxes_nm[1] = 5.0;
        profile->semiaxes_nm[2] = 5.0;
        fprintf(stderr, "[warn] scheduled nucleation could not read semiaxes from source summary; using 5 nm spherical fallback geometry.\n");
    }

    double xb_ref_sum = 0.0;
    int xb_ref_count = 0;
    for (size_t f = 0; f < profile->families.size(); ++f) {
        if (!profile->families[f].xB.empty()) {
            xb_ref_sum += profile->families[f].xB.back();
            xb_ref_count++;
        }
    }
    profile->xB_matrix_reference = xb_ref_sum / fmax((double)xb_ref_count, 1.0);
    profile->xB_matrix_near = profile->xB_matrix_reference;
    profile->using_average_family = 0;
    profile->loaded = 1;

    printf("[SCHEDULED NUC TEST] loaded faceted profile CSV: %s families=%zu semiaxes_nm=(%.4f, %.4f, %.4f) xB_ref=%.8e\n",
           csv_path, profile->families.size(),
           profile->semiaxes_nm[0], profile->semiaxes_nm[1], profile->semiaxes_nm[2],
           profile->xB_matrix_reference);
    return 1;
}

static int family_index_from_octant_label(const ScheduledNucProfileSet *profile, int sx, int sy, int sz) {
    char label[32];
    snprintf(label, sizeof(label), "%+d%+d%+d", sx, sy, sz);
    for (size_t i = 0; i < profile->families.size(); ++i) {
        if (strcmp(profile->families[i].label, label) == 0) return (int)i;
    }
    return -1;
}

static const ScheduledNucFamilyProfile *select_profile_family(const ScheduledNucProfileSet *profile,
                                                              double qx, double qy, double qz,
                                                              int *used_average) {
    int sx = (qx >= 0.0) ? 1 : -1;
    int sy = (qy >= 0.0) ? 1 : -1;
    int sz = (qz >= 0.0) ? 1 : -1;
    int idx = family_index_from_octant_label(profile, sx, sy, sz);
    if (idx >= 0) {
        if (used_average) *used_average = 0;
        return &profile->families[(size_t)idx];
    }
    if (used_average) *used_average = 1;
    return &profile->avg_family;
}

static double smoothstep01_scalar(double s) {
    if (s <= 0.0) return 0.0;
    if (s >= 1.0) return 1.0;
    return s * s * (3.0 - 2.0 * s);
}

static double w_box_for_event(const ScheduledNucEvent *ev, double x, double y, double z,
                              double half_box_nm, double blend_margin_nm) {
    double qx = x - ev->cx;
    double qy = y - ev->cy;
    double qz = z - ev->cz;
    double m = fmin(half_box_nm - fabs(qx), fmin(half_box_nm - fabs(qy), half_box_nm - fabs(qz)));
    if (m <= 0.0) return 0.0;
    if (m >= blend_margin_nm) return 1.0;
    return smoothstep01_scalar(m / fmax(blend_margin_nm, 1.0e-30));
}

static double outside_distance_to_source_box(const ScheduledNucEvent *ev, double x, double y, double z,
                                             double half_box_nm) {
    double ax = fmax(fabs(x - ev->cx) - half_box_nm, 0.0);
    double ay = fmax(fabs(y - ev->cy) - half_box_nm, 0.0);
    double az = fmax(fabs(z - ev->cz) - half_box_nm, 0.0);
    return sqrt(ax * ax + ay * ay + az * az);
}

static double compute_mean_xBtot_host(const std::vector<double> &phi, const std::vector<double> &xB,
                                      double vB_frac) {
    long double sum = 0.0;
    const size_t n = phi.size();
    for (size_t i = 0; i < n; ++i) {
        double h = h_of_phi(clamp01(phi[i]));
        sum += (1.0 - h) * xB[i] + vB_frac * h;
    }
    return (double)(sum / (long double)n);
}

static void host_minmax_mean_phi_xB(const std::vector<double> &phi, const std::vector<double> &xB,
                                    double *phi_min, double *phi_max, double *phi_mean,
                                    double *xb_min, double *xb_max, double *xb_mean,
                                    double *mean_h) {
    double pmin = 1.0e300, pmax = -1.0e300, xmin = 1.0e300, xmax = -1.0e300;
    long double ps = 0.0, xs = 0.0, hs = 0.0;
    for (size_t i = 0; i < phi.size(); ++i) {
        double p = clamp01(phi[i]);
        double x = xB[i];
        pmin = fmin(pmin, p);
        pmax = fmax(pmax, p);
        xmin = fmin(xmin, x);
        xmax = fmax(xmax, x);
        ps += p;
        xs += x;
        hs += h_of_phi(p);
    }
    double inv = 1.0 / fmax((double)phi.size(), 1.0);
    if (phi_min) *phi_min = pmin;
    if (phi_max) *phi_max = pmax;
    if (phi_mean) *phi_mean = (double)ps * inv;
    if (xb_min) *xb_min = xmin;
    if (xb_max) *xb_max = xmax;
    if (xb_mean) *xb_mean = (double)xs * inv;
    if (mean_h) *mean_h = (double)hs * inv;
}

static int write_host_vtk_scalar(const char *path, const std::vector<double> &field,
                                 int Nx, int Ny, int Nz, double dx_nm, const char *name) {
    FILE *fp = fopen(path, "w");
    if (!fp) return 0;
    fprintf(fp, "# vtk DataFile Version 3.0\n%s\nASCII\nDATASET STRUCTURED_POINTS\n", name);
    fprintf(fp, "DIMENSIONS %d %d %d\n", Nx, Ny, Nz);
    fprintf(fp, "ORIGIN 0 0 0\n");
    fprintf(fp, "SPACING %.12g %.12g %.12g\n", dx_nm, dx_nm, dx_nm);
    fprintf(fp, "POINT_DATA %zu\n", field.size());
    fprintf(fp, "SCALARS %s double 1\nLOOKUP_TABLE default\n", name);
    for (size_t i = 0; i < field.size(); ++i) {
        fprintf(fp, "%.10e\n", field[i]);
    }
    fclose(fp);
    return 1;
}

static int apply_scheduled_events_cpu(ScheduledNucRuntime *rt, PFParams *P, int step,
                                      double *d_phi_r, double *d_Y_r, double *d_xB_r,
                                      int total_r, size_t size_r,
                                      const char *case_output_dir) {
    if (!rt || !P || !P->scheduled_nuc_enabled || P->mode != 0) return 1;
    std::vector<int> event_indices;
    for (size_t i = 0; i < rt->events.size(); ++i) {
        if (!rt->events[i].fired && rt->events[i].step == step) {
            event_indices.push_back((int)i);
        }
    }
    if (event_indices.empty()) return 1;

    printf("[SCHEDULED NUC TEST] using CPU roundtrip profile insertion at step %d events=%zu\n",
           step, event_indices.size());

    std::vector<double> phi((size_t)total_r), xB((size_t)total_r), Y((size_t)total_r);
    CUDA_CHECK(cudaMemcpy(phi.data(), d_phi_r, size_r, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(xB.data(), d_xB_r, size_r, cudaMemcpyDeviceToHost));

    const double M_before = compute_mean_xBtot_host(phi, xB, P->v_B);
    double phi_min_before, phi_max_before, phi_mean_before, xb_min_before, xb_max_before, xb_mean_before, mean_h_before;
    host_minmax_mean_phi_xB(phi, xB, &phi_min_before, &phi_max_before, &phi_mean_before,
                            &xb_min_before, &xb_max_before, &xb_mean_before, &mean_h_before);

    const double dx_nm = P->dx;
    const double source_box_nm = 400.0 * P->scheduled_nuc_source_dx_nm;
    const double half_box_nm = 0.5 * source_box_nm;
    const double box_blend_nm = fmax(3.0, P->scheduled_nuc_edge_sample_outer_nm);
    const double scale_phi = (P->scheduled_nuc_scale_interface_width > 0.0)
        ? P->scheduled_nuc_scale_interface_width
        : (P->scheduled_nuc_target_lambda_nm / fmax(P->scheduled_nuc_source_lambda_nm, 1.0e-30));
    const double scale_xb = (P->scheduled_nuc_scale_xB_profile_width > 0.0)
        ? P->scheduled_nuc_scale_xB_profile_width
        : scale_phi;
    const double geom_scale = P->scheduled_nuc_scale_geometry;
    const double a = fmax(rt->profile.semiaxes_nm[0] * geom_scale, 1.0e-6);
    const double b = fmax(rt->profile.semiaxes_nm[1] * geom_scale, 1.0e-6);
    const double c = fmax(rt->profile.semiaxes_nm[2] * geom_scale, 1.0e-6);

    std::vector<float> W_new_max((size_t)total_r, 0.0f);
    std::vector<float> local_weight((size_t)total_r, 0.0f);
    long long profile_queries = 0;
    long long profile_oor = 0;
    double xB_edge_used = 0.0;
    double C_local_last = 0.0;
    double clip_lower_frac = 0.0, clip_upper_frac = 0.0;
    int overlap_warning = 0;

    for (int ev_pos = 0; ev_pos < (int)event_indices.size(); ++ev_pos) {
        ScheduledNucEvent &ev = rt->events[(size_t)event_indices[(size_t)ev_pos]];
        if (ev.cx < 0.0 || ev.cy < 0.0 || ev.cz < 0.0 ||
            ev.cx >= P->Nx * dx_nm || ev.cy >= P->Ny * dx_nm || ev.cz >= P->Nz * dx_nm) {
            fprintf(stderr, "[fatal] scheduled nucleus center outside target box: step=%d center=(%.3f,%.3f,%.3f)\n",
                    step, ev.cx, ev.cy, ev.cz);
            return 0;
        }

        std::vector<double> edge_samples;
        edge_samples.reserve(4096);
        for (int i = 0; i < P->Nx; ++i) {
            double xx = (i + 0.5) * dx_nm;
            for (int j = 0; j < P->Ny; ++j) {
                double yy = (j + 0.5) * dx_nm;
                for (int k = 0; k < P->Nz; ++k) {
                    double zz = (k + 0.5) * dx_nm;
                    double d_out = outside_distance_to_source_box(&ev, xx, yy, zz, half_box_nm);
                    if (d_out < P->scheduled_nuc_edge_sample_inner_nm ||
                        d_out > P->scheduled_nuc_edge_sample_outer_nm) continue;
                    size_t idx = ((size_t)i * P->Ny + (size_t)j) * P->Nz + (size_t)k;
                    edge_samples.push_back(xB[idx]);
                }
            }
        }
        if (edge_samples.empty()) {
            fprintf(stderr, "[fatal] scheduled nucleation edge shell sample count is zero at step=%d center=(%.3f,%.3f,%.3f)\n",
                    step, ev.cx, ev.cy, ev.cz);
            return 0;
        }
        double xB_edge = 0.0;
        if (strcmp(P->scheduled_nuc_edge_sample_stat, "median") == 0) {
            std::sort(edge_samples.begin(), edge_samples.end());
            xB_edge = edge_samples[edge_samples.size() / 2];
        } else {
            long double s = 0.0;
            for (double v : edge_samples) s += v;
            xB_edge = (double)(s / (long double)edge_samples.size());
        }
        xB_edge_used = xB_edge;
        printf("[SCHEDULED NUC TEST] event step=%d center=(%.3f,%.3f,%.3f) xB_edge=%.8e samples=%zu\n",
               step, ev.cx, ev.cy, ev.cz, xB_edge, edge_samples.size());

        for (int i = 0; i < P->Nx; ++i) {
            double xx = (i + 0.5) * dx_nm;
            for (int j = 0; j < P->Ny; ++j) {
                double yy = (j + 0.5) * dx_nm;
                for (int k = 0; k < P->Nz; ++k) {
                    double zz = (k + 0.5) * dx_nm;
                    double W_box = w_box_for_event(&ev, xx, yy, zz, half_box_nm, box_blend_nm);
                    if (W_box <= 0.0) continue;
                    size_t idx = ((size_t)i * P->Ny + (size_t)j) * P->Nz + (size_t)k;
                    W_new_max[idx] = (float)fmax((double)W_new_max[idx], W_box);

                    double qx = xx - ev.cx;
                    double qy = yy - ev.cy;
                    double qz = zz - ev.cz;
                    double rho = sqrt((qx / a) * (qx / a) + (qy / b) * (qy / b) + (qz / c) * (qz / c));
                    double rr = sqrt(qx * qx + qy * qy + qz * qz);
                    double boundary_radius = (rho > 1.0e-12) ? (rr / rho) : ((a + b + c) / 3.0);
                    double d_target = (rho - 1.0) * boundary_radius;
                    int used_avg = 0;
                    const ScheduledNucFamilyProfile *fam = select_profile_family(&rt->profile, qx, qy, qz, &used_avg);
                    if (used_avg) rt->profile.using_average_family = 1;
                    int oor_phi = 0, oor_xb = 0;
                    double phi_i = interp_profile_clamped(fam->d_nm, fam->phi, d_target / scale_phi, &oor_phi);
                    double xb_prof = interp_profile_clamped(fam->d_nm, fam->xB, d_target / scale_xb, &oor_xb);
                    profile_queries += 2;
                    profile_oor += (oor_phi + oor_xb);
                    double xb_local = rt->profile.xB_matrix_near +
                                      P->scheduled_nuc_alpha_interface * (xb_prof - rt->profile.xB_matrix_reference);
                    double xb_candidate = xB_edge + W_box * (xb_local - xB_edge);
                    double phi_new = fmax(phi[idx], clamp01(phi_i));
                    if (phi[idx] > P->scheduled_nuc_phi_matrix_threshold && phi_i > P->scheduled_nuc_phi_matrix_threshold) {
                        overlap_warning = 1;
                    }
                    phi[idx] = phi_new;
                    xB[idx] = fmin(xB[idx], xb_candidate);
                }
            }
        }
        ev.fired = 1;
        rt->fired_events.push_back(ev);
    }

    const double M_after_embed = compute_mean_xBtot_host(phi, xB, P->v_B);

    // Build local compensation shell around the newly inserted source boxes only.
    for (int i = 0; i < P->Nx; ++i) {
        double xx = (i + 0.5) * dx_nm;
        for (int j = 0; j < P->Ny; ++j) {
            double yy = (j + 0.5) * dx_nm;
            for (int k = 0; k < P->Nz; ++k) {
                double zz = (k + 0.5) * dx_nm;
                size_t idx = ((size_t)i * P->Ny + (size_t)j) * P->Nz + (size_t)k;
                if (W_new_max[idx] > P->scheduled_nuc_W_comp_threshold) continue;
                if (phi[idx] > P->scheduled_nuc_phi_matrix_threshold) continue;
                double best_w = 0.0;
                for (int ev_pos = 0; ev_pos < (int)event_indices.size(); ++ev_pos) {
                    ScheduledNucEvent &ev = rt->events[(size_t)event_indices[(size_t)ev_pos]];
                    double d_out = outside_distance_to_source_box(&ev, xx, yy, zz, half_box_nm);
                    if (d_out < P->scheduled_nuc_local_comp_inner_nm ||
                        d_out > P->scheduled_nuc_local_comp_outer_nm) continue;
                    double w = 1.0;
                    double taper = P->scheduled_nuc_local_comp_taper_nm;
                    if (taper > 0.0) {
                        double w_in = smoothstep01_scalar((d_out - P->scheduled_nuc_local_comp_inner_nm) / taper);
                        double w_out = smoothstep01_scalar((P->scheduled_nuc_local_comp_outer_nm - d_out) / taper);
                        w = fmin(w_in, w_out);
                    }
                    best_w = fmax(best_w, w);
                }
                local_weight[idx] = (float)best_w;
            }
        }
    }

    double M_now = M_after_embed;
    long long active_w = 0;
    for (size_t i = 0; i < local_weight.size(); ++i) if (local_weight[i] > 1.0e-12f) active_w++;
    for (int iter = 0; iter < P->scheduled_nuc_mass_iters; ++iter) {
        long double denom_sum = 0.0;
        for (size_t idx = 0; idx < phi.size(); ++idx) {
            double h = h_of_phi(clamp01(phi[idx]));
            denom_sum += (1.0 - h) * (double)local_weight[idx];
        }
        double denom = (double)(denom_sum / (long double)phi.size());
        if (denom <= 1.0e-30) {
            fprintf(stderr, "[fatal] scheduled nucleation local compensation mask is empty or inactive.\n");
            return 0;
        }
        double mass_error = M_before - M_now;
        double C_local = mass_error / denom;
        C_local_last = C_local;
        long long clip_lo = 0, clip_hi = 0;
        for (size_t idx = 0; idx < xB.size(); ++idx) {
            if (local_weight[idx] <= 0.0f) continue;
            double xb = xB[idx] + C_local * (double)local_weight[idx];
            if (xb < P->scheduled_nuc_xB_min) { xb = P->scheduled_nuc_xB_min; clip_lo++; }
            if (xb > P->scheduled_nuc_xB_max) { xb = P->scheduled_nuc_xB_max; clip_hi++; }
            xB[idx] = xb;
        }
        clip_lower_frac = (double)clip_lo / fmax((double)xB.size(), 1.0);
        clip_upper_frac = (double)clip_hi / fmax((double)xB.size(), 1.0);
        M_now = compute_mean_xBtot_host(phi, xB, P->v_B);
        if (fabs(M_now - M_before) <= P->scheduled_nuc_mass_tol) break;
    }

    for (size_t idx = 0; idx < xB.size(); ++idx) {
        double xb = xB[idx];
        if (xb < P->scheduled_nuc_xB_min) xb = P->scheduled_nuc_xB_min;
        if (xb > P->scheduled_nuc_xB_max) xb = P->scheduled_nuc_xB_max;
        xB[idx] = xb;
        Y[idx] = logit_from_fraction(xb, P->xB_eps, P->Y_clip);
    }

    double phi_min_after, phi_max_after, phi_mean_after, xb_min_after, xb_max_after, xb_mean_after, mean_h_after;
    host_minmax_mean_phi_xB(phi, xB, &phi_min_after, &phi_max_after, &phi_mean_after,
                            &xb_min_after, &xb_max_after, &xb_mean_after, &mean_h_after);
    const double event_mass_error = M_now - M_before;
    const double rel_event_mass_error = event_mass_error / fmax(fabs(M_before), 1.0e-30);
    const double active_frac = (double)active_w / fmax((double)total_r, 1.0);
    const double oor_frac = (double)profile_oor / fmax((double)profile_queries, 1.0);

    CUDA_CHECK(cudaMemcpy(d_phi_r, phi.data(), size_r, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xB_r, xB.data(), size_r, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y_r, Y.data(), size_r, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    if (rt->events_csv) {
        for (size_t ii = 0; ii < event_indices.size(); ++ii) {
            ScheduledNucEvent &ev = rt->events[(size_t)event_indices[ii]];
            rt->event_counter += 1;
            fprintf(rt->events_csv,
                    "%d,%d,%s,"
                    "%.10e,%.10e,%.10e,%.10e,"
                    "%.10e,%.10e,%.10e,%.10e,%.10e,"
                    "%.10e,%.10e,"
                    "%.10e,%.10e,%.10e,%.10e,"
                    "%.10e,%.10e,"
                    "%.10e,%.10e,%.10e,%d\n",
                    rt->event_counter, step, P->scheduled_nuc_source_case_label,
                    ev.cx, ev.cy, ev.cz, xB_edge_used,
                    M_before, M_after_embed, M_now, event_mass_error, rel_event_mass_error,
                    C_local_last, active_frac,
                    xb_min_before, xb_max_before, xb_min_after, xb_max_after,
                    phi_max_after, mean_h_after,
                    clip_lower_frac, clip_upper_frac, oor_frac, overlap_warning);
        }
        fflush(rt->events_csv);
    }

    printf("[SCHEDULED NUC TEST] step=%d M_before=%.10e M_after_embed=%.10e M_after_comp=%.10e rel_event_err=%.3e C_local=%.3e xB=[%.6e,%.6e] mean_h=%.6e\n",
           step, M_before, M_after_embed, M_now, rel_event_mass_error, C_local_last,
           xb_min_after, xb_max_after, mean_h_after);
    if (fabs(rel_event_mass_error) > 1.0e-5) {
        fprintf(stderr, "[warn] scheduled nucleation event mass error exceeds 1e-5: %.6e\n", rel_event_mass_error);
    } else if (fabs(rel_event_mass_error) > 1.0e-6) {
        fprintf(stderr, "[warn] scheduled nucleation event mass error exceeds 1e-6: %.6e\n", rel_event_mass_error);
    }
    if (clip_lower_frac > 0.0 || clip_upper_frac > 0.0) {
        fprintf(stderr, "[warn] scheduled nucleation clipping occurred: lower=%.3e upper=%.3e\n",
                clip_lower_frac, clip_upper_frac);
    }
    if (rt->profile.using_average_family) {
        fprintf(stderr, "[warn] scheduled nucleation used averaged profile for at least one query; check family labels in profile CSV.\n");
    }

    if (P->scheduled_nuc_write_event_vtk && case_output_dir && case_output_dir[0] != '\0') {
        char path[4096];
        snprintf(path, sizeof(path), "%s/event_step%04d_phi_after_comp.vtk", case_output_dir, step);
        write_host_vtk_scalar(path, phi, P->Nx, P->Ny, P->Nz, dx_nm, "phi");
        snprintf(path, sizeof(path), "%s/event_step%04d_xB_after_comp.vtk", case_output_dir, step);
        write_host_vtk_scalar(path, xB, P->Nx, P->Ny, P->Nz, dx_nm, "xB");
    }
    return 1;
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
    double vf_target = compute_effective_vf_target(P);
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
    if (P->ic_phi_seed_radius > 0.0) {
        log_kv_text("vf_target_source", "seed radius %.6f", P->ic_phi_seed_radius);
    }
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

static void sync_thermo_runtime_flags(const PFParams *P) {
    const int enabled = (P && P->thermo_convex_extrapolation_enabled) ? 1 : 0;
    h_thermo_convex_extrapolation_enabled = enabled;
    CUDA_CHECK(cudaMemcpyToSymbol(d_thermo_convex_extrapolation_enabled, &enabled, sizeof(int)));
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
    P->pf_params_schema_version = 0;
    P->Nx = 400; P->Ny = 400; P->Nz = 400;
    P->dx = P->dy = P->dz = -1.0;
    P->dt = 1.0e-2;
    P->t_real_unit = -1.0;
    P->nsteps = 10;
    P->out_every = 10;
    P->csv_out_every = 1;  // CSV输出间隔，默认为每个时间步
    P->dimension = 3;
    P->seed = 12345;
    snprintf(P->model_mode, sizeof(P->model_mode), "two_phase");
    P->gp_xB_fixed = 0.35;
    P->gp_delta_g0 = 0.0;
    P->gp_delta_g_stab = 0.0;
    P->gp_W_eta = 0.0;
    P->gp_kappa_eta = 0.0;
    P->gp_L_eta = 0.0;
    P->gp_W_eta_phys_input = NAN;
    P->gp_kappa_eta_phys_input = NAN;
    P->gp_L_eta_phys_input = NAN;
    P->gp_W_eta_code_input = NAN;
    P->gp_kappa_eta_code_input = NAN;
    P->gp_L_eta_code_input = NAN;
    P->gp_W_eta_legacy_specified = 0;
    P->gp_kappa_eta_legacy_specified = 0;
    P->gp_L_eta_legacy_specified = 0;
    P->gp_W_eta_phys_specified = 0;
    P->gp_kappa_eta_phys_specified = 0;
    P->gp_L_eta_phys_specified = 0;
    P->gp_W_eta_code_specified = 0;
    P->gp_kappa_eta_code_specified = 0;
    P->gp_L_eta_code_specified = 0;
    P->gp_W_eta_input_mode_resolved = 0;
    P->gp_kappa_eta_input_mode_resolved = 0;
    P->gp_gamma_alpha_gp = -1.0;
    P->gp_l_eta_nm = -1.0;
    P->gp_D_ratio = 1.0;
    P->gp_M_int_eta = -1.0;
    P->gp_eps_iso = 0.0;
    P->gp_M_GP = 0.0;
    P->gp_M_beta = 0.0;
    P->gp_elastic_enabled = 0;
    P->gp_elastic_active_eta = 0;
    P->gp_elastic_active_phi = 0;
    P->gp_elastic_derivative_scale = 1.0;
    P->gp_nuc_enabled = 0;
    P->gp_nuc_check_interval = 10;
    P->gp_nuc_phi_threshold = 0.05;
    P->gp_nuc_eta_threshold = 0.05;
    P->gp_nuc_h_alpha_threshold = 0.95;
    P->gp_nuc_J0 = 1.0e12;
    P->gp_nuc_gamma = 0.1;
    P->gp_nuc_seed_radius = 0.10;
    P->gp_nuc_seed_peak = 0.05;
    P->gp_nuc_seed_iface_width = 0.10;
    P->gp_nuc_patch_radius = 0.35;
    P->gp_nuc_shell_inner_radius = 0.15;
    P->gp_nuc_shell_outer_radius = 0.35;
    P->gp_nuc_max_events_per_step = 1;
    snprintf(P->gp_nuc_mass_mode, sizeof(P->gp_nuc_mass_mode), "local_compensate");
    P->gp_to_beta_enabled = 0;
    P->gp_to_beta_check_interval = 10;
    P->gp_to_beta_eta_threshold = 0.5;
    P->gp_to_beta_radius_threshold = 0.0;
    P->gp_to_beta_xB_threshold = 0.0;
    P->gp_to_beta_seed_radius = 0.15;
    P->gp_to_beta_seed_peak = 1.0;
    P->gp_to_beta_seed_iface_width = 0.10;
    P->gp_to_beta_patch_radius = 0.40;
    P->gp_to_beta_shell_inner_radius = 0.20;
    P->gp_to_beta_shell_outer_radius = 0.40;
    P->gp_to_beta_max_events_per_step = 1;
    P->gp_to_beta_stochastic_enabled = 0;
    P->gp_to_beta_J0_site = 1.0e30;
    P->gp_to_beta_gamma = 0.05;
    P->gp_to_beta_drive_const = 1.0;
    P->gp_to_beta_max_events_per_check = 1;
    snprintf(P->gp_to_beta_barrier_mode, sizeof(P->gp_to_beta_barrier_mode), "cnt_simple");
    snprintf(P->gp_to_beta_drive_mode, sizeof(P->gp_to_beta_drive_mode), "local_simple");
    snprintf(P->gp_to_beta_mass_mode, sizeof(P->gp_to_beta_mass_mode), "report");
    snprintf(P->gp_to_beta_eta_deplete_mode, sizeof(P->gp_to_beta_eta_deplete_mode),
             "multiply_1_minus_hphi_seed");
    snprintf(P->gp_to_beta_phi_insert_mode, sizeof(P->gp_to_beta_phi_insert_mode), "max");
    P->gp_to_beta_conversion_mass_audit_enabled = 0;
    P->gp_to_beta_stop_after_conversion_audit = 0;
    snprintf(P->gp_to_beta_conversion_audit_prefix,
             sizeof(P->gp_to_beta_conversion_audit_prefix),
             "gp_to_beta_conversion_mass_audit");
    P->gp_to_beta_feasibility_gate_enabled = 1;
    P->gp_to_beta_min_shell_capacity_factor = 1.05;
    P->gp_to_beta_reject_if_infeasible = 1;
    P->gp_to_beta_allow_seed_amplitude_scaling = 1;
    P->gp_to_beta_min_seed_amplitude = 0.05;
    P->gp_to_beta_event_cooldown_steps = 0;
    P->gp_to_beta_min_event_spacing = 0.0;
    P->gp_to_beta_event_exclusion_radius = 0.0;
    P->gp_to_beta_max_events_global = 1000000000;
    P->gp_to_beta_max_events_per_window = 1000000000;
    P->gp_to_beta_event_window_steps = 0;
    P->post_conversion_y_update_audit_enabled = 0;
    P->post_conversion_y_update_audit_steps = 5;
    snprintf(P->post_conversion_y_update_audit_prefix,
             sizeof(P->post_conversion_y_update_audit_prefix),
             "post_conversion_y_update_audit");
    P->y_update_k0_audit_enabled = 0;
    P->y_update_k0_audit_steps = 5;
    snprintf(P->y_update_k0_audit_prefix,
             sizeof(P->y_update_k0_audit_prefix),
             "y_update_k0_audit");
    P->y_update_mass_projection_enabled = 0;
    P->y_update_mass_projection_report_enabled = 0;
    P->y_update_mass_projection_max_iter = 30;
    P->y_update_mass_projection_tol = 1.0e-12;
    snprintf(P->y_update_mass_projection_target_mode,
             sizeof(P->y_update_mass_projection_target_mode),
             "pre_Y_update");
    snprintf(P->gp_C_mode, sizeof(P->gp_C_mode), "alpha");
    snprintf(P->gp_eps_mode, sizeof(P->gp_eps_mode), "isotropic");
    snprintf(P->gp_init_mode, sizeof(P->gp_init_mode), "none");
    snprintf(P->gp_init_mass_mode, sizeof(P->gp_init_mass_mode), "report");
    P->gp_eta_seed_radius = 0.0;
    P->gp_eta_seed_peak = 1.0;
    P->gp_eta_seed_center_x = 0.0;
    P->gp_eta_seed_center_y = 0.0;
    P->gp_eta_seed_center_z = 0.0;
    P->gp_eta_iface_width = 0.0;
    P->gp_obs_target_radius_nm = 1.0;
    P->gp_obs_eta_peak = 0.2;
    P->gp_obs_iface_width_nm = 0.2;
    snprintf(P->gp_obs_profile_type, sizeof(P->gp_obs_profile_type), "tanh");
    snprintf(P->gp_obs_match_mode, sizeof(P->gp_obs_match_mode), "match_integral_h_volume");
    snprintf(P->gp_obs_compensation_mode, sizeof(P->gp_obs_compensation_mode), "smooth_radial_depletion");
    P->gp_obs_depletion_radius_factor = 3.0;
    P->gp_obs_depletion_smooth_width_factor = 0.5;
    P->gp_obs_min_xB_alpha = 0.0;
    P->gp_obs_max_xB_alpha = 1.0;
    P->gp_raw_reaction_drive_only = 0;
    P->gp_raw_reaction_drive_use_raw_units_debug = 0;
    P->gp_reaction_nu_A = 0.65;
    P->gp_reaction_nu_B = 0.35;
    P->gp_xB_eq_alpha_for_eta = -1.0;
    strncpy(P->gp_L_eta_mode, "manual", sizeof(P->gp_L_eta_mode) - 1);
    P->gp_L_eta_mode[sizeof(P->gp_L_eta_mode) - 1] = '\0';
    P->gp_M_eta_phys = -1.0;
    P->gp_M_eta_ratio_to_crit = -1.0;
    P->gp_kinetic_ref_enabled = 1;
    P->gp_kinetic_ref_apply = 0;
    P->phi_eta_step_delta_diag_enabled = 0;
    P->phi_eta_step_delta_diag_every = 1;
    P->phi_eta_step_delta_diag_max_steps = 200;
    snprintf(P->phi_eta_step_delta_diag_prefix,
             sizeof(P->phi_eta_step_delta_diag_prefix),
             "phi_eta_step_delta");
    P->phi_eta_rhs_attribution_diag_enabled = 0;
    P->phi_eta_rhs_attribution_diag_every = 1;
    P->phi_eta_rhs_attribution_diag_max_steps = 200;
    snprintf(P->phi_eta_rhs_attribution_diag_prefix,
             sizeof(P->phi_eta_rhs_attribution_diag_prefix),
             "phi_eta_rhs_attribution");
    P->gp_h_alpha_eps = 1.0e-8;
    snprintf(P->gp_eta_mass_limiter, sizeof(P->gp_eta_mass_limiter), "off");
    snprintf(P->gp_y_update_mode, sizeof(P->gp_y_update_mode), "old_rhs");
    P->gp_y_picard_iters = 3;
    
    // 物理输入必须由 --pf-param-file 提供；这里使用哨兵值以防漏传
    P->W = -1.0;
    P->kappa_phi = -1.0;
    P->L_phi = -1.0;
    P->D_alpha = -1.0;
    P->D_compound = -1.0;
    
    P->temperature_C = -1.0;
    P->thermo_convex_extrapolation_enabled = 0;
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
    P->minimize_continue_from_vtk = 0;
    P->continue_phi_vtk_path[0] = '\0';
    P->continue_xB_vtk_path[0] = '\0';
    P->init_mode_raw_fields = 0;
    P->init_phi_raw_path[0] = '\0';
    P->init_xB_raw_path[0] = '\0';
    P->init_eta_raw_path[0] = '\0';
    P->init_meta_path[0] = '\0';

    // Scheduled nucleation test: explicit opt-in only. Defaults are inert.
    P->scheduled_nuc_enabled = 0;
    P->scheduled_nuc_source_dyn_dir[0] = '\0';
    P->scheduled_nuc_profile_dir[0] = '\0';
    snprintf(P->scheduled_nuc_source_step, sizeof(P->scheduled_nuc_source_step), "latest");
    P->scheduled_nuc_source_phi_vtk[0] = '\0';
    P->scheduled_nuc_source_xB_vtk[0] = '\0';
    P->scheduled_nuc_steps_csv[0] = '\0';
    P->scheduled_nuc_centers_nm[0] = '\0';
    snprintf(P->scheduled_nuc_source_case_label, sizeof(P->scheduled_nuc_source_case_label),
             "T400_xB0p030_no_strain");
    snprintf(P->scheduled_nuc_xB_edge_mode, sizeof(P->scheduled_nuc_xB_edge_mode),
             "sample-current-background-shell");
    snprintf(P->scheduled_nuc_edge_sample_stat, sizeof(P->scheduled_nuc_edge_sample_stat), "mean");
    P->scheduled_nuc_edge_sample_inner_nm = 0.0;
    P->scheduled_nuc_edge_sample_outer_nm = 3.0;
    P->scheduled_nuc_local_comp_inner_nm = 0.0;
    P->scheduled_nuc_local_comp_outer_nm = 20.0;
    P->scheduled_nuc_local_comp_taper_nm = 5.0;
    P->scheduled_nuc_source_dx_nm = 0.1;
    P->scheduled_nuc_source_lambda_nm = 0.6;
    P->scheduled_nuc_target_lambda_nm = 4.0;
    P->scheduled_nuc_scale_geometry = 1.0;
    P->scheduled_nuc_scale_interface_width = 0.0;   // auto: target/source lambda
    P->scheduled_nuc_scale_xB_profile_width = 0.0;  // auto: same as phi interface scale
    P->scheduled_nuc_alpha_interface = 0.25;
    P->scheduled_nuc_xB_min = 1.0e-8;
    P->scheduled_nuc_xB_max = 0.035;
    P->scheduled_nuc_phi_matrix_threshold = 0.05;
    P->scheduled_nuc_W_comp_threshold = 1.0e-3;
    P->scheduled_nuc_mass_iters = 10;
    P->scheduled_nuc_mass_tol = 1.0e-9;
    P->scheduled_nuc_write_event_vtk = 0;
    P->scheduled_nuc_fallback_analytic_sphere = 0;

    P->dynamics_mass_diag_enabled = 0;
    P->dynamics_mass_diag_interval = 1;
    P->enable_Y_rhs_previous_time_level = 0;
    P->disable_Y_rhs_gamma_term = 0;
    P->Y_rhs_term_h_scale = 1.0;
    P->enable_Y_rhs_picard = 0;
    P->Y_rhs_picard_iters = 1;
    P->Y_rhs_picard_omega = 1.0;

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
    atomicAddDoubleCompat(sum_pos, pos_grid * spacing);
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
    const double *d_eta_r,
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
        d_phi_r, d_eta_r, d_xB_r,  // Optimization(4): 需要 phi/eta 和 xB 来现场计算 eps0
        d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
        d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
        (float)P->eps_xx00, (float)P->eps_yy00, (float)P->eps_zz00,
        (float)P->eps_yz00, (float)P->eps_xz00, (float)P->eps_xy00,
        (double)P->eps_iso_over_vB,
        is_gp_zone_mode(P) ? 1 : 0,
        (is_gp_zone_mode(P) && P->gp_elastic_enabled) ? 1 : 0,
        P->gp_eps_iso,
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

static int parse_unsigned_long_value(const char *text, unsigned long *out) {
    if (!text || !out) return 0;
    errno = 0;
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || (end && *end != '\0')) return 0;
    *out = value;
    return 1;
}

static int is_valid_model_mode(const char *mode) {
    if (!mode || mode[0] == '\0') return 0;
    return (strcmp(mode, "two_phase") == 0 || strcmp(mode, "gp_zone") == 0);
}

static int is_gp_zone_mode(const PFParams *P) {
    return P && strcmp(P->model_mode, "gp_zone") == 0;
}

static const char *xBtot_output_stem(const PFParams *P) {
    return is_gp_zone_mode(P) ? "xBtot_gp" : "xBtot";
}

static const char *xBtot_vtk_field_name(const PFParams *P) {
    return is_gp_zone_mode(P) ? "xBtot_gp" : "xB_tot";
}

static const char *mean_xBtot_label(const PFParams *P) {
    return is_gp_zone_mode(P) ? "mean_xBtot_gp" : "mean_xBtot";
}

static double gp_y_update_mode_code(const PFParams *P) {
    if (!P) return 0.0;
    if (strcmp(P->gp_y_update_mode, "old_rhs") == 0) return 0.0;
    if (strcmp(P->gp_y_update_mode, "picard_storage") == 0) return 1.0;
    if (strcmp(P->gp_y_update_mode, "storage_exact") == 0) return 2.0;
    if (strcmp(P->gp_y_update_mode, "conservative_y_rhs") == 0) return 3.0;
    return -1.0;
}

static double compute_mu_C_gp_host(double xB_alpha,
                                   double temperature_K,
                                   double mu_reference_scale,
                                   double Vm_alpha_0,
                                   double dVm_alpha_dxB) {
    double x = clamp_fraction_eps(xB_alpha);
    double muA = mu_A_dimless(x, temperature_K, mu_reference_scale);
    double muB = mu_B_dimless(x, temperature_K, mu_reference_scale);
    double c_ref = 1.0 / Vm_alpha_of_xB(x, Vm_alpha_0, dVm_alpha_dxB);
    return c_ref * (muB - muA);
}

static double compute_mu_C_gp_slope_numeric_host(double xB_ref,
                                                 double temperature_K,
                                                 double mu_reference_scale,
                                                 double Vm_alpha_0,
                                                 double dVm_alpha_dxB) {
    const double dx = 1.0e-5;
    double x_plus = clamp_fraction_eps(xB_ref + dx);
    double x_minus = clamp_fraction_eps(xB_ref - dx);
    double mu_plus = compute_mu_C_gp_host(x_plus, temperature_K, mu_reference_scale,
                                          Vm_alpha_0, dVm_alpha_dxB);
    double mu_minus = compute_mu_C_gp_host(x_minus, temperature_K, mu_reference_scale,
                                           Vm_alpha_0, dVm_alpha_dxB);
    return (mu_plus - mu_minus) / fmax(x_plus - x_minus, 1.0e-30);
}

static double compute_g_alpha_host(double xB_alpha,
                                   double temperature_K,
                                   double mu_reference_scale) {
    double x = clamp_fraction_eps(xB_alpha);
    (void)mu_reference_scale;
    double muA = mu_PbTe_raw(temperature_K, x);
    double muB = mu_Ag2Te_raw(temperature_K, x);
    return (1.0 - x) * muA + x * muB;
}

static double compute_g_alpha_raw_regular_solution_host(double xB_alpha,
                                                        double temperature_K) {
    double x = clamp_fraction_eps(xB_alpha);
    double muA = mu_PbTe_calphad(temperature_K, x);
    double muB = mu_Ag2Te_calphad(temperature_K, x);
    return (1.0 - x) * muA + x * muB;
}

static double compute_gp_mu_reference_mech_mix_host(
        double xB_GP, double temperature_K, double delta_g_stab) {
    double G_A = G_PbTe_Solid(temperature_K);
    double G_B = G_Ag2Te_Solid(temperature_K);
    return (1.0 - xB_GP) * G_A + xB_GP * G_B - delta_g_stab;
}

static double compute_eta_interface_W_phys_host(double gamma_alpha_gp_Jm2,
                                                double l_eta_m) {
    if (!(gamma_alpha_gp_Jm2 > 0.0) || !(l_eta_m > 0.0)) return NAN;
    return 12.0 * gamma_alpha_gp_Jm2 / l_eta_m;
}

static double compute_eta_interface_kappa_phys_host(double gamma_alpha_gp_Jm2,
                                                    double l_eta_m) {
    if (!(gamma_alpha_gp_Jm2 > 0.0) || !(l_eta_m > 0.0)) return NAN;
    return 1.5 * gamma_alpha_gp_Jm2 * l_eta_m;
}

static double compute_interface_W_phys_ln9_host(double gamma_Jm2,
                                                double lambda_m) {
    if (!(gamma_Jm2 > 0.0) || !(lambda_m > 0.0)) return NAN;
    return 6.0 * gamma_Jm2 * log(9.0) / lambda_m;
}

static double compute_interface_kappa_phys_ln9_host(double gamma_Jm2,
                                                    double lambda_m) {
    if (!(gamma_Jm2 > 0.0) || !(lambda_m > 0.0)) return NAN;
    return 3.0 * gamma_Jm2 * lambda_m / log(9.0);
}

static double compute_eta_ref_dx_phys_m_host(const PFParams *P) {
    if (!P) return NAN;
    const double dx_phys_m =
        (P->ic_phi_iface_w > 1.0e-30)
            ? (P->lambda_sm_m / (2.0 * P->ic_phi_iface_w))
            : (P->dx * 1.0e-9);
    if (!(dx_phys_m > 0.0) || !(fabs(P->dx) > 1.0e-30)) return NAN;
    return dx_phys_m / P->dx;
}

static double compute_eta_ref_w_phys_host(const PFParams *P) {
    if (!P) return NAN;
    if (P->gamma_Jm2 > 0.0 && P->lambda_sm_m > 0.0) {
        return 12.0 * P->gamma_Jm2 / P->lambda_sm_m;
    }
    fprintf(stderr,
            "[fatal] invalid eta reference interface inputs: gamma_Jm2=%.6e lambda_sm_m=%.6e\n",
            P->gamma_Jm2, P->lambda_sm_m);
    return NAN;
}

static double compute_eta_ref_D_alpha_phys_host(const PFParams *P) {
    if (!P) return NAN;
    const double dx_ref_m = compute_eta_ref_dx_phys_m_host(P);
    if (!(dx_ref_m > 0.0) || !(P->t_real_unit > 0.0) || !(P->D_alpha > 0.0)) return NAN;
    return P->D_alpha * dx_ref_m * dx_ref_m / P->t_real_unit;
}

static double compute_eta_ref_L_ref_m_host(const PFParams *P) {
    if (!P) return NAN;
    const double D_alpha_phys = compute_eta_ref_D_alpha_phys_host(P);
    if (!(D_alpha_phys > 0.0) || !(P->t_real_unit > 0.0)) return NAN;
    return sqrt(D_alpha_phys * P->t_real_unit);
}

static double compute_gp_eta_zeta_host(double muA_x,
                                       double muB_x,
                                       double nu_A,
                                       double nu_B,
                                       double x_eq_alpha) {
    const double pref = nu_A * muA_x + nu_B * muB_x;
    const double bracket = nu_B / fmax(nu_A + nu_B, 1.0e-30) - x_eq_alpha;
    return pref * bracket;
}

static double xi_eq_profile_host(double x, double w, double kappa) {
    const double s = sqrt(w / (2.0 * kappa));
    return 0.5 * (1.0 - tanh(s * x));
}

static double dxi_dx_profile_host(double x, double w, double kappa) {
    const double s = sqrt(w / (2.0 * kappa));
    const double c = cosh(s * x);
    const double sech2 = 1.0 / (c * c);
    return -0.5 * s * sech2;
}

static double compute_gp_eta_zeta0_host(double l_eta_m,
                                        double W_eta_phys,
                                        double kappa_eta_phys,
                                        double D_alpha_phys,
                                        double D_eff_phys,
                                        int n_quad) {
    if (!(l_eta_m > 0.0) || !(W_eta_phys > 0.0) || !(kappa_eta_phys > 0.0) ||
        !(D_alpha_phys > 0.0) || !(D_eff_phys > 0.0) || n_quad < 5) {
        return NAN;
    }
    const double x_min = -0.5 * l_eta_m;
    const double x_max =  0.5 * l_eta_m;
    const double dx = (x_max - x_min) / (double)(n_quad - 1);
    std::vector<double> h_vals((size_t)n_quad, 0.0);
    std::vector<double> hprime_vals((size_t)n_quad, 0.0);
    std::vector<double> dxi_vals((size_t)n_quad, 0.0);
    std::vector<double> I_vals((size_t)n_quad, 0.0);
    std::vector<double> inner((size_t)n_quad, 0.0);
    for (int i = 0; i < n_quad; ++i) {
        const double x = x_min + dx * (double)i;
        const double xi = xi_eq_profile_host(x, W_eta_phys, kappa_eta_phys);
        dxi_vals[(size_t)i] = dxi_dx_profile_host(x, W_eta_phys, kappa_eta_phys);
        h_vals[(size_t)i] = h_of_eta(xi);
        hprime_vals[(size_t)i] = h_prime_of_eta(xi);
        const double Dmix = (1.0 - h_vals[(size_t)i]) * D_alpha_phys + h_vals[(size_t)i] * D_eff_phys;
        inner[(size_t)i] = (1.0 - h_vals[(size_t)i]) / fmax(Dmix, 1.0e-32);
    }
    for (int i = 1; i < n_quad; ++i) {
        I_vals[(size_t)i] = I_vals[(size_t)(i - 1)] + 0.5 * dx * (inner[(size_t)i] + inner[(size_t)(i - 1)]);
    }
    double outer_int = 0.0;
    for (int i = 1; i < n_quad; ++i) {
        const double outer_prev = hprime_vals[(size_t)(i - 1)] * dxi_vals[(size_t)(i - 1)] * I_vals[(size_t)(i - 1)];
        const double outer_curr = hprime_vals[(size_t)i] * dxi_vals[(size_t)i] * I_vals[(size_t)i];
        outer_int += 0.5 * dx * (outer_prev + outer_curr);
    }
    return -2.0 * D_alpha_phys / l_eta_m * outer_int;
}

enum {
    GP_ETA_INPUT_MODE_DEFAULT_INTERNAL = 0,
    GP_ETA_INPUT_MODE_CODE = 1,
    GP_ETA_INPUT_MODE_PHYSICAL_CONVERTED = 2,
    GP_ETA_INPUT_MODE_LEGACY_PHYSICAL_CONVERTED = 3,
    GP_ETA_INPUT_MODE_LEGACY_CODE_AS_IS = 4,
    GP_ETA_INPUT_MODE_DERIVED_PHYSICAL_CONVERTED = 5,
};

static const char *gp_eta_input_mode_label(int mode) {
    switch (mode) {
        case GP_ETA_INPUT_MODE_CODE: return "code";
        case GP_ETA_INPUT_MODE_PHYSICAL_CONVERTED: return "physical_converted";
        case GP_ETA_INPUT_MODE_LEGACY_PHYSICAL_CONVERTED: return "legacy_physical_converted";
        case GP_ETA_INPUT_MODE_LEGACY_CODE_AS_IS: return "legacy_code_as_is";
        case GP_ETA_INPUT_MODE_DERIVED_PHYSICAL_CONVERTED: return "derived_physical_converted";
        default: return "default_internal";
    }
}

static int gp_L_eta_mode_is_sto_S218b(const char *mode) {
    return mode && strcmp(mode, "sto_S218b") == 0;
}

static int gp_L_eta_mode_is_sto_S218b_override(const char *mode) {
    return mode && strcmp(mode, "sto_S218b_override") == 0;
}

static const char *classify_gp_eta_Mratio(double ratio) {
    if (!isfinite(ratio) || ratio <= 0.0) return "unavailable";
    if (ratio < 1.0e-2) return "strongly reaction-controlled";
    if (ratio < 1.0e-1) return "reaction-controlled";
    if (ratio <= 10.0) return "mixed-control";
    return "diffusion-controlled";
}

static int normalize_gp_eta_interface_units_host(PFParams *P) {
    if (!P) return 0;

    const double w_ref_phys = compute_eta_ref_w_phys_host(P);
    const double dx_ref_m = compute_eta_ref_dx_phys_m_host(P);
    const double l_eta_m =
        (P->gp_l_eta_nm > 0.0) ? (P->gp_l_eta_nm * 1.0e-9) : NAN;
    const int can_derive_from_gamma_lambda =
        isfinite(P->gp_gamma_alpha_gp) && P->gp_gamma_alpha_gp > 0.0 &&
        isfinite(l_eta_m) && l_eta_m > 0.0;

    P->gp_W_eta_input_mode_resolved = GP_ETA_INPUT_MODE_DEFAULT_INTERNAL;
    P->gp_kappa_eta_input_mode_resolved = GP_ETA_INPUT_MODE_DEFAULT_INTERNAL;

    if (P->gp_W_eta_code_specified) {
        P->gp_W_eta = P->gp_W_eta_code_input;
        P->gp_W_eta_input_mode_resolved = GP_ETA_INPUT_MODE_CODE;
        if (P->gp_W_eta_phys_specified) {
            const double expected_code = P->gp_W_eta_phys_input / fmax(w_ref_phys, 1.0e-30);
            if (isfinite(expected_code) && fabs(P->gp_W_eta_code_input - expected_code) / fmax(fabs(expected_code), 1.0e-30) > 1.0e-6) {
                fprintf(stderr,
                        "[fatal] gp_W_eta_code and gp_W_eta_phys are inconsistent (code=%.8e expected=%.8e)\n",
                        P->gp_W_eta_code_input, expected_code);
                return 0;
            }
        }
    } else if (P->gp_W_eta_phys_specified) {
        if (!(w_ref_phys > 0.0)) {
            fprintf(stderr, "[fatal] cannot convert gp_W_eta_phys to code units: invalid w_ref=%.6e\n", w_ref_phys);
            return 0;
        }
        P->gp_W_eta = P->gp_W_eta_phys_input / w_ref_phys;
        P->gp_W_eta_code_input = P->gp_W_eta;
        P->gp_W_eta_input_mode_resolved = GP_ETA_INPUT_MODE_PHYSICAL_CONVERTED;
    } else if (P->gp_W_eta_legacy_specified) {
        fprintf(stderr,
                "[warn] gp_W_eta is deprecated; please use gp_W_eta_code/gp_W_eta_phys instead.\n");
        P->gp_W_eta_code_input = P->gp_W_eta;
        P->gp_W_eta_input_mode_resolved = GP_ETA_INPUT_MODE_LEGACY_CODE_AS_IS;
    } else if (can_derive_from_gamma_lambda) {
        if (!(w_ref_phys > 0.0)) {
            fprintf(stderr, "[fatal] cannot derive gp_W_eta from gamma/lambda: invalid w_ref=%.6e\n", w_ref_phys);
            return 0;
        }
        P->gp_W_eta_phys_input = compute_eta_interface_W_phys_host(P->gp_gamma_alpha_gp, l_eta_m);
        P->gp_W_eta = P->gp_W_eta_phys_input / w_ref_phys;
        P->gp_W_eta_code_input = P->gp_W_eta;
        P->gp_W_eta_input_mode_resolved = GP_ETA_INPUT_MODE_DERIVED_PHYSICAL_CONVERTED;
    }
    if (!P->gp_W_eta_phys_specified && P->gp_W_eta_code_specified && isfinite(P->gp_W_eta_code_input) && w_ref_phys > 0.0) {
        P->gp_W_eta_phys_input = P->gp_W_eta_code_input * w_ref_phys;
    }

    if (P->gp_kappa_eta_code_specified) {
        P->gp_kappa_eta = P->gp_kappa_eta_code_input;
        P->gp_kappa_eta_input_mode_resolved = GP_ETA_INPUT_MODE_CODE;
        if (P->gp_kappa_eta_phys_specified) {
            const double expected_code = P->gp_kappa_eta_phys_input / fmax(w_ref_phys * dx_ref_m * dx_ref_m, 1.0e-30);
            if (isfinite(expected_code) && fabs(P->gp_kappa_eta_code_input - expected_code) / fmax(fabs(expected_code), 1.0e-30) > 1.0e-6) {
                fprintf(stderr,
                        "[fatal] gp_kappa_eta_code and gp_kappa_eta_phys are inconsistent (code=%.8e expected=%.8e)\n",
                        P->gp_kappa_eta_code_input, expected_code);
                return 0;
            }
        }
    } else if (P->gp_kappa_eta_phys_specified) {
        if (!(w_ref_phys > 0.0) || !(dx_ref_m > 0.0)) {
            fprintf(stderr,
                    "[fatal] cannot convert gp_kappa_eta_phys to code units: invalid w_ref=%.6e or dx_ref=%.6e\n",
                    w_ref_phys, dx_ref_m);
            return 0;
        }
        P->gp_kappa_eta = P->gp_kappa_eta_phys_input / (w_ref_phys * dx_ref_m * dx_ref_m);
        P->gp_kappa_eta_code_input = P->gp_kappa_eta;
        P->gp_kappa_eta_input_mode_resolved = GP_ETA_INPUT_MODE_PHYSICAL_CONVERTED;
    } else if (P->gp_kappa_eta_legacy_specified) {
        fprintf(stderr,
                "[warn] gp_kappa_eta is deprecated; please use gp_kappa_eta_code/gp_kappa_eta_phys instead.\n");
        P->gp_kappa_eta_code_input = P->gp_kappa_eta;
        P->gp_kappa_eta_input_mode_resolved = GP_ETA_INPUT_MODE_LEGACY_CODE_AS_IS;
    } else if (can_derive_from_gamma_lambda) {
        if (!(w_ref_phys > 0.0) || !(dx_ref_m > 0.0)) {
            fprintf(stderr,
                    "[fatal] cannot derive gp_kappa_eta from gamma/lambda: invalid w_ref=%.6e or dx_ref=%.6e\n",
                    w_ref_phys, dx_ref_m);
            return 0;
        }
        P->gp_kappa_eta_phys_input = compute_eta_interface_kappa_phys_host(P->gp_gamma_alpha_gp, l_eta_m);
        P->gp_kappa_eta = P->gp_kappa_eta_phys_input / (w_ref_phys * dx_ref_m * dx_ref_m);
        P->gp_kappa_eta_code_input = P->gp_kappa_eta;
        P->gp_kappa_eta_input_mode_resolved = GP_ETA_INPUT_MODE_DERIVED_PHYSICAL_CONVERTED;
    }
    if (!P->gp_kappa_eta_phys_specified && P->gp_kappa_eta_code_specified && isfinite(P->gp_kappa_eta_code_input) && w_ref_phys > 0.0 && dx_ref_m > 0.0) {
        P->gp_kappa_eta_phys_input = P->gp_kappa_eta_code_input * w_ref_phys * dx_ref_m * dx_ref_m;
    }

    if (P->gp_L_eta_code_specified) {
        P->gp_L_eta = P->gp_L_eta_code_input;
        if (P->gp_L_eta_phys_specified) {
            if (!(w_ref_phys > 0.0) || !(P->t_real_unit > 0.0)) {
                fprintf(stderr,
                        "[fatal] cannot validate gp_L_eta_phys: invalid w_ref=%.6e or t_real_unit=%.6e\n",
                        w_ref_phys, P->t_real_unit);
                return 0;
            }
            const double expected_code = P->gp_L_eta_phys_input * w_ref_phys * P->t_real_unit;
            if (isfinite(expected_code) &&
                fabs(P->gp_L_eta_code_input - expected_code) / fmax(fabs(expected_code), 1.0e-30) > 1.0e-6) {
                fprintf(stderr,
                        "[fatal] gp_L_eta_code and gp_L_eta_phys are inconsistent (code=%.8e expected=%.8e)\n",
                        P->gp_L_eta_code_input, expected_code);
                return 0;
            }
        }
    } else if (P->gp_L_eta_phys_specified) {
        if (!(w_ref_phys > 0.0) || !(P->t_real_unit > 0.0)) {
            fprintf(stderr,
                    "[fatal] cannot convert gp_L_eta_phys to code units: invalid w_ref=%.6e or t_real_unit=%.6e\n",
                    w_ref_phys, P->t_real_unit);
            return 0;
        }
        const double expected_code = P->gp_L_eta_phys_input * w_ref_phys * P->t_real_unit;
        P->gp_L_eta = expected_code;
        P->gp_L_eta_code_input = expected_code;
    } else if (P->gp_L_eta_legacy_specified) {
        fprintf(stderr,
                "[warn] gp_L_eta is deprecated; please use gp_L_eta_code/gp_L_eta_phys instead.\n");
        P->gp_L_eta_code_input = P->gp_L_eta;
    } else if (P->gp_L_eta_mode[0] != '\0' && strcmp(P->gp_L_eta_mode, "sto_S218b_override") == 0) {
        /* Final override happens later once the sto_S218b reference has been computed. */
    }
    if (!P->gp_L_eta_phys_specified && P->gp_L_eta_code_specified && isfinite(P->gp_L_eta_code_input) && w_ref_phys > 0.0 && P->t_real_unit > 0.0) {
        P->gp_L_eta_phys_input = P->gp_L_eta_code_input / (w_ref_phys * P->t_real_unit);
    }

    return 1;
}

typedef struct {
    int enabled;
    int applied;
    int reaction_ref_available;
    int uses_abs_zeta_product;
    char reaction_ref_reason[128];
    char gp_L_eta_mode[32];
    char eta_regime_classification[64];
    char gp_W_eta_input_source[32];
    char gp_kappa_eta_input_source[32];
    double T_K;
    double xB0;
    double D_alpha_ref_phys;
    double L_ref_m;
    double t0_diff_s;
    double dt_code;
    double dt_phys_s;
    double total_time_phys_s;
    double gp_gamma_alpha_gp;
    double gp_l_eta_nm;
    double W_eta_phys;
    double kappa_eta_phys;
    double gp_W_eta_phys_input;
    double gp_kappa_eta_phys_input;
    double gp_W_eta_code_input;
    double gp_kappa_eta_code_input;
    double gp_W_eta_kernel_used;
    double gp_kappa_eta_kernel_used;
    double gp_W_eta_current;
    double gp_kappa_eta_current;
    double gp_W_eta_ref_code;
    double gp_kappa_eta_ref_code;
    double gp_W_eta_rel_mismatch;
    double gp_kappa_eta_rel_mismatch;
    double gp_W_eta_kernel_vs_ref_ratio;
    double gp_kappa_eta_kernel_vs_ref_ratio;
    double gp_L_eta_phys_input;
    double gp_L_eta_code_input;
    double gp_L_eta_ref_code;
    double gp_L_eta_rel_mismatch;
    double gp_L_eta_kernel_vs_ref_ratio;
    double zeta_eta;
    double zeta0_eta;
    double D_eff_phys;
    double c_tot_phys;
    double v_A;
    double v_B;
    double Mcrit_eta;
    double gp_M_eta_phys_input;
    double gp_M_eta_ratio_to_crit_input;
    double M_eta_used;
    double M_eta_over_Mcrit;
    double L_eta_reaction_approx_phys;
    double L_eta_reaction_approx_code;
    double L_eta_full_phys;
    double L_eta_full_code;
    double L_eta_full_over_L_diff;
    double L_eta_full_over_L_reaction_approx;
    double L_eta_reaction_ref_phys;
    double L_eta_reaction_ref_code;
    double L_eta_diff_ref_phys;
    double L_eta_diff_ref_code;
    double gp_L_eta_current;
    double ratio_current_to_reaction_ref;
    double ratio_current_to_diff_ref;
    double S_eta_W;
    double S_eta_grad;
    double S_eta_total;
    double S_phi_W;
    double S_phi_grad;
    double S_phi_total;
    double w_ref_phys;
    double dx_ref_m;
    double l_eta_m;
    double kmax2;
    double mu_reference_scale;
    double gp_xB_eq_alpha_for_eta;
    double gp_xB_gp_ref;
    double phi_gamma_alpha_beta;
    double phi_lambda_nm;
    double W_phi_phys_expected_doc;
    double kappa_phi_phys_expected_doc;
    double W_phi_code_expected_doc;
    double kappa_phi_code_expected_doc;
    double W_phi_kernel_used;
    double kappa_phi_kernel_used;
    double W_phi_kernel_vs_doc_ratio;
    double kappa_phi_kernel_vs_doc_ratio;
    double W_eta_phys_doc;
    double kappa_eta_phys_doc;
    double gp_W_eta_code_doc;
    double gp_kappa_eta_code_doc;
    double gp_W_eta_kernel_vs_doc_ratio;
    double gp_kappa_eta_kernel_vs_doc_ratio;
    double W_doc_over_W_ln9;
    double kappa_doc_over_kappa_ln9;
    double W_phi_phys;
    double kappa_phi_phys;
    double L_phi_phys;
    double W_phi_code;
    double kappa_phi_code;
    double L_phi_code;
} GPEtaKineticRefDiag;

static GPEtaKineticRefDiag compute_gp_eta_kinetic_ref_diag_host(const PFParams *P) {
    GPEtaKineticRefDiag out;
    memset(&out, 0, sizeof(out));
    out.enabled = (P && P->gp_kinetic_ref_enabled) ? 1 : 0;
    out.applied = (P && P->gp_kinetic_ref_apply) ? 1 : 0;
    out.reaction_ref_available = 0;
    out.uses_abs_zeta_product = 1;
    snprintf(out.reaction_ref_reason, sizeof(out.reaction_ref_reason),
             "reaction/interface reference unavailable: M_GP not specified");
    snprintf(out.gp_L_eta_mode, sizeof(out.gp_L_eta_mode), "%s", (P ? P->gp_L_eta_mode : "manual"));
    snprintf(out.eta_regime_classification, sizeof(out.eta_regime_classification), "%s", "unavailable");
    out.L_eta_reaction_ref_phys = NAN;
    out.L_eta_reaction_ref_code = NAN;
    out.L_eta_reaction_approx_phys = NAN;
    out.L_eta_reaction_approx_code = NAN;
    out.L_eta_full_phys = NAN;
    out.L_eta_full_code = NAN;
    out.L_eta_full_over_L_diff = NAN;
    out.L_eta_full_over_L_reaction_approx = NAN;
    out.ratio_current_to_reaction_ref = NAN;
    out.ratio_current_to_diff_ref = NAN;
    out.W_eta_phys = NAN;
    out.kappa_eta_phys = NAN;
    snprintf(out.gp_W_eta_input_source, sizeof(out.gp_W_eta_input_source), "%s",
             gp_eta_input_mode_label(GP_ETA_INPUT_MODE_DEFAULT_INTERNAL));
    snprintf(out.gp_kappa_eta_input_source, sizeof(out.gp_kappa_eta_input_source), "%s",
             gp_eta_input_mode_label(GP_ETA_INPUT_MODE_DEFAULT_INTERNAL));
    out.gp_W_eta_phys_input = NAN;
    out.gp_kappa_eta_phys_input = NAN;
    out.gp_W_eta_code_input = NAN;
    out.gp_kappa_eta_code_input = NAN;
    out.gp_W_eta_kernel_used = NAN;
    out.gp_kappa_eta_kernel_used = NAN;
    out.gp_W_eta_ref_code = NAN;
    out.gp_kappa_eta_ref_code = NAN;
    out.gp_W_eta_rel_mismatch = NAN;
    out.gp_kappa_eta_rel_mismatch = NAN;
    out.gp_W_eta_kernel_vs_ref_ratio = NAN;
    out.gp_kappa_eta_kernel_vs_ref_ratio = NAN;
    out.gp_L_eta_phys_input = NAN;
    out.gp_L_eta_code_input = NAN;
    out.gp_L_eta_ref_code = NAN;
    out.gp_L_eta_rel_mismatch = NAN;
    out.gp_L_eta_kernel_vs_ref_ratio = NAN;
    out.zeta_eta = NAN;
    out.zeta0_eta = NAN;
    out.D_eff_phys = NAN;
    out.c_tot_phys = NAN;
    out.v_A = NAN;
    out.v_B = NAN;
    out.Mcrit_eta = NAN;
    out.gp_M_eta_phys_input = NAN;
    out.gp_M_eta_ratio_to_crit_input = NAN;
    out.M_eta_used = NAN;
    out.M_eta_over_Mcrit = NAN;
    out.L_eta_diff_ref_phys = NAN;
    out.L_eta_diff_ref_code = NAN;
    out.w_ref_phys = NAN;
    out.dx_ref_m = NAN;
    out.l_eta_m = NAN;
    out.kmax2 = NAN;
    out.W_phi_phys_expected_doc = NAN;
    out.kappa_phi_phys_expected_doc = NAN;
    out.W_phi_code_expected_doc = NAN;
    out.kappa_phi_code_expected_doc = NAN;
    out.W_phi_kernel_used = NAN;
    out.kappa_phi_kernel_used = NAN;
    out.W_phi_kernel_vs_doc_ratio = NAN;
    out.kappa_phi_kernel_vs_doc_ratio = NAN;
    out.W_eta_phys_doc = NAN;
    out.kappa_eta_phys_doc = NAN;
    out.gp_W_eta_code_doc = NAN;
    out.gp_kappa_eta_code_doc = NAN;
    out.gp_W_eta_kernel_vs_doc_ratio = NAN;
    out.gp_kappa_eta_kernel_vs_doc_ratio = NAN;
    out.W_doc_over_W_ln9 = NAN;
    out.kappa_doc_over_kappa_ln9 = NAN;
    if (!P || !out.enabled) return out;

    out.T_K = P->temperature_C + 273.15;
    out.xB0 = P->ic_23d_xB_out;
    out.D_alpha_ref_phys = compute_eta_ref_D_alpha_phys_host(P);
    out.L_ref_m = compute_eta_ref_L_ref_m_host(P);
    out.t0_diff_s = (P->t_real_unit > 0.0) ? P->t_real_unit : NAN;
    out.dt_code = P->dt;
    out.dt_phys_s = (isfinite(out.t0_diff_s) ? P->dt * out.t0_diff_s : NAN);
    out.total_time_phys_s = (isfinite(out.t0_diff_s) ? (double)P->nsteps * P->dt * out.t0_diff_s : NAN);
    out.gp_gamma_alpha_gp = P->gp_gamma_alpha_gp;
    out.gp_l_eta_nm = P->gp_l_eta_nm;
    snprintf(out.gp_W_eta_input_source, sizeof(out.gp_W_eta_input_source), "%s",
             gp_eta_input_mode_label(P->gp_W_eta_input_mode_resolved));
    snprintf(out.gp_kappa_eta_input_source, sizeof(out.gp_kappa_eta_input_source), "%s",
             gp_eta_input_mode_label(P->gp_kappa_eta_input_mode_resolved));
    out.gp_W_eta_phys_input = P->gp_W_eta_phys_input;
    out.gp_kappa_eta_phys_input = P->gp_kappa_eta_phys_input;
    out.gp_W_eta_code_input = P->gp_W_eta_code_input;
    out.gp_kappa_eta_code_input = P->gp_kappa_eta_code_input;
    out.gp_W_eta_kernel_used = P->gp_W_eta;
    out.gp_kappa_eta_kernel_used = P->gp_kappa_eta;
    out.gp_W_eta_current = P->gp_W_eta;
    out.gp_kappa_eta_current = P->gp_kappa_eta;
    out.gp_L_eta_current = P->gp_L_eta;
    out.gp_L_eta_phys_input = P->gp_L_eta_phys_input;
    out.gp_L_eta_code_input = P->gp_L_eta_code_input;
    out.gp_M_eta_phys_input = (P->gp_M_eta_phys > 0.0) ? P->gp_M_eta_phys : NAN;
    out.gp_M_eta_ratio_to_crit_input =
        (P->gp_M_eta_ratio_to_crit > 0.0) ? P->gp_M_eta_ratio_to_crit : NAN;
    out.applied = (gp_L_eta_mode_is_sto_S218b(P->gp_L_eta_mode) ||
                   gp_L_eta_mode_is_sto_S218b_override(P->gp_L_eta_mode)) ? 1 : out.applied;
    out.w_ref_phys = compute_eta_ref_w_phys_host(P);
    out.dx_ref_m = compute_eta_ref_dx_phys_m_host(P);
    out.mu_reference_scale = P->mu_reference_scale;
    out.gp_xB_eq_alpha_for_eta =
        (P->gp_xB_eq_alpha_for_eta > 0.0) ? P->gp_xB_eq_alpha_for_eta : P->ic_xB_eq_matrix;
    out.gp_xB_gp_ref = P->gp_xB_fixed;
    out.v_A = P->gp_reaction_nu_A;
    out.v_B = P->gp_reaction_nu_B;
    out.phi_gamma_alpha_beta = P->gamma_Jm2;
    out.phi_lambda_nm = (P->lambda_sm_m > 0.0) ? (P->lambda_sm_m * 1.0e9) : NAN;
    out.W_phi_code = P->W;
    out.kappa_phi_code = P->kappa_phi;
    out.L_phi_code = P->L_phi;
    out.W_phi_kernel_used = P->W;
    out.kappa_phi_kernel_used = P->kappa_phi;
    out.l_eta_m = (P->gp_l_eta_nm > 0.0) ? (P->gp_l_eta_nm * 1.0e-9) : NAN;
    out.kmax2 =
        pow(M_PI / fmax(P->dx, 1.0e-30), 2.0) +
        pow(M_PI / fmax(P->dy, 1.0e-30), 2.0) +
        pow(M_PI / fmax(P->dz, 1.0e-30), 2.0);

    if (isfinite(out.w_ref_phys) && out.w_ref_phys > 0.0 &&
        isfinite(out.dx_ref_m) && out.dx_ref_m > 0.0) {
        out.W_eta_phys_doc = P->gp_W_eta_phys_input;
        out.kappa_eta_phys_doc = P->gp_kappa_eta_phys_input;
        out.W_eta_phys = out.W_eta_phys_doc;
        out.kappa_eta_phys = out.kappa_eta_phys_doc;
        if (isfinite(out.W_eta_phys_doc)) {
            out.gp_W_eta_code_doc = out.W_eta_phys_doc / out.w_ref_phys;
        }
        if (isfinite(out.kappa_eta_phys_doc)) {
            out.gp_kappa_eta_code_doc = out.kappa_eta_phys_doc / (out.w_ref_phys * out.dx_ref_m * out.dx_ref_m);
        }
        if (isfinite(out.gp_W_eta_code_doc) && fabs(out.gp_W_eta_code_doc) > 1.0e-30) {
            out.gp_W_eta_rel_mismatch = (P->gp_W_eta - out.gp_W_eta_code_doc) / out.gp_W_eta_code_doc;
            out.gp_W_eta_kernel_vs_ref_ratio = P->gp_W_eta / out.gp_W_eta_code_doc;
            out.gp_W_eta_kernel_vs_doc_ratio = out.gp_W_eta_kernel_vs_ref_ratio;
        }
        if (isfinite(out.gp_kappa_eta_code_doc) && fabs(out.gp_kappa_eta_code_doc) > 1.0e-30) {
            out.gp_kappa_eta_rel_mismatch = (P->gp_kappa_eta - out.gp_kappa_eta_code_doc) / out.gp_kappa_eta_code_doc;
            out.gp_kappa_eta_kernel_vs_ref_ratio = P->gp_kappa_eta / out.gp_kappa_eta_code_doc;
            out.gp_kappa_eta_kernel_vs_doc_ratio = out.gp_kappa_eta_kernel_vs_ref_ratio;
        }
        if (isfinite(out.W_eta_phys_doc) && isfinite(out.kappa_eta_phys_doc)) {
            const double W_eta_ln9 = compute_interface_W_phys_ln9_host(out.gp_gamma_alpha_gp, out.l_eta_m);
            const double kappa_eta_ln9 = compute_interface_kappa_phys_ln9_host(out.gp_gamma_alpha_gp, out.l_eta_m);
            if (isfinite(W_eta_ln9) && fabs(W_eta_ln9) > 1.0e-30) {
                out.W_doc_over_W_ln9 = out.W_eta_phys_doc / W_eta_ln9;
            }
            if (isfinite(kappa_eta_ln9) && fabs(kappa_eta_ln9) > 1.0e-30) {
                out.kappa_doc_over_kappa_ln9 = out.kappa_eta_phys_doc / kappa_eta_ln9;
            }
        }
    }
    if (isfinite(out.phi_gamma_alpha_beta) && out.phi_gamma_alpha_beta > 0.0 &&
        isfinite(P->lambda_sm_m) && P->lambda_sm_m > 0.0) {
        out.W_phi_phys_expected_doc = 12.0 * out.phi_gamma_alpha_beta / P->lambda_sm_m;
        out.kappa_phi_phys_expected_doc = 1.5 * out.phi_gamma_alpha_beta * P->lambda_sm_m;
        if (isfinite(out.w_ref_phys) && out.w_ref_phys > 0.0) {
            out.W_phi_code_expected_doc = out.W_phi_phys_expected_doc / out.w_ref_phys;
            if (isfinite(out.dx_ref_m) && out.dx_ref_m > 0.0) {
                out.kappa_phi_code_expected_doc =
                    out.kappa_phi_phys_expected_doc / (out.w_ref_phys * out.dx_ref_m * out.dx_ref_m);
            }
        }
        if (isfinite(out.W_phi_code_expected_doc) && fabs(out.W_phi_code_expected_doc) > 1.0e-30) {
            out.W_phi_kernel_vs_doc_ratio = out.W_phi_kernel_used / out.W_phi_code_expected_doc;
        }
        if (isfinite(out.kappa_phi_code_expected_doc) && fabs(out.kappa_phi_code_expected_doc) > 1.0e-30) {
            out.kappa_phi_kernel_vs_doc_ratio = out.kappa_phi_kernel_used / out.kappa_phi_code_expected_doc;
        }
    }
    if (isfinite(out.w_ref_phys) && out.w_ref_phys > 0.0 &&
        isfinite(out.dx_ref_m) && out.dx_ref_m > 0.0) {
        out.W_phi_phys = out.W_phi_code * out.w_ref_phys;
        out.kappa_phi_phys = out.kappa_phi_code * out.w_ref_phys * out.dx_ref_m * out.dx_ref_m;
    } else {
        out.W_phi_phys = NAN;
        out.kappa_phi_phys = NAN;
    }
    if (isfinite(out.w_ref_phys) && out.w_ref_phys > 0.0 &&
        isfinite(out.t0_diff_s) && out.t0_diff_s > 0.0) {
        out.L_phi_phys = out.L_phi_code / (out.w_ref_phys * out.t0_diff_s);
    } else {
        out.L_phi_phys = NAN;
    }

    if (isfinite(out.D_alpha_ref_phys) && out.D_alpha_ref_phys > 0.0) {
        out.D_eff_phys = out.D_alpha_ref_phys * fmax(P->gp_D_ratio, 0.0);
    }
    if (isfinite(P->Vm_alpha_0_phys_m3mol) && P->Vm_alpha_0_phys_m3mol > 0.0) {
        out.c_tot_phys = 1.0 / P->Vm_alpha_0_phys_m3mol;
    }
    if (isfinite(out.T_K) && out.T_K > 0.0 &&
        isfinite(out.gp_xB_eq_alpha_for_eta) && out.gp_xB_eq_alpha_for_eta > 0.0 && out.gp_xB_eq_alpha_for_eta < 1.0) {
        const double muA_x = mu_PbTe_calphad(out.T_K, out.gp_xB_eq_alpha_for_eta);
        const double muB_x = mu_Ag2Te_calphad(out.T_K, out.gp_xB_eq_alpha_for_eta);
        out.zeta_eta = compute_gp_eta_zeta_host(muA_x, muB_x,
                                                P->gp_reaction_nu_A, P->gp_reaction_nu_B,
                                                out.gp_xB_eq_alpha_for_eta);
    }
    if (isfinite(out.l_eta_m) && isfinite(out.W_eta_phys) && isfinite(out.kappa_eta_phys) &&
        isfinite(out.D_alpha_ref_phys) && isfinite(out.D_eff_phys)) {
        out.zeta0_eta = compute_gp_eta_zeta0_host(out.l_eta_m, out.W_eta_phys, out.kappa_eta_phys,
                                                  out.D_alpha_ref_phys, out.D_eff_phys, 2001);
    }
    if (isfinite(out.D_eff_phys) && isfinite(out.l_eta_m) && out.l_eta_m > 0.0 &&
        isfinite(out.zeta_eta) && isfinite(out.zeta0_eta) &&
        isfinite(out.c_tot_phys) && out.c_tot_phys > 0.0) {
        const double zprod = out.zeta_eta * out.zeta0_eta;
        const double abs_zprod = fabs(zprod);
        if (abs_zprod > 1.0e-30) {
            const double v_sum = P->gp_reaction_nu_A + P->gp_reaction_nu_B;
            const double prefactor = 2.0 * v_sum / (3.0 * out.c_tot_phys * out.l_eta_m);
            out.Mcrit_eta = 2.0 * out.D_alpha_ref_phys / (abs_zprod * out.l_eta_m);
            out.L_eta_diff_ref_phys =
                4.0 * v_sum * out.D_alpha_ref_phys /
                (3.0 * out.c_tot_phys * out.l_eta_m * out.l_eta_m * abs_zprod);
            if (isfinite(out.w_ref_phys) && isfinite(out.t0_diff_s)) {
                out.L_eta_diff_ref_code = out.L_eta_diff_ref_phys * out.w_ref_phys * out.t0_diff_s;
            }

            if (P->gp_M_eta_phys > 0.0) {
                out.gp_M_eta_phys_input = P->gp_M_eta_phys;
                out.M_eta_used = P->gp_M_eta_phys;
                if (P->gp_M_eta_ratio_to_crit > 0.0) {
                    fprintf(stderr,
                            "[warn] both gp_M_eta_phys and gp_M_eta_ratio_to_crit provided; using physical M.\n");
                    snprintf(out.reaction_ref_reason, sizeof(out.reaction_ref_reason),
                             "both physical M and ratio provided; using physical M.");
                }
            } else if (P->gp_M_eta_ratio_to_crit > 0.0) {
                out.gp_M_eta_ratio_to_crit_input = P->gp_M_eta_ratio_to_crit;
                out.M_eta_used = P->gp_M_eta_ratio_to_crit * out.Mcrit_eta;
            } else if (P->gp_M_int_eta > 0.0) {
                out.gp_M_eta_phys_input = P->gp_M_int_eta;
                out.M_eta_used = P->gp_M_int_eta;
                snprintf(out.reaction_ref_reason, sizeof(out.reaction_ref_reason),
                         "using legacy gp_M_int_eta as M_eta physical input");
            }

            if (isfinite(out.Mcrit_eta) && out.Mcrit_eta > 0.0 &&
                isfinite(out.M_eta_used) && out.M_eta_used > 0.0) {
                out.M_eta_over_Mcrit = out.M_eta_used / out.Mcrit_eta;
                snprintf(out.eta_regime_classification, sizeof(out.eta_regime_classification),
                         "%s", classify_gp_eta_Mratio(out.M_eta_over_Mcrit));
                out.L_eta_reaction_approx_phys = prefactor * out.M_eta_used;
                out.L_eta_full_phys =
                    prefactor / (1.0 / out.M_eta_used + abs_zprod * out.l_eta_m / (2.0 * out.D_alpha_ref_phys));
                if (isfinite(out.w_ref_phys) && isfinite(out.t0_diff_s)) {
                    out.L_eta_reaction_approx_code = out.L_eta_reaction_approx_phys * out.w_ref_phys * out.t0_diff_s;
                    out.L_eta_full_code = out.L_eta_full_phys * out.w_ref_phys * out.t0_diff_s;
                }
                out.L_eta_reaction_ref_phys = out.L_eta_reaction_approx_phys;
                out.L_eta_reaction_ref_code = out.L_eta_reaction_approx_code;
                out.reaction_ref_available = 1;
                snprintf(out.reaction_ref_reason, sizeof(out.reaction_ref_reason), "available");
                if (isfinite(out.L_eta_diff_ref_phys) && fabs(out.L_eta_diff_ref_phys) > 1.0e-30) {
                    out.L_eta_full_over_L_diff = out.L_eta_full_phys / out.L_eta_diff_ref_phys;
                }
                if (isfinite(out.L_eta_reaction_approx_phys) && fabs(out.L_eta_reaction_approx_phys) > 1.0e-30) {
                    out.L_eta_full_over_L_reaction_approx = out.L_eta_full_phys / out.L_eta_reaction_approx_phys;
                }
            } else if (gp_L_eta_mode_is_sto_S218b(P->gp_L_eta_mode) ||
                       gp_L_eta_mode_is_sto_S218b_override(P->gp_L_eta_mode)) {
                snprintf(out.reaction_ref_reason, sizeof(out.reaction_ref_reason),
                         "reaction/interface reference unavailable: M_GP not specified");
            }
        }
    }
    if (isfinite(out.L_eta_reaction_ref_code) && fabs(out.L_eta_reaction_ref_code) > 1.0e-30) {
        out.ratio_current_to_reaction_ref = P->gp_L_eta / out.L_eta_reaction_ref_code;
    }
    if (isfinite(out.L_eta_diff_ref_code) && fabs(out.L_eta_diff_ref_code) > 1.0e-30) {
        out.ratio_current_to_diff_ref = P->gp_L_eta / out.L_eta_diff_ref_code;
    }
    if (isfinite(out.gp_L_eta_phys_input) && isfinite(out.w_ref_phys) && isfinite(out.t0_diff_s) &&
        out.gp_L_eta_phys_input > 0.0 && out.w_ref_phys > 0.0 && out.t0_diff_s > 0.0) {
        out.gp_L_eta_ref_code = out.gp_L_eta_phys_input * out.w_ref_phys * out.t0_diff_s;
        if (fabs(out.gp_L_eta_ref_code) > 1.0e-30) {
            out.gp_L_eta_rel_mismatch = (P->gp_L_eta - out.gp_L_eta_ref_code) / out.gp_L_eta_ref_code;
            out.gp_L_eta_kernel_vs_ref_ratio = P->gp_L_eta / out.gp_L_eta_ref_code;
        }
    }
    out.S_eta_W = P->gp_L_eta * P->dt * P->gp_W_eta;
    out.S_eta_grad = P->gp_L_eta * P->dt * P->gp_kappa_eta * out.kmax2;
    out.S_eta_total = out.S_eta_W + out.S_eta_grad;
    out.S_phi_W = P->L_phi * P->dt * P->W;
    out.S_phi_grad = P->L_phi * P->dt * P->kappa_phi * out.kmax2;
    out.S_phi_total = out.S_phi_W + out.S_phi_grad;
    return out;
}

static void print_gp_eta_kinetic_ref_diag(const GPEtaKineticRefDiag *D) {
    if (!D || !D->enabled) return;
    printf("\n[STO/pseudobinary thin-interface parameterization diagnostics]\n");
    printf("  interface_parameterization     = document_w_12gamma_over_lambda_kappa_1p5gamma_lambda\n");
    printf("  framework                      = phi and eta use the same STO-type pseudobinary reaction phase-field parameterization\n");
    printf("  w_ref                          = %.8e J/m^3\n", D->w_ref_phys);
    printf("  dx_ref                         = %.8e m\n", D->dx_ref_m);
    printf("  t0_diff                        = %.8e s\n", D->t0_diff_s);
    printf("  mu_reference_scale             = %.8e J/mol\n", D->mu_reference_scale);
    printf("\n");
    printf("  [phi thin-interface inputs/code]\n");
    printf("    gamma_alpha_beta             = %.8e J/m^2\n", D->phi_gamma_alpha_beta);
    printf("    lambda_phi                   = %.8e nm\n", D->phi_lambda_nm);
    printf("    W_phi_phys_expected_doc      = %.8e J/m^3\n", D->W_phi_phys_expected_doc);
    printf("    kappa_phi_phys_expected_doc  = %.8e J/m\n", D->kappa_phi_phys_expected_doc);
    printf("    W_phi_code_expected_doc      = %.8e\n", D->W_phi_code_expected_doc);
    printf("    kappa_phi_code_expected_doc  = %.8e\n", D->kappa_phi_code_expected_doc);
    printf("    W_phi_kernel_used            = %.8e\n", D->W_phi_kernel_used);
    printf("    kappa_phi_kernel_used        = %.8e\n", D->kappa_phi_kernel_used);
    printf("    W_phi_kernel_vs_doc_ratio    = %.8e\n", D->W_phi_kernel_vs_doc_ratio);
    printf("    kappa_phi_kernel_vs_doc_ratio = %.8e\n", D->kappa_phi_kernel_vs_doc_ratio);
    printf("    W_phi_phys                   = %.8e J/m^3\n", D->W_phi_phys);
    printf("    kappa_phi_phys               = %.8e J/m\n", D->kappa_phi_phys);
    printf("    L_phi_phys                   = %.8e\n", D->L_phi_phys);
    printf("    W_phi_code                   = %.8e\n", D->W_phi_code);
    printf("    kappa_phi_code               = %.8e\n", D->kappa_phi_code);
    printf("    L_phi_code                   = %.8e\n", D->L_phi_code);
    printf("\n");
    printf("  [eta thin-interface inputs/code]\n");
    printf("    gp_gamma_alpha_gp            = %.8e J/m^2\n", D->gp_gamma_alpha_gp);
    printf("    gp_l_eta_nm                  = %.8e nm\n", D->gp_l_eta_nm);
    printf("    W_eta_phys_doc               = %.8e J/m^3\n", D->W_eta_phys_doc);
    printf("    kappa_eta_phys_doc           = %.8e J/m\n", D->kappa_eta_phys_doc);
    printf("  gp_W_eta_input_source          = %s\n", D->gp_W_eta_input_source);
    printf("  gp_kappa_eta_input_source      = %s\n", D->gp_kappa_eta_input_source);
    printf("  gp_W_eta_phys_input            = %.8e J/m^3\n", D->gp_W_eta_phys_input);
    printf("  gp_kappa_eta_phys_input        = %.8e J/m\n", D->gp_kappa_eta_phys_input);
    printf("  gp_W_eta_code_input            = %.8e\n", D->gp_W_eta_code_input);
    printf("  gp_kappa_eta_code_input        = %.8e\n", D->gp_kappa_eta_code_input);
    printf("  gp_W_eta_code_doc              = %.8e\n", D->gp_W_eta_code_doc);
    printf("  gp_kappa_eta_code_doc          = %.8e\n", D->gp_kappa_eta_code_doc);
    printf("  gp_W_eta_kernel_used           = %.8e\n", D->gp_W_eta_kernel_used);
    printf("  gp_kappa_eta_kernel_used       = %.8e\n", D->gp_kappa_eta_kernel_used);
    printf("  gp_W_eta_kernel_vs_doc_ratio   = %.8e\n", D->gp_W_eta_kernel_vs_doc_ratio);
    printf("  gp_kappa_eta_kernel_vs_doc_ratio = %.8e\n", D->gp_kappa_eta_kernel_vs_doc_ratio);
    printf("  W_doc_over_W_ln9               = %.8e\n", D->W_doc_over_W_ln9);
    printf("  kappa_doc_over_kappa_ln9       = %.8e\n", D->kappa_doc_over_kappa_ln9);
    printf("  gp_xB_eq_alpha_for_eta         = %.8e\n", D->gp_xB_eq_alpha_for_eta);
    printf("  gp_xB_gp_ref(reuse gp_xB_fixed)= %.8e\n", D->gp_xB_gp_ref);
    printf("\n");
    printf("  [GP STO-type kinetic references]\n");
    printf("  D_alpha_ref                    = %.8e m^2/s\n", D->D_alpha_ref_phys);
    printf("  L_ref                          = %.8e m\n", D->L_ref_m);
    printf("  dt_code                        = %.8e\n", D->dt_code);
    printf("  dt_phys                        = %.8e s\n", D->dt_phys_s);
    printf("  total_time_phys                = %.8e s\n", D->total_time_phys_s);
    printf("  c_tot                          = %.8e mol/m^3\n", D->c_tot_phys);
    printf("  v_A                            = %.8e\n", D->v_A);
    printf("  v_B                            = %.8e\n", D->v_B);
    printf("  zeta_eta                       = %.8e\n", D->zeta_eta);
    printf("  zeta0_eta                      = %.8e\n", D->zeta0_eta);
    printf("  D_eff                          = %.8e m^2/s\n", D->D_eff_phys);
    printf("  L_eta_diff_ref_phys            = %.8e\n", D->L_eta_diff_ref_phys);
    printf("  L_eta_diff_ref_code            = %.8e\n", D->L_eta_diff_ref_code);
    printf("  gp_L_eta_current_code          = %.8e\n", D->gp_L_eta_current);
    printf("  ratio_current_to_reaction_ref  = %.8e\n", D->ratio_current_to_reaction_ref);
    printf("  ratio_current_to_diff_ref      = %.8e\n", D->ratio_current_to_diff_ref);
    printf("  S_eta_W                        = %.8e\n", D->S_eta_W);
    printf("  S_eta_grad                     = %.8e\n", D->S_eta_grad);
    printf("  S_eta_total                    = %.8e\n", D->S_eta_total);
    printf("  S_phi_W                        = %.8e\n", D->S_phi_W);
    printf("  S_phi_grad                     = %.8e\n", D->S_phi_grad);
    printf("  S_phi_total                    = %.8e\n", D->S_phi_total);
    if (isfinite(D->ratio_current_to_reaction_ref)) {
        if (D->ratio_current_to_reaction_ref < 0.1) {
            printf("  regime_vs_reaction_ref         = eta slower than supplied reaction reference\n");
        } else if (D->ratio_current_to_reaction_ref <= 10.0) {
            printf("  regime_vs_reaction_ref         = eta near finite reaction/interface-controlled reference\n");
        } else {
            printf("  regime_vs_reaction_ref         = eta larger than supplied reaction reference\n");
        }
    }
    if (isfinite(D->ratio_current_to_diff_ref)) {
        if (D->ratio_current_to_diff_ref < 0.1) {
            printf("  regime_vs_diff_ref             = eta is below STO diffusion-controlled limit\n");
        } else if (D->ratio_current_to_diff_ref <= 10.0) {
            printf("  regime_vs_diff_ref             = eta approaches STO diffusion-controlled limit; check projection-like behavior\n");
        } else {
            printf("  regime_vs_diff_ref             = eta exceeds STO diffusion-controlled limit; check projection-like behavior\n");
        }
    }
    if (D->S_eta_W > 1.0) {
        printf("  WARNING                        : explicit source term may be unstable (S_eta_W=%.3e > 1.0)\n", D->S_eta_W);
    }
    if (isfinite(D->ratio_current_to_diff_ref) && D->ratio_current_to_diff_ref > 1.0) {
        printf("  WARNING                        : chosen eta mobility exceeds STO diffusion-limit reference; this may be inappropriate for a GP precursor\n");
    }
    printf("\n");
    printf("[GP eta STO-SM Eq. S2.18b kinetic factor]\n");
    printf("  gp_L_eta_mode                  = %s\n", D->gp_L_eta_mode);
    printf("  lambda_eta                     = %.8e m\n", D->l_eta_m);
    printf("  gp_gamma_alpha_gp              = %.8e J/m^2\n", D->gp_gamma_alpha_gp);
    printf("  D_alpha                        = %.8e m^2/s\n", D->D_alpha_ref_phys);
    printf("  c                              = %.8e mol/m^3\n", D->c_tot_phys);
    printf("  v_A                            = %.8e\n", D->v_A);
    printf("  v_B                            = %.8e\n", D->v_B);
    printf("  zeta_eta                       = %.8e\n", D->zeta_eta);
    printf("  zeta0_eta                      = %.8e\n", D->zeta0_eta);
    printf("  uses_abs_zeta_product          = %d\n", D->uses_abs_zeta_product);
    printf("  Mcrit_eta                      = %.8e\n", D->Mcrit_eta);
    printf("  gp_M_eta_phys_input            = %.8e\n", D->gp_M_eta_phys_input);
    printf("  gp_M_eta_ratio_to_crit_input   = %.8e\n", D->gp_M_eta_ratio_to_crit_input);
    printf("  M_eta_used                     = %.8e\n", D->M_eta_used);
    printf("  M_eta_over_Mcrit               = %.8e\n", D->M_eta_over_Mcrit);
    printf("  L_eta_reaction_approx_phys     = %.8e\n", D->L_eta_reaction_approx_phys);
    printf("  L_eta_reaction_approx_code     = %.8e\n", D->L_eta_reaction_approx_code);
    printf("  L_eta_full_phys                = %.8e\n", D->L_eta_full_phys);
    printf("  L_eta_full_code                = %.8e\n", D->L_eta_full_code);
    printf("  L_eta_diff_phys                = %.8e\n", D->L_eta_diff_ref_phys);
    printf("  L_eta_diff_code                = %.8e\n", D->L_eta_diff_ref_code);
    printf("  L_eta_full_over_L_diff         = %.8e\n", D->L_eta_full_over_L_diff);
    printf("  L_eta_full_over_L_reaction_approx = %.8e\n", D->L_eta_full_over_L_reaction_approx);
    printf("  gp_L_eta_kernel_used           = %.8e\n", D->gp_L_eta_current);
    printf("  eta_regime_classification      = %s\n", D->eta_regime_classification);
    if (D->reaction_ref_available) {
        printf("  gp_L_eta computed from STO-SM Eq. S2.18b = %s\n",
               (gp_L_eta_mode_is_sto_S218b(D->gp_L_eta_mode) ||
                gp_L_eta_mode_is_sto_S218b_override(D->gp_L_eta_mode)) ? "yes" : "no");
        if (gp_L_eta_mode_is_sto_S218b(D->gp_L_eta_mode) ||
            gp_L_eta_mode_is_sto_S218b_override(D->gp_L_eta_mode)) {
            printf("  gp_L_eta computed from STO-SM Eq. S2.18b\n");
        }
    } else if (gp_L_eta_mode_is_sto_S218b(D->gp_L_eta_mode) ||
               gp_L_eta_mode_is_sto_S218b_override(D->gp_L_eta_mode)) {
        printf("  reaction/interface reference unavailable: %s\n", D->reaction_ref_reason);
    }
}

static void write_gp_eta_kinetic_ref_diag_csv(const char *case_output_dir,
                                              const GPEtaKineticRefDiag *D) {
    if (!case_output_dir || !case_output_dir[0] || !D || !D->enabled) return;
    char path[4096];
    snprintf(path, sizeof(path), "%s/gp_eta_kinetic_reference_diagnostics.csv", case_output_dir);
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "[warn] cannot open GP kinetic reference diagnostics CSV: %s\n", path);
        return;
    }
    fprintf(fp,
            "interface_parameterization,T,xB0,D_alpha_ref,L_ref,t0_diff,dt_code,dt_phys,gp_gamma_alpha_gp,gp_l_eta_nm,"
            "gp_L_eta_mode,c_tot_phys,v_A,v_B,Mcrit_eta,gp_M_eta_phys_input,gp_M_eta_ratio_to_crit_input,M_eta_used,M_eta_over_Mcrit,"
            "L_eta_reaction_approx_phys,L_eta_reaction_approx_code,L_eta_full_phys,L_eta_full_code,L_eta_full_over_L_diff,L_eta_full_over_L_reaction_approx,eta_regime_classification,"
            "phi_gamma_alpha_beta,phi_lambda_nm,W_phi_phys_expected_doc,kappa_phi_phys_expected_doc,W_phi_code_expected_doc,kappa_phi_code_expected_doc,"
            "W_phi_kernel_used,kappa_phi_kernel_used,W_phi_kernel_vs_doc_ratio,kappa_phi_kernel_vs_doc_ratio,"
            "W_phi_phys,kappa_phi_phys,L_phi_phys,W_phi_code,kappa_phi_code,L_phi_code,"
            "gp_W_eta_input_source,gp_kappa_eta_input_source,"
            "W_eta_phys_doc,kappa_eta_phys_doc,gp_W_eta_phys_input,gp_kappa_eta_phys_input,"
            "gp_W_eta_code_input,gp_kappa_eta_code_input,gp_W_eta_kernel_used,gp_kappa_eta_kernel_used,gp_W_eta_code_doc,gp_kappa_eta_code_doc,"
            "gp_W_eta_kernel_vs_doc_ratio,gp_kappa_eta_kernel_vs_doc_ratio,W_doc_over_W_ln9,kappa_doc_over_kappa_ln9,"
            "zeta_eta,zeta0_eta,D_eff,L_eta_reaction_ref_phys,L_eta_reaction_ref_code,L_eta_diff_ref_phys,L_eta_diff_ref_code,"
            "gp_L_eta_current,ratio_current_to_reaction_ref,ratio_current_to_diff_ref,S_eta_W,S_eta_grad,S_eta_total,"
            "S_phi_W,S_phi_grad,S_phi_total\n");
    fprintf(fp,
            "%s,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
            "%s,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
            "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%s,"
            "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
            "%.10e,%.10e,%.10e,%.10e,"
            "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
            "%s,%s,"
            "%.10e,%.10e,%.10e,%.10e,"
            "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
            "%.10e,%.10e,%.10e,%.10e,"
            "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
            "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
            "%.10e,%.10e,%.10e\n",
            "document_w_12gamma_over_lambda_kappa_1p5gamma_lambda",
            D->T_K, D->xB0, D->D_alpha_ref_phys, D->L_ref_m, D->t0_diff_s,
            D->dt_code, D->dt_phys_s, D->gp_gamma_alpha_gp, D->gp_l_eta_nm,
            D->gp_L_eta_mode, D->c_tot_phys, D->v_A, D->v_B, D->Mcrit_eta,
            D->gp_M_eta_phys_input, D->gp_M_eta_ratio_to_crit_input, D->M_eta_used, D->M_eta_over_Mcrit,
            D->L_eta_reaction_approx_phys, D->L_eta_reaction_approx_code, D->L_eta_full_phys, D->L_eta_full_code,
            D->L_eta_full_over_L_diff, D->L_eta_full_over_L_reaction_approx, D->eta_regime_classification,
            D->phi_gamma_alpha_beta, D->phi_lambda_nm, D->W_phi_phys_expected_doc, D->kappa_phi_phys_expected_doc,
            D->W_phi_code_expected_doc, D->kappa_phi_code_expected_doc,
            D->W_phi_kernel_used, D->kappa_phi_kernel_used, D->W_phi_kernel_vs_doc_ratio, D->kappa_phi_kernel_vs_doc_ratio,
            D->W_phi_phys, D->kappa_phi_phys, D->L_phi_phys, D->W_phi_code, D->kappa_phi_code, D->L_phi_code,
            D->gp_W_eta_input_source, D->gp_kappa_eta_input_source,
            D->W_eta_phys_doc, D->kappa_eta_phys_doc, D->gp_W_eta_phys_input, D->gp_kappa_eta_phys_input,
            D->gp_W_eta_code_input, D->gp_kappa_eta_code_input, D->gp_W_eta_kernel_used, D->gp_kappa_eta_kernel_used,
            D->gp_W_eta_code_doc, D->gp_kappa_eta_code_doc,
            D->gp_W_eta_kernel_vs_doc_ratio, D->gp_kappa_eta_kernel_vs_doc_ratio,
            D->W_doc_over_W_ln9, D->kappa_doc_over_kappa_ln9,
            D->zeta_eta, D->zeta0_eta, D->D_eff_phys, D->L_eta_reaction_ref_phys,
            D->L_eta_reaction_ref_code, D->L_eta_diff_ref_phys, D->L_eta_diff_ref_code,
            D->gp_L_eta_current, D->ratio_current_to_reaction_ref, D->ratio_current_to_diff_ref,
            D->S_eta_W, D->S_eta_grad, D->S_eta_total,
            D->S_phi_W, D->S_phi_grad, D->S_phi_total);
    fclose(fp);
}

static void write_gp_eta_S218b_kinetic_diag_csv(const char *case_output_dir,
                                                const GPEtaKineticRefDiag *D) {
    if (!case_output_dir || !case_output_dir[0] || !D || !D->enabled) return;
    char path[4096];
    snprintf(path, sizeof(path), "%s/gp_eta_S218b_kinetic_diagnostics.csv", case_output_dir);
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "[warn] cannot open GP S2.18b kinetic diagnostics CSV: %s\n", path);
        return;
    }
    fprintf(fp,
            "T,xB0,gp_L_eta_mode,lambda_eta,gp_gamma_alpha_gp,D_alpha,c,v_A,v_B,zeta_eta,zeta0_eta,"
            "Mcrit_eta,gp_M_eta_phys_input,gp_M_eta_ratio_to_crit_input,M_eta_used,M_eta_over_Mcrit,"
            "L_eta_reaction_approx_phys,L_eta_reaction_approx_code,L_eta_full_phys,L_eta_full_code,"
            "L_eta_diff_phys,L_eta_diff_code,L_eta_full_over_L_diff,L_eta_full_over_L_reaction_approx,"
            "gp_L_eta_kernel_used,S_eta_W,S_eta_grad,S_eta_total,eta_regime_classification\n");
    fprintf(fp,
            "%.10e,%.10e,%s,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,"
            "%.10e,%.10e,%.10e,%.10e,%.10e,"
            "%.10e,%.10e,%.10e,%.10e,"
            "%.10e,%.10e,%.10e,%.10e,"
            "%.10e,%.10e,%.10e,%.10e,%s\n",
            D->T_K, D->xB0, D->gp_L_eta_mode, D->l_eta_m, D->gp_gamma_alpha_gp,
            D->D_alpha_ref_phys, D->c_tot_phys, D->v_A, D->v_B, D->zeta_eta, D->zeta0_eta,
            D->Mcrit_eta, D->gp_M_eta_phys_input, D->gp_M_eta_ratio_to_crit_input, D->M_eta_used, D->M_eta_over_Mcrit,
            D->L_eta_reaction_approx_phys, D->L_eta_reaction_approx_code, D->L_eta_full_phys, D->L_eta_full_code,
            D->L_eta_diff_ref_phys, D->L_eta_diff_ref_code, D->L_eta_full_over_L_diff, D->L_eta_full_over_L_reaction_approx,
            D->gp_L_eta_current, D->S_eta_W, D->S_eta_grad, D->S_eta_total, D->eta_regime_classification);
    fclose(fp);
}

typedef struct {
    const char *label;
    double physical;
    double code_expected;
    double kernel_used;
    double ratio;
} UnitConversionRow;

typedef struct {
    double w_ref_phys;
    double dx_ref_m;
    double t0_diff_s;
    double mu_reference_scale;
    double kmax2;
    double S_phi_W;
    double S_phi_grad;
    double S_phi_total;
    double S_eta_W;
    double S_eta_grad;
    double S_eta_total;
    double CFL_diff;
    UnitConversionRow rows[6];
} UnitConversionSummary;

static double resolve_or_derive_physical_value(double physical_input,
                                               double kernel_used,
                                               double scale,
                                               const char *label) {
    if (isfinite(physical_input) && physical_input > 0.0) {
        return physical_input;
    }
    if (isfinite(kernel_used) && isfinite(scale) && scale > 0.0) {
        fprintf(stderr, "[warn] %s physical value missing; deriving from code value.\n", label);
        return kernel_used / scale;
    }
    return NAN;
}

static int compute_unit_conversion_summary(const PFParams *P, UnitConversionSummary *S) {
    if (!P || !S) return 0;
    memset(S, 0, sizeof(*S));
    S->w_ref_phys = compute_eta_ref_w_phys_host(P);
    S->dx_ref_m = compute_eta_ref_dx_phys_m_host(P);
    S->t0_diff_s = P->t_real_unit;
    S->mu_reference_scale = P->mu_reference_scale;
    if (!(isfinite(S->w_ref_phys) && S->w_ref_phys > 0.0 &&
          isfinite(S->dx_ref_m) && S->dx_ref_m > 0.0 &&
          isfinite(S->t0_diff_s) && S->t0_diff_s > 0.0)) {
        fprintf(stderr, "[fatal] cannot compute unit conversion summary: invalid reference scales\n");
        return 0;
    }

    const double kmax2 =
        pow(M_PI / fmax(P->dx, 1.0e-30), 2.0) +
        pow(M_PI / fmax(P->dy, 1.0e-30), 2.0) +
        pow(M_PI / fmax(P->dz, 1.0e-30), 2.0);
    S->kmax2 = kmax2;

    const double phi_W_phys = 12.0 * P->gamma_Jm2 / P->lambda_sm_m;
    const double phi_kappa_phys = 1.5 * P->gamma_Jm2 * P->lambda_sm_m;
    const double phi_L_phys = (isfinite(P->L_phi) && P->L_phi > 0.0)
        ? (P->L_phi / (S->w_ref_phys * S->t0_diff_s))
        : NAN;

    const double eta_W_phys = resolve_or_derive_physical_value(P->gp_W_eta_phys_input, P->gp_W_eta, S->w_ref_phys, "gp_W_eta");
    const double eta_kappa_phys = resolve_or_derive_physical_value(P->gp_kappa_eta_phys_input, P->gp_kappa_eta, S->w_ref_phys * S->dx_ref_m * S->dx_ref_m, "gp_kappa_eta");
    const double eta_L_phys = resolve_or_derive_physical_value(P->gp_L_eta_phys_input, P->gp_L_eta, S->w_ref_phys * S->t0_diff_s, "gp_L_eta");

    S->rows[0] = (UnitConversionRow){
        "W_phi",
        phi_W_phys,
        phi_W_phys / S->w_ref_phys,
        P->W,
        NAN,
    };
    S->rows[1] = (UnitConversionRow){
        "kappa_phi",
        phi_kappa_phys,
        phi_kappa_phys / (S->w_ref_phys * S->dx_ref_m * S->dx_ref_m),
        P->kappa_phi,
        NAN,
    };
    S->rows[2] = (UnitConversionRow){
        "L_phi",
        phi_L_phys,
        isfinite(phi_L_phys) ? (phi_L_phys * S->w_ref_phys * S->t0_diff_s) : NAN,
        P->L_phi,
        NAN,
    };
    S->rows[3] = (UnitConversionRow){
        "W_eta",
        eta_W_phys,
        isfinite(eta_W_phys) ? (eta_W_phys / S->w_ref_phys) : NAN,
        P->gp_W_eta,
        NAN,
    };
    S->rows[4] = (UnitConversionRow){
        "kappa_eta",
        eta_kappa_phys,
        isfinite(eta_kappa_phys) ? (eta_kappa_phys / (S->w_ref_phys * S->dx_ref_m * S->dx_ref_m)) : NAN,
        P->gp_kappa_eta,
        NAN,
    };
    S->rows[5] = (UnitConversionRow){
        "L_eta",
        eta_L_phys,
        isfinite(eta_L_phys) ? (eta_L_phys * S->w_ref_phys * S->t0_diff_s) : NAN,
        P->gp_L_eta,
        NAN,
    };

    for (int i = 0; i < 6; ++i) {
        const double expected = S->rows[i].code_expected;
        const double used = S->rows[i].kernel_used;
        if (!(isfinite(expected) && fabs(expected) > 1.0e-30 &&
              isfinite(used) && isfinite(S->rows[i].physical) && S->rows[i].physical > 0.0)) {
            fprintf(stderr, "[fatal] unit conversion summary row '%s' is invalid\n", S->rows[i].label);
            return 0;
        }
        S->rows[i].ratio = used / expected;
        const double rel_err = fabs(S->rows[i].ratio - 1.0);
        if (rel_err > 1.0e-4) {
            fprintf(stderr,
                    "[fatal] %s code/phys mismatch: physical=%.8e expected_code=%.8e kernel_used=%.8e ratio=%.8e\n",
                    S->rows[i].label, S->rows[i].physical, expected, used, S->rows[i].ratio);
            return 0;
        }
        if (rel_err > 1.0e-6) {
            fprintf(stderr,
                    "[warn] %s code/phys mismatch exceeds 1e-6: ratio=%.8e\n",
                    S->rows[i].label, S->rows[i].ratio);
        }
    }

    S->S_phi_W = P->L_phi * P->dt * P->W;
    S->S_phi_grad = P->L_phi * P->dt * P->kappa_phi * kmax2;
    S->S_phi_total = S->S_phi_W + S->S_phi_grad;
    S->S_eta_W = P->gp_L_eta * P->dt * P->gp_W_eta;
    S->S_eta_grad = P->gp_L_eta * P->dt * P->gp_kappa_eta * kmax2;
    S->S_eta_total = S->S_eta_W + S->S_eta_grad;
    S->CFL_diff = (isfinite(P->D_alpha) && P->D_alpha > 0.0 &&
                   isfinite(P->dt) && P->dt > 0.0)
                      ? (P->D_alpha * P->dt / (fmin(P->dx, fmin(P->dy, P->dz)) * fmin(P->dx, fmin(P->dy, P->dz))))
                      : NAN;
    return 1;
}

static int print_unit_conversion_summary_table(const PFParams *P) {
    UnitConversionSummary S;
    if (!compute_unit_conversion_summary(P, &S)) return 0;

    printf("\n[Unit Conversion Summary]\n");
    printf("  Reference scales:\n");
    printf("    w_ref (J/m^3)      = %.8e\n", S.w_ref_phys);
    printf("    dx_ref (m)         = %.8e\n", S.dx_ref_m);
    printf("    t0_diff (s)        = %.8e\n", S.t0_diff_s);
    printf("    mu_reference_scale = %.8e\n", S.mu_reference_scale);
    printf("\n");
    printf("                       physical            code               kernel_used        ratio(kernel/expected)\n");
    for (int i = 0; i < 6; ++i) {
        printf("  %-18s %-18.8e %-18.8e %-18.8e %-18.8e\n",
               S.rows[i].label, S.rows[i].physical, S.rows[i].code_expected, S.rows[i].kernel_used, S.rows[i].ratio);
    }
    printf("\n");
    printf("  Stability indicators (S_W = L·dt·W; S_grad = L·dt·κ·k_max²):\n");
    printf("    S_phi_W    = %.8e     S_phi_grad = %.8e     S_phi_total = %.8e\n",
           S.S_phi_W, S.S_phi_grad, S.S_phi_total);
    printf("    S_eta_W    = %.8e     S_eta_grad = %.8e     S_eta_total = %.8e\n",
           S.S_eta_W, S.S_eta_grad, S.S_eta_total);
    printf("    CFL_diff   = %.8e\n", S.CFL_diff);
    printf("    Note: semi-implicit denom damps S_grad; instability risk comes from explicit source term S_W. Production-safe envelope: S_W < 1.\n");
    return 1;
}

static int write_unit_conversion_summary_csv(const char *case_output_dir, const PFParams *P) {
    if (!case_output_dir || !case_output_dir[0]) return 1;
    UnitConversionSummary S;
    if (!compute_unit_conversion_summary(P, &S)) return 0;
    char path[4096];
    snprintf(path, sizeof(path), "%s/unit_conversion_summary.csv", case_output_dir);
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "[warn] cannot open unit conversion summary CSV: %s\n", path);
        return 0;
    }
    fprintf(fp, "label,physical,code_expected,kernel_used,ratio\n");
    for (int i = 0; i < 6; ++i) {
        fprintf(fp, "%s,%.10e,%.10e,%.10e,%.10e\n",
                S.rows[i].label, S.rows[i].physical, S.rows[i].code_expected, S.rows[i].kernel_used, S.rows[i].ratio);
    }
    fprintf(fp, "S_phi_W,,,,%.10e\n", S.S_phi_W);
    fprintf(fp, "S_phi_grad,,,,%.10e\n", S.S_phi_grad);
    fprintf(fp, "S_phi_total,,,,%.10e\n", S.S_phi_total);
    fprintf(fp, "S_eta_W,,,,%.10e\n", S.S_eta_W);
    fprintf(fp, "S_eta_grad,,,,%.10e\n", S.S_eta_grad);
    fprintf(fp, "S_eta_total,,,,%.10e\n", S.S_eta_total);
    fprintf(fp, "CFL_diff,,,,%.10e\n", S.CFL_diff);
    fclose(fp);
    return 1;
}

static double compute_delta_g_nuc_GP_host(double xB_alpha_local,
                                          double temperature_K,
                                          double mu_reference_scale,
                                          double xB_GP_fixed) {
    double x = clamp_fraction_eps(xB_alpha_local);
    double x_gp = clamp_fraction_eps(xB_GP_fixed);
    (void)mu_reference_scale;
    double muA = mu_PbTe_raw(temperature_K, x);
    double muB = mu_Ag2Te_raw(temperature_K, x);
    double g_alpha = (1.0 - x) * muA + x * muB;
    double mu_exchange_alpha = muB - muA;
    double g_gp = compute_g_alpha_host(x_gp, temperature_K, mu_reference_scale);
    return g_gp - (g_alpha + (x_gp - x) * mu_exchange_alpha);
}

static double compute_local_Vm_alpha_phys_host(double xB_alpha, const PFParams *P) {
    double Vm_ratio = 1.0;
    if (fabs(P->Vm_alpha_0) > 1.0e-30) {
        Vm_ratio = Vm_alpha_of_xB(clamp_fraction_eps(xB_alpha), P->Vm_alpha_0, P->dVm_alpha_dxB) / P->Vm_alpha_0;
    }
    return fmax(P->Vm_alpha_0_phys_m3mol * Vm_ratio, 1.0e-30);
}

static inline uint64_t splitmix64_host(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static double uniform01_from_key_host(uint64_t key) {
    const uint64_t bits = splitmix64_host(key);
    return ((bits >> 11) * (1.0 / 9007199254740992.0));
}

static double h_inverse_bisection_host(double target) {
    if (target <= 0.0) return 0.0;
    if (target >= 1.0) return 1.0;
    double lo = 0.0;
    double hi = 1.0;
    for (int iter = 0; iter < 80; ++iter) {
        double mid = 0.5 * (lo + hi);
        double h_mid = h_of_phi(mid);
        if (h_mid < target) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
}

static void init_gp_storage_stats_host(double *stats) {
    if (!stats) return;
    for (int i = 0; i < MASS_DIAG_GP_STORAGE_STATS_COUNT; ++i) stats[i] = 0.0;
    stats[MASS_DIAG_GP_STORAGE_MIN_HALPHA] = 1.0e300;
    stats[MASS_DIAG_GP_STORAGE_MIN_HGP] = 1.0e300;
    stats[MASS_DIAG_GP_STORAGE_MIN_ETA] = 1.0e300;
    stats[MASS_DIAG_GP_STORAGE_MIN_XB_ALPHA] = 1.0e300;
    stats[MASS_DIAG_GP_STORAGE_MAX_HALPHA] = -1.0e300;
    stats[MASS_DIAG_GP_STORAGE_MAX_HGP] = -1.0e300;
    stats[MASS_DIAG_GP_STORAGE_MAX_ETA] = -1.0e300;
    stats[MASS_DIAG_GP_STORAGE_MAX_XB_ALPHA] = -1.0e300;
}

static void init_gp_eta_feasibility_stats_host(double *stats) {
    if (!stats) return;
    for (int i = 0; i < GP_ETA_FEAS_STATS_COUNT; ++i) stats[i] = 0.0;
    stats[GP_ETA_FEAS_MIN_RECOVERED_XB_BEFORE] = 1.0e300;
    stats[GP_ETA_FEAS_MAX_RECOVERED_XB_BEFORE] = -1.0e300;
    stats[GP_ETA_FEAS_MIN_RECOVERED_XB_AFTER] = 1.0e300;
    stats[GP_ETA_FEAS_MAX_RECOVERED_XB_AFTER] = -1.0e300;
    stats[GP_ETA_FEAS_MAX_ETA_TENT] = -1.0e300;
    stats[GP_ETA_FEAS_MAX_HGP_NEW] = -1.0e300;
    stats[GP_ETA_FEAS_MAX_ETA_NEW] = -1.0e300;
}

static void init_gp_transport_stats_host(double *stats) {
    if (!stats) return;
    for (int i = 0; i < GP_TRANSPORT_STATS_COUNT; ++i) stats[i] = 0.0;
    stats[GP_TRANSPORT_MIN_HALPHA] = 1.0e300;
    stats[GP_TRANSPORT_MAX_HALPHA] = -1.0e300;
    stats[GP_TRANSPORT_MIN_M_ALPHA] = 1.0e300;
    stats[GP_TRANSPORT_MAX_M_ALPHA] = -1.0e300;
    stats[GP_TRANSPORT_MIN_M_EFF] = 1.0e300;
    stats[GP_TRANSPORT_MAX_M_EFF] = -1.0e300;
}

static void init_gp_elastic_stats_host(double *stats) {
    if (!stats) return;
    for (int i = 0; i < GP_ELASTIC_STATS_COUNT; ++i) stats[i] = 0.0;
    stats[GP_ELASTIC_STATS_MIN_SIGMA_HYDRO] = 1.0e300;
    stats[GP_ELASTIC_STATS_MAX_SIGMA_HYDRO] = -1.0e300;
    stats[GP_ELASTIC_STATS_MIN_EPS0_GP_DIAG] = 1.0e300;
    stats[GP_ELASTIC_STATS_MAX_EPS0_GP_DIAG] = -1.0e300;
    stats[GP_ELASTIC_STATS_MIN_MINUS_DELTA_MU_R_GP] = 1.0e300;
    stats[GP_ELASTIC_STATS_MAX_MINUS_DELTA_MU_R_GP] = -1.0e300;
}

static void init_mass_diag_y_rhs_stats_host(double *stats) {
    if (!stats) return;
    for (int i = 0; i < MASS_DIAG_Y_RHS_STATS_COUNT; ++i) stats[i] = 0.0;
    stats[MASS_DIAG_Y_RHS_MIN_H] = 1.0e300;
    stats[MASS_DIAG_Y_RHS_MIN_HQ] = 1.0e300;
    stats[MASS_DIAG_Y_RHS_MAX_HQ] = -1.0e300;
}

static void compute_scalar_field_stats(const double *d_array,
                                       int total_size,
                                       double *mean_val,
                                       double *min_val,
                                       double *max_val) {
    if (mean_val) *mean_val = 0.0;
    if (min_val) *min_val = 0.0;
    if (max_val) *max_val = 0.0;
    if (!d_array || total_size <= 0) return;
    double sum = gpu_reduce_sum(d_array, total_size);
    double min_local = 0.0;
    double max_local = 0.0;
    gpu_reduce_min_max(d_array, total_size, &min_local, &max_local);
    if (mean_val) *mean_val = sum / (double)total_size;
    if (min_val) *min_val = min_local;
    if (max_val) *max_val = max_local;
}

typedef struct {
    double min_before;
    double max_before;
    double mean_before;
    double min_after;
    double max_after;
    double mean_after;
    double delta_min;
    double delta_max;
    double delta_mean;
    double delta_abs_mean;
    double delta_abs_max;
    double delta_l2;
    double frac_abs_gt_1e12;
    double frac_abs_gt_1e8;
    double frac_abs_gt_1e5;
} StepDeltaFieldStats;

typedef struct {
    int step;
    double time_code;
    double dt;
    StepDeltaFieldStats phi;
    StepDeltaFieldStats eta;
    StepDeltaFieldStats xB;
    double Y_min_before;
    double Y_max_before;
    double Y_mean_before;
    double Y_min_after;
    double Y_max_after;
    double Y_mean_after;
    double dY_abs_mean;
    double dY_abs_max;
    double xBtot_mean_before;
    double xBtot_mean_after;
    double xBtot_delta;
    double xBtot_rel_delta;
    long long xB_clip_low_count;
    long long xB_clip_high_count;
    long long phi_out_of_bounds_count;
    long long eta_out_of_bounds_count;
    int nan_inf_flag;
    double eta_far_field_mean;
    double eta_far_field_max;
    double phi_far_field_mean;
    double phi_far_field_max;
    double eta_max_minus_far;
    double phi_max_minus_far;
    double deta_abs_mean_over_dphi_abs_mean;
    double deta_abs_max_over_dphi_abs_max;
    double deta_l2_over_dphi_l2;
    double S_phi_W;
    double S_phi_grad;
    double S_phi_total;
    double S_eta_W;
    double S_eta_grad;
    double S_eta_total;
    double gp_L_eta;
    double L_phi;
} PhiEtaStepDeltaRow;

static void compute_host_field_delta_stats(const double *before,
                                           const double *after,
                                           int total_size,
                                           StepDeltaFieldStats *out,
                                           int *nan_inf_flag) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!before || !after || total_size <= 0) return;
    out->min_before = out->max_before = before[0];
    out->min_after = out->max_after = after[0];
    double sum_before = 0.0;
    double sum_after = 0.0;
    double sum_delta = 0.0;
    double sum_abs_delta = 0.0;
    double sum_sq_delta = 0.0;
    long long count_gt_1e12 = 0;
    long long count_gt_1e8 = 0;
    long long count_gt_1e5 = 0;
    for (int idx = 0; idx < total_size; ++idx) {
        const double b = before[idx];
        const double a = after[idx];
        if (!isfinite(b) || !isfinite(a)) {
            if (nan_inf_flag) *nan_inf_flag = 1;
            continue;
        }
        out->min_before = fmin(out->min_before, b);
        out->max_before = fmax(out->max_before, b);
        out->min_after = fmin(out->min_after, a);
        out->max_after = fmax(out->max_after, a);
        sum_before += b;
        sum_after += a;
        const double d = a - b;
        const double ad = fabs(d);
        if (idx == 0) {
            out->delta_min = d;
            out->delta_max = d;
            out->delta_abs_max = ad;
        } else {
            out->delta_min = fmin(out->delta_min, d);
            out->delta_max = fmax(out->delta_max, d);
            out->delta_abs_max = fmax(out->delta_abs_max, ad);
        }
        sum_delta += d;
        sum_abs_delta += ad;
        sum_sq_delta += d * d;
        if (ad > 1.0e-12) ++count_gt_1e12;
        if (ad > 1.0e-8) ++count_gt_1e8;
        if (ad > 1.0e-5) ++count_gt_1e5;
    }
    const double denom = fmax((double)total_size, 1.0);
    out->mean_before = sum_before / denom;
    out->mean_after = sum_after / denom;
    out->delta_mean = sum_delta / denom;
    out->delta_abs_mean = sum_abs_delta / denom;
    out->delta_l2 = sqrt(sum_sq_delta / denom);
    out->frac_abs_gt_1e12 = (double)count_gt_1e12 / denom;
    out->frac_abs_gt_1e8 = (double)count_gt_1e8 / denom;
    out->frac_abs_gt_1e5 = (double)count_gt_1e5 / denom;
}

static void compute_host_field_basic_stats(const double *field,
                                           int total_size,
                                           double *mean_val,
                                           double *min_val,
                                           double *max_val,
                                           int *nan_inf_flag) {
    if (mean_val) *mean_val = 0.0;
    if (min_val) *min_val = 0.0;
    if (max_val) *max_val = 0.0;
    if (!field || total_size <= 0) return;
    double sum = 0.0;
    double min_local = field[0];
    double max_local = field[0];
    for (int idx = 0; idx < total_size; ++idx) {
        const double v = field[idx];
        if (!isfinite(v)) {
            if (nan_inf_flag) *nan_inf_flag = 1;
            continue;
        }
        min_local = fmin(min_local, v);
        max_local = fmax(max_local, v);
        sum += v;
    }
    const double denom = fmax((double)total_size, 1.0);
    if (mean_val) *mean_val = sum / denom;
    if (min_val) *min_val = min_local;
    if (max_val) *max_val = max_local;
}

static int host_argmax_field(const double *field, int total_size, int *nan_inf_flag) {
    if (!field || total_size <= 0) return 0;
    int best = 0;
    double best_val = field[0];
    for (int idx = 0; idx < total_size; ++idx) {
        const double v = field[idx];
        if (!isfinite(v)) {
            if (nan_inf_flag) *nan_inf_flag = 1;
            continue;
        }
        if (v > best_val) {
            best_val = v;
            best = idx;
        }
    }
    return best;
}

static void compute_far_field_stats_host(const double *field,
                                         int total_size,
                                         const PFParams *P,
                                         int center_idx,
                                         double *mean_val,
                                         double *max_val,
                                         int *nan_inf_flag) {
    if (mean_val) *mean_val = 0.0;
    if (max_val) *max_val = 0.0;
    if (!field || !P || total_size <= 0) return;
    int ic = 0, jc = 0, kc = 0;
    host_decode_index(center_idx, P->Ny, P->Nz, &ic, &jc, &kc);
    const double cx = ic * P->dx;
    const double cy = jc * P->dy;
    const double cz = kc * P->dz;
    const double Lx = P->Nx * P->dx;
    const double Ly = P->Ny * P->dy;
    const double Lz = P->Nz * P->dz;
    const double minL = fmin(Lx, fmin(Ly, Lz));
    const double far_r = 0.30 * minL;
    double sum = 0.0;
    double max_local = -1.0e300;
    long long count = 0;
    for (int idx = 0; idx < total_size; ++idx) {
        int i = 0, j = 0, k = 0;
        host_decode_index(idx, P->Ny, P->Nz, &i, &j, &k);
        const double x = i * P->dx;
        const double y = j * P->dy;
        const double z = k * P->dz;
        const double rx = periodic_delta(x, cx, Lx);
        const double ry = periodic_delta(y, cy, Ly);
        const double rz = periodic_delta(z, cz, Lz);
        const double r = sqrt(rx * rx + ry * ry + rz * rz);
        if (r < far_r) continue;
        const double v = field[idx];
        if (!isfinite(v)) {
            if (nan_inf_flag) *nan_inf_flag = 1;
            continue;
        }
        sum += v;
        max_local = fmax(max_local, v);
        ++count;
    }
    if (count <= 0) {
        if (mean_val) *mean_val = NAN;
        if (max_val) *max_val = NAN;
        return;
    }
    if (mean_val) *mean_val = sum / (double)count;
    if (max_val) *max_val = max_local;
}

static void write_phi_eta_step_delta_csv_header(FILE *fp) {
    if (!fp) return;
    fprintf(fp,
            "step,time_code,dt,"
            "phi_min_before,phi_max_before,phi_mean_before,phi_min_after,phi_max_after,phi_mean_after,"
            "dphi_min,dphi_max,dphi_mean,dphi_abs_mean,dphi_abs_max,dphi_l2,"
            "dphi_nonzero_fraction_abs_gt_1e-12,dphi_nonzero_fraction_abs_gt_1e-8,dphi_nonzero_fraction_abs_gt_1e-5,"
            "eta_min_before,eta_max_before,eta_mean_before,eta_min_after,eta_max_after,eta_mean_after,"
            "deta_min,deta_max,deta_mean,deta_abs_mean,deta_abs_max,deta_l2,"
            "deta_nonzero_fraction_abs_gt_1e-12,deta_nonzero_fraction_abs_gt_1e-8,deta_nonzero_fraction_abs_gt_1e-5,"
            "deta_abs_mean_over_dphi_abs_mean,deta_abs_max_over_dphi_abs_max,deta_l2_over_dphi_l2,"
            "xB_min_before,xB_max_before,xB_mean_before,xB_min_after,xB_max_after,xB_mean_after,"
            "dxB_min,dxB_max,dxB_abs_mean,dxB_abs_max,"
            "Y_min_before,Y_max_before,Y_mean_before,Y_min_after,Y_max_after,Y_mean_after,dY_abs_mean,dY_abs_max,"
            "xBtot_mean_before,xBtot_mean_after,xBtot_delta,xBtot_rel_delta,"
            "xB_clip_low_count,xB_clip_high_count,phi_out_of_bounds_count,eta_out_of_bounds_count,nan_inf_flag,"
            "eta_far_field_mean,eta_far_field_max,phi_far_field_mean,phi_far_field_max,eta_max_minus_far,phi_max_minus_far,"
            "S_phi_W,S_phi_grad,S_phi_total,S_eta_W,S_eta_grad,S_eta_total,gp_L_eta,L_phi\n");
}

static void write_phi_eta_step_delta_csv_row(FILE *fp,
                                             const PhiEtaStepDeltaRow *r) {
    if (!fp || !r) return;
#define CSV_PD(v) fprintf(fp, "%.10e", (double)(v))
#define CSV_PI(v) fprintf(fp, "%d", (int)(v))
#define CSV_PLL(v) fprintf(fp, "%lld", (long long)(v))
#define CSV_PC() fputc(',', fp)
    CSV_PI(r->step); CSV_PC();
    CSV_PD(r->time_code); CSV_PC();
    CSV_PD(r->dt); CSV_PC();
    CSV_PD(r->phi.min_before); CSV_PC();
    CSV_PD(r->phi.max_before); CSV_PC();
    CSV_PD(r->phi.mean_before); CSV_PC();
    CSV_PD(r->phi.min_after); CSV_PC();
    CSV_PD(r->phi.max_after); CSV_PC();
    CSV_PD(r->phi.mean_after); CSV_PC();
    CSV_PD(r->phi.delta_min); CSV_PC();
    CSV_PD(r->phi.delta_max); CSV_PC();
    CSV_PD(r->phi.delta_mean); CSV_PC();
    CSV_PD(r->phi.delta_abs_mean); CSV_PC();
    CSV_PD(r->phi.delta_abs_max); CSV_PC();
    CSV_PD(r->phi.delta_l2); CSV_PC();
    CSV_PD(r->phi.frac_abs_gt_1e12); CSV_PC();
    CSV_PD(r->phi.frac_abs_gt_1e8); CSV_PC();
    CSV_PD(r->phi.frac_abs_gt_1e5); CSV_PC();
    CSV_PD(r->eta.min_before); CSV_PC();
    CSV_PD(r->eta.max_before); CSV_PC();
    CSV_PD(r->eta.mean_before); CSV_PC();
    CSV_PD(r->eta.min_after); CSV_PC();
    CSV_PD(r->eta.max_after); CSV_PC();
    CSV_PD(r->eta.mean_after); CSV_PC();
    CSV_PD(r->eta.delta_min); CSV_PC();
    CSV_PD(r->eta.delta_max); CSV_PC();
    CSV_PD(r->eta.delta_mean); CSV_PC();
    CSV_PD(r->eta.delta_abs_mean); CSV_PC();
    CSV_PD(r->eta.delta_abs_max); CSV_PC();
    CSV_PD(r->eta.delta_l2); CSV_PC();
    CSV_PD(r->eta.frac_abs_gt_1e12); CSV_PC();
    CSV_PD(r->eta.frac_abs_gt_1e8); CSV_PC();
    CSV_PD(r->eta.frac_abs_gt_1e5); CSV_PC();
    CSV_PD(r->deta_abs_mean_over_dphi_abs_mean); CSV_PC();
    CSV_PD(r->deta_abs_max_over_dphi_abs_max); CSV_PC();
    CSV_PD(r->deta_l2_over_dphi_l2); CSV_PC();
    CSV_PD(r->xB.min_before); CSV_PC();
    CSV_PD(r->xB.max_before); CSV_PC();
    CSV_PD(r->xB.mean_before); CSV_PC();
    CSV_PD(r->xB.min_after); CSV_PC();
    CSV_PD(r->xB.max_after); CSV_PC();
    CSV_PD(r->xB.mean_after); CSV_PC();
    CSV_PD(r->xB.delta_min); CSV_PC();
    CSV_PD(r->xB.delta_max); CSV_PC();
    CSV_PD(r->xB.delta_abs_mean); CSV_PC();
    CSV_PD(r->xB.delta_abs_max); CSV_PC();
    CSV_PD(r->Y_min_before); CSV_PC();
    CSV_PD(r->Y_max_before); CSV_PC();
    CSV_PD(r->Y_mean_before); CSV_PC();
    CSV_PD(r->Y_min_after); CSV_PC();
    CSV_PD(r->Y_max_after); CSV_PC();
    CSV_PD(r->Y_mean_after); CSV_PC();
    CSV_PD(r->dY_abs_mean); CSV_PC();
    CSV_PD(r->dY_abs_max); CSV_PC();
    CSV_PD(r->xBtot_mean_before); CSV_PC();
    CSV_PD(r->xBtot_mean_after); CSV_PC();
    CSV_PD(r->xBtot_delta); CSV_PC();
    CSV_PD(r->xBtot_rel_delta); CSV_PC();
    CSV_PLL(r->xB_clip_low_count); CSV_PC();
    CSV_PLL(r->xB_clip_high_count); CSV_PC();
    CSV_PLL(r->phi_out_of_bounds_count); CSV_PC();
    CSV_PLL(r->eta_out_of_bounds_count); CSV_PC();
    CSV_PI(r->nan_inf_flag); CSV_PC();
    CSV_PD(r->eta_far_field_mean); CSV_PC();
    CSV_PD(r->eta_far_field_max); CSV_PC();
    CSV_PD(r->phi_far_field_mean); CSV_PC();
    CSV_PD(r->phi_far_field_max); CSV_PC();
    CSV_PD(r->eta_max_minus_far); CSV_PC();
    CSV_PD(r->phi_max_minus_far); CSV_PC();
    CSV_PD(r->S_phi_W); CSV_PC();
    CSV_PD(r->S_phi_grad); CSV_PC();
    CSV_PD(r->S_phi_total); CSV_PC();
    CSV_PD(r->S_eta_W); CSV_PC();
    CSV_PD(r->S_eta_grad); CSV_PC();
    CSV_PD(r->S_eta_total); CSV_PC();
    CSV_PD(r->gp_L_eta); CSV_PC();
    CSV_PD(r->L_phi);
    fputc('\n', fp);
#undef CSV_PD
#undef CSV_PI
#undef CSV_PLL
#undef CSV_PC
}

typedef struct {
    double signed_mean;
    double abs_mean;
    double abs_max;
    double rms;
} AttributionFieldStats;

typedef struct {
    int step;
    double time_code;
    double dt;
    AttributionFieldStats phi_rhs_total_explicit;
    AttributionFieldStats phi_rhs_chem;
    AttributionFieldStats phi_rhs_double_well;
    AttributionFieldStats phi_rhs_elastic;
    AttributionFieldStats phi_update_explicit_est;
    AttributionFieldStats dphi_actual;
    AttributionFieldStats eta_rhs_total_explicit;
    AttributionFieldStats eta_rhs_bulk_or_chemical;
    AttributionFieldStats eta_rhs_double_well;
    AttributionFieldStats eta_rhs_elastic;
    AttributionFieldStats eta_update_explicit_est;
    AttributionFieldStats deta_actual;
    double phi_grad_denom_min;
    double phi_grad_denom_max;
    double phi_grad_denom_mean;
    double phi_grad_denom_at_kmax;
    double eta_grad_denom_min;
    double eta_grad_denom_max;
    double eta_grad_denom_mean;
    double eta_grad_denom_at_kmax;
    double eta_rhs_total_absmax_over_phi_rhs_total_absmax;
    double eta_rhs_bulk_absmax_over_phi_chem_absmax;
    double eta_rhs_dw_absmax_over_phi_dw_absmax;
    double eta_rhs_elastic_absmax_over_phi_elastic_absmax;
    double eta_Ldt_rhs_absmax_over_phi_Ldt_rhs_absmax;
    double eta_Ldt_bulk_absmax_over_phi_Ldt_chem_absmax;
    double eta_Ldt_dw_absmax_over_phi_Ldt_dw_absmax;
    double eta_Ldt_elastic_absmax_over_phi_Ldt_elastic_absmax;
    double deta_absmax_over_dphi_absmax;
    double deta_absmean_over_dphi_absmean;
    double deta_rms_over_dphi_rms;
    double phi_damping_stronger_than_eta;
    double phi_actual_over_explicit_est_absmax;
    double eta_actual_over_explicit_est_absmax;
    double corr_abs_deta_abs_dxB;
    double corr_abs_deta_abs_dY;
    double corr_abs_deta_local_xB;
    double corr_abs_deta_eta_before;
    double corr_abs_deta_phi_before;
    int max_abs_deta_idx;
    int max_abs_deta_i;
    int max_abs_deta_j;
    int max_abs_deta_k;
    double max_abs_deta_value;
    double max_abs_deta_eta_before;
    double max_abs_deta_phi_before;
    double max_abs_deta_xB_before;
    double max_abs_deta_Y_before;
    double max_abs_deta_eta_rhs_bulk;
    double max_abs_deta_eta_rhs_dw;
    double max_abs_deta_eta_rhs_total;
    double max_abs_deta_dxB;
    int max_abs_dphi_idx;
    int max_abs_dphi_i;
    int max_abs_dphi_j;
    int max_abs_dphi_k;
    double max_abs_dphi_value;
    double max_abs_dphi_eta_before;
    double max_abs_dphi_phi_before;
    double max_abs_dphi_xB_before;
    double max_abs_dphi_Y_before;
    double max_abs_dphi_phi_rhs_chem;
    double max_abs_dphi_phi_rhs_dw;
    double max_abs_dphi_phi_rhs_total;
    double max_abs_dphi_dxB;
    double xB_min_after;
    double xB_max_after;
    double xBtot_rel_delta;
    double eta_far_field_max;
    long long xB_clip_low_count;
    long long xB_clip_high_count;
    int nan_inf_flag;
    double S_phi_W;
    double S_phi_grad;
    double S_phi_total;
    double S_eta_W;
    double S_eta_grad;
    double S_eta_total;
    double gp_L_eta;
    double L_phi;
} PhiEtaRHSAttributionRow;

static void compute_attribution_field_stats_host(const double *field,
                                                 int total_size,
                                                 AttributionFieldStats *out,
                                                 int *nan_inf_flag) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!field || total_size <= 0) return;
    double sum = 0.0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double max_abs = 0.0;
    for (int idx = 0; idx < total_size; ++idx) {
        const double v = field[idx];
        if (!isfinite(v)) {
            if (nan_inf_flag) *nan_inf_flag = 1;
            continue;
        }
        const double av = fabs(v);
        sum += v;
        sum_abs += av;
        sum_sq += v * v;
        max_abs = fmax(max_abs, av);
    }
    const double denom = fmax((double)total_size, 1.0);
    out->signed_mean = sum / denom;
    out->abs_mean = sum_abs / denom;
    out->abs_max = max_abs;
    out->rms = sqrt(sum_sq / denom);
}

static double compute_corr_host(const double *a,
                                const double *b,
                                int total_size,
                                int *nan_inf_flag) {
    if (!a || !b || total_size <= 1) return NAN;
    double mean_a = 0.0, mean_b = 0.0;
    long long n = 0;
    for (int idx = 0; idx < total_size; ++idx) {
        const double va = a[idx];
        const double vb = b[idx];
        if (!isfinite(va) || !isfinite(vb)) {
            if (nan_inf_flag) *nan_inf_flag = 1;
            continue;
        }
        mean_a += va;
        mean_b += vb;
        ++n;
    }
    if (n <= 1) return NAN;
    mean_a /= (double)n;
    mean_b /= (double)n;
    double cov = 0.0, var_a = 0.0, var_b = 0.0;
    for (int idx = 0; idx < total_size; ++idx) {
        const double va = a[idx];
        const double vb = b[idx];
        if (!isfinite(va) || !isfinite(vb)) continue;
        const double da = va - mean_a;
        const double db = vb - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    if (var_a <= 1.0e-300 || var_b <= 1.0e-300) return NAN;
    return cov / sqrt(var_a * var_b);
}

static int host_argmax_abs_delta_field(const double *before,
                                       const double *after,
                                       int total_size,
                                       double *value_out,
                                       int *nan_inf_flag) {
    if (!before || !after || total_size <= 0) return 0;
    int best = 0;
    double best_abs = -1.0;
    double best_val = 0.0;
    for (int idx = 0; idx < total_size; ++idx) {
        const double d = after[idx] - before[idx];
        if (!isfinite(d)) {
            if (nan_inf_flag) *nan_inf_flag = 1;
            continue;
        }
        const double ad = fabs(d);
        if (ad > best_abs) {
            best_abs = ad;
            best = idx;
            best_val = d;
        }
    }
    if (value_out) *value_out = best_val;
    return best;
}

static int host_argmax_abs_field(const double *field,
                                 int total_size,
                                 double *value_out,
                                 int *nan_inf_flag) {
    if (!field || total_size <= 0) return 0;
    int best = 0;
    double best_abs = -1.0;
    double best_val = 0.0;
    for (int idx = 0; idx < total_size; ++idx) {
        const double v = field[idx];
        if (!isfinite(v)) {
            if (nan_inf_flag) *nan_inf_flag = 1;
            continue;
        }
        const double av = fabs(v);
        if (av > best_abs) {
            best_abs = av;
            best = idx;
            best_val = v;
        }
    }
    if (value_out) *value_out = best_val;
    return best;
}

static void linear_index_to_ijk_host(const PFParams *P,
                                     int idx,
                                     int *i_out,
                                     int *j_out,
                                     int *k_out) {
    if (i_out) *i_out = 0;
    if (j_out) *j_out = 0;
    if (k_out) *k_out = 0;
    if (!P || idx < 0 || P->Ny <= 0 || P->Nz <= 0) return;
    const int k = idx % P->Nz;
    const int rem = idx / P->Nz;
    const int j = rem % P->Ny;
    const int i = rem / P->Ny;
    if (i_out) *i_out = i;
    if (j_out) *j_out = j;
    if (k_out) *k_out = k;
}

static void compute_fourier_denom_stats_host(const PFParams *P,
                                             double mobility,
                                             double kappa,
                                             double dt,
                                             double *min_val,
                                             double *max_val,
                                             double *mean_val,
                                             double *at_kmax) {
    if (min_val) *min_val = NAN;
    if (max_val) *max_val = NAN;
    if (mean_val) *mean_val = NAN;
    if (at_kmax) *at_kmax = NAN;
    if (!P || P->Nx <= 0 || P->Ny <= 0 || P->Nz <= 0) return;
    const int NzC = P->Nz / 2 + 1;
    double min_d = 1.0e300;
    double max_d = -1.0e300;
    double sum_d = 0.0;
    long long count = 0;
    for (int i = 0; i < P->Nx; ++i) {
        const double kx = (i <= P->Nx / 2)
                              ? (2.0 * M_PI * i / (P->dx * P->Nx))
                              : (2.0 * M_PI * (i - P->Nx) / (P->dx * P->Nx));
        for (int j = 0; j < P->Ny; ++j) {
            const double ky = (j <= P->Ny / 2)
                                  ? (2.0 * M_PI * j / (P->dy * P->Ny))
                                  : (2.0 * M_PI * (j - P->Ny) / (P->dy * P->Ny));
            for (int k = 0; k < NzC; ++k) {
                const double kz = 2.0 * M_PI * k / (P->dz * P->Nz);
                const double k2 = kx * kx + ky * ky + kz * kz;
                const double denom = 1.0 + mobility * dt * kappa * k2;
                min_d = fmin(min_d, denom);
                max_d = fmax(max_d, denom);
                sum_d += denom;
                ++count;
            }
        }
    }
    const double kx_max = M_PI / fmax(P->dx, 1.0e-30);
    const double ky_max = M_PI / fmax(P->dy, 1.0e-30);
    const double kz_max = M_PI / fmax(P->dz, 1.0e-30);
    const double kmax2 = kx_max * kx_max + ky_max * ky_max + kz_max * kz_max;
    if (min_val) *min_val = min_d;
    if (max_val) *max_val = max_d;
    if (mean_val) *mean_val = sum_d / fmax((double)count, 1.0);
    if (at_kmax) *at_kmax = 1.0 + mobility * dt * kappa * kmax2;
}

static void write_phi_eta_rhs_attribution_csv_header(FILE *fp) {
    if (!fp) return;
    fprintf(fp,
            "step,time_code,dt,"
            "phi_rhs_total_explicit_signed_mean,phi_rhs_total_explicit_abs_mean,phi_rhs_total_explicit_abs_max,phi_rhs_total_explicit_rms,"
            "phi_rhs_chem_signed_mean,phi_rhs_chem_abs_mean,phi_rhs_chem_abs_max,phi_rhs_chem_rms,"
            "phi_rhs_double_well_signed_mean,phi_rhs_double_well_abs_mean,phi_rhs_double_well_abs_max,phi_rhs_double_well_rms,"
            "phi_rhs_elastic_signed_mean,phi_rhs_elastic_abs_mean,phi_rhs_elastic_abs_max,phi_rhs_elastic_rms,"
            "phi_update_explicit_est_signed_mean,phi_update_explicit_est_abs_mean,phi_update_explicit_est_abs_max,phi_update_explicit_est_rms,"
            "dphi_actual_signed_mean,dphi_actual_abs_mean,dphi_actual_abs_max,dphi_actual_rms,"
            "eta_rhs_total_explicit_signed_mean,eta_rhs_total_explicit_abs_mean,eta_rhs_total_explicit_abs_max,eta_rhs_total_explicit_rms,"
            "eta_rhs_bulk_signed_mean,eta_rhs_bulk_abs_mean,eta_rhs_bulk_abs_max,eta_rhs_bulk_rms,"
            "eta_rhs_double_well_signed_mean,eta_rhs_double_well_abs_mean,eta_rhs_double_well_abs_max,eta_rhs_double_well_rms,"
            "eta_rhs_elastic_signed_mean,eta_rhs_elastic_abs_mean,eta_rhs_elastic_abs_max,eta_rhs_elastic_rms,"
            "eta_update_explicit_est_signed_mean,eta_update_explicit_est_abs_mean,eta_update_explicit_est_abs_max,eta_update_explicit_est_rms,"
            "deta_actual_signed_mean,deta_actual_abs_mean,deta_actual_abs_max,deta_actual_rms,"
            "phi_grad_denom_min,phi_grad_denom_max,phi_grad_denom_mean,phi_grad_denom_at_kmax,"
            "eta_grad_denom_min,eta_grad_denom_max,eta_grad_denom_mean,eta_grad_denom_at_kmax,"
            "eta_rhs_total_absmax_over_phi_rhs_total_absmax,eta_rhs_bulk_absmax_over_phi_chem_absmax,eta_rhs_dw_absmax_over_phi_dw_absmax,eta_rhs_elastic_absmax_over_phi_elastic_absmax,"
            "eta_Ldt_rhs_absmax_over_phi_Ldt_rhs_absmax,eta_Ldt_bulk_absmax_over_phi_Ldt_chem_absmax,eta_Ldt_dw_absmax_over_phi_Ldt_dw_absmax,eta_Ldt_elastic_absmax_over_phi_Ldt_elastic_absmax,"
            "deta_absmax_over_dphi_absmax,deta_absmean_over_dphi_absmean,deta_rms_over_dphi_rms,"
            "phi_damping_stronger_than_eta,phi_actual_over_explicit_est_absmax,eta_actual_over_explicit_est_absmax,"
            "corr_abs_deta_abs_dxB,corr_abs_deta_abs_dY,corr_abs_deta_local_xB,corr_abs_deta_eta_before,corr_abs_deta_phi_before,"
            "max_abs_deta_idx,max_abs_deta_i,max_abs_deta_j,max_abs_deta_k,max_abs_deta_value,max_abs_deta_eta_before,max_abs_deta_phi_before,max_abs_deta_xB_before,max_abs_deta_Y_before,max_abs_deta_eta_rhs_bulk,max_abs_deta_eta_rhs_dw,max_abs_deta_eta_rhs_total,max_abs_deta_dxB,"
            "max_abs_dphi_idx,max_abs_dphi_i,max_abs_dphi_j,max_abs_dphi_k,max_abs_dphi_value,max_abs_dphi_eta_before,max_abs_dphi_phi_before,max_abs_dphi_xB_before,max_abs_dphi_Y_before,max_abs_dphi_phi_rhs_chem,max_abs_dphi_phi_rhs_dw,max_abs_dphi_phi_rhs_total,max_abs_dphi_dxB,"
            "xB_min_after,xB_max_after,xBtot_rel_delta,eta_far_field_max,xB_clip_low_count,xB_clip_high_count,nan_inf_flag,"
            "S_phi_W,S_phi_grad,S_phi_total,S_eta_W,S_eta_grad,S_eta_total,gp_L_eta,L_phi\n");
}

static void write_phi_eta_rhs_attribution_csv_row(FILE *fp,
                                                  const PhiEtaRHSAttributionRow *r) {
    if (!fp || !r) return;
#define CSV_PD(v) fprintf(fp, "%.10e", (double)(v))
#define CSV_PI(v) fprintf(fp, "%d", (int)(v))
#define CSV_PLL(v) fprintf(fp, "%lld", (long long)(v))
#define CSV_PC() fputc(',', fp)
#define CSV_PSTAT(s) CSV_PD((s).signed_mean); CSV_PC(); CSV_PD((s).abs_mean); CSV_PC(); CSV_PD((s).abs_max); CSV_PC(); CSV_PD((s).rms)
    CSV_PI(r->step); CSV_PC(); CSV_PD(r->time_code); CSV_PC(); CSV_PD(r->dt); CSV_PC();
    CSV_PSTAT(r->phi_rhs_total_explicit); CSV_PC();
    CSV_PSTAT(r->phi_rhs_chem); CSV_PC();
    CSV_PSTAT(r->phi_rhs_double_well); CSV_PC();
    CSV_PSTAT(r->phi_rhs_elastic); CSV_PC();
    CSV_PSTAT(r->phi_update_explicit_est); CSV_PC();
    CSV_PSTAT(r->dphi_actual); CSV_PC();
    CSV_PSTAT(r->eta_rhs_total_explicit); CSV_PC();
    CSV_PSTAT(r->eta_rhs_bulk_or_chemical); CSV_PC();
    CSV_PSTAT(r->eta_rhs_double_well); CSV_PC();
    CSV_PSTAT(r->eta_rhs_elastic); CSV_PC();
    CSV_PSTAT(r->eta_update_explicit_est); CSV_PC();
    CSV_PSTAT(r->deta_actual); CSV_PC();
    CSV_PD(r->phi_grad_denom_min); CSV_PC(); CSV_PD(r->phi_grad_denom_max); CSV_PC(); CSV_PD(r->phi_grad_denom_mean); CSV_PC(); CSV_PD(r->phi_grad_denom_at_kmax); CSV_PC();
    CSV_PD(r->eta_grad_denom_min); CSV_PC(); CSV_PD(r->eta_grad_denom_max); CSV_PC(); CSV_PD(r->eta_grad_denom_mean); CSV_PC(); CSV_PD(r->eta_grad_denom_at_kmax); CSV_PC();
    CSV_PD(r->eta_rhs_total_absmax_over_phi_rhs_total_absmax); CSV_PC(); CSV_PD(r->eta_rhs_bulk_absmax_over_phi_chem_absmax); CSV_PC(); CSV_PD(r->eta_rhs_dw_absmax_over_phi_dw_absmax); CSV_PC(); CSV_PD(r->eta_rhs_elastic_absmax_over_phi_elastic_absmax); CSV_PC();
    CSV_PD(r->eta_Ldt_rhs_absmax_over_phi_Ldt_rhs_absmax); CSV_PC(); CSV_PD(r->eta_Ldt_bulk_absmax_over_phi_Ldt_chem_absmax); CSV_PC(); CSV_PD(r->eta_Ldt_dw_absmax_over_phi_Ldt_dw_absmax); CSV_PC(); CSV_PD(r->eta_Ldt_elastic_absmax_over_phi_Ldt_elastic_absmax); CSV_PC();
    CSV_PD(r->deta_absmax_over_dphi_absmax); CSV_PC(); CSV_PD(r->deta_absmean_over_dphi_absmean); CSV_PC(); CSV_PD(r->deta_rms_over_dphi_rms); CSV_PC();
    CSV_PD(r->phi_damping_stronger_than_eta); CSV_PC(); CSV_PD(r->phi_actual_over_explicit_est_absmax); CSV_PC(); CSV_PD(r->eta_actual_over_explicit_est_absmax); CSV_PC();
    CSV_PD(r->corr_abs_deta_abs_dxB); CSV_PC(); CSV_PD(r->corr_abs_deta_abs_dY); CSV_PC(); CSV_PD(r->corr_abs_deta_local_xB); CSV_PC(); CSV_PD(r->corr_abs_deta_eta_before); CSV_PC(); CSV_PD(r->corr_abs_deta_phi_before); CSV_PC();
    CSV_PI(r->max_abs_deta_idx); CSV_PC(); CSV_PI(r->max_abs_deta_i); CSV_PC(); CSV_PI(r->max_abs_deta_j); CSV_PC(); CSV_PI(r->max_abs_deta_k); CSV_PC(); CSV_PD(r->max_abs_deta_value); CSV_PC(); CSV_PD(r->max_abs_deta_eta_before); CSV_PC(); CSV_PD(r->max_abs_deta_phi_before); CSV_PC(); CSV_PD(r->max_abs_deta_xB_before); CSV_PC(); CSV_PD(r->max_abs_deta_Y_before); CSV_PC(); CSV_PD(r->max_abs_deta_eta_rhs_bulk); CSV_PC(); CSV_PD(r->max_abs_deta_eta_rhs_dw); CSV_PC(); CSV_PD(r->max_abs_deta_eta_rhs_total); CSV_PC(); CSV_PD(r->max_abs_deta_dxB); CSV_PC();
    CSV_PI(r->max_abs_dphi_idx); CSV_PC(); CSV_PI(r->max_abs_dphi_i); CSV_PC(); CSV_PI(r->max_abs_dphi_j); CSV_PC(); CSV_PI(r->max_abs_dphi_k); CSV_PC(); CSV_PD(r->max_abs_dphi_value); CSV_PC(); CSV_PD(r->max_abs_dphi_eta_before); CSV_PC(); CSV_PD(r->max_abs_dphi_phi_before); CSV_PC(); CSV_PD(r->max_abs_dphi_xB_before); CSV_PC(); CSV_PD(r->max_abs_dphi_Y_before); CSV_PC(); CSV_PD(r->max_abs_dphi_phi_rhs_chem); CSV_PC(); CSV_PD(r->max_abs_dphi_phi_rhs_dw); CSV_PC(); CSV_PD(r->max_abs_dphi_phi_rhs_total); CSV_PC(); CSV_PD(r->max_abs_dphi_dxB); CSV_PC();
    CSV_PD(r->xB_min_after); CSV_PC(); CSV_PD(r->xB_max_after); CSV_PC(); CSV_PD(r->xBtot_rel_delta); CSV_PC(); CSV_PD(r->eta_far_field_max); CSV_PC(); CSV_PLL(r->xB_clip_low_count); CSV_PC(); CSV_PLL(r->xB_clip_high_count); CSV_PC(); CSV_PI(r->nan_inf_flag); CSV_PC();
    CSV_PD(r->S_phi_W); CSV_PC(); CSV_PD(r->S_phi_grad); CSV_PC(); CSV_PD(r->S_phi_total); CSV_PC(); CSV_PD(r->S_eta_W); CSV_PC(); CSV_PD(r->S_eta_grad); CSV_PC(); CSV_PD(r->S_eta_total); CSV_PC(); CSV_PD(r->gp_L_eta); CSV_PC(); CSV_PD(r->L_phi);
    fputc('\n', fp);
#undef CSV_PD
#undef CSV_PI
#undef CSV_PLL
#undef CSV_PC
#undef CSV_PSTAT
}

static void write_eta_bulk_max_location_csv_header(FILE *fp) {
    if (!fp) return;
    fprintf(fp,
            "step,time_code,dt,max_abs_dgbulk_idx,max_abs_dgbulk_i,max_abs_dgbulk_j,max_abs_dgbulk_k,"
            "max_abs_dgbulk_value,eta_before,phi_before,xB_before,Y_before,h_eta,hp_eta,"
            "mu_PbTe_raw,mu_Ag2Te_raw,gp_mu_reference_raw,drive_raw_formula_value,dgbulk_deta_raw_formula_value,"
            "energy_scale,mu_PbTe_dimless,mu_Ag2Te_dimless,gp_mu_reference_dimless,drive_dimless_formula_value,dgbulk_deta_dimless_formula_value,"
            "dgbulk_deta_kernel_used,eta_rhs_bulk,eta_rhs_double_well,eta_rhs_elastic,eta_rhs_total,dxB,deta,"
            "eta_rhs_bulk_absmax,eta_rhs_double_well_absmax,eta_rhs_elastic_absmax,eta_rhs_total_absmax,"
            "phi_rhs_chem_at_loc,phi_rhs_total_at_loc,phi_rhs_dw_at_loc,phi_rhs_elastic_at_loc,"
            "phi_chem_rhs_absmax,phi_double_well_absmax,phi_elastic_absmax,phi_rhs_total_absmax,"
            "eta_bulk_rhs_absmax_over_phi_chem_rhs_absmax,eta_Ldt_bulk_absmax_over_phi_Ldt_chem_absmax,"
            "gp_W_eta_kernel_used,gp_kappa_eta_kernel_used,gp_L_eta,S_eta_W,S_eta_grad,S_eta_total,"
            "W_phi,kappa_phi,L_phi,S_phi_W,S_phi_grad,S_phi_total,"
            "region_class,is_eta_interface,is_far_field_like,is_phi_nonzero,"
            "eta_peak_idx,eta_peak_i,eta_peak_j,eta_peak_k,eta_peak_value\n");
}

static int is_valid_gp_init_mode(const char *mode) {
    if (!mode || mode[0] == '\0') return 0;
    return (strcmp(mode, "none") == 0 ||
            strcmp(mode, "single_sphere") == 0 ||
            strcmp(mode, "observed_gp_diffuse") == 0);
}

static int is_valid_gp_init_mass_mode(const char *mode) {
    if (!mode || mode[0] == '\0') return 0;
    return (strcmp(mode, "report") == 0 ||
            strcmp(mode, "adjust_background") == 0 ||
            strcmp(mode, "local_compensate") == 0);
}

static int is_valid_gp_obs_profile_type(const char *mode) {
    if (!mode || mode[0] == '\0') return 0;
    return (strcmp(mode, "tanh") == 0 ||
            strcmp(mode, "gaussian") == 0 ||
            strcmp(mode, "compact_smooth") == 0);
}

static int is_valid_gp_obs_match_mode(const char *mode) {
    if (!mode || mode[0] == '\0') return 0;
    return strcmp(mode, "match_integral_h_volume") == 0;
}

static int is_valid_gp_obs_compensation_mode(const char *mode) {
    if (!mode || mode[0] == '\0') return 0;
    return (strcmp(mode, "smooth_radial_depletion") == 0 ||
            strcmp(mode, "wide_shell") == 0 ||
            strcmp(mode, "hybrid") == 0);
}

static int is_valid_gp_nuc_mass_mode(const char *mode) {
    if (!mode || mode[0] == '\0') return 0;
    return (strcmp(mode, "report") == 0 ||
            strcmp(mode, "local_compensate") == 0);
}

static int is_valid_gp_to_beta_mass_mode(const char *mode) {
    if (!mode || mode[0] == '\0') return 0;
    return (strcmp(mode, "report") == 0 ||
            strcmp(mode, "local_compensate") == 0);
}

static int is_valid_gp_to_beta_eta_deplete_mode(const char *mode) {
    if (!mode || mode[0] == '\0') return 0;
    return strcmp(mode, "multiply_1_minus_hphi_seed") == 0;
}

static int is_valid_gp_to_beta_phi_insert_mode(const char *mode) {
    if (!mode || mode[0] == '\0') return 0;
    return strcmp(mode, "max") == 0;
}

static int is_valid_gp_to_beta_barrier_mode(const char *mode) {
    if (!mode || mode[0] == '\0') return 0;
    return strcmp(mode, "cnt_simple") == 0;
}

static int is_valid_gp_to_beta_drive_mode(const char *mode) {
    if (!mode || mode[0] == '\0') return 0;
    return (strcmp(mode, "constant") == 0 ||
            strcmp(mode, "local_simple") == 0);
}

static int is_valid_gp_eta_mass_limiter(const char *mode) {
    if (!mode || mode[0] == '\0') return 0;
    return (strcmp(mode, "off") == 0 || strcmp(mode, "local_clip") == 0);
}

static int is_valid_gp_y_update_mode(const char *mode) {
    if (!mode || mode[0] == '\0') return 0;
    return (strcmp(mode, "old_rhs") == 0 ||
            strcmp(mode, "conservative_y_rhs") == 0 ||
            strcmp(mode, "picard_storage") == 0 ||
            strcmp(mode, "storage_exact") == 0);
}

static int is_valid_gp_C_mode(const char *mode) {
    if (!mode || mode[0] == '\0') return 0;
    return strcmp(mode, "alpha") == 0;
}

static int is_valid_gp_eps_mode(const char *mode) {
    if (!mode || mode[0] == '\0') return 0;
    return strcmp(mode, "isotropic") == 0;
}

static int is_valid_gp_L_eta_mode(const char *mode) {
    if (!mode || mode[0] == '\0') return 0;
    return (strcmp(mode, "manual") == 0 ||
            strcmp(mode, "sto_S218b") == 0 ||
            strcmp(mode, "sto_S218b_override") == 0);
}

static int is_valid_y_update_mass_projection_target_mode(const char *mode) {
    if (!mode || mode[0] == '\0') return 0;
    return (strcmp(mode, "pre_Y_update") == 0 ||
            strcmp(mode, "post_conversion_baseline") == 0);
}

static double host_mean_xBtot_two_phase(const double *phi_r,
                                        const double *xB_alpha_r,
                                        const PFParams *P,
                                        int total_size) {
    double sum = 0.0;
    for (int i = 0; i < total_size; ++i) {
        double h_beta = h_of_phi(clamp01_local(phi_r[i]));
        double xB_alpha = clamp_eps(xB_alpha_r[i], P->xB_eps);
        sum += (1.0 - h_beta) * xB_alpha + h_beta;
    }
    return sum / fmax((double)total_size, 1.0);
}

static double host_mean_xBtot_gp(const double *phi_r,
                                 const double *eta_r,
                                 const double *xB_alpha_r,
                                 const PFParams *P,
                                 int total_size) {
    double sum = 0.0;
    for (int i = 0; i < total_size; ++i) {
        double h_alpha = 0.0;
        double h_GP = 0.0;
        double h_beta = 0.0;
        phase_fractions_gp(clamp01_local(phi_r[i]), clamp01_local(eta_r[i]),
                           &h_alpha, &h_GP, &h_beta);
        double xB_alpha = clamp_eps(xB_alpha_r[i], P->xB_eps);
        sum += h_alpha * xB_alpha + h_GP * P->gp_xB_fixed + h_beta;
    }
    return sum / fmax((double)total_size, 1.0);
}

static int gp_recovered_xB_valid_host(double xBtot_available,
                                      double h_beta,
                                      double A,
                                      double h_GP,
                                      double xB_GP,
                                      double xB_min,
                                      double xB_max,
                                      double eps_h,
                                      double *xB_rec_out) {
    if (!isfinite(xBtot_available) || !isfinite(h_beta) || !isfinite(A) || !isfinite(h_GP)) return 0;
    if (A < 0.0 || h_GP < 0.0 || h_GP > A) return 0;
    double h_alpha = A - h_GP;
    if (!(h_alpha >= eps_h)) return 0;
    double numerator = xBtot_available - h_GP * xB_GP - h_beta;
    if (!isfinite(numerator)) return 0;
    double xB_rec = numerator / h_alpha;
    if (!isfinite(xB_rec)) return 0;
    if (xB_rec_out) *xB_rec_out = xB_rec;
    return (xB_rec >= xB_min && xB_rec <= xB_max) ? 1 : 0;
}

static double gp_limited_hGP_host(double xBtot_available,
                                  double h_beta,
                                  double A,
                                  double h_GP_tent,
                                  double xB_GP,
                                  double xB_min,
                                  double xB_max,
                                  double eps_h) {
    double h_hi = fmin(fmax(h_GP_tent, 0.0), fmax(A, 0.0));
    double xB_tmp = 0.0;
    if (gp_recovered_xB_valid_host(xBtot_available, h_beta, A, h_hi, xB_GP,
                                   xB_min, xB_max, eps_h, &xB_tmp)) {
        return h_hi;
    }
    double lo = 0.0;
    double hi = h_hi;
    for (int iter = 0; iter < 80; ++iter) {
        double mid = 0.5 * (lo + hi);
        if (gp_recovered_xB_valid_host(xBtot_available, h_beta, A, mid, xB_GP,
                                       xB_min, xB_max, eps_h, &xB_tmp)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

static int apply_gp_eta_pointwise_feasibility_host(const double *phi_r,
                                                   double *eta_r,
                                                   const double *xB_alpha_r,
                                                   const PFParams *P,
                                                   int total_size,
                                                   long long *limited_count_out,
                                                   double *eta_max_before_out,
                                                   double *eta_max_after_out) {
    long long limited_count = 0;
    double eta_max_before = 0.0;
    double eta_max_after = 0.0;
    for (int i = 0; i < total_size; ++i) {
        double phi = clamp01_local(phi_r[i]);
        double eta_old = clamp01_local(eta_r[i]);
        double xB_old = clamp_eps(xB_alpha_r[i], P->xB_eps);
        double h_beta = h_of_phi(phi);
        double A = fmax(1.0 - h_beta, 0.0);
        double h_GP_tent = A * h_of_eta(eta_old);
        // Initialization limiter uses the local matrix-held Ag as the available
        // storage budget before introducing a new GP seed at this point.
        double xBtot_available = A * xB_old + h_beta;
        double xB_rec_before = 0.0;
        int valid_before = gp_recovered_xB_valid_host(xBtot_available, h_beta, A, h_GP_tent,
                                                      P->gp_xB_fixed, P->xB_eps, 1.0 - P->xB_eps,
                                                      P->gp_h_alpha_eps, &xB_rec_before);
        double eta_new = eta_old;
        if (!valid_before && h_GP_tent > 0.0 && A > P->gp_h_alpha_eps) {
            double h_GP_allowed = gp_limited_hGP_host(xBtot_available, h_beta, A, h_GP_tent,
                                                      P->gp_xB_fixed, P->xB_eps, 1.0 - P->xB_eps,
                                                      P->gp_h_alpha_eps);
            double h_eta_allowed = clamp01_local(h_GP_allowed / fmax(A, P->gp_h_alpha_eps));
            eta_new = h_inverse_bisection_host(h_eta_allowed);
            eta_new = clamp01_local(eta_new);
            if (fabs(eta_new - eta_old) > 0.0) ++limited_count;
        }
        eta_r[i] = eta_new;
        if (eta_old > eta_max_before) eta_max_before = eta_old;
        if (eta_new > eta_max_after) eta_max_after = eta_new;
    }
    if (limited_count_out) *limited_count_out = limited_count;
    if (eta_max_before_out) *eta_max_before_out = eta_max_before;
    if (eta_max_after_out) *eta_max_after_out = eta_max_after;
    return 1;
}

static void recompute_host_xBtot_field(const double *phi_r,
                                       const double *eta_r,
                                       const double *xB_alpha_r,
                                       double *xBtot_r,
                                       const PFParams *P,
                                       int total_size) {
    for (int i = 0; i < total_size; ++i) {
        if (is_gp_zone_mode(P)) {
            double h_alpha = 0.0;
            double h_GP = 0.0;
            double h_beta = 0.0;
            phase_fractions_gp(clamp01_local(phi_r[i]), clamp01_local(eta_r[i]),
                               &h_alpha, &h_GP, &h_beta);
            xBtot_r[i] = h_alpha * clamp_eps(xB_alpha_r[i], P->xB_eps) + h_GP * P->gp_xB_fixed + h_beta;
        } else {
            double h_beta = h_of_phi(clamp01_local(phi_r[i]));
            xBtot_r[i] = (1.0 - h_beta) * clamp_eps(xB_alpha_r[i], P->xB_eps) + h_beta;
        }
    }
}

static inline int host_linear_index_3d(int i, int j, int k, int Ny, int Nz) {
    return (i * Ny + j) * Nz + k;
}

static double host_sum_xBtot_gp_total(const double *phi_r,
                                      const double *eta_r,
                                      const double *xB_alpha_r,
                                      const PFParams *P,
                                      int total_size) {
    return host_mean_xBtot_gp(phi_r, eta_r, xB_alpha_r, P, total_size) * (double)total_size;
}

static double host_sum_hGP_volume(const double *phi_r,
                                  const double *eta_r,
                                  const PFParams *P,
                                  int total_size) {
    const double dV = P->dx * P->dy * P->dz;
    double sum = 0.0;
    for (int i = 0; i < total_size; ++i) {
        double h_alpha = 0.0, h_GP = 0.0, h_beta = 0.0;
        phase_fractions_gp(clamp01_local(phi_r[i]), clamp01_local(eta_r[i]), &h_alpha, &h_GP, &h_beta);
        (void)h_alpha;
        (void)h_beta;
        sum += h_GP * dV;
    }
    return sum;
}

static double observed_gp_profile_value_host(double r,
                                             double radius_param,
                                             double eta_peak,
                                             double width,
                                             const char *profile_type) {
    const double rp = fmax(radius_param, 1.0e-30);
    const double w = fmax(width, 1.0e-30);
    double eta = 0.0;
    if (strcmp(profile_type, "gaussian") == 0) {
        eta = eta_peak * exp(-(r * r) / fmax(rp * rp, 1.0e-30));
    } else if (strcmp(profile_type, "compact_smooth") == 0) {
        if (r < rp) {
            double q = 1.0 - (r / rp) * (r / rp);
            eta = eta_peak * q * q;
        } else {
            eta = 0.0;
        }
    } else {
        eta = eta_peak * 0.5 * (1.0 - tanh((r - rp) / w));
    }
    return clamp01_local(eta);
}

static double observed_gp_profile_h_volume_host(double radius_param,
                                                double eta_peak,
                                                double width,
                                                const char *profile_type,
                                                const PFParams *P,
                                                double cx,
                                                double cy,
                                                double cz) {
    const double Lx = P->Nx * P->dx;
    const double Ly = P->Ny * P->dy;
    const double Lz = P->Nz * P->dz;
    const double dV = P->dx * P->dy * P->dz;
    const int total_size = P->Nx * P->Ny * P->Nz;
    double Vh = 0.0;
    for (int idx = 0; idx < total_size; ++idx) {
        int i = 0, j = 0, k = 0;
        host_decode_index(idx, P->Ny, P->Nz, &i, &j, &k);
        double x = i * P->dx;
        double y = j * P->dy;
        double z = k * P->dz;
        double rx = periodic_delta(x, cx, Lx);
        double ry = periodic_delta(y, cy, Ly);
        double rz = periodic_delta(z, cz, Lz);
        double r = sqrt(rx * rx + ry * ry + rz * rz);
        double eta = observed_gp_profile_value_host(r, radius_param, eta_peak, width, profile_type);
        Vh += h_of_eta(eta) * dV;
    }
    return Vh;
}

static double observed_gp_comp_weight_host(double r,
                                           double R_target,
                                           double R_dep,
                                           double smooth_w,
                                           const char *mode) {
    const double w = fmax(smooth_w, 1.0e-30);
    const double r_core = fmax(R_target, 0.0);
    const double r_dep_eff = fmax(R_dep, r_core + w);
    const double outside_core = 0.5 * (1.0 + tanh((r - r_core) / w));
    const double inside_dep = 0.5 * (1.0 - tanh((r - r_dep_eff) / w));
    const double gaussian_tail = exp(-(r * r) / fmax(r_dep_eff * r_dep_eff, 1.0e-30));
    double weight = 0.0;
    if (strcmp(mode, "wide_shell") == 0) {
        weight = outside_core * inside_dep;
    } else if (strcmp(mode, "hybrid") == 0) {
        weight = outside_core * (0.65 * inside_dep + 0.35 * gaussian_tail);
    } else {
        weight = outside_core * gaussian_tail;
    }
    if (!isfinite(weight) || weight < 0.0) weight = 0.0;
    return weight;
}

static double host_max_abs_grad_scalar_periodic(const double *field, const PFParams *P) {
    double max_abs_grad = 0.0;
    for (int i = 0; i < P->Nx; ++i) {
        int ip = (i + 1) % P->Nx;
        int im = (i - 1 + P->Nx) % P->Nx;
        for (int j = 0; j < P->Ny; ++j) {
            int jp = (j + 1) % P->Ny;
            int jm = (j - 1 + P->Ny) % P->Ny;
            for (int k = 0; k < P->Nz; ++k) {
                int kp = (k + 1) % P->Nz;
                int km = (k - 1 + P->Nz) % P->Nz;
                int idx = host_linear_index_3d(i, j, k, P->Ny, P->Nz);
                (void)idx;
                double dfdx = (field[host_linear_index_3d(ip, j, k, P->Ny, P->Nz)] -
                               field[host_linear_index_3d(im, j, k, P->Ny, P->Nz)]) / (2.0 * P->dx);
                double dfdy = (field[host_linear_index_3d(i, jp, k, P->Ny, P->Nz)] -
                               field[host_linear_index_3d(i, jm, k, P->Ny, P->Nz)]) / (2.0 * P->dy);
                double dfdz = (field[host_linear_index_3d(i, j, kp, P->Ny, P->Nz)] -
                               field[host_linear_index_3d(i, j, km, P->Ny, P->Nz)]) / (2.0 * P->dz);
                double mag = sqrt(dfdx * dfdx + dfdy * dfdy + dfdz * dfdz);
                if (mag > max_abs_grad) max_abs_grad = mag;
            }
        }
    }
    return max_abs_grad;
}

static double host_max_abs_grad_mu_from_xB_periodic(const double *xB_alpha_r, const PFParams *P) {
    const int total_size = P->Nx * P->Ny * P->Nz;
    std::vector<double> mu_field((size_t)total_size, 0.0);
    const double temperature_K = P->temperature_C + 273.15;
    for (int idx = 0; idx < total_size; ++idx) {
        mu_field[(size_t)idx] = compute_mu_C_gp_host(xB_alpha_r[idx], temperature_K,
                                                     P->mu_reference_scale,
                                                     P->Vm_alpha_0,
                                                     P->dVm_alpha_dxB);
    }
    return host_max_abs_grad_scalar_periodic(mu_field.data(), P);
}

static int initialize_observed_gp_diffuse_host(double *phi_r,
                                               double *eta_r,
                                               double *Y_r,
                                               double *xB_alpha_r,
                                               double *xBtot_r,
                                               const PFParams *P,
                                               int total_size,
                                               double cx,
                                               double cy,
                                               double cz) {
    const double xB_min_bound = fmax(P->gp_obs_min_xB_alpha, P->xB_eps);
    const double xB_max_bound = fmin(P->gp_obs_max_xB_alpha, 1.0 - P->xB_eps);
    const double R_target = P->gp_obs_target_radius_nm;
    const double eta_peak = P->gp_obs_eta_peak;
    const double w_eta = fmax(P->gp_obs_iface_width_nm, P->dx);
    const double V_target = (4.0 / 3.0) * M_PI * R_target * R_target * R_target;
    const double Lx = P->Nx * P->dx;
    const double Ly = P->Ny * P->dy;
    const double Lz = P->Nz * P->dz;
    const double dV = P->dx * P->dy * P->dz;
    const double R_dep = fmax(P->gp_obs_depletion_radius_factor * R_target, R_target + P->dx);
    const double depletion_smooth_width = fmax(P->gp_obs_depletion_smooth_width_factor * R_target, P->dx);

    if (!is_valid_gp_obs_profile_type(P->gp_obs_profile_type) ||
        !is_valid_gp_obs_match_mode(P->gp_obs_match_mode) ||
        !is_valid_gp_obs_compensation_mode(P->gp_obs_compensation_mode)) {
        fprintf(stderr, "[fatal] observed_gp_diffuse received invalid profile/match/compensation mode\n");
        return 0;
    }

    double R_lo = fmax(0.25 * R_target, 0.1 * P->dx);
    double R_hi = fmax(2.0 * R_target, R_lo + P->dx);
    double V_hi = observed_gp_profile_h_volume_host(R_hi, eta_peak, w_eta, P->gp_obs_profile_type, P, cx, cy, cz);
    const double half_box = 0.5 * fmin(Lx, fmin(Ly, Lz));
    while (V_hi < V_target && R_hi < half_box) {
        R_hi *= 1.5;
        V_hi = observed_gp_profile_h_volume_host(R_hi, eta_peak, w_eta, P->gp_obs_profile_type, P, cx, cy, cz);
    }
    if (V_hi < V_target) {
        log_section_header("Observed GP Diffuse Initialization");
        log_kv_text("gp_obs_init_success", "%d", 0);
        log_kv_text("gp_obs_init_failure_reason", "%s", "volume_match_bracket_failed");
        log_kv_text("gp_obs_target_radius_nm", "%.8e", R_target);
        log_kv_text("gp_obs_eta_peak", "%.8e", eta_peak);
        log_kv_text("gp_obs_V_target", "%.8e", V_target);
        log_kv_text("gp_obs_Vh_hi", "%.8e", V_hi);
        return 0;
    }

    double R_profile = R_hi;
    for (int iter = 0; iter < 80; ++iter) {
        double mid = 0.5 * (R_lo + R_hi);
        double V_mid = observed_gp_profile_h_volume_host(mid, eta_peak, w_eta, P->gp_obs_profile_type, P, cx, cy, cz);
        if (V_mid < V_target) {
            R_lo = mid;
        } else {
            R_hi = mid;
        }
        R_profile = 0.5 * (R_lo + R_hi);
    }

    std::vector<double> eta_trial((size_t)total_size, 0.0);
    std::vector<double> xB_trial((size_t)total_size, 0.0);
    std::vector<double> weight((size_t)total_size, 0.0);
    const double mass_before = host_sum_xBtot_gp_total(phi_r, eta_r, xB_alpha_r, P, total_size);
    double xB_shell_min_before = INFINITY;
    double xB_shell_max_before = -INFINITY;

    for (int idx = 0; idx < total_size; ++idx) {
        int i = 0, j = 0, k = 0;
        host_decode_index(idx, P->Ny, P->Nz, &i, &j, &k);
        double x = i * P->dx;
        double y = j * P->dy;
        double z = k * P->dz;
        double rx = periodic_delta(x, cx, Lx);
        double ry = periodic_delta(y, cy, Ly);
        double rz = periodic_delta(z, cz, Lz);
        double r = sqrt(rx * rx + ry * ry + rz * rz);
        double eta_seed = observed_gp_profile_value_host(r, R_profile, eta_peak, w_eta, P->gp_obs_profile_type);
        eta_trial[(size_t)idx] = fmax(clamp01_local(eta_r[idx]), eta_seed);
        xB_trial[(size_t)idx] = clamp_eps(xB_alpha_r[idx], P->xB_eps);
        weight[(size_t)idx] = observed_gp_comp_weight_host(r, R_target, R_dep,
                                                           depletion_smooth_width,
                                                           P->gp_obs_compensation_mode);
        if (weight[(size_t)idx] > 1.0e-14) {
            double xv = xB_trial[(size_t)idx];
            if (xv < xB_shell_min_before) xB_shell_min_before = xv;
            if (xv > xB_shell_max_before) xB_shell_max_before = xv;
        }
    }

    double mass_after_seed = host_sum_xBtot_gp_total(phi_r, eta_trial.data(), xB_trial.data(), P, total_size);
    const double demand = mass_after_seed - mass_before;
    const int remove_mass = (demand > 0.0) ? 1 : 0;

    auto mass_after_compensation_for_amp = [&](double amp,
                                               std::vector<double> *xB_out,
                                               double *mass_delta_out,
                                               double *xmin_out,
                                               double *xmax_out) {
        double total_mass = 0.0;
        double total_delta_mass = 0.0;
        double xmin = INFINITY;
        double xmax = -INFINITY;
        for (int idx = 0; idx < total_size; ++idx) {
            double xB_old = clamp_eps(xB_trial[(size_t)idx], P->xB_eps);
            double h_alpha = 0.0, h_GP = 0.0, h_beta = 0.0;
            phase_fractions_gp(clamp01_local(phi_r[idx]), clamp01_local(eta_trial[(size_t)idx]), &h_alpha, &h_GP, &h_beta);
            double signed_delta = remove_mass ? (-amp * weight[(size_t)idx]) : (amp * weight[(size_t)idx]);
            double xB_new = xB_old + signed_delta;
            if (xB_new < xB_min_bound) xB_new = xB_min_bound;
            if (xB_new > xB_max_bound) xB_new = xB_max_bound;
            if (xB_out) (*xB_out)[(size_t)idx] = xB_new;
            total_delta_mass += h_alpha * (xB_new - xB_old);
            total_mass += h_alpha * xB_new + h_GP * P->gp_xB_fixed + h_beta;
            if (xB_new < xmin) xmin = xB_new;
            if (xB_new > xmax) xmax = xB_new;
        }
        if (mass_delta_out) *mass_delta_out = total_delta_mass;
        if (xmin_out) *xmin_out = xmin;
        if (xmax_out) *xmax_out = xmax;
        return total_mass;
    };

    double compensation_amp = 0.0;
    int init_success = 1;
    if (fabs(demand) > 1.0e-14) {
        double lo = 0.0;
        double hi = 1.0;
        double mass_delta_hi = 0.0;
        double xmin_tmp = 0.0, xmax_tmp = 0.0;
        mass_after_compensation_for_amp(hi, NULL, &mass_delta_hi, &xmin_tmp, &xmax_tmp);
        double supplied_hi = remove_mass ? (-mass_delta_hi) : mass_delta_hi;
        for (int iter = 0; iter < 40 && supplied_hi < fabs(demand); ++iter) {
            hi *= 2.0;
            mass_after_compensation_for_amp(hi, NULL, &mass_delta_hi, &xmin_tmp, &xmax_tmp);
            supplied_hi = remove_mass ? (-mass_delta_hi) : mass_delta_hi;
        }
        if (supplied_hi + 1.0e-14 < fabs(demand)) {
            init_success = 0;
        } else {
            for (int iter = 0; iter < 80; ++iter) {
                double mid = 0.5 * (lo + hi);
                double mass_delta_mid = 0.0;
                mass_after_compensation_for_amp(mid, NULL, &mass_delta_mid, &xmin_tmp, &xmax_tmp);
                double supplied_mid = remove_mass ? (-mass_delta_mid) : mass_delta_mid;
                if (supplied_mid < fabs(demand)) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            compensation_amp = hi;
        }
    }

    double xB_min_init = NAN;
    double xB_max_init = NAN;
    double mass_after = mass_after_seed;
    double compensation_mass = 0.0;
    if (init_success) {
        if (fabs(demand) > 1.0e-14) {
            mass_after = mass_after_compensation_for_amp(compensation_amp, &xB_trial, &compensation_mass,
                                                         &xB_min_init, &xB_max_init);
        } else {
            for (int idx = 0; idx < total_size; ++idx) {
                xB_trial[(size_t)idx] = clamp_eps(xB_trial[(size_t)idx], P->xB_eps);
            }
            mass_after = host_sum_xBtot_gp_total(phi_r, eta_trial.data(), xB_trial.data(), P, total_size);
            xB_min_init = *std::min_element(xB_trial.begin(), xB_trial.end());
            xB_max_init = *std::max_element(xB_trial.begin(), xB_trial.end());
        }
    }

    for (int idx = 0; idx < total_size; ++idx) {
        eta_r[idx] = eta_trial[(size_t)idx];
        xB_alpha_r[idx] = xB_trial[(size_t)idx];
        if (Y_r) {
            Y_r[idx] = logit_from_fraction(xB_alpha_r[idx], P->xB_eps, P->Y_clip);
        }
    }
    recompute_host_xBtot_field(phi_r, eta_r, xB_alpha_r, xBtot_r, P, total_size);

    const double V_h_final = host_sum_hGP_volume(phi_r, eta_r, P, total_size);
    const double R_eff_h_final = cbrt(fmax(3.0 * V_h_final / (4.0 * M_PI), 0.0));
    const double mass_error = mass_after - mass_before;
    const double match_error = V_h_final - V_target;
    const double max_abs_grad_xB_init = host_max_abs_grad_scalar_periodic(xB_alpha_r, P);
    const double max_abs_grad_mu_init = host_max_abs_grad_mu_from_xB_periodic(xB_alpha_r, P);

    log_section_header("Observed GP Diffuse Initialization");
    log_kv_text("gp_obs_init_success", "%d", init_success);
    log_kv_text("gp_obs_target_radius_nm", "%.8e", R_target);
    log_kv_text("gp_obs_eta_peak", "%.8e", eta_peak);
    log_kv_text("gp_obs_profile_type", "%s", P->gp_obs_profile_type);
    log_kv_text("gp_obs_match_mode", "%s", P->gp_obs_match_mode);
    log_kv_text("gp_obs_compensation_mode", "%s", P->gp_obs_compensation_mode);
    log_kv_text("gp_obs_iface_width_nm", "%.8e", w_eta);
    log_kv_text("gp_obs_R_profile_used_nm", "%.8e", R_profile);
    log_kv_text("gp_obs_V_target", "%.8e", V_target);
    log_kv_text("gp_obs_V_h_final", "%.8e", V_h_final);
    log_kv_text("gp_obs_R_eff_h_final_nm", "%.8e", R_eff_h_final);
    log_kv_text("gp_obs_match_error", "%.8e", match_error);
    log_kv_text("gp_obs_mass_before", "%.8e", mass_before);
    log_kv_text("gp_obs_mass_after", "%.8e", mass_after);
    log_kv_text("gp_obs_mass_error", "%.8e", mass_error);
    log_kv_text("gp_obs_depletion_radius_used_nm", "%.8e", R_dep);
    log_kv_text("gp_obs_depletion_smooth_width_nm", "%.8e", depletion_smooth_width);
    log_kv_text("gp_obs_compensation_amplitude", "%.8e", compensation_amp);
    log_kv_text("gp_obs_compensation_mass", "%.8e", compensation_mass);
    log_kv_text("gp_obs_xB_min_init", "%.8e", xB_min_init);
    log_kv_text("gp_obs_xB_max_init", "%.8e", xB_max_init);
    log_kv_text("gp_obs_xB_shell_min_before_event", "%.8e", xB_shell_min_before);
    log_kv_text("gp_obs_xB_shell_max_before_event", "%.8e", xB_shell_max_before);
    log_kv_text("gp_obs_max_abs_grad_xB_init", "%.8e", max_abs_grad_xB_init);
    log_kv_text("gp_obs_estimated_max_abs_grad_mu_init", "%.8e", max_abs_grad_mu_init);

    return init_success;
}

static int apply_gp_eta_initialization_host(double *phi_r,
                                            double *eta_r,
                                            double *Y_r,
                                            double *xB_alpha_r,
                                            double *xBtot_r,
                                            const PFParams *P,
                                            int total_size) {
    if (!is_gp_zone_mode(P)) return 1;
    if (!is_valid_gp_init_mode(P->gp_init_mode)) {
        fprintf(stderr, "[fatal] invalid gp_init_mode = '%s'\n", P->gp_init_mode);
        return 0;
    }
    if (!is_valid_gp_init_mass_mode(P->gp_init_mass_mode)) {
        fprintf(stderr, "[fatal] invalid gp_init_mass_mode = '%s'\n", P->gp_init_mass_mode);
        return 0;
    }
    if (strcmp(P->gp_init_mode, "none") == 0) {
        recompute_host_xBtot_field(phi_r, eta_r, xB_alpha_r, xBtot_r, P, total_size);
        return 1;
    }
    if (strcmp(P->gp_init_mode, "single_sphere") != 0 &&
        strcmp(P->gp_init_mode, "observed_gp_diffuse") != 0) {
        fprintf(stderr, "[fatal] unsupported gp_init_mode = '%s'\n", P->gp_init_mode);
        return 0;
    }
    if (strcmp(P->gp_init_mode, "single_sphere") == 0 && !(P->gp_eta_seed_radius > 0.0)) {
        fprintf(stderr, "[fatal] gp_init_mode=single_sphere requires gp_eta_seed_radius > 0\n");
        return 0;
    }
    if (strcmp(P->gp_init_mode, "single_sphere") == 0 && !(P->gp_eta_seed_peak > 0.0)) {
        fprintf(stderr, "[fatal] gp_init_mode=single_sphere requires gp_eta_seed_peak > 0\n");
        return 0;
    }

    const double Lx = P->Nx * P->dx;
    const double Ly = P->Ny * P->dy;
    const double Lz = P->Nz * P->dz;
    const double cx = (P->gp_eta_seed_center_x == 0.0 && P->gp_eta_seed_center_y == 0.0 &&
                       P->gp_eta_seed_center_z == 0.0) ? (0.5 * Lx) : P->gp_eta_seed_center_x;
    const double cy = (P->gp_eta_seed_center_x == 0.0 && P->gp_eta_seed_center_y == 0.0 &&
                       P->gp_eta_seed_center_z == 0.0) ? (0.5 * Ly) : P->gp_eta_seed_center_y;
    const double cz = (P->gp_eta_seed_center_x == 0.0 && P->gp_eta_seed_center_y == 0.0 &&
                       P->gp_eta_seed_center_z == 0.0) ? (0.5 * Lz) : P->gp_eta_seed_center_z;
    if (strcmp(P->gp_init_mode, "observed_gp_diffuse") == 0) {
        return initialize_observed_gp_diffuse_host(phi_r, eta_r, Y_r, xB_alpha_r, xBtot_r,
                                                   P, total_size, cx, cy, cz);
    }
    const double w_eta = (P->gp_eta_iface_width > 0.0)
                             ? P->gp_eta_iface_width
                             : ((P->ic_phi_iface_w > 0.0) ? (P->ic_phi_iface_w * P->dx) : P->dx);
    const double mean_before = host_mean_xBtot_gp(phi_r, eta_r, xB_alpha_r, P, total_size);

    for (int idx = 0; idx < total_size; ++idx) {
        int i = 0, j = 0, k = 0;
        host_decode_index(idx, P->Ny, P->Nz, &i, &j, &k);
        double x = i * P->dx;
        double y = j * P->dy;
        double z = k * P->dz;
        double rx = periodic_delta(x, cx, Lx);
        double ry = periodic_delta(y, cy, Ly);
        double rz = periodic_delta(z, cz, Lz);
        double r = sqrt(rx * rx + ry * ry + rz * rz);
        double eta_seed = P->gp_eta_seed_peak *
                          0.5 * (1.0 - tanh((r - P->gp_eta_seed_radius) / fmax(w_eta, 1.0e-30)));
        eta_seed = clamp01_local(eta_seed);
        if (eta_seed > eta_r[idx]) {
            eta_r[idx] = eta_seed;
        }
    }

    double mean_after_seed = host_mean_xBtot_gp(phi_r, eta_r, xB_alpha_r, P, total_size);
    double mean_after_adjust = mean_after_seed;
    long long clip_count = 0;
    double gp_insert_mass_before = (double)total_size * mean_before;
    double gp_insert_mass_after = (double)total_size * mean_after_seed;
    double gp_insert_mass_error = gp_insert_mass_after - gp_insert_mass_before;
    int gp_insert_compensation_success = 0;
    double gp_insert_eta_scaled_factor = 1.0;
    long long eta_seed_limiter_count = 0;
    double eta_seed_max_before_limiter = 0.0;
    double eta_seed_max_after_limiter = 0.0;
    double eta_seed_scaled_or_limited_mass_change = 0.0;

    if (strcmp(P->gp_init_mass_mode, "adjust_background") == 0) {
        double sum_h_alpha = 0.0;
        for (int i = 0; i < total_size; ++i) {
            double h_alpha = 0.0;
            phase_fractions_gp(clamp01_local(phi_r[i]), clamp01_local(eta_r[i]), &h_alpha, NULL, NULL);
            sum_h_alpha += h_alpha;
        }
        if (sum_h_alpha <= 1.0e-30) {
            fprintf(stderr, "[fatal] gp_init_mass_mode=adjust_background but h_alpha support is empty\n");
            return 0;
        }
        double delta_xB = ((double)total_size * (mean_before - mean_after_seed)) / sum_h_alpha;
        for (int i = 0; i < total_size; ++i) {
            double xB_old = xB_alpha_r[i];
            double xB_new = clamp_eps(xB_old + delta_xB, P->xB_eps);
            if (fabs(xB_new - (xB_old + delta_xB)) > 0.0) {
                ++clip_count;
            }
            xB_alpha_r[i] = xB_new;
            if (Y_r) {
                Y_r[i] = logit_from_fraction(xB_new, P->xB_eps, P->Y_clip);
            }
        }
        mean_after_adjust = host_mean_xBtot_gp(phi_r, eta_r, xB_alpha_r, P, total_size);
        gp_insert_mass_error = (double)total_size * (mean_after_adjust - mean_before);
        gp_insert_compensation_success = 1;
    } else if (strcmp(P->gp_init_mass_mode, "local_compensate") == 0) {
        const double shell_inner = P->gp_eta_seed_radius;
        const double shell_outer = P->gp_eta_seed_radius + fmax(3.0 * w_eta, P->dx);
        const double target_mass = gp_insert_mass_before;
        double scale_lo = 0.0;
        double scale_hi = 1.0;
        double best_scale = 0.0;
        double best_shell_capacity = 0.0;
        for (int iter = 0; iter < 40; ++iter) {
            double scale_mid = 0.5 * (scale_lo + scale_hi);
            double trial_mass = 0.0;
            double shell_capacity = 0.0;
            for (int idx = 0; idx < total_size; ++idx) {
                int i = 0, j = 0, k = 0;
                host_decode_index(idx, P->Ny, P->Nz, &i, &j, &k);
                double x = i * P->dx;
                double y = j * P->dy;
                double z = k * P->dz;
                double rx = periodic_delta(x, cx, Lx);
                double ry = periodic_delta(y, cy, Ly);
                double rz = periodic_delta(z, cz, Lz);
                double r = sqrt(rx * rx + ry * ry + rz * rz);
                double eta_trial = clamp01_local(scale_mid * eta_r[idx]);
                double h_alpha = 0.0, h_GP = 0.0, h_beta = 0.0;
                phase_fractions_gp(clamp01_local(phi_r[idx]), eta_trial, &h_alpha, &h_GP, &h_beta);
                trial_mass += h_alpha * clamp_eps(xB_alpha_r[idx], P->xB_eps) + h_GP * P->gp_xB_fixed + h_beta;
                if (r >= shell_inner && r <= shell_outer) {
                    phase_fractions_gp(clamp01_local(phi_r[idx]), eta_trial, &h_alpha, NULL, NULL);
                    shell_capacity += h_alpha * fmax(xB_alpha_r[idx] - P->xB_eps, 0.0);
                }
            }
            double demand = trial_mass - target_mass;
            if (demand <= shell_capacity + 1.0e-14) {
                best_scale = scale_mid;
                best_shell_capacity = shell_capacity;
                scale_lo = scale_mid;
            } else {
                scale_hi = scale_mid;
            }
        }
        gp_insert_eta_scaled_factor = best_scale;
        for (int idx = 0; idx < total_size; ++idx) {
            eta_r[idx] = clamp01_local(best_scale * eta_r[idx]);
        }
        double mass_after_scaled = host_mean_xBtot_gp(phi_r, eta_r, xB_alpha_r, P, total_size) * (double)total_size;
        double demand_mass = mass_after_scaled - target_mass;
        if (demand_mass <= best_shell_capacity + 1.0e-14) {
            double shell_removal_sum = 0.0;
            for (int idx = 0; idx < total_size; ++idx) {
                int i = 0, j = 0, k = 0;
                host_decode_index(idx, P->Ny, P->Nz, &i, &j, &k);
                double x = i * P->dx;
                double y = j * P->dy;
                double z = k * P->dz;
                double rx = periodic_delta(x, cx, Lx);
                double ry = periodic_delta(y, cy, Ly);
                double rz = periodic_delta(z, cz, Lz);
                double r = sqrt(rx * rx + ry * ry + rz * rz);
                if (r < shell_inner || r > shell_outer) continue;
                double h_alpha = 0.0;
                phase_fractions_gp(clamp01_local(phi_r[idx]), clamp01_local(eta_r[idx]), &h_alpha, NULL, NULL);
                shell_removal_sum += h_alpha * fmax(xB_alpha_r[idx] - P->xB_eps, 0.0);
            }
            if (shell_removal_sum > 1.0e-30) {
                for (int idx = 0; idx < total_size; ++idx) {
                    int i = 0, j = 0, k = 0;
                    host_decode_index(idx, P->Ny, P->Nz, &i, &j, &k);
                    double x = i * P->dx;
                    double y = j * P->dy;
                    double z = k * P->dz;
                    double rx = periodic_delta(x, cx, Lx);
                    double ry = periodic_delta(y, cy, Ly);
                    double rz = periodic_delta(z, cz, Lz);
                    double r = sqrt(rx * rx + ry * ry + rz * rz);
                    if (r < shell_inner || r > shell_outer) continue;
                    double h_alpha = 0.0;
                    phase_fractions_gp(clamp01_local(phi_r[idx]), clamp01_local(eta_r[idx]), &h_alpha, NULL, NULL);
                    double capacity = h_alpha * fmax(xB_alpha_r[idx] - P->xB_eps, 0.0);
                    if (capacity <= 0.0) continue;
                    double mass_take = demand_mass * (capacity / shell_removal_sum);
                    double dxB = mass_take / fmax(h_alpha, 1.0e-30);
                    xB_alpha_r[idx] = clamp_eps(xB_alpha_r[idx] - dxB, P->xB_eps);
                    if (Y_r) {
                        Y_r[idx] = logit_from_fraction(xB_alpha_r[idx], P->xB_eps, P->Y_clip);
                    }
                }
                mean_after_adjust = host_mean_xBtot_gp(phi_r, eta_r, xB_alpha_r, P, total_size);
                gp_insert_mass_after = (double)total_size * mean_after_adjust;
                gp_insert_mass_error = gp_insert_mass_after - gp_insert_mass_before;
                gp_insert_compensation_success = (fabs(gp_insert_mass_error) < 1.0e-8) ? 1 : 0;
            }
        }
    }

    if (strcmp(P->gp_eta_mass_limiter, "local_clip") == 0) {
        double mass_before_limiter = host_mean_xBtot_gp(phi_r, eta_r, xB_alpha_r, P, total_size) * (double)total_size;
        if (!apply_gp_eta_pointwise_feasibility_host(phi_r, eta_r, xB_alpha_r, P, total_size,
                                                     &eta_seed_limiter_count,
                                                     &eta_seed_max_before_limiter,
                                                     &eta_seed_max_after_limiter)) {
            return 0;
        }
        double mass_after_limiter = host_mean_xBtot_gp(phi_r, eta_r, xB_alpha_r, P, total_size) * (double)total_size;
        eta_seed_scaled_or_limited_mass_change = mass_after_limiter - mass_before_limiter;
        if (strcmp(P->gp_init_mass_mode, "local_compensate") == 0 &&
            fabs(eta_seed_scaled_or_limited_mass_change) > 1.0e-14) {
            const double target_mass = gp_insert_mass_before;
            const double shell_inner = P->gp_eta_seed_radius;
            const double shell_outer = P->gp_eta_seed_radius + fmax(3.0 * w_eta, P->dx);
            double mass_now = mass_after_limiter;
            double excess_mass = mass_now - target_mass;
            if (excess_mass > 0.0) {
                double shell_removal_sum = 0.0;
                for (int idx = 0; idx < total_size; ++idx) {
                    int i = 0, j = 0, k = 0;
                    host_decode_index(idx, P->Ny, P->Nz, &i, &j, &k);
                    double x = i * P->dx;
                    double y = j * P->dy;
                    double z = k * P->dz;
                    double rx = periodic_delta(x, cx, Lx);
                    double ry = periodic_delta(y, cy, Ly);
                    double rz = periodic_delta(z, cz, Lz);
                    double r = sqrt(rx * rx + ry * ry + rz * rz);
                    if (r < shell_inner || r > shell_outer) continue;
                    double h_alpha = 0.0;
                    phase_fractions_gp(clamp01_local(phi_r[idx]), clamp01_local(eta_r[idx]), &h_alpha, NULL, NULL);
                    shell_removal_sum += h_alpha * fmax(xB_alpha_r[idx] - P->xB_eps, 0.0);
                }
                if (shell_removal_sum > 1.0e-30) {
                    for (int idx = 0; idx < total_size; ++idx) {
                        int i = 0, j = 0, k = 0;
                        host_decode_index(idx, P->Ny, P->Nz, &i, &j, &k);
                        double x = i * P->dx;
                        double y = j * P->dy;
                        double z = k * P->dz;
                        double rx = periodic_delta(x, cx, Lx);
                        double ry = periodic_delta(y, cy, Ly);
                        double rz = periodic_delta(z, cz, Lz);
                        double r = sqrt(rx * rx + ry * ry + rz * rz);
                        if (r < shell_inner || r > shell_outer) continue;
                        double h_alpha = 0.0;
                        phase_fractions_gp(clamp01_local(phi_r[idx]), clamp01_local(eta_r[idx]), &h_alpha, NULL, NULL);
                        double capacity = h_alpha * fmax(xB_alpha_r[idx] - P->xB_eps, 0.0);
                        if (capacity <= 0.0) continue;
                        double mass_take = excess_mass * (capacity / shell_removal_sum);
                        double dxB = mass_take / fmax(h_alpha, 1.0e-30);
                        xB_alpha_r[idx] = clamp_eps(xB_alpha_r[idx] - dxB, P->xB_eps);
                        if (Y_r) {
                            Y_r[idx] = logit_from_fraction(xB_alpha_r[idx], P->xB_eps, P->Y_clip);
                        }
                    }
                }
            }
            mean_after_adjust = host_mean_xBtot_gp(phi_r, eta_r, xB_alpha_r, P, total_size);
            gp_insert_mass_after = (double)total_size * mean_after_adjust;
            gp_insert_mass_error = gp_insert_mass_after - gp_insert_mass_before;
            gp_insert_compensation_success = (fabs(gp_insert_mass_error) < 1.0e-8) ? 1 : 0;
        }
    }

    recompute_host_xBtot_field(phi_r, eta_r, xB_alpha_r, xBtot_r, P, total_size);

    log_section_header("GP Initialization");
    log_kv_text("gp_init_mode", "%s", P->gp_init_mode);
    log_kv_text("gp_init_mass_mode", "%s", P->gp_init_mass_mode);
    log_kv_text("gp_eta_seed_center", "(%.6f, %.6f, %.6f)", cx, cy, cz);
    log_kv_text("gp_eta_seed_radius", "%.6f", P->gp_eta_seed_radius);
    log_kv_text("gp_eta_seed_peak", "%.6f", P->gp_eta_seed_peak);
    log_kv_text("gp_eta_iface_width", "%.6f", w_eta);
    log_kv_text("mean_xBtot_gp_before_seed", "%.8e", mean_before);
    log_kv_text("mean_xBtot_gp_after_seed", "%.8e", mean_after_seed);
    log_kv_text("delta_mean_xBtot_gp_seed", "%.8e", mean_after_seed - mean_before);
    if (strcmp(P->gp_init_mass_mode, "adjust_background") == 0) {
        log_kv_text("gp_init_adjust_clip_count", "%lld", clip_count);
        log_kv_text("mean_xBtot_gp_after_adjust", "%.8e", mean_after_adjust);
        log_kv_text("delta_mean_xBtot_gp_adjust", "%.8e", mean_after_adjust - mean_before);
    } else if (strcmp(P->gp_init_mass_mode, "local_compensate") == 0) {
        log_kv_text("gp_insert_mass_before", "%.8e", gp_insert_mass_before);
        log_kv_text("gp_insert_mass_after", "%.8e", gp_insert_mass_after);
        log_kv_text("gp_insert_mass_error", "%.8e", gp_insert_mass_error);
        log_kv_text("gp_insert_compensation_success", "%d", gp_insert_compensation_success);
        log_kv_text("gp_insert_eta_scaled_factor", "%.8e", gp_insert_eta_scaled_factor);
    }
    if (strcmp(P->gp_eta_mass_limiter, "local_clip") == 0) {
        log_kv_text("eta_seed_max_before_limiter", "%.8e", eta_seed_max_before_limiter);
        log_kv_text("eta_seed_max_after_limiter", "%.8e", eta_seed_max_after_limiter);
        log_kv_text("eta_seed_limiter_count", "%lld", eta_seed_limiter_count);
        log_kv_text("eta_seed_scaled_or_limited_mass_change", "%.8e", eta_seed_scaled_or_limited_mass_change);
    }
    return 1;
}

static void launch_compute_model_xBtot_kernel(const PFParams *P,
                                              const double *phi_r,
                                              const double *eta_r,
                                              const double *xB_alpha_r,
                                              double *xBtot_r,
                                              int total_size) {
    if (is_gp_zone_mode(P)) {
        launch_compute_xBtot_gp_kernel(phi_r, eta_r, xB_alpha_r, xBtot_r, P->gp_xB_fixed, total_size);
    } else {
        launch_compute_xBtot_kernel(phi_r, xB_alpha_r, xBtot_r, P->v_B, total_size);
    }
}

typedef struct {
    int event_triggered;
    int compensation_success;
    int center_i, center_j, center_k;
    double center_x, center_y, center_z;
    double local_xB_alpha;
    double local_xBtot_gp;
    double delta_g_nuc;
    double drive;
    double deltaG_star_J;
    double J_rate;
    double P_event;
    double mass_before;
    double mass_after_raw;
    double mass_after_comp;
    double mass_error_raw;
    double mass_error_comp;
    double xB_clip_count_event;
    double eta_inserted_volume;
    double seed_scale;
} GpNucStepResult;

typedef struct {
    int event_triggered;
    int compensation_success;
    int stochastic_enabled;
    int event_accepted;
    int stop_after_conversion_audit_requested;
    int feasibility_checked;
    int throttling_checked;
    int event_rejected_infeasible;
    int seed_scaled_due_to_capacity;
    int cooldown_or_spacing_rejected;
    int rejected_cooldown;
    int rejected_spacing;
    int rejected_window_limit;
    int rejected_global_limit;
    int accepted_events_in_window;
    int accepted_events_total;
    int center_i, center_j, center_k;
    double center_x, center_y, center_z;
    double eta_max;
    double R_eff_GP;
    double local_xB_alpha;
    double local_xBtot_gp;
    double drive_beta_given_GP;
    double deltaG_star_J;
    double J_site;
    double P_event;
    double random_u;
    double phi_seed_radius;
    double patch_radius;
    double mass_before;
    double mass_after_raw;
    double mass_after_comp;
    double mass_error_raw;
    double mass_error_comp;
    double shell_capacity_add;
    double shell_capacity_remove;
    double capacity_ratio;
    double seed_amplitude_original;
    double seed_amplitude_scaled;
    double xB_clip_count_event;
    double eta_depleted_amount;
    double phi_inserted_volume;
    double xB_shell_min_before_event;
    double xB_shell_max_before_event;
    double nearest_event_distance;
    double steps_since_nearest_event;
} GpToBetaStepResult;

typedef struct {
    double sum_xBtot;
    double mean_xBtot;
    double mean_xB;
    double min_xB;
    double max_xB;
    double mean_phi;
    double min_phi;
    double max_phi;
    double mean_eta;
    double min_eta;
    double max_eta;
    double mean_h_gp;
    double mean_h_beta;
    double patch_mass;
    double storage_residual_mean;
    double storage_residual_max_abs;
    double storage_residual_sum;
} GpToBetaAuditStats;

static inline double host_xBtot_gp_point_value(double phi, double eta, double xB_alpha,
                                               const PFParams *P) {
    double h_alpha = 0.0, h_GP = 0.0, h_beta = 0.0;
    phase_fractions_gp(clamp01_local(phi), clamp01_local(eta), &h_alpha, &h_GP, &h_beta);
    return h_alpha * clamp_eps(xB_alpha, P->xB_eps) + h_GP * P->gp_xB_fixed + h_beta;
}

static void compute_gp_to_beta_audit_stats_host(const std::vector<double> &phi_field,
                                                const std::vector<double> &eta_field,
                                                const std::vector<double> &xB_field,
                                                const PFParams *P,
                                                int total_r,
                                                double cx, double cy, double cz,
                                                double patch_r,
                                                GpToBetaAuditStats *out) {
    if (!out || !P || total_r <= 0) return;
    memset(out, 0, sizeof(*out));
    out->min_xB = INFINITY;
    out->max_xB = -INFINITY;
    out->min_phi = INFINITY;
    out->max_phi = -INFINITY;
    out->min_eta = INFINITY;
    out->max_eta = -INFINITY;
    out->storage_residual_max_abs = 0.0;
    const double inv_total = 1.0 / (double)total_r;
    const double dV = P->dx * P->dy * P->dz;
    const double Lx = P->Nx * P->dx;
    const double Ly = P->Ny * P->dy;
    const double Lz = P->Nz * P->dz;
    for (int idx = 0; idx < total_r; ++idx) {
        int i = 0, j = 0, k = 0;
        host_decode_index(idx, P->Ny, P->Nz, &i, &j, &k);
        double x = i * P->dx;
        double y = j * P->dy;
        double z = k * P->dz;
        const double phi = clamp01_local(phi_field[(size_t)idx]);
        const double eta = clamp01_local(eta_field[(size_t)idx]);
        const double xB = clamp_eps(xB_field[(size_t)idx], P->xB_eps);
        double h_alpha = 0.0, h_GP = 0.0, h_beta = 0.0;
        phase_fractions_gp(phi, eta, &h_alpha, &h_GP, &h_beta);
        const double xBtot = host_xBtot_gp_point_value(phi, eta, xB, P);
        const double residual = xBtot - host_xBtot_gp_point_value(phi, eta, xB, P);
        out->sum_xBtot += xBtot;
        out->mean_xB += xB;
        out->mean_phi += phi;
        out->mean_eta += eta;
        out->mean_h_gp += h_GP;
        out->mean_h_beta += h_beta;
        out->storage_residual_sum += residual;
        out->storage_residual_max_abs = fmax(out->storage_residual_max_abs, fabs(residual));
        if (xB < out->min_xB) out->min_xB = xB;
        if (xB > out->max_xB) out->max_xB = xB;
        if (phi < out->min_phi) out->min_phi = phi;
        if (phi > out->max_phi) out->max_phi = phi;
        if (eta < out->min_eta) out->min_eta = eta;
        if (eta > out->max_eta) out->max_eta = eta;
        double rx = periodic_delta(x, cx, Lx);
        double ry = periodic_delta(y, cy, Ly);
        double rz = periodic_delta(z, cz, Lz);
        double r = sqrt(rx * rx + ry * ry + rz * rz);
        if (r <= patch_r) {
            out->patch_mass += xBtot * dV;
        }
    }
    out->mean_xBtot = out->sum_xBtot * inv_total;
    out->mean_xB *= inv_total;
    out->mean_phi *= inv_total;
    out->mean_eta *= inv_total;
    out->mean_h_gp *= inv_total;
    out->mean_h_beta *= inv_total;
    out->storage_residual_mean = out->storage_residual_sum * inv_total;
}

static inline double gp_to_beta_seed_profile_value(double r, const PFParams *P) {
    const double w = fmax(P->gp_to_beta_seed_iface_width, 1.0e-30);
    return clamp01_local(P->gp_to_beta_seed_peak *
                         0.5 * (1.0 - tanh((r - P->gp_to_beta_seed_radius) / w)));
}

static inline double gp_nuc_seed_profile_value(double r, const PFParams *P, double seed_scale) {
    const double w = fmax(P->gp_nuc_seed_iface_width, 1.0e-30);
    return clamp01_local(seed_scale * P->gp_nuc_seed_peak *
                         0.5 * (1.0 - tanh((r - P->gp_nuc_seed_radius) / w)));
}

static double compute_gp_to_beta_drive_local_simple_host(double local_xB_alpha,
                                                         double local_xBtot_gp,
                                                         double eta_max,
                                                         double R_eff_GP,
                                                         const PFParams *P) {
    double x_term = fmax(local_xBtot_gp - fmax(P->gp_to_beta_xB_threshold, 0.0), 0.0);
    double eta_term = fmax(eta_max - fmax(P->gp_to_beta_eta_threshold, 0.0), 0.0);
    double r_term = fmax(R_eff_GP - fmax(P->gp_to_beta_radius_threshold, 0.0), 0.0);
    (void)local_xB_alpha;
    return x_term + eta_term + r_term;
}

static int apply_gp_nucleation_event_cpu(GpNucRuntime *rt, const PFParams *P, int step,
                                         double *d_phi_r, double *d_eta_r, double *d_Y_r, double *d_xB_r,
                                         int total_r, size_t size_r, GpNucStepResult *result_out) {
    if (result_out) memset(result_out, 0, sizeof(*result_out));
    if (!rt || !P || !is_gp_zone_mode(P) || !P->gp_nuc_enabled || P->mode != 0) return 1;
    if (P->gp_nuc_check_interval <= 0 || (step % P->gp_nuc_check_interval) != 0) return 1;
    if (P->gp_nuc_max_events_per_step <= 0) return 1;

    std::vector<double> phi((size_t)total_r), eta((size_t)total_r), xB((size_t)total_r), Y((size_t)total_r);
    CUDA_CHECK(cudaMemcpy(phi.data(), d_phi_r, size_r, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(eta.data(), d_eta_r, size_r, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(xB.data(), d_xB_r, size_r, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Y.data(), d_Y_r, size_r, cudaMemcpyDeviceToHost));

    double temperature_K = P->temperature_C + 273.15;
    if (temperature_K < 1.0) temperature_K = 1.0;
    const double dx_phys_m = (P->ic_phi_iface_w > 1e-30) ? (P->lambda_sm_m / (2.0 * P->ic_phi_iface_w)) : (P->dx * 1.0e-9);
    const double dy_phys_m = (P->ic_phi_iface_w > 1e-30) ? (P->lambda_sm_m / (2.0 * P->ic_phi_iface_w)) : (P->dy * 1.0e-9);
    const double dz_phys_m = (P->ic_phi_iface_w > 1e-30) ? (P->lambda_sm_m / (2.0 * P->ic_phi_iface_w)) : (P->dz * 1.0e-9);
    const double dt_real_s = fmax(P->dt * P->t_real_unit, 1.0e-30);
    const double dV_phys = fmax(dx_phys_m * dy_phys_m * dz_phys_m, 1.0e-30);
    const double dV = P->dx * P->dy * P->dz;
    const double Lx = P->Nx * P->dx;
    const double Ly = P->Ny * P->dy;
    const double Lz = P->Nz * P->dz;
    const double gamma_gp = fmax(P->gp_nuc_gamma, 0.0);
    const double J0_gp = fmax(P->gp_nuc_J0, 0.0);
    const double kB = 1.380649e-23;

    int best_idx = -1;
    double best_margin = -1.0;
    double best_delta_g = 0.0;
    double best_drive = 0.0;
    double best_dGstar = 0.0;
    double best_J = 0.0;
    double best_P = 0.0;
    for (int idx = 0; idx < total_r; ++idx) {
        double phi_local = clamp01_local(phi[idx]);
        double eta_local = clamp01_local(eta[idx]);
        double xB_local = clamp_eps(xB[idx], P->xB_eps);
        double h_alpha = 0.0, h_GP = 0.0, h_beta = 0.0;
        phase_fractions_gp(phi_local, eta_local, &h_alpha, &h_GP, &h_beta);
        if (!(h_alpha > P->gp_nuc_h_alpha_threshold)) continue;
        if (!(phi_local < P->gp_nuc_phi_threshold)) continue;
        if (!(eta_local < P->gp_nuc_eta_threshold)) continue;

        double delta_g = compute_delta_g_nuc_GP_host(xB_local, temperature_K,
                                                     P->mu_reference_scale, P->gp_xB_fixed);
        double drive = -delta_g;
        if (!(drive > 0.0)) continue;
        double Vm_phys = compute_local_Vm_alpha_phys_host(xB_local, P);
        double drive_vol = drive / Vm_phys;
        if (!(drive_vol > 0.0)) continue;
        double deltaG_star = (16.0 * M_PI * gamma_gp * gamma_gp * gamma_gp) /
                             fmax(3.0 * drive_vol * drive_vol, 1.0e-300);
        double expo = -deltaG_star / (kB * temperature_K);
        if (expo < -700.0) expo = -700.0;
        double J_rate = J0_gp * exp(expo);
        double lambda = J_rate * dt_real_s * dV_phys;
        double P_event = 1.0 - exp(-fmin(lambda, 700.0));
        if (!(P_event > 0.0)) continue;

        uint64_t key = ((uint64_t)P->seed << 32) ^
                       ((uint64_t)(unsigned int)step * 0x9e3779b97f4a7c15ULL) ^
                       ((uint64_t)(unsigned int)idx * 0xbf58476d1ce4e5b9ULL);
        double u = uniform01_from_key_host(key);
        if (u >= P_event) continue;
        double margin = P_event - u;
        if (margin > best_margin) {
            best_margin = margin;
            best_idx = idx;
            best_delta_g = delta_g;
            best_drive = drive;
            best_dGstar = deltaG_star;
            best_J = J_rate;
            best_P = P_event;
        }
    }
    if (best_idx < 0) return 1;

    int ic = 0, jc = 0, kc = 0;
    host_decode_index(best_idx, P->Ny, P->Nz, &ic, &jc, &kc);
    const double cx = ic * P->dx;
    const double cy = jc * P->dy;
    const double cz = kc * P->dz;
    const double patch_r = P->gp_nuc_patch_radius;
    const double shell_inner = P->gp_nuc_shell_inner_radius;
    const double shell_outer = P->gp_nuc_shell_outer_radius;

    auto compute_patch_mass = [&](const std::vector<double> &eta_field,
                                  const std::vector<double> &xB_field) {
        double mass = 0.0;
        for (int idx = 0; idx < total_r; ++idx) {
            int i = 0, j = 0, k = 0;
            host_decode_index(idx, P->Ny, P->Nz, &i, &j, &k);
            double xcoord = i * P->dx;
            double ycoord = j * P->dy;
            double zcoord = k * P->dz;
            double rx = periodic_delta(xcoord, cx, Lx);
            double ry = periodic_delta(ycoord, cy, Ly);
            double rz = periodic_delta(zcoord, cz, Lz);
            double r = sqrt(rx * rx + ry * ry + rz * rz);
            if (r > patch_r) continue;
            mass += host_xBtot_gp_point_value(phi[idx], eta_field[idx], xB_field[idx], P) * dV;
        }
        return mass;
    };
    auto compute_shell_capacity = [&](const std::vector<double> &eta_field,
                                      const std::vector<double> &xB_field,
                                      int remove_mass) {
        double capacity = 0.0;
        for (int idx = 0; idx < total_r; ++idx) {
            int i = 0, j = 0, k = 0;
            host_decode_index(idx, P->Ny, P->Nz, &i, &j, &k);
            double xcoord = i * P->dx;
            double ycoord = j * P->dy;
            double zcoord = k * P->dz;
            double rx = periodic_delta(xcoord, cx, Lx);
            double ry = periodic_delta(ycoord, cy, Ly);
            double rz = periodic_delta(zcoord, cz, Lz);
            double r = sqrt(rx * rx + ry * ry + rz * rz);
            if (r < shell_inner || r > shell_outer || r > patch_r) continue;
            double h_alpha = 0.0;
            phase_fractions_gp(clamp01_local(phi[idx]), clamp01_local(eta_field[idx]), &h_alpha, NULL, NULL);
            if (h_alpha <= P->gp_h_alpha_eps) continue;
            if (remove_mass) {
                capacity += h_alpha * fmax(xB_field[idx] - P->xB_eps, 0.0) * dV;
            } else {
                capacity += h_alpha * fmax((1.0 - P->xB_eps) - xB_field[idx], 0.0) * dV;
            }
        }
        return capacity;
    };
    auto mass_change_is_feasible = [&](double delta_mass,
                                       const std::vector<double> &eta_field,
                                       const std::vector<double> &xB_field) {
        if (fabs(delta_mass) <= 1.0e-14) return true;
        if (delta_mass > 0.0) {
            return compute_shell_capacity(eta_field, xB_field, 1) + 1.0e-14 >= delta_mass;
        }
        return compute_shell_capacity(eta_field, xB_field, 0) + 1.0e-14 >= -delta_mass;
    };
    auto build_seeded_eta = [&](double seed_scale, std::vector<double> *eta_out, double *eta_inserted_volume_out) {
        double eta_inserted = 0.0;
        for (int idx = 0; idx < total_r; ++idx) {
            int i = 0, j = 0, k = 0;
            host_decode_index(idx, P->Ny, P->Nz, &i, &j, &k);
            double xcoord = i * P->dx;
            double ycoord = j * P->dy;
            double zcoord = k * P->dz;
            double rx = periodic_delta(xcoord, cx, Lx);
            double ry = periodic_delta(ycoord, cy, Ly);
            double rz = periodic_delta(zcoord, cz, Lz);
            double r = sqrt(rx * rx + ry * ry + rz * rz);
            double eta_seed = gp_nuc_seed_profile_value(r, P, seed_scale);
            double eta_old = clamp01_local(eta[idx]);
            double eta_new = fmax(eta_old, eta_seed);
            (*eta_out)[idx] = eta_new;
            eta_inserted += fmax(h_of_phi(eta_new) - h_of_phi(eta_old), 0.0) * dV;
        }
        if (eta_inserted_volume_out) *eta_inserted_volume_out = eta_inserted;
    };

    const double local_xB_alpha = clamp_eps(xB[best_idx], P->xB_eps);
    const double local_xBtot_gp = host_xBtot_gp_point_value(phi[best_idx], eta[best_idx], xB[best_idx], P);
    const double M_before = compute_patch_mass(eta, xB);
    std::vector<double> eta_new = eta;
    std::vector<double> xB_new = xB;
    double seed_scale = 1.0;
    double eta_inserted_volume = 0.0;
    build_seeded_eta(seed_scale, &eta_new, &eta_inserted_volume);
    double M_after_raw = compute_patch_mass(eta_new, xB);
    if (strcmp(P->gp_nuc_mass_mode, "local_compensate") == 0) {
        double delta_raw = M_after_raw - M_before;
        if (!mass_change_is_feasible(delta_raw, eta_new, xB_new)) {
            double lo = 0.0, hi = 1.0;
            for (int iter = 0; iter < 32; ++iter) {
                double mid = 0.5 * (lo + hi);
                std::vector<double> eta_mid = eta;
                double eta_inserted_mid = 0.0;
                build_seeded_eta(mid, &eta_mid, &eta_inserted_mid);
                double M_mid = compute_patch_mass(eta_mid, xB);
                double delta_mid = M_mid - M_before;
                if (mass_change_is_feasible(delta_mid, eta_mid, xB_new)) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            seed_scale = lo;
            build_seeded_eta(seed_scale, &eta_new, &eta_inserted_volume);
            M_after_raw = compute_patch_mass(eta_new, xB);
        }
    }

    double M_after_comp = M_after_raw;
    int compensation_success = 0;
    long long xB_clip_count_event = 0;
    if (strcmp(P->gp_nuc_mass_mode, "local_compensate") == 0) {
        std::vector<int> active;
        std::vector<double> weight;
        for (int idx = 0; idx < total_r; ++idx) {
            int i = 0, j = 0, k = 0;
            host_decode_index(idx, P->Ny, P->Nz, &i, &j, &k);
            double xcoord = i * P->dx;
            double ycoord = j * P->dy;
            double zcoord = k * P->dz;
            double rx = periodic_delta(xcoord, cx, Lx);
            double ry = periodic_delta(ycoord, cy, Ly);
            double rz = periodic_delta(zcoord, cz, Lz);
            double r = sqrt(rx * rx + ry * ry + rz * rz);
            if (r < shell_inner || r > shell_outer || r > patch_r) continue;
            double h_alpha = 0.0;
            phase_fractions_gp(clamp01_local(phi[idx]), clamp01_local(eta_new[idx]), &h_alpha, NULL, NULL);
            if (h_alpha <= P->gp_h_alpha_eps) continue;
            active.push_back(idx);
            weight.push_back(h_alpha);
        }
        double remaining = M_before - M_after_raw;
        std::vector<int> current = active;
        std::vector<double> current_w = weight;
        for (int iter = 0; iter < total_r && !current.empty() && fabs(remaining) > 1.0e-12; ++iter) {
            double denom = 0.0;
            for (size_t n = 0; n < current.size(); ++n) denom += current_w[n] * current_w[n] * dV;
            if (denom <= 1.0e-30) break;
            double lambda = remaining / denom;
            int any_bound = 0;
            std::vector<int> next;
            std::vector<double> next_w;
            for (size_t n = 0; n < current.size(); ++n) {
                int idx = current[n];
                double w = current_w[n];
                double trial = xB_new[idx] + lambda * w;
                double bounded = clamp_eps(trial, P->xB_eps);
                if (fabs(bounded - trial) > 0.0) {
                    any_bound = 1;
                    ++xB_clip_count_event;
                    remaining -= w * (bounded - xB_new[idx]) * dV;
                    xB_new[idx] = bounded;
                } else {
                    next.push_back(idx);
                    next_w.push_back(w);
                }
            }
            if (!any_bound) {
                for (size_t n = 0; n < current.size(); ++n) xB_new[current[n]] += lambda * current_w[n];
                remaining = 0.0;
                break;
            }
            current.swap(next);
            current_w.swap(next_w);
        }
        for (int idx = 0; idx < total_r; ++idx) {
            xB_new[idx] = clamp_eps(xB_new[idx], P->xB_eps);
            Y[idx] = logit_from_fraction(xB_new[idx], P->xB_eps, P->Y_clip);
        }
        M_after_comp = compute_patch_mass(eta_new, xB_new);
        compensation_success = (fabs(M_after_comp - M_before) <= fmax(1.0e-10, 1.0e-8 * fabs(M_before))) ? 1 : 0;
    }

    CUDA_CHECK(cudaMemcpy(d_eta_r, eta_new.data(), size_r, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xB_r, xB_new.data(), size_r, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y_r, Y.data(), size_r, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    if (rt->events_csv) {
        rt->event_counter += 1;
        fprintf(rt->events_csv,
                "%d,%d,%d,%d,%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%d,%lld,%.10e\n",
                step, rt->event_counter, ic, jc, kc, cx, cy, cz,
                local_xB_alpha, local_xBtot_gp, best_delta_g, best_drive, best_dGstar, best_J, best_P,
                M_before, M_after_raw, M_after_comp, compensation_success, xB_clip_count_event, seed_scale);
        fflush(rt->events_csv);
    }
    if (result_out) {
        result_out->event_triggered = 1;
        result_out->compensation_success = compensation_success;
        result_out->center_i = ic;
        result_out->center_j = jc;
        result_out->center_k = kc;
        result_out->center_x = cx;
        result_out->center_y = cy;
        result_out->center_z = cz;
        result_out->local_xB_alpha = local_xB_alpha;
        result_out->local_xBtot_gp = local_xBtot_gp;
        result_out->delta_g_nuc = best_delta_g;
        result_out->drive = best_drive;
        result_out->deltaG_star_J = best_dGstar;
        result_out->J_rate = best_J;
        result_out->P_event = best_P;
        result_out->mass_before = M_before;
        result_out->mass_after_raw = M_after_raw;
        result_out->mass_after_comp = M_after_comp;
        result_out->mass_error_raw = M_after_raw - M_before;
        result_out->mass_error_comp = M_after_comp - M_before;
        result_out->xB_clip_count_event = (double)xB_clip_count_event;
        result_out->eta_inserted_volume = eta_inserted_volume;
        result_out->seed_scale = seed_scale;
    }
    printf("[GP-NUC] step=%d center=(%d,%d,%d) xB_local=%.6e xBtot_local=%.6e delta_g=%.6e drive=%.6e P=%.6e M_before=%.10e M_after_raw=%.10e M_after_comp=%.10e success=%d clip=%lld seed_scale=%.6e\n",
           step, ic, jc, kc, local_xB_alpha, local_xBtot_gp, best_delta_g, best_drive, best_P,
           M_before, M_after_raw, M_after_comp, compensation_success, xB_clip_count_event, seed_scale);
    return 1;
}

static int apply_gp_to_beta_event_cpu(GpToBetaRuntime *rt, const PFParams *P, int step,
                                      double *d_phi_r, double *d_eta_r, double *d_Y_r, double *d_xB_r,
                                      int total_r, size_t size_r, GpToBetaStepResult *result_out) {
    if (result_out) memset(result_out, 0, sizeof(*result_out));
    if (!rt || !P || !is_gp_zone_mode(P) || !P->gp_to_beta_enabled || P->mode != 0) return 1;
    if (P->gp_to_beta_check_interval <= 0 || (step % P->gp_to_beta_check_interval) != 0) return 1;
    if (P->gp_to_beta_max_events_per_step <= 0) return 1;

    std::vector<double> phi((size_t)total_r), eta((size_t)total_r), xB((size_t)total_r), Y((size_t)total_r);
    CUDA_CHECK(cudaMemcpy(phi.data(), d_phi_r, size_r, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(eta.data(), d_eta_r, size_r, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(xB.data(), d_xB_r, size_r, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Y.data(), d_Y_r, size_r, cudaMemcpyDeviceToHost));

    int idx_max = -1;
    double eta_max = -1.0;
    for (int idx = 0; idx < total_r; ++idx) {
        double e = clamp01_local(eta[idx]);
        if (e > eta_max) {
            eta_max = e;
            idx_max = idx;
        }
    }
    if (idx_max < 0 || !(eta_max > P->gp_to_beta_eta_threshold)) return 1;
    if (clamp01_local(phi[idx_max]) >= 0.5) return 1;

    int ic = 0, jc = 0, kc = 0;
    host_decode_index(idx_max, P->Ny, P->Nz, &ic, &jc, &kc);
    const double cx = ic * P->dx;
    const double cy = jc * P->dy;
    const double cz = kc * P->dz;
    const double Lx = P->Nx * P->dx;
    const double Ly = P->Ny * P->dy;
    const double Lz = P->Nz * P->dz;
    const double dV = P->dx * P->dy * P->dz;
    const double patch_r = P->gp_to_beta_patch_radius;
    const double shell_inner = P->gp_to_beta_shell_inner_radius;
    const double shell_outer = P->gp_to_beta_shell_outer_radius;

    double V_GP_patch = 0.0;
    double M_before = 0.0;
    for (int idx = 0; idx < total_r; ++idx) {
        int i = 0, j = 0, k = 0;
        host_decode_index(idx, P->Ny, P->Nz, &i, &j, &k);
        double x = i * P->dx;
        double y = j * P->dy;
        double z = k * P->dz;
        double rx = periodic_delta(x, cx, Lx);
        double ry = periodic_delta(y, cy, Ly);
        double rz = periodic_delta(z, cz, Lz);
        double r = sqrt(rx * rx + ry * ry + rz * rz);
        if (r > patch_r) continue;
        double h_alpha = 0.0, h_GP = 0.0, h_beta = 0.0;
        phase_fractions_gp(clamp01_local(phi[idx]), clamp01_local(eta[idx]), &h_alpha, &h_GP, &h_beta);
        V_GP_patch += h_GP * dV;
        M_before += host_xBtot_gp_point_value(phi[idx], eta[idx], xB[idx], P) * dV;
    }
    const double R_eff = cbrt(fmax(3.0 * V_GP_patch / (4.0 * M_PI), 0.0));
    const double local_xB_alpha = clamp_eps(xB[idx_max], P->xB_eps);
    const double local_xBtot_gp = host_xBtot_gp_point_value(phi[idx_max], eta[idx_max], xB[idx_max], P);
    if (!(R_eff > P->gp_to_beta_radius_threshold) ||
        !(local_xBtot_gp > P->gp_to_beta_xB_threshold)) {
        return 1;
    }
    int throttling_checked = 0;
    int rejected_cooldown = 0;
    int rejected_spacing = 0;
    int rejected_window_limit = 0;
    int rejected_global_limit = 0;
    int accepted_events_in_window = 0;
    int accepted_events_total = (int)rt->accepted_events.size();
    double nearest_event_distance = NAN;
    double steps_since_nearest_event = NAN;
    const int stochastic_enabled = P->gp_to_beta_stochastic_enabled ? 1 : 0;
    int event_accepted = 1;
    double drive_beta_given_GP = 0.0;
    double deltaG_star_J = 0.0;
    double J_site = 0.0;
    double P_event = 1.0;
    double random_u = 0.0;
    if (stochastic_enabled) {
        double temperature_K = P->temperature_C + 273.15;
        if (temperature_K < 1.0) temperature_K = 1.0;
        const double kB = 1.380649e-23;
        if (strcmp(P->gp_to_beta_drive_mode, "constant") == 0) {
            drive_beta_given_GP = fmax(P->gp_to_beta_drive_const, 0.0);
        } else {
            drive_beta_given_GP = compute_gp_to_beta_drive_local_simple_host(
                local_xB_alpha, local_xBtot_gp, eta_max, R_eff, P);
        }
        if (!(drive_beta_given_GP > 0.0)) {
            event_accepted = 0;
        } else {
            double delta_g_v = drive_beta_given_GP;
            deltaG_star_J = (16.0 * M_PI * pow(fmax(P->gp_to_beta_gamma, 0.0), 3.0)) /
                            fmax(3.0 * delta_g_v * delta_g_v, 1.0e-300);
            double expo = -deltaG_star_J / (kB * temperature_K);
            if (expo < -700.0) expo = -700.0;
            J_site = fmax(P->gp_to_beta_J0_site, 0.0) * exp(expo);
            P_event = 1.0 - exp(-fmin(J_site * fmax(P->dt * P->t_real_unit, 1.0e-30), 700.0));
            uint64_t key = ((uint64_t)P->seed << 32) ^
                           ((uint64_t)(unsigned int)step * 0xd6e8feb86659fd93ULL) ^
                           ((uint64_t)(unsigned int)idx_max * 0xa0761d6478bd642fULL);
            random_u = uniform01_from_key_host(key);
            event_accepted = (random_u < P_event) ? 1 : 0;
        }
        if (!event_accepted) {
            if (rt->events_csv) {
                rt->event_counter += 1;
                fprintf(rt->events_csv, "%d,%d,%d,%d,%d,", step, rt->event_counter, ic, jc, kc);
                fprintf(rt->events_csv, "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,",
                        cx, cy, cz, eta_max, R_eff, local_xB_alpha, local_xBtot_gp);
                fprintf(rt->events_csv, "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,",
                        P->gp_to_beta_seed_radius, patch_r, 0.0, 0.0, 0.0, 0.0, 0.0);
                fprintf(rt->events_csv, "%d,%d,", stochastic_enabled, event_accepted);
                fprintf(rt->events_csv, "%.10e,%.10e,%.10e,%.10e,%.10e,",
                        drive_beta_given_GP, deltaG_star_J, J_site, P_event, random_u);
                fprintf(rt->events_csv, "%d,%lld,", 0, 0LL);
                fprintf(rt->events_csv, "%d,%.10e,%.10e,%.10e,%.10e,", 0, 0.0, 0.0, 0.0, 0.0);
                fprintf(rt->events_csv, "%d,%.10e,%.10e,%d,%d,%.10e,%.10e,",
                        0, P->gp_to_beta_seed_peak, P->gp_to_beta_seed_peak, 0, 0, 0.0, 0.0);
                fprintf(rt->events_csv, "%d,%d,%d,%d,%d,%.10e,%.10e,%d,%d\n",
                        0, 0, 0, 0, 0, NAN, NAN, 0, accepted_events_total);
                fflush(rt->events_csv);
            }
            if (result_out) {
                result_out->stochastic_enabled = stochastic_enabled;
                result_out->event_accepted = 0;
                result_out->center_i = ic;
                result_out->center_j = jc;
                result_out->center_k = kc;
                result_out->center_x = cx;
                result_out->center_y = cy;
                result_out->center_z = cz;
                result_out->eta_max = eta_max;
                result_out->R_eff_GP = R_eff;
                result_out->local_xB_alpha = local_xB_alpha;
                result_out->local_xBtot_gp = local_xBtot_gp;
                result_out->drive_beta_given_GP = drive_beta_given_GP;
                result_out->deltaG_star_J = deltaG_star_J;
                result_out->J_site = J_site;
                result_out->P_event = P_event;
                result_out->random_u = random_u;
            }
            return 1;
        }
    }

    std::vector<double> phi_new = phi;
    std::vector<double> eta_new = eta;
    std::vector<double> xB_new = xB;
    std::vector<double> xB_pre_clip = xB;
    const double xB_min = P->xB_eps;
    const double xB_max = 1.0 - P->xB_eps;
    int feasibility_checked = 0;
    int event_rejected_infeasible = 0;
    int seed_scaled_due_to_capacity = 0;
    int cooldown_or_spacing_rejected = 0;
    double shell_capacity_add = 0.0;
    double shell_capacity_remove = 0.0;
    double capacity_ratio = NAN;
    const double seed_amplitude_original = P->gp_to_beta_seed_peak;
    double seed_amplitude_scaled = P->gp_to_beta_seed_peak;
    double xB_shell_min_before_event = NAN;
    double xB_shell_max_before_event = NAN;
    {
        throttling_checked = 1;
        double nearest_dist_local = INFINITY;
        int nearest_step_local = -1;
        int recent_window_count = 0;
        for (size_t n = 0; n < rt->accepted_events.size(); ++n) {
            const GpToBetaAcceptedEvent &ev = rt->accepted_events[n];
            const double ex = ev.i * P->dx;
            const double ey = ev.j * P->dy;
            const double ez = ev.k * P->dz;
            const double dxp = periodic_delta(cx, ex, Lx);
            const double dyp = periodic_delta(cy, ey, Ly);
            const double dzp = periodic_delta(cz, ez, Lz);
            const double dist = sqrt(dxp * dxp + dyp * dyp + dzp * dzp);
            if (dist < nearest_dist_local) {
                nearest_dist_local = dist;
                nearest_step_local = ev.step;
            }
            if (P->gp_to_beta_event_window_steps > 0 &&
                (step - ev.step) >= 0 &&
                (step - ev.step) < P->gp_to_beta_event_window_steps) {
                recent_window_count += 1;
            }
        }
        if (!rt->accepted_events.empty()) {
            nearest_event_distance = nearest_dist_local;
            if (nearest_step_local >= 0) {
                steps_since_nearest_event = (double)(step - nearest_step_local);
            }
        }
        accepted_events_in_window = recent_window_count;
        accepted_events_total = (int)rt->accepted_events.size();
        if (P->gp_to_beta_max_events_global > 0 &&
            accepted_events_total >= P->gp_to_beta_max_events_global) {
            rejected_global_limit = 1;
        }
        if (P->gp_to_beta_event_window_steps > 0 &&
            P->gp_to_beta_max_events_per_window > 0 &&
            accepted_events_in_window >= P->gp_to_beta_max_events_per_window) {
            rejected_window_limit = 1;
        }
        if (P->gp_to_beta_event_cooldown_steps > 0 &&
            rt->last_accepted_step > -std::numeric_limits<int>::max() / 2 &&
            (step - rt->last_accepted_step) < P->gp_to_beta_event_cooldown_steps) {
            rejected_cooldown = 1;
        }
        if (isfinite(nearest_event_distance)) {
            if ((P->gp_to_beta_min_event_spacing > 0.0 &&
                 nearest_event_distance < P->gp_to_beta_min_event_spacing) ||
                (P->gp_to_beta_event_exclusion_radius > 0.0 &&
                 nearest_event_distance < P->gp_to_beta_event_exclusion_radius)) {
                rejected_spacing = 1;
            }
        }
        cooldown_or_spacing_rejected = (rejected_cooldown || rejected_spacing) ? 1 : 0;
    }
    GpToBetaAuditStats audit_stats;
    double audit_sum_before = 0.0;
    double audit_sum_prev = 0.0;
    const int audit_event_id = rt->event_counter + 1;
    auto emit_audit_stage = [&](const char *stage,
                                const std::vector<double> &phi_field,
                                const std::vector<double> &eta_field,
                                const std::vector<double> &xB_field,
                                double patch_mass_before_stage,
                                double compensation_mass,
                                double clipped_mass_loss,
                                long long num_clipped_low,
                                long long num_clipped_high,
                                const char *notes) {
        if (!rt->audit_csv || !P->gp_to_beta_conversion_mass_audit_enabled) return;
        compute_gp_to_beta_audit_stats_host(phi_field, eta_field, xB_field, P, total_r,
                                            cx, cy, cz, patch_r, &audit_stats);
        fprintf(rt->audit_csv, "%d,%s,%d,%d,%d,%d,",
                step, stage, audit_event_id, ic, jc, kc);
        fprintf(rt->audit_csv, "%.17e,%.17e,%.17e,%.17e,",
                audit_stats.mean_xBtot,
                audit_stats.sum_xBtot,
                audit_stats.sum_xBtot - audit_sum_prev,
                audit_stats.sum_xBtot - audit_sum_before);
        fprintf(rt->audit_csv, "%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,",
                audit_stats.mean_xB,
                audit_stats.min_xB,
                audit_stats.max_xB,
                audit_stats.mean_phi,
                audit_stats.min_phi,
                audit_stats.max_phi,
                audit_stats.mean_eta,
                audit_stats.min_eta,
                audit_stats.max_eta);
        fprintf(rt->audit_csv, "%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,",
                audit_stats.mean_h_gp,
                audit_stats.mean_h_beta,
                patch_mass_before_stage,
                audit_stats.patch_mass,
                audit_stats.patch_mass - patch_mass_before_stage,
                compensation_mass);
        fprintf(rt->audit_csv, "%.17e,%lld,%lld,%.17e,%.17e,%.17e,%s\n",
                clipped_mass_loss,
                num_clipped_low,
                num_clipped_high,
                audit_stats.storage_residual_mean,
                audit_stats.storage_residual_max_abs,
                audit_stats.storage_residual_sum,
                (notes ? notes : ""));
        fflush(rt->audit_csv);
        audit_sum_prev = audit_stats.sum_xBtot;
    };
    if (rejected_cooldown || rejected_spacing || rejected_window_limit || rejected_global_limit) {
        if (rejected_cooldown) rt->count_events_rejected_cooldown += 1;
        if (rejected_spacing) rt->count_events_rejected_spacing += 1;
        if (rejected_window_limit) rt->count_events_rejected_window += 1;
        if (rejected_global_limit) rt->count_events_rejected_global += 1;
        if (rt->events_csv) {
            rt->event_counter += 1;
            fprintf(rt->events_csv, "%d,%d,%d,%d,%d,", step, rt->event_counter, ic, jc, kc);
            fprintf(rt->events_csv, "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,",
                    cx, cy, cz, eta_max, R_eff, local_xB_alpha, local_xBtot_gp);
            fprintf(rt->events_csv, "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,",
                    P->gp_to_beta_seed_radius, patch_r, 0.0, 0.0, 0.0, 0.0, 0.0);
            fprintf(rt->events_csv, "%d,%d,", stochastic_enabled, 0);
            fprintf(rt->events_csv, "%.10e,%.10e,%.10e,%.10e,%.10e,",
                    drive_beta_given_GP, deltaG_star_J, J_site, P_event, random_u);
            fprintf(rt->events_csv, "%d,%lld,", 0, 0LL);
            fprintf(rt->events_csv, "%d,%.10e,%.10e,%.10e,%.10e,",
                    0, 0.0, 0.0, 0.0, NAN);
            fprintf(rt->events_csv, "%d,%.10e,%.10e,%d,%d,%.10e,%.10e,",
                    0, seed_amplitude_original, seed_amplitude_original, 0,
                    cooldown_or_spacing_rejected, NAN, NAN);
            fprintf(rt->events_csv, "%d,%d,%d,%d,%d,%.10e,%.10e,%d,%d\n",
                    throttling_checked, rejected_cooldown, rejected_spacing,
                    rejected_window_limit, rejected_global_limit,
                    nearest_event_distance, steps_since_nearest_event,
                    accepted_events_in_window, accepted_events_total);
            fflush(rt->events_csv);
        }
        if (result_out) {
            result_out->stochastic_enabled = stochastic_enabled;
            result_out->event_accepted = 0;
            result_out->throttling_checked = throttling_checked;
            result_out->rejected_cooldown = rejected_cooldown;
            result_out->rejected_spacing = rejected_spacing;
            result_out->rejected_window_limit = rejected_window_limit;
            result_out->rejected_global_limit = rejected_global_limit;
            result_out->cooldown_or_spacing_rejected = cooldown_or_spacing_rejected;
            result_out->accepted_events_in_window = accepted_events_in_window;
            result_out->accepted_events_total = accepted_events_total;
            result_out->nearest_event_distance = nearest_event_distance;
            result_out->steps_since_nearest_event = steps_since_nearest_event;
            result_out->center_i = ic;
            result_out->center_j = jc;
            result_out->center_k = kc;
            result_out->center_x = cx;
            result_out->center_y = cy;
            result_out->center_z = cz;
            result_out->eta_max = eta_max;
            result_out->R_eff_GP = R_eff;
            result_out->local_xB_alpha = local_xB_alpha;
            result_out->local_xBtot_gp = local_xBtot_gp;
            result_out->drive_beta_given_GP = drive_beta_given_GP;
            result_out->deltaG_star_J = deltaG_star_J;
            result_out->J_site = J_site;
            result_out->P_event = P_event;
            result_out->random_u = random_u;
        }
        return 1;
    }
    if (rt->audit_csv && P->gp_to_beta_conversion_mass_audit_enabled) {
        compute_gp_to_beta_audit_stats_host(phi, eta, xB, P, total_r, cx, cy, cz, patch_r, &audit_stats);
        audit_sum_before = audit_stats.sum_xBtot;
        audit_sum_prev = audit_sum_before;
        emit_audit_stage("before_conversion", phi, eta, xB, M_before, 0.0, 0.0, 0, 0, "");
    }
    auto build_seeded_fields = [&](double seed_scale,
                                   std::vector<double> *phi_out,
                                   std::vector<double> *eta_out,
                                   double *eta_dep_out,
                                   double *phi_vol_out) {
        double eta_dep = 0.0;
        double phi_vol = 0.0;
        for (int idx = 0; idx < total_r; ++idx) {
            int i = 0, j = 0, k = 0;
            host_decode_index(idx, P->Ny, P->Nz, &i, &j, &k);
            double x = i * P->dx;
            double y = j * P->dy;
            double z = k * P->dz;
            double rx = periodic_delta(x, cx, Lx);
            double ry = periodic_delta(y, cy, Ly);
            double rz = periodic_delta(z, cz, Lz);
            double r = sqrt(rx * rx + ry * ry + rz * rz);
            double phi_seed = clamp01_local(seed_scale * gp_to_beta_seed_profile_value(r, P));
            double phi_old = clamp01_local(phi[idx]);
            double eta_old = clamp01_local(eta[idx]);
            double phi_after = (strcmp(P->gp_to_beta_phi_insert_mode, "max") == 0)
                                   ? fmax(phi_old, phi_seed)
                                   : phi_old;
            double eta_after = eta_old;
            if (strcmp(P->gp_to_beta_eta_deplete_mode, "multiply_1_minus_hphi_seed") == 0) {
                eta_after = eta_old * (1.0 - h_of_phi(phi_seed));
            }
            (*phi_out)[idx] = clamp01_local(phi_after);
            (*eta_out)[idx] = clamp01_local(eta_after);
            eta_dep += fmax(eta_old - eta_after, 0.0) * dV;
            phi_vol += fmax(h_of_phi(phi_after) - h_of_phi(phi_old), 0.0) * dV;
        }
        if (eta_dep_out) *eta_dep_out = eta_dep;
        if (phi_vol_out) *phi_vol_out = phi_vol;
    };
    auto compute_patch_mass = [&](const std::vector<double> &phi_field,
                                  const std::vector<double> &eta_field,
                                  const std::vector<double> &xB_field) {
        double mass = 0.0;
        for (int idx = 0; idx < total_r; ++idx) {
            int i = 0, j = 0, k = 0;
            host_decode_index(idx, P->Ny, P->Nz, &i, &j, &k);
            double x = i * P->dx;
            double y = j * P->dy;
            double z = k * P->dz;
            double rx = periodic_delta(x, cx, Lx);
            double ry = periodic_delta(y, cy, Ly);
            double rz = periodic_delta(z, cz, Lz);
            double r = sqrt(rx * rx + ry * ry + rz * rz);
            if (r > patch_r) continue;
            mass += host_xBtot_gp_point_value(phi_field[idx], eta_field[idx], xB_field[idx], P) * dV;
        }
        return mass;
    };
    auto compute_shell_capacity = [&](const std::vector<double> &phi_field,
                                      const std::vector<double> &eta_field,
                                      const std::vector<double> &xB_field,
                                      int remove_mass) {
        double capacity = 0.0;
        for (int idx = 0; idx < total_r; ++idx) {
            int i = 0, j = 0, k = 0;
            host_decode_index(idx, P->Ny, P->Nz, &i, &j, &k);
            double x = i * P->dx;
            double y = j * P->dy;
            double z = k * P->dz;
            double rx = periodic_delta(x, cx, Lx);
            double ry = periodic_delta(y, cy, Ly);
            double rz = periodic_delta(z, cz, Lz);
            double r = sqrt(rx * rx + ry * ry + rz * rz);
            if (r < shell_inner || r > shell_outer || r > patch_r) continue;
            double h_alpha = 0.0;
            phase_fractions_gp(clamp01_local(phi_field[idx]), clamp01_local(eta_field[idx]), &h_alpha, NULL, NULL);
            if (h_alpha <= P->gp_h_alpha_eps) continue;
            if (remove_mass) {
                capacity += h_alpha * fmax(xB_field[idx] - xB_min, 0.0) * dV;
            } else {
                capacity += h_alpha * fmax(xB_max - xB_field[idx], 0.0) * dV;
            }
        }
        return capacity;
    };
    auto compute_shell_xB_bounds = [&](const std::vector<double> &xB_field,
                                       double *xmin_out,
                                       double *xmax_out) {
        double xmin = INFINITY;
        double xmax = -INFINITY;
        int found = 0;
        for (int idx = 0; idx < total_r; ++idx) {
            int i = 0, j = 0, k = 0;
            host_decode_index(idx, P->Ny, P->Nz, &i, &j, &k);
            double x = i * P->dx;
            double y = j * P->dy;
            double z = k * P->dz;
            double rx = periodic_delta(x, cx, Lx);
            double ry = periodic_delta(y, cy, Ly);
            double rz = periodic_delta(z, cz, Lz);
            double r = sqrt(rx * rx + ry * ry + rz * rz);
            if (r < shell_inner || r > shell_outer || r > patch_r) continue;
            double xv = clamp_eps(xB_field[idx], P->xB_eps);
            if (xv < xmin) xmin = xv;
            if (xv > xmax) xmax = xv;
            found = 1;
        }
        if (!found) {
            xmin = NAN;
            xmax = NAN;
        }
        if (xmin_out) *xmin_out = xmin;
        if (xmax_out) *xmax_out = xmax;
    };
    auto mass_change_is_feasible = [&](double delta_mass,
                                       const std::vector<double> &phi_field,
                                       const std::vector<double> &eta_field,
                                       const std::vector<double> &xB_field) {
        if (fabs(delta_mass) <= 1.0e-14) return true;
        const double capacity = (delta_mass > 0.0)
                                    ? compute_shell_capacity(phi_field, eta_field, xB_field, 1)
                                    : compute_shell_capacity(phi_field, eta_field, xB_field, 0);
        return capacity + 1.0e-14 >= fabs(delta_mass) * P->gp_to_beta_min_shell_capacity_factor;
    };

    double seed_scale = 1.0;
    double eta_depleted_amount = 0.0;
    double phi_inserted_volume = 0.0;
    build_seeded_fields(seed_scale, &phi_new, &eta_new, &eta_depleted_amount, &phi_inserted_volume);
    double M_after_raw = compute_patch_mass(phi_new, eta_new, xB);
    const double mass_error_raw_predicted = M_after_raw - M_before;
    compute_shell_xB_bounds(xB, &xB_shell_min_before_event, &xB_shell_max_before_event);
    shell_capacity_remove = compute_shell_capacity(phi_new, eta_new, xB, 1);
    shell_capacity_add = compute_shell_capacity(phi_new, eta_new, xB, 0);
    if (fabs(mass_error_raw_predicted) <= 1.0e-14) {
        capacity_ratio = INFINITY;
    } else if (mass_error_raw_predicted > 0.0) {
        capacity_ratio = shell_capacity_remove / fabs(mass_error_raw_predicted);
    } else {
        capacity_ratio = shell_capacity_add / fabs(mass_error_raw_predicted);
    }

    if (P->gp_to_beta_feasibility_gate_enabled) {
        feasibility_checked = 1;
        if (strcmp(P->gp_to_beta_mass_mode, "local_compensate") == 0 &&
            !mass_change_is_feasible(mass_error_raw_predicted, phi_new, eta_new, xB_new)) {
            if (P->gp_to_beta_allow_seed_amplitude_scaling) {
                double lo = 0.0, hi = 1.0;
                int found = 0;
                for (int iter = 0; iter < 40; ++iter) {
                    double mid = 0.5 * (lo + hi);
                    std::vector<double> phi_mid = phi;
                    std::vector<double> eta_mid = eta;
                    double eta_dep_mid = 0.0;
                    double phi_vol_mid = 0.0;
                    build_seeded_fields(mid, &phi_mid, &eta_mid, &eta_dep_mid, &phi_vol_mid);
                    double M_mid = compute_patch_mass(phi_mid, eta_mid, xB);
                    double delta_mid = M_mid - M_before;
                    if (mass_change_is_feasible(delta_mid, phi_mid, eta_mid, xB_new)) {
                        found = 1;
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                if (found) {
                    const double amplitude_candidate = lo * P->gp_to_beta_seed_peak;
                    if (amplitude_candidate >= P->gp_to_beta_min_seed_amplitude) {
                        seed_scale = lo;
                        seed_scaled_due_to_capacity = (seed_scale < 1.0 - 1.0e-12) ? 1 : 0;
                        seed_amplitude_scaled = amplitude_candidate;
                        build_seeded_fields(seed_scale, &phi_new, &eta_new, &eta_depleted_amount, &phi_inserted_volume);
                        M_after_raw = compute_patch_mass(phi_new, eta_new, xB);
                        shell_capacity_remove = compute_shell_capacity(phi_new, eta_new, xB, 1);
                        shell_capacity_add = compute_shell_capacity(phi_new, eta_new, xB, 0);
                        if (fabs(M_after_raw - M_before) <= 1.0e-14) {
                            capacity_ratio = INFINITY;
                        } else if ((M_after_raw - M_before) > 0.0) {
                            capacity_ratio = shell_capacity_remove / fabs(M_after_raw - M_before);
                        } else {
                            capacity_ratio = shell_capacity_add / fabs(M_after_raw - M_before);
                        }
                    } else if (P->gp_to_beta_reject_if_infeasible) {
                        event_accepted = 0;
                        event_rejected_infeasible = 1;
                    }
                } else if (P->gp_to_beta_reject_if_infeasible) {
                    event_accepted = 0;
                    event_rejected_infeasible = 1;
                }
            } else if (P->gp_to_beta_reject_if_infeasible) {
                event_accepted = 0;
                event_rejected_infeasible = 1;
            }
        }
    }
    if (seed_amplitude_scaled == P->gp_to_beta_seed_peak) {
        seed_amplitude_scaled = seed_scale * P->gp_to_beta_seed_peak;
    }
    if (!event_accepted) {
        if (event_rejected_infeasible) {
            rt->count_events_rejected_capacity += 1;
        }
        if (rt->events_csv) {
            rt->event_counter += 1;
            fprintf(rt->events_csv, "%d,%d,%d,%d,%d,", step, rt->event_counter, ic, jc, kc);
            fprintf(rt->events_csv, "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,",
                    cx, cy, cz, eta_max, R_eff, local_xB_alpha, local_xBtot_gp);
            fprintf(rt->events_csv, "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,",
                    P->gp_to_beta_seed_radius, patch_r,
                    M_before, M_after_raw, M_after_raw,
                    M_after_raw - M_before, M_after_raw - M_before);
            fprintf(rt->events_csv, "%d,%d,", stochastic_enabled, 0);
            fprintf(rt->events_csv, "%.10e,%.10e,%.10e,%.10e,%.10e,",
                    drive_beta_given_GP, deltaG_star_J, J_site, P_event, random_u);
            fprintf(rt->events_csv, "%d,%lld,", 0, 0LL);
            fprintf(rt->events_csv, "%d,%.10e,%.10e,%.10e,%.10e,",
                    feasibility_checked, mass_error_raw_predicted,
                    shell_capacity_add, shell_capacity_remove, capacity_ratio);
            fprintf(rt->events_csv, "%d,%.10e,%.10e,%d,%d,%.10e,%.10e,",
                    event_rejected_infeasible, seed_amplitude_original, seed_amplitude_scaled,
                    seed_scaled_due_to_capacity, cooldown_or_spacing_rejected,
                    xB_shell_min_before_event, xB_shell_max_before_event);
            fprintf(rt->events_csv, "%d,%d,%d,%d,%d,%.10e,%.10e,%d,%d\n",
                    throttling_checked, rejected_cooldown, rejected_spacing,
                    rejected_window_limit, rejected_global_limit,
                    nearest_event_distance, steps_since_nearest_event,
                    accepted_events_in_window, accepted_events_total);
            fflush(rt->events_csv);
        }
        if (result_out) {
            result_out->stochastic_enabled = stochastic_enabled;
            result_out->event_accepted = 0;
            result_out->throttling_checked = throttling_checked;
            result_out->rejected_cooldown = rejected_cooldown;
            result_out->rejected_spacing = rejected_spacing;
            result_out->rejected_window_limit = rejected_window_limit;
            result_out->rejected_global_limit = rejected_global_limit;
            result_out->accepted_events_in_window = accepted_events_in_window;
            result_out->accepted_events_total = accepted_events_total;
            result_out->nearest_event_distance = nearest_event_distance;
            result_out->steps_since_nearest_event = steps_since_nearest_event;
            result_out->center_i = ic;
            result_out->center_j = jc;
            result_out->center_k = kc;
            result_out->center_x = cx;
            result_out->center_y = cy;
            result_out->center_z = cz;
            result_out->eta_max = eta_max;
            result_out->R_eff_GP = R_eff;
            result_out->local_xB_alpha = local_xB_alpha;
            result_out->local_xBtot_gp = local_xBtot_gp;
            result_out->drive_beta_given_GP = drive_beta_given_GP;
            result_out->deltaG_star_J = deltaG_star_J;
            result_out->J_site = J_site;
            result_out->P_event = P_event;
            result_out->random_u = random_u;
            result_out->feasibility_checked = feasibility_checked;
            result_out->event_rejected_infeasible = event_rejected_infeasible;
            result_out->seed_scaled_due_to_capacity = seed_scaled_due_to_capacity;
            result_out->cooldown_or_spacing_rejected = cooldown_or_spacing_rejected;
            result_out->shell_capacity_add = shell_capacity_add;
            result_out->shell_capacity_remove = shell_capacity_remove;
            result_out->capacity_ratio = capacity_ratio;
            result_out->seed_amplitude_original = seed_amplitude_original;
            result_out->seed_amplitude_scaled = seed_amplitude_scaled;
            result_out->mass_before = M_before;
            result_out->mass_after_raw = M_after_raw;
            result_out->mass_error_raw = M_after_raw - M_before;
            result_out->xB_shell_min_before_event = xB_shell_min_before_event;
            result_out->xB_shell_max_before_event = xB_shell_max_before_event;
        }
        return 1;
    }

    emit_audit_stage("after_phi_eta_update", phi_new, eta_new, xB, M_before, 0.0, 0.0, 0, 0, "");
    emit_audit_stage("after_xB_or_xBtot_update", phi_new, eta_new, xB, M_before, 0.0, 0.0, 0, 0,
                     "same_as_after_phi_eta_update");

    double M_after_comp = M_after_raw;
    int compensation_success = 0;
    long long xB_clip_count_event = 0;
    long long xB_clip_count_low_event = 0;
    long long xB_clip_count_high_event = 0;
    double clipped_mass_loss = 0.0;
    double compensation_mass = 0.0;
    if (strcmp(P->gp_to_beta_mass_mode, "local_compensate") == 0) {
        std::vector<int> active;
        std::vector<double> weight;
        active.reserve((size_t)total_r);
        weight.reserve((size_t)total_r);
        for (int idx = 0; idx < total_r; ++idx) {
            int i = 0, j = 0, k = 0;
            host_decode_index(idx, P->Ny, P->Nz, &i, &j, &k);
            double x = i * P->dx;
            double y = j * P->dy;
            double z = k * P->dz;
            double rx = periodic_delta(x, cx, Lx);
            double ry = periodic_delta(y, cy, Ly);
            double rz = periodic_delta(z, cz, Lz);
            double r = sqrt(rx * rx + ry * ry + rz * rz);
            if (r < shell_inner || r > shell_outer || r > patch_r) continue;
            double h_alpha = 0.0;
            phase_fractions_gp(clamp01_local(phi_new[idx]), clamp01_local(eta_new[idx]), &h_alpha, NULL, NULL);
            if (h_alpha <= P->gp_h_alpha_eps) continue;
            active.push_back(idx);
            weight.push_back(h_alpha);
        }

        double remaining = M_before - M_after_raw;
        std::vector<int> current = active;
        std::vector<double> current_w = weight;
        for (int iter = 0; iter < total_r && !current.empty() && fabs(remaining) > 1.0e-12; ++iter) {
            double denom = 0.0;
            for (size_t n = 0; n < current.size(); ++n) {
                denom += current_w[n] * current_w[n] * dV;
            }
            if (denom <= 1.0e-30) break;
            double lambda = remaining / denom;
            int any_bound = 0;
            std::vector<int> next;
            std::vector<double> next_w;
            next.reserve(current.size());
            next_w.reserve(current_w.size());
            for (size_t n = 0; n < current.size(); ++n) {
                int idx = current[n];
                double w = current_w[n];
                double trial = xB_new[idx] + lambda * w;
                double bounded = trial;
                if (bounded < P->xB_eps) bounded = P->xB_eps;
                if (bounded > 1.0 - P->xB_eps) bounded = 1.0 - P->xB_eps;
                if (fabs(bounded - trial) > 0.0) {
                    any_bound = 1;
                    ++xB_clip_count_event;
                    if (trial < P->xB_eps) ++xB_clip_count_low_event;
                    if (trial > 1.0 - P->xB_eps) ++xB_clip_count_high_event;
                    remaining -= w * (bounded - xB_new[idx]) * dV;
                    xB_new[idx] = bounded;
                } else {
                    next.push_back(idx);
                    next_w.push_back(w);
                }
            }
            if (!any_bound) {
                for (size_t n = 0; n < current.size(); ++n) {
                    int idx = current[n];
                    xB_new[idx] += lambda * current_w[n];
                }
                remaining = 0.0;
                break;
            }
            current.swap(next);
            current_w.swap(next_w);
        }
        xB_pre_clip = xB_new;
        for (int idx = 0; idx < total_r; ++idx) {
            double unclamped = xB_new[idx];
            double clamped = clamp_eps(unclamped, P->xB_eps);
            if (fabs(clamped - unclamped) > 0.0) {
                ++xB_clip_count_event;
                if (unclamped < P->xB_eps) ++xB_clip_count_low_event;
                if (unclamped > 1.0 - P->xB_eps) ++xB_clip_count_high_event;
            }
            xB_new[idx] = clamped;
            Y[idx] = logit_from_fraction(xB_new[idx], P->xB_eps, P->Y_clip);
        }
        M_after_comp = 0.0;
        for (int idx = 0; idx < total_r; ++idx) {
            int i = 0, j = 0, k = 0;
            host_decode_index(idx, P->Ny, P->Nz, &i, &j, &k);
            double x = i * P->dx;
            double y = j * P->dy;
            double z = k * P->dz;
            double rx = periodic_delta(x, cx, Lx);
            double ry = periodic_delta(y, cy, Ly);
            double rz = periodic_delta(z, cz, Lz);
            double r = sqrt(rx * rx + ry * ry + rz * rz);
            if (r > patch_r) continue;
            M_after_comp += host_xBtot_gp_point_value(phi_new[idx], eta_new[idx], xB_new[idx], P) * dV;
        }
        compensation_mass = M_after_comp - M_after_raw;
        compensation_success = (fabs(M_after_comp - M_before) <= fmax(1.0e-10, 1.0e-8 * fabs(M_before))) ? 1 : 0;
    }
    emit_audit_stage("after_compensation", phi_new, eta_new, xB_new, M_before,
                     compensation_mass, 0.0, 0, 0, "");
    emit_audit_stage("after_Y_reconstruction", phi_new, eta_new, xB_new, M_before,
                     compensation_mass, 0.0, 0, 0, "same_as_after_compensation");
    if (rt->audit_csv && P->gp_to_beta_conversion_mass_audit_enabled) {
        GpToBetaAuditStats pre_clip_stats;
        GpToBetaAuditStats post_clip_stats;
        compute_gp_to_beta_audit_stats_host(phi_new, eta_new, xB_pre_clip, P, total_r,
                                            cx, cy, cz, patch_r, &pre_clip_stats);
        compute_gp_to_beta_audit_stats_host(phi_new, eta_new, xB_new, P, total_r,
                                            cx, cy, cz, patch_r, &post_clip_stats);
        clipped_mass_loss = post_clip_stats.sum_xBtot - pre_clip_stats.sum_xBtot;
    }
    emit_audit_stage("after_clipping", phi_new, eta_new, xB_new, M_before,
                     compensation_mass, clipped_mass_loss,
                     xB_clip_count_low_event, xB_clip_count_high_event, "");

    CUDA_CHECK(cudaMemcpy(d_phi_r, phi_new.data(), size_r, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_eta_r, eta_new.data(), size_r, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xB_r, xB_new.data(), size_r, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y_r, Y.data(), size_r, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    emit_audit_stage("end_of_event_step", phi_new, eta_new, xB_new, M_before,
                     compensation_mass, clipped_mass_loss,
                     xB_clip_count_low_event, xB_clip_count_high_event, "");

    if (rt->events_csv) {
        const int accepted_events_total_after = accepted_events_total + 1;
        const int accepted_events_in_window_after =
            (P->gp_to_beta_event_window_steps > 0) ? (accepted_events_in_window + 1) : accepted_events_in_window;
        rt->event_counter += 1;
        fprintf(rt->events_csv, "%d,%d,%d,%d,%d,", step, rt->event_counter, ic, jc, kc);
        fprintf(rt->events_csv, "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,",
                cx, cy, cz, eta_max, R_eff, local_xB_alpha, local_xBtot_gp);
        fprintf(rt->events_csv, "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,",
                P->gp_to_beta_seed_radius, patch_r,
                M_before, M_after_raw, M_after_comp,
                M_after_raw - M_before, M_after_comp - M_before);
        fprintf(rt->events_csv, "%d,%d,", stochastic_enabled, event_accepted);
        fprintf(rt->events_csv, "%.10e,%.10e,%.10e,%.10e,%.10e,",
                drive_beta_given_GP, deltaG_star_J, J_site, P_event, random_u);
        fprintf(rt->events_csv, "%d,%lld,", compensation_success, xB_clip_count_event);
        fprintf(rt->events_csv, "%d,%.10e,%.10e,%.10e,%.10e,",
                feasibility_checked, mass_error_raw_predicted,
                shell_capacity_add, shell_capacity_remove, capacity_ratio);
        fprintf(rt->events_csv, "%d,%.10e,%.10e,%d,%d,%.10e,%.10e,",
                event_rejected_infeasible, seed_amplitude_original, seed_amplitude_scaled,
                seed_scaled_due_to_capacity, cooldown_or_spacing_rejected,
                xB_shell_min_before_event, xB_shell_max_before_event);
        fprintf(rt->events_csv, "%d,%d,%d,%d,%d,%.10e,%.10e,%d,%d\n",
                throttling_checked, rejected_cooldown, rejected_spacing,
                rejected_window_limit, rejected_global_limit,
                nearest_event_distance, steps_since_nearest_event,
                accepted_events_in_window_after, accepted_events_total_after);
        fflush(rt->events_csv);
    }

    if (result_out) {
        result_out->event_triggered = 1;
        result_out->compensation_success = compensation_success;
        result_out->stochastic_enabled = stochastic_enabled;
        result_out->event_accepted = event_accepted;
        result_out->stop_after_conversion_audit_requested =
            (P->gp_to_beta_stop_after_conversion_audit && P->gp_to_beta_conversion_mass_audit_enabled) ? 1 : 0;
        result_out->center_i = ic;
        result_out->center_j = jc;
        result_out->center_k = kc;
        result_out->center_x = cx;
        result_out->center_y = cy;
        result_out->center_z = cz;
        result_out->eta_max = eta_max;
        result_out->R_eff_GP = R_eff;
        result_out->local_xB_alpha = local_xB_alpha;
        result_out->local_xBtot_gp = local_xBtot_gp;
        result_out->drive_beta_given_GP = drive_beta_given_GP;
        result_out->deltaG_star_J = deltaG_star_J;
        result_out->J_site = J_site;
        result_out->P_event = P_event;
        result_out->random_u = random_u;
        result_out->phi_seed_radius = P->gp_to_beta_seed_radius;
        result_out->patch_radius = patch_r;
        result_out->mass_before = M_before;
        result_out->mass_after_raw = M_after_raw;
        result_out->mass_after_comp = M_after_comp;
        result_out->mass_error_raw = mass_error_raw_predicted;
        result_out->mass_error_comp = M_after_comp - M_before;
        result_out->xB_clip_count_event = (double)xB_clip_count_event;
        result_out->eta_depleted_amount = eta_depleted_amount;
        result_out->phi_inserted_volume = phi_inserted_volume;
        result_out->feasibility_checked = feasibility_checked;
        result_out->event_rejected_infeasible = event_rejected_infeasible;
        result_out->seed_scaled_due_to_capacity = seed_scaled_due_to_capacity;
        result_out->cooldown_or_spacing_rejected = cooldown_or_spacing_rejected;
        result_out->throttling_checked = throttling_checked;
        result_out->rejected_cooldown = rejected_cooldown;
        result_out->rejected_spacing = rejected_spacing;
        result_out->rejected_window_limit = rejected_window_limit;
        result_out->rejected_global_limit = rejected_global_limit;
        result_out->accepted_events_in_window =
            (P->gp_to_beta_event_window_steps > 0) ? (accepted_events_in_window + 1) : accepted_events_in_window;
        result_out->accepted_events_total = accepted_events_total + 1;
        result_out->shell_capacity_add = shell_capacity_add;
        result_out->shell_capacity_remove = shell_capacity_remove;
        result_out->capacity_ratio = capacity_ratio;
        result_out->seed_amplitude_original = seed_amplitude_original;
        result_out->seed_amplitude_scaled = seed_amplitude_scaled;
        result_out->xB_shell_min_before_event = xB_shell_min_before_event;
        result_out->xB_shell_max_before_event = xB_shell_max_before_event;
        result_out->nearest_event_distance = nearest_event_distance;
        result_out->steps_since_nearest_event = steps_since_nearest_event;
    }

    if (seed_scaled_due_to_capacity) {
        rt->count_events_scaled_capacity += 1;
    }
    rt->count_events_accepted += 1;
    if (isfinite(capacity_ratio)) {
        rt->min_capacity_ratio = fmin(rt->min_capacity_ratio, capacity_ratio);
        rt->sum_capacity_ratio += capacity_ratio;
        rt->capacity_ratio_count += 1;
    }
    rt->last_accepted_step = step;
    rt->last_accepted_i = ic;
    rt->last_accepted_j = jc;
    rt->last_accepted_k = kc;
    rt->accepted_events.push_back({step, ic, jc, kc});

    printf("[GP->BETA] step=%d center=(%d,%d,%d) eta_max=%.6e R_eff=%.6e xB_local=%.6e xBtot_local=%.6e M_before=%.10e M_after_raw=%.10e M_after_comp=%.10e success=%d clip=%lld\n",
           step, ic, jc, kc, eta_max, R_eff, local_xB_alpha, local_xBtot_gp,
           M_before, M_after_raw, M_after_comp, compensation_success, xB_clip_count_event);
    return 1;
}

static double gpu_reduce_sum_model_xBtot(const PFParams *P,
                                         const double *phi_r,
                                         const double *eta_r,
                                         const double *xB_alpha_r,
                                         int total_size) {
    if (is_gp_zone_mode(P)) {
        return gpu_reduce_sum_xBtot_gp(phi_r, eta_r, xB_alpha_r, P->gp_xB_fixed, total_size);
    }
    return gpu_reduce_sum_xBtot(phi_r, xB_alpha_r, P->v_B, total_size);
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

#define TRY_SET_ULONG(name, field) \
    if (strcmp((key), (name)) == 0) { \
        unsigned long parsed = 0; \
        if (!parse_unsigned_long_value((value), &parsed)) { \
            fprintf(stderr, "[fatal] %s:%d invalid unsigned long value for %s: %s\n", \
                    (path ? path : "<pf-param-file>"), (line_no), (name), (value)); \
            return -1; \
        } \
        P->field = parsed; \
        return 1; \
    }

    TRY_SET_INT("pf_params_schema_version", pf_params_schema_version);
    TRY_SET_ULONG("seed", seed);
    TRY_SET_DOUBLE("W", W);
    TRY_SET_DOUBLE("dt", dt);
    TRY_SET_DOUBLE("dx", dx);
    TRY_SET_DOUBLE("dy", dy);
    TRY_SET_DOUBLE("dz", dz);
    TRY_SET_DOUBLE("t_real_unit", t_real_unit);
    TRY_SET_DOUBLE("gp_xB_fixed", gp_xB_fixed);
    TRY_SET_DOUBLE("gp_delta_g0", gp_delta_g0);
    TRY_SET_DOUBLE("gp_delta_g_stab", gp_delta_g_stab);
    if (strcmp(key, "gp_W_eta") == 0) {
        double parsed = 0.0;
        if (!parse_double_value(value, &parsed)) {
            fprintf(stderr, "[fatal] %s:%d invalid numeric value for gp_W_eta: %s\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        P->gp_W_eta = parsed;
        P->gp_W_eta_code_input = parsed;
        P->gp_W_eta_code_specified = 1;
        P->gp_W_eta_legacy_specified = 1;
        fprintf(stderr, "[warn] gp_W_eta is deprecated; please use gp_W_eta_code instead.\n");
        return 1;
    }
    if (strcmp(key, "gp_kappa_eta") == 0) {
        double parsed = 0.0;
        if (!parse_double_value(value, &parsed)) {
            fprintf(stderr, "[fatal] %s:%d invalid numeric value for gp_kappa_eta: %s\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        P->gp_kappa_eta = parsed;
        P->gp_kappa_eta_code_input = parsed;
        P->gp_kappa_eta_code_specified = 1;
        P->gp_kappa_eta_legacy_specified = 1;
        fprintf(stderr, "[warn] gp_kappa_eta is deprecated; please use gp_kappa_eta_code instead.\n");
        return 1;
    }
    if (strcmp(key, "gp_L_eta") == 0) {
        double parsed = 0.0;
        if (!parse_double_value(value, &parsed)) {
            fprintf(stderr, "[fatal] %s:%d invalid numeric value for gp_L_eta: %s\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        P->gp_L_eta = parsed;
        P->gp_L_eta_code_input = parsed;
        P->gp_L_eta_code_specified = 1;
        P->gp_L_eta_legacy_specified = 1;
        fprintf(stderr, "[warn] gp_L_eta is deprecated; please use gp_L_eta_code instead.\n");
        return 1;
    }
    if (strcmp(key, "gp_W_eta_phys") == 0) {
        double parsed = 0.0;
        if (!parse_double_value(value, &parsed)) {
            fprintf(stderr, "[fatal] %s:%d invalid numeric value for gp_W_eta_phys: %s\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        P->gp_W_eta_phys_input = parsed;
        P->gp_W_eta_phys_specified = 1;
        return 1;
    }
    if (strcmp(key, "gp_kappa_eta_phys") == 0) {
        double parsed = 0.0;
        if (!parse_double_value(value, &parsed)) {
            fprintf(stderr, "[fatal] %s:%d invalid numeric value for gp_kappa_eta_phys: %s\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        P->gp_kappa_eta_phys_input = parsed;
        P->gp_kappa_eta_phys_specified = 1;
        return 1;
    }
    if (strcmp(key, "gp_L_eta_phys") == 0) {
        double parsed = 0.0;
        if (!parse_double_value(value, &parsed)) {
            fprintf(stderr, "[fatal] %s:%d invalid numeric value for gp_L_eta_phys: %s\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        P->gp_L_eta_phys_input = parsed;
        P->gp_L_eta_phys_specified = 1;
        return 1;
    }
    if (strcmp(key, "gp_W_eta_code") == 0) {
        double parsed = 0.0;
        if (!parse_double_value(value, &parsed)) {
            fprintf(stderr, "[fatal] %s:%d invalid numeric value for gp_W_eta_code: %s\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        P->gp_W_eta_code_input = parsed;
        P->gp_W_eta_code_specified = 1;
        return 1;
    }
    if (strcmp(key, "gp_L_eta_code") == 0) {
        double parsed = 0.0;
        if (!parse_double_value(value, &parsed)) {
            fprintf(stderr, "[fatal] %s:%d invalid numeric value for gp_L_eta_code: %s\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        P->gp_L_eta_code_input = parsed;
        P->gp_L_eta_code_specified = 1;
        return 1;
    }
    if (strcmp(key, "gp_kappa_eta_code") == 0) {
        double parsed = 0.0;
        if (!parse_double_value(value, &parsed)) {
            fprintf(stderr, "[fatal] %s:%d invalid numeric value for gp_kappa_eta_code: %s\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        P->gp_kappa_eta_code_input = parsed;
        P->gp_kappa_eta_code_specified = 1;
        return 1;
    }
    TRY_SET_DOUBLE("gp_L_eta", gp_L_eta);
    TRY_SET_DOUBLE("gp_gamma_alpha_gp", gp_gamma_alpha_gp);
    TRY_SET_DOUBLE("gp_l_eta_nm", gp_l_eta_nm);
    TRY_SET_DOUBLE("gp_D_ratio", gp_D_ratio);
    TRY_SET_DOUBLE("gp_M_int_eta", gp_M_int_eta);
    TRY_SET_DOUBLE("gp_M_eta_phys", gp_M_eta_phys);
    TRY_SET_DOUBLE("gp_M_eta_ratio_to_crit", gp_M_eta_ratio_to_crit);
    TRY_SET_DOUBLE("gp_eps_iso", gp_eps_iso);
    TRY_SET_DOUBLE("gp_M_GP", gp_M_GP);
    TRY_SET_DOUBLE("gp_M_beta", gp_M_beta);
    TRY_SET_INT("gp_elastic_enabled", gp_elastic_enabled);
    TRY_SET_INT("gp_elastic_active_eta", gp_elastic_active_eta);
    TRY_SET_INT("gp_elastic_active_phi", gp_elastic_active_phi);
    TRY_SET_DOUBLE("gp_elastic_derivative_scale", gp_elastic_derivative_scale);
    TRY_SET_INT("gp_nuc_enabled", gp_nuc_enabled);
    TRY_SET_INT("gp_nuc_check_interval", gp_nuc_check_interval);
    TRY_SET_DOUBLE("gp_nuc_phi_threshold", gp_nuc_phi_threshold);
    TRY_SET_DOUBLE("gp_nuc_eta_threshold", gp_nuc_eta_threshold);
    TRY_SET_DOUBLE("gp_nuc_h_alpha_threshold", gp_nuc_h_alpha_threshold);
    TRY_SET_DOUBLE("gp_nuc_J0", gp_nuc_J0);
    TRY_SET_DOUBLE("gp_nuc_gamma", gp_nuc_gamma);
    TRY_SET_DOUBLE("gp_nuc_seed_radius", gp_nuc_seed_radius);
    TRY_SET_DOUBLE("gp_nuc_seed_peak", gp_nuc_seed_peak);
    TRY_SET_DOUBLE("gp_nuc_seed_iface_width", gp_nuc_seed_iface_width);
    TRY_SET_DOUBLE("gp_nuc_patch_radius", gp_nuc_patch_radius);
    TRY_SET_DOUBLE("gp_nuc_shell_inner_radius", gp_nuc_shell_inner_radius);
    TRY_SET_DOUBLE("gp_nuc_shell_outer_radius", gp_nuc_shell_outer_radius);
    TRY_SET_INT("gp_nuc_max_events_per_step", gp_nuc_max_events_per_step);
    TRY_SET_INT("gp_to_beta_enabled", gp_to_beta_enabled);
    TRY_SET_INT("gp_to_beta_check_interval", gp_to_beta_check_interval);
    TRY_SET_DOUBLE("gp_to_beta_eta_threshold", gp_to_beta_eta_threshold);
    TRY_SET_DOUBLE("gp_to_beta_radius_threshold", gp_to_beta_radius_threshold);
    TRY_SET_DOUBLE("gp_to_beta_xB_threshold", gp_to_beta_xB_threshold);
    TRY_SET_DOUBLE("gp_to_beta_seed_radius", gp_to_beta_seed_radius);
    TRY_SET_DOUBLE("gp_to_beta_seed_peak", gp_to_beta_seed_peak);
    TRY_SET_DOUBLE("gp_to_beta_seed_iface_width", gp_to_beta_seed_iface_width);
    TRY_SET_DOUBLE("gp_to_beta_patch_radius", gp_to_beta_patch_radius);
    TRY_SET_DOUBLE("gp_to_beta_shell_inner_radius", gp_to_beta_shell_inner_radius);
    TRY_SET_DOUBLE("gp_to_beta_shell_outer_radius", gp_to_beta_shell_outer_radius);
    TRY_SET_INT("gp_to_beta_max_events_per_step", gp_to_beta_max_events_per_step);
    TRY_SET_INT("gp_to_beta_stochastic_enabled", gp_to_beta_stochastic_enabled);
    TRY_SET_DOUBLE("gp_to_beta_J0_site", gp_to_beta_J0_site);
    TRY_SET_DOUBLE("gp_to_beta_gamma", gp_to_beta_gamma);
    TRY_SET_DOUBLE("gp_to_beta_drive_const", gp_to_beta_drive_const);
    TRY_SET_INT("gp_to_beta_max_events_per_check", gp_to_beta_max_events_per_check);
    TRY_SET_INT("gp_to_beta_conversion_mass_audit_enabled", gp_to_beta_conversion_mass_audit_enabled);
    TRY_SET_INT("gp_to_beta_stop_after_conversion_audit", gp_to_beta_stop_after_conversion_audit);
    TRY_SET_INT("gp_to_beta_feasibility_gate_enabled", gp_to_beta_feasibility_gate_enabled);
    TRY_SET_DOUBLE("gp_to_beta_min_shell_capacity_factor", gp_to_beta_min_shell_capacity_factor);
    TRY_SET_INT("gp_to_beta_reject_if_infeasible", gp_to_beta_reject_if_infeasible);
    TRY_SET_INT("gp_to_beta_allow_seed_amplitude_scaling", gp_to_beta_allow_seed_amplitude_scaling);
    TRY_SET_DOUBLE("gp_to_beta_min_seed_amplitude", gp_to_beta_min_seed_amplitude);
    TRY_SET_INT("gp_to_beta_event_cooldown_steps", gp_to_beta_event_cooldown_steps);
    TRY_SET_DOUBLE("gp_to_beta_min_event_spacing", gp_to_beta_min_event_spacing);
    TRY_SET_DOUBLE("gp_to_beta_event_exclusion_radius", gp_to_beta_event_exclusion_radius);
    TRY_SET_INT("gp_to_beta_max_events_global", gp_to_beta_max_events_global);
    TRY_SET_INT("gp_to_beta_max_events_per_window", gp_to_beta_max_events_per_window);
    TRY_SET_INT("gp_to_beta_event_window_steps", gp_to_beta_event_window_steps);
    TRY_SET_INT("post_conversion_y_update_audit_enabled", post_conversion_y_update_audit_enabled);
    TRY_SET_INT("post_conversion_y_update_audit_steps", post_conversion_y_update_audit_steps);
    TRY_SET_INT("y_update_k0_audit_enabled", y_update_k0_audit_enabled);
    TRY_SET_INT("y_update_k0_audit_steps", y_update_k0_audit_steps);
    TRY_SET_INT("y_update_mass_projection_enabled", y_update_mass_projection_enabled);
    TRY_SET_INT("y_update_mass_projection_report_enabled", y_update_mass_projection_report_enabled);
    TRY_SET_INT("y_update_mass_projection_max_iter", y_update_mass_projection_max_iter);
    TRY_SET_DOUBLE("y_update_mass_projection_tol", y_update_mass_projection_tol);
    TRY_SET_DOUBLE("gp_eta_seed_radius", gp_eta_seed_radius);
    TRY_SET_DOUBLE("gp_eta_seed_peak", gp_eta_seed_peak);
    TRY_SET_DOUBLE("gp_eta_seed_center_x", gp_eta_seed_center_x);
    TRY_SET_DOUBLE("gp_eta_seed_center_y", gp_eta_seed_center_y);
    TRY_SET_DOUBLE("gp_eta_seed_center_z", gp_eta_seed_center_z);
    TRY_SET_DOUBLE("gp_eta_iface_width", gp_eta_iface_width);
    TRY_SET_DOUBLE("gp_obs_target_radius_nm", gp_obs_target_radius_nm);
    TRY_SET_DOUBLE("gp_obs_eta_peak", gp_obs_eta_peak);
    TRY_SET_DOUBLE("gp_obs_iface_width_nm", gp_obs_iface_width_nm);
    TRY_SET_DOUBLE("gp_obs_depletion_radius_factor", gp_obs_depletion_radius_factor);
    TRY_SET_DOUBLE("gp_obs_depletion_smooth_width_factor", gp_obs_depletion_smooth_width_factor);
    TRY_SET_DOUBLE("gp_obs_min_xB_alpha", gp_obs_min_xB_alpha);
    TRY_SET_DOUBLE("gp_obs_max_xB_alpha", gp_obs_max_xB_alpha);
    TRY_SET_INT("gp_raw_reaction_drive_only", gp_raw_reaction_drive_only);
    TRY_SET_INT("gp_raw_reaction_drive_use_raw_units_debug", gp_raw_reaction_drive_use_raw_units_debug);
    TRY_SET_DOUBLE("gp_reaction_nu_A", gp_reaction_nu_A);
    TRY_SET_DOUBLE("gp_reaction_nu_B", gp_reaction_nu_B);
    TRY_SET_DOUBLE("gp_xB_eq_alpha_for_eta", gp_xB_eq_alpha_for_eta);
    TRY_SET_INT("gp_kinetic_ref_enabled", gp_kinetic_ref_enabled);
    TRY_SET_INT("gp_kinetic_ref_apply", gp_kinetic_ref_apply);
    TRY_SET_INT("phi_eta_step_delta_diag_enabled", phi_eta_step_delta_diag_enabled);
    TRY_SET_INT("phi_eta_step_delta_diag_every", phi_eta_step_delta_diag_every);
    TRY_SET_INT("phi_eta_step_delta_diag_max_steps", phi_eta_step_delta_diag_max_steps);
    if (strcmp(key, "phi_eta_step_delta_diag_prefix") == 0) {
        snprintf(P->phi_eta_step_delta_diag_prefix,
                 sizeof(P->phi_eta_step_delta_diag_prefix), "%s", value);
        return 1;
    }
    TRY_SET_INT("phi_eta_rhs_attribution_diag_enabled", phi_eta_rhs_attribution_diag_enabled);
    TRY_SET_INT("phi_eta_rhs_attribution_diag_every", phi_eta_rhs_attribution_diag_every);
    TRY_SET_INT("phi_eta_rhs_attribution_diag_max_steps", phi_eta_rhs_attribution_diag_max_steps);
    if (strcmp(key, "phi_eta_rhs_attribution_diag_prefix") == 0) {
        snprintf(P->phi_eta_rhs_attribution_diag_prefix,
                 sizeof(P->phi_eta_rhs_attribution_diag_prefix), "%s", value);
        return 1;
    }
    TRY_SET_DOUBLE("gp_h_alpha_eps", gp_h_alpha_eps);
    TRY_SET_INT("gp_y_picard_iters", gp_y_picard_iters);
    TRY_SET_DOUBLE("kappa_phi", kappa_phi);
    TRY_SET_DOUBLE("L_phi", L_phi);
    TRY_SET_DOUBLE("D_alpha", D_alpha);
    TRY_SET_DOUBLE("D_compound", D_compound);
    TRY_SET_DOUBLE("temperature_C", temperature_C);
    TRY_SET_INT("thermo_convex_extrapolation_enabled", thermo_convex_extrapolation_enabled);
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
    TRY_SET_DOUBLE("E0_xx", E0_xx);
    TRY_SET_DOUBLE("E0_yy", E0_yy);
    TRY_SET_DOUBLE("E0_zz", E0_zz);
    TRY_SET_DOUBLE("E0_yz", E0_yz);
    TRY_SET_DOUBLE("E0_xz", E0_xz);
    TRY_SET_DOUBLE("E0_xy", E0_xy);

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

    if (strcmp(key, "model_mode") == 0) {
        if (!is_valid_model_mode(value)) {
            fprintf(stderr,
                    "[fatal] %s:%d invalid model_mode: %s (expected two_phase or gp_zone)\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        strncpy(P->model_mode, value, sizeof(P->model_mode) - 1);
        P->model_mode[sizeof(P->model_mode) - 1] = '\0';
        return 1;
    }
    if (strcmp(key, "gp_init_mode") == 0) {
        strncpy(P->gp_init_mode, value, sizeof(P->gp_init_mode) - 1);
        P->gp_init_mode[sizeof(P->gp_init_mode) - 1] = '\0';
        return 1;
    }
    if (strcmp(key, "gp_init_mass_mode") == 0) {
        strncpy(P->gp_init_mass_mode, value, sizeof(P->gp_init_mass_mode) - 1);
        P->gp_init_mass_mode[sizeof(P->gp_init_mass_mode) - 1] = '\0';
        return 1;
    }
    if (strcmp(key, "gp_obs_profile_type") == 0) {
        if (!is_valid_gp_obs_profile_type(value)) {
            fprintf(stderr, "[fatal] %s:%d invalid gp_obs_profile_type: %s (expected tanh, gaussian, or compact_smooth)\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        strncpy(P->gp_obs_profile_type, value, sizeof(P->gp_obs_profile_type) - 1);
        P->gp_obs_profile_type[sizeof(P->gp_obs_profile_type) - 1] = '\0';
        return 1;
    }
    if (strcmp(key, "gp_obs_match_mode") == 0) {
        if (!is_valid_gp_obs_match_mode(value)) {
            fprintf(stderr, "[fatal] %s:%d invalid gp_obs_match_mode: %s (expected match_integral_h_volume)\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        strncpy(P->gp_obs_match_mode, value, sizeof(P->gp_obs_match_mode) - 1);
        P->gp_obs_match_mode[sizeof(P->gp_obs_match_mode) - 1] = '\0';
        return 1;
    }
    if (strcmp(key, "gp_obs_compensation_mode") == 0) {
        if (!is_valid_gp_obs_compensation_mode(value)) {
            fprintf(stderr, "[fatal] %s:%d invalid gp_obs_compensation_mode: %s (expected smooth_radial_depletion, wide_shell, or hybrid)\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        strncpy(P->gp_obs_compensation_mode, value, sizeof(P->gp_obs_compensation_mode) - 1);
        P->gp_obs_compensation_mode[sizeof(P->gp_obs_compensation_mode) - 1] = '\0';
        return 1;
    }
    if (strcmp(key, "gp_nuc_mass_mode") == 0) {
        if (!is_valid_gp_nuc_mass_mode(value)) {
            fprintf(stderr, "[fatal] %s:%d invalid gp_nuc_mass_mode: %s (expected report or local_compensate)\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        strncpy(P->gp_nuc_mass_mode, value, sizeof(P->gp_nuc_mass_mode) - 1);
        P->gp_nuc_mass_mode[sizeof(P->gp_nuc_mass_mode) - 1] = '\0';
        return 1;
    }
    if (strcmp(key, "gp_to_beta_mass_mode") == 0) {
        if (!is_valid_gp_to_beta_mass_mode(value)) {
            fprintf(stderr, "[fatal] %s:%d invalid gp_to_beta_mass_mode: %s (expected report or local_compensate)\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        strncpy(P->gp_to_beta_mass_mode, value, sizeof(P->gp_to_beta_mass_mode) - 1);
        P->gp_to_beta_mass_mode[sizeof(P->gp_to_beta_mass_mode) - 1] = '\0';
        return 1;
    }
    if (strcmp(key, "gp_to_beta_barrier_mode") == 0) {
        if (!is_valid_gp_to_beta_barrier_mode(value)) {
            fprintf(stderr, "[fatal] %s:%d invalid gp_to_beta_barrier_mode: %s (expected cnt_simple)\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        strncpy(P->gp_to_beta_barrier_mode, value, sizeof(P->gp_to_beta_barrier_mode) - 1);
        P->gp_to_beta_barrier_mode[sizeof(P->gp_to_beta_barrier_mode) - 1] = '\0';
        return 1;
    }
    if (strcmp(key, "gp_to_beta_drive_mode") == 0) {
        if (!is_valid_gp_to_beta_drive_mode(value)) {
            fprintf(stderr, "[fatal] %s:%d invalid gp_to_beta_drive_mode: %s (expected constant or local_simple)\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        strncpy(P->gp_to_beta_drive_mode, value, sizeof(P->gp_to_beta_drive_mode) - 1);
        P->gp_to_beta_drive_mode[sizeof(P->gp_to_beta_drive_mode) - 1] = '\0';
        return 1;
    }
    if (strcmp(key, "gp_to_beta_eta_deplete_mode") == 0) {
        if (!is_valid_gp_to_beta_eta_deplete_mode(value)) {
            fprintf(stderr, "[fatal] %s:%d invalid gp_to_beta_eta_deplete_mode: %s\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        strncpy(P->gp_to_beta_eta_deplete_mode, value, sizeof(P->gp_to_beta_eta_deplete_mode) - 1);
        P->gp_to_beta_eta_deplete_mode[sizeof(P->gp_to_beta_eta_deplete_mode) - 1] = '\0';
        return 1;
    }
    if (strcmp(key, "gp_to_beta_phi_insert_mode") == 0) {
        if (!is_valid_gp_to_beta_phi_insert_mode(value)) {
            fprintf(stderr, "[fatal] %s:%d invalid gp_to_beta_phi_insert_mode: %s (expected max)\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        strncpy(P->gp_to_beta_phi_insert_mode, value, sizeof(P->gp_to_beta_phi_insert_mode) - 1);
        P->gp_to_beta_phi_insert_mode[sizeof(P->gp_to_beta_phi_insert_mode) - 1] = '\0';
        return 1;
    }
    if (strcmp(key, "gp_to_beta_conversion_audit_prefix") == 0) {
        strncpy(P->gp_to_beta_conversion_audit_prefix, value,
                sizeof(P->gp_to_beta_conversion_audit_prefix) - 1);
        P->gp_to_beta_conversion_audit_prefix[sizeof(P->gp_to_beta_conversion_audit_prefix) - 1] = '\0';
        return 1;
    }
    if (strcmp(key, "post_conversion_y_update_audit_prefix") == 0) {
        strncpy(P->post_conversion_y_update_audit_prefix, value,
                sizeof(P->post_conversion_y_update_audit_prefix) - 1);
        P->post_conversion_y_update_audit_prefix[sizeof(P->post_conversion_y_update_audit_prefix) - 1] = '\0';
        return 1;
    }
    if (strcmp(key, "y_update_k0_audit_prefix") == 0) {
        strncpy(P->y_update_k0_audit_prefix, value,
                sizeof(P->y_update_k0_audit_prefix) - 1);
        P->y_update_k0_audit_prefix[sizeof(P->y_update_k0_audit_prefix) - 1] = '\0';
        return 1;
    }
    if (strcmp(key, "y_update_mass_projection_target_mode") == 0) {
        if (!is_valid_y_update_mass_projection_target_mode(value)) {
            fprintf(stderr, "[fatal] %s:%d invalid y_update_mass_projection_target_mode: %s (expected pre_Y_update or post_conversion_baseline)\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        strncpy(P->y_update_mass_projection_target_mode, value,
                sizeof(P->y_update_mass_projection_target_mode) - 1);
        P->y_update_mass_projection_target_mode[sizeof(P->y_update_mass_projection_target_mode) - 1] = '\0';
        return 1;
    }
    if (strcmp(key, "gp_eta_mass_limiter") == 0) {
        if (!is_valid_gp_eta_mass_limiter(value)) {
            fprintf(stderr, "[fatal] %s:%d invalid gp_eta_mass_limiter: %s (expected off or local_clip)\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        strncpy(P->gp_eta_mass_limiter, value, sizeof(P->gp_eta_mass_limiter) - 1);
        P->gp_eta_mass_limiter[sizeof(P->gp_eta_mass_limiter) - 1] = '\0';
        return 1;
    }
    if (strcmp(key, "gp_y_update_mode") == 0) {
        if (!is_valid_gp_y_update_mode(value)) {
            fprintf(stderr, "[fatal] %s:%d invalid gp_y_update_mode: %s (expected old_rhs, conservative_y_rhs, picard_storage, or storage_exact)\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        strncpy(P->gp_y_update_mode, value, sizeof(P->gp_y_update_mode) - 1);
        P->gp_y_update_mode[sizeof(P->gp_y_update_mode) - 1] = '\0';
        return 1;
    }
    if (strcmp(key, "gp_C_mode") == 0) {
        if (!is_valid_gp_C_mode(value)) {
            fprintf(stderr, "[fatal] %s:%d invalid gp_C_mode: %s (expected alpha)\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        strncpy(P->gp_C_mode, value, sizeof(P->gp_C_mode) - 1);
        P->gp_C_mode[sizeof(P->gp_C_mode) - 1] = '\0';
        return 1;
    }
    if (strcmp(key, "gp_eps_mode") == 0) {
        if (!is_valid_gp_eps_mode(value)) {
            fprintf(stderr, "[fatal] %s:%d invalid gp_eps_mode: %s (expected isotropic)\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        strncpy(P->gp_eps_mode, value, sizeof(P->gp_eps_mode) - 1);
        P->gp_eps_mode[sizeof(P->gp_eps_mode) - 1] = '\0';
        return 1;
    }
    if (strcmp(key, "gp_L_eta_mode") == 0) {
        if (!is_valid_gp_L_eta_mode(value)) {
            fprintf(stderr, "[fatal] %s:%d invalid gp_L_eta_mode: %s (expected manual, sto_S218b, or sto_S218b_override)\n",
                    (path ? path : "<pf-param-file>"), line_no, value);
            return -1;
        }
        strncpy(P->gp_L_eta_mode, value, sizeof(P->gp_L_eta_mode) - 1);
        P->gp_L_eta_mode[sizeof(P->gp_L_eta_mode) - 1] = '\0';
        return 1;
    }
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
    char continue_phi_vtk_pre_scan[4096] = {0};
    char effective_pf_param_file[4096] = {0};
    char continue_case_pf_param_file[4096] = {0};

    // Optional: interpret radius in physical nm and convert to internal length units later,
    // after we can infer (dx_phys_m_run, unit_to_m_run) from PF inputs.
    int has_radius_phys_nm = 0;
    double radius_phys_nm = 0.0;
    int has_xB_matrix_override = 0;
    double xB_matrix_override = 0.0;

    for (int i = 1; i < argc; ++i) {
        const char *v = get_flag_value(argc, argv, &i, "--pf-param-file");
        if (v != NULL) {
            pf_param_file = v;
        }
        v = get_flag_value(argc, argv, &i, "--continue-phi-vtk");
        if (v != NULL) {
            snprintf(continue_phi_vtk_pre_scan, sizeof(continue_phi_vtk_pre_scan), "%s", v);
        }
    }
    if (!wants_help && pf_param_file == NULL) {
        fprintf(stderr,
                "[fatal] 物理参数不再使用 main_cuda 内置默认值。请通过 --pf-param-file <path> 提供完整物理输入。\n");
        fprintf(stderr,
                "        可先运行: python3 Unit_Psedobinary.py --input-json physical_inputs.example.json --output-pf-param-file /tmp/generated_pf.params\n");
        return 2;
    }
    if (argc >= 2 && argv[1][0] != '-') { P.Nx = atoi(argv[1]); }
    if (argc >= 3 && argv[2][0] != '-') { P.Ny = atoi(argv[2]); }
    if (argc >= 4 && argv[3][0] != '-') { P.Nz = atoi(argv[3]); }
    if (argc >= 5 && argv[4][0] != '-') { P.dt = atof(argv[4]); }
    if (argc >= 6 && argv[5][0] != '-') { P.nsteps = atoi(argv[5]); }
    if (argc >= 7 && argv[6][0] != '-') { P.out_every = atoi(argv[6]); }
    if (argc >= 8 && argv[7][0] != '-') { P.csv_out_every = atoi(argv[7]); }  // CSV输出间隔参数
    if (argc >= 9 && argv[8][0] != '-') { P.elastic_enabled = atoi(argv[8]); }  // 弹性功能开关（0/1）
    if (pf_param_file != NULL) {
        snprintf(effective_pf_param_file, sizeof(effective_pf_param_file), "%s", pf_param_file);
        if (!load_pfparams_override_file(&P, effective_pf_param_file)) {
            return 2;
        }
        // 物理输入文件中的 dt 作为默认时间步；若后续显式给出 --minimize-dt，仍允许覆盖。
        P.minimize_dt = P.dt;
    }
    if (continue_phi_vtk_pre_scan[0] != '\0' &&
        build_continue_case_pf_param_path(continue_phi_vtk_pre_scan,
                                          continue_case_pf_param_file,
                                          sizeof(continue_case_pf_param_file))) {
        struct stat st;
        if (stat(continue_case_pf_param_file, &st) == 0) {
            if (pf_param_file == NULL) {
                if (!load_pfparams_override_file(&P, continue_case_pf_param_file)) {
                    return 2;
                }
                snprintf(effective_pf_param_file, sizeof(effective_pf_param_file), "%s", continue_case_pf_param_file);
                P.minimize_dt = P.dt;
                printf("[continue-param] loaded stored PF params from %s\n", continue_case_pf_param_file);
            } else {
                printf("[continue-param] explicit --pf-param-file takes precedence; ignoring stored PF params at %s\n",
                       continue_case_pf_param_file);
            }
        } else {
            printf("[continue-param] no stored PF params next to continue VTK, fallback to %s\n",
                   effective_pf_param_file);
        }
    }
    if (P.pf_params_schema_version != PF_PARAMS_SCHEMA_VERSION) {
        fprintf(stderr,
                "[fatal] pf_params_schema_version mismatch: file=%d expected=%d. 请重新跑 Unit_Psedobinary.py\n",
                P.pf_params_schema_version, PF_PARAMS_SCHEMA_VERSION);
        return 2;
    }
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
            printf("    注: dt 位置参数仅作 legacy 回退；若 --pf-param-file 中提供 dt，则以参数文件为准。\n");
            printf("\nFlags (optional):\n");
            printf("  --mode=dynamics|dynamics-continue|minimize|minimize-continue\n");
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
            printf("  --radius <val>          seed radius in internal length units (same units as P.dx)\n");
            printf("  --radius-phys-nm <nm>   seed radius in physical nm (converted using dx_phys inferred from lambda_sm_m/(2*ic_phi_iface_w))\n");
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
            printf("  --seed <int>            global deterministic RNG seed for event sampling and init randomness\n");
            printf("  --gp-to-beta-conversion-mass-audit-enabled 0|1\n");
            printf("  --gp-to-beta-stop-after-conversion-audit 0|1\n");
            printf("  --post-conversion-y-update-audit-enabled 0|1\n");
            printf("  --post-conversion-y-update-audit-steps <n>\n");
            printf("  --y-update-mass-projection-enabled 0|1\n");
            printf("  --y-update-mass-projection-report-enabled 0|1\n");
            printf("  --y-update-mass-projection-max-iter <n>\n");
            printf("  --y-update-mass-projection-tol <val>\n");
            printf("  --y-update-mass-projection-target-mode pre_Y_update|post_conversion_baseline\n");
            printf("  --init-test-id <int>    preset index for batch tests (0..7, <0 to disable)\n");
            printf("  --continue-phi-vtk <path>  continuation: load phi from ASCII VTK and continue minimize\n");
            printf("  --continue-xB-vtk <path>   continuation(full-model): optional xB ASCII VTK; if missing, rebuild xB/Y from phi via init logic\n");
            printf("                               若 VTK 同目录存在 pf_input.params，则 continue 会优先使用该参数快照\n");
            printf("  --init-mode raw_fields      initialize from Python-generated raw phi/xB[/eta] fields\n");
            printf("  --init-phi-raw <path>       raw_fields phi_init.raw (C order, idx=i*(Ny*Nz)+j*Nz+k)\n");
            printf("  --init-xB-raw <path>        raw_fields xB_init.raw\n");
            printf("  --init-eta-raw <path>       raw_fields optional eta_init.raw (default zero)\n");
            printf("  --init-meta <path>          raw_fields init_meta.json with grid/dx/lambda/dtype metadata\n");
            printf("\nScheduled nucleation TEST mode (explicit opt-in, dynamics only):\n");
            printf("  --enable-scheduled-nucleation-test\n");
            printf("  --scheduled-nuc-source-dyn-dir <dir>       no-strain dynamic-continue source dir\n");
            printf("  --scheduled-nuc-profile-dir <dir>          faceted profile dir containing faceted_family_profiles.csv\n");
            printf("  --scheduled-nuc-source-step latest|N       diagnostic source step label (default latest)\n");
            printf("  --scheduled-nuc-steps 100,300,600\n");
            printf("  --scheduled-nuc-centers-nm \"100,100,100;180,100,100;100,180,100\"\n");
            printf("  --scheduled-nuc-xB-edge-mode sample-current-background-shell\n");
            printf("  --scheduled-nuc-local-comp-outer-nm 20.0\n");
            printf("  --scheduled-nuc-local-comp-taper-nm 5.0\n");
            printf("  --scheduled-nucleation-write-event-vtk 0|1\n");
            printf("\nDynamics mass-drift diagnostics (explicit opt-in):\n");
            printf("  --enable-dynamics-mass-diagnostics\n");
            printf("  --dynamics-mass-diag-interval <n>   write diagnostics every n steps (default 1)\n");
            printf("  --enable-Y-rhs-previous-time-level  use phi^n/Y^n for explicit Y-RHS terms (default off)\n");
            printf("  --disable-Y-rhs-gamma-term          diagnostic mode only; set term_gamma = 0 (default off)\n");
            printf("  --Y-rhs-term-h-scale <s>            diagnostic scaling for term_h (default 1.0)\n");
            printf("  --enable-Y-rhs-picard               diagnostic mode only; Picard iterate gamma*dYdt coupling (default off)\n");
            printf("  --Y-rhs-picard-iters <n>            Picard iterations for Y RHS gamma*dYdt (default 1)\n");
            printf("  --Y-rhs-picard-omega <w>            Picard under-relaxation omega in (0,1] (default 1.0)\n");
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
            else if (strcmp(v, "minimize-continue") == 0) {
                P.mode = 1;
                P.minimize_continue_from_vtk = 1;
            }
            else if (strcmp(v, "dynamics-continue") == 0) {
                P.mode = 0;
            }
            else P.mode = 0;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--init-mode")) != NULL) {
            if (strcmp(v, "raw_fields") == 0) {
                P.init_mode_raw_fields = 1;
            } else {
                fprintf(stderr, "[fatal] unsupported --init-mode '%s'. Supported: raw_fields\n", v);
                return 2;
            }
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--init-phi-raw")) != NULL) {
            strncpy(P.init_phi_raw_path, v, sizeof(P.init_phi_raw_path) - 1);
            P.init_phi_raw_path[sizeof(P.init_phi_raw_path) - 1] = '\0';
            P.init_mode_raw_fields = 1;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--init-xB-raw")) != NULL) {
            strncpy(P.init_xB_raw_path, v, sizeof(P.init_xB_raw_path) - 1);
            P.init_xB_raw_path[sizeof(P.init_xB_raw_path) - 1] = '\0';
            P.init_mode_raw_fields = 1;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--init-eta-raw")) != NULL) {
            strncpy(P.init_eta_raw_path, v, sizeof(P.init_eta_raw_path) - 1);
            P.init_eta_raw_path[sizeof(P.init_eta_raw_path) - 1] = '\0';
            P.init_mode_raw_fields = 1;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--init-meta")) != NULL) {
            strncpy(P.init_meta_path, v, sizeof(P.init_meta_path) - 1);
            P.init_meta_path[sizeof(P.init_meta_path) - 1] = '\0';
            P.init_mode_raw_fields = 1;
            continue;
        }
        if (strcmp(argv[i], "--enable-scheduled-nucleation-test") == 0) {
            P.scheduled_nuc_enabled = 1;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-source-dyn-dir")) != NULL) {
            strncpy(P.scheduled_nuc_source_dyn_dir, v, sizeof(P.scheduled_nuc_source_dyn_dir) - 1);
            P.scheduled_nuc_source_dyn_dir[sizeof(P.scheduled_nuc_source_dyn_dir) - 1] = '\0';
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-profile-dir")) != NULL) {
            strncpy(P.scheduled_nuc_profile_dir, v, sizeof(P.scheduled_nuc_profile_dir) - 1);
            P.scheduled_nuc_profile_dir[sizeof(P.scheduled_nuc_profile_dir) - 1] = '\0';
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-source-step")) != NULL) {
            strncpy(P.scheduled_nuc_source_step, v, sizeof(P.scheduled_nuc_source_step) - 1);
            P.scheduled_nuc_source_step[sizeof(P.scheduled_nuc_source_step) - 1] = '\0';
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-source-phi-vtk")) != NULL) {
            strncpy(P.scheduled_nuc_source_phi_vtk, v, sizeof(P.scheduled_nuc_source_phi_vtk) - 1);
            P.scheduled_nuc_source_phi_vtk[sizeof(P.scheduled_nuc_source_phi_vtk) - 1] = '\0';
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-source-xB-vtk")) != NULL) {
            strncpy(P.scheduled_nuc_source_xB_vtk, v, sizeof(P.scheduled_nuc_source_xB_vtk) - 1);
            P.scheduled_nuc_source_xB_vtk[sizeof(P.scheduled_nuc_source_xB_vtk) - 1] = '\0';
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-source-case-label")) != NULL) {
            strncpy(P.scheduled_nuc_source_case_label, v, sizeof(P.scheduled_nuc_source_case_label) - 1);
            P.scheduled_nuc_source_case_label[sizeof(P.scheduled_nuc_source_case_label) - 1] = '\0';
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-steps")) != NULL) {
            strncpy(P.scheduled_nuc_steps_csv, v, sizeof(P.scheduled_nuc_steps_csv) - 1);
            P.scheduled_nuc_steps_csv[sizeof(P.scheduled_nuc_steps_csv) - 1] = '\0';
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-centers-nm")) != NULL) {
            strncpy(P.scheduled_nuc_centers_nm, v, sizeof(P.scheduled_nuc_centers_nm) - 1);
            P.scheduled_nuc_centers_nm[sizeof(P.scheduled_nuc_centers_nm) - 1] = '\0';
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-xB-edge-mode")) != NULL) {
            strncpy(P.scheduled_nuc_xB_edge_mode, v, sizeof(P.scheduled_nuc_xB_edge_mode) - 1);
            P.scheduled_nuc_xB_edge_mode[sizeof(P.scheduled_nuc_xB_edge_mode) - 1] = '\0';
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-edge-sample-stat")) != NULL) {
            strncpy(P.scheduled_nuc_edge_sample_stat, v, sizeof(P.scheduled_nuc_edge_sample_stat) - 1);
            P.scheduled_nuc_edge_sample_stat[sizeof(P.scheduled_nuc_edge_sample_stat) - 1] = '\0';
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-edge-sample-inner-nm")) != NULL) {
            P.scheduled_nuc_edge_sample_inner_nm = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-edge-sample-outer-nm")) != NULL) {
            P.scheduled_nuc_edge_sample_outer_nm = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-local-comp-inner-nm")) != NULL) {
            P.scheduled_nuc_local_comp_inner_nm = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-local-comp-outer-nm")) != NULL) {
            P.scheduled_nuc_local_comp_outer_nm = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-local-comp-taper-nm")) != NULL) {
            P.scheduled_nuc_local_comp_taper_nm = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-source-dx-nm")) != NULL) {
            P.scheduled_nuc_source_dx_nm = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-source-lambda-nm")) != NULL) {
            P.scheduled_nuc_source_lambda_nm = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-target-lambda-nm")) != NULL) {
            P.scheduled_nuc_target_lambda_nm = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-scale-geometry")) != NULL) {
            P.scheduled_nuc_scale_geometry = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-scale-interface-width")) != NULL) {
            P.scheduled_nuc_scale_interface_width = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-scale-xB-profile-width")) != NULL) {
            P.scheduled_nuc_scale_xB_profile_width = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-alpha-interface")) != NULL) {
            P.scheduled_nuc_alpha_interface = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-xB-min")) != NULL) {
            P.scheduled_nuc_xB_min = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-xB-max")) != NULL) {
            P.scheduled_nuc_xB_max = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-phi-matrix-threshold")) != NULL) {
            P.scheduled_nuc_phi_matrix_threshold = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-W-comp-threshold")) != NULL) {
            P.scheduled_nuc_W_comp_threshold = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-mass-iters")) != NULL) {
            P.scheduled_nuc_mass_iters = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nuc-mass-tol")) != NULL) {
            P.scheduled_nuc_mass_tol = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--scheduled-nucleation-write-event-vtk")) != NULL) {
            P.scheduled_nuc_write_event_vtk = atoi(v) ? 1 : 0;
            continue;
        }
        if (strcmp(argv[i], "--scheduled-nuc-fallback-analytic-sphere") == 0) {
            P.scheduled_nuc_fallback_analytic_sphere = 1;
            continue;
        }
        if (strcmp(argv[i], "--enable-dynamics-mass-diagnostics") == 0) {
            P.dynamics_mass_diag_enabled = 1;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--dynamics-mass-diag-interval")) != NULL) {
            P.dynamics_mass_diag_enabled = 1;
            P.dynamics_mass_diag_interval = atoi(v);
            if (P.dynamics_mass_diag_interval <= 0) P.dynamics_mass_diag_interval = 1;
            continue;
        }
        if (strcmp(argv[i], "--enable-Y-rhs-previous-time-level") == 0 ||
            strcmp(argv[i], "--enable-y-rhs-prev-time-level") == 0) {
            P.enable_Y_rhs_previous_time_level = 1;
            continue;
        }
        if (strcmp(argv[i], "--disable-Y-rhs-gamma-term") == 0 ||
            strcmp(argv[i], "--disable-y-rhs-gamma-term") == 0) {
            P.disable_Y_rhs_gamma_term = 1;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--Y-rhs-term-h-scale")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--y-rhs-term-h-scale")) != NULL) {
            P.Y_rhs_term_h_scale = atof(v);
            continue;
        }
        if (strcmp(argv[i], "--enable-Y-rhs-picard") == 0 ||
            strcmp(argv[i], "--enable-y-rhs-picard") == 0) {
            P.enable_Y_rhs_picard = 1;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--Y-rhs-picard-iters")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--y-rhs-picard-iters")) != NULL) {
            P.enable_Y_rhs_picard = 1;
            P.Y_rhs_picard_iters = atoi(v);
            if (P.Y_rhs_picard_iters <= 0) P.Y_rhs_picard_iters = 1;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--Y-rhs-picard-omega")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--y-rhs-picard-omega")) != NULL) {
            P.enable_Y_rhs_picard = 1;
            P.Y_rhs_picard_omega = atof(v);
            if (!(P.Y_rhs_picard_omega > 0.0 && P.Y_rhs_picard_omega <= 1.0)) {
                P.Y_rhs_picard_omega = 1.0;
            }
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--pf-param-file")) != NULL) {
            (void)v;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--Nx")) != NULL) {
            P.Nx = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--Ny")) != NULL) {
            P.Ny = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--Nz")) != NULL) {
            P.Nz = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--dt")) != NULL) {
            P.dt = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--nsteps")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--steps")) != NULL) {
            P.nsteps = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--out-every")) != NULL) {
            P.out_every = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--csv-out-every")) != NULL) {
            P.csv_out_every = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--dx-nm")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--interface-width-nm")) != NULL) {
            // Accepted for command readability. The actual PF discretization still comes
            // from --pf-param-file; raw init meta validation catches any mismatch.
            (void)v;
            continue;
        }
        if (strcmp(argv[i], "--no-random-init") == 0 ||
            strcmp(argv[i], "--disable-extra-nucleation") == 0) {
            // raw_fields never uses random IC or extra nucleation; keep these no-op flags for scripts.
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
        if ((v = get_flag_value(argc, argv, &i, "--continue-phi-vtk")) != NULL) {
            strncpy(P.continue_phi_vtk_path, v, sizeof(P.continue_phi_vtk_path) - 1);
            P.continue_phi_vtk_path[sizeof(P.continue_phi_vtk_path) - 1] = '\0';
            P.minimize_continue_from_vtk = 1;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--continue-xB-vtk")) != NULL) {
            strncpy(P.continue_xB_vtk_path, v, sizeof(P.continue_xB_vtk_path) - 1);
            P.continue_xB_vtk_path[sizeof(P.continue_xB_vtk_path) - 1] = '\0';
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
        if ((v = get_flag_value(argc, argv, &i, "--radius-phys-nm")) != NULL) {
            has_radius_phys_nm = 1;
            radius_phys_nm = atof(v);
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
        if ((v = get_flag_value(argc, argv, &i, "--external-strain")) != NULL) {
            double s = atof(v);
            P.E0_xx = P.E0_yy = P.E0_zz = P.E0_yz = P.E0_xz = P.E0_xy = s;
            if (fabs(s) < 1.0e-30) {
                P.E0_xx = P.E0_yy = P.E0_zz = P.E0_yz = P.E0_xz = P.E0_xy = 0.0;
            }
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--external-strain-exx")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--E0-xx")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--E0_xx")) != NULL) {
            P.E0_xx = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--external-strain-eyy")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--E0-yy")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--E0_yy")) != NULL) {
            P.E0_yy = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--external-strain-ezz")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--E0-zz")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--E0_zz")) != NULL) {
            P.E0_zz = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--external-strain-eyz")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--E0-yz")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--E0_yz")) != NULL) {
            P.E0_yz = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--external-strain-exz")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--E0-xz")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--E0_xz")) != NULL) {
            P.E0_xz = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--external-strain-exy")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--E0-xy")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--E0_xy")) != NULL) {
            P.E0_xy = atof(v);
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
        if ((v = get_flag_value(argc, argv, &i, "--seed")) != NULL) {
            P.seed = (unsigned long)strtoul(v, NULL, 10);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--xB_matrix")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--xB-matrix")) != NULL) {
            xB_matrix_override = atof(v);
            has_xB_matrix_override = 1;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-xB-fixed")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_xB_fixed")) != NULL) {
            P.gp_xB_fixed = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-delta-g0")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_delta_g0")) != NULL) {
            P.gp_delta_g0 = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-delta-g-stab")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_delta_g_stab")) != NULL) {
            P.gp_delta_g_stab = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-L-eta")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_L_eta")) != NULL) {
            P.gp_L_eta = atof(v);
            P.gp_L_eta_code_input = P.gp_L_eta;
            P.gp_L_eta_code_specified = 1;
            P.gp_L_eta_legacy_specified = 1;
            fprintf(stderr, "[warn] gp_L_eta is deprecated; please use gp_L_eta_code instead.\n");
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-W-eta-phys")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_W_eta_phys")) != NULL) {
            P.gp_W_eta_phys_input = atof(v);
            P.gp_W_eta_phys_specified = 1;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-kappa-eta-phys")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_kappa_eta_phys")) != NULL) {
            P.gp_kappa_eta_phys_input = atof(v);
            P.gp_kappa_eta_phys_specified = 1;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-W-eta-code")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_W_eta_code")) != NULL) {
            P.gp_W_eta_code_input = atof(v);
            P.gp_W_eta_code_specified = 1;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-kappa-eta-code")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_kappa_eta_code")) != NULL) {
            P.gp_kappa_eta_code_input = atof(v);
            P.gp_kappa_eta_code_specified = 1;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-W-eta")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_W_eta")) != NULL) {
            P.gp_W_eta = atof(v);
            P.gp_W_eta_code_input = P.gp_W_eta;
            P.gp_W_eta_code_specified = 1;
            P.gp_W_eta_legacy_specified = 1;
            fprintf(stderr, "[warn] gp_W_eta is deprecated; please use gp_W_eta_code instead.\n");
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-kappa-eta")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_kappa_eta")) != NULL) {
            P.gp_kappa_eta = atof(v);
            P.gp_kappa_eta_code_input = P.gp_kappa_eta;
            P.gp_kappa_eta_code_specified = 1;
            P.gp_kappa_eta_legacy_specified = 1;
            fprintf(stderr, "[warn] gp_kappa_eta is deprecated; please use gp_kappa_eta_code instead.\n");
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-gamma-alpha-gp")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_gamma_alpha_gp")) != NULL) {
            P.gp_gamma_alpha_gp = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-l-eta-nm")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_l_eta_nm")) != NULL) {
            P.gp_l_eta_nm = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-D-ratio")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_D_ratio")) != NULL) {
            P.gp_D_ratio = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-L-eta-mode")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_L_eta_mode")) != NULL) {
            if (!is_valid_gp_L_eta_mode(v)) {
                fprintf(stderr, "[fatal] invalid --gp_L_eta_mode: %s (expected manual, sto_S218b, or sto_S218b_override)\n", v);
                return 2;
            }
            strncpy(P.gp_L_eta_mode, v, sizeof(P.gp_L_eta_mode) - 1);
            P.gp_L_eta_mode[sizeof(P.gp_L_eta_mode) - 1] = '\0';
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-M-int-eta")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_M_int_eta")) != NULL) {
            P.gp_M_int_eta = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-M-eta-phys")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_M_eta_phys")) != NULL) {
            P.gp_M_eta_phys = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-M-eta-ratio-to-crit")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_M_eta_ratio_to_crit")) != NULL) {
            P.gp_M_eta_ratio_to_crit = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-xB-eq-alpha-for-eta")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_xB_eq_alpha_for_eta")) != NULL) {
            P.gp_xB_eq_alpha_for_eta = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-kinetic-ref-enabled")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_kinetic_ref_enabled")) != NULL) {
            P.gp_kinetic_ref_enabled = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-kinetic-ref-apply")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_kinetic_ref_apply")) != NULL) {
            P.gp_kinetic_ref_apply = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--phi-eta-step-delta-diag-enabled")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--phi_eta_step_delta_diag_enabled")) != NULL) {
            P.phi_eta_step_delta_diag_enabled = atoi(v) ? 1 : 0;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--phi-eta-step-delta-diag-every")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--phi_eta_step_delta_diag_every")) != NULL) {
            P.phi_eta_step_delta_diag_every = atoi(v);
            if (P.phi_eta_step_delta_diag_every <= 0) P.phi_eta_step_delta_diag_every = 1;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--phi-eta-step-delta-diag-max-steps")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--phi_eta_step_delta_diag_max_steps")) != NULL) {
            P.phi_eta_step_delta_diag_max_steps = atoi(v);
            if (P.phi_eta_step_delta_diag_max_steps < 0) P.phi_eta_step_delta_diag_max_steps = 0;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--phi-eta-step-delta-diag-prefix")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--phi_eta_step_delta_diag_prefix")) != NULL) {
            snprintf(P.phi_eta_step_delta_diag_prefix,
                     sizeof(P.phi_eta_step_delta_diag_prefix), "%s", v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--phi-eta-rhs-attribution-diag-enabled")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--phi_eta_rhs_attribution_diag_enabled")) != NULL) {
            P.phi_eta_rhs_attribution_diag_enabled = atoi(v) ? 1 : 0;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--phi-eta-rhs-attribution-diag-every")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--phi_eta_rhs_attribution_diag_every")) != NULL) {
            P.phi_eta_rhs_attribution_diag_every = atoi(v);
            if (P.phi_eta_rhs_attribution_diag_every <= 0) P.phi_eta_rhs_attribution_diag_every = 1;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--phi-eta-rhs-attribution-diag-max-steps")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--phi_eta_rhs_attribution_diag_max_steps")) != NULL) {
            P.phi_eta_rhs_attribution_diag_max_steps = atoi(v);
            if (P.phi_eta_rhs_attribution_diag_max_steps < 0) P.phi_eta_rhs_attribution_diag_max_steps = 0;
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--phi-eta-rhs-attribution-diag-prefix")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--phi_eta_rhs_attribution_diag_prefix")) != NULL) {
            snprintf(P.phi_eta_rhs_attribution_diag_prefix,
                     sizeof(P.phi_eta_rhs_attribution_diag_prefix), "%s", v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-raw-reaction-drive-only")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_raw_reaction_drive_only")) != NULL) {
            P.gp_raw_reaction_drive_only = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-raw-reaction-drive-use-raw-units-debug")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_raw_reaction_drive_use_raw_units_debug")) != NULL) {
            P.gp_raw_reaction_drive_use_raw_units_debug = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-y-update-mode")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--gp_y_update_mode")) != NULL) {
            if (!is_valid_gp_y_update_mode(v)) {
                fprintf(stderr,
                        "[fatal] invalid --gp_y_update_mode: %s (expected old_rhs, conservative_y_rhs, picard_storage, or storage_exact)\n",
                        v);
                return 2;
            }
            strncpy(P.gp_y_update_mode, v, sizeof(P.gp_y_update_mode) - 1);
            P.gp_y_update_mode[sizeof(P.gp_y_update_mode) - 1] = '\0';
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-to-beta-conversion-mass-audit-enabled")) != NULL) {
            P.gp_to_beta_conversion_mass_audit_enabled = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-to-beta-stop-after-conversion-audit")) != NULL) {
            P.gp_to_beta_stop_after_conversion_audit = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--post-conversion-y-update-audit-enabled")) != NULL) {
            P.post_conversion_y_update_audit_enabled = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--post-conversion-y-update-audit-steps")) != NULL) {
            P.post_conversion_y_update_audit_steps = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--post-conversion-y-update-audit-prefix")) != NULL) {
            strncpy(P.post_conversion_y_update_audit_prefix, v,
                    sizeof(P.post_conversion_y_update_audit_prefix) - 1);
            P.post_conversion_y_update_audit_prefix[sizeof(P.post_conversion_y_update_audit_prefix) - 1] = '\0';
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--y-update-k0-audit-enabled")) != NULL) {
            P.y_update_k0_audit_enabled = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--y-update-k0-audit-steps")) != NULL) {
            P.y_update_k0_audit_steps = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--y-update-k0-audit-prefix")) != NULL) {
            strncpy(P.y_update_k0_audit_prefix, v,
                    sizeof(P.y_update_k0_audit_prefix) - 1);
            P.y_update_k0_audit_prefix[sizeof(P.y_update_k0_audit_prefix) - 1] = '\0';
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--y-update-mass-projection-enabled")) != NULL) {
            P.y_update_mass_projection_enabled = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--y-update-mass-projection-report-enabled")) != NULL) {
            P.y_update_mass_projection_report_enabled = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--y-update-mass-projection-max-iter")) != NULL) {
            P.y_update_mass_projection_max_iter = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--y-update-mass-projection-tol")) != NULL) {
            P.y_update_mass_projection_tol = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--y-update-mass-projection-target-mode")) != NULL) {
            strncpy(P.y_update_mass_projection_target_mode, v,
                    sizeof(P.y_update_mass_projection_target_mode) - 1);
            P.y_update_mass_projection_target_mode[sizeof(P.y_update_mass_projection_target_mode) - 1] = '\0';
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-init-mode")) != NULL) {
            strncpy(P.gp_init_mode, v, sizeof(P.gp_init_mode) - 1);
            P.gp_init_mode[sizeof(P.gp_init_mode) - 1] = '\0';
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-init-mass-mode")) != NULL) {
            strncpy(P.gp_init_mass_mode, v, sizeof(P.gp_init_mass_mode) - 1);
            P.gp_init_mass_mode[sizeof(P.gp_init_mass_mode) - 1] = '\0';
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-obs-target-radius-nm")) != NULL) {
            P.gp_obs_target_radius_nm = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-obs-eta-peak")) != NULL) {
            P.gp_obs_eta_peak = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-obs-iface-width-nm")) != NULL) {
            P.gp_obs_iface_width_nm = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-obs-profile-type")) != NULL) {
            strncpy(P.gp_obs_profile_type, v, sizeof(P.gp_obs_profile_type) - 1);
            P.gp_obs_profile_type[sizeof(P.gp_obs_profile_type) - 1] = '\0';
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-obs-match-mode")) != NULL) {
            strncpy(P.gp_obs_match_mode, v, sizeof(P.gp_obs_match_mode) - 1);
            P.gp_obs_match_mode[sizeof(P.gp_obs_match_mode) - 1] = '\0';
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-obs-compensation-mode")) != NULL) {
            strncpy(P.gp_obs_compensation_mode, v, sizeof(P.gp_obs_compensation_mode) - 1);
            P.gp_obs_compensation_mode[sizeof(P.gp_obs_compensation_mode) - 1] = '\0';
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-obs-depletion-radius-factor")) != NULL) {
            P.gp_obs_depletion_radius_factor = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-obs-depletion-smooth-width-factor")) != NULL) {
            P.gp_obs_depletion_smooth_width_factor = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-obs-min-xB-alpha")) != NULL) {
            P.gp_obs_min_xB_alpha = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-obs-max-xB-alpha")) != NULL) {
            P.gp_obs_max_xB_alpha = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-to-beta-event-cooldown-steps")) != NULL) {
            P.gp_to_beta_event_cooldown_steps = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-to-beta-min-event-spacing")) != NULL) {
            P.gp_to_beta_min_event_spacing = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-to-beta-event-exclusion-radius")) != NULL) {
            P.gp_to_beta_event_exclusion_radius = atof(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-to-beta-max-events-global")) != NULL) {
            P.gp_to_beta_max_events_global = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-to-beta-max-events-per-window")) != NULL) {
            P.gp_to_beta_max_events_per_window = atoi(v);
            continue;
        }
        if ((v = get_flag_value(argc, argv, &i, "--gp-to-beta-event-window-steps")) != NULL) {
            P.gp_to_beta_event_window_steps = atoi(v);
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
        if ((v = get_flag_value(argc, argv, &i, "--thermo-convex-extrapolation-enabled")) != NULL ||
            (v = get_flag_value(argc, argv, &i, "--thermo_convex_extrapolation_enabled")) != NULL) {
            P.thermo_convex_extrapolation_enabled = atoi(v);
            continue;
        }
        // TODO: 若后续增加 --mu-reference-scale / --v-A / --v-B 等热力学相关 flag，
        // 也应在此处解析并在解析完成后调用 pfparams_refresh_thermo(&P)。
    }

    // 在解析完所有可选参数（包括温度等热力学参数）之后，刷新热力学量。
    pfparams_refresh_thermo(&P);
    if (has_xB_matrix_override) {
        P.ic_xB_eq_matrix = xB_matrix_override;
    }
    sync_thermo_runtime_flags(&P);
    P.xB_ref_for_eps_c = P.ic_xB_eq_matrix;
    if (!validate_physical_params_ready(&P)) {
        return 2;
    }
    if (!normalize_gp_eta_interface_units_host(&P)) {
        return 2;
    }
    {
        const double w_phys_A = 12.0 * P.gamma_Jm2 / P.lambda_sm_m;
        const double w_phys_B = (fabs(P.mu_reference_scale) > 1.0e-30 && P.Vm_alpha_0_phys_m3mol > 0.0)
                                    ? (P.mu_reference_scale * (1.0 / P.Vm_alpha_0_phys_m3mol))
                                    : NAN;
        if (!(isfinite(w_phys_A) && w_phys_A > 0.0)) {
            fprintf(stderr, "[fatal] invalid w_phys_A derived from gamma_Jm2/lambda_sm_m\n");
            return 2;
        }
        if (isfinite(w_phys_B) && w_phys_B > 0.0) {
            const double rel = fabs(w_phys_A - w_phys_B) / fmax(fabs(w_phys_A), 1.0e-30);
            if (rel > 1.0e-6) {
                fprintf(stderr,
                        "[fatal] w_phys consistency mismatch: A=%.8e B=%.8e. 请重新跑 Unit_Psedobinary.py\n",
                        w_phys_A, w_phys_B);
                return 2;
            }
        }
        const double dx_phys = compute_eta_ref_dx_phys_m_host(&P);
        if (!(isfinite(dx_phys) && dx_phys > 0.0)) {
            fprintf(stderr, "[fatal] cannot resolve phys_dx_ref for D_alpha consistency check\n");
            return 2;
        }
        const double D_alpha_phys_expected = D_Ag_in_PbTe_m2_per_s(P.temperature_C + 273.15);
        const double D_alpha_phys_inferred = P.D_alpha * dx_phys * dx_phys / P.t_real_unit;
        const double L_ref_inferred = sqrt(D_alpha_phys_inferred * P.t_real_unit);
        if (!(isfinite(D_alpha_phys_inferred) && D_alpha_phys_inferred > 0.0 &&
              isfinite(L_ref_inferred) && L_ref_inferred > 0.0)) {
            fprintf(stderr, "[fatal] invalid D_alpha/t_real_unit consistency; 请重新跑 Unit_Psedobinary.py\n");
            return 2;
        }
        const double identity_err = fabs((L_ref_inferred * L_ref_inferred) / (D_alpha_phys_inferred * P.t_real_unit) - 1.0);
        if (identity_err > 1.0e-6) {
            fprintf(stderr, "[fatal] L_ref/D_alpha/t_real_unit identity failed; 请重新跑 Unit_Psedobinary.py\n");
            return 2;
        }
        const double rel_D_alpha = fabs(D_alpha_phys_inferred - D_alpha_phys_expected) / fmax(D_alpha_phys_expected, 1.0e-30);
        if (rel_D_alpha > 1.0e-3) {
            fprintf(stderr,
                    "[fatal] D_alpha consistency mismatch: inferred=%.8e expected=%.8e. 请重新跑 Unit_Psedobinary.py\n",
                    D_alpha_phys_inferred, D_alpha_phys_expected);
            return 2;
        }
    }

    if (P.mode == 1) {
        P.nsteps = P.minimize_max_iter;
        P.dt = P.minimize_dt;
    }
    if (P.minimize_continue_from_vtk) {
        if (P.mode != 0 && P.mode != 1) {
            fprintf(stderr, "[fatal] continuation 仅支持 dynamics/minimize 模式。请使用 --mode=dynamics-continue、--mode=minimize-continue 或 --mode=minimize。\n");
            return 2;
        }
        if (P.continue_phi_vtk_path[0] == '\0') {
            fprintf(stderr, "[fatal] continuation 缺少 --continue-phi-vtk <path>。\n");
            return 2;
        }
    }
    if (P.init_mode_raw_fields) {
        if (P.minimize_continue_from_vtk) {
            fprintf(stderr, "[fatal] --init-mode raw_fields cannot be combined with --continue-phi-vtk/--continue-xB-vtk.\n");
            return 2;
        }
        if (!(P.mode == 0 || (P.mode == 1 && P.minimize_full_model == 1))) {
            fprintf(stderr, "[fatal] --init-mode raw_fields requires dynamics mode or full-model minimize (--mode=minimize --minimize-full-model).\n");
            return 2;
        }
        if (P.init_phi_raw_path[0] == '\0' || P.init_xB_raw_path[0] == '\0' || P.init_meta_path[0] == '\0') {
            fprintf(stderr, "[fatal] raw_fields requires --init-phi-raw, --init-xB-raw, and --init-meta.\n");
            return 2;
        }
    }
    if (P.scheduled_nuc_enabled) {
        if (P.mode != 0) {
            fprintf(stderr, "[fatal] scheduled nucleation test is dynamics-only.\n");
            return 2;
        }
        if (P.scheduled_nuc_profile_dir[0] == '\0' && !P.scheduled_nuc_fallback_analytic_sphere) {
            fprintf(stderr, "[fatal] scheduled nucleation test requires --scheduled-nuc-profile-dir for the no-strain dynamic-continue source.\n");
            return 2;
        }
        if (P.scheduled_nuc_source_dyn_dir[0] == '\0') {
            fprintf(stderr, "[fatal] scheduled nucleation test requires --scheduled-nuc-source-dyn-dir for source diagnostics/geometry.\n");
            return 2;
        }
        if (strcmp(P.scheduled_nuc_xB_edge_mode, "sample-current-background-shell") != 0) {
            fprintf(stderr, "[fatal] scheduled nucleation test currently supports only --scheduled-nuc-xB-edge-mode sample-current-background-shell.\n");
            return 2;
        }
        double e0_norm = fabs(P.E0_xx) + fabs(P.E0_yy) + fabs(P.E0_zz) +
                         fabs(P.E0_yz) + fabs(P.E0_xz) + fabs(P.E0_xy);
        if (e0_norm > 1.0e-14 && !P.elastic_enabled) {
            fprintf(stderr,
                    "[warn] scheduled nucleation test has nonzero external strain but elastic=0; strain will not affect elastic terms.\n");
        }
        if (P.scheduled_nuc_scale_interface_width <= 0.0) {
            P.scheduled_nuc_scale_interface_width =
                P.scheduled_nuc_target_lambda_nm / fmax(P.scheduled_nuc_source_lambda_nm, 1.0e-30);
        }
        if (P.scheduled_nuc_scale_xB_profile_width <= 0.0) {
            P.scheduled_nuc_scale_xB_profile_width = P.scheduled_nuc_scale_interface_width;
        }
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
    const int gp_elastic_solver_enabled =
        (is_gp_zone_mode(&P) && P.gp_elastic_enabled) ? 1 : 0;
    const int gp_active_eta_coupling_enabled =
        (gp_elastic_solver_enabled && P.gp_elastic_active_eta) ? 1 : 0;
    const int gp_active_phi_coupling_enabled =
        (gp_elastic_solver_enabled && P.gp_elastic_active_phi) ? 1 : 0;
    const int phi_elastic_coupling_enabled =
        (P.elastic_enabled && (!gp_elastic_solver_enabled || gp_active_phi_coupling_enabled)) ? 1 : 0;

    if (P.out_every > P.nsteps) {
        fprintf(stderr, "[warn] 输出间隔 (%d) 大于总时间步数 (%d)，将调整为最后一步输出\n", P.out_every, P.nsteps);
        P.out_every = P.nsteps;
    }
    if (P.csv_out_every > P.nsteps) {
        fprintf(stderr, "[warn] CSV输出间隔 (%d) 大于总时间步数 (%d)，将调整为最后一步输出\n", P.csv_out_every, P.nsteps);
        P.csv_out_every = P.nsteps;
    }

    RawInitMeta raw_init_meta;
    memset(&raw_init_meta, 0, sizeof(raw_init_meta));
    if (P.init_mode_raw_fields) {
        if (!load_raw_init_meta(P.init_meta_path, &raw_init_meta)) {
            return 2;
        }
    }

    {
        double temperature_K = P.temperature_C + 273.15;
        char grid_buf[64];
        // NOTE: In this codebase, P.dx/P.dy/P.dz are typically *grid units* (often 1.0),
        // while the physical spacing is provided via (lambda_sm_m, ic_phi_iface_w) where:
        //   ic_phi_iface_w = (lambda_sm / dx_phys) / 2
        // So infer dx_phys from that relation for correct physical reporting and naming.
        const double dx_phys_m = (P.ic_phi_iface_w > 1e-30) ? (P.lambda_sm_m / (2.0 * P.ic_phi_iface_w)) : (P.dx * 1.0e-9);
        const double dy_phys_m = (P.ic_phi_iface_w > 1e-30) ? (P.lambda_sm_m / (2.0 * P.ic_phi_iface_w)) : (P.dy * 1.0e-9);
        const double dz_phys_m = (P.ic_phi_iface_w > 1e-30) ? (P.lambda_sm_m / (2.0 * P.ic_phi_iface_w)) : (P.dz * 1.0e-9);
        const double Lx_phys_m = P.Nx * dx_phys_m;
        const double Ly_phys_m = P.Ny * dy_phys_m;
        const double Lz_phys_m = P.Nz * dz_phys_m;
        const double lambda_over_dx = (dx_phys_m > 0.0) ? (P.lambda_sm_m / dx_phys_m) : 0.0;
        const char *mode_label =
            (P.mode == 0)
                ? (P.init_mode_raw_fields ? "dynamics raw_fields"
                   : (P.minimize_continue_from_vtk ? "dynamics-continue" : "dynamics"))
                : (P.minimize_full_model
                       ? "minimize (full-model: chem+diffusion+volume)"
                       : "minimize (phi-only)");
        if (P.init_mode_raw_fields &&
            !validate_raw_init_meta_against_run(&raw_init_meta, &P,
                                                dx_phys_m * 1.0e9,
                                                P.lambda_sm_m * 1.0e9)) {
            return 2;
        }
        snprintf(grid_buf, sizeof(grid_buf), "%dx%dx%d", P.Nx, P.Ny, P.Nz);

        log_section_header("Run Configuration");
        log_kv_text("mode", "%s", mode_label);
        log_kv_text("model_mode", "%s", P.model_mode);
        if (is_gp_zone_mode(&P)) {
            const double gp_mu_ref_mech_mix =
                compute_gp_mu_reference_mech_mix_host(
                    P.gp_xB_fixed, temperature_K, P.gp_delta_g_stab);
            const double x_probe_matrix = clamp_fraction_eps(P.ic_xB_eq_matrix);
            const double minus_dmu_probe_matrix =
                P.gp_reaction_nu_A * mu_PbTe_calphad(temperature_K, x_probe_matrix) +
                P.gp_reaction_nu_B * mu_Ag2Te_calphad(temperature_K, x_probe_matrix) -
                gp_mu_ref_mech_mix;
            const double minus_dmu_probe_003 =
                P.gp_reaction_nu_A * mu_PbTe_calphad(temperature_K, 0.03) +
                P.gp_reaction_nu_B * mu_Ag2Te_calphad(temperature_K, 0.03) -
                gp_mu_ref_mech_mix;
            const double minus_dmu_probe_005 =
                P.gp_reaction_nu_A * mu_PbTe_calphad(temperature_K, 0.05) +
                P.gp_reaction_nu_B * mu_Ag2Te_calphad(temperature_K, 0.05) -
                gp_mu_ref_mech_mix;
            printf("[GP-REF] mode          : mechanical-mixture\n");
            printf("[GP-REF] G_PbTe_Solid  : %.4f J/mol\n",
                   G_PbTe_Solid(temperature_K));
            printf("[GP-REF] G_Ag2Te_Solid : %.4f J/mol\n",
                   G_Ag2Te_Solid(temperature_K));
            printf("[GP-REF] mu_GP0        : %.4f J/mol\n",
                   gp_mu_ref_mech_mix);
            printf("[GP-REF] delta_g_stab  : %.1f J/mol\n",
                   P.gp_delta_g_stab);
            printf("[GP-REF] minus_delta_mu_r_GP(xB_matrix=%.4f) : %.4f J/mol\n",
                   x_probe_matrix, minus_dmu_probe_matrix);
            printf("[GP-REF] minus_delta_mu_r_GP(xB=0.0300)      : %.4f J/mol\n",
                   minus_dmu_probe_003);
            printf("[GP-REF] minus_delta_mu_r_GP(xB=0.0500)      : %.4f J/mol\n",
                   minus_dmu_probe_005);
            log_kv_text("gp_elastic_enabled", "%d", P.gp_elastic_enabled);
            log_kv_text("gp_elastic_active_eta", "%d", P.gp_elastic_active_eta);
            log_kv_text("gp_elastic_active_phi", "%d", P.gp_elastic_active_phi);
            log_kv_text("gp_elastic_derivative_scale", "%.6e", P.gp_elastic_derivative_scale);
            log_kv_text("gp_nuc_enabled", "%d", P.gp_nuc_enabled);
            log_kv_text("gp_nuc_check_interval", "%d", P.gp_nuc_check_interval);
            log_kv_text("gp_nuc_phi_threshold", "%.6e", P.gp_nuc_phi_threshold);
            log_kv_text("gp_nuc_eta_threshold", "%.6e", P.gp_nuc_eta_threshold);
            log_kv_text("gp_nuc_h_alpha_threshold", "%.6e", P.gp_nuc_h_alpha_threshold);
            log_kv_text("gp_nuc_J0", "%.6e", P.gp_nuc_J0);
            log_kv_text("gp_nuc_gamma", "%.6e", P.gp_nuc_gamma);
            log_kv_text("gp_nuc_seed_radius", "%.6e", P.gp_nuc_seed_radius);
            log_kv_text("gp_nuc_seed_peak", "%.6e", P.gp_nuc_seed_peak);
            log_kv_text("gp_nuc_seed_iface_width", "%.6e", P.gp_nuc_seed_iface_width);
            log_kv_text("gp_nuc_patch_radius", "%.6e", P.gp_nuc_patch_radius);
            log_kv_text("gp_nuc_shell_inner_radius", "%.6e", P.gp_nuc_shell_inner_radius);
            log_kv_text("gp_nuc_shell_outer_radius", "%.6e", P.gp_nuc_shell_outer_radius);
            log_kv_text("gp_nuc_max_events_per_step", "%d", P.gp_nuc_max_events_per_step);
            log_kv_text("gp_nuc_mass_mode", "%s", P.gp_nuc_mass_mode);
            log_kv_text("gp_to_beta_enabled", "%d", P.gp_to_beta_enabled);
            log_kv_text("gp_to_beta_check_interval", "%d", P.gp_to_beta_check_interval);
            log_kv_text("gp_to_beta_eta_threshold", "%.6e", P.gp_to_beta_eta_threshold);
            log_kv_text("gp_to_beta_radius_threshold", "%.6e", P.gp_to_beta_radius_threshold);
            log_kv_text("gp_to_beta_xB_threshold", "%.6e", P.gp_to_beta_xB_threshold);
            log_kv_text("gp_to_beta_seed_radius", "%.6e", P.gp_to_beta_seed_radius);
            log_kv_text("gp_to_beta_seed_peak", "%.6e", P.gp_to_beta_seed_peak);
            log_kv_text("gp_to_beta_seed_iface_width", "%.6e", P.gp_to_beta_seed_iface_width);
            log_kv_text("gp_to_beta_patch_radius", "%.6e", P.gp_to_beta_patch_radius);
            log_kv_text("gp_to_beta_shell_inner_radius", "%.6e", P.gp_to_beta_shell_inner_radius);
            log_kv_text("gp_to_beta_shell_outer_radius", "%.6e", P.gp_to_beta_shell_outer_radius);
            log_kv_text("gp_to_beta_max_events_per_step", "%d", P.gp_to_beta_max_events_per_step);
            log_kv_text("gp_to_beta_stochastic_enabled", "%d", P.gp_to_beta_stochastic_enabled);
            log_kv_text("gp_to_beta_J0_site", "%.6e", P.gp_to_beta_J0_site);
            log_kv_text("gp_to_beta_gamma", "%.6e", P.gp_to_beta_gamma);
            log_kv_text("gp_to_beta_drive_const", "%.6e", P.gp_to_beta_drive_const);
            log_kv_text("gp_to_beta_max_events_per_check", "%d", P.gp_to_beta_max_events_per_check);
            log_kv_text("gp_to_beta_barrier_mode", "%s", P.gp_to_beta_barrier_mode);
            log_kv_text("gp_to_beta_drive_mode", "%s", P.gp_to_beta_drive_mode);
            log_kv_text("gp_to_beta_conversion_mass_audit_enabled", "%d",
                        P.gp_to_beta_conversion_mass_audit_enabled);
            log_kv_text("gp_to_beta_stop_after_conversion_audit", "%d",
                        P.gp_to_beta_stop_after_conversion_audit);
            log_kv_text("gp_to_beta_conversion_audit_prefix", "%s",
                        P.gp_to_beta_conversion_audit_prefix);
            log_kv_text("gp_to_beta_feasibility_gate_enabled", "%d",
                        P.gp_to_beta_feasibility_gate_enabled);
            log_kv_text("gp_to_beta_min_shell_capacity_factor", "%.6e",
                        P.gp_to_beta_min_shell_capacity_factor);
            log_kv_text("gp_to_beta_reject_if_infeasible", "%d",
                        P.gp_to_beta_reject_if_infeasible);
            log_kv_text("gp_to_beta_allow_seed_amplitude_scaling", "%d",
                        P.gp_to_beta_allow_seed_amplitude_scaling);
            log_kv_text("gp_to_beta_min_seed_amplitude", "%.6e",
                        P.gp_to_beta_min_seed_amplitude);
            log_kv_text("gp_to_beta_event_cooldown_steps", "%d",
                        P.gp_to_beta_event_cooldown_steps);
            log_kv_text("gp_to_beta_min_event_spacing", "%.6e",
                        P.gp_to_beta_min_event_spacing);
            log_kv_text("gp_to_beta_event_exclusion_radius", "%.6e",
                        P.gp_to_beta_event_exclusion_radius);
            log_kv_text("gp_to_beta_max_events_global", "%d",
                        P.gp_to_beta_max_events_global);
            log_kv_text("gp_to_beta_max_events_per_window", "%d",
                        P.gp_to_beta_max_events_per_window);
            log_kv_text("gp_to_beta_event_window_steps", "%d",
                        P.gp_to_beta_event_window_steps);
            log_kv_text("post_conversion_y_update_audit_enabled", "%d",
                        P.post_conversion_y_update_audit_enabled);
            log_kv_text("post_conversion_y_update_audit_steps", "%d",
                        P.post_conversion_y_update_audit_steps);
            log_kv_text("post_conversion_y_update_audit_prefix", "%s",
                        P.post_conversion_y_update_audit_prefix);
            log_kv_text("y_update_k0_audit_enabled", "%d",
                        P.y_update_k0_audit_enabled);
            log_kv_text("y_update_k0_audit_steps", "%d",
                        P.y_update_k0_audit_steps);
            log_kv_text("y_update_k0_audit_prefix", "%s",
                        P.y_update_k0_audit_prefix);
            log_kv_text("y_update_mass_projection_enabled", "%d",
                        P.y_update_mass_projection_enabled);
            log_kv_text("y_update_mass_projection_report_enabled", "%d",
                        P.y_update_mass_projection_report_enabled);
            log_kv_text("y_update_mass_projection_max_iter", "%d",
                        P.y_update_mass_projection_max_iter);
            log_kv_text("y_update_mass_projection_tol", "%.6e",
                        P.y_update_mass_projection_tol);
            log_kv_text("y_update_mass_projection_target_mode", "%s",
                        P.y_update_mass_projection_target_mode);
            log_kv_text("gp_to_beta_mass_mode", "%s", P.gp_to_beta_mass_mode);
            log_kv_text("gp_to_beta_eta_deplete_mode", "%s", P.gp_to_beta_eta_deplete_mode);
            log_kv_text("gp_to_beta_phi_insert_mode", "%s", P.gp_to_beta_phi_insert_mode);
            log_kv_text("gp_C_mode", "%s", P.gp_C_mode);
            log_kv_text("gp_eps_mode", "%s", P.gp_eps_mode);
            log_kv_text("gp_eps_iso", "%.6e", P.gp_eps_iso);
            log_kv_text("gp_eta_mass_limiter", "%s", P.gp_eta_mass_limiter);
            log_kv_text("gp_y_update_mode", "%s", P.gp_y_update_mode);
            log_kv_text("gp_y_picard_iters", "%d", P.gp_y_picard_iters);
            log_kv_text("gp_raw_reaction_drive_only", "%d", P.gp_raw_reaction_drive_only);
            log_kv_text("gp_raw_reaction_drive_use_raw_units_debug", "%d",
                        P.gp_raw_reaction_drive_use_raw_units_debug);
            log_kv_text("gp_reaction_nu_A", "%.8e", P.gp_reaction_nu_A);
            log_kv_text("gp_reaction_nu_B", "%.8e", P.gp_reaction_nu_B);
            log_kv_text("gp_gamma_alpha_gp", "%.8e", P.gp_gamma_alpha_gp);
            log_kv_text("gp_l_eta_nm", "%.8e", P.gp_l_eta_nm);
            log_kv_text("gp_D_ratio", "%.8e", P.gp_D_ratio);
            log_kv_text("gp_L_eta_mode", "%s", P.gp_L_eta_mode);
            log_kv_text("gp_M_int_eta", "%.8e", P.gp_M_int_eta);
            log_kv_text("gp_M_eta_phys", "%.8e", P.gp_M_eta_phys);
            log_kv_text("gp_M_eta_ratio_to_crit", "%.8e", P.gp_M_eta_ratio_to_crit);
            log_kv_text("gp_W_eta_input_source", "%s",
                        gp_eta_input_mode_label(P.gp_W_eta_input_mode_resolved));
            log_kv_text("gp_kappa_eta_input_source", "%s",
                        gp_eta_input_mode_label(P.gp_kappa_eta_input_mode_resolved));
            log_kv_text("gp_W_eta_phys_input", "%.8e", P.gp_W_eta_phys_input);
            log_kv_text("gp_kappa_eta_phys_input", "%.8e", P.gp_kappa_eta_phys_input);
            log_kv_text("gp_W_eta_code_input", "%.8e", P.gp_W_eta_code_input);
            log_kv_text("gp_kappa_eta_code_input", "%.8e", P.gp_kappa_eta_code_input);
            log_kv_text("gp_xB_eq_alpha_for_eta", "%.8e", P.gp_xB_eq_alpha_for_eta);
            log_kv_text("gp_xB_gp_ref(reuse_gp_xB_fixed)", "%.8e", P.gp_xB_fixed);
            log_kv_text("gp_kinetic_ref_enabled", "%d", P.gp_kinetic_ref_enabled);
            log_kv_text("gp_kinetic_ref_apply", "%d", P.gp_kinetic_ref_apply);
            log_kv_text("phi_eta_step_delta_diag_enabled", "%d", P.phi_eta_step_delta_diag_enabled);
            log_kv_text("phi_eta_step_delta_diag_every", "%d", P.phi_eta_step_delta_diag_every);
            log_kv_text("phi_eta_step_delta_diag_max_steps", "%d", P.phi_eta_step_delta_diag_max_steps);
            log_kv_text("phi_eta_step_delta_diag_prefix", "%s", P.phi_eta_step_delta_diag_prefix);
            log_kv_text("phi_eta_rhs_attribution_diag_enabled", "%d", P.phi_eta_rhs_attribution_diag_enabled);
            log_kv_text("phi_eta_rhs_attribution_diag_every", "%d", P.phi_eta_rhs_attribution_diag_every);
            log_kv_text("phi_eta_rhs_attribution_diag_max_steps", "%d", P.phi_eta_rhs_attribution_diag_max_steps);
            log_kv_text("phi_eta_rhs_attribution_diag_prefix", "%s", P.phi_eta_rhs_attribution_diag_prefix);
            log_kv_text("gp_delta_g_stab", "%.8e", P.gp_delta_g_stab);
            log_kv_text("gp_mu_reference_raw", "%.8e",
                        compute_gp_mu_reference_mech_mix_host(
                            P.gp_xB_fixed, temperature_K, P.gp_delta_g_stab));
            if (P.gp_raw_reaction_drive_only && P.gp_raw_reaction_drive_use_raw_units_debug) {
                fprintf(stdout,
                        "WARNING: using raw-unit GP eta dgbulk_deta path; this is not compatible with code-unit W/kappa and is for debugging only.\n");
            }
        }
        log_kv_text("init_mode", "%s", P.init_mode_raw_fields ? "raw_fields" : "standard");
        if (effective_pf_param_file[0] != '\0') {
            log_kv_text("pf_param_file", "%s", effective_pf_param_file);
        }
        log_kv_text("grid", "%s", grid_buf);
        log_kv_text("dx / dy / dz (grid)", "%.6f / %.6f / %.6f", P.dx, P.dy, P.dz);
        log_kv_text("cell_size_phys_nm", "%.6f / %.6f / %.6f",
                    dx_phys_m * 1.0e9, dy_phys_m * 1.0e9, dz_phys_m * 1.0e9);
        log_kv_text("system_size_phys_nm", "%.6f / %.6f / %.6f",
                    Lx_phys_m * 1.0e9, Ly_phys_m * 1.0e9, Lz_phys_m * 1.0e9);
        log_kv_text("system_size_phys_m", "%.6e / %.6e / %.6e",
                    Lx_phys_m, Ly_phys_m, Lz_phys_m);
        log_kv_text("t_real_unit_s", "%.6e", P.t_real_unit);
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
        log_kv_text("external_strain(E0)", "xx=%.6e yy=%.6e zz=%.6e yz=%.6e xz=%.6e xy=%.6e",
                    P.E0_xx, P.E0_yy, P.E0_zz, P.E0_yz, P.E0_xz, P.E0_xy);
        log_kv_text("diag_vtk", "%s", log_enabled_cn(P.diag_vtk_enabled));
        log_kv_text("diag_elastic_bulk", "%s", log_enabled_cn(P.diag_elastic_bulk_penalty_enabled));
        if (P.scheduled_nuc_enabled) {
            log_kv_text("scheduled_nucleation_test", "enabled");
            log_kv_text("scheduled_source_case", "%s", P.scheduled_nuc_source_case_label);
            log_kv_text("scheduled_source_dyn_dir", "%s", P.scheduled_nuc_source_dyn_dir);
            log_kv_text("scheduled_profile_dir", "%s", P.scheduled_nuc_profile_dir);
            log_kv_text("scheduled_source_dx/lambda_nm", "%.6f / %.6f",
                        P.scheduled_nuc_source_dx_nm, P.scheduled_nuc_source_lambda_nm);
            log_kv_text("scheduled_target_lambda_nm", "%.6f", P.scheduled_nuc_target_lambda_nm);
            log_kv_text("scheduled_scale_geom/interface/xB", "%.6f / %.6f / %.6f",
                        P.scheduled_nuc_scale_geometry,
                        P.scheduled_nuc_scale_interface_width,
                        P.scheduled_nuc_scale_xB_profile_width);
            log_kv_text("scheduled_steps", "%s", P.scheduled_nuc_steps_csv);
            log_kv_text("scheduled_centers_nm", "%s", P.scheduled_nuc_centers_nm);
        } else {
            log_kv_text("scheduled_nucleation_test", "disabled");
        }
        log_kv_text("dynamics_mass_diagnostics", "%s (interval=%d)",
                    P.dynamics_mass_diag_enabled ? "enabled" : "disabled",
                    P.dynamics_mass_diag_interval);
        log_kv_text("Y_rhs_previous_time_level", "%s",
                    P.enable_Y_rhs_previous_time_level ? "enabled" : "disabled");
        printf("[RUN PARAM] Nx Ny Nz = %d %d %d, dx_nm = %.6f, interface_width_nm / lambda_sm = %.6f, dt = %.6e, init_mode = %s, scheduled nucleation test = %s\n",
               P.Nx, P.Ny, P.Nz,
               dx_phys_m * 1.0e9,
               P.lambda_sm_m * 1.0e9,
               P.dt,
               P.init_mode_raw_fields ? "raw_fields" : "standard",
               P.scheduled_nuc_enabled ? "enabled" : "disabled");

        log_section_header("Thermodynamics");
        log_kv_text("T_C", "%.6f", P.temperature_C);
        log_kv_text("T_K", "%.6f", temperature_K);
        log_kv_text("thermo_convex_extrapolation_enabled", "%d",
                    P.thermo_convex_extrapolation_enabled);
        log_kv_text("xB_eq", "%.8e", P.ic_xB_eq_matrix);
        log_kv_text("mu0_compound_hat", "%.8e", P.mu0_compound);
        log_kv_text("mu_reference_scale", "%.8e", P.mu_reference_scale);
        if (is_gp_zone_mode(&P)) {
            log_kv_text("gp_mu_reference_raw", "%.8e",
                        compute_gp_mu_reference_mech_mix_host(
                            P.gp_xB_fixed, temperature_K, P.gp_delta_g_stab));
        }
        if (is_gp_zone_mode(&P) && P.gp_nuc_enabled) {
            log_kv_text("gp_nuc_probe_delta_g_x0.03_Tcurrent", "%.8e",
                        compute_delta_g_nuc_GP_host(0.03, temperature_K, P.mu_reference_scale, P.gp_xB_fixed));
        }

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
    char results_root[4096];
    const char *results_root_env = getenv("CUDA_STO_RESULTS_ROOT");
    if (results_root_env && results_root_env[0] != '\0') {
        snprintf(results_root, sizeof(results_root), "%s", results_root_env);
    } else {
        snprintf(results_root, sizeof(results_root), "%s", "Results");
    }
    char run_dir_name[256];
    char output_dir[4096];
    char output_pf_input_file[4096];
    // For naming: infer the physical dx (m) from (lambda_sm_m, ic_phi_iface_w) when available.
    // Then convert "internal length units" (the same units used by P.dx and --radius) into meters.
    const double dx_phys_m_run = (P.ic_phi_iface_w > 1e-30)
                                     ? (P.lambda_sm_m / (2.0 * P.ic_phi_iface_w))
                                     : (P.dx * 1.0e-9);
    const double unit_to_m_run = (fabs(P.dx) > 1e-30) ? (dx_phys_m_run / P.dx) : 1.0e-9;

    // Apply --radius-phys-nm if present.
    if (has_radius_phys_nm) {
        if (radius_phys_nm < 0.0) {
            fprintf(stderr, "[fatal] --radius-phys-nm must be >= 0, got %.6f\n", radius_phys_nm);
            return 2;
        }
        if (radius_phys_nm == 0.0 && !P.init_mode_raw_fields) {
            fprintf(stderr, "[fatal] --radius-phys-nm=0 is only allowed with --init-mode raw_fields for matrix-only reference initialization.\n");
            return 2;
        }
        if (unit_to_m_run <= 0.0) {
            fprintf(stderr, "[fatal] invalid unit_to_m_run=%.6e, cannot convert physical radius.\n", unit_to_m_run);
            return 2;
        }
        // Convert physical nm -> internal length units.
        P.ic_phi_seed_radius = (radius_phys_nm * 1.0e-9) / unit_to_m_run;
    }

    // ============================================================
    // Consistency diagnostics: interface width / length scaling
    // ============================================================
    if (P.ic_phi_iface_w > 1e-30) {
        const double ic = P.ic_phi_iface_w;
        // Strict dual-dx interpretation:
        // - ic_phi_iface_w is set by the PF discretization width: ic = lambda_sm / (2 * pf_dx)
        // - P.dx is the solver-space spacing normalized by phys_dx_ref: P.dx = pf_dx / phys_dx_ref
        // - kappa_phi remains calibrated on phys_dx_ref, so the consistent expectation is
        //   kappa_phi = 0.5 * (lambda_sm / (2*phys_dx_ref))^2 = 0.5 * (ic * P.dx)^2
        const double kappa_expected = 0.5 * (ic * P.dx) * (ic * P.dx);
        const double denom = fabs(kappa_expected) > 1e-30 ? fabs(kappa_expected) : 1.0;
        const double rel = fabs(P.kappa_phi - kappa_expected) / denom;
        log_section_header("Interface-Width Consistency (Diagnostics)");
        log_kv_text("dx_internal", "%.6g", P.dx);
        log_kv_text("dx_phys_nm(inferred)", "%.6f", dx_phys_m_run * 1.0e9);
        log_kv_text("unit_to_nm(internal->phys)", "%.6e", unit_to_m_run * 1.0e9);
        log_kv_text("ic_phi_iface_w(grids_halfwidth)", "%.6f", ic);
        log_kv_text("interface_width_grids(2*ic)", "%.6f", 2.0 * ic);
        log_kv_text("interface_width_nm(2*ic*dx_phys)", "%.6f", (2.0 * ic) * (dx_phys_m_run * 1.0e9));
        log_kv_text("kappa_phi_loaded", "%.8e", P.kappa_phi);
        log_kv_text("kappa_phi_expected_from_ic(0.5*(ic*dx)^2)", "%.8e", kappa_expected);
        log_kv_text("kappa_phi_rel_error", "%.3e", rel);
        if (rel > 5e-3) {
            fprintf(stderr,
                    "[warn] kappa_phi appears inconsistent with ic_phi_iface_w. "
                    "This can cause the equilibrium interface width to differ from the seeded width.\n");
            fprintf(stderr,
                    "       loaded kappa_phi=%.8e, expected ~ %.8e (rel_err=%.3e). "
                    "Check dx/lambda_sm_m/ic_phi_iface_w consistency and regenerate pf_input.params.\n",
                    P.kappa_phi, kappa_expected, rel);
        }

        // Seed sharpness diagnostic: if R is too small compared to w, phi center cannot reach ~1.
        if (P.ic_phi_seed_radius > 0.0 && P.dx > 0.0) {
            const double w_init = ic * P.dx;
            if (w_init > 1e-30) {
                const double R_over_w = P.ic_phi_seed_radius / w_init;
                const double phi_center = 0.5 * (1.0 + tanh(R_over_w));
                log_kv_text("seed_R_internal", "%.6g", P.ic_phi_seed_radius);
                log_kv_text("seed_w_init_internal(ic*dx)", "%.6g", w_init);
                log_kv_text("seed_R_over_w", "%.6g", R_over_w);
                log_kv_text("seed_phi_center_ideal(0.5*(1+tanh(R/w)))", "%.6f", phi_center);
                if (phi_center < 0.95) {
                    fprintf(stderr,
                            "[warn] Seed radius is not large compared to interface width: "
                            "phi_center_ideal=%.3f (<0.95). This often indicates a radius-unit mismatch "
                            "(physical nm vs internal units) or too-small R.\n",
                            phi_center);
                }
            }
        }
        fflush(stdout);
    }
    mkdir(results_root, 0755);
    if (P.minimize_continue_from_vtk) {
        char source_output_root[4096];
        char source_output_root_base[256];
        source_output_root[0] = '\0';
        source_output_root_base[0] = '\0';

        if (derive_continue_output_root(P.continue_phi_vtk_path,
                                        source_output_root,
                                        sizeof(source_output_root))) {
            path_basename_copy(source_output_root,
                               source_output_root_base,
                               sizeof(source_output_root_base));
        }

        if (results_root_env && results_root_env[0] != '\0' &&
            source_output_root_base[0] != '\0') {
            snprintf(output_dir, sizeof(output_dir), "%s/%s", results_root, source_output_root_base);
        } else if (source_output_root[0] != '\0') {
            snprintf(output_dir, sizeof(output_dir), "%s", source_output_root);
        } else {
            snprintf(output_dir, sizeof(output_dir), "%s", results_root);
        }
        mkdir(output_dir, 0755);
    } else {
        if (P.diag_elastic_bulk_penalty_enabled || P.mode == 1) {
            // Name run directory with *physical* seed radius (nm).
            // Note: --radius is interpreted in the same internal length unit as P.dx (typically nm),
            // so convert via unit_to_m_run to get a consistent physical nm label.
            double R_phys_nm = P.ic_phi_seed_radius * unit_to_m_run * 1.0e9;
            snprintf(run_dir_name, sizeof(run_dir_name),
                     "%s_T%.0f_cuda_%dx%dx%d_dt%.3g_steps%d_r%.3fnm_xB%.3f",
                     out_prefix, P.temperature_C,
                     P.Nx, P.Ny, P.Nz, P.dt, P.nsteps,
                     R_phys_nm, P.ic_23d_xB_out);
        } else {
            snprintf(run_dir_name, sizeof(run_dir_name),
                     "%s_T%.0f_cuda_%dx%dx%d_dt%.3g_steps%d_xB%.3f",
                     out_prefix, P.temperature_C,
                     P.Nx, P.Ny, P.Nz, P.dt, P.nsteps,
                     P.ic_23d_xB_out);
        }
        snprintf(output_dir, sizeof(output_dir), "%s/%s", results_root, run_dir_name);
        mkdir(output_dir, 0755);
    }
    output_pf_input_file[0] = '\0';
    if (P.mode == 0 && !P.minimize_continue_from_vtk) {
        snprintf(output_pf_input_file, sizeof(output_pf_input_file), "%s/pf_input.params", output_dir);
        if (effective_pf_param_file[0] != '\0') {
            if (!copy_text_file(effective_pf_param_file, output_pf_input_file)) {
                fprintf(stderr, "[warn] 无法将 PF 参数快照复制到结果根目录: %s -> %s\n",
                        effective_pf_param_file, output_pf_input_file);
            }
        }
    }

    // 若用户未指定 init_case_tag，则根据初始化参数自动生成一个简洁标签
    if (P.init_case_tag[0] == '\0' && P.minimize_continue_from_vtk) {
        derive_next_continue_case_tag(P.continue_phi_vtk_path, P.init_case_tag, sizeof(P.init_case_tag), P.mode);
    }
    if (P.init_case_tag[0] == '\0' && P.init_mode_raw_fields) {
        snprintf(P.init_case_tag, sizeof(P.init_case_tag), "raw_fields_relax");
    }
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
    char case_pf_input_file[4096];
    char source_case_tag[256];
    if (P.minimize_continue_from_vtk) {
        char source_case_output_dir[4096];
        source_case_output_dir[0] = '\0';
        if (derive_continue_case_output_dir(P.continue_phi_vtk_path,
                                            source_case_output_dir,
                                            sizeof(source_case_output_dir))) {
            path_basename_copy(source_case_output_dir, source_case_tag, sizeof(source_case_tag));
        } else {
            snprintf(source_case_tag, sizeof(source_case_tag), "%s", P.init_case_tag);
        }
        // For continue runs, keep a dedicated subdirectory under the resolved output root
        // so that dynamic outputs do not pollute the original discrete-case directory.
        snprintf(case_output_dir, sizeof(case_output_dir), "%s/%s", output_dir, P.init_case_tag);
        mkdir(case_output_dir, 0755);
    } else {
        snprintf(case_output_dir, sizeof(case_output_dir), "%s/%s", output_dir, P.init_case_tag);
        mkdir(case_output_dir, 0755);
        snprintf(source_case_tag, sizeof(source_case_tag), "%s", P.init_case_tag);
    }
    snprintf(case_pf_input_file, sizeof(case_pf_input_file), "%s/pf_input.params", case_output_dir);
    if (effective_pf_param_file[0] != '\0') {
        if (!copy_text_file(effective_pf_param_file, case_pf_input_file)) {
            fprintf(stderr, "[warn] 无法将 PF 参数快照复制到结果目录: %s -> %s\n",
                    effective_pf_param_file, case_pf_input_file);
        }
    }

    // 统一用于 VTK 文件名后缀（优先 init_case_tag；若为空则回退 case_<init_test_id>）
    char vtk_case_tag_buf[128];
    const char *vtk_case_tag = P.init_case_tag;
    if (vtk_case_tag[0] == '\0') {
        snprintf(vtk_case_tag_buf, sizeof(vtk_case_tag_buf), "case_%d", P.init_test_id);
        vtk_case_tag = vtk_case_tag_buf;
    }
    const char *csv_case_tag = P.minimize_continue_from_vtk ? source_case_tag : vtk_case_tag;
    g_dynamic_outputs_use_case_dir = (P.mode == 0 && (P.minimize_continue_from_vtk || P.init_mode_raw_fields || P.scheduled_nuc_enabled)) ? 1 : 0;

    log_section_header("Output Layout");
    log_kv_text("output_root", "%s", output_dir);
    if (effective_pf_param_file[0] != '\0' && output_pf_input_file[0] != '\0') {
        log_kv_text("output_pf_input", "%s", output_pf_input_file);
    }
    log_kv_text("init_case_tag", "%s", P.init_case_tag);
    log_kv_text("case_output_dir", "%s", case_output_dir);
    if (effective_pf_param_file[0] != '\0') {
        log_kv_text("case_pf_input", "%s", case_pf_input_file);
    }
    if (is_gp_zone_mode(&P) && P.gp_kinetic_ref_enabled) {
        GPEtaKineticRefDiag gp_eta_kin_diag = compute_gp_eta_kinetic_ref_diag_host(&P);
        if (gp_L_eta_mode_is_sto_S218b_override(P.gp_L_eta_mode)) {
            if (!(gp_eta_kin_diag.M_eta_used > 0.0) || !(gp_eta_kin_diag.Mcrit_eta > 0.0) ||
                !(gp_eta_kin_diag.L_eta_full_code > 0.0)) {
                fprintf(stderr,
                        "[fatal] gp_L_eta_mode=sto_S218b_override requires gp_M_eta_phys or gp_M_eta_ratio_to_crit and valid STO-SM kinetic reference inputs.\n");
                return 2;
            }
            P.gp_L_eta = gp_eta_kin_diag.L_eta_full_code;
            gp_eta_kin_diag = compute_gp_eta_kinetic_ref_diag_host(&P);
        }
        print_gp_eta_kinetic_ref_diag(&gp_eta_kin_diag);
        write_gp_eta_kinetic_ref_diag_csv(case_output_dir, &gp_eta_kin_diag);
        write_gp_eta_S218b_kinetic_diag_csv(case_output_dir, &gp_eta_kin_diag);
    }
    if (!print_unit_conversion_summary_table(&P)) {
        return 2;
    }
    if (!write_unit_conversion_summary_csv(case_output_dir, &P)) {
        return 2;
    }
        log_kv_text("vtk_mode", "%s",
                    (P.mode == 0)
                        ? ((P.minimize_continue_from_vtk || P.init_mode_raw_fields)
                               ? "dynamic/raw-continue (with case suffix)"
                               : "dynamic (no case suffix)")
                        : "minimize (with case suffix)");
    if (P.mode != 0) {
        log_kv_text("vtk_filename_suffix", "%s", vtk_case_tag);
    }
    if (P.minimize_continue_from_vtk) {
        log_kv_text("continue_phi_vtk", "%s", P.continue_phi_vtk_path);
        if (P.continue_xB_vtk_path[0] != '\0') {
            log_kv_text("continue_xB_vtk", "%s", P.continue_xB_vtk_path);
        }
    }
    
    // 创建CSV文件：dynamics 用 vf_precip_vs_time.csv，minimize 用 energy_minimize.csv
    FILE *csv_fp = NULL;
    FILE *relax_fp = NULL;
    FILE *mass_diag_fp = NULL;
    FILE *phi_eta_step_delta_fp = NULL;
    FILE *phi_eta_rhs_attribution_fp = NULL;
    FILE *eta_bulk_max_location_fp = NULL;
    FILE *energy_fp = NULL; // minimize mode energy log
    int csv_step_offset = 0;
    double csv_time_offset = 0.0;
    double csv_real_time_offset = 0.0;
    int energy_iter_offset = 0;
    if (P.mode == 0 || (P.mode == 1 && P.minimize_full_model == 1)) {
        char csv_path[4096];
        int append_existing_csv = 0;
        snprintf(csv_path, sizeof(csv_path), "%s/vf_precip_vs_time_%s.csv", case_output_dir, csv_case_tag);
        if (P.minimize_continue_from_vtk && file_exists_nonempty(csv_path)) {
            append_existing_csv = 1;
            read_last_vf_csv_state(csv_path, &csv_step_offset, &csv_time_offset, &csv_real_time_offset);
        }
        csv_fp = fopen(csv_path, append_existing_csv ? "a" : "w");
        if (!csv_fp) {
            fprintf(stderr, "[warn] 无法创建CSV文件 %s\n", csv_path);
        } else if (!append_existing_csv) {
            if (P.diag_elastic_bulk_penalty_enabled) {
                fprintf(csv_fp, "step,t_code,t_real_s,vf_precip,R_avg,sum_h,sum_gel,E_el_bulk_hat,E_el_bulk_Jm3,Delta_mu_el_Jmol\n");
            } else {
                fprintf(csv_fp, "step,t_code,t_real_s,vf_precip,R_avg\n");
            }
            fflush(csv_fp);
        }
    }
    if ((P.init_mode_raw_fields || P.scheduled_nuc_enabled) && P.mode == 0) {
        char relax_csv_path[4096];
        snprintf(relax_csv_path, sizeof(relax_csv_path), "%s/relaxation_diagnostics.csv", case_output_dir);
        relax_fp = fopen(relax_csv_path, "w");
        if (!relax_fp) {
            fprintf(stderr, "[warn] cannot open relaxation diagnostics csv %s\n", relax_csv_path);
        } else {
            fprintf(relax_fp,
                    "step,time,t_real_s,mean_phi,mean_hphi,mean_xB,%s,"
                    "initial_%s,delta_%s,relative_drift_vs_initial,"
                    "xB_min,xB_max,phi_min,phi_max\n",
                    mean_xBtot_label(&P),
                    mean_xBtot_label(&P),
                    mean_xBtot_label(&P));
            fflush(relax_fp);
            log_kv_text("relaxation_diagnostics_csv", "%s", relax_csv_path);
        }
    }
    
    int NzC = P.Nz / 2 + 1;
    int total_r = P.Nx * P.Ny * P.Nz;
    int total_k = P.Nx * P.Ny * NzC;
    size_t size_r = total_r * sizeof(double);
    size_t size_k = total_k * sizeof(cufftDoubleComplex);
    if (P.mode == 0 && P.dynamics_mass_diag_enabled) {
        char mass_diag_csv_path[4096];
        snprintf(mass_diag_csv_path, sizeof(mass_diag_csv_path), "%s/dynamics_mass_diagnostics.csv", case_output_dir);
        mass_diag_fp = fopen(mass_diag_csv_path, "w");
        if (!mass_diag_fp) {
            fprintf(stderr, "[warn] cannot open dynamics mass diagnostics csv %s\n", mass_diag_csv_path);
        } else {
            write_mass_diag_csv_header(mass_diag_fp, &P);
            fflush(mass_diag_fp);
            log_kv_text("dynamics_mass_diagnostics_csv", "%s", mass_diag_csv_path);
        }
    }
    if (P.mode == 0 && P.phi_eta_step_delta_diag_enabled) {
        char phi_eta_diag_csv_path[4096];
        snprintf(phi_eta_diag_csv_path, sizeof(phi_eta_diag_csv_path), "%s/%s_per_step.csv",
                 case_output_dir, P.phi_eta_step_delta_diag_prefix);
        phi_eta_step_delta_fp = fopen(phi_eta_diag_csv_path, "w");
        if (!phi_eta_step_delta_fp) {
            fprintf(stderr, "[warn] cannot open phi/eta per-step diagnostics csv %s\n",
                    phi_eta_diag_csv_path);
        } else {
            write_phi_eta_step_delta_csv_header(phi_eta_step_delta_fp);
            fflush(phi_eta_step_delta_fp);
            log_kv_text("phi_eta_step_delta_csv", "%s", phi_eta_diag_csv_path);
        }
    }
    if (P.mode == 0 && P.phi_eta_rhs_attribution_diag_enabled) {
        char phi_eta_rhs_attr_csv_path[4096];
        snprintf(phi_eta_rhs_attr_csv_path, sizeof(phi_eta_rhs_attr_csv_path), "%s/%s_per_step.csv",
                 case_output_dir, P.phi_eta_rhs_attribution_diag_prefix);
        phi_eta_rhs_attribution_fp = fopen(phi_eta_rhs_attr_csv_path, "w");
        if (!phi_eta_rhs_attribution_fp) {
            fprintf(stderr, "[warn] cannot open phi/eta RHS attribution csv %s\n",
                    phi_eta_rhs_attr_csv_path);
        } else {
            write_phi_eta_rhs_attribution_csv_header(phi_eta_rhs_attribution_fp);
            fflush(phi_eta_rhs_attribution_fp);
            log_kv_text("phi_eta_rhs_attribution_csv", "%s", phi_eta_rhs_attr_csv_path);
        }
        char eta_bulk_max_csv_path[4096];
        snprintf(eta_bulk_max_csv_path, sizeof(eta_bulk_max_csv_path), "%s/%s_eta_bulk_max_location.csv",
                 case_output_dir, P.phi_eta_rhs_attribution_diag_prefix);
        eta_bulk_max_location_fp = fopen(eta_bulk_max_csv_path, "w");
        if (!eta_bulk_max_location_fp) {
            fprintf(stderr, "[warn] cannot open eta bulk max-location csv %s\n",
                    eta_bulk_max_csv_path);
        } else {
            write_eta_bulk_max_location_csv_header(eta_bulk_max_location_fp);
            fflush(eta_bulk_max_location_fp);
            log_kv_text("eta_bulk_max_location_csv", "%s", eta_bulk_max_csv_path);
        }
    }

    ScheduledNucRuntime scheduled_runtime;
    scheduled_runtime.events_csv = NULL;
    scheduled_runtime.event_counter = 0;
    GpNucRuntime gp_nuc_runtime;
    gp_nuc_runtime.events_csv = NULL;
    gp_nuc_runtime.event_counter = 0;
    GpToBetaRuntime gp_to_beta_runtime;
    gp_to_beta_runtime.events_csv = NULL;
    gp_to_beta_runtime.audit_csv = NULL;
    gp_to_beta_runtime.event_counter = 0;
    gp_to_beta_runtime.audit_prefix[0] = '\0';
    gp_to_beta_runtime.count_events_rejected_capacity = 0;
    gp_to_beta_runtime.count_events_rejected_cooldown = 0;
    gp_to_beta_runtime.count_events_rejected_spacing = 0;
    gp_to_beta_runtime.count_events_rejected_window = 0;
    gp_to_beta_runtime.count_events_rejected_global = 0;
    gp_to_beta_runtime.count_events_scaled_capacity = 0;
    gp_to_beta_runtime.count_events_accepted = 0;
    gp_to_beta_runtime.min_capacity_ratio = INFINITY;
    gp_to_beta_runtime.sum_capacity_ratio = 0.0;
    gp_to_beta_runtime.capacity_ratio_count = 0;
    gp_to_beta_runtime.last_accepted_step = -std::numeric_limits<int>::max();
    gp_to_beta_runtime.last_accepted_i = -1;
    gp_to_beta_runtime.last_accepted_j = -1;
    gp_to_beta_runtime.last_accepted_k = -1;
    gp_to_beta_runtime.accepted_events.clear();
    PostConversionYAuditRuntime post_conversion_audit_runtime;
    memset(&post_conversion_audit_runtime, 0, sizeof(post_conversion_audit_runtime));
    YUpdateK0AuditRuntime y_update_k0_audit_runtime;
    memset(&y_update_k0_audit_runtime, 0, sizeof(y_update_k0_audit_runtime));
    FILE *y_update_mass_projection_fp = NULL;
    if (P.scheduled_nuc_enabled) {
        if (!parse_scheduled_events(&P, &scheduled_runtime.events)) {
            return 2;
        }
        if (!load_scheduled_profile_csv(&P, &scheduled_runtime.profile)) {
            return 2;
        }
        char scheduled_csv_path[4096];
        snprintf(scheduled_csv_path, sizeof(scheduled_csv_path), "%s/scheduled_nucleation_events.csv", case_output_dir);
        scheduled_runtime.events_csv = fopen(scheduled_csv_path, "w");
        if (!scheduled_runtime.events_csv) {
            fprintf(stderr, "[fatal] cannot open scheduled nucleation events CSV: %s\n", scheduled_csv_path);
            return 2;
        }
        fprintf(scheduled_runtime.events_csv,
                "event_id,step,source_case_label,center_x_nm,center_y_nm,center_z_nm,xB_edge,"
                "M_before_event,M_after_embed_before_comp,M_after_comp,event_mass_error,relative_event_mass_error,"
                "C_local,local_comp_weight_active_fraction,xB_min_before,xB_max_before,xB_min_after,xB_max_after,"
                "phi_max_after,mean_hphi_after,clip_lower_fraction,clip_upper_fraction,profile_queries_out_of_range_fraction,overlap_warning\n");
        fflush(scheduled_runtime.events_csv);
        log_kv_text("scheduled_nucleation_events_csv", "%s", scheduled_csv_path);
    }
    if (is_gp_zone_mode(&P) && P.gp_nuc_enabled) {
        char gp_nuc_csv_path[4096];
        snprintf(gp_nuc_csv_path, sizeof(gp_nuc_csv_path), "%s/gp_nucleation_events.csv", case_output_dir);
        gp_nuc_runtime.events_csv = fopen(gp_nuc_csv_path, "w");
        if (!gp_nuc_runtime.events_csv) {
            fprintf(stderr, "[fatal] cannot open gp nucleation events CSV: %s\n", gp_nuc_csv_path);
            return 2;
        }
        fprintf(gp_nuc_runtime.events_csv,
                "step,event_id,center_i,center_j,center_k,center_x,center_y,center_z,local_xB_alpha,local_xBtot_gp,delta_g_nuc,drive,deltaG_star_J,J_rate,P_event,"
                "mass_before,mass_after_raw,mass_after_comp,compensation_success,xB_clip_count_event,seed_scale\n");
        fflush(gp_nuc_runtime.events_csv);
        log_kv_text("gp_nucleation_events_csv", "%s", gp_nuc_csv_path);
    }
    if (is_gp_zone_mode(&P) && P.gp_to_beta_enabled) {
        char gp_to_beta_csv_path[4096];
        snprintf(gp_to_beta_csv_path, sizeof(gp_to_beta_csv_path), "%s/gp_to_beta_events.csv", case_output_dir);
        gp_to_beta_runtime.events_csv = fopen(gp_to_beta_csv_path, "w");
        if (!gp_to_beta_runtime.events_csv) {
            fprintf(stderr, "[fatal] cannot open gp_to_beta events CSV: %s\n", gp_to_beta_csv_path);
            return 2;
        }
        fprintf(gp_to_beta_runtime.events_csv,
                "step,event_id,center_i,center_j,center_k,center_x,center_y,center_z,eta_max,R_eff_GP,local_xB_alpha,local_xBtot_gp,"
                "phi_seed_radius,patch_radius,mass_before,mass_after_raw,mass_after_comp,mass_error_raw,mass_error_comp,stochastic_enabled,event_accepted,"
                "drive_beta_given_GP,deltaG_star,J_site,P_event,random_u,compensation_success,xB_clip_count_event,"
                "feasibility_checked,mass_error_raw_predicted,shell_capacity_add,shell_capacity_remove,capacity_ratio,"
                "event_rejected_infeasible,seed_amplitude_original,seed_amplitude_scaled,seed_scaled_due_to_capacity,"
                "cooldown_or_spacing_rejected,xB_shell_min_before_event,xB_shell_max_before_event,"
                "throttling_checked,rejected_cooldown,rejected_spacing,rejected_window_limit,rejected_global_limit,"
                "nearest_event_distance,steps_since_nearest_event,accepted_events_in_window,accepted_events_total\n");
        fflush(gp_to_beta_runtime.events_csv);
        log_kv_text("gp_to_beta_events_csv", "%s", gp_to_beta_csv_path);
        snprintf(gp_to_beta_runtime.audit_prefix, sizeof(gp_to_beta_runtime.audit_prefix), "%s",
                 P.gp_to_beta_conversion_audit_prefix);
        if (P.gp_to_beta_conversion_mass_audit_enabled) {
            char gp_to_beta_audit_csv_path[4096];
            snprintf(gp_to_beta_audit_csv_path, sizeof(gp_to_beta_audit_csv_path),
                     "%s/%s.csv", case_output_dir, gp_to_beta_runtime.audit_prefix);
            gp_to_beta_runtime.audit_csv = fopen(gp_to_beta_audit_csv_path, "w");
            if (!gp_to_beta_runtime.audit_csv) {
                fprintf(stderr, "[fatal] cannot open gp_to_beta conversion audit CSV: %s\n",
                        gp_to_beta_audit_csv_path);
                return 2;
            }
            fprintf(gp_to_beta_runtime.audit_csv,
                    "step,stage,event_id,center_i,center_j,center_k,mean_xBtot,sum_xBtot,"
                    "delta_sum_xBtot_from_previous_stage,delta_sum_xBtot_from_before_conversion,"
                    "mean_xB,min_xB,max_xB,mean_phi,min_phi,max_phi,mean_eta,min_eta,max_eta,"
                    "mean_h_gp,mean_h_beta,patch_mass_before,patch_mass_after,delta_patch_mass,"
                    "compensation_mass,clipped_mass_loss,num_clipped_low,num_clipped_high,"
                    "storage_residual_mean,storage_residual_max_abs,storage_residual_sum,notes\n");
            fflush(gp_to_beta_runtime.audit_csv);
            log_kv_text("gp_to_beta_conversion_audit_csv", "%s", gp_to_beta_audit_csv_path);
        }
        if (P.post_conversion_y_update_audit_enabled) {
            char post_conv_audit_csv_path[4096];
            snprintf(post_conv_audit_csv_path, sizeof(post_conv_audit_csv_path),
                     "%s/%s.csv", case_output_dir, P.post_conversion_y_update_audit_prefix);
            post_conversion_audit_runtime.csv = fopen(post_conv_audit_csv_path, "w");
            if (!post_conversion_audit_runtime.csv) {
                fprintf(stderr, "[fatal] cannot open post-conversion Y audit CSV: %s\n",
                        post_conv_audit_csv_path);
                return 2;
            }
            fprintf(post_conversion_audit_runtime.csv,
                    "step,stage,sum_xBtot,delta_sum_xBtot_from_previous_stage,"
                    "delta_sum_xBtot_from_post_conversion_baseline,mean_xBtot,mean_xB,min_xB,max_xB,mean_Y,"
                    "Y_k0_before,Y_k0_after,delta_Y_k0,Y_update_k0_drift,Y_update_nonzero_mode_drift,"
                    "clipped_mass_loss,num_clipped_low,num_clipped_high,storage_residual_sum,storage_residual_max_abs\n");
            fflush(post_conversion_audit_runtime.csv);
            snprintf(post_conversion_audit_runtime.prefix, sizeof(post_conversion_audit_runtime.prefix), "%s",
                     P.post_conversion_y_update_audit_prefix);
            log_kv_text("post_conversion_y_update_audit_csv", "%s", post_conv_audit_csv_path);
        }
        if (P.y_update_k0_audit_enabled) {
            char y_update_k0_audit_csv_path[4096];
            snprintf(y_update_k0_audit_csv_path, sizeof(y_update_k0_audit_csv_path),
                     "%s/%s.csv", case_output_dir, P.y_update_k0_audit_prefix);
            y_update_k0_audit_runtime.csv = fopen(y_update_k0_audit_csv_path, "w");
            if (!y_update_k0_audit_runtime.csv) {
                fprintf(stderr, "[fatal] cannot open Y-update k0 audit CSV: %s\n",
                        y_update_k0_audit_csv_path);
                return 2;
            }
            fprintf(y_update_k0_audit_runtime.csv,
                    "step,audit_step_index,stage,post_conversion_step,"
                    "sum_xBtot_before_Y,sum_xBtot_after_Y,delta_sum_xBtot_Y_update,target_sum_xBtot,"
                    "Y_k0_before,Y_k0_after,delta_Y_k0,"
                    "RHS_k0_total,RHS_k0_linear,RHS_k0_nonlinear,RHS_k0_stabilization_add,RHS_k0_stabilization_subtract,"
                    "RHS_k0_source,RHS_k0_transport,RHS_k0_unknown,"
                    "mean_Y_before,mean_Y_after,mean_xB_before,mean_xB_after,"
                    "min_xB_before,max_xB_before,min_xB_after,max_xB_after,"
                    "mean_phi,mean_eta,mean_h_gp,mean_h_beta,"
                    "storage_residual_sum_before,storage_residual_sum_after,"
                    "storage_residual_max_abs_before,storage_residual_max_abs_after,notes\n");
            fflush(y_update_k0_audit_runtime.csv);
            snprintf(y_update_k0_audit_runtime.prefix, sizeof(y_update_k0_audit_runtime.prefix), "%s",
                     P.y_update_k0_audit_prefix);
            log_kv_text("y_update_k0_audit_csv", "%s", y_update_k0_audit_csv_path);
        }
        if (P.y_update_mass_projection_enabled &&
            (P.y_update_mass_projection_report_enabled || P.y_update_k0_audit_enabled)) {
            char y_update_mass_projection_csv_path[4096];
            snprintf(y_update_mass_projection_csv_path, sizeof(y_update_mass_projection_csv_path),
                     "%s/y_update_mass_projection.csv", case_output_dir);
            y_update_mass_projection_fp = fopen(y_update_mass_projection_csv_path, "w");
            if (!y_update_mass_projection_fp) {
                fprintf(stderr, "[fatal] cannot open Y-update mass projection CSV: %s\n",
                        y_update_mass_projection_csv_path);
                return 2;
            }
            fprintf(y_update_mass_projection_fp,
                    "step,post_conversion_step,target_mode,target_sum_xBtot,"
                    "sum_xBtot_before_projection,sum_xBtot_after_projection,"
                    "delta_before_projection,delta_after_projection,"
                    "lambda_shift,num_iter,converged,projection_residual,"
                    "min_xB_before_projection,max_xB_before_projection,"
                    "min_xB_after_projection,max_xB_after_projection,"
                    "num_clipped_low_after_projection,num_clipped_high_after_projection,notes\n");
            fflush(y_update_mass_projection_fp);
            log_kv_text("y_update_mass_projection_csv", "%s", y_update_mass_projection_csv_path);
        }
    }
    
    double temperature_K = P.temperature_C + 273.15;
    double invN = 1.0 / (double)total_r;
    // baseline for bulk chemical free-energy (stoichiometric compound PF)
    double xB_ref = NAN;
    double g_bulk0_hat = NAN;
    double F_el0_hat = 0.0;  // elastic baseline: 默认 0，可选后续测一次 phi=0,xB=xB_ref
    double muB_farfield_hat = NAN; // CNT far-field chemical potential (hat)
    
    // 分配CPU内存（用于初始化和输出）
    double *h_phi_r = (double*)malloc(size_r);
    double *h_eta_r = (double*)malloc(size_r);
    double *h_Y_r = (double*)malloc(size_r);
    double *h_xB_r = (double*)malloc(size_r);
    double *h_xBtot_r = (double*)malloc(size_r);
    
    // 优化：不再分配d_diag_stats，使用直接归约函数节省显存
    
    // 初始化场
    if (P.init_mode_raw_fields) {
        log_section_header("Initialization");
        log_kv_text("path", "%s", "raw_fields phi/xB from Python embedding");
        fflush(stdout);
        if (!load_raw_init_fields(h_phi_r, h_eta_r, h_Y_r, h_xB_r, h_xBtot_r, &P, total_r, &raw_init_meta, 1)) {
            return 2;
        }
    } else if (P.minimize_continue_from_vtk) {
        log_section_header("Initialization");
        if ((P.mode == 0 || P.minimize_full_model) && P.continue_xB_vtk_path[0] != '\0') {
            log_kv_text("path", "%s", "continue from phi/xB VTK");
        } else if (P.mode == 0 || P.minimize_full_model) {
            log_kv_text("path", "%s", "continue from phi VTK + rebuilt xB/Y");
        } else {
            log_kv_text("path", "%s", "continue from phi VTK");
        }
        fflush(stdout);
        if (!load_continue_fields_from_vtk(h_phi_r, h_Y_r, h_xB_r, h_xBtot_r, &P, total_r)) {
            return 2;
        }
        memset(h_eta_r, 0, size_r);
    } else if (P.mode == 1 && P.minimize_full_model == 0) {
        // 旧行为：phi-only minimize，不初始化xB/Y
        log_section_header("Initialization");
        log_kv_text("path", "phi-only minimize");
        fflush(stdout);
        initialize_phi_only_cuda(h_phi_r, &P, total_r);
        memset(h_eta_r, 0, size_r);
        memset(h_Y_r, 0, size_r);
        memset(h_xB_r, 0, size_r);
        memset(h_xBtot_r, 0, size_r);
    } else if (P.scheduled_nuc_enabled) {
        log_section_header("Initialization");
        log_kv_text("path", "scheduled nucleation test uniform no-nucleus field");
        double xb0 = (P.ic_23d_xB_out > 0.0) ? P.ic_23d_xB_out : 0.030;
        xb0 = fmin(P.scheduled_nuc_xB_max, fmax(P.scheduled_nuc_xB_min, xb0));
        for (int i = 0; i < total_r; ++i) {
            h_phi_r[i] = 0.0;
            h_eta_r[i] = 0.0;
            h_xB_r[i] = xb0;
            h_Y_r[i] = logit_from_fraction(xb0, P.xB_eps, P.Y_clip);
            h_xBtot_r[i] = xb0;
        }
        log_kv_text("uniform_phi", "%.6f", 0.0);
        log_kv_text("uniform_xB", "%.8e", xb0);
    } else {
        // dynamics 与 full-model minimize 共用：初始化 phi/Y/xB
        log_section_header("Initialization");
        log_kv_text("path", "mass-conserving phi/xB/Y");
        fflush(stdout);
        initialize_fields_cuda(h_phi_r, h_Y_r, h_xB_r, &P, total_r);
        memset(h_eta_r, 0, size_r);
        for (int i = 0; i < total_r; i++) {
            double phi = h_phi_r[i];
            double h = h_of_phi(phi);
            double xB = clamp01(h_xB_r[i]);
            h_xBtot_r[i] = (1.0 - h) * xB + P.v_B * h;
        }
    }

    if (is_gp_zone_mode(&P)) {
        if (!apply_gp_eta_initialization_host(h_phi_r, h_eta_r, h_Y_r, h_xB_r, h_xBtot_r, &P, total_r)) {
            return 2;
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
        int append_existing_energy = 0;
        snprintf(epath, sizeof(epath), "%s/energy_minimize_T%.0f_r%.2f_xB%.3f_%s.csv",
                 case_output_dir, P.temperature_C, P.ic_phi_seed_radius, P.ic_23d_xB_out, csv_case_tag);
        if (P.minimize_continue_from_vtk && file_exists_nonempty(epath)) {
            append_existing_energy = 1;
            read_last_energy_iter(epath, &energy_iter_offset);
        }
        energy_fp = fopen(epath, append_existing_energy ? "a" : "w");
        if (!energy_fp) {
            fprintf(stderr, "[warn] cannot open energy csv %s\n", epath);
        } else if (!append_existing_energy) {
            // Minimization energy log (excess-free-energy + CNT-based summaries):
            // iter,dt,mean_h,V0,lambda,
            // F_surf_hat,F_el_hat,F_chem_excess_hat,F_total_excess_hat,
            // F_chem_CNT_hat,F_total_CNT_hat,
            // total_interface_sum,total_el_core_sum,
            // rms_res,rms_dphi,rms_dY,rel_dF,vol_err_rel,post_proj_iters_used
            fprintf(energy_fp,
                    "iter,dt,mean_h,V0,lambda,"
                    "F_surf_hat,F_el_hat,F_chem_excess_hat,F_total_excess_hat,"
                    "F_chem_CNT_hat,F_total_CNT_hat,"
                    "total_interface_sum,total_el_core_sum,"
                    "rms_res,rms_dphi,rms_dY,rel_dF,vol_err_rel,post_proj_iters_used\n");
            fflush(energy_fp);
        }
    }

    // 输出t=0的VTK文件（先将CPU数据复制到临时GPU数组）
    char filename[4096];  // 用于VTK输出文件名
    double *d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, size_r));

    if (P.mode == 0) {
        recompute_host_xBtot_field(h_phi_r, h_eta_r, h_xB_r, h_xBtot_r, &P, total_r);
        CUDA_CHECK(cudaMemcpy(d_temp, h_xBtot_r, size_r, cudaMemcpyHostToDevice));
        build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir,
                            xBtot_output_stem(&P), VTK_NAME_STEP, 0, vtk_case_tag, 0);
        int ret1 = write_vtk_cuda(d_temp, P.Nx, P.Ny, P.Nz, xBtot_vtk_field_name(&P), 0, filename);
        if (!ret1) {
            fprintf(stderr, "ERROR: Failed to write initial %s VTK file: %s\n",
                    xBtot_output_stem(&P), filename);
        }
        if (is_gp_zone_mode(&P)) {
            build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir,
                                "xBtot_gp", VTK_NAME_INIT, 0, vtk_case_tag, 0);
            int ret_gp_init = write_vtk_cuda(d_temp, P.Nx, P.Ny, P.Nz, "xBtot_gp", 0, filename);
            if (!ret_gp_init) {
                fprintf(stderr, "ERROR: Failed to write initial xBtot_gp_init VTK file: %s\n", filename);
            }
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
    if (is_gp_zone_mode(&P)) {
        CUDA_CHECK(cudaMemcpy(d_temp, h_eta_r, size_r, cudaMemcpyHostToDevice));
        build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "eta", VTK_NAME_INIT, 0, vtk_case_tag, P.mode);
        int ret_eta_init = write_vtk_cuda(d_temp, P.Nx, P.Ny, P.Nz, "eta", 0, filename);
        if (!ret_eta_init) {
            fprintf(stderr, "ERROR: Failed to write initial eta VTK file: %s\n", filename);
        }
    }

    if (P.mode == 0) {
        CUDA_CHECK(cudaMemcpy(d_temp, h_xB_r, size_r, cudaMemcpyHostToDevice));
        build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "xB", VTK_NAME_STEP, 0, vtk_case_tag, 0);
        int ret3 = write_vtk_cuda(d_temp, P.Nx, P.Ny, P.Nz, "xB", 0, filename);
        if (!ret3) {
            fprintf(stderr, "ERROR: Failed to write initial xB VTK file: %s\n", filename);
        }
    } else if (P.mode == 1 && P.minimize_full_model == 1) {
        CUDA_CHECK(cudaMemcpy(d_temp, h_xB_r, size_r, cudaMemcpyHostToDevice));
        build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "xB", VTK_NAME_INIT, 0, vtk_case_tag, P.mode);
        int ret3 = write_vtk_cuda(d_temp, P.Nx, P.Ny, P.Nz, "xB", 0, filename);
        if (!ret3) {
            fprintf(stderr, "ERROR: Failed to write initial xB VTK file: %s\n", filename);
        } else {
            printf("写出初始化 xB vtk: %s\n", filename);
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
            (phi_elastic_coupling_enabled ? P.elastic_shift_dimless : 0.0),
            total_r,
            phi_elastic_coupling_enabled);
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
    double *d_phi_r, *d_eta_r, *d_Y_r, *d_xB_r;
    double *d_phi_rhs_r, *d_eta_prev_r, *d_eta_rhs_r, *d_phi_n_saved, *d_Y_n_saved, *d_dY_dt_prev_r;
    double *d_dY_dt_picard_r, *d_dY_dt_picard_old_r;
    double *d_Y_projection_base_r = NULL;
    // 优化：移除d_xBtot_r，仅在需要输出VTK时临时计算
    
    // 工作空间（Y方程相关）
    double *d_mu_x_r;
    // Optimization: d_xB_prev_r 和 d_divJ_r 复用同一内存
    double *d_divJ_r, *d_xB_prev_r;
    double *d_xB_gp_old_r = NULL;
    double *d_xB_old_diag_r = NULL;
    // Optimization: d_phi_rhs_r 在步骤1完成后可复用为 d_lapY_r
    double *d_lapY_r, *d_Y_rhs_r;

    // k空间数组
    // Optimization: phi_k 与 Y_k 使用 in-place 更新，不再常驻 phi_k_new、Y_k_new
    cufftDoubleComplex *d_phi_k, *d_eta_k, *d_eta_rhs_k, *d_phi_rhs_k;
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
    CUDA_CHECK(cudaMalloc(&d_eta_r, size_r));
    CUDA_CHECK(cudaMalloc(&d_Y_r, size_r));
    CUDA_CHECK(cudaMalloc(&d_xB_r, size_r));
    CUDA_CHECK(cudaMalloc(&d_phi_rhs_r, size_r));
    CUDA_CHECK(cudaMalloc(&d_eta_prev_r, size_r));
    CUDA_CHECK(cudaMalloc(&d_eta_rhs_r, size_r));
    CUDA_CHECK(cudaMalloc(&d_phi_n_saved, size_r));
    CUDA_CHECK(cudaMalloc(&d_Y_n_saved, size_r));
    CUDA_CHECK(cudaMalloc(&d_dY_dt_prev_r, size_r));
    CUDA_CHECK(cudaMalloc(&d_dY_dt_picard_r, size_r));
    CUDA_CHECK(cudaMalloc(&d_dY_dt_picard_old_r, size_r));
    if (P.y_update_mass_projection_enabled || P.y_update_k0_audit_enabled) {
        CUDA_CHECK(cudaMalloc(&d_Y_projection_base_r, size_r));
    }
    // 优化：不再分配d_xBtot_r，节省1GB显存
    // 优化：d_phi_rhs_r 在步骤1完成后可复用为 d_lapY_r（节省1GB）
    d_lapY_r = d_phi_rhs_r;  // 复用指针
    
    CUDA_CHECK(cudaMalloc(&d_mu_x_r, size_r));
    // Optimization: d_Y_rhs_r 复用 d_mu_x_r（Y_rhs 在 mu_x 完全使用后写入）
    d_Y_rhs_r = d_mu_x_r;
    // Optimization: 移除 grad_mu_x/y/z_r 与 Jx/Jy/Jz_r/k 常驻，改用 scratch + divJ 串行累加
    CUDA_CHECK(cudaMalloc(&d_divJ_r, size_r));
    d_xB_prev_r = d_divJ_r;
    CUDA_CHECK(cudaMalloc(&d_xB_gp_old_r, size_r));
    if (P.dynamics_mass_diag_enabled && P.mode == 0) {
        CUDA_CHECK(cudaMalloc(&d_xB_old_diag_r, size_r));
    }
    // 优化：不再分配d_DY_values，节省1GB显存
    
    CUDA_CHECK(cudaMalloc(&d_phi_k, size_k));
    CUDA_CHECK(cudaMalloc(&d_eta_k, size_k));
    CUDA_CHECK(cudaMalloc(&d_eta_rhs_k, size_k));
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
    double *d_mass_diag_phi_stats = NULL;
    double *d_mass_diag_Y_stats = NULL;
    double *d_mass_diag_Y_rhs_stats = NULL;
    double *d_mass_diag_gp_storage_stats = NULL;
    double *d_gp_Y_update_stats = NULL;
    double *d_gp_eta_feas_stats = NULL;
    double *d_gp_transport_stats = NULL;
    double *d_gp_elastic_stats = NULL;
    double *d_picard_diff_stats = NULL;
    double *d_phi_rhs_bulk_diag_r = NULL;
    double *d_phi_rhs_dw_diag_r = NULL;
    double *d_phi_rhs_elastic_diag_r = NULL;
    double *d_eta_rhs_bulk_diag_r = NULL;
    double *d_eta_rhs_dw_diag_r = NULL;
    double *d_eta_rhs_elastic_diag_r = NULL;
    if (P.dynamics_mass_diag_enabled && P.mode == 0) {
        CUDA_CHECK(cudaMalloc(&d_mass_diag_phi_stats, MASS_DIAG_PHI_STATS_COUNT * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_mass_diag_Y_stats, MASS_DIAG_Y_STATS_COUNT * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_mass_diag_Y_rhs_stats, MASS_DIAG_Y_RHS_STATS_COUNT * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_mass_diag_gp_storage_stats, MASS_DIAG_GP_STORAGE_STATS_COUNT * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_gp_Y_update_stats, GP_Y_UPDATE_STATS_COUNT * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_gp_eta_feas_stats, GP_ETA_FEAS_STATS_COUNT * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_gp_transport_stats, GP_TRANSPORT_STATS_COUNT * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_gp_elastic_stats, GP_ELASTIC_STATS_COUNT * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_picard_diff_stats, 2 * sizeof(double)));
    }
    if (P.mode == 0 && P.phi_eta_rhs_attribution_diag_enabled) {
        CUDA_CHECK(cudaMalloc(&d_phi_rhs_bulk_diag_r, size_r));
        CUDA_CHECK(cudaMalloc(&d_phi_rhs_dw_diag_r, size_r));
        CUDA_CHECK(cudaMalloc(&d_phi_rhs_elastic_diag_r, size_r));
        CUDA_CHECK(cudaMalloc(&d_eta_rhs_bulk_diag_r, size_r));
        CUDA_CHECK(cudaMalloc(&d_eta_rhs_dw_diag_r, size_r));
        CUDA_CHECK(cudaMalloc(&d_eta_rhs_elastic_diag_r, size_r));
    }
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
    CUDA_CHECK(cudaMemcpy(d_eta_r, h_eta_r, size_r, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y_r, h_Y_r, size_r, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_xB_r, h_xB_r, size_r, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_eta_prev_r, h_eta_r, size_r, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_eta_rhs_r, 0, size_r));
    CUDA_CHECK(cudaMemset(d_eta_k, 0, size_k));
    CUDA_CHECK(cudaMemset(d_eta_rhs_k, 0, size_k));

    // minimize mode: phi-only, no xB rebuild
    // 初始化工作数组
    CUDA_CHECK(cudaMemset(d_dY_dt_prev_r, 0, size_r));
    CUDA_CHECK(cudaMemset(d_dY_dt_picard_r, 0, size_r));
    CUDA_CHECK(cudaMemset(d_dY_dt_picard_old_r, 0, size_r));
    if (d_xB_old_diag_r) {
        CUDA_CHECK(cudaMemset(d_xB_old_diag_r, 0, size_r));
    }
    
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
        mean_xB_tot = gpu_reduce_sum_model_xBtot(&P, d_phi_r, d_eta_r, d_xB_r, total_r) / (double)total_r;
        gpu_reduce_min_max(d_xB_r, total_r, &min_xB, &max_xB);
        gpu_reduce_min_max_meff(d_phi_r, d_xB_r, total_r,
                                P.D_alpha, P.D_compound, P.Vm_alpha_0, P.dVm_alpha_dxB, P.Vm_compound,
                                temperature_K, P.mu_reference_scale, &min_Meff, &max_Meff);
    }
    double relaxation_initial_mean_xBtot = mean_xB_tot;
    DynamicsMassDiagSummary mass_diag_summary;
    memset(&mass_diag_summary, 0, sizeof(mass_diag_summary));
    mass_diag_summary.initial_mean_xBtot = mean_xB_tot;
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

    if (csv_fp && !need_delay_step0_csv && !P.minimize_continue_from_vtk) {
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
    if (relax_fp && P.mode == 0) {
        write_relaxation_diag_row(relax_fp, 0, 0.0, 0.0,
                                  d_phi_r, d_eta_r, d_xB_r, &P, total_r,
                                  relaxation_initial_mean_xBtot);
    }
    
    // 性能计时
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    GeometrySummaryRuntime geometry_summary_runtime;
    memset(&geometry_summary_runtime, 0, sizeof(geometry_summary_runtime));
    geometry_summary_runtime.enabled = env_flag_enabled("GEOMETRY_SUMMARY_EVERY_STEP");
    geometry_summary_runtime.write_each_step = env_flag_enabled("GEOMETRY_SUMMARY_WRITE_EACH_STEP");
    geometry_summary_runtime.print_each_step = env_flag_enabled("GEOMETRY_SUMMARY_PRINT_EACH_STEP");
    
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
    int stop_after_conversion_audit_done = 0;
    const double step_loop_wall_t0 = wall_time_sec_monotonic();
    double *step_wall_s = (nsteps_run > 0) ? (double *)calloc((size_t)nsteps_run, sizeof(double)) : NULL;
    const int phi_eta_step_delta_diag_enabled_runtime =
        (P.mode == 0 && P.phi_eta_step_delta_diag_enabled && phi_eta_step_delta_fp);
    const int phi_eta_rhs_attribution_diag_enabled_runtime =
        (P.mode == 0 && P.phi_eta_rhs_attribution_diag_enabled && phi_eta_rhs_attribution_fp);
    const int phi_eta_field_snapshot_runtime =
        (phi_eta_step_delta_diag_enabled_runtime || phi_eta_rhs_attribution_diag_enabled_runtime);
    std::vector<double> phi_before_step_host;
    std::vector<double> phi_after_phi_update_host;
    std::vector<double> eta_before_step_host;
    std::vector<double> eta_after_eta_update_host;
    std::vector<double> xB_before_step_host;
    std::vector<double> xB_after_step_host;
    std::vector<double> Y_before_step_host;
    std::vector<double> Y_after_step_host;
    std::vector<double> phi_rhs_total_explicit_host;
    std::vector<double> phi_rhs_bulk_host;
    std::vector<double> phi_rhs_dw_host;
    std::vector<double> phi_rhs_elastic_host;
    std::vector<double> eta_rhs_total_explicit_host;
    std::vector<double> eta_rhs_bulk_host;
    std::vector<double> eta_rhs_dw_host;
    std::vector<double> eta_rhs_elastic_host;
    if (phi_eta_field_snapshot_runtime) {
        phi_before_step_host.resize((size_t)total_r);
        phi_after_phi_update_host.resize((size_t)total_r);
        eta_before_step_host.resize((size_t)total_r);
        eta_after_eta_update_host.resize((size_t)total_r);
        xB_before_step_host.resize((size_t)total_r);
        xB_after_step_host.resize((size_t)total_r);
        Y_before_step_host.resize((size_t)total_r);
        Y_after_step_host.resize((size_t)total_r);
    }
    if (phi_eta_rhs_attribution_diag_enabled_runtime) {
        phi_rhs_total_explicit_host.resize((size_t)total_r);
        phi_rhs_bulk_host.resize((size_t)total_r);
        phi_rhs_dw_host.resize((size_t)total_r);
        phi_rhs_elastic_host.resize((size_t)total_r);
        eta_rhs_total_explicit_host.resize((size_t)total_r);
        eta_rhs_bulk_host.resize((size_t)total_r);
        eta_rhs_dw_host.resize((size_t)total_r);
        eta_rhs_elastic_host.resize((size_t)total_r);
    }

    for (int step = 1; step <= nsteps_run; step++) {
        const double step_wall_t0 = wall_time_sec_monotonic();
        steps_completed = step;
        // 保存上一时间步
        CUDA_CHECK(cudaMemcpy(d_phi_n_saved, d_phi_r, size_r, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_eta_prev_r, d_eta_r, size_r, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_Y_n_saved, d_Y_r, size_r, cudaMemcpyDeviceToDevice));
        const int do_mass_diag =
            (P.mode == 0 && P.dynamics_mass_diag_enabled &&
             (step % P.dynamics_mass_diag_interval == 0));
        const int do_phi_eta_step_delta_diag =
            (phi_eta_step_delta_diag_enabled_runtime &&
             step <= P.phi_eta_step_delta_diag_max_steps &&
             (step % P.phi_eta_step_delta_diag_every == 0));
        const int do_phi_eta_rhs_attribution_diag =
            (phi_eta_rhs_attribution_diag_enabled_runtime &&
             step <= P.phi_eta_rhs_attribution_diag_max_steps &&
             (step % P.phi_eta_rhs_attribution_diag_every == 0));
        const int do_phi_eta_field_snapshot =
            (do_phi_eta_step_delta_diag || do_phi_eta_rhs_attribution_diag);
        DynamicsMassDiagRow mass_diag_row;
        memset(&mass_diag_row, 0, sizeof(mass_diag_row));
        if (do_phi_eta_field_snapshot) {
            CUDA_CHECK(cudaMemcpy(phi_before_step_host.data(), d_phi_r, size_r, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(eta_before_step_host.data(), d_eta_r, size_r, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(xB_before_step_host.data(), d_xB_r, size_r, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(Y_before_step_host.data(), d_Y_r, size_r, cudaMemcpyDeviceToHost));
        }
        if (do_mass_diag) {
            mass_diag_row.step = step;
            mass_diag_row.time_code = step * P.dt;
            mass_diag_row.y_rhs_prev_time_level_enabled = P.enable_Y_rhs_previous_time_level ? 1.0 : 0.0;
            mass_diag_row.y_rhs_picard_enabled = P.enable_Y_rhs_picard ? 1.0 : 0.0;
            mass_diag_row.y_rhs_picard_iters = (double)P.Y_rhs_picard_iters;
            mass_diag_row.y_rhs_picard_omega = P.Y_rhs_picard_omega;
            mass_diag_row.gp_y_update_mode_code = is_gp_zone_mode(&P) ? gp_y_update_mode_code(&P) : 0.0;
            mass_diag_row.dmuC_dx_ref_numeric = NAN;
            mass_diag_row.mean_xBtot_before_step =
                gpu_reduce_sum_model_xBtot(&P, d_phi_r, d_eta_r, d_xB_r, total_r) / (double)total_r;
            if (is_gp_zone_mode(&P)) {
                mass_diag_row.mean_xBtot_gp_before_step = mass_diag_row.mean_xBtot_before_step;
            }
            mass_diag_row.mean_xB_before_phi_update =
                gpu_reduce_sum(d_xB_r, total_r) / (double)total_r;
        }
        if (post_conversion_audit_runtime.csv &&
            P.post_conversion_y_update_audit_enabled) {
            capture_post_conversion_device_state(&P, d_phi_r, d_eta_r, d_Y_r, d_xB_r,
                                                 plan_r2c_Y, d_Y_k, total_r, total_k,
                                                 &post_conversion_audit_runtime.before_step_sum_xBtot,
                                                 &post_conversion_audit_runtime.before_step_mean_xB,
                                                 &post_conversion_audit_runtime.before_step_min_xB,
                                                 &post_conversion_audit_runtime.before_step_max_xB,
                                                 &post_conversion_audit_runtime.before_step_mean_Y,
                                                 &post_conversion_audit_runtime.before_step_Y_k0_re,
                                                 &post_conversion_audit_runtime.before_step_Y_k0_im);
        }
        if ((y_update_k0_audit_runtime.active &&
             y_update_k0_audit_runtime.rows_remaining_steps > 0) ||
            y_update_k0_audit_runtime.projection_armed) {
            capture_post_conversion_device_state(&P, d_phi_r, d_eta_r, d_Y_r, d_xB_r,
                                                 plan_r2c_Y, d_Y_k, total_r, total_k,
                                                 &y_update_k0_audit_runtime.pre_Y_sum_xBtot,
                                                 &y_update_k0_audit_runtime.pre_Y_mean_xB,
                                                 &y_update_k0_audit_runtime.pre_Y_min_xB,
                                                 &y_update_k0_audit_runtime.pre_Y_max_xB,
                                                 &y_update_k0_audit_runtime.pre_Y_mean_Y,
                                                 &y_update_k0_audit_runtime.pre_Y_Y_k0_re,
                                                 &y_update_k0_audit_runtime.pre_Y_Y_k0_im);
        }

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
                    d_eta_r,
                    d_xB_r,
                    d_uxx_r, d_uyy_r, d_uzz_r,
                    d_uxy_r, d_uxz_r, d_uyz_r,
                    eps_xx00_f, eps_yy00_f, eps_zz00_f,
                    eps_yz00_f, eps_xz00_f, eps_xy00_f,
                    (double)P.eps_iso_over_vB,
                    is_gp_zone_mode(&P) ? 1 : 0,
                    gp_elastic_solver_enabled,
                    P.gp_eps_iso,
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
                    d_eta_r,
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
                    is_gp_zone_mode(&P) ? 1 : 0,
                    gp_elastic_solver_enabled,
                    P.gp_eps_iso,
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
                d_eta_r,
                (P.mode == 1 && P.minimize_full_model == 0) ? NULL : d_xB_r,
                eps_xx00_f, eps_yy00_f, eps_zz00_f,
                eps_yz00_f, eps_xz00_f, eps_xy00_f,
                (double)P.eps_iso_over_vB,
                is_gp_zone_mode(&P) ? 1 : 0,
                gp_elastic_solver_enabled,
                P.gp_eps_iso,
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

            if (do_mass_diag && gp_elastic_solver_enabled) {
                double gp_elastic_stats_init[GP_ELASTIC_STATS_COUNT];
                double gp_elastic_stats[GP_ELASTIC_STATS_COUNT] = {0.0};
                init_gp_elastic_stats_host(gp_elastic_stats_init);
                CUDA_CHECK(cudaMemcpy(d_gp_elastic_stats, gp_elastic_stats_init,
                                      GP_ELASTIC_STATS_COUNT * sizeof(double),
                                      cudaMemcpyHostToDevice));
                double *d_gel_diag_tmp = (double *)d_scratch_r_double;
                launch_compute_gel_density_kernel(
                    d_uxx_r, d_uyy_r, d_uzz_r,
                    d_uxy_r, d_uxz_r, d_uyz_r,
                    d_phi_r,
                    d_eta_r,
                    (P.mode == 1 && P.minimize_full_model == 0) ? NULL : d_xB_r,
                    d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                    d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                    (float)P.eps_xx00, (float)P.eps_yy00, (float)P.eps_zz00,
                    (float)P.eps_yz00, (float)P.eps_xz00, (float)P.eps_xy00,
                    (double)P.eps_iso_over_vB,
                    is_gp_zone_mode(&P) ? 1 : 0,
                    gp_elastic_solver_enabled,
                    P.gp_eps_iso,
                    d_gel_diag_tmp,
                    total_r);
                launch_compute_gp_elastic_stats_kernel(
                    d_phi_r, d_eta_r, d_gel_diag_tmp,
                    d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                    P.gp_eps_iso, d_gp_elastic_stats, total_r);
                CUDA_CHECK(cudaMemcpy(gp_elastic_stats, d_gp_elastic_stats,
                                      GP_ELASTIC_STATS_COUNT * sizeof(double),
                                      cudaMemcpyDeviceToHost));
                mass_diag_row.mean_elastic_energy =
                    gp_elastic_stats[GP_ELASTIC_STATS_SUM_GEL] / (double)total_r;
                mass_diag_row.max_elastic_energy =
                    gp_elastic_stats[GP_ELASTIC_STATS_MAX_GEL];
                mass_diag_row.stress_hydro_min =
                    gp_elastic_stats[GP_ELASTIC_STATS_MIN_SIGMA_HYDRO];
                mass_diag_row.stress_hydro_max =
                    gp_elastic_stats[GP_ELASTIC_STATS_MAX_SIGMA_HYDRO];
                mass_diag_row.eps0_GP_contrib_min =
                    gp_elastic_stats[GP_ELASTIC_STATS_MIN_EPS0_GP_DIAG];
                mass_diag_row.eps0_GP_contrib_max =
                    gp_elastic_stats[GP_ELASTIC_STATS_MAX_EPS0_GP_DIAG];
            }

            // === 诊断：弹性 bulk 惩罚(能量密度)（只需在t=0力学平衡求解后输出一次）===
            // 注意：这里在 step=1 且更新phi之前执行，此时的 phi/sigma 对应 t=0。
            if (need_elastic_bulk_diag && !el_bulk_diag_ran) {
                el_bulk_diag = compute_elastic_bulk_penalty(
                    d_phi_r,
                    d_eta_r,
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
                if (need_delay_step0_csv && csv_fp && !P.minimize_continue_from_vtk) {
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
                if (P.ic_phi_seed_radius > 0.0) {
                    V0_target = compute_effective_vf_target(&P);
                } else {
                    V0_target = mean_h_now; // lock to initial volume fraction
                }
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
                    d_eta_r,
                    (P.mode == 1 && P.minimize_full_model == 0) ? NULL : d_xB_r,
                    d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                    d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                    (float)P.eps_xx00, (float)P.eps_yy00, (float)P.eps_zz00,
                    (float)P.eps_yz00, (float)P.eps_xz00, (float)P.eps_xy00,
                    (double)P.eps_iso_over_vB,
                    is_gp_zone_mode(&P) ? 1 : 0,
                    gp_elastic_solver_enabled,
                    P.gp_eps_iso,
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
                if (is_gp_zone_mode(&P)) {
                    launch_compute_mu_C_gp_kernel(d_Y_r, d_xB_r, d_mu_x_r,
                                                  temperature_K, P.mu_reference_scale,
                                                  P.Vm_alpha_0, P.dVm_alpha_dxB,
                                                  P.Y_clip, P.xB_eps,
                                                  d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                                                  P.eps_iso_over_vB,
                                                  total_r,
                                                  P.elastic_enabled);
                } else {
                    launch_compute_mu_x_kernel(d_Y_r, d_phi_r, d_xB_r, d_mu_x_r,
                                              temperature_K, P.mu_reference_scale,
                                              P.v_A, P.v_B, P.mu0_compound,
                                              P.Vm_compound, P.Vm_alpha_0,
                                              P.dVm_alpha_dxB, P.Y_clip, P.xB_eps,
                                              d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                                              P.eps_iso_over_vB,
                                              total_r,
                                              P.elastic_enabled);
                }

                // g_explicit = chem + W*g'(phi) + elastic（不含梯度项）
                launch_compute_phi_rhs_kernel(
                    d_phi_r, d_eta_r, d_xB_r, d_phi_rhs_r,
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
                    is_gp_zone_mode(&P) ? 1 : 0,
                    gp_elastic_solver_enabled,
                    gp_active_phi_coupling_enabled,
                    P.gp_eps_iso,
                    P.gp_elastic_derivative_scale,
                    (phi_elastic_coupling_enabled ? P.elastic_shift_dimless : 0.0),
                    /* disable_chem = */ 0,
                    total_r,
                    phi_elastic_coupling_enabled,
                    (do_mass_diag && gp_elastic_solver_enabled) ? d_gp_elastic_stats : NULL);
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
                if (is_gp_zone_mode(&P)) {
                    launch_compute_mu_C_gp_kernel(d_Y_r, d_xB_prev_r, d_mu_x_r,
                                                  temperature_K, P.mu_reference_scale,
                                                  P.Vm_alpha_0, P.dVm_alpha_dxB,
                                                  P.Y_clip, P.xB_eps,
                                                  d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                                                  P.eps_iso_over_vB,
                                                  total_r,
                                                  P.elastic_enabled);
                } else {
                    launch_compute_mu_x_kernel(d_Y_r, d_phi_r, d_xB_prev_r, d_mu_x_r,
                                              temperature_K, P.mu_reference_scale,
                                              P.v_A, P.v_B, P.mu0_compound,
                                              P.Vm_compound, P.Vm_alpha_0,
                                              P.dVm_alpha_dxB, P.Y_clip, P.xB_eps,
                                              d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                                              P.eps_iso_over_vB,
                                              total_r,
                                              P.elastic_enabled);
                }

                // 2) 使用 (phi_r, xB_tmp) 计算 g_explicit，再后续减 Lap/加 λh'
                launch_compute_phi_rhs_kernel(
                    d_phi_r, d_eta_r, d_xB_prev_r, d_res_r,
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
                    is_gp_zone_mode(&P) ? 1 : 0,
                    gp_elastic_solver_enabled,
                    gp_active_phi_coupling_enabled,
                    P.gp_eps_iso,
                    P.gp_elastic_derivative_scale,
                    (phi_elastic_coupling_enabled ? P.elastic_shift_dimless : 0.0),
                    /* disable_chem = */ 0,
                    total_r,
                    phi_elastic_coupling_enabled,
                    NULL);
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
            launch_compute_phi_rhs_kernel(d_phi_r, d_eta_r, d_xB_r, d_phi_rhs_r,
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
                                         is_gp_zone_mode(&P) ? 1 : 0,
                                         gp_elastic_solver_enabled,
                                         gp_active_phi_coupling_enabled,
                                         P.gp_eps_iso,
                                         P.gp_elastic_derivative_scale,
                                         (phi_elastic_coupling_enabled ? P.elastic_shift_dimless : 0.0),
                                         0,
                                         total_r,
                                         phi_elastic_coupling_enabled,
                                         (do_mass_diag && gp_elastic_solver_enabled) ? d_gp_elastic_stats : NULL);
            if (do_phi_eta_rhs_attribution_diag) {
                launch_compute_f_phi_chem_f_phi_dw_kernel(
                    d_phi_r, d_xB_r,
                    d_phi_rhs_bulk_diag_r, d_phi_rhs_dw_diag_r,
                    temperature_K, P.mu_reference_scale,
                    P.v_A, P.v_B, P.mu0_compound,
                    P.Vm_compound, P.Vm_alpha_0, P.dVm_alpha_dxB, P.W,
                    (phi_elastic_coupling_enabled ? P.elastic_shift_dimless : 0.0), total_r);
                launch_compute_f_phi_bulk_dgel_dphi_kernel(
                    d_phi_r, d_xB_r,
                    d_phi_rhs_bulk_diag_r, d_phi_rhs_elastic_diag_r,
                    temperature_K, P.mu_reference_scale,
                    P.v_A, P.v_B, P.mu0_compound,
                    P.Vm_compound, P.Vm_alpha_0, P.dVm_alpha_dxB,
                    d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                    d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                    d_uxx_r, d_uyy_r, d_uzz_r, d_uxy_r, d_uxz_r, d_uyz_r,
                    P.S_p_11, P.S_p_12, P.S_p_13, P.S_p_14, P.S_p_15, P.S_p_16,
                    P.S_p_22, P.S_p_23, P.S_p_24, P.S_p_25, P.S_p_26,
                    P.S_p_33, P.S_p_34, P.S_p_35, P.S_p_36,
                    P.S_p_44, P.S_p_45, P.S_p_46, P.S_p_55, P.S_p_56, P.S_p_66,
                    P.eps_xx00, P.eps_yy00, P.eps_zz00,
                    P.eps_yz00, P.eps_xz00, P.eps_xy00,
                    P.eps_iso_over_vB,
                    (phi_elastic_coupling_enabled ? P.elastic_shift_dimless : 0.0),
                    total_r, phi_elastic_coupling_enabled);
                CUDA_CHECK(cudaMemcpy(phi_rhs_total_explicit_host.data(), d_phi_rhs_r, size_r, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(phi_rhs_bulk_host.data(), d_phi_rhs_bulk_diag_r, size_r, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(phi_rhs_dw_host.data(), d_phi_rhs_dw_diag_r, size_r, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(phi_rhs_elastic_host.data(), d_phi_rhs_elastic_diag_r, size_r, cudaMemcpyDeviceToHost));
            }
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
            if (do_mass_diag) {
                double phi_stats[MASS_DIAG_PHI_STATS_COUNT] = {0.0};
                CUDA_CHECK(cudaMemset(d_mass_diag_phi_stats, 0, MASS_DIAG_PHI_STATS_COUNT * sizeof(double)));
                launch_phi_mass_diagnostics_kernel(d_phi_r, d_phi_n_saved, d_xB_r,
                                                   d_mass_diag_phi_stats, invN, P.v_B, total_r);
                CUDA_CHECK(cudaMemcpy(phi_stats, d_mass_diag_phi_stats,
                                      MASS_DIAG_PHI_STATS_COUNT * sizeof(double),
                                      cudaMemcpyDeviceToHost));
                const double inv_total = 1.0 / (double)total_r;
                mass_diag_row.mean_xBtot_before_phi_clip =
                    phi_stats[MASS_DIAG_PHI_SUM_XBTOT_RAW] * inv_total;
                mass_diag_row.mean_xBtot_after_phi_clip =
                    phi_stats[MASS_DIAG_PHI_SUM_XBTOT_CLAMPED] * inv_total;
                mass_diag_row.mean_h_before_phi_update =
                    phi_stats[MASS_DIAG_PHI_SUM_H_BEFORE] * inv_total;
                mass_diag_row.mean_h_after_phi_update =
                    phi_stats[MASS_DIAG_PHI_SUM_H_AFTER] * inv_total;
                mass_diag_row.delta_mean_h_phi_update =
                    mass_diag_row.mean_h_after_phi_update - mass_diag_row.mean_h_before_phi_update;
                mass_diag_row.mean_phi_before =
                    phi_stats[MASS_DIAG_PHI_SUM_PHI_BEFORE] * inv_total;
                mass_diag_row.mean_phi_after =
                    phi_stats[MASS_DIAG_PHI_SUM_PHI_AFTER] * inv_total;
                mass_diag_row.predicted_delta_xBtot_from_phi_change =
                    phi_stats[MASS_DIAG_PHI_SUM_PRED_DELTA_XBTOT] * inv_total;
                mass_diag_row.phi_clip_count_low =
                    phi_stats[MASS_DIAG_PHI_CLIP_LOW_COUNT];
                mass_diag_row.phi_clip_count_high =
                    phi_stats[MASS_DIAG_PHI_CLIP_HIGH_COUNT];
                mass_diag_row.delta_mass_phi_clip =
                    mass_diag_row.mean_xBtot_after_phi_clip - mass_diag_row.mean_xBtot_before_phi_clip;
            }
            launch_phi_normalize_and_clamp_kernel(d_phi_r, invN, total_r);
            if (do_mass_diag) {
                mass_diag_row.mean_xBtot_after_phi_update =
                    gpu_reduce_sum_model_xBtot(&P, d_phi_r, d_eta_r, d_xB_r, total_r) / (double)total_r;
                if (is_gp_zone_mode(&P)) {
                    mass_diag_row.mean_xBtot_gp_after_phi_update = mass_diag_row.mean_xBtot_after_phi_update;
                }
                mass_diag_row.delta_mass_phi_update =
                    mass_diag_row.mean_xBtot_after_phi_update - mass_diag_row.mean_xBtot_before_step;
            }
            if (do_phi_eta_field_snapshot) {
                CUDA_CHECK(cudaMemcpy(phi_after_phi_update_host.data(), d_phi_r, size_r,
                                      cudaMemcpyDeviceToHost));
            }
        }

        if (P.mode == 0 && is_gp_zone_mode(&P) ) {
            const double gp_mu_ref =
                compute_gp_mu_reference_mech_mix_host(
                    P.gp_xB_fixed, temperature_K, P.gp_delta_g_stab);
            CUFFT_CHECK(cufftExecD2Z(plan_r2c_phi, d_eta_r, d_eta_k));
            launch_dealias_kernel(d_eta_k, P.Nx, P.Ny, P.Nz, NzC,
                                 P.dx, P.dy, P.dz, total_k);
            launch_compute_eta_rhs_kernel(d_eta_r, d_phi_r, d_xB_r,
                                          d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                                          d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                                          d_eta_rhs_r,
                                          temperature_K, P.mu_reference_scale,
                                          P.Vm_alpha_0, P.dVm_alpha_dxB,
                                          P.gp_xB_fixed, P.gp_delta_g0,
                                          P.gp_raw_reaction_drive_only,
                                          P.gp_raw_reaction_drive_use_raw_units_debug,
                                          P.gp_reaction_nu_A, P.gp_reaction_nu_B,
                                          gp_mu_ref,
                                          P.gp_W_eta, P.eps_iso_over_vB,
                                          gp_elastic_solver_enabled,
                                          gp_active_eta_coupling_enabled,
                                          P.gp_eps_iso,
                                          P.gp_elastic_derivative_scale,
                                          total_r,
                                          (do_mass_diag && gp_elastic_solver_enabled) ? d_gp_elastic_stats : NULL);
            if (do_phi_eta_rhs_attribution_diag) {
                launch_compute_eta_rhs_components_kernel(
                    d_eta_r, d_phi_r, d_xB_r,
                    d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                    d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                    d_eta_rhs_bulk_diag_r, d_eta_rhs_dw_diag_r, d_eta_rhs_elastic_diag_r, NULL, NULL,
                    temperature_K, P.mu_reference_scale,
                    P.Vm_alpha_0, P.dVm_alpha_dxB,
                    P.gp_xB_fixed, P.gp_delta_g0,
                    P.gp_raw_reaction_drive_only,
                    P.gp_raw_reaction_drive_use_raw_units_debug,
                    P.gp_reaction_nu_A, P.gp_reaction_nu_B,
                    gp_mu_ref,
                    P.gp_W_eta, P.eps_iso_over_vB,
                    gp_elastic_solver_enabled,
                    gp_active_eta_coupling_enabled,
                    P.gp_eps_iso,
                    P.gp_elastic_derivative_scale,
                    total_r);
                CUDA_CHECK(cudaMemcpy(eta_rhs_total_explicit_host.data(), d_eta_rhs_r, size_r, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(eta_rhs_bulk_host.data(), d_eta_rhs_bulk_diag_r, size_r, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(eta_rhs_dw_host.data(), d_eta_rhs_dw_diag_r, size_r, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(eta_rhs_elastic_host.data(), d_eta_rhs_elastic_diag_r, size_r, cudaMemcpyDeviceToHost));
            }
            CUFFT_CHECK(cufftExecD2Z(plan_r2c_phi_rhs, d_eta_rhs_r, d_eta_rhs_k));
            launch_dealias_kernel(d_eta_rhs_k, P.Nx, P.Ny, P.Nz, NzC,
                                 P.dx, P.dy, P.dz, total_k);
            launch_eta_semi_implicit_update_kernel(d_eta_k, d_eta_rhs_k, KS.d_k2,
                                                   d_eta_k, P.gp_L_eta, P.gp_kappa_eta,
                                                   P.dt, total_k);
            launch_dealias_kernel(d_eta_k, P.Nx, P.Ny, P.Nz, NzC,
                                 P.dx, P.dy, P.dz, total_k);
            CUFFT_CHECK(cufftExecZ2D(plan_c2r_phi, d_eta_k, d_eta_r));
            launch_eta_normalize_and_clamp_kernel(d_eta_r, invN, total_r);
            if (do_mass_diag) {
                mass_diag_row.mean_xBtot_after_eta_update =
                    gpu_reduce_sum_model_xBtot(&P, d_phi_r, d_eta_r, d_xB_r, total_r) / (double)total_r;
                mass_diag_row.mean_xBtot_gp_after_eta_update = mass_diag_row.mean_xBtot_after_eta_update;
                mass_diag_row.delta_mass_eta_update =
                    mass_diag_row.mean_xBtot_after_eta_update - mass_diag_row.mean_xBtot_after_phi_update;
            }
            if (do_phi_eta_field_snapshot) {
                CUDA_CHECK(cudaMemcpy(eta_after_eta_update_host.data(), d_eta_r, size_r,
                                      cudaMemcpyDeviceToHost));
            }
        } else if (do_mass_diag && is_gp_zone_mode(&P)) {
            mass_diag_row.mean_xBtot_after_eta_update = mass_diag_row.mean_xBtot_after_phi_update;
            mass_diag_row.mean_xBtot_gp_after_eta_update = mass_diag_row.mean_xBtot_after_eta_update;
            mass_diag_row.delta_mass_eta_update = 0.0;
        }
        if (do_phi_eta_field_snapshot &&
            !(P.mode == 0 && is_gp_zone_mode(&P) )) {
            CUDA_CHECK(cudaMemcpy(eta_after_eta_update_host.data(), d_eta_r, size_r,
                                  cudaMemcpyDeviceToHost));
        }
        if (do_mass_diag && is_gp_zone_mode(&P)) {
            mass_diag_row.mean_xBtot_after_gp_to_beta_event = mass_diag_row.mean_xBtot_after_eta_update;
            mass_diag_row.mean_xBtot_gp_after_gp_to_beta_event = mass_diag_row.mean_xBtot_after_eta_update;
            mass_diag_row.delta_mass_gp_to_beta_event = 0.0;
        }
        if (is_gp_zone_mode(&P) && P.gp_nuc_enabled && P.mode == 0) {
            GpNucStepResult gp_nuc_result;
            if (!apply_gp_nucleation_event_cpu(&gp_nuc_runtime, &P, step,
                                               d_phi_r, d_eta_r, d_Y_r, d_xB_r,
                                               total_r, size_r, &gp_nuc_result)) {
                fprintf(stderr, "[fatal] gp nucleation event insertion failed at step %d\n", step);
                return 2;
            }
            if (do_mass_diag && gp_nuc_result.event_triggered) {
                mass_diag_row.mean_xBtot_after_gp_to_beta_event =
                    gpu_reduce_sum_model_xBtot(&P, d_phi_r, d_eta_r, d_xB_r, total_r) / (double)total_r;
                mass_diag_row.mean_xBtot_gp_after_gp_to_beta_event =
                    mass_diag_row.mean_xBtot_after_gp_to_beta_event;
                mass_diag_row.delta_mass_gp_to_beta_event =
                    mass_diag_row.mean_xBtot_after_gp_to_beta_event - mass_diag_row.mean_xBtot_after_eta_update;
            }
        }
        if (is_gp_zone_mode(&P) && P.gp_to_beta_enabled && P.mode == 0) {
            GpToBetaStepResult gp_to_beta_result;
            if (!apply_gp_to_beta_event_cpu(&gp_to_beta_runtime, &P, step,
                                            d_phi_r, d_eta_r, d_Y_r, d_xB_r,
                                            total_r, size_r, &gp_to_beta_result)) {
                fprintf(stderr, "[fatal] gp_to_beta event conversion failed at step %d\n", step);
                return 2;
            }
            if (do_mass_diag) {
                mass_diag_row.mean_xBtot_after_gp_to_beta_event =
                    gpu_reduce_sum_model_xBtot(&P, d_phi_r, d_eta_r, d_xB_r, total_r) / (double)total_r;
                mass_diag_row.mean_xBtot_gp_after_gp_to_beta_event =
                    mass_diag_row.mean_xBtot_after_gp_to_beta_event;
                mass_diag_row.delta_mass_gp_to_beta_event =
                    mass_diag_row.mean_xBtot_after_gp_to_beta_event - mass_diag_row.mean_xBtot_after_eta_update;
            }
            if (post_conversion_audit_runtime.csv &&
                post_conversion_audit_runtime.active == 0 &&
                post_conversion_audit_runtime.rows_remaining_steps == 0 &&
                P.post_conversion_y_update_audit_enabled &&
                gp_to_beta_result.event_triggered &&
                gp_to_beta_result.event_accepted) {
                post_conversion_audit_runtime.active = 1;
                post_conversion_audit_runtime.trigger_step = step;
                post_conversion_audit_runtime.rows_remaining_steps =
                    (P.post_conversion_y_update_audit_steps > 0) ? P.post_conversion_y_update_audit_steps : 1;
                post_conversion_audit_runtime.baseline_sum_xBtot =
                    gpu_reduce_sum_model_xBtot(&P, d_phi_r, d_eta_r, d_xB_r, total_r);
            }
            if ((P.y_update_k0_audit_enabled || P.y_update_mass_projection_enabled) &&
                gp_to_beta_result.event_triggered &&
                gp_to_beta_result.event_accepted) {
                y_update_k0_audit_runtime.projection_armed =
                    P.y_update_mass_projection_enabled ? 1 : 0;
                y_update_k0_audit_runtime.trigger_step = step;
                y_update_k0_audit_runtime.post_conversion_step_index = 0;
                y_update_k0_audit_runtime.post_conversion_baseline_sum_xBtot =
                    gpu_reduce_sum_model_xBtot(&P, d_phi_r, d_eta_r, d_xB_r, total_r);
                if (P.y_update_k0_audit_enabled) {
                    y_update_k0_audit_runtime.active = 1;
                    y_update_k0_audit_runtime.rows_remaining_steps =
                        (P.y_update_k0_audit_steps > 0) ? P.y_update_k0_audit_steps : 1;
                } else {
                    y_update_k0_audit_runtime.active = 0;
                    y_update_k0_audit_runtime.rows_remaining_steps = 0;
                }
                capture_post_conversion_device_state(&P, d_phi_r, d_eta_r, d_Y_r, d_xB_r,
                                                     plan_r2c_Y, d_Y_k, total_r, total_k,
                                                     &y_update_k0_audit_runtime.pre_Y_sum_xBtot,
                                                     &y_update_k0_audit_runtime.pre_Y_mean_xB,
                                                     &y_update_k0_audit_runtime.pre_Y_min_xB,
                                                     &y_update_k0_audit_runtime.pre_Y_max_xB,
                                                     &y_update_k0_audit_runtime.pre_Y_mean_Y,
                                                     &y_update_k0_audit_runtime.pre_Y_Y_k0_re,
                                                     &y_update_k0_audit_runtime.pre_Y_Y_k0_im);
            }
            if (post_conversion_audit_runtime.csv && post_conversion_audit_runtime.active &&
                post_conversion_audit_runtime.rows_remaining_steps > 0) {
                capture_post_conversion_device_state(&P, d_phi_r, d_eta_r, d_Y_r, d_xB_r,
                                                     plan_r2c_Y, d_Y_k, total_r, total_k,
                                                     &post_conversion_audit_runtime.pre_Y_sum_xBtot,
                                                     &post_conversion_audit_runtime.pre_Y_mean_xB,
                                                     &post_conversion_audit_runtime.pre_Y_min_xB,
                                                     &post_conversion_audit_runtime.pre_Y_max_xB,
                                                     &post_conversion_audit_runtime.pre_Y_mean_Y,
                                                     &post_conversion_audit_runtime.pre_Y_Y_k0_re,
                                                     &post_conversion_audit_runtime.pre_Y_Y_k0_im);
            }
            if (gp_to_beta_result.stop_after_conversion_audit_requested &&
                gp_to_beta_result.event_triggered &&
                gp_to_beta_result.event_accepted) {
                write_gp_to_beta_conversion_audit_fields(&P, case_output_dir,
                                                         d_phi_r, d_eta_r, d_xB_r,
                                                         (double *)d_scratch_r_double, total_r);
                stop_after_conversion_audit_done = 1;
            }
        }
        if (stop_after_conversion_audit_done) {
            if (do_mass_diag && mass_diag_fp) {
                write_mass_diag_row_csv(mass_diag_fp, &mass_diag_row, &P);
                fflush(mass_diag_fp);
            }
            break;
        }
        if (do_mass_diag && gp_elastic_solver_enabled) {
            double gp_elastic_stats[GP_ELASTIC_STATS_COUNT] = {0.0};
            CUDA_CHECK(cudaMemcpy(gp_elastic_stats, d_gp_elastic_stats,
                                  GP_ELASTIC_STATS_COUNT * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            mass_diag_row.mean_elastic_energy =
                gp_elastic_stats[GP_ELASTIC_STATS_SUM_GEL] / (double)total_r;
            mass_diag_row.max_elastic_energy =
                gp_elastic_stats[GP_ELASTIC_STATS_MAX_GEL];
            mass_diag_row.stress_hydro_min =
                gp_elastic_stats[GP_ELASTIC_STATS_MIN_SIGMA_HYDRO];
            mass_diag_row.stress_hydro_max =
                gp_elastic_stats[GP_ELASTIC_STATS_MAX_SIGMA_HYDRO];
            mass_diag_row.eps0_GP_contrib_min =
                gp_elastic_stats[GP_ELASTIC_STATS_MIN_EPS0_GP_DIAG];
            mass_diag_row.eps0_GP_contrib_max =
                gp_elastic_stats[GP_ELASTIC_STATS_MAX_EPS0_GP_DIAG];
            mass_diag_row.mean_dgel_deta_el =
                gp_elastic_stats[GP_ELASTIC_STATS_SUM_DGEL_DETA] / (double)total_r;
            mass_diag_row.max_abs_dgel_deta_el =
                gp_elastic_stats[GP_ELASTIC_STATS_MAXABS_DGEL_DETA];
            mass_diag_row.mean_dgel_dphi_el =
                gp_elastic_stats[GP_ELASTIC_STATS_SUM_DGEL_DPHI] / (double)total_r;
            mass_diag_row.max_abs_dgel_dphi_el =
                gp_elastic_stats[GP_ELASTIC_STATS_MAXABS_DGEL_DPHI];
            mass_diag_row.mean_eta_rhs_elastic_part =
                gp_elastic_stats[GP_ELASTIC_STATS_SUM_ETA_RHS_ELASTIC] / (double)total_r;
            mass_diag_row.max_abs_eta_rhs_elastic_part =
                gp_elastic_stats[GP_ELASTIC_STATS_MAXABS_ETA_RHS_ELASTIC];
            mass_diag_row.mean_phi_rhs_elastic_part =
                gp_elastic_stats[GP_ELASTIC_STATS_SUM_PHI_RHS_ELASTIC] / (double)total_r;
            mass_diag_row.max_abs_phi_rhs_elastic_part =
                gp_elastic_stats[GP_ELASTIC_STATS_MAXABS_PHI_RHS_ELASTIC];
            mass_diag_row.gp_mu_reference_raw =
                P.gp_raw_reaction_drive_only
                    ? compute_gp_mu_reference_mech_mix_host(
                        P.gp_xB_fixed, temperature_K, P.gp_delta_g_stab)
                    : NAN;
            mass_diag_row.gp_minus_delta_mu_r_mean =
                P.gp_raw_reaction_drive_only
                    ? (gp_elastic_stats[GP_ELASTIC_STATS_SUM_MINUS_DELTA_MU_R_GP] / (double)total_r)
                    : NAN;
            mass_diag_row.gp_minus_delta_mu_r_min =
                P.gp_raw_reaction_drive_only
                    ? gp_elastic_stats[GP_ELASTIC_STATS_MIN_MINUS_DELTA_MU_R_GP]
                    : NAN;
            mass_diag_row.gp_minus_delta_mu_r_max =
                P.gp_raw_reaction_drive_only
                    ? gp_elastic_stats[GP_ELASTIC_STATS_MAX_MINUS_DELTA_MU_R_GP]
                    : NAN;
        }
        
        // Optimization: 步骤1完成，d_phi_rhs_k 可复用为 d_mu_x_k
        d_mu_x_k = d_phi_rhs_k;
        
        // ============================================================
        // 步骤2：Y方程的更新
        // ============================================================
        int minimize_should_stop = 0;

        if (P.mode == 0 || (P.mode == 1 && P.minimize_full_model == 1)) {
        
        const double *phi_for_Y_explicit = P.enable_Y_rhs_previous_time_level ? d_phi_n_saved : d_phi_r;
        const double *eta_for_Y_explicit = P.enable_Y_rhs_previous_time_level ? d_eta_prev_r : d_eta_r;

        // 2.1 计算mu_x（full-model minimize 与 dynamics 共用）
        if (is_gp_zone_mode(&P)) {
            launch_compute_mu_C_gp_kernel(d_Y_r, d_xB_r, d_mu_x_r,
                                          temperature_K, P.mu_reference_scale,
                                          P.Vm_alpha_0, P.dVm_alpha_dxB,
                                          P.Y_clip, P.xB_eps,
                                          d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                                          P.eps_iso_over_vB,
                                          total_r,
                                          P.elastic_enabled);
        } else {
            launch_compute_mu_x_kernel(d_Y_r, phi_for_Y_explicit, d_xB_r, d_mu_x_r,
                                      temperature_K, P.mu_reference_scale,
                                      P.v_A, P.v_B, P.mu0_compound,
                                      P.Vm_compound, P.Vm_alpha_0,
                                      P.dVm_alpha_dxB, P.Y_clip, P.xB_eps,
                                      d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                                      P.eps_iso_over_vB,
                                      total_r,
                                      P.elastic_enabled);
        }
        if (do_mass_diag && is_gp_zone_mode(&P)) {
            compute_scalar_field_stats(d_mu_x_r, total_r,
                                       &mass_diag_row.mu_C_mean,
                                       &mass_diag_row.mu_C_min,
                                       &mass_diag_row.mu_C_max);
            double gp_transport_stats_init[GP_TRANSPORT_STATS_COUNT];
            double gp_transport_stats[GP_TRANSPORT_STATS_COUNT] = {0.0};
            init_gp_transport_stats_host(gp_transport_stats_init);
            CUDA_CHECK(cudaMemcpy(d_gp_transport_stats, gp_transport_stats_init,
                                  GP_TRANSPORT_STATS_COUNT * sizeof(double),
                                  cudaMemcpyHostToDevice));
            launch_compute_gp_transport_stats_kernel(
                d_phi_r, d_eta_r, d_xB_r, P.gp_xB_fixed,
                P.D_alpha, P.gp_M_GP, P.gp_M_beta,
                P.Vm_alpha_0, P.dVm_alpha_dxB, P.Vm_compound,
                temperature_K, P.mu_reference_scale,
                                                d_gp_transport_stats, total_r);
            CUDA_CHECK(cudaMemcpy(gp_transport_stats, d_gp_transport_stats,
                                  GP_TRANSPORT_STATS_COUNT * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            mass_diag_row.xB_alpha_perturb_maxabs =
                gp_transport_stats[GP_TRANSPORT_MAXABS_XB_PERTURB];
            mass_diag_row.h_alpha_transport_min =
                gp_transport_stats[GP_TRANSPORT_MIN_HALPHA];
            mass_diag_row.h_alpha_transport_max =
                gp_transport_stats[GP_TRANSPORT_MAX_HALPHA];
            mass_diag_row.M_alpha_min =
                gp_transport_stats[GP_TRANSPORT_MIN_M_ALPHA];
            mass_diag_row.M_alpha_max =
                gp_transport_stats[GP_TRANSPORT_MAX_M_ALPHA];
            mass_diag_row.M_eff_min =
                gp_transport_stats[GP_TRANSPORT_MIN_M_EFF];
            mass_diag_row.M_eff_max =
                gp_transport_stats[GP_TRANSPORT_MAX_M_EFF];
        }
        
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
        CUDA_CHECK(cudaMemcpy(d_xB_gp_old_r, d_xB_r, size_r, cudaMemcpyDeviceToDevice));
        if (d_xB_old_diag_r) {
            CUDA_CHECK(cudaMemcpy(d_xB_old_diag_r, d_xB_r, size_r, cudaMemcpyDeviceToDevice));
        }
        for (int alpha = 0; alpha < 3; alpha++) {
            // Optimization: use scratch_k_z for grad_mu_k then J_k sequentially
            launch_compute_gradient_single_component_k_kernel(
                d_mu_x_k, scratch_k, alpha,
                P.Nx, P.Ny, P.Nz, NzC, P.dx, P.dy, P.dz, total_k);
            launch_dealias_kernel(scratch_k, P.Nx, P.Ny, P.Nz, NzC,
                                 P.dx, P.dy, P.dz, total_k);
            CUFFT_CHECK(cufftExecZ2D(plan_c2r_xB, scratch_k, scratch_r));
            launch_normalize_only_kernel(scratch_r, invN, total_r);
            if (do_mass_diag && is_gp_zone_mode(&P)) {
                double grad_min = 0.0;
                double grad_max = 0.0;
                gpu_reduce_min_max(scratch_r, total_r, &grad_min, &grad_max);
                if (alpha == 0) {
                    mass_diag_row.grad_mu_x_min = grad_min;
                    mass_diag_row.grad_mu_x_max = grad_max;
                } else if (alpha == 1) {
                    mass_diag_row.grad_mu_y_min = grad_min;
                    mass_diag_row.grad_mu_y_max = grad_max;
                } else {
                    mass_diag_row.grad_mu_z_min = grad_min;
                    mass_diag_row.grad_mu_z_max = grad_max;
                }
            }
            // Optimization: reuse scratch_r_d for grad_mu_r and J_r (same buffer)
            if (is_gp_zone_mode(&P)) {
                launch_compute_flux_single_component_gp_kernel(
                    scratch_r, phi_for_Y_explicit, d_eta_r, d_xB_prev_r, scratch_r,
                    P.D_alpha, P.gp_M_GP, P.gp_M_beta,
                    P.Vm_alpha_0, P.dVm_alpha_dxB,
                    P.Vm_compound, temperature_K, P.mu_reference_scale,
                    total_r);
            } else {
                launch_compute_flux_single_component_kernel(
                    scratch_r, phi_for_Y_explicit, d_xB_prev_r, scratch_r,
                    P.D_alpha, P.D_compound, P.Vm_alpha_0, P.dVm_alpha_dxB,
                    P.Vm_compound, temperature_K, P.mu_reference_scale, total_r);
            }
            if (do_mass_diag && is_gp_zone_mode(&P)) {
                double flux_min = 0.0;
                double flux_max = 0.0;
                gpu_reduce_min_max(scratch_r, total_r, &flux_min, &flux_max);
                if (alpha == 0) {
                    mass_diag_row.flux_x_min = flux_min;
                    mass_diag_row.flux_x_max = flux_max;
                } else if (alpha == 1) {
                    mass_diag_row.flux_y_min = flux_min;
                    mass_diag_row.flux_y_max = flux_max;
                } else {
                    mass_diag_row.flux_z_min = flux_min;
                    mass_diag_row.flux_z_max = flux_max;
                }
            }
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
        if (do_mass_diag) {
            cuDoubleComplex divJ_k0 = make_cuDoubleComplex(0.0, 0.0);
            CUDA_CHECK(cudaMemcpy(&divJ_k0, d_divJ_k, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            mass_diag_row.divJ_k0_real_before_update = cuCreal(divJ_k0);
            mass_diag_row.divJ_k0_imag_before_update = cuCimag(divJ_k0);
        }
        launch_normalize_only_kernel(d_divJ_r, invN, total_r);

        if (is_gp_zone_mode(&P)) {
            const int eta_limiter_enabled = (strcmp(P.gp_eta_mass_limiter, "local_clip") == 0) ? 1 : 0;
            if (do_mass_diag) {
                double eta_feas_stats_init[GP_ETA_FEAS_STATS_COUNT];
                init_gp_eta_feasibility_stats_host(eta_feas_stats_init);
                CUDA_CHECK(cudaMemcpy(d_gp_eta_feas_stats, eta_feas_stats_init,
                                      GP_ETA_FEAS_STATS_COUNT * sizeof(double),
                                      cudaMemcpyHostToDevice));
            }
            launch_gp_eta_feasibility_and_limiter_kernel(
                d_eta_r, d_eta_prev_r, d_phi_r, d_phi_n_saved, d_xB_gp_old_r,
                d_divJ_r, P.dt, P.gp_xB_fixed,
                P.xB_eps, 1.0 - P.xB_eps, P.gp_h_alpha_eps,
                eta_limiter_enabled,
                do_mass_diag ? d_gp_eta_feas_stats : NULL, total_r);
            if (eta_limiter_enabled) {
                launch_zero_divJ_k_kernel(d_divJ_k, total_k);
                for (int alpha = 0; alpha < 3; alpha++) {
                    launch_compute_gradient_single_component_k_kernel(
                        d_mu_x_k, scratch_k, alpha,
                        P.Nx, P.Ny, P.Nz, NzC, P.dx, P.dy, P.dz, total_k);
                    launch_dealias_kernel(scratch_k, P.Nx, P.Ny, P.Nz, NzC,
                                         P.dx, P.dy, P.dz, total_k);
                    CUFFT_CHECK(cufftExecZ2D(plan_c2r_xB, scratch_k, scratch_r));
                    launch_normalize_only_kernel(scratch_r, invN, total_r);
                    launch_compute_flux_single_component_gp_kernel(
                        scratch_r, phi_for_Y_explicit, d_eta_r, d_xB_prev_r, scratch_r,
                        P.D_alpha, P.gp_M_GP, P.gp_M_beta,
                        P.Vm_alpha_0, P.dVm_alpha_dxB,
                        P.Vm_compound, temperature_K, P.mu_reference_scale,
                                                total_r);
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
                if (do_mass_diag) {
                    double eta_feas_stats_init[GP_ETA_FEAS_STATS_COUNT];
                    init_gp_eta_feasibility_stats_host(eta_feas_stats_init);
                    CUDA_CHECK(cudaMemcpy(d_gp_eta_feas_stats, eta_feas_stats_init,
                                          GP_ETA_FEAS_STATS_COUNT * sizeof(double),
                                          cudaMemcpyHostToDevice));
                }
                launch_gp_eta_feasibility_and_limiter_kernel(
                    d_eta_r, d_eta_prev_r, d_phi_r, d_phi_n_saved, d_xB_gp_old_r,
                    d_divJ_r, P.dt, P.gp_xB_fixed,
                    P.xB_eps, 1.0 - P.xB_eps, P.gp_h_alpha_eps,
                    eta_limiter_enabled,
                    do_mass_diag ? d_gp_eta_feas_stats : NULL, total_r);
            }
            if (do_mass_diag) {
                double eta_feas_stats[GP_ETA_FEAS_STATS_COUNT] = {0.0};
                CUDA_CHECK(cudaMemcpy(eta_feas_stats, d_gp_eta_feas_stats,
                                      GP_ETA_FEAS_STATS_COUNT * sizeof(double),
                                      cudaMemcpyDeviceToHost));
                mass_diag_row.recovered_xB_min = eta_feas_stats[GP_ETA_FEAS_MIN_RECOVERED_XB_BEFORE];
                mass_diag_row.recovered_xB_max = eta_feas_stats[GP_ETA_FEAS_MAX_RECOVERED_XB_BEFORE];
                mass_diag_row.recovered_xB_gt_xBmax_count = eta_feas_stats[GP_ETA_FEAS_GT_XBMAX_COUNT_BEFORE];
                mass_diag_row.recovered_xB_lt_xBmin_count = eta_feas_stats[GP_ETA_FEAS_LT_XBMIN_COUNT_BEFORE];
                mass_diag_row.recovered_xB_min_after_limiter = eta_feas_stats[GP_ETA_FEAS_MIN_RECOVERED_XB_AFTER];
                mass_diag_row.recovered_xB_max_after_limiter = eta_feas_stats[GP_ETA_FEAS_MAX_RECOVERED_XB_AFTER];
                mass_diag_row.recovered_xB_gt_xBmax_count_after_limiter = eta_feas_stats[GP_ETA_FEAS_GT_XBMAX_COUNT_AFTER];
                mass_diag_row.recovered_xB_lt_xBmin_count_after_limiter = eta_feas_stats[GP_ETA_FEAS_LT_XBMIN_COUNT_AFTER];
                mass_diag_row.recovered_xB_invalid_count_before_limiter = eta_feas_stats[GP_ETA_FEAS_INVALID_COUNT_BEFORE];
                mass_diag_row.recovered_xB_invalid_count_after_limiter = eta_feas_stats[GP_ETA_FEAS_INVALID_COUNT_AFTER];
                mass_diag_row.recovered_xB_invalid_mass = eta_feas_stats[GP_ETA_FEAS_INVALID_MASS] / (double)total_r;
                mass_diag_row.eta_tentative_max = eta_feas_stats[GP_ETA_FEAS_MAX_ETA_TENT];
                mass_diag_row.h_GP_new_max = eta_feas_stats[GP_ETA_FEAS_MAX_HGP_NEW];
                mass_diag_row.eta_new_max = eta_feas_stats[GP_ETA_FEAS_MAX_ETA_NEW];
                mass_diag_row.eta_limiter_count = eta_feas_stats[GP_ETA_FEAS_LIMITER_COUNT];
                mass_diag_row.eta_limiter_max_delta = eta_feas_stats[GP_ETA_FEAS_LIMITER_MAX_DELTA];
                mass_diag_row.eta_limiter_mass_prevented = eta_feas_stats[GP_ETA_FEAS_LIMITER_MASS_PREVENTED] / (double)total_r;
            }
            if (do_mass_diag) {
                double eta_min_local = 0.0;
                double eta_max_local = 0.0;
                double eta_diff_stats[2] = {0.0, 0.0};
                gpu_reduce_min_max(d_eta_r, total_r, &eta_min_local, &eta_max_local);
                CUDA_CHECK(cudaMemset(d_picard_diff_stats, 0, 2 * sizeof(double)));
                launch_compute_diff_sumsq_max_kernel(d_eta_r, d_eta_prev_r,
                                                     d_picard_diff_stats, total_r);
                CUDA_CHECK(cudaMemcpy(eta_diff_stats, d_picard_diff_stats,
                                      2 * sizeof(double), cudaMemcpyDeviceToHost));
                mass_diag_row.eta_step_max = eta_max_local;
                mass_diag_row.eta_step_max_delta = eta_diff_stats[1];
                mass_diag_row.eta_integral =
                    gpu_reduce_sum(d_eta_r, total_r) * (P.dx * P.dy * P.dz);
                compute_scalar_field_stats(d_divJ_r, total_r,
                                           &mass_diag_row.mean_divJ,
                                           &mass_diag_row.divJ_min,
                                           &mass_diag_row.divJ_max);
                mass_diag_row.mean_divJ_effective = mass_diag_row.mean_divJ;
                mass_diag_row.dt_divJ_min = P.dt * mass_diag_row.divJ_min;
                mass_diag_row.dt_divJ_max = P.dt * mass_diag_row.divJ_max;
                mass_diag_row.maxabs_divJ =
                    fmax(fabs(mass_diag_row.divJ_min), fabs(mass_diag_row.divJ_max));
                mass_diag_row.max_abs_xBtot_target_delta =
                    P.dt * mass_diag_row.maxabs_divJ;
                const double max_abs_dt_divJ =
                    fmax(fabs(mass_diag_row.dt_divJ_min), fabs(mass_diag_row.dt_divJ_max));
                if (max_abs_dt_divJ > 1.0e-1) {
                    printf("[gp_zone][step=%d] WARNING TRANSPORT_UNSTABLE_RISK max_abs(dt*divJ)=%.6e\n",
                           step, max_abs_dt_divJ);
                } else if (max_abs_dt_divJ > 1.0e-2) {
                    printf("[gp_zone][step=%d] WARNING TRANSPORT_TOO_AGGRESSIVE max_abs(dt*divJ)=%.6e\n",
                           step, max_abs_dt_divJ);
                }
            }
        }
        
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
        double global_sum_DY = is_gp_zone_mode(&P)
                                   ? gpu_reduce_sum_DY_gp(d_Y_r, phi_for_Y_explicit,
                                                          eta_for_Y_explicit,
                                                          P.D_alpha, total_r)
                                   : gpu_reduce_sum_DY(d_Y_r, phi_for_Y_explicit,
                                                       P.D_alpha, P.D_compound, total_r);
        double mean_DY = global_sum_DY / (double)total_r;
        
        // 2.10-2.14 Y 更新：
        // - two_phase / gp old_rhs / gp conservative_y_rhs 继续沿用半隐式 Fourier 链
        // - gp picard_storage / storage_exact 改走 storage-consistent 本地点更新
        const int gp_mode_old_rhs =
            (is_gp_zone_mode(&P) && strcmp(P.gp_y_update_mode, "old_rhs") == 0);
        const int gp_mode_conservative_y_rhs =
            (is_gp_zone_mode(&P) && strcmp(P.gp_y_update_mode, "conservative_y_rhs") == 0);
        const int gp_mode_picard_storage =
            (is_gp_zone_mode(&P) && strcmp(P.gp_y_update_mode, "picard_storage") == 0);
        const int gp_mode_storage_exact =
            (is_gp_zone_mode(&P) && strcmp(P.gp_y_update_mode, "storage_exact") == 0);
        const int gp_mode_direct_storage = gp_mode_picard_storage || gp_mode_storage_exact;
        const int use_Y_rhs_picard =
            (!gp_mode_direct_storage && P.mode == 0 && P.enable_Y_rhs_picard) ? 1 : 0;
        double Y_upper_cap = P.Y_clip;
        if (P.mode == 1) {
            double xB_cap = P.minimize_xB_max_safe;
            if (xB_cap < P.xB_eps) xB_cap = P.xB_eps;
            if (xB_cap > 1.0 - P.xB_eps) xB_cap = 1.0 - P.xB_eps;
            double y_cap_from_xB = log(xB_cap / (1.0 - xB_cap));
            if (y_cap_from_xB < Y_upper_cap) Y_upper_cap = y_cap_from_xB;
        }

        if (!gp_mode_direct_storage) {
            const int Y_rhs_picard_iters = use_Y_rhs_picard ? ((P.Y_rhs_picard_iters > 0) ? P.Y_rhs_picard_iters : 1) : 1;
            const double Y_rhs_picard_omega = (P.Y_rhs_picard_omega > 0.0 && P.Y_rhs_picard_omega <= 1.0)
                                                  ? P.Y_rhs_picard_omega : 1.0;
            const double *Y_for_Y_rhs = use_Y_rhs_picard ? d_Y_n_saved : d_Y_r;

            if (use_Y_rhs_picard) {
                CUDA_CHECK(cudaMemcpy(d_dY_dt_picard_r, d_dY_dt_prev_r, size_r, cudaMemcpyDeviceToDevice));
            }
            CUFFT_CHECK(cufftExecD2Z(plan_r2c_Y, d_Y_n_saved, d_divJ_k)); // 保存 Y^n 的 k-space
            launch_dealias_kernel(d_divJ_k, P.Nx, P.Ny, P.Nz, NzC,
                                 P.dx, P.dy, P.dz, total_k);

            for (int pic_iter = 0; pic_iter < Y_rhs_picard_iters; ++pic_iter) {
                if (use_Y_rhs_picard) {
                    CUDA_CHECK(cudaMemcpy(d_dY_dt_picard_old_r, d_dY_dt_picard_r, size_r, cudaMemcpyDeviceToDevice));
                }
                if (do_mass_diag) {
                    double y_rhs_stats_init[MASS_DIAG_Y_RHS_STATS_COUNT];
                    init_mass_diag_y_rhs_stats_host(y_rhs_stats_init);
                    CUDA_CHECK(cudaMemcpy(d_mass_diag_Y_rhs_stats, y_rhs_stats_init,
                                          MASS_DIAG_Y_RHS_STATS_COUNT * sizeof(double),
                                          cudaMemcpyHostToDevice));
                }
                if (is_gp_zone_mode(&P)) {
                    if (gp_mode_conservative_y_rhs) {
                        launch_compute_Y_rhs_gp_conservative_kernel(
                            d_divJ_r, d_phi_r, d_phi_n_saved,
                            d_eta_r, d_eta_prev_r, d_lapY_r, Y_for_Y_rhs,
                            use_Y_rhs_picard ? d_dY_dt_picard_r : d_dY_dt_prev_r,
                            d_Y_rhs_r, P.dt, P.gp_xB_fixed, mean_DY, total_r,
                            P.enable_Y_rhs_previous_time_level ? 1 : 0,
                            P.disable_Y_rhs_gamma_term ? 1 : 0,
                            do_mass_diag ? d_mass_diag_Y_rhs_stats : NULL);
                    } else {
                        launch_compute_Y_rhs_gp_kernel(d_divJ_r, d_phi_r, d_phi_n_saved,
                                                       d_eta_r, d_eta_prev_r, d_lapY_r, Y_for_Y_rhs,
                                                       use_Y_rhs_picard ? d_dY_dt_picard_r : d_dY_dt_prev_r,
                                                       d_Y_rhs_r, P.dt, P.gp_xB_fixed, mean_DY,
                                                       P.gp_h_alpha_eps, total_r,
                                                       P.enable_Y_rhs_previous_time_level ? 1 : 0,
                                                       P.disable_Y_rhs_gamma_term ? 1 : 0,
                                                       do_mass_diag ? d_mass_diag_Y_rhs_stats : NULL);
                    }
                } else {
                    launch_compute_Y_rhs_kernel(d_divJ_r, d_phi_r, d_phi_n_saved,
                                                d_lapY_r, Y_for_Y_rhs,
                                                use_Y_rhs_picard ? d_dY_dt_picard_r : d_dY_dt_prev_r,
                                                d_Y_rhs_r, P.dt, P.v_B, mean_DY, total_r,
                                                P.enable_Y_rhs_previous_time_level ? 1 : 0,
                                                P.disable_Y_rhs_gamma_term ? 1 : 0,
                                                P.Y_rhs_term_h_scale,
                                                do_mass_diag ? d_mass_diag_Y_rhs_stats : NULL);
                }

                CUFFT_CHECK(cufftExecD2Z(plan_r2c_Y, d_Y_rhs_r, d_Y_rhs_k));
                launch_dealias_kernel(d_Y_rhs_k, P.Nx, P.Ny, P.Nz, NzC,
                                     P.dx, P.dy, P.dz, total_k);

                launch_Y_semi_implicit_update_kernel(d_divJ_k, d_Y_rhs_k, KS.d_k2,
                                                    d_Y_k, mean_DY, P.dt, total_k);
                launch_dealias_kernel(d_Y_k, P.Nx, P.Ny, P.Nz, NzC,
                                     P.dx, P.dy, P.dz, total_k);

                CUFFT_CHECK(cufftExecZ2D(plan_c2r_Y, d_Y_k, d_Y_r));
                launch_Y_normalize_and_clamp_kernel(d_Y_r, d_xB_r, invN,
                                                   P.Y_clip, Y_upper_cap, P.xB_eps, total_r);

                if (use_Y_rhs_picard) {
                    launch_update_dY_dt_prev_kernel(d_Y_r, d_Y_n_saved, d_dY_dt_picard_r, P.dt, total_r);
                    if (Y_rhs_picard_omega < 1.0) {
                        launch_relax_dY_dt_guess_kernel(d_dY_dt_picard_r, d_dY_dt_picard_old_r,
                                                        d_dY_dt_picard_r, Y_rhs_picard_omega, total_r);
                    }
                    if (do_mass_diag) {
                        CUDA_CHECK(cudaMemset(d_picard_diff_stats, 0, 2 * sizeof(double)));
                        launch_compute_diff_sumsq_max_kernel(d_dY_dt_picard_r, d_dY_dt_picard_old_r,
                                                             d_picard_diff_stats, total_r);
                        if (pic_iter == 0) {
                            mass_diag_row.picard_mean_xBtot_after_iter_0 =
                                gpu_reduce_sum_model_xBtot(&P, d_phi_r, d_eta_r, d_xB_r, total_r) / (double)total_r;
                        }
                        if (pic_iter == Y_rhs_picard_iters - 1) {
                            double picard_stats[2] = {0.0, 0.0};
                            CUDA_CHECK(cudaMemcpy(picard_stats, d_picard_diff_stats, 2 * sizeof(double),
                                                  cudaMemcpyDeviceToHost));
                            mass_diag_row.picard_rms_dYdt_change_final =
                                sqrt(picard_stats[0] / (double)total_r);
                            mass_diag_row.picard_maxabs_dYdt_change_final = picard_stats[1];
                        }
                    }
                }
            } // end Picard / single-pass Y solve loop
        } else {
            if (do_mass_diag) {
                double gp_storage_stats_init[MASS_DIAG_GP_STORAGE_STATS_COUNT];
                init_gp_storage_stats_host(gp_storage_stats_init);
                CUDA_CHECK(cudaMemcpy(d_mass_diag_gp_storage_stats, gp_storage_stats_init,
                                      MASS_DIAG_GP_STORAGE_STATS_COUNT * sizeof(double),
                                      cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(d_gp_Y_update_stats, 0, GP_Y_UPDATE_STATS_COUNT * sizeof(double)));
            }

            if (gp_mode_storage_exact) {
                launch_gp_storage_exact_Y_update_kernel(
                    d_divJ_r, d_phi_r, d_phi_n_saved, d_eta_r, d_eta_prev_r,
                    d_Y_n_saved, d_Y_r, d_xB_r, P.dt, P.gp_xB_fixed, P.Y_clip,
                    Y_upper_cap, P.xB_eps, P.gp_h_alpha_eps,
                    0,
                    do_mass_diag ? d_gp_Y_update_stats : NULL, total_r);
            } else {
                const int gp_picard_iters = (P.gp_y_picard_iters > 0) ? P.gp_y_picard_iters : 1;
                for (int gp_iter = 0; gp_iter < gp_picard_iters; ++gp_iter) {
                    launch_gp_picard_storage_Y_update_kernel(
                        d_divJ_r, d_phi_r, d_phi_n_saved, d_eta_r, d_eta_prev_r,
                        d_Y_n_saved, d_Y_r, d_xB_r, P.dt, P.gp_xB_fixed, P.Y_clip,
                        Y_upper_cap, P.xB_eps,
                        (do_mass_diag && gp_iter == gp_picard_iters - 1) ? d_gp_Y_update_stats : NULL,
                        total_r);
                    if (do_mass_diag && gp_iter == 0) {
                        mass_diag_row.picard_mean_xBtot_after_iter_0 =
                            gpu_reduce_sum_model_xBtot(&P, d_phi_r, d_eta_r, d_xB_r, total_r) / (double)total_r;
                    }
                }
            }
        }

        if (is_gp_zone_mode(&P) &&
            gp_mode_direct_storage &&
            (y_update_k0_audit_runtime.projection_armed ||
             (y_update_k0_audit_runtime.active &&
              y_update_k0_audit_runtime.rows_remaining_steps > 0))) {
            const int audit_step_index =
                (y_update_k0_audit_runtime.active && y_update_k0_audit_runtime.rows_remaining_steps > 0)
                    ? ((P.y_update_k0_audit_steps - y_update_k0_audit_runtime.rows_remaining_steps) + 1)
                    : 0;
            const int post_conversion_step = step - y_update_k0_audit_runtime.trigger_step + 1;
            const double sum_xBtot_before_Y = y_update_k0_audit_runtime.pre_Y_sum_xBtot;
            double sum_xBtot_after_Y = 0.0;
            double mean_xB_after = 0.0, min_xB_after = 0.0, max_xB_after = 0.0;
            double mean_Y_after = 0.0, Y_k0_after_re = 0.0, Y_k0_after_im = 0.0;
            const double mean_phi_now = gpu_reduce_sum(d_phi_r, total_r) / (double)total_r;
            const double mean_eta_now = gpu_reduce_sum(d_eta_r, total_r) / (double)total_r;
            const double mean_h_beta_now = gpu_compute_vf_from_h(d_phi_r, total_r);
            const double mean_h_gp_now = NAN;
            capture_post_conversion_device_state(&P, d_phi_r, d_eta_r, d_Y_r, d_xB_r,
                                                 plan_r2c_Y, d_Y_k, total_r, total_k,
                                                 &sum_xBtot_after_Y,
                                                 &mean_xB_after,
                                                 &min_xB_after,
                                                 &max_xB_after,
                                                 &mean_Y_after,
                                                 &Y_k0_after_re,
                                                 &Y_k0_after_im);
            const double delta_Y_k0_re =
                Y_k0_after_re - y_update_k0_audit_runtime.pre_Y_Y_k0_re;
            const double rhs_k0_total =
                (P.dt > 0.0) ? (delta_Y_k0_re / P.dt) : NAN;
            const double target_sum_for_projection =
                (strcmp(P.y_update_mass_projection_target_mode, "post_conversion_baseline") == 0)
                    ? y_update_k0_audit_runtime.post_conversion_baseline_sum_xBtot
                    : sum_xBtot_before_Y;
            if (y_update_k0_audit_runtime.csv &&
                y_update_k0_audit_runtime.active &&
                y_update_k0_audit_runtime.rows_remaining_steps > 0) {
                write_y_update_k0_audit_row(
                    y_update_k0_audit_runtime.csv,
                    step,
                    audit_step_index,
                    "after_Y_update",
                    post_conversion_step,
                    sum_xBtot_before_Y,
                    sum_xBtot_after_Y,
                    sum_xBtot_after_Y - sum_xBtot_before_Y,
                    target_sum_for_projection,
                    y_update_k0_audit_runtime.pre_Y_Y_k0_re,
                    Y_k0_after_re,
                    delta_Y_k0_re,
                    rhs_k0_total,
                    NAN, NAN, NAN, NAN, NAN, NAN, rhs_k0_total,
                    y_update_k0_audit_runtime.pre_Y_mean_Y,
                    mean_Y_after,
                    y_update_k0_audit_runtime.pre_Y_mean_xB,
                    mean_xB_after,
                    y_update_k0_audit_runtime.pre_Y_min_xB,
                    y_update_k0_audit_runtime.pre_Y_max_xB,
                    min_xB_after,
                    max_xB_after,
                    mean_phi_now,
                    mean_eta_now,
                    mean_h_gp_now,
                    mean_h_beta_now,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    "direct_storage_total_only");
                fflush(y_update_k0_audit_runtime.csv);
            }
            if (P.y_update_mass_projection_enabled && d_Y_projection_base_r) {
                double lambda_shift = 0.0;
                double sum_before_proj = 0.0, sum_after_proj = 0.0;
                double min_xB_before_proj = 0.0, max_xB_before_proj = 0.0;
                double min_xB_after_proj = 0.0, max_xB_after_proj = 0.0;
                int converged = 0;
                if (!run_y_update_mass_projection(
                        &P, step, post_conversion_step,
                        P.y_update_mass_projection_target_mode,
                        target_sum_for_projection,
                        d_phi_r, d_eta_r, d_Y_projection_base_r,
                        d_Y_r, d_xB_r, total_r, y_update_mass_projection_fp,
                        &lambda_shift,
                        &sum_before_proj,
                        &sum_after_proj,
                        &min_xB_before_proj,
                        &max_xB_before_proj,
                        &min_xB_after_proj,
                        &max_xB_after_proj,
                        &converged)) {
                    fprintf(stderr, "[fatal] Y mass projection failed at step %d\n", step);
                    return 2;
                }
                if (y_update_k0_audit_runtime.csv &&
                    y_update_k0_audit_runtime.active &&
                    y_update_k0_audit_runtime.rows_remaining_steps > 0) {
                    double mean_xB_after_proj = 0.0, mean_Y_after_proj = 0.0;
                    double Y_k0_after_proj_re = 0.0, Y_k0_after_proj_im = 0.0;
                    capture_post_conversion_device_state(&P, d_phi_r, d_eta_r, d_Y_r, d_xB_r,
                                                         plan_r2c_Y, d_Y_k, total_r, total_k,
                                                         NULL,
                                                         &mean_xB_after_proj,
                                                         NULL,
                                                         NULL,
                                                         &mean_Y_after_proj,
                                                         &Y_k0_after_proj_re,
                                                         &Y_k0_after_proj_im);
                    const double delta_Y_k0_proj =
                        Y_k0_after_proj_re - y_update_k0_audit_runtime.pre_Y_Y_k0_re;
                    const double rhs_k0_total_proj =
                        (P.dt > 0.0) ? (delta_Y_k0_proj / P.dt) : NAN;
                    write_y_update_k0_audit_row(
                        y_update_k0_audit_runtime.csv,
                        step,
                        audit_step_index,
                        "after_projection",
                        post_conversion_step,
                        sum_xBtot_before_Y,
                        sum_after_proj,
                        sum_after_proj - sum_xBtot_before_Y,
                        target_sum_for_projection,
                        y_update_k0_audit_runtime.pre_Y_Y_k0_re,
                        Y_k0_after_proj_re,
                        delta_Y_k0_proj,
                        rhs_k0_total_proj,
                        NAN, NAN, NAN, NAN, NAN, NAN, rhs_k0_total_proj,
                        y_update_k0_audit_runtime.pre_Y_mean_Y,
                        mean_Y_after_proj,
                        y_update_k0_audit_runtime.pre_Y_mean_xB,
                        mean_xB_after_proj,
                        y_update_k0_audit_runtime.pre_Y_min_xB,
                        y_update_k0_audit_runtime.pre_Y_max_xB,
                        min_xB_after_proj,
                        max_xB_after_proj,
                        mean_phi_now,
                        mean_eta_now,
                        mean_h_gp_now,
                        mean_h_beta_now,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        converged ? "projection_converged" : "projection_not_converged");
                    fflush(y_update_k0_audit_runtime.csv);
                }
            }
            y_update_k0_audit_runtime.post_conversion_step_index = post_conversion_step;
            if (y_update_k0_audit_runtime.active &&
                y_update_k0_audit_runtime.rows_remaining_steps > 0) {
                y_update_k0_audit_runtime.rows_remaining_steps -= 1;
                if (y_update_k0_audit_runtime.rows_remaining_steps <= 0) {
                    y_update_k0_audit_runtime.active = 0;
                }
            }
        }

        if (do_mass_diag) {
            const double inv_total = 1.0 / (double)total_r;
            if (!gp_mode_direct_storage) {
                cuDoubleComplex Y_k0_before = make_cuDoubleComplex(0.0, 0.0);
                cuDoubleComplex Y_rhs_k0 = make_cuDoubleComplex(0.0, 0.0);
                cuDoubleComplex Y_k0_after = make_cuDoubleComplex(0.0, 0.0);
                double y_rhs_stats[MASS_DIAG_Y_RHS_STATS_COUNT] = {0.0};
                CUDA_CHECK(cudaMemcpy(&Y_k0_before, d_divJ_k,
                                      sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&Y_rhs_k0, d_Y_rhs_k, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&Y_k0_after, d_Y_k, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(y_rhs_stats, d_mass_diag_Y_rhs_stats,
                                      MASS_DIAG_Y_RHS_STATS_COUNT * sizeof(double),
                                      cudaMemcpyDeviceToHost));
                mass_diag_row.Y_k0_before_re = cuCreal(Y_k0_before);
                mass_diag_row.Y_k0_before_im = cuCimag(Y_k0_before);
                mass_diag_row.Y_rhs_k0_re = cuCreal(Y_rhs_k0);
                mass_diag_row.Y_rhs_k0_im = cuCimag(Y_rhs_k0);
                mass_diag_row.Y_k0_after_re = cuCreal(Y_k0_after);
                mass_diag_row.Y_k0_after_im = cuCimag(Y_k0_after);
                mass_diag_row.delta_Y_k0_re = mass_diag_row.Y_k0_after_re - mass_diag_row.Y_k0_before_re;
                mass_diag_row.delta_Y_k0_im = mass_diag_row.Y_k0_after_im - mass_diag_row.Y_k0_before_im;
                mass_diag_row.mean_h_old_for_Y_rhs = y_rhs_stats[MASS_DIAG_Y_RHS_SUM_H_OLD] * inv_total;
                mass_diag_row.mean_h_new_for_Y_rhs = y_rhs_stats[MASS_DIAG_Y_RHS_SUM_H_NEW] * inv_total;
                mass_diag_row.mean_h_alpha_q_for_Y_rhs = y_rhs_stats[MASS_DIAG_Y_RHS_SUM_HQ] * inv_total;
                mass_diag_row.min_h_alpha_for_Y_rhs = y_rhs_stats[MASS_DIAG_Y_RHS_MIN_H];
                mass_diag_row.min_h_alpha_q_for_Y_rhs = y_rhs_stats[MASS_DIAG_Y_RHS_MIN_HQ];
                mass_diag_row.max_h_alpha_q_for_Y_rhs = y_rhs_stats[MASS_DIAG_Y_RHS_MAX_HQ];
                mass_diag_row.mean_gamma_local = y_rhs_stats[MASS_DIAG_Y_RHS_SUM_GAMMA_LOCAL] * inv_total;
                mass_diag_row.mean_term_h = y_rhs_stats[MASS_DIAG_Y_RHS_SUM_TERM_H] * inv_total;
                mass_diag_row.mean_term_lap = y_rhs_stats[MASS_DIAG_Y_RHS_SUM_TERM_LAP] * inv_total;
                mass_diag_row.mean_term_gamma = y_rhs_stats[MASS_DIAG_Y_RHS_SUM_TERM_GAMMA] * inv_total;
                mass_diag_row.mean_divJ = y_rhs_stats[MASS_DIAG_Y_RHS_SUM_DIVJ] * inv_total;
                mass_diag_row.mean_divJ_effective = mass_diag_row.mean_divJ;
                mass_diag_row.mean_lapY = y_rhs_stats[MASS_DIAG_Y_RHS_SUM_LAPY] * inv_total;
                mass_diag_row.mean_Y_rhs_total = y_rhs_stats[MASS_DIAG_Y_RHS_SUM_TOTAL] * inv_total;
                mass_diag_row.k0_divJ_re = y_rhs_stats[MASS_DIAG_Y_RHS_SUM_DIVJ];
                mass_diag_row.k0_divJ_im = 0.0;
                mass_diag_row.k0_term_h_re = y_rhs_stats[MASS_DIAG_Y_RHS_SUM_TERM_H];
                mass_diag_row.k0_term_h_im = 0.0;
                mass_diag_row.k0_term_lap_re = y_rhs_stats[MASS_DIAG_Y_RHS_SUM_TERM_LAP];
                mass_diag_row.k0_term_lap_im = 0.0;
                mass_diag_row.k0_term_gamma_re = y_rhs_stats[MASS_DIAG_Y_RHS_SUM_TERM_GAMMA];
                mass_diag_row.k0_term_gamma_im = 0.0;
                mass_diag_row.k0_Y_rhs_total_re = y_rhs_stats[MASS_DIAG_Y_RHS_SUM_TOTAL];
                mass_diag_row.k0_Y_rhs_total_im = 0.0;
                mass_diag_row.l1_divJ = y_rhs_stats[MASS_DIAG_Y_RHS_SUMABS_DIVJ] * inv_total;
                mass_diag_row.l1_term_h = y_rhs_stats[MASS_DIAG_Y_RHS_SUMABS_TERM_H] * inv_total;
                mass_diag_row.l1_term_lap = y_rhs_stats[MASS_DIAG_Y_RHS_SUMABS_TERM_LAP] * inv_total;
                mass_diag_row.l1_term_gamma = y_rhs_stats[MASS_DIAG_Y_RHS_SUMABS_TERM_GAMMA] * inv_total;
                mass_diag_row.l1_Y_rhs_total = y_rhs_stats[MASS_DIAG_Y_RHS_SUMABS_TOTAL] * inv_total;
                mass_diag_row.l2_divJ = sqrt(fmax(0.0, y_rhs_stats[MASS_DIAG_Y_RHS_SUMSQ_DIVJ] * inv_total));
                mass_diag_row.l2_term_h = sqrt(fmax(0.0, y_rhs_stats[MASS_DIAG_Y_RHS_SUMSQ_TERM_H] * inv_total));
                mass_diag_row.l2_term_lap = sqrt(fmax(0.0, y_rhs_stats[MASS_DIAG_Y_RHS_SUMSQ_TERM_LAP] * inv_total));
                mass_diag_row.l2_term_gamma = sqrt(fmax(0.0, y_rhs_stats[MASS_DIAG_Y_RHS_SUMSQ_TERM_GAMMA] * inv_total));
                mass_diag_row.l2_Y_rhs_total = sqrt(fmax(0.0, y_rhs_stats[MASS_DIAG_Y_RHS_SUMSQ_TOTAL] * inv_total));
                mass_diag_row.maxabs_divJ = y_rhs_stats[MASS_DIAG_Y_RHS_MAXABS_DIVJ];
                mass_diag_row.maxabs_term_h = y_rhs_stats[MASS_DIAG_Y_RHS_MAXABS_TERM_H];
                mass_diag_row.maxabs_term_lap = y_rhs_stats[MASS_DIAG_Y_RHS_MAXABS_TERM_LAP];
                mass_diag_row.maxabs_term_gamma = y_rhs_stats[MASS_DIAG_Y_RHS_MAXABS_TERM_GAMMA];
                mass_diag_row.maxabs_Y_rhs_total = y_rhs_stats[MASS_DIAG_Y_RHS_MAXABS_TOTAL];
                mass_diag_row.max_abs_fY = y_rhs_stats[MASS_DIAG_Y_RHS_MAXABS_TOTAL];
                mass_diag_row.max_abs_lagged_dYdt = y_rhs_stats[MASS_DIAG_Y_RHS_MAXABS_LAGGED_DYDT];
            } else {
                cuDoubleComplex Y_k0_before = make_cuDoubleComplex(0.0, 0.0);
                cuDoubleComplex Y_k0_after = make_cuDoubleComplex(0.0, 0.0);
                CUFFT_CHECK(cufftExecD2Z(plan_r2c_Y, d_Y_n_saved, d_Y_k));
                launch_dealias_kernel(d_Y_k, P.Nx, P.Ny, P.Nz, NzC,
                                     P.dx, P.dy, P.dz, total_k);
                CUDA_CHECK(cudaMemcpy(&Y_k0_before, d_Y_k, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                CUFFT_CHECK(cufftExecD2Z(plan_r2c_Y, d_Y_r, d_Y_k));
                launch_dealias_kernel(d_Y_k, P.Nx, P.Ny, P.Nz, NzC,
                                     P.dx, P.dy, P.dz, total_k);
                CUDA_CHECK(cudaMemcpy(&Y_k0_after, d_Y_k, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                mass_diag_row.Y_k0_before_re = cuCreal(Y_k0_before);
                mass_diag_row.Y_k0_before_im = cuCimag(Y_k0_before);
                mass_diag_row.Y_k0_after_re = cuCreal(Y_k0_after);
                mass_diag_row.Y_k0_after_im = cuCimag(Y_k0_after);
                mass_diag_row.delta_Y_k0_re = mass_diag_row.Y_k0_after_re - mass_diag_row.Y_k0_before_re;
                mass_diag_row.delta_Y_k0_im = mass_diag_row.Y_k0_after_im - mass_diag_row.Y_k0_before_im;
            }
            mass_diag_row.mean_xBtot_old = mass_diag_row.mean_xBtot_before_step;
        }

        // 最终 Y 诊断与接受；direct storage 模式走离散 storage closure 诊断
        {
            if (do_mass_diag) {
                const double inv_total = 1.0 / (double)total_r;
                if (gp_mode_direct_storage) {
                    double gp_storage_stats[MASS_DIAG_GP_STORAGE_STATS_COUNT];
                    double gp_update_stats[GP_Y_UPDATE_STATS_COUNT] = {0.0};
                    double xB_diff_stats[2] = {0.0, 0.0};
                    launch_gp_storage_diagnostics_kernel(
                        d_phi_n_saved, d_eta_prev_r, d_xB_old_diag_r ? d_xB_old_diag_r : d_xB_prev_r,
                        d_phi_r, d_eta_r, d_xB_r, d_divJ_r, P.dt, P.gp_xB_fixed,
                        P.gp_h_alpha_eps, d_mass_diag_gp_storage_stats, total_r);
                    CUDA_CHECK(cudaMemcpy(gp_storage_stats, d_mass_diag_gp_storage_stats,
                                          MASS_DIAG_GP_STORAGE_STATS_COUNT * sizeof(double),
                                          cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(gp_update_stats, d_gp_Y_update_stats,
                                          GP_Y_UPDATE_STATS_COUNT * sizeof(double),
                                          cudaMemcpyDeviceToHost));
                    if (d_xB_old_diag_r) {
                        CUDA_CHECK(cudaMemset(d_picard_diff_stats, 0, 2 * sizeof(double)));
                        launch_compute_diff_sumsq_max_kernel(d_xB_r, d_xB_old_diag_r,
                                                             d_picard_diff_stats, total_r);
                        CUDA_CHECK(cudaMemcpy(xB_diff_stats, d_picard_diff_stats,
                                              2 * sizeof(double), cudaMemcpyDeviceToHost));
                        mass_diag_row.max_abs_xB_alpha_delta = xB_diff_stats[1];
                    }
                    mass_diag_row.mean_xBtot_gp_before_Y =
                        gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_XBTOT_BEFORE] * inv_total;
                    mass_diag_row.mean_xB_alpha_before_Y =
                        gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_XB_ALPHA_BEFORE] * inv_total;
                    mass_diag_row.mean_storage_GP_before_Y =
                        gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_STORAGE_GP_BEFORE] * inv_total;
                    mass_diag_row.mean_storage_beta_before_Y =
                        gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_STORAGE_BETA_BEFORE] * inv_total;
                    mass_diag_row.mean_xBtot_after_Y_update =
                        gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_XBTOT_AFTER] * inv_total;
                    mass_diag_row.mean_xBtot_gp_after_Y_update = mass_diag_row.mean_xBtot_after_Y_update;
                    mass_diag_row.mean_xB_alpha_after_Y =
                        gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_XB_ALPHA_AFTER] * inv_total;
                    mass_diag_row.mean_xB_after_Y_update = mass_diag_row.mean_xB_alpha_after_Y;
                    mass_diag_row.mean_xB_after_clipping = mass_diag_row.mean_xB_alpha_after_Y;
                    mass_diag_row.mean_storage_GP_after_Y =
                        gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_STORAGE_GP_AFTER] * inv_total;
                    mass_diag_row.mean_storage_beta_after_Y =
                        gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_STORAGE_BETA_AFTER] * inv_total;
                    mass_diag_row.mean_delta_xB_alpha =
                        gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_DELTA_XB_ALPHA] * inv_total;
                    mass_diag_row.mean_delta_storage_GP =
                        gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_DELTA_STORAGE_GP] * inv_total;
                    mass_diag_row.mean_delta_storage_beta =
                        gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_DELTA_STORAGE_BETA] * inv_total;
                    mass_diag_row.mean_delta_xBtot_reconstructed =
                        mass_diag_row.mean_delta_xB_alpha +
                        mass_diag_row.mean_delta_storage_GP +
                        mass_diag_row.mean_delta_storage_beta;
                    mass_diag_row.mean_dt_divJ =
                        gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_DT_DIVJ] * inv_total;
                    mass_diag_row.mean_divJ =
                        gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_DIVJ] * inv_total;
                    mass_diag_row.mean_divJ_effective = mass_diag_row.mean_divJ;
                    mass_diag_row.gp_closure_error =
                        gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_CLOSURE_ERROR] * inv_total;
                    mass_diag_row.gp_small_h_alpha_count =
                        gp_storage_stats[MASS_DIAG_GP_STORAGE_SMALL_HALPHA_COUNT];
                    mass_diag_row.gp_min_h_alpha = gp_storage_stats[MASS_DIAG_GP_STORAGE_MIN_HALPHA];
                    mass_diag_row.gp_max_h_alpha = gp_storage_stats[MASS_DIAG_GP_STORAGE_MAX_HALPHA];
                    mass_diag_row.gp_min_h_GP = gp_storage_stats[MASS_DIAG_GP_STORAGE_MIN_HGP];
                    mass_diag_row.gp_max_h_GP = gp_storage_stats[MASS_DIAG_GP_STORAGE_MAX_HGP];
                    mass_diag_row.gp_min_eta = gp_storage_stats[MASS_DIAG_GP_STORAGE_MIN_ETA];
                    mass_diag_row.gp_max_eta = gp_storage_stats[MASS_DIAG_GP_STORAGE_MAX_ETA];
                    mass_diag_row.gp_min_xB_alpha = gp_storage_stats[MASS_DIAG_GP_STORAGE_MIN_XB_ALPHA];
                    mass_diag_row.gp_max_xB_alpha = gp_storage_stats[MASS_DIAG_GP_STORAGE_MAX_XB_ALPHA];
                    mass_diag_row.gp_clipping_mass_error =
                        gp_update_stats[GP_Y_UPDATE_SUM_CLIPPING_MASS_ERROR] * inv_total;
                    mass_diag_row.gp_small_h_alpha_mass_error =
                        gp_update_stats[GP_Y_UPDATE_SUM_SMALL_HALPHA_MASS_ERROR] * inv_total;
                    mass_diag_row.Y_clip_count_high = gp_update_stats[GP_Y_UPDATE_Y_CLIP_COUNT];
                    mass_diag_row.xB_clip_count_high = gp_update_stats[GP_Y_UPDATE_XB_CLIP_COUNT];
                    mass_diag_row.Y_clip_count_low = 0.0;
                    mass_diag_row.xB_clip_count_low = 0.0;
                    mass_diag_row.mean_xBtot_after_Y_to_xB = mass_diag_row.mean_xBtot_after_Y_update;
                    mass_diag_row.mean_xBtot_before_clipping = mass_diag_row.mean_xBtot_after_Y_update;
                    mass_diag_row.mean_xBtot_after_clipping = mass_diag_row.mean_xBtot_after_Y_update;
                    mass_diag_row.mean_xBtot_before_Y_clip = mass_diag_row.mean_xBtot_after_Y_update - mass_diag_row.gp_clipping_mass_error;
                    mass_diag_row.mean_xBtot_after_Y_clip = mass_diag_row.mean_xBtot_after_Y_update;
                    mass_diag_row.mean_xBtot_before_xB_clip = mass_diag_row.mean_xBtot_after_Y_update - mass_diag_row.gp_clipping_mass_error;
                    mass_diag_row.mean_xBtot_after_xB_clip = mass_diag_row.mean_xBtot_after_Y_update;
                    mass_diag_row.delta_mass_Y_clip = mass_diag_row.gp_clipping_mass_error;
                    mass_diag_row.delta_mass_xB_clip = 0.0;
                } else {
                    double y_stats[MASS_DIAG_Y_STATS_COUNT] = {0.0};
                    CUDA_CHECK(cudaMemset(d_mass_diag_Y_stats, 0, MASS_DIAG_Y_STATS_COUNT * sizeof(double)));
                    if (is_gp_zone_mode(&P)) {
                        launch_Y_mass_diagnostics_gp_kernel(
                            d_Y_r, d_phi_r, d_eta_r,
                            d_xB_old_diag_r ? d_xB_old_diag_r : d_xB_prev_r,
                            d_mass_diag_Y_stats, invN,
                            P.Y_clip, Y_upper_cap, P.xB_eps, P.gp_xB_fixed, total_r);
                    } else {
                        launch_Y_mass_diagnostics_kernel(
                            d_Y_r, d_phi_r, d_xB_old_diag_r ? d_xB_old_diag_r : d_xB_prev_r,
                            d_mass_diag_Y_stats, invN,
                            P.Y_clip, Y_upper_cap, P.xB_eps, P.v_B, total_r);
                    }
                    CUDA_CHECK(cudaMemcpy(y_stats, d_mass_diag_Y_stats,
                                          MASS_DIAG_Y_STATS_COUNT * sizeof(double),
                                          cudaMemcpyDeviceToHost));
                    mass_diag_row.mean_xBtot_after_Y_update =
                        y_stats[MASS_DIAG_Y_SUM_XBTOT_RAWY] * inv_total;
                    if (is_gp_zone_mode(&P)) {
                        mass_diag_row.mean_xBtot_gp_after_Y_update = mass_diag_row.mean_xBtot_after_Y_update;
                    }
                    mass_diag_row.mean_xBtot_after_Y_to_xB =
                        y_stats[MASS_DIAG_Y_SUM_XBTOT_YCLAMP_PRE_XBCLIP] * inv_total;
                    mass_diag_row.mean_xBtot_before_clipping = mass_diag_row.mean_xBtot_after_Y_to_xB;
                    mass_diag_row.mean_xBtot_before_Y_clip =
                        y_stats[MASS_DIAG_Y_SUM_XBTOT_RAWY] * inv_total;
                    mass_diag_row.mean_xBtot_after_Y_clip =
                        y_stats[MASS_DIAG_Y_SUM_XBTOT_YCLAMP_PRE_XBCLIP] * inv_total;
                    mass_diag_row.mean_xBtot_before_xB_clip =
                        y_stats[MASS_DIAG_Y_SUM_XBTOT_YCLAMP_PRE_XBCLIP] * inv_total;
                    mass_diag_row.mean_xBtot_after_xB_clip =
                        y_stats[MASS_DIAG_Y_SUM_XBTOT_FINAL] * inv_total;
                    mass_diag_row.mean_xB_after_Y_update =
                        y_stats[MASS_DIAG_Y_SUM_XB_RAWY] * inv_total;
                    mass_diag_row.mean_xB_after_clipping =
                        y_stats[MASS_DIAG_Y_SUM_XB_FINAL] * inv_total;
                    mass_diag_row.Y_clip_count_low = y_stats[MASS_DIAG_Y_CLIP_LOW_COUNT];
                    mass_diag_row.Y_clip_count_high = y_stats[MASS_DIAG_Y_CLIP_HIGH_COUNT];
                    mass_diag_row.xB_clip_count_low = y_stats[MASS_DIAG_Y_XB_CLIP_LOW_COUNT];
                    mass_diag_row.xB_clip_count_high = y_stats[MASS_DIAG_Y_XB_CLIP_HIGH_COUNT];
                    if (is_gp_zone_mode(&P)) {
                        double gp_storage_stats_init[MASS_DIAG_GP_STORAGE_STATS_COUNT];
                        double gp_storage_stats[MASS_DIAG_GP_STORAGE_STATS_COUNT];
                        init_gp_storage_stats_host(gp_storage_stats_init);
                        CUDA_CHECK(cudaMemcpy(d_mass_diag_gp_storage_stats, gp_storage_stats_init,
                                              MASS_DIAG_GP_STORAGE_STATS_COUNT * sizeof(double),
                                              cudaMemcpyHostToDevice));
                        launch_gp_storage_diagnostics_kernel(
                            d_phi_n_saved, d_eta_prev_r, d_xB_old_diag_r ? d_xB_old_diag_r : d_xB_prev_r,
                            d_phi_r, d_eta_r, d_xB_r, d_divJ_r, P.dt, P.gp_xB_fixed,
                            P.gp_h_alpha_eps, d_mass_diag_gp_storage_stats, total_r);
                        CUDA_CHECK(cudaMemcpy(gp_storage_stats, d_mass_diag_gp_storage_stats,
                                              MASS_DIAG_GP_STORAGE_STATS_COUNT * sizeof(double),
                                              cudaMemcpyDeviceToHost));
                        mass_diag_row.mean_xBtot_gp_before_Y =
                            gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_XBTOT_BEFORE] * inv_total;
                        mass_diag_row.mean_xB_alpha_before_Y =
                            gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_XB_ALPHA_BEFORE] * inv_total;
                        mass_diag_row.mean_storage_GP_before_Y =
                            gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_STORAGE_GP_BEFORE] * inv_total;
                        mass_diag_row.mean_storage_beta_before_Y =
                            gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_STORAGE_BETA_BEFORE] * inv_total;
                        mass_diag_row.mean_xB_alpha_after_Y =
                            gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_XB_ALPHA_AFTER] * inv_total;
                        mass_diag_row.mean_storage_GP_after_Y =
                            gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_STORAGE_GP_AFTER] * inv_total;
                        mass_diag_row.mean_storage_beta_after_Y =
                            gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_STORAGE_BETA_AFTER] * inv_total;
                        mass_diag_row.mean_delta_xB_alpha =
                            gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_DELTA_XB_ALPHA] * inv_total;
                        mass_diag_row.mean_delta_storage_GP =
                            gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_DELTA_STORAGE_GP] * inv_total;
                        mass_diag_row.mean_delta_storage_beta =
                            gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_DELTA_STORAGE_BETA] * inv_total;
                        mass_diag_row.mean_delta_xBtot_reconstructed =
                            mass_diag_row.mean_delta_xB_alpha +
                            mass_diag_row.mean_delta_storage_GP +
                            mass_diag_row.mean_delta_storage_beta;
                        mass_diag_row.mean_dt_divJ =
                            gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_DT_DIVJ] * inv_total;
                        mass_diag_row.gp_closure_error =
                            gp_storage_stats[MASS_DIAG_GP_STORAGE_SUM_CLOSURE_ERROR] * inv_total;
                        mass_diag_row.gp_small_h_alpha_count =
                            gp_storage_stats[MASS_DIAG_GP_STORAGE_SMALL_HALPHA_COUNT];
                        mass_diag_row.gp_min_h_alpha = gp_storage_stats[MASS_DIAG_GP_STORAGE_MIN_HALPHA];
                        mass_diag_row.gp_max_h_alpha = gp_storage_stats[MASS_DIAG_GP_STORAGE_MAX_HALPHA];
                        mass_diag_row.gp_min_h_GP = gp_storage_stats[MASS_DIAG_GP_STORAGE_MIN_HGP];
                        mass_diag_row.gp_max_h_GP = gp_storage_stats[MASS_DIAG_GP_STORAGE_MAX_HGP];
                        mass_diag_row.gp_min_eta = gp_storage_stats[MASS_DIAG_GP_STORAGE_MIN_ETA];
                        mass_diag_row.gp_max_eta = gp_storage_stats[MASS_DIAG_GP_STORAGE_MAX_ETA];
                        mass_diag_row.gp_min_xB_alpha = gp_storage_stats[MASS_DIAG_GP_STORAGE_MIN_XB_ALPHA];
                        mass_diag_row.gp_max_xB_alpha = gp_storage_stats[MASS_DIAG_GP_STORAGE_MAX_XB_ALPHA];
                    }
                }
                const double mean_xBtot_before_Y_stage =
                    (is_gp_zone_mode(&P) ? mass_diag_row.mean_xBtot_after_gp_to_beta_event
                                         : mass_diag_row.mean_xBtot_after_phi_update);
                mass_diag_row.delta_mass_Y_update =
                    mass_diag_row.mean_xBtot_after_Y_update - mean_xBtot_before_Y_stage;
                mass_diag_row.delta_mass_Y_to_xB =
                    mass_diag_row.mean_xBtot_after_Y_to_xB - mass_diag_row.mean_xBtot_after_Y_update;
                mass_diag_row.delta_mass_Y_clip =
                    mass_diag_row.mean_xBtot_after_Y_clip - mass_diag_row.mean_xBtot_before_Y_clip;
                mass_diag_row.delta_mass_xB_clip =
                    mass_diag_row.mean_xBtot_after_xB_clip - mass_diag_row.mean_xBtot_before_xB_clip;
                mass_diag_row.delta_M_phi =
                    mass_diag_row.mean_xBtot_after_phi_update - mass_diag_row.mean_xBtot_old;
                mass_diag_row.delta_M_Y_actual =
                    mass_diag_row.mean_xBtot_after_Y_update - mean_xBtot_before_Y_stage;
                mass_diag_row.delta_M_Y_required =
                    mass_diag_row.mean_xBtot_old - mean_xBtot_before_Y_stage;
                if (fabs(mass_diag_row.delta_M_Y_required) > 1.0e-30) {
                    mass_diag_row.Y_compensation_ratio =
                        mass_diag_row.delta_M_Y_actual / mass_diag_row.delta_M_Y_required;
                } else {
                    mass_diag_row.Y_compensation_ratio = 0.0;
                }
                mass_diag_row.picard_mean_xBtot_after_final = mass_diag_row.mean_xBtot_after_Y_update;
                mass_diag_row.picard_Y_compensation_ratio_final = mass_diag_row.Y_compensation_ratio;
                if (!use_Y_rhs_picard) {
                    mass_diag_row.picard_mean_xBtot_after_iter_0 = mass_diag_row.mean_xBtot_after_Y_update;
                }
                mass_diag_row.predicted_delta_M_from_plus_divJ = P.dt * mass_diag_row.mean_divJ_effective;
                mass_diag_row.predicted_delta_M_from_minus_divJ = -P.dt * mass_diag_row.mean_divJ_effective;
                mass_diag_row.conservative_residual_plus =
                    mass_diag_row.mean_xBtot_after_Y_update - mass_diag_row.mean_xBtot_old -
                    P.dt * mass_diag_row.mean_divJ_effective;
                mass_diag_row.conservative_residual_minus =
                    mass_diag_row.mean_xBtot_after_Y_update - mass_diag_row.mean_xBtot_old +
                    P.dt * mass_diag_row.mean_divJ_effective;
                mass_diag_row.conservative_balance_residual = mass_diag_row.conservative_residual_minus;
            }
            if (do_mass_diag) {
                mass_diag_row.mean_xBtot_after_clipping =
                    gpu_reduce_sum_model_xBtot(&P, d_phi_r, d_eta_r, d_xB_r, total_r) / (double)total_r;
                mass_diag_row.mean_xBtot_end_step = mass_diag_row.mean_xBtot_after_clipping;
                if (is_gp_zone_mode(&P)) {
                    mass_diag_row.mean_xBtot_gp_end_step = mass_diag_row.mean_xBtot_end_step;
                }
                mass_diag_row.delta_mass_clipping =
                    mass_diag_row.mean_xBtot_after_clipping - mass_diag_row.mean_xBtot_before_clipping;
                mass_diag_row.total_delta_mass_step =
                    mass_diag_row.mean_xBtot_end_step - mass_diag_row.mean_xBtot_before_step;
            }
        }

        // 2.15 更新 dY_dt_prev；Picard 模式接受最终 self-consistent guess，baseline 保持旧行为
        if (use_Y_rhs_picard) {
            CUDA_CHECK(cudaMemcpy(d_dY_dt_prev_r, d_dY_dt_picard_r, size_r, cudaMemcpyDeviceToDevice));
        } else {
            launch_update_dY_dt_prev_kernel(d_Y_r, d_Y_n_saved, d_dY_dt_prev_r, P.dt, total_r);
        }
        } // end if dynamics mode (Y update)

        if (do_phi_eta_field_snapshot) {
            CUDA_CHECK(cudaMemcpy(xB_after_step_host.data(), d_xB_r, size_r, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(Y_after_step_host.data(), d_Y_r, size_r, cudaMemcpyDeviceToHost));
        }
        if (do_phi_eta_step_delta_diag) {
            PhiEtaStepDeltaRow step_delta_row;
            memset(&step_delta_row, 0, sizeof(step_delta_row));
            step_delta_row.step = step;
            step_delta_row.time_code = step * P.dt;
            step_delta_row.dt = P.dt;
            compute_host_field_delta_stats(phi_before_step_host.data(),
                                           phi_after_phi_update_host.data(),
                                           total_r, &step_delta_row.phi,
                                           &step_delta_row.nan_inf_flag);
            compute_host_field_delta_stats(eta_before_step_host.data(),
                                           eta_after_eta_update_host.data(),
                                           total_r, &step_delta_row.eta,
                                           &step_delta_row.nan_inf_flag);
            compute_host_field_delta_stats(xB_before_step_host.data(),
                                           xB_after_step_host.data(),
                                           total_r, &step_delta_row.xB,
                                           &step_delta_row.nan_inf_flag);
            compute_host_field_basic_stats(Y_before_step_host.data(), total_r,
                                           &step_delta_row.Y_mean_before,
                                           &step_delta_row.Y_min_before,
                                           &step_delta_row.Y_max_before,
                                           &step_delta_row.nan_inf_flag);
            compute_host_field_basic_stats(Y_after_step_host.data(), total_r,
                                           &step_delta_row.Y_mean_after,
                                           &step_delta_row.Y_min_after,
                                           &step_delta_row.Y_max_after,
                                           &step_delta_row.nan_inf_flag);
            {
                double sum_abs_dY = 0.0;
                double max_abs_dY = 0.0;
                for (int idx = 0; idx < total_r; ++idx) {
                    const double db = Y_after_step_host[(size_t)idx] - Y_before_step_host[(size_t)idx];
                    const double adb = fabs(db);
                    sum_abs_dY += adb;
                    max_abs_dY = fmax(max_abs_dY, adb);
                }
                step_delta_row.dY_abs_mean = sum_abs_dY / fmax((double)total_r, 1.0);
                step_delta_row.dY_abs_max = max_abs_dY;
            }
            step_delta_row.xBtot_mean_before =
                is_gp_zone_mode(&P)
                    ? host_mean_xBtot_gp(phi_before_step_host.data(),
                                         eta_before_step_host.data(),
                                         xB_before_step_host.data(), &P, total_r)
                    : host_mean_xBtot_two_phase(phi_before_step_host.data(),
                                                xB_before_step_host.data(), &P, total_r);
            step_delta_row.xBtot_mean_after =
                is_gp_zone_mode(&P)
                    ? host_mean_xBtot_gp(phi_after_phi_update_host.data(),
                                         eta_after_eta_update_host.data(),
                                         xB_after_step_host.data(), &P, total_r)
                    : host_mean_xBtot_two_phase(phi_after_phi_update_host.data(),
                                                xB_after_step_host.data(), &P, total_r);
            step_delta_row.xBtot_delta =
                step_delta_row.xBtot_mean_after - step_delta_row.xBtot_mean_before;
            step_delta_row.xBtot_rel_delta =
                (fabs(step_delta_row.xBtot_mean_before) > 1.0e-30)
                    ? (step_delta_row.xBtot_delta / step_delta_row.xBtot_mean_before)
                    : NAN;
            {
                const double xB_low = P.xB_eps * 1.0000001;
                const double xB_high = 1.0 - P.xB_eps * 1.0000001;
                for (int idx = 0; idx < total_r; ++idx) {
                    const double xb = xB_after_step_host[(size_t)idx];
                    const double ph = phi_after_phi_update_host[(size_t)idx];
                    const double et = eta_after_eta_update_host[(size_t)idx];
                    if (xb <= xB_low) ++step_delta_row.xB_clip_low_count;
                    if (xb >= xB_high) ++step_delta_row.xB_clip_high_count;
                    if (ph < -1.0e-12 || ph > 1.0 + 1.0e-12) ++step_delta_row.phi_out_of_bounds_count;
                    if (et < -1.0e-12 || et > 1.0 + 1.0e-12) ++step_delta_row.eta_out_of_bounds_count;
                    if (!isfinite(xb) || !isfinite(ph) || !isfinite(et) ||
                        !isfinite(Y_after_step_host[(size_t)idx])) {
                        step_delta_row.nan_inf_flag = 1;
                    }
                }
            }
            {
                const int eta_peak_idx = host_argmax_field(eta_after_eta_update_host.data(), total_r,
                                                           &step_delta_row.nan_inf_flag);
                const int phi_peak_idx = host_argmax_field(phi_after_phi_update_host.data(), total_r,
                                                           &step_delta_row.nan_inf_flag);
                compute_far_field_stats_host(eta_after_eta_update_host.data(), total_r, &P, eta_peak_idx,
                                             &step_delta_row.eta_far_field_mean,
                                             &step_delta_row.eta_far_field_max,
                                             &step_delta_row.nan_inf_flag);
                compute_far_field_stats_host(phi_after_phi_update_host.data(), total_r, &P, phi_peak_idx,
                                             &step_delta_row.phi_far_field_mean,
                                             &step_delta_row.phi_far_field_max,
                                             &step_delta_row.nan_inf_flag);
                step_delta_row.eta_max_minus_far =
                    step_delta_row.eta.max_after - step_delta_row.eta_far_field_mean;
                step_delta_row.phi_max_minus_far =
                    step_delta_row.phi.max_after - step_delta_row.phi_far_field_mean;
            }
            step_delta_row.deta_abs_mean_over_dphi_abs_mean =
                (step_delta_row.phi.delta_abs_mean > 1.0e-300)
                    ? (step_delta_row.eta.delta_abs_mean / step_delta_row.phi.delta_abs_mean)
                    : NAN;
            step_delta_row.deta_abs_max_over_dphi_abs_max =
                (step_delta_row.phi.delta_abs_max > 1.0e-300)
                    ? (step_delta_row.eta.delta_abs_max / step_delta_row.phi.delta_abs_max)
                    : NAN;
            step_delta_row.deta_l2_over_dphi_l2 =
                (step_delta_row.phi.delta_l2 > 1.0e-300)
                    ? (step_delta_row.eta.delta_l2 / step_delta_row.phi.delta_l2)
                    : NAN;
            {
                const double kmax2 =
                    pow(M_PI / fmax(P.dx, 1.0e-30), 2.0) +
                    pow(M_PI / fmax(P.dy, 1.0e-30), 2.0) +
                    pow(M_PI / fmax(P.dz, 1.0e-30), 2.0);
                step_delta_row.S_phi_W = P.L_phi * P.dt * P.W;
                step_delta_row.S_phi_grad = P.L_phi * P.dt * P.kappa_phi * kmax2;
                step_delta_row.S_phi_total = step_delta_row.S_phi_W + step_delta_row.S_phi_grad;
                step_delta_row.S_eta_W = P.gp_L_eta * P.dt * P.gp_W_eta;
                step_delta_row.S_eta_grad = P.gp_L_eta * P.dt * P.gp_kappa_eta * kmax2;
                step_delta_row.S_eta_total = step_delta_row.S_eta_W + step_delta_row.S_eta_grad;
                step_delta_row.gp_L_eta = P.gp_L_eta;
                step_delta_row.L_phi = P.L_phi;
            }
            write_phi_eta_step_delta_csv_row(phi_eta_step_delta_fp, &step_delta_row);
            fflush(phi_eta_step_delta_fp);
        }
        if (do_phi_eta_rhs_attribution_diag) {
            PhiEtaRHSAttributionRow attr_row;
            memset(&attr_row, 0, sizeof(attr_row));
            attr_row.step = step;
            attr_row.time_code = step * P.dt;
            attr_row.dt = P.dt;

            compute_attribution_field_stats_host(phi_rhs_total_explicit_host.data(), total_r,
                                                 &attr_row.phi_rhs_total_explicit,
                                                 &attr_row.nan_inf_flag);
            compute_attribution_field_stats_host(phi_rhs_bulk_host.data(), total_r,
                                                 &attr_row.phi_rhs_chem,
                                                 &attr_row.nan_inf_flag);
            compute_attribution_field_stats_host(phi_rhs_dw_host.data(), total_r,
                                                 &attr_row.phi_rhs_double_well,
                                                 &attr_row.nan_inf_flag);
            compute_attribution_field_stats_host(phi_rhs_elastic_host.data(), total_r,
                                                 &attr_row.phi_rhs_elastic,
                                                 &attr_row.nan_inf_flag);
            compute_attribution_field_stats_host(eta_rhs_total_explicit_host.data(), total_r,
                                                 &attr_row.eta_rhs_total_explicit,
                                                 &attr_row.nan_inf_flag);
            compute_attribution_field_stats_host(eta_rhs_bulk_host.data(), total_r,
                                                 &attr_row.eta_rhs_bulk_or_chemical,
                                                 &attr_row.nan_inf_flag);
            compute_attribution_field_stats_host(eta_rhs_dw_host.data(), total_r,
                                                 &attr_row.eta_rhs_double_well,
                                                 &attr_row.nan_inf_flag);
            compute_attribution_field_stats_host(eta_rhs_elastic_host.data(), total_r,
                                                 &attr_row.eta_rhs_elastic,
                                                 &attr_row.nan_inf_flag);

            std::vector<double> phi_update_explicit_est_host((size_t)total_r, 0.0);
            std::vector<double> eta_update_explicit_est_host((size_t)total_r, 0.0);
            std::vector<double> dphi_actual_host((size_t)total_r, 0.0);
            std::vector<double> deta_actual_host((size_t)total_r, 0.0);
            std::vector<double> abs_deta_host((size_t)total_r, 0.0);
            std::vector<double> abs_dxB_host((size_t)total_r, 0.0);
            std::vector<double> abs_dY_host((size_t)total_r, 0.0);
            for (int idx = 0; idx < total_r; ++idx) {
                phi_update_explicit_est_host[(size_t)idx] =
                    -P.L_phi * dt_phi * phi_rhs_total_explicit_host[(size_t)idx];
                eta_update_explicit_est_host[(size_t)idx] =
                    -P.gp_L_eta * P.dt * eta_rhs_total_explicit_host[(size_t)idx];
                dphi_actual_host[(size_t)idx] =
                    phi_after_phi_update_host[(size_t)idx] - phi_before_step_host[(size_t)idx];
                deta_actual_host[(size_t)idx] =
                    eta_after_eta_update_host[(size_t)idx] - eta_before_step_host[(size_t)idx];
                abs_deta_host[(size_t)idx] = fabs(deta_actual_host[(size_t)idx]);
                abs_dxB_host[(size_t)idx] =
                    fabs(xB_after_step_host[(size_t)idx] - xB_before_step_host[(size_t)idx]);
                abs_dY_host[(size_t)idx] =
                    fabs(Y_after_step_host[(size_t)idx] - Y_before_step_host[(size_t)idx]);
            }
            compute_attribution_field_stats_host(phi_update_explicit_est_host.data(), total_r,
                                                 &attr_row.phi_update_explicit_est,
                                                 &attr_row.nan_inf_flag);
            compute_attribution_field_stats_host(eta_update_explicit_est_host.data(), total_r,
                                                 &attr_row.eta_update_explicit_est,
                                                 &attr_row.nan_inf_flag);
            compute_attribution_field_stats_host(dphi_actual_host.data(), total_r,
                                                 &attr_row.dphi_actual,
                                                 &attr_row.nan_inf_flag);
            compute_attribution_field_stats_host(deta_actual_host.data(), total_r,
                                                 &attr_row.deta_actual,
                                                 &attr_row.nan_inf_flag);

            compute_fourier_denom_stats_host(&P, P.L_phi, P.kappa_phi, dt_phi,
                                             &attr_row.phi_grad_denom_min,
                                             &attr_row.phi_grad_denom_max,
                                             &attr_row.phi_grad_denom_mean,
                                             &attr_row.phi_grad_denom_at_kmax);
            compute_fourier_denom_stats_host(&P, P.gp_L_eta, P.gp_kappa_eta, P.dt,
                                             &attr_row.eta_grad_denom_min,
                                             &attr_row.eta_grad_denom_max,
                                             &attr_row.eta_grad_denom_mean,
                                             &attr_row.eta_grad_denom_at_kmax);

            auto safe_ratio = [](double a, double b) {
                return (fabs(b) > 1.0e-300) ? (a / b) : NAN;
            };
            attr_row.eta_rhs_total_absmax_over_phi_rhs_total_absmax =
                safe_ratio(attr_row.eta_rhs_total_explicit.abs_max,
                           attr_row.phi_rhs_total_explicit.abs_max);
            attr_row.eta_rhs_bulk_absmax_over_phi_chem_absmax =
                safe_ratio(attr_row.eta_rhs_bulk_or_chemical.abs_max,
                           attr_row.phi_rhs_chem.abs_max);
            attr_row.eta_rhs_dw_absmax_over_phi_dw_absmax =
                safe_ratio(attr_row.eta_rhs_double_well.abs_max,
                           attr_row.phi_rhs_double_well.abs_max);
            attr_row.eta_rhs_elastic_absmax_over_phi_elastic_absmax =
                safe_ratio(attr_row.eta_rhs_elastic.abs_max,
                           attr_row.phi_rhs_elastic.abs_max);
            attr_row.eta_Ldt_rhs_absmax_over_phi_Ldt_rhs_absmax =
                safe_ratio(attr_row.eta_update_explicit_est.abs_max,
                           attr_row.phi_update_explicit_est.abs_max);
            attr_row.eta_Ldt_bulk_absmax_over_phi_Ldt_chem_absmax =
                safe_ratio(P.gp_L_eta * P.dt * attr_row.eta_rhs_bulk_or_chemical.abs_max,
                           P.L_phi * dt_phi * attr_row.phi_rhs_chem.abs_max);
            attr_row.eta_Ldt_dw_absmax_over_phi_Ldt_dw_absmax =
                safe_ratio(P.gp_L_eta * P.dt * attr_row.eta_rhs_double_well.abs_max,
                           P.L_phi * dt_phi * attr_row.phi_rhs_double_well.abs_max);
            attr_row.eta_Ldt_elastic_absmax_over_phi_Ldt_elastic_absmax =
                safe_ratio(P.gp_L_eta * P.dt * attr_row.eta_rhs_elastic.abs_max,
                           P.L_phi * dt_phi * attr_row.phi_rhs_elastic.abs_max);
            attr_row.deta_absmax_over_dphi_absmax =
                safe_ratio(attr_row.deta_actual.abs_max, attr_row.dphi_actual.abs_max);
            attr_row.deta_absmean_over_dphi_absmean =
                safe_ratio(attr_row.deta_actual.abs_mean, attr_row.dphi_actual.abs_mean);
            attr_row.deta_rms_over_dphi_rms =
                safe_ratio(attr_row.deta_actual.rms, attr_row.dphi_actual.rms);
            attr_row.phi_damping_stronger_than_eta =
                safe_ratio(attr_row.phi_grad_denom_at_kmax, attr_row.eta_grad_denom_at_kmax);
            attr_row.phi_actual_over_explicit_est_absmax =
                safe_ratio(attr_row.dphi_actual.abs_max, attr_row.phi_update_explicit_est.abs_max);
            attr_row.eta_actual_over_explicit_est_absmax =
                safe_ratio(attr_row.deta_actual.abs_max, attr_row.eta_update_explicit_est.abs_max);

            attr_row.corr_abs_deta_abs_dxB =
                compute_corr_host(abs_deta_host.data(), abs_dxB_host.data(), total_r,
                                  &attr_row.nan_inf_flag);
            attr_row.corr_abs_deta_abs_dY =
                compute_corr_host(abs_deta_host.data(), abs_dY_host.data(), total_r,
                                  &attr_row.nan_inf_flag);
            attr_row.corr_abs_deta_local_xB =
                compute_corr_host(abs_deta_host.data(), xB_before_step_host.data(), total_r,
                                  &attr_row.nan_inf_flag);
            attr_row.corr_abs_deta_eta_before =
                compute_corr_host(abs_deta_host.data(), eta_before_step_host.data(), total_r,
                                  &attr_row.nan_inf_flag);
            attr_row.corr_abs_deta_phi_before =
                compute_corr_host(abs_deta_host.data(), phi_before_step_host.data(), total_r,
                                  &attr_row.nan_inf_flag);

            {
                double max_abs_deta_value = 0.0;
                const int idx = host_argmax_abs_delta_field(eta_before_step_host.data(),
                                                            eta_after_eta_update_host.data(),
                                                            total_r, &max_abs_deta_value,
                                                            &attr_row.nan_inf_flag);
                attr_row.max_abs_deta_idx = idx;
                linear_index_to_ijk_host(&P, idx, &attr_row.max_abs_deta_i,
                                         &attr_row.max_abs_deta_j, &attr_row.max_abs_deta_k);
                attr_row.max_abs_deta_value = max_abs_deta_value;
                attr_row.max_abs_deta_eta_before = eta_before_step_host[(size_t)idx];
                attr_row.max_abs_deta_phi_before = phi_before_step_host[(size_t)idx];
                attr_row.max_abs_deta_xB_before = xB_before_step_host[(size_t)idx];
                attr_row.max_abs_deta_Y_before = Y_before_step_host[(size_t)idx];
                attr_row.max_abs_deta_eta_rhs_bulk = eta_rhs_bulk_host[(size_t)idx];
                attr_row.max_abs_deta_eta_rhs_dw = eta_rhs_dw_host[(size_t)idx];
                attr_row.max_abs_deta_eta_rhs_total = eta_rhs_total_explicit_host[(size_t)idx];
                attr_row.max_abs_deta_dxB =
                    xB_after_step_host[(size_t)idx] - xB_before_step_host[(size_t)idx];
            }
            {
                double max_abs_dphi_value = 0.0;
                const int idx = host_argmax_abs_delta_field(phi_before_step_host.data(),
                                                            phi_after_phi_update_host.data(),
                                                            total_r, &max_abs_dphi_value,
                                                            &attr_row.nan_inf_flag);
                attr_row.max_abs_dphi_idx = idx;
                linear_index_to_ijk_host(&P, idx, &attr_row.max_abs_dphi_i,
                                         &attr_row.max_abs_dphi_j, &attr_row.max_abs_dphi_k);
                attr_row.max_abs_dphi_value = max_abs_dphi_value;
                attr_row.max_abs_dphi_eta_before = eta_before_step_host[(size_t)idx];
                attr_row.max_abs_dphi_phi_before = phi_before_step_host[(size_t)idx];
                attr_row.max_abs_dphi_xB_before = xB_before_step_host[(size_t)idx];
                attr_row.max_abs_dphi_Y_before = Y_before_step_host[(size_t)idx];
                attr_row.max_abs_dphi_phi_rhs_chem = phi_rhs_bulk_host[(size_t)idx];
                attr_row.max_abs_dphi_phi_rhs_dw = phi_rhs_dw_host[(size_t)idx];
                attr_row.max_abs_dphi_phi_rhs_total = phi_rhs_total_explicit_host[(size_t)idx];
                attr_row.max_abs_dphi_dxB =
                    xB_after_step_host[(size_t)idx] - xB_before_step_host[(size_t)idx];
            }

            compute_host_field_basic_stats(xB_after_step_host.data(), total_r, NULL,
                                           &attr_row.xB_min_after, &attr_row.xB_max_after,
                                           &attr_row.nan_inf_flag);
            attr_row.xBtot_rel_delta =
                (is_gp_zone_mode(&P)
                    ? host_mean_xBtot_gp(phi_after_phi_update_host.data(),
                                         eta_after_eta_update_host.data(),
                                         xB_after_step_host.data(), &P, total_r)
                    : host_mean_xBtot_two_phase(phi_after_phi_update_host.data(),
                                                xB_after_step_host.data(), &P, total_r));
            {
                const double xBtot_mean_before =
                    is_gp_zone_mode(&P)
                        ? host_mean_xBtot_gp(phi_before_step_host.data(),
                                             eta_before_step_host.data(),
                                             xB_before_step_host.data(), &P, total_r)
                        : host_mean_xBtot_two_phase(phi_before_step_host.data(),
                                                    xB_before_step_host.data(), &P, total_r);
                const double xBtot_mean_after = attr_row.xBtot_rel_delta;
                attr_row.xBtot_rel_delta =
                    (fabs(xBtot_mean_before) > 1.0e-300)
                        ? ((xBtot_mean_after - xBtot_mean_before) / xBtot_mean_before)
                        : NAN;
            }
            {
                const int eta_peak_idx = host_argmax_field(eta_after_eta_update_host.data(), total_r,
                                                           &attr_row.nan_inf_flag);
                double eta_far_field_mean = NAN;
                compute_far_field_stats_host(eta_after_eta_update_host.data(), total_r, &P, eta_peak_idx,
                                             &eta_far_field_mean, &attr_row.eta_far_field_max,
                                             &attr_row.nan_inf_flag);
            }
            {
                const double xB_low = P.xB_eps * 1.0000001;
                const double xB_high = 1.0 - P.xB_eps * 1.0000001;
                for (int idx = 0; idx < total_r; ++idx) {
                    const double xb = xB_after_step_host[(size_t)idx];
                    if (xb <= xB_low) ++attr_row.xB_clip_low_count;
                    if (xb >= xB_high) ++attr_row.xB_clip_high_count;
                    if (!isfinite(xb) ||
                        !isfinite(phi_after_phi_update_host[(size_t)idx]) ||
                        !isfinite(eta_after_eta_update_host[(size_t)idx]) ||
                        !isfinite(Y_after_step_host[(size_t)idx])) {
                        attr_row.nan_inf_flag = 1;
                    }
                }
            }
            {
                const double kmax2 =
                    pow(M_PI / fmax(P.dx, 1.0e-30), 2.0) +
                    pow(M_PI / fmax(P.dy, 1.0e-30), 2.0) +
                    pow(M_PI / fmax(P.dz, 1.0e-30), 2.0);
                attr_row.S_phi_W = P.L_phi * dt_phi * P.W;
                attr_row.S_phi_grad = P.L_phi * dt_phi * P.kappa_phi * kmax2;
                attr_row.S_phi_total = attr_row.S_phi_W + attr_row.S_phi_grad;
                attr_row.S_eta_W = P.gp_L_eta * P.dt * P.gp_W_eta;
                attr_row.S_eta_grad = P.gp_L_eta * P.dt * P.gp_kappa_eta * kmax2;
                attr_row.S_eta_total = attr_row.S_eta_W + attr_row.S_eta_grad;
                attr_row.gp_L_eta = P.gp_L_eta;
                attr_row.L_phi = P.L_phi;
            }
            write_phi_eta_rhs_attribution_csv_row(phi_eta_rhs_attribution_fp, &attr_row);
            fflush(phi_eta_rhs_attribution_fp);
            if (eta_bulk_max_location_fp) {
                auto safe_ratio_local = [](double a, double b) {
                    return (fabs(b) > 1.0e-300) ? (a / b) : NAN;
                };
                double max_abs_bulk_value = 0.0;
                const int idx = host_argmax_abs_field(eta_rhs_bulk_host.data(), total_r,
                                                      &max_abs_bulk_value,
                                                      &attr_row.nan_inf_flag);
                int ii = 0, jj = 0, kk = 0;
                linear_index_to_ijk_host(&P, idx, &ii, &jj, &kk);
                const double eta_before = eta_before_step_host[(size_t)idx];
                const double phi_before = phi_before_step_host[(size_t)idx];
                const double xB_before = xB_before_step_host[(size_t)idx];
                const double Y_before = Y_before_step_host[(size_t)idx];
                const double h_eta = h_of_phi(eta_before);
                const double hp_eta = h_prime_of_phi(eta_before);
                const double h_phi = h_of_phi(phi_before);
                const double c_ref = 1.0 / Vm_alpha_of_xB(xB_before, P.Vm_alpha_0, P.dVm_alpha_dxB);
                const double gp_mu_ref_raw =
                    compute_gp_mu_reference_mech_mix_host(
                        P.gp_xB_fixed, temperature_K, P.gp_delta_g_stab);
                const double mu_PbTe_raw = mu_PbTe_calphad(temperature_K, xB_before);
                const double mu_Ag2Te_raw = mu_Ag2Te_calphad(temperature_K, xB_before);
                const double energy_scale =
                    (fabs(P.mu_reference_scale) > 1.0e-300) ? P.mu_reference_scale : 1.0;
                const double mu_PbTe_dimless = mu_PbTe_raw / energy_scale;
                const double mu_Ag2Te_dimless = mu_Ag2Te_raw / energy_scale;
                const double gp_mu_reference_dimless = gp_mu_ref_raw / energy_scale;
                const double drive_raw_formula_value =
                    P.gp_reaction_nu_A * mu_PbTe_raw +
                    P.gp_reaction_nu_B * mu_Ag2Te_raw -
                    gp_mu_ref_raw;
                const double drive_dimless_formula_value =
                    P.gp_reaction_nu_A * mu_PbTe_dimless +
                    P.gp_reaction_nu_B * mu_Ag2Te_dimless -
                    gp_mu_reference_dimless;
                const double dgbulk_deta_raw_formula_value =
                    c_ref * (1.0 - h_phi) * hp_eta * (-drive_raw_formula_value);
                const double dgbulk_deta_dimless_formula_value =
                    c_ref * (1.0 - h_phi) * hp_eta * (-drive_dimless_formula_value);
                const double dgbulk_deta_kernel_used =
                    eta_rhs_bulk_host[(size_t)idx];
                const double dxB = xB_after_step_host[(size_t)idx] - xB_before;
                const double deta = eta_after_eta_update_host[(size_t)idx] - eta_before;
                const double kmax2 =
                    pow(M_PI / fmax(P.dx, 1.0e-30), 2.0) +
                    pow(M_PI / fmax(P.dy, 1.0e-30), 2.0) +
                    pow(M_PI / fmax(P.dz, 1.0e-30), 2.0);
                const double S_eta_W = P.gp_L_eta * P.dt * P.gp_W_eta;
                const double S_eta_grad = P.gp_L_eta * P.dt * P.gp_kappa_eta * kmax2;
                const double S_eta_total = S_eta_W + S_eta_grad;
                const double S_phi_W = P.L_phi * dt_phi * P.W;
                const double S_phi_grad = P.L_phi * dt_phi * P.kappa_phi * kmax2;
                const double S_phi_total = S_phi_W + S_phi_grad;
                const char *region_class = "far_field";
                int is_eta_interface = 0;
                int is_far_field_like = 0;
                if (eta_before > 0.1 && eta_before < 0.9) {
                    region_class = "interface";
                    is_eta_interface = 1;
                } else if (eta_before >= 0.9) {
                    region_class = "core";
                } else {
                    is_far_field_like = 1;
                }
                const int is_phi_nonzero = (phi_before > 1.0e-8) ? 1 : 0;
                const int eta_peak_idx = host_argmax_field(eta_after_eta_update_host.data(), total_r,
                                                           &attr_row.nan_inf_flag);
                int eta_peak_i = 0, eta_peak_j = 0, eta_peak_k = 0;
                linear_index_to_ijk_host(&P, eta_peak_idx,
                                         &eta_peak_i, &eta_peak_j, &eta_peak_k);
                #define CSV_D(v) fprintf(eta_bulk_max_location_fp, "%.10e", (double)(v))
                #define CSV_I(v) fprintf(eta_bulk_max_location_fp, "%d", (int)(v))
                #define CSV_S(v) fprintf(eta_bulk_max_location_fp, "%s", (v))
                #define CSV_C() fputc(',', eta_bulk_max_location_fp)
                CSV_I(step); CSV_C(); CSV_D(step * P.dt); CSV_C(); CSV_D(P.dt); CSV_C();
                CSV_I(idx); CSV_C(); CSV_I(ii); CSV_C(); CSV_I(jj); CSV_C(); CSV_I(kk); CSV_C();
                CSV_D(max_abs_bulk_value); CSV_C(); CSV_D(eta_before); CSV_C(); CSV_D(phi_before); CSV_C(); CSV_D(xB_before); CSV_C(); CSV_D(Y_before); CSV_C(); CSV_D(h_eta); CSV_C(); CSV_D(hp_eta); CSV_C();
                CSV_D(mu_PbTe_raw); CSV_C(); CSV_D(mu_Ag2Te_raw); CSV_C(); CSV_D(gp_mu_ref_raw); CSV_C(); CSV_D(drive_raw_formula_value); CSV_C(); CSV_D(dgbulk_deta_raw_formula_value); CSV_C();
                CSV_D(energy_scale); CSV_C(); CSV_D(mu_PbTe_dimless); CSV_C(); CSV_D(mu_Ag2Te_dimless); CSV_C(); CSV_D(gp_mu_reference_dimless); CSV_C(); CSV_D(drive_dimless_formula_value); CSV_C(); CSV_D(dgbulk_deta_dimless_formula_value); CSV_C();
                CSV_D(dgbulk_deta_kernel_used); CSV_C();
                CSV_D(eta_rhs_bulk_host[(size_t)idx]); CSV_C(); CSV_D(eta_rhs_dw_host[(size_t)idx]); CSV_C(); CSV_D(eta_rhs_elastic_host[(size_t)idx]); CSV_C(); CSV_D(eta_rhs_total_explicit_host[(size_t)idx]); CSV_C(); CSV_D(dxB); CSV_C(); CSV_D(deta); CSV_C();
                CSV_D(attr_row.eta_rhs_bulk_or_chemical.abs_max); CSV_C(); CSV_D(attr_row.eta_rhs_double_well.abs_max); CSV_C(); CSV_D(attr_row.eta_rhs_elastic.abs_max); CSV_C(); CSV_D(attr_row.eta_rhs_total_explicit.abs_max); CSV_C();
                CSV_D(phi_rhs_bulk_host[(size_t)idx]); CSV_C(); CSV_D(phi_rhs_total_explicit_host[(size_t)idx]); CSV_C(); CSV_D(phi_rhs_dw_host[(size_t)idx]); CSV_C(); CSV_D(phi_rhs_elastic_host[(size_t)idx]); CSV_C();
                CSV_D(attr_row.phi_rhs_chem.abs_max); CSV_C(); CSV_D(attr_row.phi_rhs_double_well.abs_max); CSV_C(); CSV_D(attr_row.phi_rhs_elastic.abs_max); CSV_C(); CSV_D(attr_row.phi_rhs_total_explicit.abs_max); CSV_C();
                CSV_D(safe_ratio_local(attr_row.eta_rhs_bulk_or_chemical.abs_max, attr_row.phi_rhs_chem.abs_max)); CSV_C();
                CSV_D(safe_ratio_local(P.gp_L_eta * P.dt * attr_row.eta_rhs_bulk_or_chemical.abs_max,
                                       P.L_phi * dt_phi * attr_row.phi_rhs_chem.abs_max)); CSV_C();
                CSV_D(P.gp_W_eta); CSV_C(); CSV_D(P.gp_kappa_eta); CSV_C(); CSV_D(P.gp_L_eta); CSV_C(); CSV_D(S_eta_W); CSV_C(); CSV_D(S_eta_grad); CSV_C(); CSV_D(S_eta_total); CSV_C();
                CSV_D(P.W); CSV_C(); CSV_D(P.kappa_phi); CSV_C(); CSV_D(P.L_phi); CSV_C(); CSV_D(S_phi_W); CSV_C(); CSV_D(S_phi_grad); CSV_C(); CSV_D(S_phi_total); CSV_C();
                CSV_S(region_class); CSV_C(); CSV_I(is_eta_interface); CSV_C(); CSV_I(is_far_field_like); CSV_C(); CSV_I(is_phi_nonzero); CSV_C();
                CSV_I(eta_peak_idx); CSV_C(); CSV_I(eta_peak_i); CSV_C(); CSV_I(eta_peak_j); CSV_C(); CSV_I(eta_peak_k); CSV_C(); CSV_D(eta_after_eta_update_host[(size_t)eta_peak_idx]);
                fputc('\n', eta_bulk_max_location_fp);
                #undef CSV_D
                #undef CSV_I
                #undef CSV_S
                #undef CSV_C
                fflush(eta_bulk_max_location_fp);
            }
        }

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
                        d_phi_r, d_eta_r, d_xB_r, d_phi_rhs_r,
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
                        is_gp_zone_mode(&P) ? 1 : 0,
                        gp_elastic_solver_enabled,
                        gp_active_phi_coupling_enabled,
                        P.gp_eps_iso,
                        P.gp_elastic_derivative_scale,
                        (phi_elastic_coupling_enabled ? P.elastic_shift_dimless : 0.0),
                        /* disable_chem = */ 0,
                        total_r,
                        phi_elastic_coupling_enabled,
                        NULL);
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
        // 6.6) 体积分数投影（minimize 模式）：显式将 <h(phi)> 投影回 V0_target
        //
        // 说明：
        // - 上面的 lambda_vol 计算不显式包含 V0_target，本质上不能保证 <h> 精确回到目标体积分数；
        // - 对于小核子（V0 很小）会表现为 vol_err_rel 长时间偏大，导致无法触发停止条件；
        // - 这里使用一阶修正：phi <- phi + lambda_correct * h'(phi)
        //   其中 lambda_correct = (V0_target - <h>) / <(h')^2>
        // - 该 kernel 已在 cuda_kernels.cu 中实现（apply_volume_projection_kernel）。
        // -----------------------------------------------------------------
        if (P.mode == 1) {
            const double vol_tiny = 1e-30;
            if (V0_target > vol_tiny && isfinite(mean_h_now)) {
                double vol_err = V0_target - mean_h_now;
                if (fabs(vol_err) > 1e-14) {
                    launch_compute_hprime_sq_values_kernel(d_phi_r, d_energy_tmp, total_r);
                    double sum_hp2_proj = gpu_reduce_sum(d_energy_tmp, total_r);
                    double mean_hp2_proj = sum_hp2_proj / (double)total_r;
                    double lambda_correct = (mean_hp2_proj > 1e-30) ? (vol_err / mean_hp2_proj) : 0.0;
                    launch_apply_volume_projection_kernel(d_phi_r, lambda_correct, total_r);
                    // 投影后重算 mean_h，供后续收敛/CSV 使用
                    mean_h_now = gpu_compute_vf_from_h(d_phi_r, total_r);
                }
            }
        }
        
        // -----------------------------------------------------------------
        // 7) 收敛判据（minimize 模式）：解耦步长控制与停止条件
        //    注意：此处的 rms_dY 使用“本次迭代 Y 更新后的场”计算，
        //    因此必须放在 Y 半隐式更新 (2.13~2.15) 之后。
        // -----------------------------------------------------------------
        if (P.mode == 1) {
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
                fprintf(energy_fp,
                        "%d,%.8e,%.8e,%.8e,%.8e,"
                        "%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,"
                        "%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%d\n",
                        step + energy_iter_offset,
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
                        P.minimize_post_projection_iters);
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
                double mean_xB_tot = gpu_reduce_sum_model_xBtot(&P, d_phi_r, d_eta_r, d_xB_r, total_r) / (double)total_r;
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
                        step + csv_step_offset, current_time + csv_time_offset, t_real + csv_real_time_offset, vf_precip, R_avg,
                        sum_h_csv, sum_gel_csv, E_hat_csv, E_Jm3_csv, dmu_csv);
            } else {
                fprintf(csv_fp, "%d,%.8f,%.8e,%.8e,%.8e\n",
                        step + csv_step_offset, current_time + csv_time_offset, t_real + csv_real_time_offset, vf_precip, R_avg);
            }
                fflush(csv_fp);
            }
            if (relax_fp && P.mode == 0) {
                write_relaxation_diag_row(relax_fp,
                                          step + csv_step_offset,
                                          current_time + csv_time_offset,
                                          t_real + csv_real_time_offset,
                                          d_phi_r,
                                          d_eta_r,
                                          d_xB_r,
                                          &P,
                                          total_r,
                                          relaxation_initial_mean_xBtot);
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
                launch_compute_model_xBtot_kernel(&P, d_phi_r, d_eta_r, d_xB_r, d_xBtot_temp, total_r);
                build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir,
                                    xBtot_output_stem(&P), VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                ret1 = write_vtk_cuda(d_xBtot_temp, P.Nx, P.Ny, P.Nz, xBtot_vtk_field_name(&P), output_step, filename);
                if (!ret1) fprintf(stderr, "ERROR: Failed to write %s VTK file: %s\n", xBtot_output_stem(&P), filename);
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
            if (is_gp_zone_mode(&P)) {
                build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "eta", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                int ret_eta = write_vtk_cuda(d_eta_r, P.Nx, P.Ny, P.Nz, "eta", output_step, filename);
                if (!ret_eta) {
                    fprintf(stderr, "ERROR: Failed to write eta VTK file: %s\n", filename);
                }
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
                    P.eps_iso_over_vB, (phi_elastic_coupling_enabled ? P.elastic_shift_dimless : 0.0),
                    total_r, phi_elastic_coupling_enabled);
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
                    (phi_elastic_coupling_enabled ? P.elastic_shift_dimless : 0.0),
                    d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                    d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                    d_uxx_r, d_uyy_r, d_uzz_r, d_uxy_r, d_uxz_r, d_uyz_r,
                    // Optimization(4): d_uxx0_r..d_uyz0_r 参数已移除
                    P.S_p_11, P.S_p_12, P.S_p_13, P.S_p_14, P.S_p_15, P.S_p_16,
                    P.S_p_22, P.S_p_23, P.S_p_24, P.S_p_25, P.S_p_26,
                    P.S_p_33, P.S_p_34, P.S_p_35, P.S_p_36,
                    P.S_p_44, P.S_p_45, P.S_p_46, P.S_p_55, P.S_p_56, P.S_p_66,
                    P.eps_xx00, P.eps_yy00, P.eps_zz00, P.eps_yz00, P.eps_xz00, P.eps_xy00,
                    P.eps_iso_over_vB, total_r, phi_elastic_coupling_enabled);
                build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "driving_force", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                int ret10 = write_vtk_cuda(scratch1, P.Nx, P.Ny, P.Nz, "driving_force", output_step, filename);
                if (!ret10) fprintf(stderr, "ERROR: Failed to write driving_force VTK file: %s\n", filename);

                if (is_gp_zone_mode(&P)) {
                    const double gp_mu_ref_raw =
                        compute_gp_mu_reference_mech_mix_host(
                            P.gp_xB_fixed, temperature_K, P.gp_delta_g_stab);
                    launch_compute_eta_rhs_components_kernel(
                        d_eta_r, d_phi_r, d_xB_r,
                        d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                        d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                        scratch0, scratch1, d_eta_rhs_r, d_phi_rhs_r, NULL,
                        temperature_K, P.mu_reference_scale,
                        P.Vm_alpha_0, P.dVm_alpha_dxB,
                        P.gp_xB_fixed, P.gp_delta_g0,
                        P.gp_raw_reaction_drive_only,
                        P.gp_raw_reaction_drive_use_raw_units_debug,
                        P.gp_reaction_nu_A, P.gp_reaction_nu_B,
                        gp_mu_ref_raw,
                        P.gp_W_eta, P.eps_iso_over_vB,
                        gp_elastic_solver_enabled,
                        gp_active_eta_coupling_enabled,
                        P.gp_eps_iso,
                        P.gp_elastic_derivative_scale,
                        total_r);
                    build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "eta_rhs_chem", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                    int ret_eta_chem = write_vtk_cuda(scratch0, P.Nx, P.Ny, P.Nz, "eta_rhs_chem", output_step, filename);
                    if (!ret_eta_chem) fprintf(stderr, "ERROR: Failed to write eta_rhs_chem VTK file: %s\n", filename);
                    build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "eta_rhs_dw", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                    int ret_eta_dw = write_vtk_cuda(scratch1, P.Nx, P.Ny, P.Nz, "eta_rhs_dw", output_step, filename);
                    if (!ret_eta_dw) fprintf(stderr, "ERROR: Failed to write eta_rhs_dw VTK file: %s\n", filename);
                    build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "eta_rhs_elastic", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                    int ret_eta_elastic = write_vtk_cuda(d_eta_rhs_r, P.Nx, P.Ny, P.Nz, "eta_rhs_elastic", output_step, filename);
                    if (!ret_eta_elastic) fprintf(stderr, "ERROR: Failed to write eta_rhs_elastic VTK file: %s\n", filename);
                    build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "eta_rhs_net_explicit", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                    int ret_eta_net = write_vtk_cuda(d_phi_rhs_r, P.Nx, P.Ny, P.Nz, "eta_rhs_net_explicit", output_step, filename);
                    if (!ret_eta_net) fprintf(stderr, "ERROR: Failed to write eta_rhs_net_explicit VTK file: %s\n", filename);

                    CUFFT_CHECK(cufftExecD2Z(plan_r2c_phi, d_eta_r, d_eta_k));
                    launch_dealias_kernel(d_eta_k, P.Nx, P.Ny, P.Nz, NzC,
                                         P.dx, P.dy, P.dz, total_k);
                    launch_compute_laplacian_k_kernel(d_eta_k, KS.d_k2, d_eta_rhs_k, total_k);
                    launch_dealias_kernel(d_eta_rhs_k, P.Nx, P.Ny, P.Nz, NzC,
                                         P.dx, P.dy, P.dz, total_k);
                    CUFFT_CHECK(cufftExecZ2D(plan_c2r_phi, d_eta_rhs_k, scratch0));
                    launch_normalize_only_kernel(scratch0, invN, total_r);
                    launch_scale_array_kernel(scratch0, -P.gp_kappa_eta, total_r);
                    build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "eta_rhs_grad", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                    int ret_eta_grad = write_vtk_cuda(scratch0, P.Nx, P.Ny, P.Nz, "eta_rhs_grad", output_step, filename);
                    if (!ret_eta_grad) fprintf(stderr, "ERROR: Failed to write eta_rhs_grad VTK file: %s\n", filename);
                    launch_add_arrays_kernel(d_phi_rhs_r, scratch0, scratch1, total_r);
                    build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "eta_rhs_full_variational", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                    int ret_eta_full = write_vtk_cuda(scratch1, P.Nx, P.Ny, P.Nz, "eta_rhs_full_variational", output_step, filename);
                    if (!ret_eta_full) fprintf(stderr, "ERROR: Failed to write eta_rhs_full_variational VTK file: %s\n", filename);
                    if (P.gp_raw_reaction_drive_only) {
                        launch_compute_eta_rhs_components_kernel(
                            d_eta_r, d_phi_r, d_xB_r,
                            d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                            d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                            NULL, NULL, NULL, NULL, scratch0,
                            temperature_K, P.mu_reference_scale,
                            P.Vm_alpha_0, P.dVm_alpha_dxB,
                            P.gp_xB_fixed, P.gp_delta_g0,
                            P.gp_raw_reaction_drive_only,
                            P.gp_raw_reaction_drive_use_raw_units_debug,
                            P.gp_reaction_nu_A, P.gp_reaction_nu_B,
                            gp_mu_ref_raw,
                            P.gp_W_eta, P.eps_iso_over_vB,
                            gp_elastic_solver_enabled,
                            gp_active_eta_coupling_enabled,
                            P.gp_eps_iso,
                            P.gp_elastic_derivative_scale,
                            total_r);
                        build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "gp_minus_delta_mu_r", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                        int ret_gp_minus_dmu = write_vtk_cuda(scratch0, P.Nx, P.Ny, P.Nz, "gp_minus_delta_mu_r", output_step, filename);
                        if (!ret_gp_minus_dmu) fprintf(stderr, "ERROR: Failed to write gp_minus_delta_mu_r VTK file: %s\n", filename);
                    }
                }

                // === 弹性诊断模式：输出弹性能密度 gel 的VTK分布 ===
                if (P.diag_elastic_bulk_penalty_enabled && P.elastic_enabled) {
                    // Optimization: gel 复用 scratch slot 0
                    double *d_gel_temp = (double *)d_scratch_r_double;
                    launch_compute_gel_density_kernel(
                        d_uxx_r, d_uyy_r, d_uzz_r,
                        d_uxy_r, d_uxz_r, d_uyz_r,
                        d_phi_r,
                        d_eta_r,
                        (P.mode == 1) ? NULL : d_xB_r,  // Optimization(4): minimize 模式 xB_r 为 NULL
                        d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                        d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                        (float)P.eps_xx00, (float)P.eps_yy00, (float)P.eps_zz00,
                        (float)P.eps_yz00, (float)P.eps_xz00, (float)P.eps_xy00,
                        (double)P.eps_iso_over_vB,
                        is_gp_zone_mode(&P) ? 1 : 0,
                        gp_elastic_solver_enabled,
                        P.gp_eps_iso,
                        d_gel_temp,
                        total_r);
                    build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "gel", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                    int ret11 = write_vtk_cuda(d_gel_temp, P.Nx, P.Ny, P.Nz, "gel", output_step, filename);
                    if (!ret11) fprintf(stderr, "ERROR: Failed to write gel VTK file: %s\n", filename);
                    
                    if (ret1 && ret2 && ret3 && ret4 && ret5 && ret6 && ret7 && ret8 && ret9 && ret10 && ret11) {
                        printf("step=%05d 完成，已输出VTK文件（phi_%d.vtk, xB_%d.vtk, %s_%d.vtk, delta_mu_r_%d.vtk, f_phi_chem_%d.vtk, dgel_dphi_%d.vtk, f_phi_dw_%d.vtk, f_phi_bulk_%d.vtk, f_phi_grad_%d.vtk, driving_force_%d.vtk, gel_%d.vtk）\n", 
                               step, output_step, output_step, xBtot_output_stem(&P), output_step, output_step, output_step, output_step,
                               output_step, output_step, output_step, output_step, output_step);
                    } else {
                        fprintf(stderr, "step=%05d 警告：诊断VTK文件输出失败！\n", step);
                    }
                } else {
                    // 诊断模式但未启用弹性bulk诊断，或弹性未启用
                    if (ret1 && ret2 && ret3 && ret4 && ret5 && ret6 && ret7 && ret8 && ret9 && ret10) {
                        printf("step=%05d 完成，已输出VTK文件（phi_%d.vtk, xB_%d.vtk, %s_%d.vtk, delta_mu_r_%d.vtk, f_phi_chem_%d.vtk, dgel_dphi_%d.vtk, f_phi_dw_%d.vtk, f_phi_bulk_%d.vtk, f_phi_grad_%d.vtk, driving_force_%d.vtk）\n", 
                               step, output_step, output_step, xBtot_output_stem(&P), output_step, output_step, output_step, output_step,
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
                        printf("step=%05d 完成，已输出VTK文件（phi_%d.vtk, xB_%d.vtk, %s_%d.vtk）\n",
                               step, output_step, output_step, xBtot_output_stem(&P), output_step);
                    } else {
                        fprintf(stderr, "step=%05d 警告：VTK文件输出失败！\n", step);
                    }
                }
            }
        }

        if (do_mass_diag) {
            if (mass_diag_fp) {
                write_mass_diag_row_csv(mass_diag_fp, &mass_diag_row, &P);
                fflush(mass_diag_fp);
            }
            if (post_conversion_audit_runtime.csv && post_conversion_audit_runtime.active &&
                post_conversion_audit_runtime.rows_remaining_steps > 0) {
                double prev_sum = post_conversion_audit_runtime.baseline_sum_xBtot;
                double current_min_xB = 0.0;
                double current_max_xB = 0.0;
                gpu_reduce_min_max(d_xB_r, total_r, &current_min_xB, &current_max_xB);
                write_post_conversion_y_audit_row(
                    post_conversion_audit_runtime.csv, step, "before_step",
                    post_conversion_audit_runtime.before_step_sum_xBtot,
                    prev_sum,
                    post_conversion_audit_runtime.baseline_sum_xBtot,
                    post_conversion_audit_runtime.before_step_sum_xBtot / (double)total_r,
                    post_conversion_audit_runtime.before_step_mean_xB,
                    post_conversion_audit_runtime.before_step_min_xB,
                    post_conversion_audit_runtime.before_step_max_xB,
                    post_conversion_audit_runtime.before_step_mean_Y,
                    post_conversion_audit_runtime.before_step_Y_k0_re,
                    post_conversion_audit_runtime.before_step_Y_k0_re,
                    0.0,
                    0.0,
                    NAN,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0);
                prev_sum = post_conversion_audit_runtime.before_step_sum_xBtot;
                write_post_conversion_y_audit_row(
                    post_conversion_audit_runtime.csv, step, "after_phi_update",
                    post_conversion_audit_runtime.pre_Y_sum_xBtot,
                    prev_sum,
                    post_conversion_audit_runtime.baseline_sum_xBtot,
                    post_conversion_audit_runtime.pre_Y_sum_xBtot / (double)total_r,
                    post_conversion_audit_runtime.pre_Y_mean_xB,
                    post_conversion_audit_runtime.pre_Y_min_xB,
                    post_conversion_audit_runtime.pre_Y_max_xB,
                    post_conversion_audit_runtime.pre_Y_mean_Y,
                    post_conversion_audit_runtime.pre_Y_Y_k0_re,
                    post_conversion_audit_runtime.pre_Y_Y_k0_re,
                    0.0,
                    0.0,
                    NAN,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0);
                prev_sum = post_conversion_audit_runtime.pre_Y_sum_xBtot;
                const double sum_after_Y = mass_diag_row.mean_xBtot_after_Y_update * (double)total_r;
                const double sum_after_Y_to_xB = mass_diag_row.mean_xBtot_after_Y_to_xB * (double)total_r;
                const double sum_after_clipping = mass_diag_row.mean_xBtot_after_clipping * (double)total_r;
                const double sum_end = mass_diag_row.mean_xBtot_end_step * (double)total_r;
                const double y_update_k0_drift = mass_diag_row.delta_mass_Y_update * (double)total_r;
                const double clipped_mass_loss = mass_diag_row.delta_mass_clipping * (double)total_r;
                write_post_conversion_y_audit_row(
                    post_conversion_audit_runtime.csv, step, "after_Y_update",
                    sum_after_Y,
                    prev_sum,
                    post_conversion_audit_runtime.baseline_sum_xBtot,
                    mass_diag_row.mean_xBtot_after_Y_update,
                    mass_diag_row.mean_xB_after_Y_update,
                    mass_diag_row.gp_min_xB_alpha,
                    mass_diag_row.gp_max_xB_alpha,
                    mass_diag_row.Y_k0_after_re / (double)total_r,
                    mass_diag_row.Y_k0_before_re,
                    mass_diag_row.Y_k0_after_re,
                    mass_diag_row.delta_Y_k0_re,
                    y_update_k0_drift,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0);
                prev_sum = sum_after_Y;
                write_post_conversion_y_audit_row(
                    post_conversion_audit_runtime.csv, step, "after_Y_to_xB",
                    sum_after_Y_to_xB,
                    prev_sum,
                    post_conversion_audit_runtime.baseline_sum_xBtot,
                    mass_diag_row.mean_xBtot_after_Y_to_xB,
                    mass_diag_row.mean_xB_after_Y_update,
                    mass_diag_row.gp_min_xB_alpha,
                    mass_diag_row.gp_max_xB_alpha,
                    mass_diag_row.Y_k0_after_re / (double)total_r,
                    mass_diag_row.Y_k0_before_re,
                    mass_diag_row.Y_k0_after_re,
                    mass_diag_row.delta_Y_k0_re,
                    y_update_k0_drift,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0);
                prev_sum = sum_after_Y_to_xB;
                write_post_conversion_y_audit_row(
                    post_conversion_audit_runtime.csv, step, "after_clipping",
                    sum_after_clipping,
                    prev_sum,
                    post_conversion_audit_runtime.baseline_sum_xBtot,
                    mass_diag_row.mean_xBtot_after_clipping,
                    mass_diag_row.mean_xB_after_clipping,
                    current_min_xB,
                    current_max_xB,
                    mass_diag_row.Y_k0_after_re / (double)total_r,
                    mass_diag_row.Y_k0_before_re,
                    mass_diag_row.Y_k0_after_re,
                    mass_diag_row.delta_Y_k0_re,
                    y_update_k0_drift,
                    0.0,
                    clipped_mass_loss,
                    mass_diag_row.xB_clip_count_low,
                    mass_diag_row.xB_clip_count_high,
                    0.0,
                    0.0);
                prev_sum = sum_after_clipping;
                write_post_conversion_y_audit_row(
                    post_conversion_audit_runtime.csv, step, "end_of_step",
                    sum_end,
                    prev_sum,
                    post_conversion_audit_runtime.baseline_sum_xBtot,
                    mass_diag_row.mean_xBtot_end_step,
                    mass_diag_row.mean_xB_after_clipping,
                    current_min_xB,
                    current_max_xB,
                    mass_diag_row.Y_k0_after_re / (double)total_r,
                    mass_diag_row.Y_k0_before_re,
                    mass_diag_row.Y_k0_after_re,
                    mass_diag_row.delta_Y_k0_re,
                    y_update_k0_drift,
                    0.0,
                    clipped_mass_loss,
                    mass_diag_row.xB_clip_count_low,
                    mass_diag_row.xB_clip_count_high,
                    0.0,
                    0.0);
                fflush(post_conversion_audit_runtime.csv);
                post_conversion_audit_runtime.rows_remaining_steps -= 1;
                if (post_conversion_audit_runtime.rows_remaining_steps <= 0) {
                    post_conversion_audit_runtime.active = 0;
                }
            }
            mass_diag_summary.final_mean_xBtot = mass_diag_row.mean_xBtot_end_step;
            mass_diag_summary.total_absolute_drift =
                mass_diag_summary.final_mean_xBtot - mass_diag_summary.initial_mean_xBtot;
            mass_diag_summary.total_relative_drift =
                mass_diag_summary.total_absolute_drift /
                fmax(fabs(mass_diag_summary.initial_mean_xBtot), 1.0e-30);
            mass_diag_summary.cumulative_delta_phi_update += mass_diag_row.delta_mass_phi_update;
            mass_diag_summary.cumulative_delta_eta_update += mass_diag_row.delta_mass_eta_update;
            mass_diag_summary.cumulative_delta_gp_to_beta_event += mass_diag_row.delta_mass_gp_to_beta_event;
            mass_diag_summary.cumulative_delta_Y_update += mass_diag_row.delta_mass_Y_update;
            mass_diag_summary.cumulative_delta_Y_to_xB += mass_diag_row.delta_mass_Y_to_xB;
            mass_diag_summary.cumulative_delta_clipping += mass_diag_row.delta_mass_clipping;
            mass_diag_summary.max_abs_step_drift =
                fmax(mass_diag_summary.max_abs_step_drift, fabs(mass_diag_row.total_delta_mass_step));
            mass_diag_summary.max_abs_delta_phi_update =
                fmax(mass_diag_summary.max_abs_delta_phi_update, fabs(mass_diag_row.delta_mass_phi_update));
            mass_diag_summary.max_abs_delta_eta_update =
                fmax(mass_diag_summary.max_abs_delta_eta_update, fabs(mass_diag_row.delta_mass_eta_update));
            mass_diag_summary.max_abs_delta_gp_to_beta_event =
                fmax(mass_diag_summary.max_abs_delta_gp_to_beta_event, fabs(mass_diag_row.delta_mass_gp_to_beta_event));
            mass_diag_summary.max_abs_delta_Y_update =
                fmax(mass_diag_summary.max_abs_delta_Y_update, fabs(mass_diag_row.delta_mass_Y_update));
            mass_diag_summary.max_abs_delta_Y_to_xB =
                fmax(mass_diag_summary.max_abs_delta_Y_to_xB, fabs(mass_diag_row.delta_mass_Y_to_xB));
            mass_diag_summary.max_abs_delta_clipping =
                fmax(mass_diag_summary.max_abs_delta_clipping, fabs(mass_diag_row.delta_mass_clipping));
            mass_diag_summary.total_clip_count_phi +=
                mass_diag_row.phi_clip_count_low + mass_diag_row.phi_clip_count_high;
            mass_diag_summary.total_clip_count_xB +=
                mass_diag_row.xB_clip_count_low + mass_diag_row.xB_clip_count_high;
            mass_diag_summary.total_clip_count_Y +=
                mass_diag_row.Y_clip_count_low + mass_diag_row.Y_clip_count_high;
            {
                const double closure_resid =
                    mass_diag_row.total_delta_mass_step -
                    (mass_diag_row.delta_mass_phi_update +
                     mass_diag_row.delta_mass_eta_update +
                     mass_diag_row.delta_mass_gp_to_beta_event +
                     mass_diag_row.delta_mass_Y_update +
                     mass_diag_row.delta_mass_Y_to_xB +
                     mass_diag_row.delta_mass_clipping);
                mass_diag_summary.max_abs_mass_closure_resid =
                    fmax(mass_diag_summary.max_abs_mass_closure_resid, fabs(closure_resid));
            }
        }

        if (P.scheduled_nuc_enabled && P.mode == 0) {
            if (!apply_scheduled_events_cpu(&scheduled_runtime, &P, step,
                                            d_phi_r, d_Y_r, d_xB_r,
                                            total_r, size_r,
                                            case_output_dir)) {
                fprintf(stderr, "[fatal] scheduled nucleation insertion failed at step %d\n", step);
                return 2;
            }
        }

        if (P.mode == 1 && minimize_should_stop) {
            // shrink "effective nsteps" so the final-output logic matches actual stop iter
            P.nsteps = step;
            break;
        }

        if (geometry_summary_runtime.enabled) {
            char step_summary_path[4096] = {0};
            char step_phi_path[4096] = {0};
            if (geometry_summary_runtime.write_each_step) {
                const char *summary_root = (P.mode == 1) ? case_output_dir : output_dir;
                const char *tag = (vtk_case_tag[0] != '\0') ? vtk_case_tag : "case_unknown";
                snprintf(step_summary_path, sizeof(step_summary_path),
                         "%s/summary_step_%05d_%s.txt", summary_root, step, tag);
                snprintf(step_phi_path, sizeof(step_phi_path),
                         "%s/phi_step_%05d_%s.vtk", summary_root, step, tag);
            }

            int ok = compute_geometry_summary_from_device(&P,
                                                          d_phi_r, h_phi_r, size_r,
                                                          d_bbox_mins, d_bbox_maxs,
                                                          d_boundary_sum, d_boundary_count,
                                                          geometry_summary_runtime.write_each_step ? step_summary_path : NULL,
                                                          geometry_summary_runtime.write_each_step ? step_phi_path : NULL,
                                                          &geometry_summary_runtime);
            if (geometry_summary_runtime.print_each_step) {
                printf("[geometry-summary] step=%05d status=%s wall_time=%.6f s calls=%d\n",
                       step, ok ? "ok" : "skipped",
                       geometry_summary_runtime.total_wall_s / fmax(geometry_summary_runtime.calls, 1),
                       geometry_summary_runtime.calls);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        if (step_wall_s) {
            step_wall_s[step - 1] = wall_time_sec_monotonic() - step_wall_t0;
        }
    }
    const double step_loop_wall_elapsed = wall_time_sec_monotonic() - step_loop_wall_t0;

    printf("\n========================================\n");
    printf("步进墙钟统计:\n");
    printf("  总步进墙钟时间: %.6f s\n", step_loop_wall_elapsed);
    printf("  平均每步墙钟时间: %.6f s\n",
           (steps_completed > 0) ? (step_loop_wall_elapsed / (double)steps_completed) : 0.0);
    if (geometry_summary_runtime.calls > 0) {
        printf("  geometry summary 调用次数: %d\n", geometry_summary_runtime.calls);
        printf("  geometry summary 有效次数: %d\n", geometry_summary_runtime.valid_calls);
        printf("  geometry summary 累计耗时: %.6f s\n", geometry_summary_runtime.total_wall_s);
        printf("  geometry summary 平均耗时: %.6f s\n",
               geometry_summary_runtime.total_wall_s / (double)geometry_summary_runtime.calls);
        printf("  geometry summary 占步进时间比例: %.2f%%\n",
               (step_loop_wall_elapsed > 0.0)
                   ? (100.0 * geometry_summary_runtime.total_wall_s / step_loop_wall_elapsed)
                   : 0.0);
    }
    printf("========================================\n");
    
    // minimize 模式：在收敛/退出后额外输出 final VTK
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
        if (is_gp_zone_mode(&P)) {
            build_case_vtk_path(filename, sizeof(filename),
                                output_dir, case_output_dir,
                                "eta", VTK_NAME_FINAL, 0, vtk_case_tag, 1);
            int ret_eta_final = write_vtk_cuda(d_eta_r, P.Nx, P.Ny, P.Nz, "eta", P.nsteps, filename);
            if (!ret_eta_final) {
                fprintf(stderr, "ERROR: Failed to write eta VTK file (final): %s\n", filename);
            }
        }

        if (P.minimize_full_model == 1) {
            build_case_vtk_path(filename, sizeof(filename),
                                output_dir, case_output_dir,
                                "xB", VTK_NAME_FINAL, 0, vtk_case_tag, 1);
            int ret_xB_final = write_vtk_cuda(d_xB_r, P.Nx, P.Ny, P.Nz, "xB", P.nsteps, filename);
            if (!ret_xB_final) {
                fprintf(stderr, "ERROR: Failed to write xB VTK file (final): %s\n", filename);
            } else {
                printf("写出 final xB vtk: %s\n", filename);
            }
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
            launch_compute_model_xBtot_kernel(&P, d_phi_r, d_eta_r, d_xB_r, d_xBtot_temp, total_r);
            build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir,
                                xBtot_output_stem(&P), VTK_NAME_STEP, output_step, vtk_case_tag, 0);
            ret1 = write_vtk_cuda(d_xBtot_temp, P.Nx, P.Ny, P.Nz, xBtot_vtk_field_name(&P), output_step, filename);
            if (!ret1) fprintf(stderr, "ERROR: Failed to write %s VTK file (final step): %s\n",
                               xBtot_output_stem(&P), filename);
            build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "xB", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
            ret3 = write_vtk_cuda(d_xB_r, P.Nx, P.Ny, P.Nz, "xB", output_step, filename);
            if (!ret3) fprintf(stderr, "ERROR: Failed to write xB VTK file (final step): %s\n", filename);
        }
        build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "phi", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
        int ret2 = write_vtk_cuda(d_phi_r, P.Nx, P.Ny, P.Nz, "phi", output_step, filename);
        if (!ret2) fprintf(stderr, "ERROR: Failed to write phi VTK file (final step): %s\n", filename);
        if (is_gp_zone_mode(&P)) {
            build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "eta", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
            int ret_eta = write_vtk_cuda(d_eta_r, P.Nx, P.Ny, P.Nz, "eta", output_step, filename);
            if (!ret_eta) fprintf(stderr, "ERROR: Failed to write eta VTK file (final step): %s\n", filename);
        }
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
                P.eps_iso_over_vB, (phi_elastic_coupling_enabled ? P.elastic_shift_dimless : 0.0),
                total_r, phi_elastic_coupling_enabled);
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
                (phi_elastic_coupling_enabled ? P.elastic_shift_dimless : 0.0),
                d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                d_uxx_r, d_uyy_r, d_uzz_r, d_uxy_r, d_uxz_r, d_uyz_r,
                // Optimization(4): d_uxx0_r..d_uyz0_r 参数已移除
                P.S_p_11, P.S_p_12, P.S_p_13, P.S_p_14, P.S_p_15, P.S_p_16,
                P.S_p_22, P.S_p_23, P.S_p_24, P.S_p_25, P.S_p_26,
                P.S_p_33, P.S_p_34, P.S_p_35, P.S_p_36,
                P.S_p_44, P.S_p_45, P.S_p_46, P.S_p_55, P.S_p_56, P.S_p_66,
                P.eps_xx00, P.eps_yy00, P.eps_zz00, P.eps_yz00, P.eps_xz00, P.eps_xy00,
                P.eps_iso_over_vB, total_r, phi_elastic_coupling_enabled);
            build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "driving_force", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
            int ret10 = write_vtk_cuda(scratch1, P.Nx, P.Ny, P.Nz, "driving_force", output_step, filename);
            if (!ret10) fprintf(stderr, "ERROR: Failed to write driving_force VTK file (final step): %s\n", filename);

            if (is_gp_zone_mode(&P)) {
                const double gp_mu_ref_raw =
                    compute_gp_mu_reference_mech_mix_host(
                        P.gp_xB_fixed, temperature_K, P.gp_delta_g_stab);
                launch_compute_eta_rhs_components_kernel(
                    d_eta_r, d_phi_r, d_xB_r,
                    d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                    d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                    scratch0, scratch1, d_eta_rhs_r, d_phi_rhs_r, NULL,
                    temperature_K, P.mu_reference_scale,
                    P.Vm_alpha_0, P.dVm_alpha_dxB,
                    P.gp_xB_fixed, P.gp_delta_g0,
                    P.gp_raw_reaction_drive_only,
                    P.gp_raw_reaction_drive_use_raw_units_debug,
                    P.gp_reaction_nu_A, P.gp_reaction_nu_B,
                    gp_mu_ref_raw,
                    P.gp_W_eta, P.eps_iso_over_vB,
                    gp_elastic_solver_enabled,
                    gp_active_eta_coupling_enabled,
                    P.gp_eps_iso,
                    P.gp_elastic_derivative_scale,
                    total_r);
                build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "eta_rhs_chem", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                int ret_eta_chem = write_vtk_cuda(scratch0, P.Nx, P.Ny, P.Nz, "eta_rhs_chem", output_step, filename);
                if (!ret_eta_chem) fprintf(stderr, "ERROR: Failed to write eta_rhs_chem VTK file (final step): %s\n", filename);
                build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "eta_rhs_dw", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                int ret_eta_dw = write_vtk_cuda(scratch1, P.Nx, P.Ny, P.Nz, "eta_rhs_dw", output_step, filename);
                if (!ret_eta_dw) fprintf(stderr, "ERROR: Failed to write eta_rhs_dw VTK file (final step): %s\n", filename);
                build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "eta_rhs_elastic", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                int ret_eta_elastic = write_vtk_cuda(d_eta_rhs_r, P.Nx, P.Ny, P.Nz, "eta_rhs_elastic", output_step, filename);
                if (!ret_eta_elastic) fprintf(stderr, "ERROR: Failed to write eta_rhs_elastic VTK file (final step): %s\n", filename);
                build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "eta_rhs_net_explicit", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                int ret_eta_net = write_vtk_cuda(d_phi_rhs_r, P.Nx, P.Ny, P.Nz, "eta_rhs_net_explicit", output_step, filename);
                if (!ret_eta_net) fprintf(stderr, "ERROR: Failed to write eta_rhs_net_explicit VTK file (final step): %s\n", filename);

                CUFFT_CHECK(cufftExecD2Z(plan_r2c_phi, d_eta_r, d_eta_k));
                launch_dealias_kernel(d_eta_k, P.Nx, P.Ny, P.Nz, NzC,
                                     P.dx, P.dy, P.dz, total_k);
                launch_compute_laplacian_k_kernel(d_eta_k, KS.d_k2, d_eta_rhs_k, total_k);
                launch_dealias_kernel(d_eta_rhs_k, P.Nx, P.Ny, P.Nz, NzC,
                                     P.dx, P.dy, P.dz, total_k);
                CUFFT_CHECK(cufftExecZ2D(plan_c2r_phi, d_eta_rhs_k, scratch0));
                launch_normalize_only_kernel(scratch0, invN, total_r);
                launch_scale_array_kernel(scratch0, -P.gp_kappa_eta, total_r);
                build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "eta_rhs_grad", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                int ret_eta_grad = write_vtk_cuda(scratch0, P.Nx, P.Ny, P.Nz, "eta_rhs_grad", output_step, filename);
                if (!ret_eta_grad) fprintf(stderr, "ERROR: Failed to write eta_rhs_grad VTK file (final step): %s\n", filename);
                launch_add_arrays_kernel(d_phi_rhs_r, scratch0, scratch1, total_r);
                build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "eta_rhs_full_variational", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                int ret_eta_full = write_vtk_cuda(scratch1, P.Nx, P.Ny, P.Nz, "eta_rhs_full_variational", output_step, filename);
                if (!ret_eta_full) fprintf(stderr, "ERROR: Failed to write eta_rhs_full_variational VTK file (final step): %s\n", filename);
                if (P.gp_raw_reaction_drive_only) {
                    launch_compute_eta_rhs_components_kernel(
                        d_eta_r, d_phi_r, d_xB_r,
                        d_sigma_xx_r, d_sigma_yy_r, d_sigma_zz_r,
                        d_sigma_xy_r, d_sigma_xz_r, d_sigma_yz_r,
                        NULL, NULL, NULL, NULL, scratch0,
                        temperature_K, P.mu_reference_scale,
                        P.Vm_alpha_0, P.dVm_alpha_dxB,
                        P.gp_xB_fixed, P.gp_delta_g0,
                        P.gp_raw_reaction_drive_only,
                        P.gp_raw_reaction_drive_use_raw_units_debug,
                        P.gp_reaction_nu_A, P.gp_reaction_nu_B,
                        gp_mu_ref_raw,
                        P.gp_W_eta, P.eps_iso_over_vB,
                        gp_elastic_solver_enabled,
                        gp_active_eta_coupling_enabled,
                        P.gp_eps_iso,
                        P.gp_elastic_derivative_scale,
                        total_r);
                    build_case_vtk_path(filename, sizeof(filename), output_dir, case_output_dir, "gp_minus_delta_mu_r", VTK_NAME_STEP, output_step, vtk_case_tag, 0);
                    int ret_gp_minus_dmu = write_vtk_cuda(scratch0, P.Nx, P.Ny, P.Nz, "gp_minus_delta_mu_r", output_step, filename);
                    if (!ret_gp_minus_dmu) fprintf(stderr, "ERROR: Failed to write gp_minus_delta_mu_r VTK file (final step): %s\n", filename);
                }
            }
            
            if (ret1 && ret2 && ret3 && ret4 && ret5 && ret6 && ret7 && ret8 && ret9 && ret10) {
                printf("step=%05d（最终步）完成，已输出VTK文件（phi_%d.vtk, xB_%d.vtk, %s_%d.vtk, delta_mu_r_%d.vtk, ...）\n",
                       final_step, output_step, output_step, xBtot_output_stem(&P), output_step, output_step);
            } else {
                fprintf(stderr, "step=%05d（最终步）警告：诊断VTK文件输出失败！\n", final_step);
            }
        } else {
            if (P.mode == 1) {
                if (ret2) printf("step=%05d（最终步）完成，已输出VTK文件（phi_%d.vtk）\n", final_step, output_step);
                else fprintf(stderr, "step=%05d（最终步）警告：VTK文件输出失败！\n", final_step);
            } else {
                if (ret1 && ret2 && ret3) {
                    printf("step=%05d（最终步）完成，已输出VTK文件（phi_%d.vtk, xB_%d.vtk, %s_%d.vtk）\n",
                           final_step, output_step, output_step, xBtot_output_stem(&P), output_step);
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
                        final_step + csv_step_offset, current_time + csv_time_offset, t_real + csv_real_time_offset, vf_precip, R_avg,
                        sum_h_csv, sum_gel_csv, E_hat_csv, E_Jm3_csv, dmu_csv);
            } else {
                fprintf(csv_fp, "%d,%.8f,%.8e,%.8e,%.8e\n",
                        final_step + csv_step_offset, current_time + csv_time_offset, t_real + csv_real_time_offset, vf_precip, R_avg);
            }
            fflush(csv_fp);
        }
    }
    if (relax_fp && P.mode == 0 && (P.nsteps % P.csv_out_every != 0)) {
        const int final_step = P.nsteps;
        const double current_time = final_step * P.dt;
        const double t_real = current_time * P.t_real_unit;
            write_relaxation_diag_row(relax_fp,
                                      final_step + csv_step_offset,
                                      current_time + csv_time_offset,
                                      t_real + csv_real_time_offset,
                                      d_phi_r,
                                      d_eta_r,
                                      d_xB_r,
                                      &P,
                                      total_r,
                                      relaxation_initial_mean_xBtot);
    }

    {
        char geometry_phi_vtk_path[4096] = {0};
        char geometry_summary_path[4096] = {0};
        int should_write_geometry_summary = 0;

        if (P.mode == 1) {
            build_case_vtk_path(geometry_phi_vtk_path, sizeof(geometry_phi_vtk_path),
                                output_dir, case_output_dir,
                                "phi", VTK_NAME_FINAL, 0, vtk_case_tag, 1);
            build_case_summary_path(geometry_summary_path, sizeof(geometry_summary_path),
                                    output_dir, case_output_dir, vtk_case_tag, 1);
            should_write_geometry_summary = 1;
        } else if (P.mode == 0) {
            int final_output_step = P.nsteps / P.out_every;
            if (P.nsteps % P.out_every != 0) final_output_step += 1;
            build_case_vtk_path(geometry_phi_vtk_path, sizeof(geometry_phi_vtk_path),
                                output_dir, case_output_dir,
                                "phi", VTK_NAME_STEP, final_output_step, vtk_case_tag, 0);
            build_case_summary_path(geometry_summary_path, sizeof(geometry_summary_path),
                                    output_dir, case_output_dir, vtk_case_tag, 0);
            should_write_geometry_summary = 1;
        }

        if (should_write_geometry_summary && geometry_summary_path[0] != '\0') {
            if (compute_geometry_summary_from_device(&P,
                                                    d_phi_r, h_phi_r, size_r,
                                                    d_bbox_mins, d_bbox_maxs,
                                                    d_boundary_sum, d_boundary_count,
                                                    geometry_summary_path,
                                                    geometry_phi_vtk_path,
                                                    NULL)) {
                printf("写出 geometry summary: %s\n", geometry_summary_path);
            } else {
                fprintf(stderr, "[warn] geometry summary skipped: no valid nucleus component found.\n");
            }
        }
    }

    if (P.mode == 0 && P.dynamics_mass_diag_enabled) {
        mass_diag_summary.final_mean_xBtot =
            gpu_reduce_sum_model_xBtot(&P, d_phi_r, d_eta_r, d_xB_r, total_r) / (double)total_r;
        mass_diag_summary.total_absolute_drift =
            mass_diag_summary.final_mean_xBtot - mass_diag_summary.initial_mean_xBtot;
        mass_diag_summary.total_relative_drift =
            mass_diag_summary.total_absolute_drift /
            fmax(fabs(mass_diag_summary.initial_mean_xBtot), 1.0e-30);
        const double accounted =
            mass_diag_summary.cumulative_delta_phi_update +
            mass_diag_summary.cumulative_delta_eta_update +
            mass_diag_summary.cumulative_delta_gp_to_beta_event +
            mass_diag_summary.cumulative_delta_Y_update +
            mass_diag_summary.cumulative_delta_Y_to_xB +
            mass_diag_summary.cumulative_delta_clipping;
        mass_diag_summary.cumulative_delta_other =
            mass_diag_summary.total_absolute_drift - accounted;
        {
            const double mismatch_phi =
                fabs(mass_diag_summary.cumulative_delta_phi_update - mass_diag_summary.cumulative_delta_phi_update);
            const double mismatch_Y =
                fabs(mass_diag_summary.cumulative_delta_Y_update - mass_diag_summary.cumulative_delta_Y_update);
            const double mismatch_conv =
                fabs(mass_diag_summary.cumulative_delta_Y_to_xB - mass_diag_summary.cumulative_delta_Y_to_xB);
            const double mismatch_clip =
                fabs(mass_diag_summary.cumulative_delta_clipping - mass_diag_summary.cumulative_delta_clipping);
            const double mismatch_abs =
                fabs(mass_diag_summary.total_absolute_drift -
                     (mass_diag_summary.final_mean_xBtot - mass_diag_summary.initial_mean_xBtot));
            mass_diag_summary.max_summary_csv_abs_mismatch =
                fmax(fmax(fmax(mismatch_phi, mismatch_Y), fmax(mismatch_conv, mismatch_clip)), mismatch_abs);
            mass_diag_summary.summary_matches_csv =
                (mass_diag_summary.max_summary_csv_abs_mismatch <= 1.0e-15) ? 1.0 : 0.0;
        }
        mass_diag_summary.suspected_primary_source =
            infer_primary_mass_source(&mass_diag_summary);

        char summary_csv_path[4096];
        snprintf(summary_csv_path, sizeof(summary_csv_path), "%s/mass_drift_summary.csv", case_output_dir);
        FILE *summary_fp = fopen(summary_csv_path, "w");
        if (!summary_fp) {
            fprintf(stderr, "[warn] cannot open mass drift summary csv %s\n", summary_csv_path);
        } else {
            if (is_gp_zone_mode(&P)) {
                fprintf(summary_fp,
                        "initial_mean_xBtot_gp,final_mean_xBtot_gp,total_absolute_drift,total_relative_drift,"
                        "cumulative_delta_phi_update,cumulative_delta_eta_update,cumulative_delta_gp_to_beta_event,cumulative_delta_Y_update,cumulative_delta_Y_to_xB,"
                        "cumulative_delta_clipping,cumulative_delta_other,max_abs_step_drift,"
                        "max_abs_delta_phi_update,max_abs_delta_eta_update,max_abs_delta_gp_to_beta_event,max_abs_delta_Y_update,max_abs_delta_Y_to_xB,max_abs_delta_clipping,"
                        "total_clip_count_phi,total_clip_count_xB,total_clip_count_Y,"
                        "summary_matches_csv,max_summary_csv_abs_mismatch,max_abs_mass_closure_resid,suspected_primary_source\n");
                fprintf(summary_fp, "%.17e,%.17e,%.17e,%.17e,",
                        mass_diag_summary.initial_mean_xBtot,
                        mass_diag_summary.final_mean_xBtot,
                        mass_diag_summary.total_absolute_drift,
                        mass_diag_summary.total_relative_drift);
                fprintf(summary_fp, "%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,",
                        mass_diag_summary.cumulative_delta_phi_update,
                        mass_diag_summary.cumulative_delta_eta_update,
                        mass_diag_summary.cumulative_delta_gp_to_beta_event,
                        mass_diag_summary.cumulative_delta_Y_update,
                        mass_diag_summary.cumulative_delta_Y_to_xB,
                        mass_diag_summary.cumulative_delta_clipping,
                        mass_diag_summary.cumulative_delta_other);
                fprintf(summary_fp, "%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,",
                        mass_diag_summary.max_abs_step_drift,
                        mass_diag_summary.max_abs_delta_phi_update,
                        mass_diag_summary.max_abs_delta_eta_update,
                        mass_diag_summary.max_abs_delta_gp_to_beta_event,
                        mass_diag_summary.max_abs_delta_Y_update,
                        mass_diag_summary.max_abs_delta_Y_to_xB,
                        mass_diag_summary.max_abs_delta_clipping);
                fprintf(summary_fp, "%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%s\n",
                        mass_diag_summary.total_clip_count_phi,
                        mass_diag_summary.total_clip_count_xB,
                        mass_diag_summary.total_clip_count_Y,
                        mass_diag_summary.summary_matches_csv,
                        mass_diag_summary.max_summary_csv_abs_mismatch,
                        mass_diag_summary.max_abs_mass_closure_resid,
                        mass_diag_summary.suspected_primary_source ? mass_diag_summary.suspected_primary_source : "mixed_or_unclear");
            } else {
                fprintf(summary_fp,
                        "initial_mean_xBtot,final_mean_xBtot,total_absolute_drift,total_relative_drift,"
                        "cumulative_delta_phi_update,cumulative_delta_Y_update,cumulative_delta_Y_to_xB,"
                        "cumulative_delta_clipping,cumulative_delta_other,max_abs_step_drift,"
                        "max_abs_delta_phi_update,max_abs_delta_Y_update,max_abs_delta_Y_to_xB,max_abs_delta_clipping,"
                        "total_clip_count_phi,total_clip_count_xB,total_clip_count_Y,"
                        "summary_matches_csv,max_summary_csv_abs_mismatch,max_abs_mass_closure_resid,suspected_primary_source\n");
                fprintf(summary_fp,
                        "%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%s\n",
                        mass_diag_summary.initial_mean_xBtot,
                        mass_diag_summary.final_mean_xBtot,
                        mass_diag_summary.total_absolute_drift,
                        mass_diag_summary.total_relative_drift,
                        mass_diag_summary.cumulative_delta_phi_update,
                        mass_diag_summary.cumulative_delta_Y_update,
                        mass_diag_summary.cumulative_delta_Y_to_xB,
                        mass_diag_summary.cumulative_delta_clipping,
                        mass_diag_summary.cumulative_delta_other,
                        mass_diag_summary.max_abs_step_drift,
                        mass_diag_summary.max_abs_delta_phi_update,
                        mass_diag_summary.max_abs_delta_Y_update,
                        mass_diag_summary.max_abs_delta_Y_to_xB,
                        mass_diag_summary.max_abs_delta_clipping,
                        mass_diag_summary.total_clip_count_phi,
                        mass_diag_summary.total_clip_count_xB,
                        mass_diag_summary.total_clip_count_Y,
                        mass_diag_summary.summary_matches_csv,
                        mass_diag_summary.max_summary_csv_abs_mismatch,
                        mass_diag_summary.max_abs_mass_closure_resid,
                        mass_diag_summary.suspected_primary_source ? mass_diag_summary.suspected_primary_source : "mixed_or_unclear");
            }
            fclose(summary_fp);
        }
        char summary_json_path[4096];
        snprintf(summary_json_path, sizeof(summary_json_path), "%s/mass_drift_summary.json", case_output_dir);
        write_mass_drift_summary_json(summary_json_path, &mass_diag_summary, &P);
        if (is_gp_zone_mode(&P)) {
            printf("[mass-drift-summary] initial=<xBtot_gp>=%.8e final=<xBtot_gp>=%.8e rel_drift=%.6e primary=%s\n",
                   mass_diag_summary.initial_mean_xBtot,
                   mass_diag_summary.final_mean_xBtot,
                   mass_diag_summary.total_relative_drift,
                   mass_diag_summary.suspected_primary_source ? mass_diag_summary.suspected_primary_source : "mixed_or_unclear");
        } else {
            printf("[mass-drift-summary] initial=<xBtot>=%.8e final=<xBtot>=%.8e rel_drift=%.6e primary=%s\n",
                   mass_diag_summary.initial_mean_xBtot,
                   mass_diag_summary.final_mean_xBtot,
                   mass_diag_summary.total_relative_drift,
                   mass_diag_summary.suspected_primary_source ? mass_diag_summary.suspected_primary_source : "mixed_or_unclear");
        }
    }

    if (P.mode == 0 && step_wall_s && steps_completed > 0) {
        PerformanceSummary perf;
        memset(&perf, 0, sizeof(perf));
        snprintf(perf.case_name, sizeof(perf.case_name), "%s", P.init_case_tag);
        perf.picard_enabled = P.enable_Y_rhs_picard ? 1.0 : 0.0;
        perf.picard_iters = (double)P.Y_rhs_picard_iters;
        perf.picard_omega = P.Y_rhs_picard_omega;
        perf.nsteps = (double)steps_completed;
        perf.dt = P.dt;
        perf.total_walltime_s = 0.0;
        perf.min_walltime_per_step_s = step_wall_s[0];
        perf.max_walltime_per_step_s = step_wall_s[0];
        double *step_wall_sorted = (double *)malloc((size_t)steps_completed * sizeof(double));
        for (int i = 0; i < steps_completed; ++i) {
            const double w = step_wall_s[i];
            perf.total_walltime_s += w;
            perf.min_walltime_per_step_s = fmin(perf.min_walltime_per_step_s, w);
            perf.max_walltime_per_step_s = fmax(perf.max_walltime_per_step_s, w);
            step_wall_sorted[i] = w;
        }
        perf.avg_walltime_per_step_s = perf.total_walltime_s / (double)steps_completed;
        qsort(step_wall_sorted, (size_t)steps_completed, sizeof(double), compare_double_asc);
        if (steps_completed % 2 == 0) {
            perf.median_walltime_per_step_s =
                0.5 * (step_wall_sorted[steps_completed / 2 - 1] + step_wall_sorted[steps_completed / 2]);
        } else {
            perf.median_walltime_per_step_s = step_wall_sorted[steps_completed / 2];
        }
        free(step_wall_sorted);
        {
            const int warmup = (steps_completed > 10) ? 10 : ((steps_completed > 5) ? 5 : 0);
            double warm_sum = 0.0;
            int warm_n = 0;
            for (int i = warmup; i < steps_completed; ++i) {
                warm_sum += step_wall_s[i];
                warm_n++;
            }
            perf.warmup_excluded_avg_walltime_per_step_s =
                (warm_n > 0) ? (warm_sum / (double)warm_n) : perf.avg_walltime_per_step_s;
        }
        perf.estimated_steps_per_hour =
            (perf.warmup_excluded_avg_walltime_per_step_s > 0.0)
                ? (3600.0 / perf.warmup_excluded_avg_walltime_per_step_s)
                : 0.0;
        perf.relative_slowdown_vs_baseline = P.enable_Y_rhs_picard ? NAN : 1.0;

        char perf_csv_path[4096];
        snprintf(perf_csv_path, sizeof(perf_csv_path), "%s/performance_summary.csv", case_output_dir);
        FILE *perf_fp = fopen(perf_csv_path, "w");
        if (perf_fp) {
            fprintf(perf_fp,
                    "case_name,picard_enabled,picard_iters,picard_omega,nsteps,dt,total_walltime_s,"
                    "avg_walltime_per_step_s,median_walltime_per_step_s,min_walltime_per_step_s,max_walltime_per_step_s,"
                    "warmup_excluded_avg_walltime_per_step_s,estimated_steps_per_hour,relative_slowdown_vs_baseline\n");
            fprintf(perf_fp,
                    "%s,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n",
                    perf.case_name, perf.picard_enabled, perf.picard_iters, perf.picard_omega,
                    perf.nsteps, perf.dt, perf.total_walltime_s, perf.avg_walltime_per_step_s,
                    perf.median_walltime_per_step_s, perf.min_walltime_per_step_s, perf.max_walltime_per_step_s,
                    perf.warmup_excluded_avg_walltime_per_step_s, perf.estimated_steps_per_hour,
                    perf.relative_slowdown_vs_baseline);
            fclose(perf_fp);
        }
        char perf_json_path[4096];
        snprintf(perf_json_path, sizeof(perf_json_path), "%s/performance_summary.json", case_output_dir);
        write_performance_summary_json(perf_json_path, &perf);
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
    CUDA_CHECK(cudaFree(d_eta_r));
    CUDA_CHECK(cudaFree(d_Y_r));
    CUDA_CHECK(cudaFree(d_xB_r));
    CUDA_CHECK(cudaFree(d_phi_rhs_r));
    CUDA_CHECK(cudaFree(d_eta_prev_r));
    CUDA_CHECK(cudaFree(d_eta_rhs_r));
    // 优化：d_phi_rhs_r被d_lapY_r复用，只需要释放一次
    CUDA_CHECK(cudaFree(d_phi_n_saved));
    CUDA_CHECK(cudaFree(d_Y_n_saved));
    if (d_Y_projection_base_r) CUDA_CHECK(cudaFree(d_Y_projection_base_r));
    CUDA_CHECK(cudaFree(d_dY_dt_prev_r));
    CUDA_CHECK(cudaFree(d_dY_dt_picard_r));
    CUDA_CHECK(cudaFree(d_dY_dt_picard_old_r));
    // 优化：不再需要释放d_xBtot_r
    CUDA_CHECK(cudaFree(d_mu_x_r));
    // Optimization: d_Y_rhs_r 复用 d_mu_x_r，不需要单独释放
    // Optimization: 移除 grad_mu/J 常驻，d_xB_prev_r 与 d_divJ_r 共享内存
    CUDA_CHECK(cudaFree(d_divJ_r));
    if (d_xB_gp_old_r) CUDA_CHECK(cudaFree(d_xB_gp_old_r));
    if (d_xB_old_diag_r) CUDA_CHECK(cudaFree(d_xB_old_diag_r));
    // 优化：不再需要释放d_DY_values
    CUDA_CHECK(cudaFree(d_phi_k));
    CUDA_CHECK(cudaFree(d_eta_k));
    CUDA_CHECK(cudaFree(d_eta_rhs_k));
    CUDA_CHECK(cudaFree(d_phi_rhs_k));
    CUDA_CHECK(cudaFree(d_Y_k));
    CUDA_CHECK(cudaFree(d_divJ_k));
    if (d_mass_diag_phi_stats) CUDA_CHECK(cudaFree(d_mass_diag_phi_stats));
    if (d_mass_diag_Y_stats) CUDA_CHECK(cudaFree(d_mass_diag_Y_stats));
    if (d_mass_diag_Y_rhs_stats) CUDA_CHECK(cudaFree(d_mass_diag_Y_rhs_stats));
    if (d_mass_diag_gp_storage_stats) CUDA_CHECK(cudaFree(d_mass_diag_gp_storage_stats));
    if (d_gp_Y_update_stats) CUDA_CHECK(cudaFree(d_gp_Y_update_stats));
    if (d_gp_eta_feas_stats) CUDA_CHECK(cudaFree(d_gp_eta_feas_stats));
    if (d_gp_transport_stats) CUDA_CHECK(cudaFree(d_gp_transport_stats));
    if (d_gp_elastic_stats) CUDA_CHECK(cudaFree(d_gp_elastic_stats));
    if (d_picard_diff_stats) CUDA_CHECK(cudaFree(d_picard_diff_stats));
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
    free(h_eta_r);
    free(h_Y_r);
    free(h_xB_r);
    free(h_xBtot_r);
    if (step_wall_s) free(step_wall_s);
    
    if (csv_fp) {
        fclose(csv_fp);
    }
    if (relax_fp) {
        fclose(relax_fp);
    }
    if (mass_diag_fp) {
        fclose(mass_diag_fp);
    }
    if (phi_eta_step_delta_fp) {
        fclose(phi_eta_step_delta_fp);
    }
    if (phi_eta_rhs_attribution_fp) {
        fclose(phi_eta_rhs_attribution_fp);
    }
    if (eta_bulk_max_location_fp) {
        fclose(eta_bulk_max_location_fp);
    }
    if (energy_fp) {
        fclose(energy_fp);
    }
    if (scheduled_runtime.events_csv) {
        fclose(scheduled_runtime.events_csv);
    }
    if (gp_nuc_runtime.events_csv) {
        fclose(gp_nuc_runtime.events_csv);
    }
    if (gp_to_beta_runtime.count_events_accepted > 0 ||
        gp_to_beta_runtime.count_events_rejected_capacity > 0 ||
        gp_to_beta_runtime.count_events_scaled_capacity > 0) {
        const double mean_capacity_ratio =
            (gp_to_beta_runtime.capacity_ratio_count > 0)
                ? (gp_to_beta_runtime.sum_capacity_ratio / (double)gp_to_beta_runtime.capacity_ratio_count)
                : NAN;
        log_kv_text("gp_to_beta_events_accepted", "%d", gp_to_beta_runtime.count_events_accepted);
        log_kv_text("gp_to_beta_events_scaled_capacity", "%d", gp_to_beta_runtime.count_events_scaled_capacity);
        log_kv_text("gp_to_beta_events_rejected_capacity", "%d", gp_to_beta_runtime.count_events_rejected_capacity);
        log_kv_text("gp_to_beta_events_rejected_cooldown", "%d", gp_to_beta_runtime.count_events_rejected_cooldown);
        log_kv_text("gp_to_beta_events_rejected_spacing", "%d", gp_to_beta_runtime.count_events_rejected_spacing);
        log_kv_text("gp_to_beta_events_rejected_window", "%d", gp_to_beta_runtime.count_events_rejected_window);
        log_kv_text("gp_to_beta_events_rejected_global", "%d", gp_to_beta_runtime.count_events_rejected_global);
        log_kv_text("gp_to_beta_capacity_ratio_min", "%.10e",
                    isfinite(gp_to_beta_runtime.min_capacity_ratio) ? gp_to_beta_runtime.min_capacity_ratio : NAN);
        log_kv_text("gp_to_beta_capacity_ratio_mean", "%.10e", mean_capacity_ratio);
    }
    if (gp_to_beta_runtime.events_csv) {
        fclose(gp_to_beta_runtime.events_csv);
    }
    if (gp_to_beta_runtime.audit_csv) {
        fclose(gp_to_beta_runtime.audit_csv);
    }
    if (post_conversion_audit_runtime.csv) {
        fclose(post_conversion_audit_runtime.csv);
    }
    if (y_update_k0_audit_runtime.csv) {
        fclose(y_update_k0_audit_runtime.csv);
    }
    if (y_update_mass_projection_fp) {
        fclose(y_update_mass_projection_fp);
    }
    
    const double wall_t1 = wall_time_sec_monotonic();
    const double wall_elapsed = wall_t1 - wall_t0;
    int hh = 0, mm = 0, ss = 0;
    format_hms(wall_elapsed, &hh, &mm, &ss);
    log_section_header("Run Summary");
    log_kv_text("mode", "%s",
                (P.mode == 0)
                    ? (P.minimize_continue_from_vtk ? "dynamic-continue" : "dynamic")
                    : "minimize");
    log_kv_text("steps_completed", "%d", steps_completed);
    log_kv_text("wall_time_s", "%.3f", wall_elapsed);
    log_kv_text("wall_time_hms", "%02d:%02d:%02d", hh, mm, ss);
    return 0;
}
