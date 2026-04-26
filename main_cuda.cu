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

static int g_dynamic_outputs_use_case_dir = 0;

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

static void build_case_legacy_summary_path(char *out, size_t out_size,
                                           const char *case_output_dir,
                                           const char *case_tag) {
    const char *tag = (case_tag && case_tag[0] != '\0') ? case_tag : "case_unknown";
    if (!out || out_size == 0) return;
    if (!case_output_dir) {
        out[0] = '\0';
        return;
    }
    snprintf(out, out_size, "%s/summary_%s.txt", case_output_dir, tag);
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
static void derive_next_continue_case_tag(const char *vtk_path, char *out, size_t out_size, int mode);
static int rebuild_full_model_composition_from_phi(double *phi_r, double *Y_r, double *xB_r, double *xBtot_r,
                                                   const PFParams *P, int total_size, int emit_logs);
static int load_continue_fields_from_vtk(double *phi_r, double *Y_r, double *xB_r, double *xBtot_r,
                                         const PFParams *P, int total_size);

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
    P->t_real_unit = -1.0;
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
    P->minimize_continue_from_vtk = 0;
    P->continue_phi_vtk_path[0] = '\0';
    P->continue_xB_vtk_path[0] = '\0';

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
    TRY_SET_DOUBLE("dt", dt);
    TRY_SET_DOUBLE("dx", dx);
    TRY_SET_DOUBLE("dy", dy);
    TRY_SET_DOUBLE("dz", dz);
    TRY_SET_DOUBLE("t_real_unit", t_real_unit);
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
            printf("  --init-test-id <int>    preset index for batch tests (0..7, <0 to disable)\n");
            printf("  --continue-phi-vtk <path>  continuation: load phi from ASCII VTK and continue minimize\n");
            printf("  --continue-xB-vtk <path>   continuation(full-model): optional xB ASCII VTK; if missing, rebuild xB/Y from phi via init logic\n");
            printf("                               若 VTK 同目录存在 pf_input.params，则 continue 会优先使用该参数快照\n");
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
                ? (P.minimize_continue_from_vtk ? "dynamics-continue" : "dynamics")
                : (P.minimize_full_model
                       ? "minimize (full-model: chem+diffusion+volume)"
                       : "minimize (phi-only)");
        snprintf(grid_buf, sizeof(grid_buf), "%dx%dx%d", P.Nx, P.Ny, P.Nz);

        log_section_header("Run Configuration");
        log_kv_text("mode", "%s", mode_label);
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
        if (radius_phys_nm <= 0.0) {
            fprintf(stderr, "[fatal] --radius-phys-nm must be > 0, got %.6f\n", radius_phys_nm);
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
    g_dynamic_outputs_use_case_dir = (P.mode == 0 && P.minimize_continue_from_vtk) ? 1 : 0;

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
        log_kv_text("vtk_mode", "%s",
                    (P.mode == 0)
                        ? (P.minimize_continue_from_vtk ? "dynamic-continue (with case suffix)" : "dynamic (no case suffix)")
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
    if (P.minimize_continue_from_vtk) {
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
    } else if (P.mode == 1 && P.minimize_full_model == 0) {
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
    const double step_loop_wall_t0 = wall_time_sec_monotonic();

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
                        step + csv_step_offset, current_time + csv_time_offset, t_real + csv_real_time_offset, vf_precip, R_avg,
                        sum_h_csv, sum_gel_csv, E_hat_csv, E_Jm3_csv, dmu_csv);
            } else {
                fprintf(csv_fp, "%d,%.8f,%.8e,%.8e,%.8e\n",
                        step + csv_step_offset, current_time + csv_time_offset, t_real + csv_real_time_offset, vf_precip, R_avg);
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
                        final_step + csv_step_offset, current_time + csv_time_offset, t_real + csv_real_time_offset, vf_precip, R_avg,
                        sum_h_csv, sum_gel_csv, E_hat_csv, E_Jm3_csv, dmu_csv);
            } else {
                fprintf(csv_fp, "%d,%.8f,%.8e,%.8e,%.8e\n",
                        final_step + csv_step_offset, current_time + csv_time_offset, t_real + csv_real_time_offset, vf_precip, R_avg);
            }
            fflush(csv_fp);
        }
    }

    {
        char geometry_phi_vtk_path[4096] = {0};
        char geometry_summary_path[4096] = {0};
        char geometry_legacy_summary_path[4096] = {0};
        int should_write_geometry_summary = 0;

        if (P.mode == 1) {
            build_case_vtk_path(geometry_phi_vtk_path, sizeof(geometry_phi_vtk_path),
                                output_dir, case_output_dir,
                                "phi", VTK_NAME_FINAL, 0, vtk_case_tag, 1);
            build_case_summary_path(geometry_summary_path, sizeof(geometry_summary_path),
                                    output_dir, case_output_dir, vtk_case_tag, 1);
            build_case_legacy_summary_path(geometry_legacy_summary_path, sizeof(geometry_legacy_summary_path),
                                           case_output_dir, vtk_case_tag);
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
                if (P.mode == 1 &&
                    geometry_legacy_summary_path[0] != '\0' &&
                    strcmp(geometry_legacy_summary_path, geometry_summary_path) != 0) {
                    if (!copy_text_file(geometry_summary_path, geometry_legacy_summary_path)) {
                        fprintf(stderr, "[warn] failed to create legacy geometry summary alias: %s\n",
                                geometry_legacy_summary_path);
                    }
                }
                printf("写出 geometry summary: %s\n", geometry_summary_path);
            } else {
                fprintf(stderr, "[warn] geometry summary skipped: no valid nucleus component found.\n");
            }
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
    log_kv_text("mode", "%s",
                (P.mode == 0)
                    ? (P.minimize_continue_from_vtk ? "dynamic-continue" : "dynamic")
                    : "minimize");
    log_kv_text("steps_completed", "%d", steps_completed);
    log_kv_text("wall_time_s", "%.3f", wall_elapsed);
    log_kv_text("wall_time_hms", "%02d:%02d:%02d", hh, mm, ss);
    return 0;
}
