#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../phase_functions.h"
#include "../thermo_utils.h"

#define CUDA_OR_DIE(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA failure %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(err__)); \
        std::exit(2); \
    } \
} while (0)

struct CellTrace {
    double phi;
    double h;
    double alpha;
    double Ctot;
    double q;
    double xB;
    double active;
    double mu;
    double g_second;
    double Gamma;
    double M_alpha;
    double M_eff;
    int convex_extension;
    int finite;
    int bound_ok;
};

struct FaceTrace {
    int face_id;
    int axis;
    int left_idx;
    int right_idx;
    int left_i;
    int left_j;
    int left_k;
    int right_i;
    int right_j;
    int right_k;
    CellTrace left;
    CellTrace right;
    double face_mobility;
    int failure_reason;
};

enum FailureReason {
    FAILURE_NONE = 0,
    FAILURE_LEFT_ALPHA_OUT_OF_RANGE = 1,
    FAILURE_RIGHT_ALPHA_OUT_OF_RANGE = 2,
    FAILURE_LEFT_MATRIX_CONSTITUTIVE = 3,
    FAILURE_RIGHT_MATRIX_CONSTITUTIVE = 4,
    FAILURE_FACE_MOBILITY = 5
};

__device__ static CellTrace evaluate_cell(
    double phi, double x_initial, double x_inactive, double D_alpha,
    double temperature_K, double energy_scale, double matrix_support_eps)
{
    CellTrace out{};
    out.phi = phi;
    out.h = h_of_phi(phi);
    out.alpha = 1.0 - out.h;
    out.Ctot = out.h + out.alpha * x_initial;
    out.q = out.Ctot - out.h;
    out.active = out.alpha > matrix_support_eps ? 1.0 : 0.0;
    out.xB = out.active > 0.5 ? out.q / out.alpha : x_inactive;
    const double muA = mu_A_dimless(out.xB, temperature_K, energy_scale);
    const double muB = mu_B_dimless(out.xB, temperature_K, energy_scale);
    const double c_bulk = c_xB_phi(out.xB, 1.0, 0.0, 1.0, out.h);
    const double mu_total = mu_tot_mix(muA, muB, out.xB, 0.0, out.h);
    out.mu = c_bulk * (muB - muA - c_bulk * mu_total * 0.0);
    out.g_second = g_alpha_second_raw(temperature_K, out.xB);
    out.Gamma = gamma_thermo_nonlinear(
        out.xB, 0.0, 1.0, 0.0, 1.0, temperature_K, energy_scale);
    out.M_alpha = matrix_mobility_alpha_candidate(
        out.xB, D_alpha, 1.0, 0.0, 1.0, temperature_K, energy_scale);
    out.M_eff = matrix_capacity_mobility_candidate(
        out.h, out.xB, D_alpha, 1.0, 0.0, 1.0,
        temperature_K, energy_scale, matrix_support_eps);
    out.convex_extension = out.xB > X_LIMIT_CONVEX ? 1 : 0;
    out.finite = isfinite(out.phi) && isfinite(out.h) &&
                 isfinite(out.alpha) && isfinite(out.Ctot) &&
                 isfinite(out.q) && isfinite(out.xB) && isfinite(out.mu) &&
                 isfinite(out.g_second) && isfinite(out.Gamma) &&
                 isfinite(out.M_alpha) && isfinite(out.M_eff);
    out.bound_ok = out.Ctot >= out.h - 1.0e-12 &&
                   out.Ctot <= 1.0 + 1.0e-12 &&
                   out.q >= -1.0e-12 && out.q <= out.alpha + 1.0e-12;
    return out;
}

__global__ static void initialize_radius3_cells(
    CellTrace *cells, int N, double radius, double width,
    double x_out, double x_eq, double D_alpha,
    double temperature_K, double energy_scale, double matrix_support_eps)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * N * N;
    if (idx >= total) return;
    const int k = idx % N;
    const int j = (idx / N) % N;
    const int i = idx / (N * N);
    const double rx = (double)i - 0.5 * (double)N;
    const double ry = (double)j - 0.5 * (double)N;
    const double rz = (double)k - 0.5 * (double)N;
    const double r_eff = sqrt((rx * rx + ry * ry + rz * rz) /
                              (radius * radius));
    const double phi = fmin(1.0, fmax(0.0,
        0.5 * (1.0 + tanh((1.0 - r_eff) * (radius / width)))));
    const double h = h_of_phi(phi);
    const double x_initial = (1.0 - h) * x_out + h * x_eq;
    cells[idx] = evaluate_cell(
        phi, x_initial, x_eq, D_alpha, temperature_K, energy_scale,
        matrix_support_eps);
}

__global__ static void trace_failed_faces(
    const CellTrace *cells, FaceTrace *faces, int *failure_count,
    int max_faces, int N, int axis)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * N * N;
    if (idx >= total) return;
    const int k = idx % N;
    const int j = (idx / N) % N;
    const int i = idx / (N * N);
    int ip = i, jp = j, kp = k;
    if (axis == 0) ip = (i + 1) % N;
    else if (axis == 1) jp = (j + 1) % N;
    else kp = (k + 1) % N;
    const int nidx = (ip * N + jp) * N + kp;
    const CellTrace left = cells[idx];
    const CellTrace right = cells[nidx];
    int reason = FAILURE_NONE;
    if (!isfinite(left.alpha) || left.alpha < 0.0 || left.alpha > 1.0) {
        reason = FAILURE_LEFT_ALPHA_OUT_OF_RANGE;
    } else if (!isfinite(right.alpha) || right.alpha < 0.0 || right.alpha > 1.0) {
        reason = FAILURE_RIGHT_ALPHA_OUT_OF_RANGE;
    } else if (!isfinite(left.M_alpha) || left.M_alpha < 0.0) {
        reason = FAILURE_LEFT_MATRIX_CONSTITUTIVE;
    } else if (!isfinite(right.M_alpha) || right.M_alpha < 0.0) {
        reason = FAILURE_RIGHT_MATRIX_CONSTITUTIVE;
    }
    double Mface = 0.0;
    if (isfinite(left.M_eff) && isfinite(right.M_eff) &&
        left.M_eff > 0.0 && right.M_eff > 0.0) {
        Mface = 2.0 * left.M_eff * right.M_eff /
                (left.M_eff + right.M_eff);
    }
    if ((!isfinite(left.M_eff) || !isfinite(right.M_eff) ||
         left.M_eff < 0.0 || right.M_eff < 0.0) && reason == FAILURE_NONE) {
        reason = FAILURE_FACE_MOBILITY;
    }
    if (reason == FAILURE_NONE) return;
    const int slot = atomicAdd(failure_count, 1);
    if (slot >= max_faces) return;
    FaceTrace rec{};
    rec.face_id = slot;
    rec.axis = axis;
    rec.left_idx = idx;
    rec.right_idx = nidx;
    rec.left_i = i; rec.left_j = j; rec.left_k = k;
    rec.right_i = ip; rec.right_j = jp; rec.right_k = kp;
    rec.left = left;
    rec.right = right;
    rec.face_mobility = Mface;
    rec.failure_reason = reason;
    faces[slot] = rec;
}

static const char *reason_name(int reason) {
    switch (reason) {
        case FAILURE_LEFT_ALPHA_OUT_OF_RANGE: return "LEFT_ALPHA_OUT_OF_RANGE";
        case FAILURE_RIGHT_ALPHA_OUT_OF_RANGE: return "RIGHT_ALPHA_OUT_OF_RANGE";
        case FAILURE_LEFT_MATRIX_CONSTITUTIVE: return "LEFT_MATRIX_CONSTITUTIVE";
        case FAILURE_RIGHT_MATRIX_CONSTITUTIVE: return "RIGHT_MATRIX_CONSTITUTIVE";
        case FAILURE_FACE_MOBILITY: return "FACE_MOBILITY";
        default: return "NONE";
    }
}

static void write_cell(FILE *fp, const CellTrace &c) {
    std::fprintf(fp,
        ",%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.0f,%.17e,%.17e,%.17e,%.17e,%.17e,%d,%d,%d",
        c.phi, c.h, c.alpha, c.Ctot, c.q, c.xB, c.active, c.mu,
        c.g_second, c.Gamma, c.M_alpha, c.M_eff,
        c.convex_extension, c.finite, c.bound_ok);
}

int main(int argc, char **argv) {
    const char *face_path = argc > 1 ? argv[1] : "radius3_failed_faces.csv";
    const char *neighbor_path = argc > 2 ? argv[2] : "radius3_failed_face_neighborhood.csv";
    constexpr int N = 16;
    constexpr int total = N * N * N;
    constexpr int max_faces = 256;
    constexpr double radius = 3.0;
    constexpr double width = 0.3;
    constexpr double x_out = 7.8305390835947344e-3;
    constexpr double x_eq = 5.90793940393781469e-3;
    constexpr double D_alpha = 9.0;
    constexpr double temperature_K = 673.15;
    constexpr double energy_scale = 1.3779024e5;
    constexpr double matrix_support_eps = 1.0e-8;

    CellTrace *d_cells = nullptr;
    FaceTrace *d_faces = nullptr;
    int *d_count = nullptr;
    CUDA_OR_DIE(cudaMalloc(&d_cells, total * sizeof(CellTrace)));
    CUDA_OR_DIE(cudaMalloc(&d_faces, max_faces * sizeof(FaceTrace)));
    CUDA_OR_DIE(cudaMalloc(&d_count, sizeof(int)));
    CUDA_OR_DIE(cudaMemset(d_count, 0, sizeof(int)));
    initialize_radius3_cells<<<(total + 255) / 256, 256>>>(
        d_cells, N, radius, width, x_out, x_eq, D_alpha,
        temperature_K, energy_scale, matrix_support_eps);
    for (int axis = 0; axis < 3; ++axis) {
        trace_failed_faces<<<(total + 255) / 256, 256>>>(
            d_cells, d_faces, d_count, max_faces, N, axis);
    }
    CUDA_OR_DIE(cudaDeviceSynchronize());

    int failure_count = 0;
    CUDA_OR_DIE(cudaMemcpy(&failure_count, d_count, sizeof(int),
                           cudaMemcpyDeviceToHost));
    if (failure_count > max_faces) {
        std::fprintf(stderr, "failure buffer overflow: %d > %d\n",
                     failure_count, max_faces);
        return 3;
    }
    std::vector<CellTrace> cells(total);
    std::vector<FaceTrace> faces((size_t)failure_count);
    CUDA_OR_DIE(cudaMemcpy(cells.data(), d_cells, total * sizeof(CellTrace),
                           cudaMemcpyDeviceToHost));
    CUDA_OR_DIE(cudaMemcpy(faces.data(), d_faces,
                           failure_count * sizeof(FaceTrace),
                           cudaMemcpyDeviceToHost));

    FILE *fp = std::fopen(face_path, "w");
    if (!fp) return 4;
    std::fprintf(fp,
        "face_id,direction,left_idx,right_idx,left_i,left_j,left_k,right_i,right_j,right_k"
        ",phi_L,h_L,matrix_capacity_L,Ctot_L,q_L,xB_alpha_L,active_support_L,mu_L,g_second_L,Gamma_L,M_alpha_L,M_eff_L,convex_extension_L,finite_L,bound_L"
        ",phi_R,h_R,matrix_capacity_R,Ctot_R,q_R,xB_alpha_R,active_support_R,mu_R,g_second_R,Gamma_R,M_alpha_R,M_eff_R,convex_extension_R,finite_R,bound_R"
        ",face_average_method,face_mobility,failure_reason\n");
    for (const FaceTrace &f : faces) {
        std::fprintf(fp, "%d,%c,%d,%d,%d,%d,%d,%d,%d,%d",
                     f.face_id, "xyz"[f.axis], f.left_idx, f.right_idx,
                     f.left_i, f.left_j, f.left_k,
                     f.right_i, f.right_j, f.right_k);
        write_cell(fp, f.left);
        write_cell(fp, f.right);
        std::fprintf(fp, ",harmonic_if_both_positive_else_zero,%.17e,%s\n",
                     f.face_mobility, reason_name(f.failure_reason));
    }
    std::fclose(fp);

    FILE *np = std::fopen(neighbor_path, "w");
    if (!np) return 5;
    std::fprintf(np,
        "face_id,endpoint,center_i,center_j,center_k,di,dj,dk,i,j,k,idx,phi,h,matrix_capacity,Ctot,q,xB_alpha,active_support,mu,g_second,Gamma,M_alpha,M_eff,convex_extension,finite,bound_ok\n");
    for (const FaceTrace &f : faces) {
        const int centers[2][3] = {
            {f.left_i, f.left_j, f.left_k},
            {f.right_i, f.right_j, f.right_k}
        };
        for (int endpoint = 0; endpoint < 2; ++endpoint) {
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    for (int dk = -1; dk <= 1; ++dk) {
                        const int i = (centers[endpoint][0] + di + N) % N;
                        const int j = (centers[endpoint][1] + dj + N) % N;
                        const int k = (centers[endpoint][2] + dk + N) % N;
                        const int idx = (i * N + j) * N + k;
                        const CellTrace &c = cells[(size_t)idx];
                        std::fprintf(np,
                            "%d,%c,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.0f,%.17e,%.17e,%.17e,%.17e,%.17e,%d,%d,%d\n",
                            f.face_id, endpoint == 0 ? 'L' : 'R',
                            centers[endpoint][0], centers[endpoint][1],
                            centers[endpoint][2], di, dj, dk, i, j, k, idx,
                            c.phi, c.h, c.alpha, c.Ctot, c.q, c.xB, c.active,
                            c.mu, c.g_second, c.Gamma, c.M_alpha, c.M_eff,
                            c.convex_extension, c.finite, c.bound_ok);
                    }
                }
            }
        }
    }
    std::fclose(np);

    int invalid_cells = 0;
    for (const CellTrace &c : cells) {
        if (!std::isfinite(c.M_eff) || c.M_eff < 0.0) ++invalid_cells;
    }
    std::printf("radius3_failed_face_count=%d\n", failure_count);
    std::printf("radius3_invalid_cell_count=%d\n", invalid_cells);
    std::printf("face_trace=%s\n", face_path);
    std::printf("neighborhood_trace=%s\n", neighbor_path);

    CUDA_OR_DIE(cudaFree(d_count));
    CUDA_OR_DIE(cudaFree(d_faces));
    CUDA_OR_DIE(cudaFree(d_cells));
    return failure_count == 36 ? 0 : 6;
}
