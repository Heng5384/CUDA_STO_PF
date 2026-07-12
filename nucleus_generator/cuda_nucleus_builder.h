#ifndef CUDA_NUCLEUS_BUILDER_H
#define CUDA_NUCLEUS_BUILDER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum CudaNucleusScaleMode {
    CUDA_NUCLEUS_SCALE_DIRECT_RESAMPLE = 0,
    CUDA_NUCLEUS_SCALE_DIFFUSE_RECONSTRUCTION = 1
} CudaNucleusScaleMode;

typedef struct CudaNucleusMetadata {
    int nx;
    int ny;
    int nz;
    double dx_nm;
    double dy_nm;
    double dz_nm;
    double r_eff_nm;
    double r_eff_over_dx;
    double interface_width_nm;
    double center_nm[3];
    double semiaxes_nm[3];
    double total_xB_before;
    double total_xB_after;
    double total_xB_error;
    double h_volume_nm3;
    double target_volume_nm3;
    double phi_min;
    double phi_max;
    double phi_grad_max;
    int connected_components_phi_gt_0p5;
    int scale_mode;
    int insertion_ready;
} CudaNucleusMetadata;

typedef struct CudaNucleusObject {
    double *phi_init;
    double *xB_init;
    CudaNucleusMetadata metadata;
} CudaNucleusObject;

static inline size_t cuda_nucleus_flat_index(const CudaNucleusMetadata *m, int i, int j, int k) {
    return ((size_t)i * (size_t)m->ny + (size_t)j) * (size_t)m->nz + (size_t)k;
}

static inline int cuda_nucleus_metadata_compatible(const CudaNucleusMetadata *m) {
    if (!m) return 0;
    if (m->nx <= 0 || m->ny <= 0 || m->nz <= 0) return 0;
    if (m->dx_nm <= 0.0 || m->dy_nm <= 0.0 || m->dz_nm <= 0.0) return 0;
    if (m->r_eff_nm <= 0.0 || m->interface_width_nm <= 0.0) return 0;
    if (m->phi_min < -1.0e-10 || m->phi_max > 1.0 + 1.0e-10) return 0;
    if (m->connected_components_phi_gt_0p5 != 1) return 0;
    if (m->total_xB_error < -1.0e-8 || m->total_xB_error > 1.0e-8) return 0;
    return m->insertion_ready ? 1 : 0;
}

#ifdef __cplusplus
}
#endif

#endif
