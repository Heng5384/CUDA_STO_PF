#ifndef IO_VTK_CUDA_H
#define IO_VTK_CUDA_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

/**
 * CUDA版本的VTK输出函数
 * 从GPU内存复制数据，重新排列为VTK格式，并写入文件
 * 
 * @param d_field GPU上的场数据（x-y-z顺序，大小为Nx*Ny*Nz）
 * @param Nx, Ny, Nz 网格尺寸
 * @param field_name VTK文件中的标量名称（如"phi"或"xB_tot"）
 * @param step 时间步数（用于文件名）
 * @param fname 输出文件名
 */
static inline int write_vtk_cuda(const double *d_field, 
                                  int Nx, int Ny, int Nz,
                                  const char *field_name,
                                  int step,
                                  const char *fname)
{
    // 分配CPU内存
    size_t data_size = Nx * Ny * Nz * sizeof(double);
    double *h_field = (double*)malloc(data_size);
    if (!h_field) {
        fprintf(stderr, "ERROR: Failed to allocate %zu bytes host memory for VTK output\n", data_size);
        return 0;
    }
    
    // 从GPU复制数据到CPU
    cudaError_t err = cudaMemcpy(h_field, d_field, data_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy failed for file %s: %s\n", fname, cudaGetErrorString(err));
        free(h_field);
        return 0;
    }
    
    // 打开输出文件
    FILE *fo = fopen(fname, "w");
    if (fo == NULL) {
        fprintf(stderr, "ERROR: Cannot create file %s (errno: %d)\n", fname, errno);
        free(h_field);
        return 0;
    }
    
    // 写入VTK文件头
    fprintf(fo, "# vtk DataFile Version 3.0\n");
    fprintf(fo, "Phase Field Simulation\n");
    fprintf(fo, "ASCII\n");
    fprintf(fo, "DATASET STRUCTURED_POINTS\n");
    
    if (Ny == 2) {
        // 2D case: DIMENSIONS Nx 1 Nz
        fprintf(fo, "DIMENSIONS %d %d %d\n", Nx, 1, Nz);
        fprintf(fo, "ASPECT_RATIO 1 1 1\n");
        fprintf(fo, "ORIGIN 0 0 0\n");
        fprintf(fo, "POINT_DATA %d\n", Nx * 1 * Nz);
    } else {
        // 3D case: DIMENSIONS Nx Ny Nz
        fprintf(fo, "DIMENSIONS %d %d %d\n", Nx, Ny, Nz);
        fprintf(fo, "ASPECT_RATIO 1 1 1\n");
        fprintf(fo, "ORIGIN 0 0 0\n");
        fprintf(fo, "POINT_DATA %d\n", Nx * Ny * Nz);
    }
    
    fprintf(fo, "SCALARS %s double 1\n", field_name);
    fprintf(fo, "LOOKUP_TABLE default\n");
    
    // 重新排列数据为VTK格式（z-y-x顺序）
    // GPU数据存储：cuFFT布局是Z轴最快（最内层），然后是Y轴，最后是X轴
    // 索引映射：idx = i * (Ny * Nz) + j * Nz + k
    // VTK需要：z-y-x顺序输出（z最外层，y中间，x最内层）
    // 输出顺序：for k, for j, for i，读取h_field[i * (Ny * Nz) + j * Nz + k]
    
    if (Ny == 2) {
        // 2D模式：只输出j=0层
        for (int k = 0; k < Nz; k++) {      // z (outermost)
            for (int j = 0; j < 1; j++) {   // y (only j=0)
                for (int i = 0; i < Nx; i++) { // x (innermost)
                    // GPU索引：Z轴最快布局 idx = i * (Ny * Nz) + j * Nz + k
                    int idx_gpu = i * (Ny * Nz) + j * Nz + k;
                    fprintf(fo, "%.5f ", h_field[idx_gpu]);
                }
            }
        }
    } else {
        // 3D模式：输出所有y层（z-y-x顺序）
        for (int k = 0; k < Nz; k++) {      // z (outermost)
            for (int j = 0; j < Ny; j++) {  // y
                for (int i = 0; i < Nx; i++) { // x (innermost)
                    // GPU索引：Z轴最快布局 idx = i * (Ny * Nz) + j * Nz + k
                    int idx_gpu = i * (Ny * Nz) + j * Nz + k;
                    fprintf(fo, "%.5f ", h_field[idx_gpu]);
                }
            }
        }
    }
    
    fprintf(fo, "\n");
    fclose(fo);
    free(h_field);
    
    return 1;
}

static inline int read_vtk_ascii_to_host(const char *fname,
                                         int expected_Nx, int expected_Ny, int expected_Nz,
                                         double *h_field,
                                         const char *field_label)
{
    FILE *fp = fopen(fname, "r");
    if (fp == NULL) {
        fprintf(stderr, "ERROR: Cannot open VTK file %s for reading (errno: %d)\n", fname, errno);
        return 0;
    }

    char line[4096];
    int vtk_Nx = -1, vtk_Ny = -1, vtk_Nz = -1;
    int found_dimensions = 0;
    int found_lookup = 0;

    while (fgets(line, sizeof(line), fp) != NULL) {
        if (!found_dimensions && strncmp(line, "DIMENSIONS", 10) == 0) {
            if (sscanf(line, "DIMENSIONS %d %d %d", &vtk_Nx, &vtk_Ny, &vtk_Nz) == 3) {
                found_dimensions = 1;
            }
            continue;
        }
        if (strncmp(line, "LOOKUP_TABLE", 12) == 0) {
            found_lookup = 1;
            break;
        }
    }

    if (!found_dimensions || !found_lookup) {
        fprintf(stderr, "ERROR: Invalid VTK file %s: missing DIMENSIONS or LOOKUP_TABLE header\n", fname);
        fclose(fp);
        return 0;
    }

    int file_is_2d = (vtk_Ny == 1 && expected_Ny == 2);
    if (!(vtk_Nx == expected_Nx &&
          vtk_Nz == expected_Nz &&
          (vtk_Ny == expected_Ny || file_is_2d))) {
        fprintf(stderr,
                "ERROR: VTK grid mismatch for %s (%s): file=%dx%dx%d expected=%dx%dx%d\n",
                fname, (field_label ? field_label : "field"),
                vtk_Nx, vtk_Ny, vtk_Nz,
                expected_Nx, expected_Ny, expected_Nz);
        fclose(fp);
        return 0;
    }

    size_t total_expected = (size_t)expected_Nx * (size_t)expected_Ny * (size_t)expected_Nz;
    for (size_t idx = 0; idx < total_expected; ++idx) {
        h_field[idx] = 0.0;
    }

    if (file_is_2d) {
        for (int k = 0; k < expected_Nz; ++k) {
            for (int i = 0; i < expected_Nx; ++i) {
                double value = 0.0;
                if (fscanf(fp, "%lf", &value) != 1) {
                    fprintf(stderr, "ERROR: Unexpected EOF while reading 2D VTK data from %s\n", fname);
                    fclose(fp);
                    return 0;
                }
                for (int j = 0; j < expected_Ny; ++j) {
                    int idx_gpu = i * (expected_Ny * expected_Nz) + j * expected_Nz + k;
                    h_field[idx_gpu] = value;
                }
            }
        }
    } else {
        for (int k = 0; k < expected_Nz; ++k) {
            for (int j = 0; j < expected_Ny; ++j) {
                for (int i = 0; i < expected_Nx; ++i) {
                    double value = 0.0;
                    if (fscanf(fp, "%lf", &value) != 1) {
                        fprintf(stderr, "ERROR: Unexpected EOF while reading VTK data from %s\n", fname);
                        fclose(fp);
                        return 0;
                    }
                    int idx_gpu = i * (expected_Ny * expected_Nz) + j * expected_Nz + k;
                    h_field[idx_gpu] = value;
                }
            }
        }
    }

    fclose(fp);
    return 1;
}

#endif // IO_VTK_CUDA_H
