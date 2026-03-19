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

#endif // IO_VTK_CUDA_H

