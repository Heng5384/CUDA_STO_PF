#include <cufft.h>
#include <cstdio>
#include <cstdlib>

static void check(cufftResult result, const char *what) {
    if (result != CUFFT_SUCCESS) {
        std::fprintf(stderr, "%s failed: cufftResult=%d\n", what, (int)result);
        std::exit(2);
    }
}

static size_t plan_bytes(int nx, int ny, int nz, cufftType type) {
    cufftHandle plan = 0;
    size_t work = 0;
    check(cufftCreate(&plan), "cufftCreate");
    check(cufftSetAutoAllocation(plan, 0), "cufftSetAutoAllocation");
    check(cufftMakePlan3d(plan, nx, ny, nz, type, &work), "cufftMakePlan3d");
    check(cufftDestroy(plan), "cufftDestroy");
    return work;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s NX NY NZ\n", argv[0]);
        return 2;
    }
    const int nx = std::atoi(argv[1]);
    const int ny = std::atoi(argv[2]);
    const int nz = std::atoi(argv[3]);
    const size_t d2z = plan_bytes(nx, ny, nz, CUFFT_D2Z);
    const size_t z2d = plan_bytes(nx, ny, nz, CUFFT_Z2D);
    const size_t r2c = plan_bytes(nx, ny, nz, CUFFT_R2C);
    const size_t c2r = plan_bytes(nx, ny, nz, CUFFT_C2R);
    std::printf("grid,d2z_work_bytes,z2d_work_bytes,r2c_work_bytes,c2r_work_bytes,double_plan_peak_bytes,elastic_plan_peak_bytes\n");
    std::printf("%dx%dx%d,%zu,%zu,%zu,%zu,%zu,%zu\n", nx, ny, nz,
                d2z, z2d, r2c, c2r, d2z + z2d, r2c + c2r);
    return 0;
}
