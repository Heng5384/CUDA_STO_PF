#include "cuda_nucleus_builder.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>

static bool read_file(const std::string &path, std::string *out) {
    std::ifstream f(path.c_str(), std::ios::in | std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    *out = ss.str();
    return true;
}

static bool extract_number(const std::string &text, const std::string &key, double *value) {
    const std::regex re("\"" + key + "\"\\s*:\\s*(-?(?:[0-9]+\\.?[0-9]*|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?)");
    std::smatch m;
    if (!std::regex_search(text, m, re)) return false;
    *value = std::atof(m[1].str().c_str());
    return true;
}

static bool extract_int(const std::string &text, const std::string &key, int *value) {
    double tmp = 0.0;
    if (!extract_number(text, key, &tmp)) return false;
    *value = static_cast<int>(std::llround(tmp));
    return true;
}

static bool extract_bool(const std::string &text, const std::string &key, int *value) {
    const std::regex re("\"" + key + "\"\\s*:\\s*(true|false)");
    std::smatch m;
    if (!std::regex_search(text, m, re)) return false;
    *value = (m[1].str() == "true") ? 1 : 0;
    return true;
}

static CudaNucleusMetadata metadata_from_json(const std::string &text) {
    CudaNucleusMetadata m = {};
    extract_int(text, "nx", &m.nx);
    extract_int(text, "ny", &m.ny);
    extract_int(text, "nz", &m.nz);
    extract_number(text, "dx_nm", &m.dx_nm);
    extract_number(text, "dy_nm", &m.dy_nm);
    extract_number(text, "dz_nm", &m.dz_nm);
    extract_number(text, "r_eff_nm", &m.r_eff_nm);
    extract_number(text, "r_eff_over_dx", &m.r_eff_over_dx);
    extract_number(text, "interface_width_nm", &m.interface_width_nm);
    extract_number(text, "total_xB_before", &m.total_xB_before);
    extract_number(text, "total_xB_after", &m.total_xB_after);
    extract_number(text, "total_xB_error", &m.total_xB_error);
    extract_number(text, "h_volume_nm3", &m.h_volume_nm3);
    extract_number(text, "target_volume_nm3", &m.target_volume_nm3);
    extract_number(text, "phi_min", &m.phi_min);
    extract_number(text, "phi_max", &m.phi_max);
    extract_number(text, "phi_grad_max", &m.phi_grad_max);
    extract_int(text, "connected_components_phi_gt_threshold", &m.connected_components_phi_gt_0p5);
    extract_bool(text, "insertion_ready", &m.insertion_ready);
    if (text.find("\"diffuse_reconstruction") != std::string::npos) {
        m.scale_mode = CUDA_NUCLEUS_SCALE_DIFFUSE_RECONSTRUCTION;
    } else {
        m.scale_mode = CUDA_NUCLEUS_SCALE_DIRECT_RESAMPLE;
    }
    return m;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: nucleus_generator_check <nucleus_metadata.json>\n";
        return 2;
    }
    std::string text;
    if (!read_file(argv[1], &text)) {
        std::cerr << "failed_to_read_metadata=" << argv[1] << "\n";
        return 2;
    }
    CudaNucleusMetadata meta = metadata_from_json(text);
    const int ok = cuda_nucleus_metadata_compatible(&meta);
    std::cout << "nucleus_metadata=" << argv[1] << "\n";
    std::cout << "grid=" << meta.nx << "x" << meta.ny << "x" << meta.nz << "\n";
    std::cout << "dx_nm=" << meta.dx_nm << "\n";
    std::cout << "r_eff_nm=" << meta.r_eff_nm << "\n";
    std::cout << "r_eff_over_dx=" << meta.r_eff_over_dx << "\n";
    std::cout << "scale_mode=" << (meta.scale_mode == CUDA_NUCLEUS_SCALE_DIFFUSE_RECONSTRUCTION ? "diffuse_reconstruction" : "direct_resample") << "\n";
    std::cout << "insertion_ready=" << (ok ? "true" : "false") << "\n";
    return ok ? 0 : 1;
}
