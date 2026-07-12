#include "cuda_nucleus_eval.cu"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

static bool read_text_file(const std::string &path, std::string *out) {
    std::ifstream f(path.c_str(), std::ios::in | std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    *out = ss.str();
    return true;
}

static bool extract_number_json(const std::string &text, const std::string &key, double *value) {
    const std::regex re("\"" + key + "\"\\s*:\\s*(-?(?:[0-9]+\\.?[0-9]*|\\.[0-9]+)(?:[eE][+-]?[0-9]+)?)");
    std::smatch m;
    if (!std::regex_search(text, m, re)) return false;
    *value = std::atof(m[1].str().c_str());
    return true;
}

static bool extract_int_json(const std::string &text, const std::string &key, int *value) {
    double tmp = 0.0;
    if (!extract_number_json(text, key, &tmp)) return false;
    *value = static_cast<int>(tmp);
    return true;
}

static bool extract_array3_json(const std::string &text, const std::string &key, double value[3]) {
    const std::regex re("\"" + key + "\"\\s*:\\s*\\[\\s*([-+0-9.eE]+)\\s*,\\s*([-+0-9.eE]+)\\s*,\\s*([-+0-9.eE]+)");
    std::smatch m;
    if (!std::regex_search(text, m, re)) return false;
    value[0] = std::atof(m[1].str().c_str());
    value[1] = std::atof(m[2].str().c_str());
    value[2] = std::atof(m[3].str().c_str());
    return true;
}

static bool descriptor_is_anisotropic(const std::string &text) {
    return text.find("\"shape_type\": \"anisotropic\"") != std::string::npos;
}

static void extract_axes_json(const std::string &text, double axes[3][3]) {
    const std::regex re("\"principal_axes\"\\s*:\\s*\\[\\s*\\[\\s*([-+0-9.eE]+)\\s*,\\s*([-+0-9.eE]+)\\s*,\\s*([-+0-9.eE]+)\\s*\\]\\s*,\\s*\\[\\s*([-+0-9.eE]+)\\s*,\\s*([-+0-9.eE]+)\\s*,\\s*([-+0-9.eE]+)\\s*\\]\\s*,\\s*\\[\\s*([-+0-9.eE]+)\\s*,\\s*([-+0-9.eE]+)\\s*,\\s*([-+0-9.eE]+)");
    std::smatch m;
    if (!std::regex_search(text, m, re)) return;
    int idx = 1;
    for (int a = 0; a < 3; ++a) {
        for (int c = 0; c < 3; ++c) {
            axes[a][c] = std::atof(m[idx++].str().c_str());
        }
    }
}

static bool load_descriptor(const std::string &path, ParametricNucleusSpec *s) {
    std::string text;
    if (!read_text_file(path, &text)) return false;
    extract_int_json(text, "nx", &s->nx);
    extract_int_json(text, "ny", &s->ny);
    extract_int_json(text, "nz", &s->nz);
    extract_number_json(text, "dx_nm", &s->dx_nm);
    extract_number_json(text, "dy_nm", &s->dy_nm);
    extract_number_json(text, "dz_nm", &s->dz_nm);
    extract_number_json(text, "r_eff_nm", &s->r_eff_nm);
    extract_number_json(text, "matrix_xB", &s->matrix_xB);
    extract_array3_json(text, "center_nm", s->center);
    extract_array3_json(text, "semiaxes_nm", s->semiaxes_nm);
    extract_axes_json(text, s->axes);
    s->anisotropic = descriptor_is_anisotropic(text);
    return s->nx > 0 && s->ny > 0 && s->nz > 0 && s->dx_nm > 0.0 && s->r_eff_nm > 0.0;
}

static bool write_raw_float32(const std::string &path, const std::vector<float> &v) {
    std::ofstream f(path.c_str(), std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char *>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    return static_cast<bool>(f);
}

static bool write_meta(
    const std::string &path,
    const ParametricNucleusSpec &s,
    const ParametricNucleusStats &st,
    const std::string &descriptor_path) {
    std::ofstream out(path.c_str());
    if (!out) return false;
    out << std::setprecision(12);
    out << "{\n";
    out << "  \"Nx\": " << s.nx << ",\n";
    out << "  \"Ny\": " << s.ny << ",\n";
    out << "  \"Nz\": " << s.nz << ",\n";
    out << "  \"dx_nm\": " << s.dx_nm << ",\n";
    out << "  \"dy_nm\": " << s.dy_nm << ",\n";
    out << "  \"dz_nm\": " << s.dz_nm << ",\n";
    out << "  \"interface_width_nm\": 0.6,\n";
    out << "  \"dtype\": \"float32\",\n";
    out << "  \"order\": \"C\",\n";
    out << "  \"phi_raw\": \"phi_init.raw\",\n";
    out << "  \"xB_raw\": \"xB_init.raw\",\n";
    out << "  \"parametric_descriptor\": \"" << descriptor_path << "\",\n";
    out << "  \"mean_xBtot\": " << st.mean_xBtot_after << ",\n";
    out << "  \"xB_max_safe\": " << st.xB_max << ",\n";
    out << "  \"mean_hphi\": " << st.mean_hphi << ",\n";
    out << "  \"mean_phi_volume_nm3\": " << st.phi_integral_nm3 << ",\n";
    out << "  \"amplitude\": " << st.amplitude << ",\n";
    out << "  \"layout_note\": \"raw files are C order, idx=i*(Ny*Nz)+j*Nz+k\"\n";
    out << "}\n";
    return true;
}

static bool write_acceptance(
    const std::string &path,
    const ParametricNucleusSpec &s,
    const ParametricNucleusStats &st,
    const std::string &runtime_status) {
    std::ofstream out(path.c_str());
    if (!out) return false;
    out << std::setprecision(12);
    out << "{\n";
    out << "  \"accepted\": " << (st.accepted ? "true" : "false") << ",\n";
    out << "  \"runtime_status\": \"" << runtime_status << "\",\n";
    out << "  \"criteria\": {\n";
    out << "    \"finite_fields\": " << (st.finite ? "true" : "false") << ",\n";
    out << "    \"mass_error_lt_1e_minus_10\": " << (st.mass_error_abs < 1.0e-10 ? "true" : "false") << ",\n";
    out << "    \"single_component\": " << (st.connected_components == 1 ? "true" : "false") << ",\n";
    out << "    \"phi_no_overflow\": " << (st.phi_max <= 1.0 + 1.0e-12 ? "true" : "false") << "\n";
    out << "  },\n";
    out << "  \"grid\": [" << s.nx << ", " << s.ny << ", " << s.nz << "],\n";
    out << "  \"dx_nm\": " << s.dx_nm << ",\n";
    out << "  \"r_eff_nm\": " << s.r_eff_nm << ",\n";
    out << "  \"anisotropic\": " << (s.anisotropic ? "true" : "false") << ",\n";
    out << "  \"amplitude\": " << st.amplitude << ",\n";
    out << "  \"target_phi_volume_nm3\": " << st.target_phi_volume_nm3 << ",\n";
    out << "  \"phi_integral_nm3\": " << st.phi_integral_nm3 << ",\n";
    out << "  \"mean_hphi\": " << st.mean_hphi << ",\n";
    out << "  \"mass_error\": " << st.mass_error << ",\n";
    out << "  \"phi_min\": " << st.phi_min << ",\n";
    out << "  \"phi_max\": " << st.phi_max << ",\n";
    out << "  \"xB_min\": " << st.xB_min << ",\n";
    out << "  \"xB_max\": " << st.xB_max << ",\n";
    out << "  \"grad_phi_max\": " << st.grad_phi_max << ",\n";
    out << "  \"connected_components\": " << st.connected_components << ",\n";
    out << "  \"largest_component_voxels\": " << st.largest_component_voxels << "\n";
    out << "}\n";
    return true;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "usage: parametric_nucleus_builder <descriptor.json> <out_dir> [runtime_status]\n";
        return 2;
    }
    const std::string descriptor_path = argv[1];
    const std::string out_dir = argv[2];
    const std::string runtime_status = (argc >= 4) ? argv[3] : "not_run";
    ParametricNucleusSpec spec;
    if (!load_descriptor(descriptor_path, &spec)) {
        std::cerr << "failed_to_load_descriptor=" << descriptor_path << "\n";
        return 2;
    }
    std::vector<float> phi;
    std::vector<float> xb;
    ParametricNucleusStats stats = build_parametric_nucleus(spec, &phi, &xb, 1.0e-10);
    const std::string phi_path = out_dir + "/phi_init.raw";
    const std::string xb_path = out_dir + "/xB_init.raw";
    const std::string meta_path = out_dir + "/init_meta.json";
    const std::string acceptance_path = out_dir + "/acceptance_summary.json";
    if (!write_raw_float32(phi_path, phi) || !write_raw_float32(xb_path, xb)) {
        std::cerr << "failed_to_write_raw_fields in " << out_dir << "\n";
        return 2;
    }
    if (!write_meta(meta_path, spec, stats, descriptor_path)) {
        std::cerr << "failed_to_write_meta=" << meta_path << "\n";
        return 2;
    }
    if (!write_acceptance(acceptance_path, spec, stats, runtime_status)) {
        std::cerr << "failed_to_write_acceptance=" << acceptance_path << "\n";
        return 2;
    }
    std::cout << "phi_raw=" << phi_path << "\n";
    std::cout << "xB_raw=" << xb_path << "\n";
    std::cout << "init_meta=" << meta_path << "\n";
    std::cout << "acceptance_summary=" << acceptance_path << "\n";
    std::cout << "accepted=" << (stats.accepted ? "true" : "false") << "\n";
    std::cout << "mass_error=" << stats.mass_error << "\n";
    return stats.accepted ? 0 : 1;
}
