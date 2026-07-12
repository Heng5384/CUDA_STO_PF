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

static bool extract_vector3(const std::string &text, const std::string &key, double v[3]) {
    const std::regex re(key + "\\s*:\\s*[\\[(]\\s*([-+0-9.eE]+)\\s*,\\s*([-+0-9.eE]+)\\s*,\\s*([-+0-9.eE]+)");
    std::smatch m;
    if (!std::regex_search(text, m, re)) return false;
    v[0] = std::atof(m[1].str().c_str());
    v[1] = std::atof(m[2].str().c_str());
    v[2] = std::atof(m[3].str().c_str());
    return true;
}

static bool extract_scalar(const std::string &text, const std::string &key, double *v) {
    const std::regex re(key + "\\s*:\\s*([-+0-9.eE]+)");
    std::smatch m;
    if (!std::regex_search(text, m, re)) return false;
    *v = std::atof(m[1].str().c_str());
    return true;
}

static bool extract_int_tuple3(const std::string &text, const std::string &key, int v[3]) {
    const std::regex re(key + "\\s*:\\s*[\\[(]\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*,\\s*([0-9]+)");
    std::smatch m;
    if (!std::regex_search(text, m, re)) return false;
    v[0] = std::atoi(m[1].str().c_str());
    v[1] = std::atoi(m[2].str().c_str());
    v[2] = std::atoi(m[3].str().c_str());
    return true;
}

static void normalize(double v[3]) {
    const double n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (n <= 1.0e-30) return;
    v[0] /= n;
    v[1] /= n;
    v[2] /= n;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "usage: dynamic_continue_parser <summary.txt> <out_descriptor.json> [--xB value] [--override-grid nx ny nz] [--center-mode grid_center]\n";
        return 2;
    }
    const std::string summary_path = argv[1];
    const std::string out_path = argv[2];
    double matrix_xB = 0.03;
    int override_grid[3] = {0, 0, 0};
    bool center_on_grid = false;
    for (int i = 3; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--xB") {
            matrix_xB = std::atof(argv[i + 1]);
            ++i;
        } else if (std::string(argv[i]) == "--override-grid" && i + 3 < argc) {
            override_grid[0] = std::atoi(argv[i + 1]);
            override_grid[1] = std::atoi(argv[i + 2]);
            override_grid[2] = std::atoi(argv[i + 3]);
            i += 3;
        } else if (std::string(argv[i]) == "--center-mode") {
            center_on_grid = (std::string(argv[i + 1]) == "grid_center");
            ++i;
        }
    }

    std::string text;
    if (!read_file(summary_path, &text)) {
        std::cerr << "failed_to_read_summary=" << summary_path << "\n";
        return 2;
    }

    int grid[3] = {0, 0, 0};
    double spacing[3] = {0.1, 0.1, 0.1};
    double center[3] = {0.0, 0.0, 0.0};
    double axes[3][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    };
    double l1 = 0.0, l2 = 0.0, l3 = 0.0;
    extract_int_tuple3(text, "grid_dimensions", grid);
    extract_vector3(text, "spacing_sim_units", spacing);
    if (!extract_vector3(text, "center_of_mass", center)) {
        std::cerr << "missing center_of_mass in " << summary_path << "\n";
        return 2;
    }
    extract_vector3(text, "long_axis", axes[0]);
    extract_vector3(text, "mid_axis", axes[1]);
    extract_vector3(text, "short_axis", axes[2]);
    normalize(axes[0]);
    normalize(axes[1]);
    normalize(axes[2]);
    extract_scalar(text, "L1_long", &l1);
    extract_scalar(text, "L2_mid", &l2);
    extract_scalar(text, "L3_short", &l3);
    if (l1 <= 0.0 || l2 <= 0.0 || l3 <= 0.0) {
        std::cerr << "missing principal lengths in " << summary_path << "\n";
        return 2;
    }
    const double a = 0.5 * l1;
    const double b = 0.5 * l2;
    const double c = 0.5 * l3;
    const double r_eff = std::cbrt(std::max(a * b * c, 1.0e-300));
    const double aspect13 = (c > 0.0) ? (a / c) : 1.0;
    const bool anisotropic = std::fabs(aspect13 - 1.0) > 0.05;
    if (override_grid[0] > 0 && override_grid[1] > 0 && override_grid[2] > 0) {
        grid[0] = override_grid[0];
        grid[1] = override_grid[1];
        grid[2] = override_grid[2];
    }
    if (center_on_grid) {
        center[0] = 0.5 * grid[0] * spacing[0];
        center[1] = 0.5 * grid[1] * spacing[1];
        center[2] = 0.5 * grid[2] * spacing[2];
    }

    std::ofstream out(out_path.c_str());
    if (!out) {
        std::cerr << "failed_to_write_descriptor=" << out_path << "\n";
        return 2;
    }
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"source_summary\": \"" << summary_path << "\",\n";
    out << "  \"shape_type\": \"" << (anisotropic ? "anisotropic" : "isotropic") << "\",\n";
    out << "  \"nx\": " << grid[0] << ", \"ny\": " << grid[1] << ", \"nz\": " << grid[2] << ",\n";
    out << "  \"dx_nm\": " << spacing[0] << ", \"dy_nm\": " << spacing[1] << ", \"dz_nm\": " << spacing[2] << ",\n";
    out << "  \"center_nm\": [" << center[0] << ", " << center[1] << ", " << center[2] << "],\n";
    out << "  \"r_eff_nm\": " << r_eff << ",\n";
    out << "  \"semiaxes_nm\": [" << a << ", " << b << ", " << c << "],\n";
    out << "  \"aspect_ratio\": [" << (a / r_eff) << ", " << (b / r_eff) << ", " << (c / r_eff) << "],\n";
    out << "  \"matrix_xB\": " << matrix_xB << ",\n";
    out << "  \"principal_axes\": [\n";
    for (int row = 0; row < 3; ++row) {
        out << "    [" << axes[row][0] << ", " << axes[row][1] << ", " << axes[row][2] << "]";
        out << (row == 2 ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"parametric_phi\": \"" << (anisotropic ? "exp(-(x-x0)^T A (x-x0))" : "exp(-|x-x0|^2/r_eff^2)") << "\",\n";
    out << "  \"volume_definition\": \"4/3*pi*semiaxes_product; amplitude rescales phi only\"\n";
    out << "}\n";
    out.close();

    std::cout << "parametric_descriptor=" << out_path << "\n";
    std::cout << "r_eff_nm=" << r_eff << "\n";
    std::cout << "shape_type=" << (anisotropic ? "anisotropic" : "isotropic") << "\n";
    return 0;
}
