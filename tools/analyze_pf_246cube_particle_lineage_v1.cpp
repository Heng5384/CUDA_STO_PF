// Deterministic periodic 3-D particle tracker for large PF checkpoints.
//
// This intentionally uses only the C++ standard library so the same source
// can run inside workstation and cluster allocations without a Python/SciPy
// dependency.  Identities propagate by voxel overlap.  Any merge, split, or
// overlap-free new component fails closed.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct Options {
    fs::path initial_phi;
    fs::path initial_xb;
    std::vector<fs::path> checkpoints;
    fs::path out;
    int nx = 246;
    int ny = 246;
    int nz = 246;
    double dx_nm = 1.0;
    double threshold = 1.0e-4;
    double physical_dt_s = 0.9909260953431841;
    double start_age_h = 6.0;
    double target_mean = 0.03;
    int expected_initial_count = 96;
    bool allow_dissolution = false;
    bool allow_merge_groups = false;
};

struct CheckpointHeader {
    uint32_t version = 0;
    uint32_t header_bytes = 0;
    uint64_t count = 0;
    uint64_t step = 0;
    int32_t nx = 0;
    int32_t ny = 0;
    int32_t nz = 0;
    double dt = 0.0;
    double temperature = 0.0;
    double target = 0.0;
};

struct FieldState {
    uint64_t step = 0;
    double dt = 0.02;
    double temperature = 380.0;
    double target_mass = 0.0;
    std::vector<double> phi;
    std::vector<double> xb;
};

struct Component {
    int dense_label = -1;
    int stable_id = -1;
    std::vector<int> lineage_members;
    uint64_t voxels = 0;
    double h_volume_nm3 = 0.0;
    double radius_nm = 0.0;
    std::array<double, 3> centroid_nm{0.0, 0.0, 0.0};
};

struct AnalysisState {
    uint64_t step = 0;
    double dt = 0.02;
    double temperature = 380.0;
    double target_mass = 0.0;
    std::vector<int32_t> labels;
    std::vector<Component> components;
    double beta_volume_fraction = 0.0;
    double mean_radius_nm = 0.0;
    double sv_nm_inv = 0.0;
    double m6_nm3 = 0.0;
    double broad_matrix_xb = 0.0;
    double broad_matrix_xag = 0.0;
    double far3_matrix_xb = 0.0;
    double far3_matrix_xag = 0.0;
    double far4_matrix_xb = 0.0;
    double far4_matrix_xag = 0.0;
    double far3_fraction = 0.0;
    double far4_fraction = 0.0;
    double far3_far4_xag_delta = 0.0;
    double canonical_inventory = 0.0;
    double mass_relative_error = 0.0;
    double phi_normalized_l1_from_initial = 0.0;
    double xb_mean_absolute_from_initial = 0.0;
    double phi_min = 0.0;
    double phi_max = 0.0;
    double xb_min = 0.0;
    double xb_max = 0.0;
    bool finite = true;
    bool bounds = true;
};

struct Event {
    uint64_t step = 0;
    std::string type;
    int stable_id = -1;
    int parent_count = 0;
    int child_count = 0;
    bool qualified_dissolution = false;
    bool qualified_merge = false;
    std::vector<int> parent_stable_ids;
    int child_stable_id = -1;
    std::string detail;
};

double h_of_phi(double phi) {
    const double p2 = phi * phi;
    return p2 * phi * (6.0 * p2 - 15.0 * phi + 10.0);
}

void squared_edt_1d(
    const std::vector<double> &input,
    std::vector<double> &output) {
    const int length = static_cast<int>(input.size());
    std::vector<int> sites(length);
    std::vector<double> boundaries(length + 1);
    int top = 0;
    sites[0] = 0;
    boundaries[0] = -std::numeric_limits<double>::infinity();
    boundaries[1] = std::numeric_limits<double>::infinity();
    for (int q = 1; q < length; ++q) {
        double intersection = 0.0;
        while (true) {
            const int site = sites[top];
            intersection =
                ((input[q] + static_cast<double>(q) * q) -
                 (input[site] + static_cast<double>(site) * site)) /
                (2.0 * (q - site));
            if (intersection > boundaries[top] || top == 0) {
                break;
            }
            --top;
        }
        ++top;
        sites[top] = q;
        boundaries[top] = intersection;
        boundaries[top + 1] = std::numeric_limits<double>::infinity();
    }
    top = 0;
    output.resize(length);
    for (int q = 0; q < length; ++q) {
        while (boundaries[top + 1] < q) {
            ++top;
        }
        const double delta = q - sites[top];
        output[q] = delta * delta + input[sites[top]];
    }
}

std::vector<double> periodic_squared_distance_to_beta(
    const std::vector<double> &phi,
    const Options &options) {
    const uint64_t count = phi.size();
    constexpr double far_value = 1.0e12;
    std::vector<double> distance(count, far_value);
    bool any_beta = false;
    for (size_t index = 0; index < phi.size(); ++index) {
        if (phi[index] >= 0.5) {
            distance[index] = 0.0;
            any_beta = true;
        }
    }
    if (!any_beta) {
        throw std::runtime_error("no phi>=0.5 beta support for distance audit");
    }
    std::vector<double> tripled;
    std::vector<double> transformed;
    auto transform = [&](int length, auto index_of) {
        tripled.resize(3 * length);
        for (int position = 0; position < length; ++position) {
            const double value = distance[index_of(position)];
            tripled[position] = value;
            tripled[position + length] = value;
            tripled[position + 2 * length] = value;
        }
        squared_edt_1d(tripled, transformed);
        for (int position = 0; position < length; ++position) {
            distance[index_of(position)] = transformed[position + length];
        }
    };
    // z lines
    for (int x = 0; x < options.nx; ++x) {
        for (int y = 0; y < options.ny; ++y) {
            const uint64_t base =
                (static_cast<uint64_t>(x) * options.ny + y) * options.nz;
            transform(options.nz, [&](int z) {
                return base + static_cast<uint64_t>(z);
            });
        }
    }
    // y lines
    for (int x = 0; x < options.nx; ++x) {
        for (int z = 0; z < options.nz; ++z) {
            transform(options.ny, [&](int y) {
                return (static_cast<uint64_t>(x) * options.ny + y) *
                           options.nz +
                       z;
            });
        }
    }
    // x lines
    for (int y = 0; y < options.ny; ++y) {
        for (int z = 0; z < options.nz; ++z) {
            transform(options.nx, [&](int x) {
                return (static_cast<uint64_t>(x) * options.ny + y) *
                           options.nz +
                       z;
            });
        }
    }
    return distance;
}

template <typename T>
T read_scalar(std::ifstream &in, const fs::path &path) {
    T value{};
    in.read(reinterpret_cast<char *>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("short checkpoint header: " + path.string());
    }
    return value;
}

CheckpointHeader read_header(std::ifstream &in, const fs::path &path) {
    std::array<char, 8> magic{};
    in.read(magic.data(), magic.size());
    if (!in) {
        throw std::runtime_error("short checkpoint prefix: " + path.string());
    }
    CheckpointHeader result;
    result.version = read_scalar<uint32_t>(in, path);
    result.header_bytes = read_scalar<uint32_t>(in, path);
    const std::string magic_text(magic.data(), magic.size());
    const bool is_v2 =
        magic_text == "PFZMCHK2" && result.version == 2 &&
        result.header_bytes == 512;
    const bool is_v3 =
        magic_text == "PFZMCHK3" && result.version == 3 &&
        result.header_bytes == 800;
    const bool is_v4 =
        magic_text == "PFZMCHK4" && result.version == 4 &&
        result.header_bytes == 904;
    if (!(is_v2 || is_v3 || is_v4)) {
        throw std::runtime_error("unsupported checkpoint schema: " + path.string());
    }
    result.count = read_scalar<uint64_t>(in, path);
    if (is_v4) {
        const uint64_t k_count = read_scalar<uint64_t>(in, path);
        result.step = read_scalar<uint64_t>(in, path);
        result.nx = read_scalar<int32_t>(in, path);
        result.ny = read_scalar<int32_t>(in, path);
        result.nz = read_scalar<int32_t>(in, path);
        const int32_t elastic_state_present =
            read_scalar<int32_t>(in, path);
        if (elastic_state_present != 1 ||
            k_count != static_cast<uint64_t>(result.nx) *
                           static_cast<uint64_t>(result.ny) *
                           static_cast<uint64_t>(result.nz / 2 + 1)) {
            throw std::runtime_error(
                "checkpoint V4 elastic-state/grid mismatch: " +
                path.string());
        }
    } else {
        result.step = read_scalar<uint64_t>(in, path);
        result.nx = read_scalar<int32_t>(in, path);
        result.ny = read_scalar<int32_t>(in, path);
        result.nz = read_scalar<int32_t>(in, path);
        (void)read_scalar<int32_t>(in, path);
    }
    result.dt = read_scalar<double>(in, path);
    result.temperature = read_scalar<double>(in, path);
    result.target = read_scalar<double>(in, path);
    (void)read_scalar<double>(in, path);
    (void)read_scalar<double>(in, path);
    (void)read_scalar<double>(in, path);
    if (
        result.count != static_cast<uint64_t>(result.nx) *
                            static_cast<uint64_t>(result.ny) *
                            static_cast<uint64_t>(result.nz)) {
        throw std::runtime_error("checkpoint count/grid mismatch: " + path.string());
    }
    in.seekg(result.header_bytes, std::ios::beg);
    if (!in) {
        throw std::runtime_error("checkpoint header seek failed: " + path.string());
    }
    return result;
}

std::vector<double> read_raw(const fs::path &path, uint64_t count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open raw field: " + path.string());
    }
    std::vector<double> values(count);
    in.read(
        reinterpret_cast<char *>(values.data()),
        static_cast<std::streamsize>(count * sizeof(double)));
    if (!in || in.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("raw field size mismatch: " + path.string());
    }
    return values;
}

FieldState read_checkpoint(const fs::path &path, const Options &options) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open checkpoint: " + path.string());
    }
    const auto header = read_header(in, path);
    if (
        header.nx != options.nx || header.ny != options.ny ||
        header.nz != options.nz) {
        throw std::runtime_error("checkpoint grid mismatch: " + path.string());
    }
    FieldState result;
    result.step = header.step;
    result.dt = header.dt;
    result.temperature = header.temperature;
    result.target_mass = header.target;
    result.phi.resize(header.count);
    result.xb.resize(header.count);
    in.read(
        reinterpret_cast<char *>(result.phi.data()),
        static_cast<std::streamsize>(header.count * sizeof(double)));
    if (!in) {
        throw std::runtime_error("short checkpoint phi: " + path.string());
    }
    // Skip Y.
    in.seekg(
        static_cast<std::streamoff>(header.count * sizeof(double)),
        std::ios::cur);
    in.read(
        reinterpret_cast<char *>(result.xb.data()),
        static_cast<std::streamsize>(header.count * sizeof(double)));
    if (!in) {
        throw std::runtime_error("short checkpoint xB: " + path.string());
    }
    return result;
}

struct Dsu {
    explicit Dsu(size_t count) : parent(count, -1) {}
    std::vector<int32_t> parent;

    int32_t find(int32_t value) {
        int32_t root = value;
        while (parent[root] != root) {
            root = parent[root];
        }
        while (parent[value] != value) {
            const int32_t next = parent[value];
            parent[value] = root;
            value = next;
        }
        return root;
    }

    void unite(int32_t left, int32_t right) {
        int32_t a = find(left);
        int32_t b = find(right);
        if (a == b) {
            return;
        }
        if (a > b) {
            std::swap(a, b);
        }
        parent[b] = a;
    }
};

AnalysisState analyze_fields(
    const FieldState &field,
    const Options &options,
    const std::vector<int> *stable_ids = nullptr) {
    const uint64_t count = static_cast<uint64_t>(options.nx) * options.ny *
                           options.nz;
    if (field.phi.size() != count || field.xb.size() != count) {
        throw std::runtime_error("field size mismatch during analysis");
    }
    Dsu dsu(count);
    std::vector<double> h(count);
    AnalysisState result;
    result.step = field.step;
    result.dt = field.dt;
    result.temperature = field.temperature;
    result.target_mass = field.target_mass;
    result.phi_min = std::numeric_limits<double>::infinity();
    result.phi_max = -std::numeric_limits<double>::infinity();
    result.xb_min = std::numeric_limits<double>::infinity();
    result.xb_max = -std::numeric_limits<double>::infinity();
    double h_sum = 0.0;
    double matrix_xb_sum = 0.0;
    double matrix_xag_sum = 0.0;
    uint64_t matrix_count = 0;
    for (uint64_t index = 0; index < count; ++index) {
        const double phi = field.phi[index];
        const double xb = field.xb[index];
        result.finite = result.finite && std::isfinite(phi) && std::isfinite(xb);
        result.phi_min = std::min(result.phi_min, phi);
        result.phi_max = std::max(result.phi_max, phi);
        result.xb_min = std::min(result.xb_min, xb);
        result.xb_max = std::max(result.xb_max, xb);
        h[index] = h_of_phi(phi);
        h_sum += h[index];
        result.canonical_inventory += h[index] + (1.0 - h[index]) * xb;
        if (h[index] < 0.005) {
            matrix_xb_sum += xb;
            matrix_xag_sum += 2.0 * xb / (2.0 + xb);
            ++matrix_count;
        }
        if (h[index] > options.threshold) {
            dsu.parent[index] = static_cast<int32_t>(index);
        }
    }
    result.bounds =
        result.phi_min >= -1.0e-6 && result.phi_max <= 1.0 + 1.0e-6 &&
        result.xb_min > 0.0 && result.xb_max < 1.0;
    if (!result.finite || matrix_count == 0) {
        throw std::runtime_error("non-finite field or empty far-matrix mask");
    }

    const uint64_t yz = static_cast<uint64_t>(options.ny) * options.nz;
    auto flat = [&](int x, int y, int z) -> int32_t {
        return static_cast<int32_t>(
            (static_cast<uint64_t>(x) * options.ny + y) * options.nz + z);
    };
    for (int x = 0; x < options.nx; ++x) {
        for (int y = 0; y < options.ny; ++y) {
            for (int z = 0; z < options.nz; ++z) {
                const int32_t here = flat(x, y, z);
                if (dsu.parent[here] < 0) {
                    continue;
                }
                const std::array<int32_t, 3> neighbors{
                    flat((x + 1) % options.nx, y, z),
                    flat(x, (y + 1) % options.ny, z),
                    flat(x, y, (z + 1) % options.nz),
                };
                for (const int32_t neighbor : neighbors) {
                    if (dsu.parent[neighbor] >= 0) {
                        dsu.unite(here, neighbor);
                    }
                }
            }
        }
    }

    std::map<int32_t, int32_t> root_to_dense;
    result.labels.assign(count, -1);
    for (uint64_t index = 0; index < count; ++index) {
        if (dsu.parent[index] < 0) {
            continue;
        }
        const int32_t root = dsu.find(static_cast<int32_t>(index));
        auto [iterator, inserted] = root_to_dense.emplace(
            root, static_cast<int32_t>(root_to_dense.size()));
        result.labels[index] = iterator->second;
    }
    result.components.resize(root_to_dense.size());
    std::vector<std::array<double, 3>> sine(root_to_dense.size());
    std::vector<std::array<double, 3>> cosine(root_to_dense.size());
    for (size_t component = 0; component < result.components.size(); ++component) {
        result.components[component].dense_label = static_cast<int>(component);
        result.components[component].stable_id =
            stable_ids ? stable_ids->at(component) : static_cast<int>(component);
    }
    for (uint64_t index = 0; index < count; ++index) {
        const int32_t label = result.labels[index];
        if (label < 0) {
            continue;
        }
        const int x = static_cast<int>(index / yz);
        const uint64_t remainder = index % yz;
        const int y = static_cast<int>(remainder / options.nz);
        const int z = static_cast<int>(remainder % options.nz);
        const std::array<int, 3> coordinate{x, y, z};
        auto &component = result.components[label];
        ++component.voxels;
        component.h_volume_nm3 += h[index] * std::pow(options.dx_nm, 3);
        for (int axis = 0; axis < 3; ++axis) {
            const int length =
                axis == 0 ? options.nx : (axis == 1 ? options.ny : options.nz);
            const double angle =
                2.0 * kPi * static_cast<double>(coordinate[axis]) / length;
            sine[label][axis] += h[index] * std::sin(angle);
            cosine[label][axis] += h[index] * std::cos(angle);
        }
    }
    double radius_sum = 0.0;
    double area_sum = 0.0;
    double r6_sum = 0.0;
    for (size_t label = 0; label < result.components.size(); ++label) {
        auto &component = result.components[label];
        if (!(component.h_volume_nm3 > 0.0)) {
            throw std::runtime_error("empty component volume");
        }
        component.radius_nm = std::cbrt(
            3.0 * component.h_volume_nm3 / (4.0 * kPi));
        for (int axis = 0; axis < 3; ++axis) {
            const int length =
                axis == 0 ? options.nx : (axis == 1 ? options.ny : options.nz);
            double angle = std::atan2(sine[label][axis], cosine[label][axis]);
            if (angle < 0.0) {
                angle += 2.0 * kPi;
            }
            component.centroid_nm[axis] =
                angle * length / (2.0 * kPi) * options.dx_nm;
        }
        radius_sum += component.radius_nm;
        area_sum += 4.0 * kPi * component.radius_nm * component.radius_nm;
        r6_sum += std::pow(component.radius_nm, 6);
    }
    const double box_volume =
        static_cast<double>(count) * std::pow(options.dx_nm, 3);
    result.beta_volume_fraction = h_sum / count;
    result.mean_radius_nm = result.components.empty()
                                ? std::numeric_limits<double>::quiet_NaN()
                                : radius_sum / result.components.size();
    result.sv_nm_inv = area_sum / box_volume;
    result.m6_nm3 = r6_sum / box_volume;
    result.broad_matrix_xb = matrix_xb_sum / matrix_count;
    result.broad_matrix_xag = matrix_xag_sum / matrix_count;
    const auto distance_squared =
        periodic_squared_distance_to_beta(field.phi, options);
    double far3_xb_sum = 0.0;
    double far3_xag_sum = 0.0;
    double far4_xb_sum = 0.0;
    double far4_xag_sum = 0.0;
    uint64_t far3_count = 0;
    uint64_t far4_count = 0;
    const double gate3_squared =
        std::pow(3.0 * 4.0 / options.dx_nm, 2);
    const double gate4_squared =
        std::pow(4.0 * 4.0 / options.dx_nm, 2);
    for (uint64_t index = 0; index < count; ++index) {
        if (field.phi[index] >= 0.5) {
            continue;
        }
        const double xb = field.xb[index];
        const double xag = 2.0 * xb / (2.0 + xb);
        if (distance_squared[index] >= gate3_squared) {
            far3_xb_sum += xb;
            far3_xag_sum += xag;
            ++far3_count;
        }
        if (distance_squared[index] >= gate4_squared) {
            far4_xb_sum += xb;
            far4_xag_sum += xag;
            ++far4_count;
        }
    }
    if (far3_count == 0 || far4_count == 0) {
        throw std::runtime_error("empty registered far-field mask");
    }
    result.far3_matrix_xb = far3_xb_sum / far3_count;
    result.far3_matrix_xag = far3_xag_sum / far3_count;
    result.far4_matrix_xb = far4_xb_sum / far4_count;
    result.far4_matrix_xag = far4_xag_sum / far4_count;
    result.far3_fraction = static_cast<double>(far3_count) / count;
    result.far4_fraction = static_cast<double>(far4_count) / count;
    result.far3_far4_xag_delta =
        std::abs(result.far3_matrix_xag - result.far4_matrix_xag);
    const double target =
        field.target_mass > 0.0 ? field.target_mass : options.target_mean * count;
    result.mass_relative_error =
        std::abs(result.canonical_inventory - target) /
        std::max(std::abs(target), 1.0);
    return result;
}

std::vector<int> propagate_ids(
    const AnalysisState &previous,
    const AnalysisState &current,
    std::vector<Event> &events,
    std::map<int, std::vector<double>> &volume_history,
    std::vector<std::vector<int>> &current_lineage_members,
    bool allow_dissolution,
    bool allow_merge_groups) {
    using Pair = std::pair<int32_t, int32_t>;
    std::map<Pair, uint64_t> overlap;
    if (previous.labels.size() != current.labels.size()) {
        throw std::runtime_error("lineage label grids differ");
    }
    for (size_t index = 0; index < previous.labels.size(); ++index) {
        const int32_t before = previous.labels[index];
        const int32_t after = current.labels[index];
        if (before >= 0 && after >= 0) {
            ++overlap[{before, after}];
        }
    }
    std::vector<std::set<int>> children(previous.components.size());
    std::vector<std::set<int>> parents(current.components.size());
    for (const auto &[pair, hits] : overlap) {
        if (hits == 0) {
            continue;
        }
        children.at(pair.first).insert(pair.second);
        parents.at(pair.second).insert(pair.first);
    }
    std::vector<int> stable_ids(current.components.size(), -1);
    current_lineage_members.assign(current.components.size(), {});
    for (size_t current_label = 0; current_label < parents.size(); ++current_label) {
        if (parents[current_label].size() > 1) {
            std::vector<int> parent_stable_ids;
            std::set<int> lineage_members;
            bool overlap_qualified = true;
            double child_overlap_fraction_sum = 0.0;
            std::ostringstream detail;
            detail << std::setprecision(17);
            bool first_detail = true;
            for (const int parent_label : parents[current_label]) {
                const auto &parent = previous.components.at(parent_label);
                parent_stable_ids.push_back(parent.stable_id);
                if (parent.lineage_members.empty()) {
                    lineage_members.insert(parent.stable_id);
                } else {
                    lineage_members.insert(
                        parent.lineage_members.begin(),
                        parent.lineage_members.end());
                }
                const uint64_t hits =
                    overlap.at({parent_label, static_cast<int>(current_label)});
                const double parent_fraction =
                    static_cast<double>(hits) /
                    std::max<uint64_t>(parent.voxels, 1);
                const double child_fraction =
                    static_cast<double>(hits) /
                    std::max<uint64_t>(
                        current.components.at(current_label).voxels, 1);
                overlap_qualified =
                    overlap_qualified && parent_fraction >= 0.5;
                child_overlap_fraction_sum += child_fraction;
                if (!first_detail) {
                    detail << '+';
                }
                first_detail = false;
                detail
                    << parent.stable_id << ':' << hits << ':'
                    << parent_fraction << ':' << child_fraction;
            }
            std::sort(parent_stable_ids.begin(), parent_stable_ids.end());
            const int group_primary =
                lineage_members.empty() ? -1 : *lineage_members.begin();
            overlap_qualified =
                overlap_qualified && group_primary >= 0 &&
                child_overlap_fraction_sum >= 0.5;
            stable_ids[current_label] = group_primary;
            current_lineage_members[current_label].assign(
                lineage_members.begin(), lineage_members.end());
            if (allow_merge_groups && overlap_qualified) {
                // The group begins a new volume history.  Reusing one parent's
                // pre-merge history would make a later group dissolution test
                // compare unlike physical objects.
                volume_history[group_primary].clear();
            }
            events.push_back({
                current.step,
                "merge",
                group_primary,
                static_cast<int>(parents[current_label].size()),
                1,
                false,
                allow_merge_groups && overlap_qualified,
                parent_stable_ids,
                group_primary,
                "parent_id:overlap_voxels:parent_fraction:child_fraction=" +
                    detail.str(),
            });
            continue;
        }
        if (parents[current_label].empty()) {
            events.push_back({
                current.step,
                "new_component_without_overlap",
                -1,
                0,
                1,
                false,
                false,
                {},
                -1,
                "new beta insertion is prohibited",
            });
            continue;
        }
        const int parent_label = *parents[current_label].begin();
        stable_ids[current_label] =
            previous.components.at(parent_label).stable_id;
        current_lineage_members[current_label] =
            previous.components.at(parent_label).lineage_members;
        if (current_lineage_members[current_label].empty()) {
            current_lineage_members[current_label].push_back(
                stable_ids[current_label]);
        }
    }
    for (size_t previous_label = 0; previous_label < children.size(); ++previous_label) {
        const int stable_id = previous.components[previous_label].stable_id;
        if (children[previous_label].size() > 1) {
            events.push_back({
                current.step,
                "split",
                stable_id,
                1,
                static_cast<int>(children[previous_label].size()),
                false,
                false,
                {stable_id},
                -1,
                "parent component overlaps multiple children",
            });
        } else if (children[previous_label].empty()) {
            const auto found = volume_history.find(stable_id);
            bool qualified = false;
            if (found != volume_history.end() && found->second.size() >= 3) {
                const auto &history = found->second;
                const size_t n = history.size();
                qualified =
                    history[n - 1] < history[n - 2] &&
                    history[n - 2] < history[n - 3] &&
                    history[n - 1] <= 0.5 * history.front();
            }
            events.push_back({
                current.step,
                "dissolution",
                stable_id,
                1,
                0,
                qualified,
                false,
                {stable_id},
                -1,
                qualified ? "continuous recorded volume decay"
                          : "insufficient continuous decay evidence",
            });
            if (!allow_dissolution) {
                events.back().qualified_dissolution = false;
            }
        }
    }
    return stable_ids;
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + argument);
            }
            return argv[++i];
        };
        if (argument == "--initial-phi") {
            options.initial_phi = value();
        } else if (argument == "--initial-xb") {
            options.initial_xb = value();
        } else if (argument == "--checkpoint") {
            options.checkpoints.emplace_back(value());
        } else if (argument == "--out") {
            options.out = value();
        } else if (argument == "--grid") {
            options.nx = options.ny = options.nz = std::stoi(value());
        } else if (argument == "--dx-nm") {
            options.dx_nm = std::stod(value());
        } else if (argument == "--threshold") {
            options.threshold = std::stod(value());
        } else if (argument == "--physical-dt-s") {
            options.physical_dt_s = std::stod(value());
        } else if (argument == "--start-age-h") {
            options.start_age_h = std::stod(value());
        } else if (argument == "--target-mean") {
            options.target_mean = std::stod(value());
        } else if (argument == "--expected-initial-count") {
            options.expected_initial_count = std::stoi(value());
        } else if (argument == "--allow-dissolution") {
            options.allow_dissolution = true;
        } else if (argument == "--allow-merge-groups") {
            options.allow_merge_groups = true;
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (
        options.initial_phi.empty() || options.initial_xb.empty() ||
        options.checkpoints.empty() || options.out.empty()) {
        throw std::runtime_error(
            "--initial-phi, --initial-xb, --checkpoint, and --out are required");
    }
    return options;
}

void ensure_new_output(const fs::path &out) {
    if (fs::exists(out)) {
        throw std::runtime_error("refusing to overwrite output: " + out.string());
    }
    fs::create_directories(out);
}

std::string join_ids(const std::vector<int> &ids) {
    std::ostringstream stream;
    for (size_t index = 0; index < ids.size(); ++index) {
        if (index) {
            stream << '+';
        }
        stream << ids[index];
    }
    return stream.str();
}

void write_outputs(
    const Options &options,
    const std::vector<AnalysisState> &states,
    const std::vector<Event> &events,
    bool passed) {
    std::ofstream ensemble(options.out / "ensemble_observables.csv");
    ensemble << std::setprecision(17);
    ensemble
        << "step,elapsed_physical_time_s,experimental_age_h,particle_count,"
           "beta_volume_fraction,mean_radius_nm,Sv_nm_inv,M6_nm3,"
           "far_field_matrix_xB,far_field_matrix_xAg,"
           "far4_matrix_xB,far4_matrix_xAg,far3_fraction,far4_fraction,"
           "far3_far4_xAg_delta,broad_matrix_xB,broad_matrix_xAg,"
           "canonical_inventory_code,"
           "mass_relative_error,phi_normalized_l1_from_initial,"
           "xB_mean_absolute_from_initial,phi_min,phi_max,xB_min,xB_max,"
           "finite,bounds\n";
    for (const auto &state : states) {
        const double elapsed = state.step * options.physical_dt_s;
        ensemble
            << state.step << ',' << elapsed << ','
            << options.start_age_h + elapsed / 3600.0 << ','
            << state.components.size() << ',' << state.beta_volume_fraction << ','
            << state.mean_radius_nm << ',' << state.sv_nm_inv << ','
            << state.m6_nm3 << ',' << state.far3_matrix_xb << ','
            << state.far3_matrix_xag << ',' << state.far4_matrix_xb << ','
            << state.far4_matrix_xag << ',' << state.far3_fraction << ','
            << state.far4_fraction << ',' << state.far3_far4_xag_delta << ','
            << state.broad_matrix_xb << ',' << state.broad_matrix_xag << ','
            << state.canonical_inventory << ','
            << state.mass_relative_error << ','
            << state.phi_normalized_l1_from_initial << ','
            << state.xb_mean_absolute_from_initial << ',' << state.phi_min
            << ',' << state.phi_max << ',' << state.xb_min << ','
            << state.xb_max << ',' << (state.finite ? 1 : 0) << ','
            << (state.bounds ? 1 : 0) << '\n';
    }

    std::ofstream lineage(options.out / "particle_lineage.csv");
    lineage << std::setprecision(17);
    lineage
        << "step,elapsed_physical_time_s,experimental_age_h,stable_particle_id,"
           "lineage_member_ids,lineage_group_size,"
           "component_label,threshold_voxel_count,h_volume_nm3,"
           "equivalent_radius_nm,centroid_x_nm,centroid_y_nm,centroid_z_nm\n";
    for (const auto &state : states) {
        const double elapsed = state.step * options.physical_dt_s;
        for (const auto &component : state.components) {
            lineage
                << state.step << ',' << elapsed << ','
                << options.start_age_h + elapsed / 3600.0 << ','
                << component.stable_id << ','
                << join_ids(component.lineage_members) << ','
                << component.lineage_members.size() << ','
                << component.dense_label << ','
                << component.voxels << ',' << component.h_volume_nm3 << ','
                << component.radius_nm << ',' << component.centroid_nm[0] << ','
                << component.centroid_nm[1] << ',' << component.centroid_nm[2]
                << '\n';
        }
    }

    std::ofstream event_file(options.out / "particle_events.csv");
    event_file
        << "step,event_type,stable_particle_id,parent_count,child_count,"
           "qualified_dissolution,qualified_merge,parent_stable_ids,"
           "child_stable_id,detail\n";
    for (const auto &event : events) {
        event_file
            << event.step << ',' << event.type << ',' << event.stable_id << ','
            << event.parent_count << ',' << event.child_count << ','
            << (event.qualified_dissolution ? 1 : 0) << ','
            << (event.qualified_merge ? 1 : 0) << ','
            << join_ids(event.parent_stable_ids) << ','
            << event.child_stable_id << ',' << event.detail
            << '\n';
    }

    const double max_mass = std::accumulate(
        states.begin(), states.end(), 0.0,
        [](double current, const AnalysisState &state) {
            return std::max(current, state.mass_relative_error);
        });
    int dissolution_count = 0;
    int merge_count = 0;
    int qualified_merge_count = 0;
    int unqualified_merge_count = 0;
    int split_count = 0;
    int new_count = 0;
    int unqualified_dissolution_count = 0;
    for (const auto &event : events) {
        if (event.type == "dissolution") {
            ++dissolution_count;
            if (!event.qualified_dissolution) {
                ++unqualified_dissolution_count;
            }
        } else if (event.type == "merge") {
            ++merge_count;
            if (event.qualified_merge) {
                ++qualified_merge_count;
            } else {
                ++unqualified_merge_count;
            }
        } else if (event.type == "split") {
            ++split_count;
        } else if (event.type == "new_component_without_overlap") {
            ++new_count;
        }
    }
    std::ofstream summary(options.out / "lineage_summary.txt");
    summary << std::setprecision(17)
            << "status="
            << (passed ? "PASS_PERIODIC_OVERLAP_PARTICLE_LINEAGE_V1"
                       : "BLOCKED_PERIODIC_OVERLAP_PARTICLE_LINEAGE_V1")
            << '\n'
            << "initial_particle_count=" << states.front().components.size() << '\n'
            << "final_particle_count=" << states.back().components.size() << '\n'
            << "snapshot_count=" << states.size() << '\n'
            << "dissolution_count=" << dissolution_count << '\n'
            << "unqualified_dissolution_count="
            << unqualified_dissolution_count << '\n'
            << "merge_count=" << merge_count << '\n'
            << "qualified_merge_count=" << qualified_merge_count << '\n'
            << "unqualified_merge_count=" << unqualified_merge_count << '\n'
            << "split_count=" << split_count << '\n'
            << "new_component_count=" << new_count << '\n'
            << "max_mass_relative_error=" << max_mass << '\n';
    std::ofstream status(options.out / "status.txt");
    status
        << (passed ? "PASS_PERIODIC_OVERLAP_PARTICLE_LINEAGE_V1"
                   : "BLOCKED_PERIODIC_OVERLAP_PARTICLE_LINEAGE_V1")
        << '\n';
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        ensure_new_output(options.out);
        const uint64_t count =
            static_cast<uint64_t>(options.nx) * options.ny * options.nz;
        FieldState initial;
        initial.step = 0;
        initial.dt = 0.02;
        initial.temperature = 380.0;
        initial.target_mass = options.target_mean * count;
        initial.phi = read_raw(options.initial_phi, count);
        initial.xb = read_raw(options.initial_xb, count);

        std::vector<AnalysisState> states;
        states.push_back(analyze_fields(initial, options));
        if (
            static_cast<int>(states.front().components.size()) !=
            options.expected_initial_count) {
            throw std::runtime_error("initial particle count mismatch");
        }
        for (auto &component : states.front().components) {
            component.lineage_members = {component.stable_id};
        }
        std::map<int, std::vector<double>> volume_history;
        for (const auto &component : states.front().components) {
            volume_history[component.stable_id].push_back(
                component.h_volume_nm3);
        }
        std::vector<Event> events;
        uint64_t prior_step = 0;
        double initial_h_sum = 0.0;
        for (const double phi : initial.phi) {
            initial_h_sum += h_of_phi(phi);
        }
        for (const auto &path : options.checkpoints) {
            const auto field = read_checkpoint(path, options);
            if (field.step <= prior_step) {
                throw std::runtime_error("checkpoints are not strictly ordered");
            }
            auto current = analyze_fields(field, options);
            double phi_l1 = 0.0;
            double xb_l1 = 0.0;
            for (size_t index = 0; index < field.phi.size(); ++index) {
                phi_l1 += std::abs(field.phi[index] - initial.phi[index]);
                xb_l1 += std::abs(field.xb[index] - initial.xb[index]);
            }
            current.phi_normalized_l1_from_initial =
                phi_l1 / std::max(initial_h_sum, 1.0);
            current.xb_mean_absolute_from_initial =
                xb_l1 / field.xb.size();
            std::vector<std::vector<int>> current_lineage_members;
            const auto stable_ids = propagate_ids(
                states.back(),
                current,
                events,
                volume_history,
                current_lineage_members,
                options.allow_dissolution,
                options.allow_merge_groups);
            for (size_t label = 0; label < current.components.size(); ++label) {
                current.components[label].stable_id = stable_ids[label];
                current.components[label].lineage_members =
                    current_lineage_members[label];
                if (stable_ids[label] >= 0) {
                    volume_history[stable_ids[label]].push_back(
                        current.components[label].h_volume_nm3);
                }
            }
            states.push_back(std::move(current));
            prior_step = field.step;
        }
        bool passed = true;
        for (const auto &state : states) {
            passed = passed && state.finite && state.bounds &&
                     state.mass_relative_error <= 1.0e-10;
        }
        for (const auto &event : events) {
            if (
                (event.type == "merge" &&
                 (!options.allow_merge_groups || !event.qualified_merge)) ||
                event.type == "split" ||
                event.type == "new_component_without_overlap" ||
                (event.type == "dissolution" &&
                 (!options.allow_dissolution ||
                  !event.qualified_dissolution))) {
                passed = false;
            }
        }
        if (!options.allow_dissolution) {
            passed = passed &&
                     states.back().components.size() ==
                         states.front().components.size();
        }
        write_outputs(options, states, events, passed);
        std::cout
            << (passed ? "PASS_PERIODIC_OVERLAP_PARTICLE_LINEAGE_V1"
                       : "BLOCKED_PERIODIC_OVERLAP_PARTICLE_LINEAGE_V1")
            << '\n';
        return passed ? 0 : 2;
    } catch (const std::exception &error) {
        std::cerr << "[fatal] " << error.what() << '\n';
        return 2;
    }
}
