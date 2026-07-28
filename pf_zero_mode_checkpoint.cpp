#include "pf_zero_mode_checkpoint.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>

namespace pf_zero_mode {
namespace {

constexpr std::uint32_t kVersion = 2U;
constexpr char kMagic[8] = {'P', 'F', 'Z', 'M', 'C', 'H', 'K', '2'};
constexpr std::size_t kSelectorBytes = 64U;

struct DiskHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t header_bytes;
    std::uint64_t element_count;
    std::uint64_t accepted_step;
    std::int32_t nx;
    std::int32_t ny;
    std::int32_t nz;
    std::int32_t reserved;
    double dt_code;
    double temperature_K;
    double target_mass_code;
    double last_lambda;
    double last_residual_code;
    double last_derivative_code;
    std::uint64_t last_iterations;
    std::uint64_t accepted_zero_mode_steps;
    std::uint64_t parameter_fingerprint;
    char zero_mode[kSelectorBytes];
    char backend[kSelectorBytes];
    char composition_mode[kSelectorBytes];
    char y_update_mode[kSelectorBytes];
    char explicit_context[kSelectorBytes];
    char reaction_discretization[kSelectorBytes];
    std::uint64_t payload_checksum;
};

bool set_error(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

bool finite_vector(const std::vector<double>& values) {
    for (double value : values) {
        if (!std::isfinite(value)) return false;
    }
    return true;
}

bool copy_selector(char* destination, const std::string& value,
                   const char* label, std::string* error) {
    if (value.empty() || value.size() >= kSelectorBytes) {
        return set_error(error, std::string(label) + " is empty or too long");
    }
    std::memset(destination, 0, kSelectorBytes);
    std::memcpy(destination, value.data(), value.size());
    return true;
}

bool write_exact(FILE* fp, const void* data, std::size_t size) {
    return size == 0U || std::fwrite(data, 1U, size, fp) == size;
}

bool read_exact(FILE* fp, void* data, std::size_t size) {
    return size == 0U || std::fread(data, 1U, size, fp) == size;
}

bool selector_equal(const char* disk, const std::string& expected) {
    return std::strncmp(disk, expected.c_str(), kSelectorBytes) == 0 &&
           disk[expected.size()] == '\0';
}

}  // namespace

std::uint64_t fnv1a64(const void* data, std::size_t size, std::uint64_t seed) {
    const unsigned char* bytes = static_cast<const unsigned char*>(data);
    std::uint64_t hash = seed;
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= static_cast<std::uint64_t>(bytes[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool validate_checkpoint(const Checkpoint& checkpoint, std::string* error) {
    if (checkpoint.nx <= 0 || checkpoint.ny <= 0 || checkpoint.nz <= 0) {
        return set_error(error, "checkpoint grid dimensions are invalid");
    }
    const std::uint64_t count =
        static_cast<std::uint64_t>(checkpoint.nx) *
        static_cast<std::uint64_t>(checkpoint.ny) *
        static_cast<std::uint64_t>(checkpoint.nz);
    if (count > static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())) {
        return set_error(error, "checkpoint element count overflows size_t");
    }
    if (checkpoint.phi.size() != count || checkpoint.Y.size() != count ||
        checkpoint.xB.size() != count ||
        checkpoint.dY_dt_prev.size() != count) {
        return set_error(error, "checkpoint field sizes do not match the grid");
    }
    if (!(checkpoint.dt_code > 0.0) || !std::isfinite(checkpoint.dt_code) ||
        !(checkpoint.temperature_K > 0.0) ||
        !std::isfinite(checkpoint.temperature_K)) {
        return set_error(error, "checkpoint time/temperature is invalid");
    }
    if (checkpoint.provenance.zero_mode != kModeV1 ||
        checkpoint.provenance.backend.empty() ||
        checkpoint.provenance.composition_mode.empty() ||
        checkpoint.provenance.y_update_mode.empty() ||
        checkpoint.provenance.explicit_context.empty() ||
        checkpoint.provenance.reaction_discretization.empty() ||
        checkpoint.provenance.parameter_fingerprint == 0U) {
        return set_error(error, "checkpoint zero-mode provenance is incomplete");
    }
    if (!std::isfinite(checkpoint.zero_mode.target_mass_code) ||
        !std::isfinite(checkpoint.zero_mode.last_lambda) ||
        !std::isfinite(checkpoint.zero_mode.last_residual_code) ||
        !std::isfinite(checkpoint.zero_mode.last_derivative_code) ||
        !finite_vector(checkpoint.phi) || !finite_vector(checkpoint.Y) ||
        !finite_vector(checkpoint.xB) ||
        !finite_vector(checkpoint.dY_dt_prev)) {
        return set_error(error, "checkpoint contains a non-finite value");
    }
    return true;
}

bool write_checkpoint(const std::string& path, const Checkpoint& checkpoint,
                      std::string* error) {
    if (!validate_checkpoint(checkpoint, error)) return false;

    DiskHeader header = {};
    std::memcpy(header.magic, kMagic, sizeof(kMagic));
    header.version = kVersion;
    header.header_bytes = sizeof(DiskHeader);
    header.element_count = checkpoint.phi.size();
    header.accepted_step = checkpoint.accepted_step;
    header.nx = checkpoint.nx;
    header.ny = checkpoint.ny;
    header.nz = checkpoint.nz;
    header.dt_code = checkpoint.dt_code;
    header.temperature_K = checkpoint.temperature_K;
    header.target_mass_code = checkpoint.zero_mode.target_mass_code;
    header.last_lambda = checkpoint.zero_mode.last_lambda;
    header.last_residual_code = checkpoint.zero_mode.last_residual_code;
    header.last_derivative_code = checkpoint.zero_mode.last_derivative_code;
    header.last_iterations = checkpoint.zero_mode.last_iterations;
    header.accepted_zero_mode_steps =
        checkpoint.zero_mode.accepted_zero_mode_steps;
    header.parameter_fingerprint =
        checkpoint.provenance.parameter_fingerprint;
    if (!copy_selector(header.zero_mode, checkpoint.provenance.zero_mode,
                       "zero_mode", error) ||
        !copy_selector(header.backend, checkpoint.provenance.backend,
                       "backend", error) ||
        !copy_selector(header.composition_mode,
                       checkpoint.provenance.composition_mode,
                       "composition_mode", error) ||
        !copy_selector(header.y_update_mode,
                       checkpoint.provenance.y_update_mode,
                       "y_update_mode", error) ||
        !copy_selector(header.explicit_context,
                       checkpoint.provenance.explicit_context,
                       "explicit_context", error) ||
        !copy_selector(header.reaction_discretization,
                       checkpoint.provenance.reaction_discretization,
                       "reaction_discretization", error)) {
        return false;
    }

    const std::string temporary = path + ".tmp";
    FILE* fp = std::fopen(temporary.c_str(), "wb+");
    if (!fp) {
        return set_error(error, "cannot open checkpoint temporary file: " +
                                std::string(std::strerror(errno)));
    }
    std::uint64_t checksum = fnv1a64(&header, sizeof(header));
    bool ok = write_exact(fp, &header, sizeof(header));
    const std::vector<double>* fields[] = {
        &checkpoint.phi, &checkpoint.Y, &checkpoint.xB,
        &checkpoint.dY_dt_prev};
    for (const std::vector<double>* field : fields) {
        const std::size_t bytes = field->size() * sizeof(double);
        if (ok) ok = write_exact(fp, field->data(), bytes);
        checksum = fnv1a64(field->data(), bytes, checksum);
    }
    if (ok) {
        header.payload_checksum = checksum;
        ok = std::fseek(fp, 0L, SEEK_SET) == 0 &&
             write_exact(fp, &header, sizeof(header)) &&
             std::fflush(fp) == 0;
    }
    if (std::fclose(fp) != 0) ok = false;
    if (!ok) {
        std::remove(temporary.c_str());
        return set_error(error, "checkpoint serialization failed");
    }
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        const std::string detail = std::strerror(errno);
        std::remove(temporary.c_str());
        return set_error(error, "checkpoint atomic rename failed: " + detail);
    }
    return true;
}

bool read_checkpoint(const std::string& path,
                     const Provenance& expected_provenance,
                     Checkpoint* checkpoint,
                     std::string* error) {
    if (!checkpoint) return set_error(error, "checkpoint output is null");
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        return set_error(error, "cannot open checkpoint: " +
                                std::string(std::strerror(errno)));
    }
    DiskHeader header = {};
    bool ok = read_exact(fp, &header, sizeof(header));
    if (!ok || std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0 ||
        header.version != kVersion ||
        header.header_bytes != sizeof(DiskHeader)) {
        std::fclose(fp);
        return set_error(error, "checkpoint header/version mismatch");
    }
    if (!selector_equal(header.zero_mode, expected_provenance.zero_mode) ||
        !selector_equal(header.backend, expected_provenance.backend) ||
        !selector_equal(header.composition_mode,
                        expected_provenance.composition_mode) ||
        !selector_equal(header.y_update_mode,
                        expected_provenance.y_update_mode) ||
        !selector_equal(header.explicit_context,
                        expected_provenance.explicit_context) ||
        !selector_equal(header.reaction_discretization,
                        expected_provenance.reaction_discretization) ||
        header.parameter_fingerprint !=
            expected_provenance.parameter_fingerprint) {
        std::fclose(fp);
        return set_error(error, "checkpoint zero-mode provenance mismatch");
    }
    if (header.element_count == 0U ||
        header.element_count >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max() / sizeof(double))) {
        std::fclose(fp);
        return set_error(error, "checkpoint element count is invalid");
    }

    Checkpoint loaded;
    loaded.accepted_step = header.accepted_step;
    loaded.nx = header.nx;
    loaded.ny = header.ny;
    loaded.nz = header.nz;
    loaded.dt_code = header.dt_code;
    loaded.temperature_K = header.temperature_K;
    loaded.provenance = expected_provenance;
    loaded.zero_mode.target_mass_code = header.target_mass_code;
    loaded.zero_mode.last_lambda = header.last_lambda;
    loaded.zero_mode.last_residual_code = header.last_residual_code;
    loaded.zero_mode.last_derivative_code = header.last_derivative_code;
    loaded.zero_mode.last_iterations = header.last_iterations;
    loaded.zero_mode.accepted_zero_mode_steps =
        header.accepted_zero_mode_steps;
    loaded.phi.resize(header.element_count);
    loaded.Y.resize(header.element_count);
    loaded.xB.resize(header.element_count);
    loaded.dY_dt_prev.resize(header.element_count);

    DiskHeader checksum_header = header;
    const std::uint64_t expected_checksum = checksum_header.payload_checksum;
    checksum_header.payload_checksum = 0U;
    std::uint64_t checksum = fnv1a64(&checksum_header,
                                    sizeof(checksum_header));
    std::vector<double>* fields[] = {
        &loaded.phi, &loaded.Y, &loaded.xB, &loaded.dY_dt_prev};
    for (std::vector<double>* field : fields) {
        const std::size_t bytes = field->size() * sizeof(double);
        if (ok) ok = read_exact(fp, field->data(), bytes);
        if (ok) checksum = fnv1a64(field->data(), bytes, checksum);
    }
    unsigned char trailing = 0U;
    if (ok && std::fread(&trailing, 1U, 1U, fp) != 0U) ok = false;
    std::fclose(fp);
    if (!ok || checksum != expected_checksum) {
        return set_error(error, "checkpoint payload checksum mismatch");
    }
    if (!validate_checkpoint(loaded, error)) return false;
    *checkpoint = std::move(loaded);
    return true;
}

}  // namespace pf_zero_mode
