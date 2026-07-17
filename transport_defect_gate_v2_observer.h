#ifndef TRANSPORT_DEFECT_GATE_V2_OBSERVER_H
#define TRANSPORT_DEFECT_GATE_V2_OBSERVER_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

// Default-off host observer. It consumes only already accepted states and never
// returns data to the solver. Event substeps remain buffered until the parent
// macro transaction commits.
class TransportDefectGateV2Observer {
  public:
    struct Metrics {
        double delta_time = 0.0;
        double sum_abs_D = 0.0;
        double sum_A = 0.0;
        double D_peak = 0.0;
        double A_scale = 0.0;
        double eta_global = 0.0;
        double eta_interface = 0.0;
        double eta_cell_max = 0.0;
        double b_signed = 0.0;
        double b_interface = 0.0;
        double r_peak = 0.0;
        double signed_D = 0.0;
        double interface_sum_abs_D = 0.0;
        double interface_sum_A = 0.0;
        double interface_signed_D = 0.0;
        double material_A_threshold = 0.0;
        double machine_roundoff_accumulation_estimate = 0.0;
        long long interface_cells = 0;
        long long material_cells = 0;
        long long accepted_rows = 0;
    };

    bool open(const char *output_dir, int total_cells, double window_time_code,
              double interface_h_eps = 1.0e-4) {
        if (!output_dir || !output_dir[0] || total_cells <= 0 ||
            !(window_time_code > 0.0) || !(interface_h_eps > 0.0) ||
            !(interface_h_eps < 0.5)) {
            fail("invalid V2 observer configuration");
            return false;
        }
        output_dir_ = output_dir;
        n_ = total_cells;
        window_time_ = window_time_code;
        interface_eps_ = interface_h_eps;
        nonzero_floor_ = std::max(
            1.0e-30,
            100.0 * std::numeric_limits<double>::epsilon() * (double)n_);
        roundoff_floor_ = nonzero_floor_;
        window_D_.assign((size_t)n_, 0.0);
        window_A_.assign((size_t)n_, 0.0);
        window_B_.assign((size_t)n_, 0.0);
        window_interface_.assign((size_t)n_, 0);
        full_D_.assign((size_t)n_, 0.0);
        full_A_.assign((size_t)n_, 0.0);
        full_B_.assign((size_t)n_, 0.0);
        full_interface_.assign((size_t)n_, 0);
        event_D_.assign((size_t)n_, 0.0);
        event_A_.assign((size_t)n_, 0.0);
        event_B_.assign((size_t)n_, 0.0);
        event_interface_.assign((size_t)n_, 0);

        char path[4096];
        std::snprintf(path, sizeof(path),
                      "%s/ctot_transport_defect_v2_window_metrics.csv",
                      output_dir_.c_str());
        metrics_fp_ = std::fopen(path, "w");
        if (!metrics_fp_) {
            fail("cannot open V2 window metrics CSV");
            return false;
        }
        std::fprintf(
            metrics_fp_,
            "record_type,window_index,time_start_code,time_end_code,"
            "delta_time_code,sum_abs_D,sum_A,D_peak,A_scale,eta_global,"
            "eta_interface,eta_cell_max,b_signed,b_interface,r_peak,"
            "signed_D,interface_sum_abs_D,interface_sum_A,"
            "interface_signed_D,interface_cells,material_cells,"
            "material_A_threshold,accepted_rows_in_record,"
            "machine_roundoff_accumulation_estimate,nonzero_floor,roundoff_floor,"
            "interface_contract\n");
        std::fflush(metrics_fp_);
        enabled_ = true;
        return write_contract();
    }

    bool enabled() const { return enabled_; }
    bool good() const { return good_; }
    const std::string &error() const { return error_; }

    void reset_event_transaction() {
        std::fill(event_D_.begin(), event_D_.end(), 0.0);
        std::fill(event_A_.begin(), event_A_.end(), 0.0);
        std::fill(event_B_.begin(), event_B_.end(), 0.0);
        std::fill(event_interface_.begin(), event_interface_.end(), 0);
        event_time_ = 0.0;
        event_rows_ = 0;
    }

    bool accumulate_accepted(double dt, const std::vector<double> &residual,
                             const std::vector<double> &C_old,
                             const std::vector<double> &C_new,
                             const std::vector<double> &phi_new,
                             bool event_transaction) {
        if (!enabled_) return true;
        if (!(dt > 0.0) || residual.size() != (size_t)n_ ||
            C_old.size() != (size_t)n_ || C_new.size() != (size_t)n_ ||
            phi_new.size() != (size_t)n_) {
            fail("invalid accepted-step input to V2 observer");
            return false;
        }
        std::vector<double> &D = event_transaction ? event_D_ : window_D_;
        std::vector<double> &A = event_transaction ? event_A_ : window_A_;
        std::vector<double> &B = event_transaction ? event_B_ : window_B_;
        std::vector<uint8_t> &interface_seen =
            event_transaction ? event_interface_ : window_interface_;
        for (int i = 0; i < n_; ++i) {
            const double dD = dt * residual[(size_t)i];
            const double dA =
                std::fabs(C_new[(size_t)i] - C_old[(size_t)i]);
            D[(size_t)i] += dD;
            A[(size_t)i] += dA;
            B[(size_t)i] += std::fabs(dD);
            const double hv = h(phi_new[(size_t)i]);
            if (hv >= interface_eps_ && hv <= 1.0 - interface_eps_) {
                interface_seen[(size_t)i] = 1;
            }
            if (!event_transaction) {
                full_D_[(size_t)i] += dD;
                full_A_[(size_t)i] += dA;
                full_B_[(size_t)i] += std::fabs(dD);
                if (hv >= interface_eps_ && hv <= 1.0 - interface_eps_) {
                    full_interface_[(size_t)i] = 1;
                }
            }
        }
        if (event_transaction) {
            event_time_ += dt;
            ++event_rows_;
            return true;
        }
        window_time_accum_ += dt;
        full_time_ += dt;
        ++window_rows_;
        ++accepted_rows_;
        return flush_complete_window_if_needed();
    }

    bool commit_event_transaction() {
        if (!enabled_ || event_rows_ == 0) return true;
        for (int i = 0; i < n_; ++i) {
            window_D_[(size_t)i] += event_D_[(size_t)i];
            window_A_[(size_t)i] += event_A_[(size_t)i];
            window_B_[(size_t)i] += event_B_[(size_t)i];
            window_interface_[(size_t)i] =
                (uint8_t)(window_interface_[(size_t)i] ||
                          event_interface_[(size_t)i]);
        }
        window_time_accum_ += event_time_;
        full_time_ += event_time_;
        window_rows_ += event_rows_;
        accepted_rows_ += event_rows_;
        accumulate_into_full(event_D_, event_A_, event_B_, event_interface_);
        reset_event_transaction();
        return flush_complete_window_if_needed();
    }

    bool finalize() {
        if (!enabled_) return true;
        if (event_rows_ != 0) {
            fail("uncommitted V2 event transaction at finalize");
            return false;
        }
        const double tol = time_tolerance(full_time_);
        if (window_time_accum_ > tol) {
            if (!write_record("PARTIAL_WINDOW", window_index_ + 1,
                              full_time_ - window_time_accum_, full_time_,
                              window_D_, window_A_, window_B_,
                              window_interface_, window_rows_)) {
                return false;
            }
        }
        if (!write_record("FULL_TRAJECTORY", 0, 0.0, full_time_, full_D_,
                          full_A_, full_B_, full_interface_, accepted_rows_)) {
            return false;
        }
        if (!write_raw("ctot_transport_defect_v2_full_D.raw", full_D_) ||
            !write_raw("ctot_transport_defect_v2_full_A.raw", full_A_) ||
            !write_raw("ctot_transport_defect_v2_full_B.raw", full_B_) ||
            !write_raw("ctot_transport_defect_v2_full_interface_seen.raw",
                       full_interface_)) {
            return false;
        }
        char path[4096];
        std::snprintf(path, sizeof(path),
                      "%s/ctot_transport_defect_v2_meta.json",
                      output_dir_.c_str());
        FILE *fp = std::fopen(path, "w");
        if (!fp) {
            fail("cannot write V2 metadata");
            return false;
        }
        std::fprintf(
            fp,
            "{\n"
            "  \"schema\": \"PHYSICALLY_NORMALIZED_TRANSPORT_DEFECT_OBSERVER_V2\",\n"
            "  \"accepted_rows\": %lld,\n"
            "  \"complete_windows\": %d,\n"
            "  \"observed_time_code\": %.17e,\n"
            "  \"window_time_code\": %.17e,\n"
            "  \"interface_h_lower\": %.17e,\n"
            "  \"interface_h_upper\": %.17e,\n"
            "  \"interface_contract\": \"UNION_OF_ACCEPTED_FINAL_STATE_INTERFACE_BAND_PER_WINDOW\",\n"
            "  \"material_cell_contract\": \"A_i_ge_1e-6_times_window_A_scale\",\n"
            "  \"nonzero_floor\": %.17e,\n"
            "  \"roundoff_floor\": %.17e,\n"
            "  \"solver_state_modified\": false\n"
            "}\n",
            accepted_rows_, window_index_, full_time_, window_time_,
            interface_eps_, 1.0 - interface_eps_, nonzero_floor_,
            roundoff_floor_);
        std::fclose(fp);
        std::fflush(metrics_fp_);
        std::fclose(metrics_fp_);
        metrics_fp_ = nullptr;
        enabled_ = false;
        return good_;
    }

    ~TransportDefectGateV2Observer() {
        if (metrics_fp_) std::fclose(metrics_fp_);
    }

  private:
    static double h(double phi) {
        return phi * phi * phi *
               (6.0 * phi * phi - 15.0 * phi + 10.0);
    }

    double time_tolerance(double scale) const {
        return std::max(1.0e-13,
                        32.0 * std::numeric_limits<double>::epsilon() *
                            std::max(1.0, std::fabs(scale)) *
                            std::max(1LL, accepted_rows_));
    }

    void fail(const char *message) {
        good_ = false;
        error_ = message ? message : "unknown V2 observer failure";
    }

    void accumulate_into_full(const std::vector<double> &D,
                              const std::vector<double> &A,
                              const std::vector<double> &B,
                              const std::vector<uint8_t> &interface_seen) {
        for (int i = 0; i < n_; ++i) {
            full_D_[(size_t)i] += D[(size_t)i];
            full_A_[(size_t)i] += A[(size_t)i];
            full_B_[(size_t)i] += B[(size_t)i];
            full_interface_[(size_t)i] =
                (uint8_t)(full_interface_[(size_t)i] ||
                          interface_seen[(size_t)i]);
        }
    }

    bool flush_complete_window_if_needed() {
        const double tol = time_tolerance(window_time_);
        if (window_time_accum_ < window_time_ - tol) return true;
        if (window_time_accum_ > window_time_ + tol) {
            fail("V2 observer window boundary overshot");
            return false;
        }
        ++window_index_;
        const double end = full_time_;
        const double start = end - window_time_accum_;
        if (!write_record("WINDOW", window_index_, start, end, window_D_,
                          window_A_, window_B_, window_interface_,
                          window_rows_)) {
            return false;
        }
        char name[256];
        std::snprintf(name, sizeof(name),
                      "ctot_transport_defect_v2_window_%04d_D.raw",
                      window_index_);
        if (!write_raw(name, window_D_)) return false;
        std::snprintf(name, sizeof(name),
                      "ctot_transport_defect_v2_window_%04d_A.raw",
                      window_index_);
        if (!write_raw(name, window_A_)) return false;
        std::snprintf(name, sizeof(name),
                      "ctot_transport_defect_v2_window_%04d_B.raw",
                      window_index_);
        if (!write_raw(name, window_B_)) return false;
        std::snprintf(name, sizeof(name),
                      "ctot_transport_defect_v2_window_%04d_interface_seen.raw",
                      window_index_);
        if (!write_raw(name, window_interface_)) return false;
        std::fill(window_D_.begin(), window_D_.end(), 0.0);
        std::fill(window_A_.begin(), window_A_.end(), 0.0);
        std::fill(window_B_.begin(), window_B_.end(), 0.0);
        std::fill(window_interface_.begin(), window_interface_.end(), 0);
        window_time_accum_ = 0.0;
        window_rows_ = 0;
        return true;
    }

    Metrics metrics(double delta_time, const std::vector<double> &D,
                    const std::vector<double> &A,
                    const std::vector<double> &B,
                    const std::vector<uint8_t> &interface_seen,
                    long long rows) const {
        Metrics m;
        m.delta_time = delta_time;
        m.accepted_rows = rows;
        double B_peak = 0.0;
        for (int i = 0; i < n_; ++i) {
            const double abs_D = std::fabs(D[(size_t)i]);
            m.sum_abs_D += abs_D;
            m.sum_A += A[(size_t)i];
            m.D_peak = std::max(m.D_peak, abs_D);
            m.A_scale = std::max(m.A_scale, A[(size_t)i]);
            B_peak = std::max(B_peak, B[(size_t)i]);
            m.signed_D += D[(size_t)i];
            if (interface_seen[(size_t)i]) {
                m.interface_sum_abs_D += abs_D;
                m.interface_sum_A += A[(size_t)i];
                m.interface_signed_D += D[(size_t)i];
                ++m.interface_cells;
            }
        }
        m.material_A_threshold = 1.0e-6 * m.A_scale;
        for (int i = 0; i < n_; ++i) {
            if (A[(size_t)i] >= m.material_A_threshold &&
                m.A_scale > 0.0) {
                m.eta_cell_max = std::max(
                    m.eta_cell_max,
                    std::fabs(D[(size_t)i]) /
                        std::max(A[(size_t)i], m.material_A_threshold));
                ++m.material_cells;
            }
        }
        m.eta_global = m.sum_abs_D / std::max(m.sum_A, nonzero_floor_);
        m.eta_interface = m.interface_sum_abs_D /
                          std::max(m.interface_sum_A, nonzero_floor_);
        m.b_signed = std::fabs(m.signed_D) /
                     std::max(m.sum_abs_D, roundoff_floor_);
        m.b_interface = std::fabs(m.interface_signed_D) /
                        std::max(m.interface_sum_abs_D, roundoff_floor_);
        m.r_peak = m.D_peak / std::max(delta_time, nonzero_floor_);
        const double eps = std::numeric_limits<double>::epsilon();
        const double n_eps = (double)std::max(0LL, rows) * eps;
        const double gamma_n = n_eps < 1.0 ? n_eps / (1.0 - n_eps)
                                            : std::numeric_limits<double>::infinity();
        m.machine_roundoff_accumulation_estimate = gamma_n * B_peak;
        return m;
    }

    bool write_record(const char *record_type, int window_index,
                      double start, double end,
                      const std::vector<double> &D,
                      const std::vector<double> &A,
                      const std::vector<double> &B,
                      const std::vector<uint8_t> &interface_seen,
                      long long rows) {
        if (!metrics_fp_) {
            fail("V2 metrics stream is closed");
            return false;
        }
        const Metrics m = metrics(end - start, D, A, B, interface_seen, rows);
        std::fprintf(
            metrics_fp_,
            "%s,%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,"
            "%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,"
            "%.17e,%.17e,%lld,%lld,%.17e,%lld,%.17e,%.17e,%.17e,%s\n",
            record_type, window_index, start, end, m.delta_time,
            m.sum_abs_D, m.sum_A, m.D_peak, m.A_scale, m.eta_global,
            m.eta_interface, m.eta_cell_max, m.b_signed, m.b_interface,
            m.r_peak, m.signed_D, m.interface_sum_abs_D,
            m.interface_sum_A, m.interface_signed_D, m.interface_cells,
            m.material_cells, m.material_A_threshold, m.accepted_rows,
            m.machine_roundoff_accumulation_estimate,
            nonzero_floor_, roundoff_floor_,
            "UNION_OF_ACCEPTED_FINAL_STATE_INTERFACE_BAND_PER_WINDOW");
        std::fflush(metrics_fp_);
        return true;
    }

    template <typename T>
    bool write_raw(const char *name, const std::vector<T> &values) {
        char path[4096];
        std::snprintf(path, sizeof(path), "%s/%s", output_dir_.c_str(), name);
        std::ofstream stream(path,
                             std::ios::out | std::ios::binary | std::ios::trunc);
        if (!stream) {
            fail("cannot open V2 raw output");
            return false;
        }
        stream.write(reinterpret_cast<const char *>(values.data()),
                     (std::streamsize)(values.size() * sizeof(T)));
        if (!stream.good()) {
            fail("cannot write V2 raw output");
            return false;
        }
        return true;
    }

    bool write_contract() {
        char path[4096];
        std::snprintf(path, sizeof(path),
                      "%s/ctot_transport_defect_v2_observer_contract.json",
                      output_dir_.c_str());
        FILE *fp = std::fopen(path, "w");
        if (!fp) {
            fail("cannot write V2 observer contract");
            return false;
        }
        std::fprintf(
            fp,
            "{\n"
            "  \"schema\": \"PHYSICALLY_NORMALIZED_TRANSPORT_DEFECT_OBSERVER_V2\",\n"
            "  \"D_i\": \"sum_accepted_dt_times_method_cold_R_C_i\",\n"
            "  \"A_i\": \"sum_accepted_abs_Ctot_np1_minus_Ctot_n_i\",\n"
            "  \"B_i\": \"sum_accepted_abs_dt_times_method_cold_R_C_i\",\n"
            "  \"interface_h_lower\": %.17e,\n"
            "  \"interface_h_upper\": %.17e,\n"
            "  \"interface_contract\": \"UNION_OF_ACCEPTED_FINAL_STATE_INTERFACE_BAND_PER_WINDOW\",\n"
            "  \"material_cell_relative_cutoff\": 1.00000000000000000e-06,\n"
            "  \"window_time_code\": %.17e,\n"
            "  \"nonzero_floor\": %.17e,\n"
            "  \"roundoff_floor\": %.17e,\n"
            "  \"roundoff_accumulation_estimate\": \"gamma_n_times_max_i_B_i\",\n"
            "  \"transaction_contract\": \"ONLY_COMMITTED_MACRO_TRANSACTIONS\",\n"
            "  \"default_off\": true,\n"
            "  \"solver_state_modified\": false\n"
            "}\n",
            interface_eps_, 1.0 - interface_eps_, window_time_,
            nonzero_floor_, roundoff_floor_);
        std::fclose(fp);
        return true;
    }

    bool enabled_ = false;
    bool good_ = true;
    std::string error_;
    std::string output_dir_;
    FILE *metrics_fp_ = nullptr;
    int n_ = 0;
    int window_index_ = 0;
    long long accepted_rows_ = 0;
    long long window_rows_ = 0;
    long long event_rows_ = 0;
    double window_time_ = 0.0;
    double window_time_accum_ = 0.0;
    double full_time_ = 0.0;
    double event_time_ = 0.0;
    double interface_eps_ = 1.0e-4;
    double nonzero_floor_ = 1.0e-30;
    double roundoff_floor_ = 1.0e-30;
    std::vector<double> window_D_, window_A_, window_B_;
    std::vector<double> full_D_, full_A_, full_B_;
    std::vector<double> event_D_, event_A_, event_B_;
    std::vector<uint8_t> window_interface_, full_interface_, event_interface_;
};

#endif
