#!/usr/bin/env python3
"""Generate the exact source-level P0 device-memory ledger.

The estimator never allocates a field and never advances a time step. cuFFT
work areas are supplied separately by tests/pf_ctot_fft_memory_estimator.cu.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Allocation:
    name: str
    dtype: str
    count_expr: str
    bytes_per_element: int
    category: str
    first_use: str
    last_use: str
    role: str
    elastic_off: bool
    elastic_on: bool
    diagnostics_off: bool
    alias_opportunity: str


ALLOCATIONS = [
    Allocation("d_phi_r", "float64", "N", 8, "persistent_accepted_state", "initialization", "shutdown", "accepted", True, True, True, "authoritative phase; no alias"),
    Allocation("d_Y_r", "float64", "N", 8, "persistent_accepted_state", "initialization", "shutdown", "derived context", True, True, True, "required between operators"),
    Allocation("d_xB_r", "float64", "N", 8, "persistent_accepted_state", "initialization", "shutdown", "derived context", True, True, True, "required between operators"),
    Allocation("d_phi_rhs_r", "float64", "N", 8, "phase_PDAS_scratch", "phase driving", "shutdown", "scratch", True, True, True, "aliases d_lapY_r"),
    Allocation("d_phi_n_saved", "float64", "N", 8, "persistent_accepted_state", "attempt save", "shutdown", "accepted", True, True, True, "rollback anchor"),
    Allocation("d_Y_n_saved", "float64", "N", 8, "persistent_accepted_state", "attempt save", "shutdown", "accepted", True, True, True, "rollback anchor"),
    Allocation("d_ctot_accepted_r", "float64", "N", 8, "persistent_accepted_state", "Ctot initialization", "shutdown", "accepted", True, True, True, "authoritative conserved state"),
    Allocation("d_ctot_work_r", "float64", "N", 8, "persistent_trial_state", "attempt start", "commit/reject", "trial", True, True, True, "cannot alias accepted during rollback"),
    Allocation("d_ctot_saved_r", "float64", "N", 8, "persistent_accepted_state", "attempt save", "commit/reject", "accepted", True, True, True, "rollback anchor"),
    Allocation("d_ctot_trial_r", "float64", "N", 8, "persistent_trial_state", "transport/phase trial", "commit/reject", "trial", True, True, True, "shared transport/PDAS trial"),
    Allocation("d_ctot_residual_r", "float64", "N", 8, "phase_PDAS_scratch", "transport residual", "commit/reject", "scratch", True, True, True, "shared transport/phase residual"),
    Allocation("d_ctot_q_alpha_r", "float64", "N", 8, "persistent_trial_state", "Ctot reconstruction", "commit", "derived context", True, True, True, "reused by PDAS residual"),
    Allocation("d_ctot_active_mask_r", "float64", "N", 8, "persistent_trial_state", "Ctot reconstruction", "commit", "mask/active code", True, True, True, "shared matrix mask/PDAS code"),
    Allocation("d_ctot_outer_C_prev_r", "float64", "N", 8, "persistent_trial_state", "outer iteration", "attempt end", "outer history", True, True, True, "needed by convergence audit"),
    Allocation("d_ctot_outer_phi_prev_r", "float64", "N", 8, "persistent_trial_state", "outer iteration", "attempt end", "outer history", True, True, True, "needed by convergence audit"),
    Allocation("d_ctot_outer_C_prev2_r", "float64", "N", 8, "persistent_trial_state", "outer initialization", "attempt end", "two-cycle history", True, True, True, "could become optional if two-cycle gate redesigned"),
    Allocation("d_ctot_outer_phi_prev2_r", "float64", "N", 8, "persistent_trial_state", "outer initialization", "attempt end", "two-cycle history", True, True, True, "could become optional if two-cycle gate redesigned"),
    Allocation("d_mu_x_r", "float64", "N", 8, "transport_scratch", "thermodynamics", "shutdown", "scratch", True, True, True, "aliases d_Y_rhs_r"),
    Allocation("d_divJ_r", "float64", "N", 8, "transport_scratch", "flux divergence", "shutdown", "scratch", True, True, True, "aliases d_xB_prev_r"),
    Allocation("d_ctot_fv_face_x", "float64", "N", 8, "transport_scratch", "face flux", "transport residual", "scratch", True, True, True, "required for conservative divergence"),
    Allocation("d_ctot_fv_face_y", "float64", "N", 8, "transport_scratch", "face flux", "transport residual", "scratch", True, True, True, "required for conservative divergence"),
    Allocation("d_ctot_fv_face_z", "float64", "N", 8, "transport_scratch", "face flux", "transport residual", "scratch", True, True, True, "required for conservative divergence"),
    Allocation("d_scratch_r_double", "float64", "2*N", 8, "phase_PDAS_scratch", "energy/phase", "shutdown", "scratch", True, True, True, "two-slot arena already shared"),
    Allocation("d_phi_k", "complex128", "K", 16, "FFT_state_scratch", "phase FFT", "shutdown", "scratch", True, True, True, "shared by all phase transforms"),
    Allocation("d_phi_rhs_k", "complex128", "K", 16, "FFT_state_scratch", "phase driving FFT", "shutdown", "scratch", True, True, True, "aliases d_mu_x_k/d_Y_rhs_k"),
    Allocation("d_Y_k", "complex128", "K", 16, "FFT_state_scratch", "old phase FFT", "shutdown", "scratch", True, True, True, "temporary spectrum"),
    Allocation("d_divJ_k", "complex128", "K", 16, "FFT_state_scratch", "spectral scratch", "shutdown", "scratch", True, True, True, "kept for legacy paths"),
    Allocation("d_scratch_k_double", "complex128", "K", 16, "FFT_state_scratch", "spectral scratch", "shutdown", "scratch", True, True, True, "single shared slot"),
    Allocation("KS.d_k2", "float64", "K", 8, "FFT_state_scratch", "k-space build", "shutdown", "coefficient", True, True, True, "read-only"),
    Allocation("KS.d_k4", "float64", "K", 8, "FFT_state_scratch", "k-space build", "shutdown", "coefficient", True, True, True, "legacy-compatible coefficient"),
    Allocation("ctot_stats_packets", "float64", "53", 8, "diagnostic_status", "initialization", "shutdown", "mandatory gates", True, True, True, "small device packets"),
    Allocation("bbox_boundary_packets", "mixed", "8", 8, "diagnostic_status", "initialization", "shutdown", "diagnostic", True, True, True, "negligible"),
    Allocation("elastic_strain_stress", "float32", "12*N", 4, "mechanical_state_scratch", "elastic initialization", "shutdown", "state/scratch", False, True, True, "six strain + six stress"),
    Allocation("elastic_base_spectra", "complex64", "9*K", 8, "mechanical_state_scratch", "elastic initialization", "shutdown", "state/scratch", False, True, True, "3 displacement + 6 strain spectra"),
    Allocation("mechanical_transaction_real", "float32", "30*N", 4, "mechanical_state_scratch", "candidate elastic setup", "shutdown", "accepted/trial/history", False, True, True, "6 each: accepted strain/stress, outer stress, accepted/trial eps0"),
    Allocation("mechanical_transaction_spectra", "complex64", "9*K", 8, "mechanical_state_scratch", "candidate elastic setup", "shutdown", "accepted/trial/history", False, True, True, "3 each: accepted/outer/inner displacement"),
    Allocation("reduction_workspace_peak", "float64", "2*ceil(N/256)+ceil(ceil(N/256)/256)", 8, "transient_reduction", "first reduction", "each reduction return", "scratch", True, True, True, "P1 persistent workspace candidate"),
]


def count(expr: str, n: int, k: int) -> int:
    blocks = (n + 255) // 256
    values = {
        "N": n, "2*N": 2 * n, "K": k, "53": 53, "8": 8,
        "12*N": 12 * n, "9*K": 9 * k, "30*N": 30 * n,
        "2*ceil(N/256)+ceil(ceil(N/256)/256)": 2 * blocks + (blocks + 255) // 256,
    }
    return values[expr]


def write_ledger(path: Path, nx: int, ny: int, nz: int) -> None:
    n = nx * ny * nz
    k = nx * ny * (nz // 2 + 1)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "type", "element_count", "bytes", "allocation_phase",
                         "first_use", "last_use", "role", "required_elastic_off",
                         "required_elastic_on", "required_diagnostics_off",
                         "alias_reuse_opportunity"])
        for item in ALLOCATIONS:
            elements = count(item.count_expr, n, k)
            writer.writerow([item.name, item.dtype, elements,
                             elements * item.bytes_per_element, item.category,
                             item.first_use, item.last_use, item.role,
                             int(item.elastic_off), int(item.elastic_on),
                             int(item.diagnostics_off), item.alias_opportunity])


def estimate_rows():
    for nx, ny, nz in [(320, 2, 320), (64, 64, 64), (128, 128, 128),
                       (256, 256, 256), (320, 320, 320)]:
        n = nx * ny * nz
        k = nx * ny * (nz // 2 + 1)
        for elastic in (False, True):
            for detailed in (False, True):
                total = 0
                categories: dict[str, int] = {}
                for item in ALLOCATIONS:
                    if elastic and not item.elastic_on:
                        continue
                    if not elastic and not item.elastic_off:
                        continue
                    if not detailed and not item.diagnostics_off:
                        continue
                    size = count(item.count_expr, n, k) * item.bytes_per_element
                    total += size
                    categories[item.category] = categories.get(item.category, 0) + size
                yield [f"{nx}x{ny}x{nz}", n, k, int(elastic), int(detailed),
                       categories.get("persistent_accepted_state", 0),
                       categories.get("persistent_trial_state", 0),
                       categories.get("transport_scratch", 0),
                       categories.get("phase_PDAS_scratch", 0),
                       categories.get("FFT_state_scratch", 0),
                       categories.get("mechanical_state_scratch", 0),
                       categories.get("diagnostic_status", 0),
                       categories.get("transient_reduction", 0), total,
                       total / 2**30]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_ledger(args.output_dir / "perf_p0_memory_ledger.csv", 320, 2, 320)
    path = args.output_dir / "perf_p0_3d_memory_estimate.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["grid", "N", "K", "elastic_enabled", "detailed_diagnostics",
                         "persistent_accepted_bytes", "persistent_trial_bytes",
                         "transport_scratch_bytes", "phase_pdas_scratch_bytes",
                         "fft_state_scratch_bytes", "mechanical_bytes",
                         "diagnostic_bytes", "transient_reduction_peak_bytes",
                         "source_allocated_bytes_excluding_cufft_context", "GiB"])
        writer.writerows(estimate_rows())


if __name__ == "__main__":
    main()
