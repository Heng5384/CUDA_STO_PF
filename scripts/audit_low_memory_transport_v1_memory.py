#!/usr/bin/env python3
"""Generate the source-exact active-manifold device-buffer ledger.

This script accounts for every persistent cudaMalloc on the frozen PF-only
production path. CUDA context and cuFFT work areas are deliberately separate
until they are measured on the target GPU.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "low_memory_transport_v1"
TARGET_DEVICE_TOTAL_BYTES = 16303 * 2**20
ADMISSION_LIMIT_BYTES = 0.85 * TARGET_DEVICE_TOTAL_BYTES


@dataclass(frozen=True)
class Buffer:
    name: str
    dtype: str
    bytes_per_element: int
    count: str
    allocation: str
    first_use: str
    last_use: str
    lifetime: str
    aliasable: str
    rollback: str
    checkpoint: str
    fft: str
    mechanics: str
    gp: str
    condition: str = "always"


B = [
    Buffer("d_bbox_mins", "int32", 4, "3", "main_cuda.cu:30131", "initial diagnostics", "shutdown", "persistent diagnostic packet", "no", "no", "no", "no", "no", "no"),
    Buffer("d_bbox_maxs", "int32", 4, "3", "main_cuda.cu:30132", "initial diagnostics", "shutdown", "persistent diagnostic packet", "no", "no", "no", "no", "no", "no"),
    Buffer("d_boundary_sum", "float64", 8, "1", "main_cuda.cu:30133", "initial diagnostics", "shutdown", "persistent diagnostic packet", "no", "no", "no", "no", "no", "no"),
    Buffer("d_boundary_count", "uint64", 8, "1", "main_cuda.cu:30134", "initial diagnostics", "shutdown", "persistent diagnostic packet", "no", "no", "no", "no", "no", "no"),
    Buffer("d_phi_r", "float64", 8, "N", "main_cuda.cu:30365", "initialization", "shutdown", "persistent authoritative", "no", "yes", "yes", "yes", "yes", "no"),
    Buffer("d_Y_r", "float64", 8, "N", "main_cuda.cu:30366", "context initialization", "shutdown", "persistent derived context", "no", "yes", "yes", "yes", "no", "no"),
    Buffer("d_xB_r", "float64", 8, "N", "main_cuda.cu:30367", "context initialization", "shutdown", "persistent derived context", "no", "reconstructed", "yes", "no", "yes", "no"),
    Buffer("d_phi_rhs_r", "float64", 8, "N", "main_cuda.cu:30368", "phase solve", "shutdown", "persistent shared scratch", "d_lapY_r", "no", "no", "yes", "yes", "no"),
    Buffer("d_phi_n_saved", "float64", 8, "N", "main_cuda.cu:30369", "attempt/history save", "shutdown", "persistent accepted history", "no", "yes", "yes", "yes", "yes", "no"),
    Buffer("d_Y_n_saved", "float64", 8, "N", "main_cuda.cu:30370", "attempt save", "shutdown", "persistent rollback context", "no", "yes", "yes", "no", "no", "no"),
    Buffer("d_ctot_accepted_r", "float64", 8, "N", "main_cuda.cu:30377", "Ctot initialization", "shutdown", "persistent authoritative", "pointer-rotation candidate", "yes", "yes", "no", "yes", "no"),
    Buffer("d_ctot_work_r", "float64", 8, "N", "main_cuda.cu:30378", "attempt start", "attempt commit/reject", "attempt trial", "pointer-rotation candidate", "yes", "no", "no", "yes", "no"),
    Buffer("d_ctot_saved_r", "float64", 8, "N", "main_cuda.cu:30379", "attempt save", "attempt commit/reject", "attempt rollback anchor", "pointer-rotation candidate", "yes", "no", "no", "yes", "no"),
    Buffer("d_ctot_trial_r", "float64", 8, "N", "main_cuda.cu:30380", "transport/phase trial", "attempt end", "attempt scratch", "conditional scratch reuse", "no", "no", "no", "yes", "no"),
    Buffer("d_ctot_history_nm1_r", "float64", 8, "N", "main_cuda.cu:30382", "BDF2 context", "shutdown", "persistent history", "no", "yes", "yes", "no", "no", "no", "BDF2"),
    Buffer("d_phi_history_nm1_r", "float64", 8, "N", "main_cuda.cu:30383", "BDF2 context", "shutdown", "persistent history", "no", "yes", "yes", "no", "no", "no", "BDF2"),
    Buffer("d_ctot_transport_anchor_r", "float64", 8, "N", "main_cuda.cu:30384", "BDF2 predictor", "attempt end", "attempt anchor", "no", "yes", "no", "no", "no", "no", "BDF2"),
    Buffer("d_phi_transport_context_r", "float64", 8, "N", "main_cuda.cu:30385", "BDF2 predictor", "attempt end", "attempt context", "no", "yes", "no", "no", "no", "no", "BDF2"),
    Buffer("d_ctot_event_macro_start_r", "float64", 8, "N", "main_cuda.cu:30387", "event transaction", "macro commit/reject", "event rollback", "lazy-allocation candidate", "yes", "no", "no", "no", "no", "event_subcycling"),
    Buffer("d_phi_event_macro_start_r", "float64", 8, "N", "main_cuda.cu:30388", "event transaction", "macro commit/reject", "event rollback", "lazy-allocation candidate", "yes", "no", "no", "no", "no", "event_subcycling"),
    Buffer("d_Y_event_macro_start_r", "float64", 8, "N", "main_cuda.cu:30389", "event transaction", "macro commit/reject", "event rollback", "lazy-allocation candidate", "yes", "no", "no", "no", "no", "event_subcycling"),
    Buffer("d_ctot_residual_r", "float64", 8, "N", "main_cuda.cu:30392", "transport residual", "attempt end", "attempt scratch", "shared with phase residual", "no", "no", "no", "yes", "no"),
    Buffer("d_ctot_line_base_residual_r", "float64", 8, "N", "main_cuda.cu:30394", "feasible direction", "transport solve end", "transport direction", "not while trial residual is live", "no", "no", "yes", "no", "no", "adaptive_coordinate"),
    Buffer("d_ctot_q_alpha_r", "float64", 8, "N", "main_cuda.cu:30396", "context reconstruction", "attempt end", "derived storage", "lifetime-reuse candidate", "no", "no", "no", "no", "no"),
    Buffer("d_ctot_active_mask_r", "float64", 8, "N", "main_cuda.cu:30397", "context reconstruction", "attempt end", "active code", "uint8 candidate", "no", "no", "no", "no", "no"),
    Buffer("d_ctot_outer_C_prev_r", "float64", 8, "N", "main_cuda.cu:30398", "outer iteration", "attempt end", "outer history", "no", "yes", "no", "no", "no", "no"),
    Buffer("d_ctot_outer_phi_prev_r", "float64", 8, "N", "main_cuda.cu:30399", "outer iteration", "attempt end", "outer history", "no", "yes", "no", "no", "no", "no"),
    Buffer("d_ctot_outer_C_prev2_r", "float64", 8, "N", "main_cuda.cu:30400", "two-cycle audit", "attempt end", "outer history", "no", "yes", "no", "no", "no", "no"),
    Buffer("d_ctot_outer_phi_prev2_r", "float64", 8, "N", "main_cuda.cu:30401", "two-cycle audit", "attempt end", "outer history", "no", "yes", "no", "no", "no", "no"),
    Buffer("d_ctot_audit_stats", "float64", 8, "12", "main_cuda.cu:30424", "Ctot initialization", "shutdown", "persistent diagnostic packet", "no", "no", "no", "no", "no", "no"),
    Buffer("d_ctot_phase_stats", "float64", 8, "11", "main_cuda.cu:30426", "phase audit", "shutdown", "persistent diagnostic packet", "no", "no", "no", "no", "no", "no"),
    Buffer("d_ctot_phase_pdas_stats", "float64", 8, "5", "main_cuda.cu:30428", "phase PDAS", "shutdown", "persistent diagnostic packet", "no", "no", "no", "no", "no", "no"),
    Buffer("d_ctot_phase_trial_decision", "PhasePdasTrialDecisionPacket", 64, "1", "main_cuda.cu:30436", "phase line search", "shutdown", "persistent diagnostic packet", "no", "no", "no", "no", "no", "no"),
    Buffer("d_ctot_phase_pcg_scalars", "float64", 8, "8", "main_cuda.cu:30438", "phase PCG", "shutdown", "persistent solver packet", "no", "no", "no", "no", "no", "no"),
    Buffer("d_ctot_phase_pcg_packet", "float64", 8, "2", "main_cuda.cu:30440", "phase PCG", "shutdown", "persistent solver packet", "no", "no", "no", "no", "no", "no"),
    Buffer("d_ctot_transport_stats", "float64", 8, "25", "main_cuda.cu:30443", "transport residual", "shutdown", "persistent diagnostic packet", "no", "no", "no", "no", "no", "no"),
    Buffer("d_ctot_fv_face_x", "float64", 8, "N", "main_cuda.cu:30448", "face flux", "transport audit", "transport scratch", "face-streaming candidate", "no", "no", "no", "no", "no"),
    Buffer("d_ctot_fv_face_y", "float64", 8, "N", "main_cuda.cu:30449", "face flux", "transport audit", "transport scratch", "face-streaming candidate", "no", "no", "no", "no", "no"),
    Buffer("d_ctot_fv_face_z", "float64", 8, "N", "main_cuda.cu:30450", "face flux", "transport audit", "transport scratch", "face-streaming candidate", "no", "no", "no", "no", "no"),
    Buffer("d_ctot_outer_face_prev_x", "float64", 8, "N", "main_cuda.cu:30452", "outer flux audit", "attempt end", "outer flux history", "recompute tradeoff only", "yes", "no", "no", "no", "no", "capacity_outer"),
    Buffer("d_ctot_outer_face_prev_y", "float64", 8, "N", "main_cuda.cu:30453", "outer flux audit", "attempt end", "outer flux history", "recompute tradeoff only", "yes", "no", "no", "no", "no", "capacity_outer"),
    Buffer("d_ctot_outer_face_prev_z", "float64", 8, "N", "main_cuda.cu:30454", "outer flux audit", "attempt end", "outer flux history", "recompute tradeoff only", "yes", "no", "no", "no", "no", "capacity_outer"),
    Buffer("d_mu_x_r", "float64", 8, "N", "main_cuda.cu:30804", "thermodynamics", "shutdown", "shared scratch", "d_Y_rhs_r", "no", "no", "yes", "no", "no"),
    Buffer("d_divJ_r", "float64", 8, "N", "main_cuda.cu:30808", "flux divergence", "shutdown", "shared scratch", "d_xB_prev_r", "no", "no", "no", "no", "no"),
    Buffer("d_scratch_r_double", "float64", 8, "2*N", "main_cuda.cu:30825", "first operator", "shutdown", "scratch arena", "two disjoint slots", "no", "no", "yes", "yes", "no"),
    Buffer("d_phi_k", "complex128", 16, "K", "main_cuda.cu:30815", "phase FFT", "shutdown", "FFT state/scratch", "no", "no", "no", "yes", "no", "no"),
    Buffer("d_phi_rhs_k", "complex128", 16, "K", "main_cuda.cu:30816", "phase FFT", "shutdown", "FFT scratch", "d_mu_x_k;d_Y_rhs_k", "no", "no", "yes", "no", "no"),
    Buffer("d_Y_k", "complex128", 16, "K", "main_cuda.cu:30817", "phase context FFT", "shutdown", "FFT scratch", "lifetime-reuse candidate", "no", "no", "yes", "no", "no"),
    Buffer("d_divJ_k", "complex128", 16, "K", "main_cuda.cu:30823", "legacy/spectral scratch", "shutdown", "FFT scratch", "legacy-gated reuse candidate", "no", "no", "yes", "no", "no"),
    Buffer("d_scratch_k_double", "complex128", 16, "K", "main_cuda.cu:30824", "first FFT operator", "shutdown", "FFT scratch arena", "single shared slot", "no", "no", "yes", "yes", "no"),
    Buffer("KS.d_k2", "float64", 8, "K", "cuda_common.cu:94", "k-space build", "shutdown", "FFT coefficient", "no", "no", "no", "yes", "no", "no"),
    Buffer("KS.d_k4", "float64", 8, "K", "cuda_common.cu:95", "k-space build", "shutdown", "FFT coefficient", "legacy-only removal candidate", "no", "no", "yes", "no", "no"),
    Buffer("phase_trial_block_summaries", "uint64x7", 56, "ceil(N/256)", "main_cuda.cu:30433", "phase line search", "shutdown", "reduction scratch", "no", "no", "no", "no", "no", "no"),
    Buffer("reduction_workspace_a", "float64", 8, "ceil(N/256)", "cuda_kernels.cu:6423", "first reduction", "shutdown", "reduction scratch", "global shared workspace", "no", "no", "no", "no", "no"),
    Buffer("reduction_workspace_b", "float64", 8, "ceil(ceil(N/256)/256)", "cuda_kernels.cu:6429", "first reduction", "shutdown", "reduction scratch", "global shared workspace", "no", "no", "no", "no", "no"),
    Buffer("reduction_workspace_zero", "float64", 8, "1", "cuda_kernels.cu:6434", "first reduction", "shutdown", "reduction scalar", "global shared workspace", "no", "no", "no", "no", "no"),
    Buffer("elastic_strain_stress", "float32", 4, "12*N", "main_cuda.cu:30920-30932", "mechanics", "shutdown", "mechanical state/scratch", "no", "yes", "no", "yes", "yes", "no", "elastic"),
    Buffer("mechanical_force", "float32", 4, "N", "main_cuda.cu:30949", "mechanical audit", "shutdown", "mechanical scratch", "no", "no", "no", "no", "yes", "no", "elastic"),
    Buffer("mechanical_transaction_real", "float32", 4, "30*N", "main_cuda.cu:30956-30960", "mechanical transaction", "shutdown", "mechanical accepted/trial/history", "no", "yes", "no", "no", "yes", "no", "elastic"),
    Buffer("elastic_base_spectra", "complex64", 8, "9*K", "main_cuda.cu:30938-30946", "mechanics FFT", "shutdown", "mechanical FFT scratch", "no", "no", "no", "yes", "yes", "no", "elastic"),
    Buffer("mechanical_transaction_spectra", "complex64", 8, "9*K", "main_cuda.cu:30951-30953", "mechanical transaction", "shutdown", "mechanical FFT history", "no", "yes", "no", "yes", "yes", "no", "elastic"),
]


def elements(expr: str, n: int, k: int) -> int:
    blocks = (n + 255) // 256
    known = {
        "N": n,
        "2*N": 2 * n,
        "K": k,
        "9*K": 9 * k,
        "12*N": 12 * n,
        "30*N": 30 * n,
        "ceil(N/256)": blocks,
        "ceil(ceil(N/256)/256)": (blocks + 255) // 256,
    }
    if expr in known:
        return known[expr]
    return int(expr)


def enabled(item: Buffer, elastic: bool) -> bool:
    return item.condition != "elastic" or elastic


def write_lifetime() -> None:
    n = 400**3
    k = 400 * 400 * (400 // 2 + 1)
    fields = list(Buffer.__dataclass_fields__) + ["bytes_at_400cube"]
    with (OUT / "device_buffer_lifetime.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in B:
            row = item.__dict__.copy()
            row["bytes_at_400cube"] = elements(item.count, n, k) * item.bytes_per_element
            writer.writerow(row)


def write_scaling() -> None:
    fields = [
        "grid", "N", "K", "configuration", "source_exact_cudaMalloc_bytes",
        "source_exact_cudaMalloc_GiB", "cufft_workspace_bytes", "cuda_context_bytes",
        "observed_peak_loss_bytes", "device_total_bytes", "admission_limit_bytes",
        "source_allocation_headroom_bytes", "admission_85pct_status",
        "evidence_status",
    ]
    with (OUT / "memory_scaling.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for side in (128, 192, 256, 320, 400):
            n = side**3
            k = side * side * (side // 2 + 1)
            for configuration, elastic in (("PF_ONLY", False), ("PF_ELASTIC", True)):
                total = sum(
                    elements(item.count, n, k) * item.bytes_per_element
                    for item in B if enabled(item, elastic)
                )
                writer.writerow({
                    "grid": f"{side}x{side}x{side}", "N": n, "K": k,
                    "configuration": configuration,
                    "source_exact_cudaMalloc_bytes": total,
                    "source_exact_cudaMalloc_GiB": total / 2**30,
                    "cufft_workspace_bytes": "PENDING_TARGET_GPU_QUERY",
                    "cuda_context_bytes": "PENDING_TARGET_GPU_MEASUREMENT",
                    "observed_peak_loss_bytes": "PENDING_TARGET_GPU_MEASUREMENT",
                    "device_total_bytes": TARGET_DEVICE_TOTAL_BYTES,
                    "admission_limit_bytes": int(ADMISSION_LIMIT_BYTES),
                    "source_allocation_headroom_bytes": int(ADMISSION_LIMIT_BYTES - total),
                    "admission_85pct_status": (
                        "FAIL_SOURCE_ALLOCATIONS_EXCEED_85PCT"
                        if total > ADMISSION_LIMIT_BYTES else
                        "PENDING_CUFFT_CONTEXT_AND_OBSERVED_PEAK"
                    ),
                    "evidence_status": "SOURCE_EXACT_EXCLUDING_CUFFT_CONTEXT",
                })
            # Static GP reservoirs are host-side site vectors in the current
            # code; no GP full-grid device array is added to the PF+elastic sum.
            total = sum(
                elements(item.count, n, k) * item.bytes_per_element
                for item in B if enabled(item, True)
            )
            writer.writerow({
                "grid": f"{side}x{side}x{side}", "N": n, "K": k,
                "configuration": "PF_ELASTIC_STATIC_GP_RESERVOIR",
                "source_exact_cudaMalloc_bytes": total,
                "source_exact_cudaMalloc_GiB": total / 2**30,
                "cufft_workspace_bytes": "PENDING_TARGET_GPU_QUERY",
                "cuda_context_bytes": "PENDING_TARGET_GPU_MEASUREMENT",
                "observed_peak_loss_bytes": "PENDING_TARGET_GPU_MEASUREMENT",
                "device_total_bytes": TARGET_DEVICE_TOTAL_BYTES,
                "admission_limit_bytes": int(ADMISSION_LIMIT_BYTES),
                "source_allocation_headroom_bytes": int(ADMISSION_LIMIT_BYTES - total),
                "admission_85pct_status": (
                    "FAIL_SOURCE_ALLOCATIONS_EXCEED_85PCT_AND_S3_GATED"
                    if total > ADMISSION_LIMIT_BYTES else
                    "PENDING_CUFFT_CONTEXT_OBSERVED_PEAK_AND_S3_GATE"
                ),
                "evidence_status": "SOURCE_EXACT_DEVICE_FIELDS_GP_SITE_VECTOR_HOST_ONLY",
            })


def write_graph() -> None:
    text = """# Device-memory dependency graph

`Ctot accepted` feeds BDF2 history/anchor, then transport work/trial/residual. A rejected attempt requires an intact accepted state and histories. `phi` follows the same accepted/history/context chain. This is why pointer rotation must rotate authoritative pointers atomically and cannot merely delete rollback storage.

Transport reconstruction produces `q_alpha`, `xB_alpha`, `Y`, active code and chemical potential. The three shared-face arrays feed one divergence field and are simultaneously needed by the existing outer face-change audit. Face streaming is therefore conditional on preserving shared-face antisymmetry and replacing that audit without losing its prior-outer reference.

The two-slot real scratch arena stores the preconditioned direction/snapshot and scalar-reduction inputs. The complex scratch and shared double FFT plans serve transport and phase serially. No low-memory globalization feature adds a full-grid field.

The current runtime `print_memory_ledger` omits several candidate allocations and is not an admission oracle. `device_buffer_lifetime.csv` is the source allocation ledger; `memory_scaling.csv` remains incomplete until target-GPU cuFFT workspaces and observed peak loss are measured.

Static GP reservoirs are host-side `GpAssistedSite` vectors in this code path. The Ctot candidate currently rejects GP/S3 at validation, so the combined PF+elastic+GP row is a union ledger, not an executable integrated production claim.
"""
    (OUT / "memory_dependency_graph.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_lifetime()
    write_scaling()
    write_graph()


if __name__ == "__main__":
    main()
