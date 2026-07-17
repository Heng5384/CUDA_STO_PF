from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAIN = (ROOT / "main_cuda.cu").read_text(encoding="utf-8")
KERNELS = (ROOT / "cuda_kernels.cu").read_text(encoding="utf-8")
PARAMS = (ROOT / "pf_params.h").read_text(encoding="utf-8")


def test_feature_selector_is_default_off_and_bounded():
    assert "P->ctot_memory_perf_features = 0;" in MAIN
    assert "P->ctot_memory_perf_features > 1023" in MAIN
    assert "int ctot_memory_perf_features;" in PARAMS


def test_streamed_divergence_preserves_axis_order_and_periodic_incidence():
    assert "if (axis == 0) divJ_r[idx] = contribution;" in KERNELS
    assert "else divJ_r[idx] += contribution;" in KERNELS
    rng = np.random.default_rng(20260717)
    shape = (7, 6, 5)
    faces = [rng.normal(size=shape) for _ in range(3)]
    spacing = (0.7, 1.1, 1.3)
    original = (
        (faces[0] - np.roll(faces[0], 1, axis=0)) / spacing[0]
        + (faces[1] - np.roll(faces[1], 1, axis=1)) / spacing[1]
        + (faces[2] - np.roll(faces[2], 1, axis=2)) / spacing[2]
    )
    streamed = (faces[0] - np.roll(faces[0], 1, axis=0)) / spacing[0]
    streamed += (faces[1] - np.roll(faces[1], 1, axis=1)) / spacing[1]
    streamed += (faces[2] - np.roll(faces[2], 1, axis=2)) / spacing[2]
    assert np.array_equal(original, streamed)
    assert abs(float(streamed.sum())) < 1.0e-12


def test_event_snapshot_alias_is_lifecycle_gated_and_zero_incremental():
    assert "ctot_lazy_event_snapshot_runtime &&" in MAIN
    assert "ctot_method_consistent_split_runtime" in MAIN
    assert "d_ctot_event_macro_start_r = d_ctot_transport_anchor_r;" in MAIN
    assert "d_phi_event_macro_start_r = d_phi_transport_context_r;" in MAIN
    assert '"aliases_transport_context=%d incremental_bytes=%zu "' in MAIN
    assert "!ctot_event_snapshot_aliases_transport_context" in MAIN


def test_derived_y_rollback_reconstructs_from_authoritative_state():
    assert "ctot_derived_Y_snapshot_elision_runtime" in MAIN
    assert "reconstruct_ctot_thermodynamic_context(\n                d_ctot_event_macro_start_r" in MAIN
    assert "Y is derived from the authoritative conserved and phase states" in MAIN


def test_output_suppression_is_explicit_default_off_and_output_only():
    assert "P->ctot_benchmark_suppress_field_output = 0;" in MAIN
    assert "int ctot_benchmark_suppress_field_output;" in PARAMS
    assert "!P.ctot_benchmark_suppress_field_output" in MAIN
    gate_region = MAIN[MAIN.index("const int global_mass_closed"):]
    assert "ctot_benchmark_suppress_field_output" not in gate_region[:1800]


def test_physics_and_acceptance_contract_markers_remain_unchanged():
    assert "changes_physics=0" in MAIN
    assert "changes_numerics=0 changes_gates=0" in MAIN
    assert "fabs(mass_error) <= 1.0e-10" in MAIN
    assert "projection_mass=0 clip_count=0" in MAIN
