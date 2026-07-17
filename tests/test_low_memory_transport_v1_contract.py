from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_default_off_and_versioned_selector():
    source = (ROOT / "main_cuda.cu").read_text()
    params = (ROOT / "pf_params.h").read_text()
    assert '"LEGACY_CURRENT"' in source
    assert '"LOW_MEMORY_GLOBALIZATION_V1"' in source
    assert "ctot_transport_efficiency_features = 0" in source
    assert "char ctot_transport_efficiency_mode[48]" in params
    assert "int ctot_transport_efficiency_features" in params
    assert "~CTOT_EFFICIENCY_IMPLEMENTED_FEATURES" in source


def test_residual_and_acceptance_gates_unchanged_by_feature_helpers():
    source = (ROOT / "main_cuda.cu").read_text()
    header = (ROOT / "low_memory_transport_v1_utils.h").read_text()
    assert "res_inf <= residual_target" in source
    assert "fabs(mass_error) <= 1.0e-10" in source
    assert "fabs(sum_divJ) <= 1.0e-10" in source
    assert "trial_linf <= production_gate" in header
    assert "0.995 * lambda_feasible" in header


def test_no_sparse_or_large_krylov_storage_added():
    header = (ROOT / "low_memory_transport_v1_utils.h").read_text().lower()
    forbidden = ("csr", "coo", "ellpack", "gmres", "fgmres", "krylov")
    assert all(token not in header for token in forbidden)


def test_checkpoint_provenance_is_written_and_read():
    source = (ROOT / "main_cuda.cu").read_text()
    assert source.count("transport_efficiency_mode") >= 8
    assert source.count("transport_efficiency_features") >= 8
    assert "CTOT_TRANSPORT_EFFICIENCY_RESTART_MIGRATION" in source
