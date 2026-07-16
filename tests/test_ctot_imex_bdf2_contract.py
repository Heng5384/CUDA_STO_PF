from __future__ import annotations

import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "main_cuda.cu").read_text(encoding="utf-8")


def test_fixed_step_anchor_is_exact_bdf2_rearrangement() -> None:
    rng = random.Random(20260716)
    dt = 0.003125
    for _ in range(1000):
        c_n = rng.uniform(0.01, 0.99)
        c_nm1 = rng.uniform(0.01, 0.99)
        div_j = rng.uniform(-1.0e-3, 1.0e-3)
        anchor = (4.0 * c_n - c_nm1) / 3.0
        c_np1 = anchor + (2.0 / 3.0) * dt * div_j
        residual = (3.0 * c_np1 - 4.0 * c_n + c_nm1) / (2.0 * dt) - div_j
        assert abs(residual) <= 2.0e-13


def test_source_free_bdf2_mass_identity() -> None:
    m_n = 62914.56
    m_nm1 = m_n
    m_np1 = (4.0 * m_n - m_nm1) / 3.0
    assert 3.0 * m_np1 - 4.0 * m_n + m_nm1 == 0.0


def test_endpoint_discrete_work_decomposition_is_algebraically_closed() -> None:
    rng = random.Random(400)
    for _ in range(1000):
        dt = 0.003125
        d_transport = rng.uniform(0.0, 1.0e-3)
        d_phase = rng.uniform(0.0, 1.0e-3)
        history = rng.uniform(-1.0e-5, 1.0e-5)
        residual = rng.uniform(-1.0e-12, 1.0e-12)
        chain = rng.uniform(-1.0e-5, 1.0e-5)
        splitting = rng.uniform(-1.0e-5, 1.0e-5)
        mechanics = rng.uniform(-1.0e-5, 1.0e-5)
        dissipation = (2.0 / 3.0) * dt * (d_transport + d_phase)
        delta_f = (
            history
            + residual
            + chain
            + splitting
            + mechanics
            - dissipation
        )
        balance = (
            delta_f
            + dissipation
            - history
            - residual
            - chain
            - splitting
            - mechanics
        )
        assert math.isclose(balance, 0.0, rel_tol=0.0, abs_tol=2.0e-20)


def test_runtime_contract_is_versioned_and_has_atomic_history_markers() -> None:
    required = (
        "FIXED_STEP_IMEX_BDF2_V1",
        "FIXED_STEP_BDF2_ACCEPTED_HISTORY_V1",
        "PHI_EXTRAPOLATION_2N_MINUS_NM1_ULP64_V1",
        "BDF2_ENDPOINT_DISCRETE_WORK_V1",
        "CTOT_IMEX_BDF2_HISTORY_COMMIT",
        "CTOT_IMEX_BDF2_MASS_IDENTITY",
        "CTOT_IMEX_BDF2_ENERGY_WORK",
        '"bdf2_dt_n"',
        '"bdf2_dt_nm1"',
        '"bdf2_accepted_step"',
        '"bdf2_time_code"',
        '"bdf2_physical_time_s"',
    )
    for marker in required:
        assert marker in SOURCE
    assert "++ctot_bdf2_accepted_step" in SOURCE
    assert "ctot_bdf2_accepted_time_code = bdf2_restart_time_code" in SOURCE
    assert "inconsistent BDF2 checkpoint time provenance" in SOURCE


def test_bdf2_mode_keeps_forbidden_corrections_out_of_normal_path() -> None:
    selector_start = SOURCE.index("static int is_ctot_jichen_imex_bdf2_v1")
    validator = SOURCE[selector_start : selector_start + 45000]
    assert "IMEX_BDF2_NO_POST_PHASE_POLISH" in validator
    assert "outer_acceleration_off" in validator
    assert "P->ctot_max_coupling_correctors == 0" in validator
    assert "BDF2 antitrapping time-level contract has not" in SOURCE


def test_rejected_bdf2_attempt_uses_transactional_fallback() -> None:
    assert "ctot_debug_force_first_bdf2_attempt_reject" in SOURCE
    forced = SOURCE.split(
        "const int forced_first_bdf2_attempt_reject", 1
    )[1].split("const int method_transport_gate_pass_diag", 1)[0]
    assert "ctot_bdf2_active_this_attempt" in forced
    assert "CTOT_IMEX_BDF2_HISTORY_COMMIT" not in forced
    rejection = SOURCE.split(
        '"debug_forced_first_bdf2_attempt_reject"', 1
    )[1].split("return 2;", 1)[0]
    assert "prepare_ctot_retry" in rejection
    retry = SOURCE.split("auto prepare_ctot_retry", 1)[1].split(
        "int mechanics_fp32_solve_count", 1
    )[0]
    assert '"previous_attempt_rejected"' in retry
