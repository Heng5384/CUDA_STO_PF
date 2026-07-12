#!/usr/bin/env python3
"""Focused host-side tests for conservative restart/checkpoint contracts."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "scripts/analyze_pf_ctot_qalpha_validation.py"


def load_analyzer():
    spec = importlib.util.spec_from_file_location("pf_checkpoint_analyzer", ANALYZER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_checkpoint(path: Path, values: np.ndarray, shape=None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    values.astype(np.float64).tofile(path / "conservative_storage_final.raw")
    meta = {
        "mode": "ctot_conservative_split",
        "dtype": "float64",
        "shape": list(shape or values.shape),
        "step": 4,
        "physical_time_code": 0.001,
        "raw_file": "conservative_storage_final.raw",
        "authoritative": True,
    }
    (path / "conservative_storage_final.json").write_text(json.dumps(meta) + "\n")


def expect_failure(callable_, expected_exception) -> None:
    try:
        callable_()
    except expected_exception:
        return
    raise AssertionError(f"expected {expected_exception.__name__}")


def main() -> int:
    analyzer = load_analyzer()
    with tempfile.TemporaryDirectory(prefix="pf_storage_recovery_") as tmp:
        tmp_path = Path(tmp)
        values = np.linspace(0.1, 0.9, 24).reshape(2, 3, 4)

        good = tmp_path / "good"
        write_checkpoint(good, values)
        loaded, meta = analyzer.read_conservative_checkpoint(good, values.shape)
        assert np.array_equal(loaded, values)
        assert meta["step"] == 4
        assert meta["physical_time_code"] == 0.001

        missing = tmp_path / "missing"
        write_checkpoint(missing, values)
        (missing / "conservative_storage_final.raw").unlink()
        expect_failure(
            lambda: analyzer.read_conservative_checkpoint(missing, values.shape),
            FileNotFoundError,
        )

        wrong_shape = tmp_path / "wrong_shape"
        write_checkpoint(wrong_shape, values, shape=(2, 2, 6))
        expect_failure(
            lambda: analyzer.read_conservative_checkpoint(wrong_shape, values.shape),
            ValueError,
        )

        wrong_size = tmp_path / "wrong_size"
        write_checkpoint(wrong_size, values)
        with (wrong_size / "conservative_storage_final.raw").open("ab") as handle:
            handle.write(b"extra")
        expect_failure(
            lambda: analyzer.read_conservative_checkpoint(wrong_size, values.shape),
            ValueError,
        )

        manifest = tmp_path / "refinement_manifest.json"
        subprocess.run(
            [sys.executable, str(ANALYZER), "--dry-run-manifest", str(manifest)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        rows = json.loads(manifest.read_text())
        fine = [row for row in rows if row["dt"] == 0.00025]
        assert len(fine) == 2
        assert all(row["nsteps"] == 2000 for row in fine)
        assert all(abs(row["physical_time_code"] - 0.5) < 1.0e-15 for row in rows)

    source = (ROOT / "main_cuda.cu").read_text()
    assert "PF_CONSERVATIVE_RESTART_LOADED" in source
    assert "PF_CONSERVATIVE_CHECKPOINT_WRITTEN" in source
    assert "--init-conservative-storage-raw requires" in source
    assert "pf_composition_mode=ctot_conservative_split" in source
    assert "qalpha_conservative_local_transaction" in source
    assert "has_trailing_byte" in source
    assert "validate_conservative_storage_checkpoint_meta" in source
    print("conservative_storage_recovery_tests=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
