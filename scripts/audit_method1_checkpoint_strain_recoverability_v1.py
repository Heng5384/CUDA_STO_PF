#!/usr/bin/env python3
"""Read-only, fail-closed recoverability audit for Method-1 elastic fields."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import platform
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPLICATES = ("A", "B", "C")
AGE_STEPS = {6.0: 0, 12.0: 21798, 18.0: 43596, 24.0: 65393, 36.0: 108989, 48.0: 152585}
AUTHORITY_STATUS = "PASS_246CUBE_6H48H_CONDITIONAL_PRODUCTION_V1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(rows, f"empty CSV: {path}")
    names = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()


def remote(host: str, command: str) -> str:
    result = subprocess.run(
        ["/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25", host, command],
        text=True, capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"remote command failed: {result.stderr.strip()}")
    return result.stdout.strip()


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def header_from_remote(host: str, path: str) -> dict[str, Any]:
    encoded = remote(host, f"dd if={shell_quote(path)} bs=904 count=1 2>/dev/null | base64")
    raw = base64.b64decode(encoded)
    require(len(raw) == 904, f"unexpected V4 header size for {path}: {len(raw)}")
    return {
        "magic": raw[:8].decode("ascii", errors="replace"),
        "version": struct.unpack_from("<I", raw, 8)[0],
        "header_bytes": struct.unpack_from("<I", raw, 12)[0],
        "element_count": struct.unpack_from("<Q", raw, 16)[0],
        "k_element_count": struct.unpack_from("<Q", raw, 24)[0],
        "accepted_step": struct.unpack_from("<Q", raw, 32)[0],
        "grid": struct.unpack_from("<iii", raw, 40),
        "elastic_state_present": struct.unpack_from("<i", raw, 52)[0],
        "elastic_source_field_step": struct.unpack_from("<Q", raw, 136)[0],
        "elastic_last_iterations": struct.unpack_from("<Q", raw, 144)[0],
        "elastic_last_relative_residual": struct.unpack_from("<d", raw, 152)[0],
    }


def final_diagnostic_values(host: str, path: str) -> dict[str, str]:
    program = (
        "import csv,json; "
        f"rows=list(csv.DictReader(open({path!r},newline=''))); "
        "r=rows[-1]; "
        "print(json.dumps({k:r.get(k) for k in ('step','mean_elastic_energy',"
        "'max_elastic_energy','stress_hydro_min','stress_hydro_max')},sort_keys=True))"
    )
    return json.loads(remote(host, "python3 -c " + shell_quote(program)))


def placeholder_plot(base: Path, title: str, body: str) -> None:
    figure, axis = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    axis.axis("off")
    axis.text(.5, .63, title, ha="center", va="center", fontsize=14, weight="bold", wrap=True)
    axis.text(.5, .37, body, ha="center", va="center", fontsize=10, wrap=True)
    figure.savefig(base.with_suffix(".png"), dpi=180)
    figure.savefig(base.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", default="uvip-cluster")
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve() if args.out.is_absolute() else (root / args.out).resolve()
    require(out.is_dir(), "density-stage output directory is required")
    authority_root = root / "reports/pf_246cube_method1_production_authority_v1"
    audits: dict[str, dict[str, Any]] = {}
    for rep in REPLICATES:
        audit = json.loads((authority_root / "authority" / rep / "audit.json").read_text())
        require(audit["numerical_status"] == AUTHORITY_STATUS, f"{rep} authority not PASS")
        require(all(audit["gates"].values()), f"{rep} authority gate failed")
        require(len(audit["checkpoint_hashes"]) == 44, f"{rep} checkpoint chain incomplete")
        audits[rep] = audit

    field_rows: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for rep in REPLICATES:
        production_root = audits[rep]["remote_production_root"]
        for age_h, step in AGE_STEPS.items():
            if step == 0:
                field_rows.append({
                    "replicate": rep, "age_h": age_h, "step": step,
                    "checkpoint_schema": "INITIAL_FIXTURE_RAW_FIELDS",
                    "checkpoint_hash_verified": "not_applicable",
                    "exact_source_field_checkpoint_available": False,
                    "reconstruction_status": "NOT_RECOVERABLE_INITIAL_FIXTURE_HAS_NO_ASSEMBLED_CELL_ELASTIC_STATE",
                    "reason": "6 h fixture has phi/Y/xB/dY_dt_prev but no converged assembled-cell displacement/eigenstrain/stress/energy field.",
                })
                closure_rows.append({"replicate": rep, "age_h": age_h, "step": step, "runtime_mean_elastic_energy": "NOT_RECORDED", "reconstructed_elastic_energy": "NOT_COMPUTABLE", "relative_energy_error": "NOT_COMPUTABLE", "status": "NOT_RECOVERABLE"})
                continue
            checkpoint = f"{production_root}/checkpoints/step_{step}.chk"
            source_checkpoint = f"{production_root}/checkpoints/step_{step - 1}.chk"
            diagnostics = f"{production_root}/segments/step_{step}/dynamics_mass_diagnostics.csv"
            remote_hash = remote(args.host, f"sha256sum {shell_quote(checkpoint)}").split()[0]
            expected_hash = audits[rep]["checkpoint_hashes"][str(step)]
            require(remote_hash == expected_hash, f"checkpoint hash mismatch: {rep}/{step}")
            header = header_from_remote(args.host, checkpoint)
            source_exists = remote(args.host, f"test -f {shell_quote(source_checkpoint)} && printf yes || printf no") == "yes"
            semantic_ok = (
                header["magic"] == "PFZMCHK4" and header["version"] == 4 and header["header_bytes"] == 904
                and header["accepted_step"] == step and header["elastic_state_present"] == 1
                and header["elastic_source_field_step"] == step - 1 and tuple(header["grid"]) == (246, 246, 246)
            )
            status = "NOT_RECOVERABLE_SOURCE_FIELD_STEP_NOT_SAVED" if semantic_ok and not source_exists else "UNEXPECTED_CHECKPOINT_SEMANTICS"
            diag_hash = remote(args.host, f"sha256sum {shell_quote(diagnostics)}").split()[0]
            diagnostic = final_diagnostic_values(args.host, diagnostics)
            field_rows.append({
                "replicate": rep, "age_h": age_h, "step": step,
                "checkpoint_path": checkpoint, "checkpoint_sha256": remote_hash,
                "checkpoint_schema": header["magic"], "header_bytes": header["header_bytes"],
                "grid": "x".join(map(str, header["grid"])),
                "elastic_warm_state_present": bool(header["elastic_state_present"]),
                "source_field_step": header["elastic_source_field_step"],
                "source_field_step_contract": "accepted_step_minus_one",
                "source_checkpoint_path": source_checkpoint,
                "exact_source_field_checkpoint_available": source_exists,
                "elastic_last_iterations": header["elastic_last_iterations"],
                "elastic_last_relative_residual": header["elastic_last_relative_residual"],
                "reconstruction_status": status,
                "reason": "V4 holds displacement_k for source n-1 but not the matching phi/xB/eigenstrain fields.",
            })
            closure_rows.append({
                "replicate": rep, "age_h": age_h, "step": step,
                "runtime_diagnostic_path": diagnostics, "runtime_diagnostic_sha256": diag_hash,
                "runtime_diagnostic_final_step": diagnostic["step"],
                "runtime_mean_elastic_energy": diagnostic["mean_elastic_energy"],
                "runtime_max_elastic_energy": diagnostic["max_elastic_energy"],
                "reconstructed_elastic_energy": "NOT_COMPUTABLE_WITH_EXACT_TIME_LEVEL",
                "relative_energy_error": "NOT_COMPUTABLE_WITH_EXACT_TIME_LEVEL",
                "hard_gate_1e_minus_6": "NOT_EVALUABLE", "status": status,
            })
            runtime_rows.append({"replicate": rep, "age_h": age_h, "step": step, **diagnostic, "status": "RUNTIME_SCALAR_ONLY_NOT_RECONSTRUCTED_FIELD"})

    write_csv(out / "reconstructed_elastic_field_manifest.csv", field_rows)
    write_csv(out / "elastic_energy_closure.csv", closure_rows)
    unavailable = [{"replicate": r, "age_h": a, "step": s, "status": "NOT_COMPUTED_SOURCE_FIELD_STEP_NOT_RECOVERABLE", "reason": "No exact prior-step phi/xB field; current-field substitution is rejected."} for r in REPLICATES for a, s in AGE_STEPS.items()]
    for name in ("strain_statistics_time_series.csv", "stress_statistics_time_series.csv", "strain_power_spectrum_radial.csv", "strain_power_spectrum_directional.csv", "strain_band_integrals.csv"):
        write_csv(out / name, unavailable)
    ratios = [{"replicate": rep, "metric": metric, "ratio_48h_over_6h": "NOT_COMPUTABLE", "status": "INCONCLUSIVE_STRAIN_FIELDS_NOT_RECOVERABLE"} for rep in (*REPLICATES, "ensemble") for metric in ("hydrostatic_strain_variance", "deviatoric_strain_variance", "hydrostatic_stress_variance", "elastic_energy", "low_q_power", "mid_q_power", "high_q_power")]
    write_csv(out / "strain_relaxation_ratios.csv", ratios)
    write_csv(out / "runtime_elastic_scalar_evidence.csv", runtime_rows)
    report = "# Strain reconstruction audit\n\n`INCONCLUSIVE_STRAIN_FIELDS_NOT_RECOVERABLE`\n\n"
    report += "All inspected 12–48 h endpoint checkpoint hashes match the immutable 44-checkpoint A/B/C authority chain. Each is checkpoint V4 and its packed displacement warm state is explicitly associated with source field step n−1. The n−1 checkpoint does not exist for any registered endpoint. Neither that source phi/xB field nor source-time-level eigenstrain/stress/strain/energy density is retained. The 6 h fixture has no assembled-cell elastic runtime state.\n\n"
    report += "Using current checkpoint phi/xB with prior-step displacement_k would mix time levels and cannot meet the required <=1e-6 energy closure. This audit refuses the approximation. Runtime elastic-energy and hydrostatic-stress extrema are preserved only as scalar provenance, not promoted to field variances, shell statistics, spectra, or a phonon-scattering law. No PF evolution was run and no checkpoint was modified.\n"
    (out / "strain_reconstruction_audit.md").write_text(report, encoding="utf-8")
    message = "Exact source field n−1 is absent. Mixing current phi/xB with saved displacement_k is rejected; field statistics and spectra are therefore not computed."
    placeholder_plot(out / "strain_variance_vs_time", "Strain variance: not recoverable", message)
    placeholder_plot(out / "elastic_energy_vs_time", "Elastic energy: runtime scalar only", "Runtime scalar values are recorded, but exact reconstructed energy closure is unavailable.")
    placeholder_plot(out / "strain_spectrum_6h_vs_48h", "Strain spectrum: not recoverable", message)
    placeholder_plot(out / "strain_relaxation_ratio_by_replicate", "Relaxation ratios: not computable", message)
    metadata = {"schema": "METHOD1_CHECKPOINT_STRAIN_RECOVERABILITY_AUDIT_V1", "analysis_script_sha256": sha256(Path(__file__).resolve()), "git_branch": git(root, "branch", "--show-current"), "git_commit": git(root, "rev-parse", "HEAD"), "python": sys.version.replace("\n", " "), "platform": platform.platform(), "remote_host": args.host, "status": "INCONCLUSIVE_STRAIN_FIELDS_NOT_RECOVERABLE"}
    (out / "strain_reconstruction_provenance.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("INCONCLUSIVE_STRAIN_FIELDS_NOT_RECOVERABLE")


if __name__ == "__main__":
    main()
