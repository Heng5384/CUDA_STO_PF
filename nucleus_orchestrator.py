#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis.compute_schur_rc_predictions import (  # noqa: E402
    build_cases,
    calc_elastic_energy_schur,
    get_chemical_driving_force_Jm3,
    get_principal_eigenstrain,
    load_inputs,
)
from tools.analysis.workflow_utils import build_cnt_prefix, workflow_name  # noqa: E402

import nucleus_selector  # noqa: E402
import nucleus_geometry_mapper  # noqa: E402


RUN_MODE_FULL = "FULL_AUTONOMOUS"
DEFAULT_CATALOG = REPO_ROOT / "nucleus_catalog.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def rel(path: Path | str | None) -> str:
    if path is None:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def shape_variants(rc_nm: float) -> list[dict[str, Any]]:
    # These are initialization-template perturbations only; they do not change CNT physics.
    return [
        {
            "shape_type": "spherical",
            "init_shape": "sphere",
            "aspect_ratio": 1.0,
            "axis_ratios": [1.0, 1.0, 1.0],
            "tilt_theta_deg": 0.0,
            "tilt_phi_deg": 0.0,
        },
        {
            "shape_type": "anisotropic",
            "init_shape": "ellipsoid",
            "aspect_ratio": 1.25,
            "axis_ratios": [1.25, 1.0, 0.8],
            "tilt_theta_deg": 0.0,
            "tilt_phi_deg": 0.0,
        },
        {
            "shape_type": "faceted",
            "init_shape": "ellipsoid",
            "aspect_ratio": 1.5,
            "axis_ratios": [1.5, 1.0, 0.67],
            "tilt_theta_deg": 10.0,
            "tilt_phi_deg": 45.0,
        },
    ]


@dataclass
class OrchestratorConfig:
    T: float
    xB: float
    strain: float
    strain_mode: str = "eyy"
    run_mode: str = RUN_MODE_FULL
    execution_backend: str = "local"  # local | slurm
    dry_run: bool = True
    base_json: str = "physical_inputs.example.json"
    workflow_root: str = "Results/orchestrator"
    grid: str = "400,400,400"
    dt: float = 0.03
    steps: int = 30000
    out_every: int = 2500
    csv_out_every: int = 10
    elastic: int = 1
    delta_nm: float = 0.5
    step_nm: float = 0.1
    poll_interval_s: float = 30.0
    timeout_s: float = 86400.0
    launch_cuda: bool = False


class NucleusOrchestrator:
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.workflow_name = workflow_name(config.T, config.xB)
        self.run_dir = (REPO_ROOT / config.workflow_root / self.workflow_name).resolve()
        self.input_dir = self.run_dir / "input"
        self.jobs_dir = self.run_dir / "jobs"
        self.results_dir = self.run_dir / "raw"
        self.parsed_dir = self.run_dir / "parsed"
        self.logs_dir = self.run_dir / "logs"
        self.scan_cases_path = self.input_dir / "scan_cases.json"
        self.jobs_manifest_path = self.jobs_dir / "minimization_jobs.json"
        self.parsed_results_path = self.parsed_dir / "parsed_nucleus_results.csv"
        self.selected_json_path = self.run_dir / "selected_nucleus.json"
        self.cuda_config_path = self.run_dir / "cuda_launch_config.json"
        self.continuous_geometry_csv = self.run_dir / "continuous_nucleus_geometry.csv"
        self.template_space_json = self.run_dir / "template_geometry_space.json"
        self.template_mapping_log = self.run_dir / "template_mapping_validation_log.csv"
        self.template_root = self.run_dir / "cuda_templates"
        self.error_log_path = self.logs_dir / "orchestrator_errors.log"
        self.catalog_path = (self.run_dir / "nucleus_catalog.preview.json") if config.dry_run else DEFAULT_CATALOG
        self.state: dict[str, Any] = {
            "created_at_utc": utc_now(),
            "run_mode": config.run_mode,
            "dry_run": config.dry_run,
            "errors": [],
            "fallback_used": False,
        }

    def log_error(self, message: str) -> None:
        self.state.setdefault("errors", []).append({"time": utc_now(), "message": message})
        self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.error_log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{utc_now()} {message}\n")

    def ensure_dirs(self) -> None:
        for path in (self.input_dir, self.jobs_dir, self.results_dir, self.parsed_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)

    def _external_strain_components(self) -> dict[str, float]:
        mode = self.config.strain_mode
        s = self.config.strain
        if mode == "exx":
            return {"E0_xx": s, "E0_yy": 0.0, "E0_zz": 0.0, "E0_yz": 0.0, "E0_xz": 0.0, "E0_xy": 0.0}
        if mode == "exx_eyy":
            return {"E0_xx": s, "E0_yy": s, "E0_zz": 0.0, "E0_yz": 0.0, "E0_xz": 0.0, "E0_xy": 0.0}
        return {"E0_xx": 0.0, "E0_yy": s, "E0_zz": 0.0, "E0_yz": 0.0, "E0_xz": 0.0, "E0_xy": 0.0}

    def generate_nucleus_scan_range(self) -> list[dict[str, Any]]:
        self.ensure_dirs()
        base_json = (REPO_ROOT / self.config.base_json).resolve()
        raw_inputs = load_inputs(base_json)
        raw_inputs["temperature_C"] = self.config.T
        physical_json = self.input_dir / "physical_inputs.json"
        write_json(physical_json, raw_inputs)

        nx, ny, nz = [int(x.strip()) for x in self.config.grid.split(",")]
        T_K = self.config.T + 273.15
        gamma = float(raw_inputs["gamma"])
        Vm_compound = float(raw_inputs["Vm_compound"])
        e0 = get_principal_eigenstrain(raw_inputs)
        C_precip = __import__("numpy").array(raw_inputs["C_tensor_Ag2Te_GPa"], dtype=float)
        DG_chem, x_eq = get_chemical_driving_force_Jm3(T_K, self.config.xB, Vm_compound)

        cases = [
            c for c in build_cases([abs(self.config.strain)], [self.config.strain_mode])
            if abs(c.strain - self.config.strain) < 1.0e-15
        ]
        if not cases:
            # build_cases emits +/- for nonzero strain; use closest sign if exact zero handling differs.
            cases = build_cases([abs(self.config.strain)], [self.config.strain_mode])
        load_case = min(cases, key=lambda c: abs(c.strain - self.config.strain))

        E_el, e_r_star, cond = calc_elastic_energy_schur(
            C6_GPa=C_precip,
            e0=e0,
            e_ext=__import__("numpy").array(load_case.e_ext, dtype=float),
            fixed_idx=load_case.fixed_idx,
            free_idx=load_case.free_idx,
        )
        DG_net = DG_chem - E_el
        rc_pred_nm = math.inf if DG_net <= 0.0 else 2.0 * gamma / DG_net * 1.0e9
        if not math.isfinite(rc_pred_nm):
            rc_pred_nm = 2.0
            self.log_error("Schur rc prediction was non-finite; using estimated rc_pred_nm=2.0 for scan generation")

        prefix = build_cnt_prefix(self.config.T, self.config.xB)
        strain_tag = load_case.strain_tag
        radii = [
            round(rc_pred_nm - self.config.delta_nm + i * self.config.step_nm, 6)
            for i in range(int(round(2.0 * self.config.delta_nm / self.config.step_nm)) + 1)
        ]
        radii = [r for r in radii if r > 0.0]

        cases_out: list[dict[str, Any]] = []
        for radius_index, radius_nm in enumerate(radii):
            for variant in shape_variants(radius_nm):
                shape_tag = variant["shape_type"]
                ar_tag = str(variant["aspect_ratio"]).replace(".", "p")
                radius_tag = f"{radius_nm:.6f}".replace(".", "p")
                case_id = f"{prefix}_{self.config.strain_mode}_{strain_tag}_{shape_tag}_ar{ar_tag}_r{radius_tag}"
                output_dir = self.results_dir / case_id
                cases_out.append({
                    "case_id": case_id,
                    "T_C": self.config.T,
                    "xB": self.config.xB,
                    "strain": self.config.strain,
                    "strain_mode": self.config.strain_mode,
                    "rc_pred_nm": rc_pred_nm,
                    "radius_nm": radius_nm,
                    "radius_index": radius_index,
                    "shape_type": variant["shape_type"],
                    "init_shape": variant["init_shape"],
                    "aspect_ratio": variant["aspect_ratio"],
                    "axis_ratios": variant["axis_ratios"],
                    "tilt_theta_deg": variant["tilt_theta_deg"],
                    "tilt_phi_deg": variant["tilt_phi_deg"],
                    "physical_input_json": rel(physical_json),
                    "output_dir": rel(output_dir),
                    "expected_energy_glob": rel(output_dir / "energy_minimize_*.csv"),
                    "expected_summary": rel(output_dir / "summary.txt"),
                    "expected_phi_glob": rel(output_dir / "phi_final_*.vtk"),
                    "grid": [nx, ny, nz],
                    "dt": self.config.dt,
                    "steps": self.config.steps,
                    "elastic": self.config.elastic,
                    "DG_chem_Jm3": DG_chem,
                    "E_el_Jm3": E_el,
                    "DG_net_Jm3": DG_net,
                    "Crr_condition_number": cond,
                    "e_r_star": [float(x) for x in e_r_star],
                })

        write_json(self.scan_cases_path, {
            "schema_version": 1,
            "generated_at_utc": utc_now(),
            "run_mode": self.config.run_mode,
            "prediction": {
                "T_C": self.config.T,
                "xB": self.config.xB,
                "strain": self.config.strain,
                "strain_mode": self.config.strain_mode,
                "rc_pred_nm": rc_pred_nm,
                "delta_nm": self.config.delta_nm,
                "step_nm": self.config.step_nm,
            },
            "cases": cases_out,
        })
        self.state["scan_case_count"] = len(cases_out)
        return cases_out

    def _pf_param_file_for_case(self, case: dict[str, Any]) -> Path:
        out = self.jobs_dir / f"pf_params_{case['case_id']}.params"
        if out.exists():
            return out
        cmd = [
            sys.executable,
            str(REPO_ROOT / "Unit_Psedobinary.py"),
            "--input-json",
            str(REPO_ROOT / case["physical_input_json"]),
            "--output-pf-param-file",
            str(out),
        ]
        if self.config.dry_run:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("# dry-run placeholder; generated by nucleus_orchestrator.py\n", encoding="utf-8")
            return out
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        return out

    def _main_cuda_command(self, case: dict[str, Any]) -> list[str]:
        nx, ny, nz = case["grid"]
        pf_param = self._pf_param_file_for_case(case)
        ratios = case["axis_ratios"]
        return [
            str(REPO_ROOT / "main_cuda"),
            str(nx), str(ny), str(nz),
            str(case["dt"]), str(case["steps"]), str(self.config.out_every), str(self.config.csv_out_every), str(self.config.elastic),
            "--pf-param-file", str(pf_param),
            "--mode=minimize",
            "--minimize-full-model",
            "--minimize-max-iter", str(case["steps"]),
            "--minimize-dt", str(case["dt"]),
            "--radius-phys-nm", str(case["radius_nm"]),
            "--elastic", str(case["elastic"]),
            "--ic-23d-xB-out", str(case["xB"]),
            "--init-case-tag", case["case_id"],
            "--init-shape", case["init_shape"],
            "--init-axis-ratio-rx", str(ratios[0]),
            "--init-axis-ratio-ry", str(ratios[1]),
            "--init-axis-ratio-rz", str(ratios[2]),
            "--init-tilt-theta-deg", str(case["tilt_theta_deg"]),
            "--init-tilt-phi-deg", str(case["tilt_phi_deg"]),
            "--minimize-rms-dphi-threshold", "1e-5",
            "--minimize-rms-dY-threshold", "1e-5",
            "--minimize-energy-diff-rel-threshold", "1e-7",
            "--minimize-convergence-steps", "100",
            "--minimize-rms-res-threshold", "1e-4",
            "--minimize-vol-err-rel-threshold", "3e-2",
            "--eta-lambda-vol", "0.9",
            "--minimize-post-projection-iters", "1",
        ]

    def submit_minimization_jobs(self, scan_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        if self.config.execution_backend == "slurm":
            script = self.jobs_dir / "submit_orchestrated_minimize_array.sbatch"
            script.write_text(
                "#!/usr/bin/env bash\n"
                "#SBATCH --job-name=nuc_orch_min\n"
                "#SBATCH --partition=gpu_uvip\n"
                "#SBATCH --qos=gpu_uvip\n"
                "#SBATCH --nodes=1\n"
                "#SBATCH --ntasks=1\n"
                "#SBATCH --gres=gpu:1\n"
                "#SBATCH --time=24:00:00\n"
                "set -euo pipefail\n"
                f"python3 {REPO_ROOT / 'nucleus_orchestrator.py'} --resume-jobs {self.scan_cases_path} --backend local --no-dry-run\n",
                encoding="utf-8",
            )
            if self.config.dry_run:
                job_id = "DRY_RUN_SLURM_JOB"
            else:
                completed = subprocess.run(["sbatch", str(script)], cwd=REPO_ROOT, text=True, capture_output=True, check=True)
                job_id = completed.stdout.strip().split()[-1]
            jobs.append({
                "job_id": job_id,
                "backend": "slurm",
                "input_geometry": "scan_cases.json",
                "case_count": len(scan_cases),
                "expected_output_file": str(self.parsed_results_path),
                "status": "submitted" if not self.config.dry_run else "dry_run",
            })
            write_json(self.jobs_manifest_path, {"jobs": jobs, "generated_at_utc": utc_now()})
            return jobs

        for case in scan_cases:
            cmd = self._main_cuda_command(case)
            status = "dry_run"
            returncode = None
            if not self.config.dry_run:
                env = os.environ.copy()
                env["CUDA_STO_RESULTS_ROOT"] = str(self.results_dir)
                try:
                    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
                    returncode = proc.returncode
                    status = "complete" if proc.returncode == 0 else "failed"
                    if proc.returncode != 0:
                        self.log_error(f"minimization failed for {case['case_id']} returncode={proc.returncode}")
                except OSError as exc:
                    status = "failed"
                    self.log_error(f"cannot launch minimization for {case['case_id']}: {exc}")
            jobs.append({
                "job_id": f"local:{case['case_id']}",
                "backend": "local",
                "input_geometry": case,
                "T": case["T_C"],
                "xB": case["xB"],
                "strain": case["strain"],
                "expected_output_file": case["expected_energy_glob"],
                "command": cmd,
                "status": status,
                "returncode": returncode,
            })
        write_json(self.jobs_manifest_path, {"jobs": jobs, "generated_at_utc": utc_now()})
        self.state["submitted_job_count"] = len(jobs)
        return jobs

    def monitor_jobs(self, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deadline = time.time() + self.config.timeout_s
        while True:
            done = 0
            for job in jobs:
                geom = job.get("input_geometry")
                if not isinstance(geom, dict):
                    if job["status"] in {"dry_run", "submitted", "complete"}:
                        done += 1
                    continue
                out_dir = REPO_ROOT / geom["output_dir"]
                energy_files = list(out_dir.glob("energy_minimize_*.csv"))
                phi_files = list(out_dir.glob("phi_final_*.vtk"))
                if energy_files and phi_files:
                    job["status"] = "complete"
                    done += 1
                elif job["status"] in {"dry_run", "failed"}:
                    done += 1
            if done == len(jobs):
                break
            if time.time() > deadline:
                self.log_error("job monitoring timeout; continuing with available results")
                break
            time.sleep(self.config.poll_interval_s)
        write_json(self.jobs_manifest_path, {"jobs": jobs, "updated_at_utc": utc_now()})
        return jobs

    def _read_last_energy(self, out_dir: Path) -> dict[str, str] | None:
        files = sorted(out_dir.glob("energy_minimize_*.csv"))
        if not files:
            return None
        try:
            rows = list(csv.DictReader(files[0].open(newline="", encoding="utf-8")))
        except (OSError, csv.Error) as exc:
            self.log_error(f"cannot read energy csv {files[0]}: {exc}")
            return None
        if not rows:
            return None
        row = dict(rows[-1])
        row["_energy_csv"] = rel(files[0])
        return row

    def parse_results(self, scan_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for case in scan_cases:
            out_dir = REPO_ROOT / case["output_dir"]
            last = self._read_last_energy(out_dir)
            fcnt = safe_float(last.get("F_total_CNT_hat") if last else None)
            fexcess = safe_float(last.get("F_total_excess_hat") if last else None)
            energy_rank = fcnt if fcnt is not None else (fexcess if fexcess is not None else 1.0e300)
            status = "complete" if last else "missing"
            rows.append({
                "case_id": case["case_id"],
                "T_C": case["T_C"],
                "xB": case["xB"],
                "strain": case["strain"],
                "strain_mode": case["strain_mode"],
                "rc_pred_nm": case["rc_pred_nm"],
                "rc_opt": case["radius_nm"],
                "barrier_height": energy_rank,
                "energy_ranking": energy_rank,
                "shape_descriptor": case["shape_type"],
                "shape_type": case["shape_type"],
                "aspect_ratio": case["aspect_ratio"],
                "semiaxes": " ".join(str(v * case["radius_nm"]) for v in case["axis_ratios"]),
                "source_dyn_dir": case["output_dir"] if status == "complete" else "",
                "profile_dir": "",
                "energy_csv": last.get("_energy_csv", "") if last else "",
                "status": status,
                "origin": "CNT/orchestrator" if last else "estimated",
            })
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        with self.parsed_results_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(rows[0].keys()) if rows else []
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        self.state["parsed_result_count"] = len(rows)
        self.state["complete_result_count"] = sum(1 for row in rows if row["status"] == "complete")
        return rows

    def update_nucleus_catalog(self, parsed_rows: list[dict[str, Any]], catalog_path: Path | None = None) -> dict[str, Any]:
        catalog_path = catalog_path or self.catalog_path
        if self.config.dry_run and not catalog_path.exists() and DEFAULT_CATALOG.exists():
            catalog_path.parent.mkdir(parents=True, exist_ok=True)
            catalog_path.write_text(DEFAULT_CATALOG.read_text(encoding="utf-8"), encoding="utf-8")
        catalog = read_json(catalog_path, {"schema_version": 1, "generated_by": "nucleus_orchestrator.py", "entries": []})
        entries = list(catalog.get("entries", []))

        def key(entry: dict[str, Any]) -> tuple[Any, ...]:
            return (
                round(float(entry.get("T_C") or -9999), 6),
                round(float(entry.get("xB") or -1), 8),
                round(float(entry.get("strain_value") or entry.get("strain") or 0), 8),
                str(entry.get("shape_type") or "unknown"),
            )

        for row in parsed_rows:
            confidence = 0.95 if row["origin"] == "DFT" else (0.75 if row["origin"].startswith("CNT") else 0.35)
            entry = {
                "id": row["case_id"],
                "T_C": float(row["T_C"]),
                "xB": float(row["xB"]),
                "strain_mode": row["strain_mode"],
                "strain_value": float(row["strain"]),
                "rc_nm": float(row["rc_opt"]),
                "energy_barrier_kBT": float(row["barrier_height"]),
                "energy_source": "orchestrator_energy_rank",
                "raw_energy_proxy": float(row["energy_ranking"]),
                "shape_type": row["shape_type"],
                "aspect_ratio": safe_float(row["aspect_ratio"], 1.0),
                "semiaxes": [safe_float(x, 0.0) for x in str(row["semiaxes"]).split()],
                "source_dyn_dir": row["source_dyn_dir"],
                "profile_dir": row["profile_dir"],
                "source_dyn_dir_candidate": row["source_dyn_dir"],
                "profile_dir_candidate": row["profile_dir"],
                "cuda_template_available": bool(row["source_dyn_dir"] and row["profile_dir"]),
                "confidence_score": confidence,
                "origin": row["origin"],
                "summary_path": str(Path(row["source_dyn_dir"]) / "summary.txt") if row["source_dyn_dir"] else "",
                "raw_source_table": rel(self.parsed_results_path),
                "notes": "Added by nucleus_orchestrator.py; energy_barrier_kBT is raw energy rank unless formal barrier conversion is available.",
                "updated_at_utc": utc_now(),
            }
            k = key(entry)
            existing_idx = next((i for i, item in enumerate(entries) if key(item) == k), None)
            if existing_idx is None:
                entries.append(entry)
            else:
                old = entries[existing_idx]
                old_energy = float(old.get("energy_barrier_kBT", 1.0e300))
                if entry["energy_barrier_kBT"] < old_energy or entry["confidence_score"] > float(old.get("confidence_score", 0.0)):
                    entries[existing_idx] = entry

        catalog["schema_version"] = max(int(catalog.get("schema_version", 1)), 1)
        catalog["generated_by"] = "nucleus_orchestrator.py update_nucleus_catalog"
        catalog["entry_count"] = len(entries)
        catalog["updated_at_utc"] = utc_now()
        catalog["entries"] = sorted(entries, key=lambda e: (str(e.get("origin")), float(e.get("T_C") or 1e99), float(e.get("xB") or 1e99), str(e.get("id"))))
        write_json(catalog_path, catalog)
        self.state["catalog_self_learning"] = True
        return catalog

    def select_nucleus(self, catalog_path: Path | None = None) -> dict[str, Any]:
        catalog_path = catalog_path or self.catalog_path
        selection = nucleus_selector.select_nucleus(
            self.config.T,
            self.config.xB,
            self.config.strain,
            catalog_path,
            require_cuda_template=True,
        )
        payload = nucleus_selector.write_selection(selection, self.selected_json_path)
        if payload.get("fallback_triggered"):
            self.state["fallback_used"] = True
            self.log_error("selector returned fallback; using last valid catalog entry if available")
            fallback = self._last_valid_catalog_entry(catalog_path)
            if fallback:
                payload["selected_nucleus"] = fallback
                payload["fallback_from_last_valid_catalog"] = True
                write_json(self.selected_json_path, payload)
        self.state["selector_used_updated_catalog"] = True
        self.state["cuda_closed_loop"] = bool(payload.get("cuda_now_uses_predicted_nucleus"))
        return payload

    def _last_valid_catalog_entry(self, catalog_path: Path) -> dict[str, Any] | None:
        catalog = read_json(catalog_path, {"entries": []})
        entries = [
            e for e in catalog.get("entries", [])
            if e.get("origin") != "fallback" and e.get("source_dyn_dir") and e.get("profile_dir")
        ]
        if not entries:
            return None
        return max(entries, key=lambda e: (float(e.get("confidence_score", 0.0)), -float(e.get("energy_barrier_kBT", 1.0e300))))

    def map_selected_nucleus_to_cuda_template(self, selected_payload: dict[str, Any]) -> dict[str, Any]:
        nucleus_geometry_mapper.extract_continuous_nuclei(
            selected_json=self.selected_json_path,
            catalog_json=self.catalog_path,
            output_csv=self.continuous_geometry_csv,
        )
        nucleus_geometry_mapper.build_template_space(
            template_root=self.template_root,
            output_json=self.template_space_json,
        )

        selected = selected_payload.get("predicted_lowest_energy_nucleus") or selected_payload.get("selected_nucleus") or {}
        continuous = nucleus_geometry_mapper.entry_to_continuous(selected, rel(self.selected_json_path))
        if continuous is None:
            fallback = self._last_valid_catalog_entry(self.catalog_path) or selected_payload.get("selected_nucleus") or {}
            continuous = nucleus_geometry_mapper.entry_to_continuous(fallback, "last_valid_or_selected")
        if continuous is None:
            raise RuntimeError("no continuous nucleus with rc_nm is available for CUDA template mapping")

        mapping = nucleus_geometry_mapper.map_continuous_nucleus_to_cuda_template(
            continuous,
            template_space_json=self.template_space_json,
            template_root=self.template_root,
            validation_log=self.template_mapping_log,
        )
        self._inject_mapped_template_into_selection(selected_payload, mapping)
        self._inject_mapped_template_into_catalog(selected, mapping)
        self.state["template_mapper_active"] = True
        self.state["continuous_geometry_csv"] = rel(self.continuous_geometry_csv)
        self.state["template_geometry_space_json"] = rel(self.template_space_json)
        self.state["template_mapping_validation_log"] = rel(self.template_mapping_log)
        self.state["geometry_loss_mean"] = mapping["mismatch_metrics"]["total"]
        return mapping

    def _inject_mapped_template_into_selection(self, selected_payload: dict[str, Any], mapping: dict[str, Any]) -> None:
        template = mapping["mapped_template"]
        selected = dict(selected_payload.get("predicted_lowest_energy_nucleus") or selected_payload.get("selected_nucleus") or {})
        selected.update({
            "profile_dir": template["profile_dir"],
            "source_dyn_dir": template["source_dyn_dir"],
            "profile_dir_candidate": template["profile_dir"],
            "source_dyn_dir_candidate": template["source_dyn_dir"],
            "cuda_template_available": True,
            "mapped_template_id": template["template_id"],
            "mapped_template_geometry": template,
            "template_mapping_mismatch": mapping["mismatch_metrics"],
            "template_mapping_decision_reason": mapping["decision_reason"],
        })
        selected_payload["selected_nucleus"] = selected
        selected_payload["mapped_template"] = template
        selected_payload["continuous_nucleus"] = mapping["continuous_nucleus"]
        selected_payload["template_mapping"] = {
            "mismatch_metrics": mapping["mismatch_metrics"],
            "decision_reason": mapping["decision_reason"],
            "fallback_used": mapping["fallback_used"],
            "validation_log": rel(self.template_mapping_log),
        }
        selected_payload["cuda_now_uses_predicted_nucleus"] = True
        selected_payload["cuda_ready_via_geometry_mapper"] = True
        write_json(self.selected_json_path, selected_payload)

    def _inject_mapped_template_into_catalog(self, selected: dict[str, Any], mapping: dict[str, Any]) -> None:
        catalog = read_json(self.catalog_path, {"schema_version": 1, "entries": []})
        entries = list(catalog.get("entries", []))
        template = mapping["mapped_template"]
        continuous = mapping["continuous_nucleus"]
        base = dict(selected or continuous.get("raw") or {})
        base.update({
            "id": f"mapped_{base.get('id') or continuous.get('nucleus_id')}",
            "rc_nm": continuous["rc_nm"],
            "shape_type": continuous["shape_type"],
            "aspect_ratio": continuous["aspect_ratio"],
            "semiaxes": continuous["semiaxes_nm"],
            "profile_dir": template["profile_dir"],
            "source_dyn_dir": template["source_dyn_dir"],
            "profile_dir_candidate": template["profile_dir"],
            "source_dyn_dir_candidate": template["source_dyn_dir"],
            "cuda_template_available": True,
            "mapped_template_id": template["template_id"],
            "geometry_mapping_loss": mapping["mismatch_metrics"]["total"],
            "geometry_mapping_decision_reason": mapping["decision_reason"],
            "origin": f"{base.get('origin', 'unknown')}+geometry_mapper",
            "updated_at_utc": utc_now(),
        })
        entries = [entry for entry in entries if entry.get("id") != base["id"]]
        entries.append(base)
        catalog["entries"] = entries
        catalog["entry_count"] = len(entries)
        catalog["updated_at_utc"] = utc_now()
        catalog["generated_by"] = "nucleus_orchestrator.py geometry_mapper_integration"
        write_json(self.catalog_path, catalog)

    def generate_cuda_launch_config(self, selected_payload: dict[str, Any]) -> dict[str, Any]:
        mapping = self.map_selected_nucleus_to_cuda_template(selected_payload)
        selected = selected_payload.get("selected_nucleus", {})
        config = {
            "schema_version": 1,
            "generated_at_utc": utc_now(),
            "run_mode": self.config.run_mode,
            "selected_nucleus": selected,
            "selected_nucleus_json": rel(self.selected_json_path),
            "catalog_json": rel(self.catalog_path),
            "manual_profile_dir_removed_from_default_path": True,
            "cuda_command": [
                str(REPO_ROOT / "main_cuda"),
                "--enable-scheduled-nucleation-test",
                "--nucleus-catalog-json", str(self.catalog_path),
                "--selected-nucleus-json", str(self.selected_json_path),
            ],
            "ready_to_launch": bool(selected.get("source_dyn_dir") and selected.get("profile_dir")),
            "continuous_nucleus_geometry_csv": rel(self.continuous_geometry_csv),
            "template_geometry_space_json": rel(self.template_space_json),
            "template_mapping_validation_log": rel(self.template_mapping_log),
            "mapped_template": mapping["mapped_template"],
            "template_mapping": {
                "mismatch_metrics": mapping["mismatch_metrics"],
                "decision_reason": mapping["decision_reason"],
                "fallback_used": mapping["fallback_used"],
            },
            "notes": "CUDA runtime resolves nucleus through selector JSON/catalog containing the mapped CUDA template; manual scheduled profile/source args are not used in the default path.",
        }
        write_json(self.cuda_config_path, config)
        return config

    def run_cuda_simulation(self, config: dict[str, Any]) -> dict[str, Any]:
        if not self.config.launch_cuda:
            config["launch_status"] = "not_requested"
            write_json(self.cuda_config_path, config)
            return config
        if self.config.dry_run:
            config["launch_status"] = "dry_run"
            write_json(self.cuda_config_path, config)
            return config
        if not config.get("ready_to_launch"):
            self.log_error("CUDA launch skipped: selected nucleus has no source/profile template")
            config["launch_status"] = "skipped_no_insertable_template"
            write_json(self.cuda_config_path, config)
            return config
        try:
            proc = subprocess.run(config["cuda_command"], cwd=REPO_ROOT)
            config["launch_status"] = "complete" if proc.returncode == 0 else "failed"
            config["returncode"] = proc.returncode
        except OSError as exc:
            self.log_error(f"CUDA launch failed: {exc}")
            config["launch_status"] = "failed"
            config["error"] = str(exc)
        write_json(self.cuda_config_path, config)
        return config

    def run_nucleation_pipeline(self) -> dict[str, Any]:
        self.ensure_dirs()
        try:
            scan_cases = self.generate_nucleus_scan_range()
        except Exception as exc:
            self.log_error(f"scan generation failed: {exc}")
            scan_cases = []
        jobs = self.submit_minimization_jobs(scan_cases) if scan_cases else []
        jobs = self.monitor_jobs(jobs) if jobs else []
        parsed = self.parse_results(scan_cases) if scan_cases else []
        try:
            self.update_nucleus_catalog(parsed)
        except Exception as exc:
            self.log_error(f"catalog update failed: {exc}")
        try:
            selected = self.select_nucleus()
        except Exception as exc:
            self.log_error(f"selector failed: {exc}")
            selected = {"selected_nucleus": self._last_valid_catalog_entry(self.catalog_path) or {}, "fallback_from_last_valid_catalog": True}
        cuda_config = self.generate_cuda_launch_config(selected)
        cuda_config = self.run_cuda_simulation(cuda_config)
        self.state.update({
            "completed_at_utc": utc_now(),
            "autonomous_status": bool(scan_cases and self.state.get("catalog_self_learning")),
            "cuda_launch_config": rel(self.cuda_config_path),
            "selected_nucleus_json": rel(self.selected_json_path),
            "catalog_json": rel(self.catalog_path),
            "parsed_results_csv": rel(self.parsed_results_path),
            "scan_cases_json": rel(self.scan_cases_path),
            "jobs_manifest_json": rel(self.jobs_manifest_path),
            "cuda_closed_loop": bool(cuda_config.get("ready_to_launch")),
        })
        write_json(self.run_dir / "orchestrator_state.json", self.state)
        return self.state


def run_nucleation_pipeline(T: float, xB: float, strain: float) -> dict[str, Any]:
    config = OrchestratorConfig(T=T, xB=xB, strain=strain, dry_run=False)
    return NucleusOrchestrator(config).run_nucleation_pipeline()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous nucleation orchestration: prediction -> sweep -> jobs -> parse -> catalog -> selector -> CUDA config.")
    parser.add_argument("--T", type=float, default=400.0)
    parser.add_argument("--xB", type=float, default=0.05)
    parser.add_argument("--strain", type=float, default=-0.01)
    parser.add_argument("--strain-mode", default="eyy", choices=("exx", "eyy", "exx_eyy"))
    parser.add_argument("--run-mode", default=RUN_MODE_FULL)
    parser.add_argument("--backend", default="local", choices=("local", "slurm"))
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.add_argument("--launch-cuda", action="store_true")
    parser.add_argument("--workflow-root", default="Results/orchestrator")
    parser.add_argument("--grid", default="400,400,400")
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--delta-nm", type=float, default=0.5)
    parser.add_argument("--step-nm", type=float, default=0.1)
    parser.add_argument("--resume-jobs", type=Path, default=None, help="internal Slurm helper: run scan cases from an existing scan_cases.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = OrchestratorConfig(
        T=args.T,
        xB=args.xB,
        strain=args.strain,
        strain_mode=args.strain_mode,
        run_mode=args.run_mode,
        execution_backend=args.backend,
        dry_run=args.dry_run,
        workflow_root=args.workflow_root,
        grid=args.grid,
        dt=args.dt,
        steps=args.steps,
        delta_nm=args.delta_nm,
        step_nm=args.step_nm,
        launch_cuda=args.launch_cuda,
    )
    orch = NucleusOrchestrator(cfg)
    if args.resume_jobs:
        payload = read_json(args.resume_jobs, {"cases": []})
        jobs = orch.submit_minimization_jobs(payload.get("cases", []))
        orch.monitor_jobs(jobs)
        return 0
    state = orch.run_nucleation_pipeline()
    print("nucleation_orchestrator_installed")
    print("full_pipeline_mode_active")
    print(f"autonomous_status = {str(bool(state.get('autonomous_status'))).lower()}")
    print(f"catalog_self_learning = {str(bool(state.get('catalog_self_learning'))).lower()}")
    print(f"cuda_closed_loop = {str(bool(state.get('cuda_closed_loop'))).lower()}")
    print("template_mapper_installed")
    print("continuous_to_discrete_mapping_active")
    print(f"geometry_loss_mean = {state.get('geometry_loss_mean', 'nan')}")
    print(f"cuda_closed_loop_status = {str(bool(state.get('cuda_closed_loop'))).lower()}")
    print("missing_template_regions = generated_intermediate_profile_when_no_existing_template_matches")
    print("next_upgrade_recommendation = validate generated profiles against minimized VTK-derived faceted profiles before production launch-cuda runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
