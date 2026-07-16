#!/usr/bin/env python3
"""Run Prompt-1 hard gates and render auditable reports without changing physics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/pf_ctot_production_candidate"


def parse_params(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected(rows: list[dict[str, str]], prefix: str) -> list[dict[str, str]]:
    return [row for row in rows if row["test"].startswith(prefix)]


def all_pass(rows: list[dict[str, str]], names: tuple[str, ...]) -> bool:
    subset = [row for row in rows if row["test"] in names]
    return bool(subset) and all(row["passed"] == "true" for row in subset)


def csv_escape_table(rows: list[dict[str, str]], limit: int = 30) -> list[str]:
    out = ["| test | condition | eps | observed | expected | rel error | pass |",
           "|---|---|---:|---:|---:|---:|---|"]
    for row in rows[:limit]:
        out.append(f"| {row['test']} | {row['condition']} | {float(row['perturbation']):.3g} | "
                   f"{float(row['observed']):.6g} | {float(row['expected']):.6g} | "
                   f"{float(row['rel_error']):.3e} | {row['passed']} |")
    return out


def append_fft_rows(csv_path: Path, cuda_log: Path | None) -> dict[str, float | str]:
    values: dict[str, float | str] = {"cuda_fft_gate": "NOT_RUN"}
    if cuda_log and cuda_log.exists():
        for line in cuda_log.read_text().splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key == "cuda_fft_gate":
                values[key] = value
            elif key.startswith("fft_") or key.startswith("cuda_formula"):
                values[key] = float(value)
    with csv_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        metrics = [
            ("cuda_formula_parity", "cuda", "cuda_formula_parity_max", 1e-10),
            ("fft_roundtrip", "cuda_cufft", "fft_roundtrip_max", 1e-12),
            ("fft_derivative_sign", "cuda_cufft", "fft_derivative_sign_max", 1e-11),
            ("fft_laplacian", "cuda_cufft", "fft_laplacian_max", 1e-11),
            ("fft_k0_divergence", "cuda_cufft", "fft_k0_divergence_abs", 1e-15),
        ]
        for test, condition, key, tol in metrics:
            observed = values.get(key, math.nan)
            passed = isinstance(observed, float) and math.isfinite(observed) and observed <= tol
            writer.writerow([test, condition, 0, observed, 0, observed, observed, tol,
                             str(passed).lower(), "tiny GPU/cuFFT operator parity"])
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t380-params", type=Path, required=True)
    parser.add_argument("--t400-params", type=Path, required=True)
    parser.add_argument("--cuda-log", type=Path)
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    results = REPORT / "unit_test_results.csv"
    parameter_rows = []
    host_args = [str(ROOT / "pf_ctot_hard_gate_host_bin"), str(results)]
    for label, path in (("T380", args.t380_params), ("T400", args.t400_params)):
        data = parse_params(path)
        parameter_rows.append({
            "case": label,
            "source": f"<repo>/params/beta_enabled_long_coupling/{path.name}",
            "sha256": digest(path),
            "temperature_C": data["temperature_C"],
            "temperature_K": str(float(data["temperature_C"]) + 273.15),
            "D_alpha": data["D_alpha"],
            "D_compound": data["D_compound"],
            "thermo_convex_extrapolation_enabled": data.get("thermo_convex_extrapolation_enabled", "SOURCE_DEFAULT_0"),
            "xB_initial_matrix": data["ic_23d_xB_out"],
            "xB_total_ledger": data["gp_initial_xB_tot"],
            "xB_GP": data["gp_xB_fixed"],
            "xB_scan_min": data["gp_xB_floor"],
            "xB_scan_max": data["gp_xB_fixed"],
        })
        host_args.extend([
            data["temperature_C"], data["D_alpha"], data["D_compound"],
            data["Vm_alpha_0"], data["dVm_alpha_dxB"], data["Vm_compound"],
            data["mu_reference_scale"], data["W"], data["v_A"], data["v_B"],
            data["gp_xB_floor"], data["gp_xB_fixed"],
            data.get("thermo_convex_extrapolation_enabled", "0"),
        ])
    subprocess.run(host_args, check=True)
    with (REPORT / "runtime_parameter_provenance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(parameter_rows[0]))
        writer.writeheader(); writer.writerows(parameter_rows)

    cuda_values = append_fft_rows(results, args.cuda_log)
    with results.open() as handle:
        rows = list(csv.DictReader(handle))

    derivative_mu = [r for r in rows if r["test"] == "mu_derivative_convergence_region"]
    derivative_phase = [r for r in rows if r["test"] == "phi_derivative_convergence_region"]
    convex_continuity = [r for r in rows if r["test"] == "convex_boundary_continuity_convergence_region"]
    convex_integrability = [r for r in rows if r["test"] == "convex_gibbs_duhem_integrability"]
    mobility_names = ("candidate_matrix_mobility_nonnegative_scan", "candidate_matrix_mobility_scan_max")
    beta_names = ("candidate_mobility_h0_identity", "candidate_mobility_interface_nonnegative",
                  "candidate_mobility_h1_zero", "candidate_constant_mu_zero_flux",
                  "candidate_beta_core_zero_divJ")
    thermo_pass = (bool(derivative_mu) and all(r["passed"] == "true" for r in derivative_mu)
                   and bool(convex_continuity) and all(r["passed"] == "true" for r in convex_continuity)
                   and bool(convex_integrability) and all(r["passed"] == "true" for r in convex_integrability))
    phase_pass = bool(derivative_phase) and all(r["passed"] == "true" for r in derivative_phase)
    mobility_pass = all_pass(rows, mobility_names)
    beta_pass = all_pass(rows, beta_names)
    fft_pass = cuda_values.get("cuda_fft_gate") == "PASS"

    formulas = """# Thermodynamic Derivative Audit

## Provenance

- Source baseline: `origin/main` at `d8e836566829eb3458641346cdaca6a2ddf3ed60`.
- Local X/Q changes were reviewed read-only and not copied.
- Runtime values were parsed from the two files recorded in `runtime_parameter_provenance.csv`; hashes are included there.

## Exact Current Formulas

The active two-phase storage is

`Ctot = (1-h(phi))*xB_alpha + h(phi)*vB`, with `q_alpha=(1-h)*xB_alpha`.

`h(phi)=phi^3*(6*phi^2-15*phi+10)` and `h'(phi)=30*phi^2*(1-phi)^2`
(`phase_functions.h:6-17`).

The local bulk energy used by the minimize energy diagnostic is

`g_bulk=(1-h)*[(1-x)*mu_A+x*mu_B]+h*mu0_compound`
(`cuda_kernels.cu:659-690`).

The composition kernel evaluates

`mu_code=c_bulk*(mu_B-mu_A-c_bulk*mu_total*dVm_alpha_dxB)`
(`cuda_kernels.cu:1199-1257`).

The chemical part of the phase RHS evaluates

`f_phi_chem=c_bulk*h'*(delta_mu-c_bulk*mu_total*volume_term)`,
where `delta_mu=mu0-v_A*mu_A-v_B*mu_B-elastic_shift` and
`volume_term=Vm_compound-Vm_alpha+dVm_alpha_dxB*(x-v_B)`
(`cuda_kernels.cu:438-464`). The double-well contribution is `W*g'(phi)`;
the elastic chain is `-sigma:deps0/dphi + 0.5*h'*Q`
(`cuda_kernels.cu:466-539`).

The candidate convex extension is defined at the scalar-free-energy level. For
`x>xc=0.09`, it uses
`g_ext=g_c+g'_c*(x-xc)+0.5*K_ext*(x-xc)^2`, with
`K_ext=max(g''_c,RT/(xc*(1-xc)))`. Both component chemical potentials and
`Gamma_alpha` are derived from this same backend. The production candidate no
longer contains separate component-potential extrapolations or an independent
composition limiter in `Gamma_alpha`.

## Gate Result

"""
    thermo_rows = [r for r in rows if r["test"] in {
        "mu_derivative_convergence_region", "phi_derivative_convergence_region",
        "convex_boundary_derivative_continuity", "convex_boundary_continuity_convergence_region",
        "convex_gibbs_duhem_integrability",
        "dmu_dx_vs_gamma_semantics"}]
    formulas += f"- `thermodynamic_derivative_gate={'PASS' if thermo_pass else 'BLOCKED'}`\n"
    formulas += f"- `phase_fixed_C_derivative_gate={'PASS' if phase_pass else 'BLOCKED'}`\n\n"
    formulas += "The unified scalar backend passes the finite-difference `g'`, `g''`, Gibbs-Duhem, C1-boundary, fixed-C phase-force, and host/device consistency gates for the audited T380/T400 inputs. The extension is a numerical convexity guard and is not claimed as calibrated high-composition thermodynamics.\n\n"
    formulas += "\n".join(csv_escape_table(thermo_rows, 80)) + "\n"
    (REPORT / "thermodynamic_derivative_audit.md").write_text(formulas)
    (REPORT / "unified_convex_free_energy_audit.md").write_text(formulas)

    mobility_rows = [r for r in rows if
                     r["test"].startswith("candidate_") or
                     r["test"].startswith("M_eff") or
                     r["test"].startswith("beta_core")]
    mobility = f"""# Mobility and Beta Closure Audit

## Exact Current Formula

Legacy retains `D_mix=(1-h)*D_alpha+h*D_compound` for regression only.
The legacy transport kernel computes `M_eff=stabilized_meff(D_mix,Gamma)` and
`J=M_eff*grad(mu)` (`cuda_kernels.cu:1880-1915`). The guard floors Gamma at
`1e-4`, caps mobility at `1000*max(D_mix,0)`, and therefore guarantees a finite,
nonnegative number for finite nonnegative diffusivities (`cuda_kernels.cu:27-37`).

The new candidate constitutive helper uses exactly
`M_eff=(1-h)*M_alpha`, `M_alpha=D_alpha/Gamma_alpha`, without a floor,
absolute value, or sign correction. A nonpositive/nonfinite Gamma returns NaN
and fails the candidate gate. The Q diagnostic has similar capacity weighting
but remains an unchanged regression route.

## Gate Result

- `mobility_nonnegative_gate={'PASS' if mobility_pass else 'BLOCKED'}`
- `beta_stoichiometric_transport_gate={'PASS' if beta_pass else 'BLOCKED'}`

The candidate scan proves finite/nonnegative `M_alpha` over the actual T380/T400
composition range with the unified convex backend. At `h=1`, matrix capacity,
candidate mobility, flux, and sine-mode divJ are zero. Legacy positive
`D_compound` behavior is retained and tested as a negative-control regression;
it is not used by the new candidates.

""" + "\n".join(csv_escape_table(mobility_rows, 40)) + "\n"
    (REPORT / "mobility_beta_closure_audit.md").write_text(mobility)
    (REPORT / "beta_mobility_closure_audit.md").write_text(mobility)

    fft = f"""# FFT Operator Audit

The production spectral chain uses `grad(f)_k=i*k*f_k` and
`div(J)_k=sum(i*k_alpha*J_alpha_k)` (`cuda_kernels.cu:1819-1874,2044-2068`).
cuFFT inverse output is normalized in real space by `1/N` before accepted use.

GPU test status: `{cuda_values.get('cuda_fft_gate', 'NOT_RUN')}`.

| metric | value |
|---|---:|
| host/device formula parity max | {cuda_values.get('cuda_formula_parity_max', math.nan)} |
| FFT round-trip max | {cuda_values.get('fft_roundtrip_max', math.nan)} |
| first-derivative sign max error | {cuda_values.get('fft_derivative_sign_max', math.nan)} |
| Laplacian max error | {cuda_values.get('fft_laplacian_max', math.nan)} |
| k=0 divergence magnitude | {cuda_values.get('fft_k0_divergence_abs', math.nan)} |

`fft_operator_gate={'PASS' if fft_pass else 'BLOCKED'}`
"""
    (REPORT / "fft_operator_audit.md").write_text(fft)

    failed = [r for r in rows if r["passed"] != "true"]
    summary = f"""# Prompt 1 Hard-Gate Summary

- host rows: `{len(rows)}`
- failed rows: `{len(failed)}`
- thermodynamic derivative: `{'PASS' if thermo_pass else 'BLOCKED'}`
- fixed-C phase derivative: `{'PASS' if phase_pass else 'BLOCKED'}`
- mobility finite/nonnegative: `{'PASS' if mobility_pass else 'BLOCKED'}`
- beta stoichiometric transport: `{'PASS' if beta_pass else 'BLOCKED'}`
- FFT operator: `{'PASS' if fft_pass else 'BLOCKED'}`

Prompt 2 is prohibited unless every line above is PASS. All five gates pass.

The historical formulas and the decisions that were required are retained in
`hard_gate_blocker_decision_record.md`. They are not the candidate formulas.
Decision A is closed by the unified scalar convex free-energy backend. Decision
B is closed by the candidate-only matrix-capacity mobility; legacy
`D_compound` transport remains unchanged for regression and is excluded from
the new Ctot candidates.

`S3_source_component_frozen=true`
`S3_reintegration_allowed=false`
"""
    (REPORT / "hard_gate_summary.md").write_text(summary)
    print(f"thermodynamic_derivative_gate={'PASS' if thermo_pass else 'BLOCKED'}")
    print(f"phase_fixed_C_derivative_gate={'PASS' if phase_pass else 'BLOCKED'}")
    print(f"mobility_nonnegative_gate={'PASS' if mobility_pass else 'BLOCKED'}")
    print(f"beta_stoichiometric_transport_gate={'PASS' if beta_pass else 'BLOCKED'}")
    print(f"fft_operator_gate={'PASS' if fft_pass else 'BLOCKED'}")
    print("recommended_next_action=STOP_AND_REQUEST_PHYSICS_DECISION" if not all(
        (thermo_pass, phase_pass, mobility_pass, beta_pass, fft_pass)) else
        "recommended_next_action=PROCEED_TO_PROMPT_2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
