#!/usr/bin/env python3
"""Prepare the reproducible Next7 independent-interface-data gate reports."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "pf_ctot_production_candidate"
STATUS = "WAITING_FOR_INDEPENDENT_INTERFACE_MOBILITY_DATA"

REQUIRED_DATA_FIELDS = {
    "record_status",
    "temperature_K",
    "phase_driving",
    "diffusion_driving",
    "measured_Vn",
    "measured_JB",
    "uncertainty_Vn",
    "uncertainty_JB",
    "covariance",
    "interface_orientation",
    "coherency_state",
    "elastic_constraint",
    "source_type",
    "source_reference",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(name: str, text: str) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(name: str, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def freeze_baseline() -> None:
    requested = [
        ("next6_mobility_closure_acceptance.md", "exact"),
        ("next6_constraint_rank_and_isolation.md", "exact"),
        ("next6_higher_order_independence.md", "exact"),
        ("next6_corrected_first_curvature_transport.md", "requested_alias"),
        ("next6_transport_equation_reaudit.md", "actual_for_requested_transport_alias"),
        ("next6_canonical_gauge_decision.md", "exact"),
        ("next6_family_uncertainty_and_production_gate.md", "requested_alias"),
        ("next6_family_uncertainty_gate.md", "actual_for_requested_family_alias"),
        ("prompt8_phase_kkt_solver_implementation.md", "exact"),
        ("perf_p1_implementation.md", "exact"),
        ("next2_p2_correctness.md", "exact"),
        ("next_stationary_equilibrium_results.md", "exact"),
        ("next2_curved_velocity_matrix.md", "validation_only_forbidden_as_fit_data"),
        ("final_terminal_output.txt", "prior_stage_terminal_frozen_before_next7_overwrite"),
    ]
    rows = []
    for name, role in requested:
        path = REPORT / name
        rows.append(
            {
                "requested_asset": name,
                "role": role,
                "present": str(path.exists()).lower(),
                "sha256": sha256(path) if path.exists() else "MISSING",
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    write_csv(
        "next7_stage0_frozen_assets.csv",
        ["requested_asset", "role", "present", "sha256", "bytes"],
        rows,
    )
    missing = [row["requested_asset"] for row in rows if row["present"] == "false"]
    write_text(
        "next7_baseline_freeze.md",
        f"""# Next7 Baseline Freeze

## Candidate

- worktree: `CUDA_STO_PF-pf-ctot-production-candidate`
- branch: `codex/pf-ctot-production-candidate`
- frozen HEAD: `{git_head()}`
- fixed-Ctot storage, one-sided `D_beta=0`, mimetic transport, semismooth PDAS,
  mechanics, accepted/trial transaction, rollback/restart, and P1/P2 remain frozen.
- GP/RSMD coupled source, GP release/growth/coarsening, direct beta inventory,
  and S3 remain disabled and untouched.

## Asset resolution

Hashes are recorded in `next7_stage0_frozen_assets.csv`. Requested names that
do not exist are aliases, not silently fabricated files:

{chr(10).join(f'- `{name}`' for name in missing) if missing else '- none'}

The corrected transport audit is stored as `next6_transport_equation_reaudit.md`.
The family gate is stored as `next6_family_uncertainty_gate.md`.

## Frozen decisions

- q-only interpolation closure: rejected.
- quartic corrected first-curvature zero set: empty.
- quintic first-order family: rank 1 / nullity 1.
- cylinder/sphere higher-order constraints: independent and incompatible.
- no canonical gauge is authorized.
- PF curved velocity data remain validation-only and forbidden for fitting.
- P1 exact replay speedup: `243.945x` for the cited 152-step profile.
- P2 accepted-step speedup at `64^3`: `4.4999161761x` elastic OFF and
  `3.4452862375x` elastic ON, with frozen bitwise correctness evidence.

No core source or runtime parameter was changed by Next7 data-gate work.
""",
    )


def accuracy_policy() -> None:
    write_text(
        "next7_production_accuracy_policy.md",
        """# Next7 Practical Production Accuracy Policy

## Non-negotiable acceptance predicates

| Quantity | Acceptance |
|---|---|
| source-free/global mass | relative error `<=1e-10` |
| local fixed-Ctot storage | roundoff residual |
| clipping | zero |
| physical/domain-wide projection | zero |
| phase bounds and KKT | pass at every accepted stage |
| base and full Onsager dissipation | nonnegative within audited roundoff |
| growth/shrink direction | agrees with independent sharp response |
| time and grid error | separately converged |

The preferred held-out velocity/radius-trajectory error is `<=2%`. An
engineering limit of `<=5%` is allowed only when it is established on held-out
data, every larger validated `R/lambda` remains within the limit, cylindrical
and spherical conclusions agree, time/grid error is substantially below 5%,
diffuse-basis uncertainty is `<=1%`, and a runtime guard blocks extrapolation.

## Error budget

| Component | Budget / rule |
|---|---|
| conservation/storage | hard predicates above; never traded against model error |
| nonlinear/KKT solve | below time/grid error and acceptance tolerances; no failed-cell suppression |
| time discretization | demonstrated by `dt` versus `dt/2`; substantially below 5% |
| spatial/interface resolution | demonstrated by `lambda/dx` refinement; substantially below 5% |
| dividing-surface mapping | `<=1%` after mapping to total-C equimolar/h-volume surface |
| diffuse basis/gauge | `<=1%` held-out spread under basis/refinement matrix |
| interface model | remainder of 2% preferred / 5% engineering budget |
| independent-data uncertainty | propagated through fitted Onsager covariance; inside applicability guard |

`O(lambda)` differences among internal diffuse surfaces are admissible only
after transformation to the accepted total-C equimolar/h-volume surface and a
`<=1%` residual mapping uncertainty. Basis shape is numerical gauge only after
it reproduces an independently identified matrix and held-out observables are
basis-insensitive.

Wrong direction, unbounded/nonmonotone 10%-500% curved errors, mass/energy
discrepancy, or selecting coefficients because beta grows are unconditional
failures. Numerical solver improvements and finite-interface physics cannot
compensate one another.
""",
    )


def interface_rank() -> None:
    vj_states = ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
    full_rows = []
    fixed_rows = []
    diagonal_rows = []
    for v, j in vj_states:
        full_rows.extend(([v, j, 0.0], [0.0, v, j]))
        fixed_rows.extend(([j, 0.0], [v, j]))
        diagonal_rows.append([j])
    models = [
        ("full_symmetric", np.asarray(full_rows), "A_I;B_I;C_I", "none"),
        ("selected_Lphi_fixes_A", np.asarray(fixed_rows), "B_I;C_I", "A_I"),
        ("independent_B_zero", np.asarray(diagonal_rows), "C_I", "A_I;B_I=0"),
    ]
    rows = []
    for name, matrix, unknown, fixed in models:
        singular = np.linalg.svd(matrix, compute_uv=False)
        rank = int(np.linalg.matrix_rank(matrix))
        rows.append(
            {
                "model": name,
                "unknown_coefficients": unknown,
                "fixed_coefficients": fixed,
                "response_rows": matrix.shape[0],
                "parameter_columns": matrix.shape[1],
                "rank": rank,
                "nullity": matrix.shape[1] - rank,
                "singular_values": ";".join(f"{value:.17g}" for value in singular),
                "required_independent_data_dimension": matrix.shape[1],
                "status": "IDENTIFIABLE_WITH_NONCOLLINEAR_INDEPENDENT_DATA",
            }
        )
    rows.append(
        {
            "model": "current_project_information",
            "unknown_coefficients": "B_I;C_I",
            "fixed_coefficients": "A_I_from_selected_Lphi_profile_combination",
            "response_rows": 0,
            "parameter_columns": 2,
            "rank": 0,
            "nullity": 2,
            "singular_values": "",
            "required_independent_data_dimension": 2,
            "status": "NO_ELIGIBLE_INDEPENDENT_RESPONSE_ROWS_PRESENT",
        }
    )
    write_csv(
        "next7_interface_rank.csv",
        [
            "model",
            "unknown_coefficients",
            "fixed_coefficients",
            "response_rows",
            "parameter_columns",
            "rank",
            "nullity",
            "singular_values",
            "required_independent_data_dimension",
            "status",
        ],
        rows,
    )
    write_text(
        "next7_interface_identifiability.md",
        r"""# Next7 Interface Onsager Identifiability

## Accepted surface, signs, and units

The normal points from beta to matrix. `V_n>0` means beta grows. The physical
matrix flux is `j_alpha=-M_alpha grad(mu_B)` and the interface transfer flux is
defined `J_B=-j_alpha dot n`, positive from matrix into beta. With zero beta
flux, the one-sided Stefan convention is

```text
J_B = V_n (v_B-x_B_alpha),  v_B=1.
```

All quantities are referred to the accepted total-C equimolar/h-volume
dividing surface:

```text
X_phase = (1/T) integral[(delta F/delta phi) partial_n(phi)] dn
          [J m^-3 K^-1]
X_diff  = (mu_B_alpha-mu_B_beta)/T
          [J mol_Beq^-1 K^-1]
V_n     [m s^-1]
J_B     [mol_Beq m^-2 s^-1]
```

The interfacial entropy production per area is
`sigma_I=X_phase*V_n+X_diff*J_B>=0`, and

```text
[X_phase]   [A_I B_I] [V_n]
[X_diff ] = [B_I C_I] [J_B].
```

Thus `A_I` has units `J s m^-4 K^-1`, `B_I` has units
`J s mol_Beq^-1 m^-1 K^-1`, and `C_I` has units
`J s m^2 mol_Beq^-2 K^-1`. Strict SPD requires
`A_I>0`, `C_I>0`, and `B_I^2<A_I*C_I`.

## Combinations already fixed

For `phi_0=0.5[1-tanh(2n/lambda)]`,

```text
integral (partial_n phi_0)^2 dn = 2/(3 lambda),
A_PF = 2/(3*T*L_phi*lambda).
```

Therefore the selected `L_phi`, equilibrium profile, `lambda`, and temperature
mathematically fix the intrinsic phase diagonal combination `A_PF`; it is not
a new fit coefficient. The accepted code finite-Lphi oracle stores the same
combination as `2/(3*L_phi*lambda_code)=1.2994628530878145e-2` for the frozen
T400 setup.

This statement is not a claim that `A_PF` has independent physical validation.
The repository explicitly records the analytic fixed-Ctot one-sided mapping as
incomplete. T400 currently uses selected `L_phi_code=85.50541544691814` and
`L_phi_phys=2.7499037546167224e-8` with finite-interface-corrected provenance.
The base T380/T400 formula rows are likewise labelled
`FORMULA_EVALUATED_BENCHMARK_NOT_DIFFUSION_CONTROLLED`. The fitter therefore
requires an explicit, provenance-bearing `A_I(T)` input and independent
phase-driving rows must validate it; the tool contains no A default.

`D_alpha`, bulk thermodynamics, and the molar-volume convention
`c_tot=1/Vm_alpha_0` fix bulk `x_B <-> mu_B` conversion and matrix transport
resistance. `D_beta=0` fixes the beta-side flux to zero. They do not determine
an additional interface-local transfer resistance `C_I`, and no accepted
identity determines reciprocal cross coupling `B_I`.

## Rank result

For each observation `(V,J)`, the full symmetric design is

```text
[X_phase] = [V J 0] [A B C]^T
[X_diff ]   [0 V J]
```

and has rank 3 under non-collinear excitation. After freezing `A_I`,

```text
X_phase-A_I*V = [J 0] [B C]^T
X_diff         = [V J] [B C]^T,
```

which has rank 2 for the required phase, diffusion, and mixed states. If and
only if independent data resolve `B_I=0` within uncertainty, the remaining
diagonal model has rank 1 and only `C_I` is fitted. Numerical singular values
for an exact non-collinear design are in `next7_interface_rank.csv`.

The reported rank is structural, not assigned from requested force labels.
The fitter rebuilds the Jacobian from the measured `(V_n,J_B)` rows and stops
if its numerical rank is below the selected model dimension. The one-sided
Stefan relation is enforced as a consistency check, not substituted to create
synthetic response rows or to inflate rank.

```text
required_independent_data_dimension=2
fixed_combination=A_I_from_selected_Lphi_equilibrium_profile
missing_combinations=B_I_and_C_I
cross_coupling_required=UNRESOLVED_PENDING_INDEPENDENT_DATA
target_variables_and_forces_well_defined=true
```
""",
    )


def audit_independent_data() -> bool:
    candidates = []
    for path in ROOT.rglob("*.csv"):
        if any(part in {".git", "next4_gold_reference"} for part in path.parts):
            continue
        try:
            with path.open(newline="", encoding="utf-8", errors="replace") as handle:
                reader = csv.DictReader(handle)
                header = set(reader.fieldnames or [])
                if not REQUIRED_DATA_FIELDS.issubset(header):
                    continue
                data_rows = 0
                template_rows = 0
                source_references = set()
                for row in reader:
                    if row.get("record_status", "").strip() == "DATA":
                        data_rows += 1
                        source_references.add(row.get("source_reference", "").strip())
                    elif row.get("record_status", "").strip() == "TEMPLATE":
                        template_rows += 1
                candidates.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "data_rows": data_rows,
                        "template_rows": template_rows,
                        "source_references": ";".join(sorted(source_references)),
                        "eligible": str(data_rows > 0).lower(),
                    }
                )
        except (OSError, csv.Error):
            continue
    write_csv(
        "next7_independent_data_audit.csv",
        ["path", "data_rows", "template_rows", "source_references", "eligible"],
        candidates,
    )
    present = any(row["data_rows"] > 0 for row in candidates)
    write_text(
        "next7_interface_data_plan.md",
        f"""# Next7 Independent Interface Data Plan

## Repository audit

Exact-schema CSV candidates found: `{len(candidates)}`.
Eligible independent `DATA` rows found: `{sum(row['data_rows'] for row in candidates)}`.
The complete audit is `next7_independent_data_audit.csv`.

```text
independent_interface_data_present={str(present).lower()}
interface_matrix_fit_status={'READY' if present else 'WAITING_FOR_DATA'}
```

Existing PF planar/curved benchmarks, seed fate, and solver timings are not
eligible independent measurements. In particular,
`next2_curved_velocity_matrix.md` remains held-out validation evidence and is
forbidden as fit input.

## Data contract

- Schema: `schemas/interface_onsager_data.schema.json`
- T380/T400 acquisition template:
  `examples/interface_onsager_data_template.csv`
- SPD fitter: `scripts/fit_interface_onsager_matrix.py`

The template contains 24 explicitly marked `TEMPLATE` rows: six non-collinear
small-driving states, each with two replicates, at 653.15 K and 673.15 K. Blank
measurement fields and template status make it impossible to fit accidentally.
Production rows must use `record_status=DATA`, positive uncertainties, a
positive-definite `(V_n,J_B)` covariance, exact sign/unit conventions,
orientation/coherency/elastic metadata, and a traceable independent source.
The requested force states alone do not establish identifiability: the fitter
must observe rank 2 in the fixed-`A_I` response Jacobian or stop with
`BLOCKED_INTERFACE_DATA_NOT_IDENTIFIABLE`.

## Minimum acquisition matrix per temperature

```text
(+phase,0), (-phase,0)
(0,+diffusion), (0,-diffusion)
two mixed, non-collinear states
at least two independent replicates per state
```

For each state, verify the linear-response window by force-amplitude halving,
measure both `V_n` and `J_B` on the total-C equimolar/h-volume surface, and
retain raw trajectories so covariance is not inferred from PF residuals.
Eligible sources are standalone planar RSMD when physically capable,
standalone MD/kMC, or independent planar migration/solute-transfer experiments.

## Low-cost campaign boundary

No repository-backed standalone planar independent engine or authorized
dataset is present, so exact executable commands and wall-time estimates would
be fabricated. If the user supplies or authorizes such a backend, prepare a
two-temperature 24-record pilot first, benchmark one replicate, and only then
estimate resources. No high-cost acquisition is authorized or launched here.

## Fit command after real data and explicit A provenance exist

```text
python3 scripts/fit_interface_onsager_matrix.py DATA.csv \\
  --model fixed_A_spd --fixed-a-json FIXED_A_BY_T.json \\
  --output-json FIT.json --output-csv FIT.csv
```

The Cholesky/Schur parameterization guarantees SPD and emits coefficient
covariance. A diagonal `B_I=0` model is selectable only after independent data
show that the cross coefficient is unresolved from zero; the fitter never
invents a default.

Final data-gate status: `{STATUS}`.
""",
    )
    return present


def design_only_reports() -> None:
    write_text(
        "next7_minimal_onsager_design.md",
        r"""# Next7 Minimal Reciprocal Closure Design

**Status: DESIGN_ONLY_DATA_GATE; no runtime implementation.**

Retain authoritative fixed `Ctot`, the mimetic face gradient `G`, divergence
`D=-G*`, positive face resistance `R`, and PDAS phase transaction. A future
accepted closure introduces only an interface-local cell-to-face operator `B`
and its exact adjoint:

```text
-G_phi = T*phi_dot + B_star*J
-G*mu  = B*phi_dot + R*J
Ctot_dot + D*J = 0
```

Eliminating the face flux locally gives

```text
J = -R^-1 (G*mu + B*phi_dot)
(T-B_star*R^-1*B) phi_dot
  = -G_phi + B_star*R^-1*G*mu.
```

This adds no global face unknown. If independent data establish `B_I=0`, only
an interface resistance amplitude is needed. If `B_I` is resolved, one
reciprocal `B/B_star` amplitude plus the missing resistance is required. The
full three-coefficient matrix is forbidden because `A_I` is already fixed by
the selected phase kinetic combination and must be validation-only.

The canonical basis must vanish in both bulks, be smooth under current `h`, be
orientation covariant, preserve `D_beta=0`, and have analytic planar-profile
normalization. No basis is selected before target `B_I,C_I` exist; selecting it
now would create an unvalidated physical default.
""",
    )
    write_text(
        "next7_discrete_reciprocity_and_spd.md",
        r"""# Next7 Discrete Reciprocity and SPD Contract

**Status: DESIGN_ONLY_DATA_GATE.**

Use the actual cell and shared-face inner products. `D=-G*` and the new
operators must satisfy `<J,B phi_dot>_face=<B_star J,phi_dot>_cell` to roundoff
using identical face weights, normals, metric factors, and interface masks.
The local kinetic block

```text
K = [[T, B_star], [B, R]]
```

is SPD iff `R` is positive and the Schur complement
`T-B_star R^-1 B` is positive on every active phase subspace. Then

```text
D_kin = <phi_dot,T phi_dot>
      + 2 <B phi_dot,J>
      + <J,R J> >= 0.
```

Required future diagnostics are phase diagonal, transport diagonal, cross
term, total quadratic form, adjoint defect, and minimum local/Schur SPD margin.
No numerical tolerance may repair a non-SPD fitted matrix.
""",
    )
    write_text(
        "next7_runtime_cost_model.md",
        """# Next7 Runtime Cost Model

**Status: DESIGN_ONLY; no timing claim.**

The algebraically eliminated design requires one interface-face localization /
cross-driving evaluation and one reciprocal cell accumulation. They should be
fused into the existing mimetic face-flux and phase-residual paths, with no
additional FFT per residual/Jv and no global face unknown. Persistent metadata
is scalar; any interface scratch must reuse current workspaces where semantics
permit.

Frozen comparison points are P2 `64^3` accepted-step times `0.101403 s`
(elastic OFF) and `0.358758 s` (elastic ON). Acceptance targets remain `<=25%`
median overhead OFF, `<=15%` ON when mechanics dominates, `<=5%` memory,
zero extra FFTs, and at most one additional median outer iteration. These are
future measured gates, not estimates and not reasons to simplify physics.
""",
    )


def blocked_reports() -> None:
    reports = {
        "next7_host_model.md": "Host inverse realization was not implemented because no independently fitted target matrix exists.",
        "next7_host_validation.md": "Host validation was not run; the independent-interface-data gate is open.",
        "next7_cuda_implementation.md": "CUDA mode `fixed_ctot_onsager_v1` was not implemented; host acceptance is a prerequisite.",
        "next7_cuda_correctness.md": "CUDA correctness, reciprocity, transaction, restart, sanitizer, and legacy-bitwise gates were not run.",
        "next7_heldout_planar_validation.md": "Held-out planar validation was not run because no model was fitted.",
        "next7_curved_validation.md": "Curved validation was not run and frozen PF curved results were not used for fitting.",
        "next7_applicability_guard.md": "No runtime applicability interval can be defined before independent fit and held-out validation.",
        "next7_time_integrator_plan.md": "Adaptive BE/BDF2 work is gated behind quantitative interface-physics acceptance and was not started.",
    }
    for name, explanation in reports.items():
        title = name.removesuffix(".md").replace("next7_", "").replace("_", " ").title()
        write_text(
            name,
            f"# Next7 {title}\n\n**Status: NOT_RUN_DATA_GATE.**\n\n{explanation}\n\n`blocking_status={STATUS}`",
        )
    write_csv(
        "next7_cuda_performance.csv",
        [
            "grid",
            "elastic",
            "P2_reference_step_time_s",
            "onsager_step_time_s",
            "runtime_overhead_ratio",
            "memory_overhead_ratio",
            "extra_FFT_count",
            "status",
        ],
        [
            {
                "grid": "64^3",
                "elastic": "OFF",
                "P2_reference_step_time_s": "0.101403",
                "onsager_step_time_s": "NOT_RUN",
                "runtime_overhead_ratio": "NOT_EVALUATED",
                "memory_overhead_ratio": "NOT_EVALUATED",
                "extra_FFT_count": "NOT_IMPLEMENTED",
                "status": "BLOCKED_INDEPENDENT_DATA_GATE",
            },
            {
                "grid": "64^3",
                "elastic": "ON",
                "P2_reference_step_time_s": "0.358758",
                "onsager_step_time_s": "NOT_RUN",
                "runtime_overhead_ratio": "NOT_EVALUATED",
                "memory_overhead_ratio": "NOT_EVALUATED",
                "extra_FFT_count": "NOT_IMPLEMENTED",
                "status": "BLOCKED_INDEPENDENT_DATA_GATE",
            },
        ],
    )


def final_reports(data_present: bool) -> None:
    write_csv(
        "next7_first_failure.csv",
        ["stage", "predicate", "observed", "required", "classification", "action"],
        [
            {
                "stage": "independent_interface_data_gate",
                "predicate": "eligible_DATA_rows_present_and_rank_sufficient",
                "observed": "0 eligible DATA rows" if not data_present else "DATA rows present",
                "required": "rank-2 independent planar response for B_I,C_I at T380/T400",
                "classification": STATUS if not data_present else "READY_FOR_FIT",
                "action": "stop_runtime_implementation" if not data_present else "run_SPD_fit",
            }
        ],
    )
    write_csv(
        "next7_change_ledger.csv",
        ["path", "change", "physics_semantics", "test_coverage"],
        [
            {
                "path": "schemas/interface_onsager_data.schema.json",
                "change": "new independent planar response contract",
                "physics_semantics": "signs, units, provenance, uncertainty only; no runtime physics",
                "test_coverage": "tests/test_interface_onsager_schema.py",
            },
            {
                "path": "examples/interface_onsager_data_template.csv",
                "change": "T380/T400 six-state replicated TEMPLATE matrix",
                "physics_semantics": "no invented measurements",
                "test_coverage": "tests/test_interface_onsager_schema.py; template rejection fit test",
            },
            {
                "path": "scripts/fit_interface_onsager_matrix.py",
                "change": "strict data validation, rank gate, SPD Cholesky/Schur fit, covariance output",
                "physics_semantics": "offline only; rejects PF curved/seed-fate/speed sources",
                "test_coverage": "tests/test_interface_onsager_fit.py",
            },
            {
                "path": "reports/pf_ctot_production_candidate/next7_*",
                "change": "baseline freeze, error policy, rank derivation, data gate, design-only reports",
                "physics_semantics": "runtime remains unchanged and gated",
                "test_coverage": "report generator plus test suite",
            },
        ],
    )
    terminal = f"""fixed_ctot_model_preserved=true
q_only_closure_status=REJECTED
required_independent_data_dimension=2
independent_interface_data_present={str(data_present).lower()}
interface_matrix_fit_status={'READY' if data_present else 'WAITING_FOR_DATA'}
interface_matrix_SPD=NOT_EVALUATED
selected_interface_closure=UNRESOLVED_PENDING_INDEPENDENT_DATA
cross_coupling_required=UNRESOLVED_PENDING_INDEPENDENT_DATA
diffuse_basis_gauge_uncertainty=NOT_EVALUATED
host_onsager_status=NOT_RUN_DATA_GATE
CUDA_onsager_status=NOT_RUN_HOST_GATE
discrete_reciprocity_status=DESIGN_CONTRACT_ONLY
total_dissipation_status=NOT_EVALUATED
P2_reference_step_time=0.101403_s_64cube_elastic_OFF
onsager_step_time=NOT_RUN
runtime_overhead_ratio=NOT_EVALUATED
memory_overhead_ratio=NOT_EVALUATED
extra_FFT_count=NOT_IMPLEMENTED
heldout_planar_status=NOT_RUN
cylindrical_status=NOT_RUN
spherical_status=NOT_RUN
minimum_R_over_lambda_for_5percent=NOT_ESTABLISHED
minimum_R_over_lambda_for_2percent=NOT_ESTABLISHED
basis_uncertainty_max=NOT_EVALUATED
adaptive_BE_status=NOT_STARTED_PHYSICS_GATE
BDF2_status=NOT_STARTED_PHYSICS_GATE
accepted_physical_time_per_wall_hour=NOT_EVALUATED
production_grid_approved=false
S3_source_component_frozen=true
S3_reintegration_allowed=false
cluster_used=false
commit_created=false
push_performed=false
recommended_next_action=provide_or_authorize_independent_planar_linear_response_data_with_covariance_at_T380_and_T400
final_status={STATUS if not data_present else 'INDEPENDENT_DATA_PRESENT_RUN_FIT'}
"""
    write_text("final_terminal_output.txt", terminal)


def main() -> int:
    freeze_baseline()
    accuracy_policy()
    interface_rank()
    data_present = audit_independent_data()
    design_only_reports()
    if data_present:
        raise RuntimeError("eligible independent data appeared; run and audit the SPD fit before proceeding")
    blocked_reports()
    final_reports(data_present)
    print((REPORT / "final_terminal_output.txt").read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
