# Transport residual-gate plumbing audit

## Verdict

`gate_plumbing_status=PASS_UNIFIED_EXISTING_THRESHOLD_WITH_TRANSACTIONAL_OBSERVER`

The existing `ctot_residual_abs_tol` and `ctot_residual_rel_tol` form one
effective threshold:

`G_eff = ctot_residual_abs_tol + ctot_residual_rel_tol * initial_res_inf`.

For this matrix only the absolute term is varied. The relative term remains
fixed at `1e-10`; every row records both requested and effective gates.

## Consumers

| Role | Current source lines |
|---|---|
| Parameter declaration | `pf_params.h:397-398` |
| Defaults | `main_cuda.cu:18505-18506` before observer instrumentation |
| Parser | `main_cuda.cu:24047-24048` before instrumentation |
| Nonlinear stopping target | `main_cuda.cu:34366-34367` before instrumentation |
| Line-search endpoint acceptance | `main_cuda.cu:34561` before instrumentation |
| Final nonlinear cold check | `main_cuda.cu:34653-34654` before instrumentation |
| Outer acceleration merit gates | `main_cuda.cu:36442-36443`, `36696-36697` before instrumentation |
| Outer transport acceptance | `main_cuda.cu:37773-37774` before instrumentation |
| Method-consistent cold gate | `main_cuda.cu:38291-38296` before instrumentation |
| Retry/failure classification | immediately follows the method gate |

All current references were re-scanned at generated lines: `[2285, 2286, 18507, 18508, 24050, 24051, 27385, 27386, 30571, 34487, 34488, 34682, 34774, 34775, 36053, 36054, 36577, 36578, 36831, 36832, 37908, 37909, 38430, 38431, 38489, 38490, 38832, 38834, 38867, 38868, 38875, 38876, 38904, 38905, 42877]`.
There is no divergent hard-coded `1e-12` final cold gate. The line-search
endpoint exception uses the same absolute parameter and therefore remains
aligned with each requested test gate.

## Observation contract

`ctot_transport_gate_trajectory_diagnostics` defaults to zero. When enabled,
it copies the method-context cold residual immediately after evaluation. A
normal accepted macro is committed directly; event substeps remain in a host
transaction buffer until `CTOT_BDF2_EVENT_MACRO_READY`. Any depth escalation
clears that buffer before rollback/replay. Rejected or later-rolled-back
attempts therefore do not enter `E_R` or `D_i`. Its host arrays are not passed
to a CUDA kernel or solver.

Observer schema: `CTOT_TRANSPORT_GATE_TRAJECTORY_V2_TRANSACTIONAL`.

The 64-step and event-bearing 360-step state hashes are identical with the
observer off and on, establishing transaction neutrality for both ordinary
and BE-subcycled qualification paths.
