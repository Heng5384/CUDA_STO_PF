# Physically scaled signed transport-defect gate V3

## Status and scope

This contract replaces neither the transport equation nor its cold residual.
It changes only the long-window observer used to decide whether a relaxed
solver gate introduces a material transport bias. Historical V1 and V2
decisions remain immutable negative controls.

For accepted step `n` and cell `i`,

```text
D_i = sum_n dt_n R_C,i^n
A_i = sum_n |C_i^(n+1) - C_i^n|
```

The interface set is the frozen moving-interface union used by the existing
transport-gate trajectory observer. With a preregistered positive `A_floor`,

```text
beta_interface = |sum_interface D_i| /
                 max(sum_interface A_i, A_floor)

beta_interface_excess =
    |sum_interface D_candidate - sum_interface D_strict| /
    max(sum_interface A_strict, A_floor)
```

The old sign-coherence statistic

```text
b_interface = |sum_interface D_i| / sum_interface |D_i|
```

is retained as `SIGN_COHERENCE_DIAGNOSTIC_ONLY`. If
`eta_interface <= 1e-6`, it is reported as
`SIGN_COHERENCE_NOT_MATERIALLY_INTERPRETABLE` and cannot independently reject
a candidate.

## Hard gates

| Metric | Limit |
|---|---:|
| `eta_global` | `1e-4` |
| `eta_interface` | `1e-3` |
| `eta_cell_max` | `1e-3` |
| `beta_interface` | `1e-4` |
| `beta_interface_excess` | `1e-4` |

All pre-existing mass, storage, KKT, energy/work, bound, retry, QoI, and
holdout gates remain in force.

## Independent holdout

The sixth physical-time window was generated after V3 registration. For the
selected `G9/dt4` candidate it produced:

| Metric | Value | Result |
|---|---:|---|
| `eta_global` | `1.26342711657069e-08` | PASS |
| `eta_interface` | `9.79778165777453e-09` | PASS |
| `eta_cell_max` | `3.24388981507639e-04` | PASS |
| `beta_interface` | `4.29739643823258e-10` | PASS |
| `beta_interface_excess` | `4.29461793033391e-10` | PASS |

The six-window interval covers `385.589863355201 s` and approximately
`3.02843 nm` of interface displacement. Evidence is frozen in
`reports/transport_residual_gate_v3/`.

`V3_status=PASS_PHYSICALLY_SCALED_SIGNED_DEFECT_GATE_V3`
