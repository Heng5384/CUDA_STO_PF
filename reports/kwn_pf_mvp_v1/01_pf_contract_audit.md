# PF contract audit for KWN–PF MVP v1

Status: `P0_CONTRACT_CONFLICT`.

## Freeze decision

`thermodynamic_contract_freeze_v1/07_frozen_thermodynamic_contract.yaml`
explicitly declares `BLOCKED_CONTRACT_CONFLICT` and has no contract hash.
Accordingly, this MVP does not select a PF thermodynamic parameter set, modify
PF/CUDA source, create a production initial condition, or launch a PF run.

| contract aspect | legacy executable/audited path | exact candidate | disposition |
|---|---|---|---|
| regular-solution parameter | `L(T)=41212.9 - 18.05*T` J mol⁻¹ | `L(T)=41504.29119633958 - 18.469276826409214*T` J mol⁻¹ | unresolved authority conflict |
| 380 °C (`653.15 K`) matrix root | `xB=0.004664951821454188` | `xB=0.004649261005504821` | candidate is reproducible but not deployed/frozen |
| source identity | legacy paths have source provenance but mixed historical runtime/binary identity | external calibration candidate lacks an approved PF source/binary/fixture binding | P0 blocker |
| required action | signed thermodynamic contract decision plus residual and downstream-impact artifacts | then matched exact-versus-legacy PF sensitivity | required before any PF coupling result |

The precise candidate values are retained in the audit because they are useful
evidence, not because they are accepted for use.  The authority decision in
`thermodynamic_contract_freeze_v1/02_authoritative_contract_decision.md`
remains controlling.

## Numerical/physical input conflicts relevant to coupling

| item | source/value | conflict or use restriction |
|---|---|---|
| alpha–beta interface energy | `physical_inputs.example.json`: `gamma=0.168 J m⁻²` | retained as a legacy physical input; no new PF retuning |
| regular interface width/grid | same file: `lambda_sm=0.6 nm`, `dx=0.1 nm` | differs from later qualified fixture settings |
| GP interface surrogate | same file: `gp_gamma_alpha_gp=0.05 J m⁻²`, `gp_l_eta=1 nm` | effective GP model, not a validated KWN phase definition |
| qualified elastic profile | `physical_override_T380_dx1nm_lambda4nm.json`: 380 °C, `dx=1 nm`, `lambda_sm=4 nm` | numerical-profile contract distinct from the example input |
| smallest handoff fixture | 96³ six-particle fixture: 380 °C, `dt_code=0.02`, `lambda_sm=4 nm`, mean `C_B_tot=0.03` | validation-only and restricted to its registered 8.0–10.5 nm library profiles |
| elastic target | project core memory: principal eigenstrain `(0.046, -0.022, -0.017)` in the existing periodic fixed-cell target | may be cited only with its existing profile provenance; no case-specific retuning |

## State and mass contract

The KWN side stores B inventory in SI mol B m⁻³:

```text
C_B,total = C_B,matrix + C_B,GP + C_B,beta.
```

Each population uses its own declared molar volume.  The matrix fraction is
computed from the two population volume fractions; the matrix composition is
recovered from the fixed total rather than clamped.  A handoff package also
records a fourth bucket, `C_B,beta_subgrid`, whenever beta material cannot be
represented by the PF resolved `phi` field.

The current PF storage relation is based on phase interpolation, e.g.
`h_beta + (1-h_beta)*xB_alpha` in beta-only mode and a legacy GP eta term when
that mode is enabled.  Its saved state does not carry a standalone KWN
subgrid-beta inventory.  Moving GP or subgrid beta inventory into `xB_alpha`
would change the physical state and is forbidden.  The adapter therefore
fails closed with `PARTIAL_PF_STATE_NOT_CLOSED`.

## Outcome

Only generic, versioned package validation is in scope now.  Beta-only PF
comparison and PF smoke execution remain blocked, and no result in this
delivery represents a PF-calibrated physical prediction.
