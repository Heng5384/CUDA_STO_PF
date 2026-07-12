# Coupling Consistency Report

## Coupling Questions

### 1. Does GP influence beta nucleation?

**Yes.**

GP influences beta nucleation through multiple mechanisms:

- Continuous eta-rich regions can be used by GP-to-beta conversion logic.
- GP-assisted hazard/barrier terms can increase beta event probability.
- GP reservoir/site events can directly insert beta seeds.
- Scheduled insertion can consume selector/mapped templates derived from external physics, depending on runtime configuration.

### 2. Is the influence variational, stochastic, or direct?

It is **mixed**:

| Route | Coupling Type | Assessment |
|---|---|---|
| Eta field in phase fractions/storage/mobility | variational-like PF coupling | Physically meaningful but GP free energy is parameterized/surrogate. |
| GP-to-beta stochastic acceptance | CNT-like stochastic coupling | Uses eta/local state and barrier/rate-like probability. |
| GP-assisted reservoir beta event | direct event injection | Mass conserving, but not variational. |
| Scheduled beta insertion | deterministic external insertion | Runtime insertion path, not spontaneous PF nucleation. |

### 3. Does beta growth feed back into composition?

**Yes.**

Beta fraction affects storage and transport through `compute_Y_rhs_gp_kernel`, phase fractions, and event mass-compensation logic.

### 4. Does beta growth feed back into GP depletion or redistribution?

**Partially.**

In continuous GP mode, phase fractions enforce competition because:

```text
h_GP = (1 - h_beta) h_eta
```

so beta suppresses local GP fraction. In event/reservoir paths, GP-assisted beta events can consume GP site mass or mark GP sites as used. These mechanisms are not fully unified.

## Energy Consistency

The couplings are **not fully energy-consistent**.

Reasons:

- PF eta/phi evolution is variational-like.
- GP-assisted reservoir logic explicitly bypasses the eta/free-energy route.
- Scheduled insertion imposes nuclei rather than deriving them from local Euler-Lagrange instability.
- CNT/hazard terms are not globally derived from the same PF free-energy functional.
- External validation reports indicate some response channels, especially temperature/strain response in the event layer, are incomplete or absent.

## Verdict

Coupling type: **hybrid**.

The system has real coupling, but it is a mixed variational/stochastic/direct-injection architecture. It should not be described as a fully variational GP-to-beta phase-field model.
