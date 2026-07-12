# CNT / Nucleation Operator Verification Report

## Operators Found

| Operator | Evidence | Type | Assessment |
|---|---|---|---|
| Scheduled beta insertion | `main_cuda.cu`: `apply_scheduled_events_cpu` | deterministic event insertion | Inserts selected/mapped profiles into PF fields. Trigger time is externally scheduled. |
| GP-assisted beta event path | `main_cuda.cu`: `trigger_gp_assisted_beta_event_host` | event-driven GP reservoir | Uses GP site/reservoir logic and mass accounting; not a variational field instability. |
| GP-to-beta conversion | `main_cuda.cu`: `apply_gp_to_beta_event_cpu` | hybrid GP field + stochastic CNT | Detects eta-rich regions and can use CNT-like acceptance probability. |
| GP nucleation event seeding | `main_cuda.cu`: `gp_nuc_enabled` path | stochastic/event GP formation | Computes local drive/barrier/rate-like quantities and seeds eta profiles. |
| Hazard-based GP assistance | `main_cuda.cu`: `gp_assisted_site_hazard`, stochastic site selection | heuristic CNT-like barrier modulation | Uses barrier and modifiers from GP strength, local composition, curvature/site density. |

## CNT-Like Ingredients

Present:

- Local driving force / barrier estimates.
- `DeltaG*`-style expressions in GP-to-beta conversion.
- Event probabilities from rates or hazard functions.
- Stochastic site selection in some GP-assisted paths.

Partial or missing:

- One unified CNT operator consistently coupled to the PF free energy.
- A single canonical `S(x)` field used everywhere.
- A guaranteed energy-consistent relation between PF free energy, CNT barrier, and inserted nucleus.

## Nucleation Classification

The current nucleation implementation is:

```text
hybrid PF + CNT/event operator
```

It is not purely PF instability and not purely CNT. It combines PF growth with event-driven nucleation, stochastic CNT-like acceptance, scheduled insertion, and GP-reservoir heuristics.

## Verdict

The nucleation operator is **partial**.

It contains CNT-like physics and runtime event triggering, but the implementation is not a single physically closed CNT-on-PF operator. Multiple paths can create beta nuclei, and not all paths use the same barrier, selector, or GP coupling definition.
