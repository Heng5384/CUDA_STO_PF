# Microstructure Evolution Capability Report

## Capability Matrix

| Capability | Status | Basis | Limitations |
|---|---:|---|---|
| Nucleation | present | Scheduled insertion, GP-assisted events, GP-to-beta conversion, GP seeding | Mostly event-driven, not purely variational spontaneous nucleation. |
| Growth | present | Phi/eta PDE evolution and composition transport | Growth depends on calibrated mobilities, thermodynamics, and inserted seed quality. |
| Coarsening | partial | Gradient penalties and order-parameter dynamics allow coarsening-like relaxation | Event/reservoir paths are not a full coarsening thermodynamic model. |
| Phase competition | partial | Alpha/GP/beta phase fractions and storage interpolation | Multiple GP/beta routes are not unified under one free-energy functional. |
| Spatial pattern formation | present | PF fields, eta morphology, stochastic/site-dependent nucleation, transport | Pattern formation may depend strongly on heuristic event parameters. |
| GP-assisted heterogeneous nucleation | present | GP-to-beta and GP-assisted hazard/event modules | Coupling is mixed and partly heuristic. |

## Interpretation

The code can generate microstructures with GP zones, beta nucleation, beta growth, and spatially heterogeneous event patterns. It is therefore operationally capable as a simulation framework.

The scientific interpretation must be careful: observed patterns arise from a hybrid of PF evolution and event insertion. They should not be presented as the result of one fully variational free-energy model unless the event paths are disabled or reformulated.

## Verdict

The microstructure capability is **partial-to-present**.

The system can produce rich microstructure evolution, but the physical mechanism is hybrid and partly non-variational.
