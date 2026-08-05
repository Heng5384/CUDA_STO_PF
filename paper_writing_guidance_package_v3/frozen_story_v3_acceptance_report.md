# Frozen story V3 acceptance report

## Final status

`CONDITIONAL_PASS_PENDING_PRODUCTION_AUTHORITY`

The writing library is logically reframed and internally frozen. The paper is not submission-ready because several P0 evidence gates remain open.

## New central line

Ag-alloyed PbTe contains a finite-lifetime interface-rich low-κ state; experimentally admissible resolved-β ensembles reveal that PSD and spatial organization select how the population coarsens, while interfacial-area loss sets the leading modeled thermal recovery.

## Primary punchline

> Ag-alloyed PbTe exhibits a metastable low-thermal-conductivity window sustained by an interface-rich precipitate population. PSD and spatial organization determine how this population coarsens, whereas the loss of interfacial area determines how the thermal window closes.

Acceptance qualification: the last clause is a strong closing statement only when “determines” is read within the frozen V2 interface-scattering contract. It is not yet a proof that `Sv` uniquely controls measured total κ.

## Sheskin and Yu roles

Sheskin and Yu are used as complementary but non-interchangeable datasets: the former defines the 6–48 h microstructure–thermal-conductivity trajectory, whereas the latter resolves the broader defect redistribution associated with long-term annealing.

Sheskin supports the low-κ state, coarsening and κ recovery. It does not provide the complete 6→48 h zT trajectory used by this paper. Yu supports GP-like objects, matrix/β redistribution, strain relaxation and Ag-decorated dislocations over AQ→48 h; it is not inserted into Sheskin's timeline or used to supply Sheskin's missing zT information.

## What Case 001–003 support

- same `REF_BROAD_METHOD1LIKE` PSD, 512 particles, total inventory, physics and numerics;
- changed centers and radius–position random assignment;
- all three reach complete 48 h checkpoints;
- conditional endpoints: 21/19/21 particles; mean R 24.561/25.402/24.916 nm; `Sv` 2.6327/2.5397/2.6654×10^6 m^-1; `M6` 137.03/168.75/126.18 nm^3;
- range/mean: R 3.37%, `Sv` 4.81%, `M6` 29.56%;
- frozen V2 MI recovery at 573.15 K: 4.10/4.21/4.07%; Δκ spread/mean 3.29%.

They support “spatially sensitive coarsening, spatially robust modeled thermal recovery” for three random fixed-PSD realizations. They do not support ensemble convergence, designed spatial control, all transport-channel robustness or production-authoritative values.

## Why “low-κ window” is justified

The term denotes the interface-rich microstructure–transport state reported around Sheskin's 6 h condition and its finite-lifetime evolution toward lower interface density and higher κ by 48 h. It is tied to a microstructural state, `Sv` loss and a measured trajectory. It is not an arbitrary time label, a claimed high-zT window or a unique equilibrium phase.

## What still cannot be stated

- true 6 h β coordinates/population were reconstructed;
- 6 h is the highest-zT state;
- Sheskin and Yu form one ageing dataset;
- random seeds prove controllable spatial architecture;
- temperature or strain stabilizes the window;
- `M6` or `Sv` alone is a universal sufficient descriptor;
- resolved β alone quantitatively reproduces measured total κ;
- present Case 001–003 values are final production authority;
- the 21-case campaign proves PSD/density/processing effects;
- the current model predicts full zT.

## Objective Acta/npj comparability assessment

| Comparator dimension | Current idea | Relevant precedent | Objective judgment |
|---|---|---|---|
| post-nucleation handoff | admissible 6 h ensembles | Acta A01 | comparable concept; weaker operator closure |
| observation/data-role discipline | four object classes and planned `H_obs` | Acta A06/A07 | conceptually aligned; implementation clearly weaker |
| bounded mechanism inference | sign/robustness plus residual | Acta A08 | comparable narrative form; weaker because no orthogonal mechanism contrasts/material plausibility closure |
| spatial mechanism | path sensitivity and planned local descriptors | Acta A03/A09 | interesting result; weaker causal evidence because random realization is not a designed field/control |
| processing design | window-lifetime target | Acta A02 | not comparable yet; no forward processing experiment |
| thermodynamic/workflow contract | versioned thermodynamics→PF→transport | npj N03/N01/N04 | comparable architecture; weaker unique contract/archive/transfer evidence |
| computational-only publishability | mechanism plus verification/failure domain | npj N05 | possible in principle; current verification/authority package incomplete |

### Direct answer: does the idea qualify to close the paper?

**Yes, at the level of scientific framing and Discussion/Conclusion rhetoric.** It is a legitimate and nontrivial punchline because it separates path selection from property consequence and is supported by unequal structural/thermal sensitivities.

**No, at the level of final submission evidence today.** The strongest causal verbs still depend on conditional post-processing, an unimplemented observation operator, a conflicted publication thermodynamic contract, unresolved thermal quantity identity, no matched finite-size bridge and no designed spatial architecture.

The idea is therefore comparable to Acta/npj papers, but the evidence package is currently below their strongest accepted examples. The novelty is not the existence of coarsening; it is the three-level separation between size hierarchy, survival network and interface-controlled modeled recovery.

## Journal and maturity decision

- Scientific story maturity: **7.8/10**.
- Punchline conceptual novelty: **7.5/10**.
- Evidence closure: **4.8/10**.
- Acta submission readiness: **4.5/10 — NO**.
- npj Computational Materials readiness: **5.0/10 — NO**.
- Most realistic route after minimum P0 closure: **Computational Materials Science**.
- Strongest aspirational route after operator/authority/causal controls: **Acta Materialia**.
- npj becomes competitive only if versioning, scalability, transfer and archive are elevated to evidence rather than documentation.

## Acceptance gates still open

Schema repair; frozen merge-aware Case 001–003 authority; all 21 cases; publication thermodynamic contract; thermal quantity identity; minimum observation operator; finite-size bridge; designed spatial contrasts; temperature/elastic proof-of-concept for processing language.
