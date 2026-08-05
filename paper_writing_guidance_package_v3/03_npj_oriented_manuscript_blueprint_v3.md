# npj Computational Materials-oriented manuscript blueprint V3

## Manuscript subject

The npj version is an experiment-constrained computational capability for propagating non-identifiable microstructure states into kinetic and thermal uncertainty. PbTe–Ag2Te remains the scientific problem, not a decorative demonstration, and the paper must not become a software-engineering report.

## Recommended structure

1. **Introduction — a materials inference gap.** Start with Sheskin's low-κ state and the mismatch between microscopy-dependent object windows. Generalize only to the computational problem of propagating admissible states when the latent microstructure is not uniquely observed.
2. **Results I — versioned ensemble architecture.** Define data roles, immutable inputs, admissibility constraints, PF propagation, observation operator and full-PSD transport handoff.
3. **Results II — thermodynamic and inventory contracts.** Show conversion/gauge/solvus provenance and explicitly report the exact-candidate versus legacy-runtime boundary. A unique publication contract is required before submission.
4. **Results III — admissible 6 h states.** Explain why a unique resolved β state is not reconstructed; version fixtures and observation assumptions.
5. **Results IV — uncertainty propagation through conserved-elastic PF.** Present Case 001–003 as fixed-PSD spatial uncertainty propagation and thermal-response robustness, not as a complete uncertainty distribution.
6. **Results V — pathway–outcome separation.** Show survival/`M6` sensitivity versus `Sv` and V2 MI recovery robustness. Retain the exact production-authority qualification.
7. **Results VI — information-preserving handoff.** Compare full PSD, `Nv+Rmean`, `Sv`, `M6` and `Sv+M6`; full PSD remains authority.
8. **Results VII — observation and transfer domains.** Demonstrate `H_obs` sensitivity and state which inputs must change for another microscopy protocol or material.
9. **Discussion.** Separate reusable capability from PbTe-specific mechanism, then discuss failure domains: thermodynamic identity, object identity, finite size, missing electronic/dislocation channels and designed spatial tests.
10. **Methods.** Provide equations, fixture schemas, version/hash manifests, post-processing, uncertainty metrics and reproduction instructions after Results/Discussion, following npj corpus practice.

## Frozen dataset sentence

Sheskin and Yu are used as complementary but non-interchangeable datasets: the former defines the 6–48 h microstructure–thermal-conductivity trajectory, whereas the latter resolves the broader defect redistribution associated with long-term annealing.

## Capability claim

The workflow propagates experimentally admissible microstructure uncertainty through conserved-elastic phase-field kinetics and a frozen full-PSD transport contract, thereby separating uncertainty in the kinetic pathway from robustness of the modeled thermal consequence.

## Required npj evidence

- one frozen publication thermodynamic contract;
- implemented and versioned observation operator;
- production-authority Case 001–003 and completed 21-case workflow;
- matched finite-size test;
- executable code/data archive and reconstruction instructions;
- module-level calibration/conditioning/validation ledger;
- explicit total-κ versus lattice-style output identity;
- at least one transfer test or appropriately bounded “formulated for” language.

## Forbidden npj drift

Do not use `generalizable`, `predictive`, `high-fidelity`, `digital twin` or `validated end-to-end` merely because modules are connected. Do not let manifests, GPU implementation or automation replace the material finding that pathway variables and thermal-outcome variables differ.
