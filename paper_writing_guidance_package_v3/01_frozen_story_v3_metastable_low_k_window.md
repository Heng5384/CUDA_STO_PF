# Frozen scientific story V3 — metastable low-κ window

## One-paragraph story

Ag-alloyed PbTe follows a finite-lifetime, low-thermal-conductivity state during Sheskin's 6→48 h ageing trajectory: the 6 h state contains a dense, interface-rich Ag-rich population, whereas continued ageing coarsens that population, reduces interfacial area and accompanies thermal-conductivity recovery. Because APT and coarse microscopy cover different object identities and visibility windows, a unique PF-resolved stoichiometric β population cannot be reconstructed at 6 h. We therefore construct experimentally admissible resolved-β ensembles constrained by Ag inventory, matrix composition and observation windows, and propagate their post-nucleation evolution with a globally conserved coherent-elastic phase-field model. Complete-checkpoint conditional analysis of three 400³ fixed-PSD spatial realizations shows that random placement changes particle survival and the large-size tail much more than it changes the frozen V2 interface-driven thermal recovery. The manuscript therefore separates three levels: PSD defines the size hierarchy, spatial organization selects the survival pathway, and loss of interfacial area closes the modeled low-κ window; Yu 2024 is used only as complementary evidence that real long-term annealing also redistributes Ag among GP-like objects, the matrix, β precipitates, strain fields and Ag-decorated dislocations.

## Five-sentence executive story

1. Sheskin's 6 h state is treated as an experimentally observed, interface-rich low-κ state that evolves toward higher κ by 48 h.
2. Microscopy-dependent visibility makes the resolved-β initial population non-identifiable, so the model uses constrained admissible ensembles instead of claiming a unique reconstructed microstructure.
3. At fixed initial PSD and inventory, Case 001–003 show spatially sensitive survival and large-tail evolution but comparatively robust V2 interface-driven thermal recovery.
4. Variables controlling the coarsening pathway are not identical to those controlling its thermal consequence: spatial organization selects the former, whereas interfacial-area loss dominates the latter within the frozen V2 contract.
5. The processing objective is therefore the lifetime of the interface-rich low-κ state, while temperature and elastic bias remain testable routes rather than demonstrated controls.

## Full scientific narrative

### 1. Experimental state and question

Sheskin 2018 reports that ageing from 6 to 48 h changes precipitate number density, size and thermal conductivity. The low-κ state is not defined as an arbitrary time interval: it is a microstructure–transport state with high Ag-rich object density, high inferred interfacial area and reduced κ, followed by coarsening and κ recovery. The question is not whether coarsening can continue, but how much an experimentally admissible resolved population can restructure, which variables select its path, and which structural information controls the modeled thermal consequence.

### 2. Observation gap and ensemble logic

PF components, APT Ag-rich objects, GP-like/unresolved objects and SEM/TEM-visible coarse precipitates are not interchangeable. We do not reconstruct a unique experimental 6 h precipitate population. Instead, we construct an ensemble of experimentally admissible resolved-β populations constrained by total Ag inventory, matrix composition and microscopy-dependent observation windows. The observation gap justifies the ensemble; it is not the manuscript's terminal message.

### 3. Post-nucleation propagation

The globally conserved coherent-elastic PF model propagates each admissible 6 h state to 48 h. It does not predict absolute nucleation. The current production trajectories use the legacy runtime thermodynamic contract; the exact-fit candidate and legacy runtime are not silently mixed. Final publication thermodynamics remain blocked until one contract is frozen and propagated.

### 4. Pathway–outcome separation

Cases 001–003 start with the same `REF_BROAD_METHOD1LIKE` PSD, 512 particles, total inventory, physics and numerics; only centers and radius–position assignment vary. Their complete 48 h checkpoints yield conditional endpoint populations of 21/19/21 particles, mean radii of 24.561/25.402/24.916 nm, `Sv` of 2.6327/2.5397/2.6654×10^6 m^-1 and `M6` of 137.03/168.75/126.18 nm^3. Relative ranges are 3.37% for mean radius, 4.81% for `Sv`, and 29.56% for `M6`. Under the frozen V2 MI contract at 573.15 K, the modeled recoveries are 4.10%, 4.21% and 4.07%; the spread in Δκ is about 3.29% of its three-case mean. At fixed initial PSD, precipitate inventory and thermodynamic conditions, random spatial realization substantially altered particle survival and the large-size tail of the evolving PSD, yet changed the predicted interface-driven thermal recovery by only approximately 3–4%. The coarsening pathway is therefore spatially sensitive, whereas the thermal response is comparatively robust.

These values are `COMPLETE_CHECKPOINT_CONDITIONAL_ANALYSIS`. They cannot be called final production authority until the 512-particle fixture/schema conflict and merge-aware post-processing are repaired and rerun.

### 5. Descriptor hierarchy

PSD defines the available donor–receiver size hierarchy and curvature competition. Spatial organization changes nearest neighbours, capture zones, diffusion competition and elastic interactions, thereby selecting which particles survive and how the large-size tail develops. `M6` describes leverage from the large-radius tail; it is not the unique determinant of κ. Within the frozen interface contract, `Sv` loss controls the direction and leading modeled magnitude of recovery. `Sv+M6` compresses the tested full-PSD transport more faithfully than `Nv+Rmean` or `Sv` alone, but full PSD remains the authority because moment closures are non-unique.

### 6. Thermal and literature boundary

The transport model is PF-informed lattice-style transport for resolved-particle/interface levers. It does not calculate full `zT`, Seebeck coefficient, electrical conductivity, carrier concentration, weighted mobility, defect-dependent electronic transport, a measured dislocation contribution, or a strict total-κ decomposition. Sheskin and Yu are used as complementary but non-interchangeable datasets: the former defines the 6–48 h microstructure–thermal-conductivity trajectory, whereas the latter resolves the broader defect redistribution associated with long-term annealing.

## Primary punchline

> Ag-alloyed PbTe exhibits a metastable low-thermal-conductivity window sustained by an interface-rich precipitate population. PSD and spatial organization determine how this population coarsens, whereas the loss of interfacial area determines how the thermal window closes.

Short forms:

- Microstructure selects the pathway; interfacial area sets the thermal outcome.
- Spatially sensitive coarsening, spatially robust thermal recovery.
- The processing target is not a unique precipitate arrangement, but the lifetime of the interface-rich low-κ state.

## Secondary bounded conclusion

> Resolved β coarsening governs the visible kinetic pathway, while the remaining transport discrepancy motivates contributions from unresolved Ag-rich populations and broader defect-state evolution.

This secondary conclusion must never replace the primary punchline or identify a unique missing mechanism.

## Evidence boundaries

- `metastable low-κ window`: supported as an experimental framing of Sheskin's 6→48 h κ/microstructure trajectory; it is not a proven zT optimum.
- `PSD defines the size hierarchy`: mechanistically supported; between-PSD effect sizes remain pending the 21-case campaign.
- `spatial organization selects the survival pathway`: conditionally supported by fixed-PSD Case 001–003 and the production-authority 246³ A/B/C ensemble; only three 400³ realizations are available.
- `interfacial-area loss sets the thermal outcome`: supported only within the frozen V2 interface-scattering contract; not a universal or total-κ causal proof.
- `thermal response is robust`: means robust to these three random realizations for the modeled interface signal, not to all spatial architectures or all transport channels.
- Yu supports broader defect redistribution; it does not extend Sheskin's time series.

## Unresolved blockers

1. Repair the 512-particle V2 fixture versus 64-particle schema conflict.
2. Freeze merge-aware unified post-processing and rerun Case 001–003 as production authority.
3. Complete all 21 cases.
4. Freeze one publication thermodynamic contract and propagate its consequences.
5. Implement the minimum observation operator with object classes and sensitivity.
6. Harmonize or explicitly bound total versus lattice-style thermal quantity identity.
7. Build the matched finite-size bridge before using robustness/convergence language.
8. Add designed spatial architectures before claiming spatial control.
9. Demonstrate temperature/history or elastic-bias control before making a processing prescription.
