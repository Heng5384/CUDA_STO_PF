# Calibration–conditioning–validation language V3

| Item | Data role | Safe wording | Evidence status | Forbidden wording |
|---|---|---|---|---|
| four solvus points | CALIBRATION | calibrated/fitted pseudo-binary description | SUPPORTED fit; uncertainty partial | validated phase diagram |
| exact-fit candidate | CALIBRATION CANDIDATE | independently reconstructed exact-fit candidate | SUPPORTED candidate | production contract used by existing runs |
| legacy runtime `41212.9-18.05T` | PRODUCTION INPUT | active legacy contract for in-flight 21-case runs | SUPPORTED provenance | final publication thermodynamics |
| 380 °C root | MODEL-DERIVED | derived from the named contract | SUPPORTED per contract | independent validation |
| 6 h matrix/inventory/PSD bounds | CONDITIONING | experiment-constrained admissible state | PARTIAL/CONDITIONAL | predicted or uniquely reconstructed state |
| PF conservation/restart/elastic residuals | NUMERICAL QUALIFICATION | numerically qualified checkpoint chain | SUPPORTED for completed chains | experimental validation |
| 246³ A/B/C | PRODUCTION-AUTHORITY SENSITIVITY | fixed-PSD random realization ensemble | SUPPORTED | processing treatments or RVE convergence |
| 400³ Case 001–003 checkpoints | COMPLETE-CHECKPOINT CONDITION | complete trajectories with conditional post-processing | CONDITIONAL | final production authority |
| Case 001–003 48 h experiment | COMPARISON/POSTDICTION | comparison with known 48 h state | PENDING OPERATOR | blind prediction |
| AQ/6 h κ used for V2 | CALIBRATION | background/interface calibration closure | SUPPORTED | independent agreement |
| 48 h V2 transport | FROZEN-CONTRACT POSTDICTION | frozen-contract interface-driven recovery | CONDITIONAL | absolute total-κ prediction |
| Yu AQ/48 h | COMPLEMENTARY MECHANISM EVIDENCE | broader defect redistribution endpoints | SUPPORTED | Sheskin 6→48 h continuation |

## Required Methods ledger sentence

“The 6 h microstructure and transport information condition the resolved-population and interface contracts, whereas the subsequent phase-field trajectory and frozen 48 h transport comparison are post-handoff outputs; agreement with conditioning data is not counted as independent validation.”

## Dataset boundary

Sheskin and Yu are used as complementary but non-interchangeable datasets: the former defines the 6–48 h microstructure–thermal-conductivity trajectory, whereas the latter resolves the broader defect redistribution associated with long-term annealing.

## Current validation ceiling

The project can claim numerical qualification, traceable conditioning, production-authority 246³ sensitivity and conditional 400³ frozen-contract postdiction. It cannot yet claim an end-to-end validated predictor because thermodynamic identity, observation identity, 400³ production authority and thermal quantity identity are not all closed.
