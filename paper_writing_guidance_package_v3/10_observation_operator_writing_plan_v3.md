# Observation operator writing plan V3

## Narrative role

The observation gap explains why an admissible ensemble is necessary. It is a Methods-level identity contract and uncertainty source, not the manuscript's dominant headline.

## Object classes

| Class | Operational identity | Comparison rule |
|---|---|---|
| PF-resolved stoichiometric β | connected β-like regions after `φ`/composition thresholding | model internal until `H_obs` is applied |
| APT-detected Ag-rich objects | composition-thresholded objects in a limited reconstructed volume | may include nonstoichiometric/unresolved classes; not one-to-one β |
| GP-like/unresolved Ag-rich population | features below β stoichiometry or experimental/PF resolution | report as a separate reservoir/bound |
| SEM/TEM-visible coarse precipitates | projection/contrast/resolution-dependent coarse objects | use method-specific 2D/3D stereology and cutoff |

## Minimum operator

Define

`H_obs[φ(r), xB(r); θ_obs] → {Nv_obs, PSD_obs, Sv_obs, morphology_obs}`.

The parameter set `θ_obs` must include:

1. point-spread blur or resolution kernel;
2. voxelization/resampling;
3. phase and/or composition threshold;
4. connected-component connectivity;
5. minimum detectable size;
6. edge censoring and ROI;
7. component versus watershed/neck separation;
8. 2D section, projection or 3D sampling mode;
9. sampling-volume/bootstrap uncertainty;
10. resolution/threshold sensitivity grid.

## Required tests

- synthetic isolated-sphere recovery across size and contrast;
- two touching particles with neck strength sweep;
- one 3D object intersected by multiple 2D sections;
- size cutoff and finite sampling-volume bias;
- component-based versus watershed-sensitive `Nv`, PSD and `Sv`;
- threshold/resolution propagation to the admissible 6 h ensemble;
- 6 h conditioning versus 48 h held-out/operator comparison.

## Main-text Methods wording

“We do not reconstruct a unique experimental 6 h precipitate population. Instead, we construct an ensemble of experimentally admissible resolved-β populations constrained by total Ag inventory, matrix composition and microscopy-dependent observation windows.”

Immediately add: “PF-resolved β, APT Ag-rich objects, GP-like/unresolved populations and SEM/TEM-visible coarse precipitates are retained as distinct object classes.”

## Reporting

- Main text: operator diagram, nominal settings, shared observables and one sensitivity panel.
- SI: pseudocode, masks, all threshold/resolution sweeps, per-object tables, 2D/3D tests, watershed alternatives and bootstrap distributions.
- Captions: object class, cutoff, ROI, `n`, sampling mode and whether the quantity conditions or evaluates the model.

## Current status

`PENDING_MINIMUM_OBSERVATION_OPERATOR_IMPLEMENTATION`. Until closure, experimental `Nv/PSD/Sv` must not be compared one-to-one with PF components, and the low-κ window may be framed through Sheskin's reported microstructure/κ association rather than claimed as quantitatively reconstructed interface area.
