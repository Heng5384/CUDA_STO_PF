# Dual-Track Unified Physics Architecture

## System Type

```text
dual-track hybrid PF + CNT calibrated nucleation system
```

The architecture separates offline nucleation calibration from runtime phase-field
microstructure evolution.

## Hard Boundary

```text
TRACK 1: CNT scanning + nucleus library
  - offline only
  - produces calibration artifacts
  - never runs inside runtime PF

TRACK 2: PF microstructure evolution + S-field nucleation
  - runtime only
  - evolves x_B, phi_GP, and phi_beta
  - never performs critical-radius scans or CNT job submission

ONLY LINK:
  - S(x) calibrated from Track 1
  - DeltaG*(T,xB,strain) fit table from Track 1
```

## Track 1: CNT Scanning Layer

Implemented by:

- `cnt_scan_pipeline.py`
- `nucleus_catalog.schema.json`
- generated `Results/dual_track_calibration/nucleus_catalog.json`
- generated `Results/dual_track_calibration/barrier_surface_fit.csv`
- generated `Results/dual_track_calibration/shape_library/`

Responsibilities:

1. Scan `DeltaG(r, T, xB, strain)` over radius and shape descriptors.
2. Extract critical radius `r*`.
3. Extract barrier `DeltaG*`.
4. Generate an energy-ranked offline nucleus catalog.
5. Fit/export the barrier surface used for calibration.

Track 1 is explicitly marked as `runtime_policy = calibration_only`.

## Track 2: PF Microstructure Evolution

Implemented by:

- `pf_dynamics_core.py`
- `s_field_definition.py`
- `coupling_interface.json`

Runtime fields:

```text
x_B(x,t)       composition field, CH form
phi_GP(x,t)    GP field, Allen-Cahn form
phi_beta(x,t)  beta field, Allen-Cahn form
```

Runtime S-field:

```text
S(x) = S(phi_GP(x), grad phi_GP(x), x_B(x))
```

Runtime beta driving force:

```text
Delta g_beta_eff(x) = S(x) * Delta g_beta_bulk(x_B)
```

Runtime beta evolution:

```text
d phi_beta / dt = -L_beta deltaF / delta phi_beta
```

No CNT nucleation kernel is allowed in Track 2.

## Coupling Layer

The coupling contract is defined in `coupling_interface.json`.

Allowed interfaces:

- `S_field`
- `DeltaG_star_surface_fit`

Forbidden runtime dependencies:

- realtime radius scan
- realtime CNT barrier calculation
- nucleus library template selection
- DFT/CE/minimization job submission
- stochastic CNT event insertion

## Validation Checks

| Check | Status | Implementation |
|---|---:|---|
| PF evolves independently | pass | `pf_dynamics_core.py` imports only `s_field_definition.py`, not CNT scan code. |
| CNT only used offline | pass | `cnt_scan_pipeline.py` writes calibration artifacts and is not used by PF core. |
| S(x) is the only spatial coupling variable | pass | Runtime beta driving uses `compute_s_field(...)`. |
| Nucleation is spatially resolved in PF | pass | `S(x)` is local and depends on GP field, GP gradient, and local composition. |
| Nucleus library used only for calibration | pass | Catalog schema requires `runtime_usable = false`. |

## What Changed Conceptually

Before:

```text
CNT / selector / template insertion could act as runtime nucleation machinery.
```

After:

```text
CNT calibrates S(x) and DeltaG*(T,xB,strain) offline.
Runtime PF evolves beta continuously through S-modulated driving force.
```

## Remaining CUDA Integration Step

The new files define the target architecture and reference implementation. The
CUDA runtime should next mirror `s_field_definition.py` and `pf_dynamics_core.py`
inside the existing kernels, while disabling scheduled/CNT/template insertion in
the default dual-track mode.
