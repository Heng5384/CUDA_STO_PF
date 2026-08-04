# AQ-to-6 h source identity audit

## Strict source separation

Yu 2024 Table S2 supplies the registered AQ two-population Debye--Callaway input.  Sheskin et al. 2018 (Tailoring) supplies independent measured transport and microstructure observations.  No Yu population, grain size, lattice parameter, or dislocation field is represented as a Tailoring measurement.

The Tailoring supporting information was not locally available.  Its main text states that the supporting material contains the raw temperature-dependent thermal-diffusivity and heat-capacity data; it does not provide a local thermal-conductivity table for this audit.  Thus the AQ/6 h thermal-conductivity values below are explicitly Figure 5c digitizations, not table values.

## Figure 5c digitization

`tailoring_figure5c_original.png` is a 600 dpi crop from page 5.  The calibration and marker coordinates are retained in the companion CSV files.  The lowest plotted marker is 30 degC = 303.15 K.  Therefore 300.00 K is a short 3.15 K extrapolation from the 30/50 degC segment; it is not a direct 300 K measurement.  The assigned 0.015 W m^-1 K^-1 uncertainty covers marker thickness, coordinate calibration and this short extrapolation only.

## Detection-scale rule

The Tailoring TEM/APT total density and the SE/FIB large-scale density are not summed.  They use different detection scales and could overlap in object identity.  The APT 37/43 and 6/43 count fractions are a single-reconstruction diagnostic, not a validated bulk two-population density.
