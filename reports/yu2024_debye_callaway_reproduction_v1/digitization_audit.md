# Digitization audit

## Experimental curves

No experimental curve digitization was needed. AQ and 48 h `kappa_lat` values come directly from SI Table S1, local SI PDF page 13. The files retain the requested `*_digitized.csv` names for interface compatibility, but every row is explicitly marked `TABLE_S1_NOT_DIGITIZED`. No smoothing, resampling, or modification was applied, and the digitization uncertainty field is zero because these are table values rather than pixel estimates.

## Published Yu model curves

Yu's calculated curves are available only as dashed lines in main-text Figure 6b. They were digitized independently as follows:

1. Main PDF page 8 was rendered with Poppler `pdftoppm` at 240 dpi to a 1985 by 2608 pixel RGB PNG.
2. The Figure 6b plot box was calibrated at:
   - x = 1101 px -> 275 K
   - x = 1564 px -> 600 K
   - y = 267 px -> 3.0 W m^-1 K^-1
   - y = 622 px -> 1.0 W m^-1 K^-1
3. The dark-blue dashed curves were isolated by the frozen RGB rule `90 < B < 150`, `R < 70`, `G < 70`.
4. Legend pixels were excluded by the plot-region mask.
5. Eighteen connected dashed segments were recovered for AQ and eighteen for 48 h.
6. Masked curve pixels were converted with the linear axis calibration. PCHIP was used only to evaluate the recovered model curve at the SI Table S1 temperatures; no extrapolation was used.
7. A vertical uncertainty of two rendered pixels gives `0.0112676 W m^-1 K^-1`.

The exact contract, source/image hashes, per-column masked curve pixels, and extracted values are frozen in:

- `data/qualification/yu2024_transport_v1/figure6b_digitization_contract.json`
- `data/qualification/yu2024_transport_v1/figure6b_model_curve_pixels.csv`
- `data/qualification/yu2024_transport_v1/yu_AQ_model_digitized.csv`
- `data/qualification/yu2024_transport_v1/yu_48h_model_digitized.csv`

## Axis-label caveat for Figures 6c and 6d

The paper labels the horizontal axis "Phonon Frequency (THz)" and extends it to approximately 18. With `Theta_D = 136 K`, the Debye angular frequency is approximately `17.8e12 rad/s`, while the cycles-per-second frequency is approximately `2.83 THz`. The plotted axis therefore appears to use `omega / 1e12 rad/s` while labelling it THz. Frequency-resolved outputs preserve both quantities and name the paper-axis convention explicitly.

## Visual QA

The rendered SI Table S1 page, SI equation pages 14-18, and main Figure 6 page were visually inspected. Equation placement, numerator/denominator structure, table values, curve identity, and calibration ticks were checked against the rendered pages rather than text extraction alone.
