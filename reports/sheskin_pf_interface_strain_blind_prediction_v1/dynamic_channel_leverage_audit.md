# Dynamic-channel leverage audit

This audit was completed before any 6 h dynamic-parameter calibration.

- Required effective-resistance ratio: `0.489063017` (magnitude diagnostic only; not substituted for the Debye integral).
- True periodic interface-area ratio, `Sv_48/Sv_6`: `0.395306476` -> `STRONG_LEVERAGE`.
- Hydrostatic mid-q power ratio: `0.306247438`.
- Hydrostatic high-q power ratio: `0.45833842`.
- Hydrostatic total-variance ratio: `1.28336297`.
- Strain leverage: `STRONG_LEVERAGE`.
- Spectral-redistribution flag: `TRUE`.

The total hydrostatic variance grows, but the registered mid/high-q power falls by more than 40% in every replicate. Therefore mean elastic energy or total strain variance would give the wrong mechanism diagnostic; the formal strain model must use the q-resolved spectrum.
