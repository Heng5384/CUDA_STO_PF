# V3 AQ background selection

Status: `PASS_AQ_ONLY_BACKGROUND_SELECTION_WITH_PROVISIONAL_HOST_MEMBERS`.

Only the seven selected AQ measured-total points were loaded for this stage. The V3 background was refitted for every host/AQ-structure member; no V2 A2 number was copied into the fit and no 6 h/48 h value was available to model selection.

H-P0 is the frozen Yu scalar host with the V2 background removed. H-P1 is unavailable because no branch-resolved PbTe host conductivity contract is qualified. H-P2 uses the 303.2 K PbTe tensor along [100], [110], [111] and a deterministic angular mean, and every H-P2 result is labelled `PROVISIONAL_HOST_ENVELOPE` because temperature dependence and a branch-resolved host lifetime authority are absent.

Retained nonprobabilistic member count: `103` (Delta-AICc <= 2 within each host/AQ structure case).

| host | retained models | AQ MAPE range (%) |
|---|---|---:|
| H-P0_YU_SCALAR | B2_omega2 | 4.045--4.146 |
| H-P2_PBTE_303K_100 | B2_omega2, B4_omega4 | 3.657--3.751 |
| H-P2_PBTE_303K_110 | B1_constant, B4_omega4 | 3.750--3.979 |
| H-P2_PBTE_303K_111 | B1_constant, B4_omega4 | 3.735--4.058 |
| H-P2_PBTE_303K_ANGULAR_MEAN | B1_constant, B4_omega4 | 3.769--3.915 |

The envelope is not a probability distribution. B1/B2/B4 are empirical nonnegative rates; BU rescales the chosen host rate but is not a revised Yu A_N. B0 is a zero-background upper-control candidate. Poor absolute chi-square fit remains a warning even when AICc selects a relative winner.
