# CNT real-unit barrier postprocess

Definitions:
- `F_total_excess_hat`: PF/full-model excess energy already referenced to the matrix/bulk baseline inside `main_cuda`.
- `F_total_CNT_hat`: CNT diagnostic using far-field chemical-potential driving force; not reference-subtracted by itself.
- `DeltaF_excess_refsub_hat`: `F_total_excess_hat - F_total_ref_hat`; strict no-nucleus-reference-subtracted PF excess barrier.
- `DeltaF_CNT_refsub_hat`: `F_total_CNT_hat - F_total_ref_hat` using the matrix-only same-strain reference.

Conversion:
- `DeltaG_J = DeltaF_hat * w_phys * V_box`
- `w_phys = 12 * gamma / lambda_sm`
- `V_box = Nx * Ny * Nz * dx^3`
- `DeltaG_kBT = DeltaG_J / (k_B * T_K)`

Processed scan points: 605
Processed cases: 55

Outputs:
- `barrier_real_units_detail.csv`
- `barrier_real_units_by_case.csv`
