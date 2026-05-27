# Step 31B: eta=1.0 Box / Depletion Validation

This report is populated after running `run_step31b_eta1_box_depletion_validation.sh`.

Diagnostic matrix:

- `96^3`, `R_GP = 1 nm`, `eta_peak = 1.0`, `dep = 2`
- `96^3`, `R_GP = 1 nm`, `eta_peak = 1.0`, `dep = 3`
- `128^3`, `R_GP = 1 nm`, `eta_peak = 1.0`, `dep = 5`
- `192^3`, `R_GP = 1 nm`, `eta_peak = 1.0`, `dep = 5`

Key questions:

1. Was the earlier `dep=5`, `96^3` failure contaminated by `R_dep > L/2` periodic overlap?
2. Does reducing `R_dep` below `L/2` improve stability at fixed box size?
3. Does increasing box size improve `dep=5` stability?
4. Can `eta_peak = 1.0` be fairly judged yet?

Interpretation rule:

Do not conclude `eta_peak = 1.0` is physically impossible unless:

- `R_dep < L/2`
- initialization gradients are reasonable
- larger-box comparison still fails
- failure is not dominated by periodic-image / compensation artifacts
