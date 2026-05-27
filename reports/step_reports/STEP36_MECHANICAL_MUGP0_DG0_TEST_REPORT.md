# Step 36 Mechanical mu_GP0, Delta_g_stab = 0 Test

## Sanity
- `mu_GP0_mechanical_mixture = -153529.2329 J/mol`
- `g_alpha_raw(0.35, 653.15 K) = -150351.3950 J/mol`
- `Delta_g_stab = 0.0 J/mol`
- `minus_Delta_mu_r_GP(xB=0.03) = 2934.3703 J/mol`
- Matrix diffusion thermodynamics branch: `raw_regular_solution`
- Mechanical-mixture reference differs from old `g_alpha_raw(0.35)` by `-3177.8379 J/mol`

## Test A: embryo-to-mature
- `V_h_init_nm3 = 0.523596973933323`
- `V_h_final_nm3 = 0.5472549699742818`
- `R_eff_h_init_nm = 0.49999942651156887`
- `R_eff_h_final_nm = 0.5074193621199743`
- `eta_max_init = 0.3999`
- `eta_max_final = 0.40024`
- `eta_integral_init = 1.0889466000000003`
- `eta_integral_final = 1.148058330000001`
- `xB_min_final = 0.00384`
- `xB_max_final = 0.0075`
- `total_relative_drift_final = -0.00034489082119478615`
- `max_abs_dt_divJ_overall = 0.00076695377532`
- `classification = embryo_not_matured`

## Test B: mature GP self-limiting
- `V_h_init_nm3 = 4.188790960736316`
- `V_h_final_nm3 = 45.59593567720144`
- `R_eff_h_init_nm = 1.0000000601565802`
- `R_eff_h_final_nm = 2.216218152082544`
- `eta_max_init = 0.99995`
- `eta_max_final = 1.0`
- `eta_integral_init = 4.500456730000002`
- `eta_integral_final = 51.58419334000002`
- `xB_min_final = 0.00254`
- `xB_max_final = 0.03327`
- `total_relative_drift_final = 0.0009221506125805539`
- `max_abs_dt_divJ_overall = 0.0010007804849`
- `relative_reff_change_5000_to_10000 = 0.8338085859264706`
- `classification = growth_like_non_saturated`

## Answers
1. With mechanical-mixture `mu_GP0` and `Delta_g_stab=0`, `xB=0.03` does support strong GP-related growth: `-Delta_mu_r_GP(xB=0.03)` is positive (`2934.4 J/mol`).
2. The `eta=0.4, R=0.5 nm` embryo does not mature toward a larger GP state.
3. The `eta=1, R=1 nm` mature GP is classified here as `growth_like_non_saturated`; under the `h(eta)` volume metric it grows from `R_eff≈1.000 nm` to `R_eff≈2.216 nm`, and the `5000 -> 10000` relative radius change is `0.834`.
4. The combined behavior is more precipitate-like than GP-zone-like: the embryo does not mature, while the mature seed keeps growing strongly rather than settling into a finite weakly saturated precursor size.
5. `Delta_g_stab=0` is not a cleanly reasonable GP-zone precursor setting under the present kinetics. It is too weak to turn the small embryo into a mature GP, but once a mature GP exists it is strong enough to drive continued non-saturated growth.
