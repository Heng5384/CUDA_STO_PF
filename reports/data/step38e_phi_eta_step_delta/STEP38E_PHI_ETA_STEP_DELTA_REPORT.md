# Phi vs Eta Per-Step Update Diagnostic

- `gp_eta_kinetic_scan_f1em5`: `max_dphi_abs_max=3.694087e-04`, `max_deta_abs_max=1.314270e-03`, `max_dxB_abs_max=8.185807e-05`, `max_xBtot_rel_delta=9.463931e-06`, `first_step_xB_clip=-1`, `first_step_eta_far_gt_0p05=-1`, `recommendation=baseline_candidate_if_xB_range_remains_stable`
- `gp_eta_kinetic_scan_f1em4`: `max_dphi_abs_max=3.694087e-04`, `max_deta_abs_max=2.097865e-03`, `max_dxB_abs_max=3.181379e-04`, `max_xBtot_rel_delta=3.927641e-05`, `first_step_xB_clip=-1`, `first_step_eta_far_gt_0p05=-1`, `recommendation=aggressive_compare_against_baseline`
- `gp_eta_kinetic_scan_f1em3`: `max_dphi_abs_max=3.694087e-04`, `max_deta_abs_max=1.995276e-02`, `max_dxB_abs_max=1.201678e-03`, `max_xBtot_rel_delta=5.692455e-04`, `first_step_xB_clip=-1`, `first_step_eta_far_gt_0p05=-1`, `recommendation=needs_review`

## Comparison
- baseline `f_eta=1e-5`: `max_deta_abs_max=1.314270e-03`, `max_eta_far_field=7.364600e-04`, `first_step_xB_clip=-1`
- aggressive `f_eta=1e-4`: `max_deta_abs_max=2.097865e-03`, `max_eta_far_field=5.194207e-03`, `first_step_xB_clip=-1`
- unstable probe `f_eta=1e-3`: `first_step_xB_clip=-1`, `max_deta_abs_max=1.995276e-02`
