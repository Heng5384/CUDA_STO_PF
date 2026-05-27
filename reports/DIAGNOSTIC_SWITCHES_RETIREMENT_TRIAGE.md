# Diagnostic Switches Retirement Triage

Rules used:

- `keep`: production-useful or core observability
- `archive`: audit/test-only, keep code but move the narrative to `archive/diagnostics/` later
- `delete`: pure debug residue, candidate for the next cleanup PR

| 旋钮 | 引入 STEP | 当前默认值 | 最后被启用的地方 | 报告结论 | 建议状态 |
|---|---|---:|---|---|---|
| `gp_nuc_enabled` | [reports/step_reports/STEP21_PRODUCTION_LIKE_PROJECTION_VALIDATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP21_PRODUCTION_LIKE_PROJECTION_VALIDATION_REPORT.md:60) | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:8596) | production GP nucleation control | keep |
| `gp_nuc_check_interval` | [reports/step_reports/STEP21_PRODUCTION_LIKE_PROJECTION_VALIDATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP21_PRODUCTION_LIKE_PROJECTION_VALIDATION_REPORT.md:67) | `10` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1206) | production GP nucleation control | keep |
| `gp_nuc_phi_threshold` | legacy | `0.05` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:8635) | production GP nucleation control | keep |
| `gp_nuc_eta_threshold` | legacy | `0.05` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:8636) | production GP nucleation control | keep |
| `gp_nuc_h_alpha_threshold` | legacy | `0.95` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:8634) | production GP nucleation control | keep |
| `gp_nuc_J0` | [reports/step_reports/STEP21_PRODUCTION_LIKE_PROJECTION_VALIDATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP21_PRODUCTION_LIKE_PROJECTION_VALIDATION_REPORT.md:182) | `1.0e12` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:8618) | production GP nucleation control | keep |
| `gp_nuc_gamma` | [reports/step_reports/STEP24_EVENT_THROTTLING_CALIBRATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP24_EVENT_THROTTLING_CALIBRATION_REPORT.md:43) | `0.1` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:8617) | production GP nucleation control | keep |
| `gp_nuc_seed_radius` | legacy | `0.10` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1220) | production GP nucleation control | keep |
| `gp_nuc_seed_peak` | legacy | `0.05` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1220) | production GP nucleation control | keep |
| `gp_nuc_seed_iface_width` | legacy | `0.10` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:8575) | production GP nucleation control | keep |
| `gp_nuc_patch_radius` | legacy | `0.35` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1210) | production GP nucleation control | keep |
| `gp_nuc_shell_inner_radius` | legacy | `0.15` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1214) | production GP nucleation control | keep |
| `gp_nuc_shell_outer_radius` | legacy | `0.35` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1215) | production GP nucleation control | keep |
| `gp_nuc_max_events_per_step` | legacy | `1` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:8598) | production GP nucleation control | keep |
| `gp_nuc_mass_mode` | legacy | "local_compensate" | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1153) | production GP nucleation control | keep |
| `gp_to_beta_enabled` | [reports/step_reports/STEP21_PRODUCTION_LIKE_PROJECTION_VALIDATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP21_PRODUCTION_LIKE_PROJECTION_VALIDATION_REPORT.md:62) | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:8901) | production GP->beta conversion control | keep |
| `gp_to_beta_check_interval` | [reports/step_reports/STEP24_EVENT_THROTTLING_CALIBRATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP24_EVENT_THROTTLING_CALIBRATION_REPORT.md:45) | `10` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1202) | production GP->beta conversion control | keep |
| `gp_to_beta_eta_threshold` | [reports/step_reports/STEP24_EVENT_THROTTLING_CALIBRATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP24_EVENT_THROTTLING_CALIBRATION_REPORT.md:198) | `0.5` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:8586) | production GP->beta conversion control | keep |
| `gp_to_beta_radius_threshold` | legacy | `0.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:8587) | production GP->beta conversion control | keep |
| `gp_to_beta_xB_threshold` | legacy | `0.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:8585) | production GP->beta conversion control | keep |
| `gp_to_beta_seed_radius` | legacy | `0.15` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1234) | production GP->beta conversion control | keep |
| `gp_to_beta_seed_peak` | legacy | `1.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1234) | production GP->beta conversion control | keep |
| `gp_to_beta_seed_iface_width` | legacy | `0.10` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:8569) | production GP->beta conversion control | keep |
| `gp_to_beta_patch_radius` | legacy | `0.40` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1224) | production GP->beta conversion control | keep |
| `gp_to_beta_shell_inner_radius` | legacy | `0.20` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1228) | production GP->beta conversion control | keep |
| `gp_to_beta_shell_outer_radius` | legacy | `0.40` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1229) | production GP->beta conversion control | keep |
| `gp_to_beta_max_events_per_step` | legacy | `1` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:8903) | production GP->beta conversion control | keep |
| `gp_to_beta_stochastic_enabled` | [reports/step_reports/STEP24_EVENT_THROTTLING_CALIBRATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP24_EVENT_THROTTLING_CALIBRATION_REPORT.md:46) | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:8970) | production GP->beta conversion control | keep |
| `gp_to_beta_J0_site` | [reports/step_reports/STEP24_EVENT_THROTTLING_CALIBRATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP24_EVENT_THROTTLING_CALIBRATION_REPORT.md:47) | `1.0e30` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:8995) | production GP->beta conversion control | keep |
| `gp_to_beta_gamma` | [reports/step_reports/STEP24_EVENT_THROTTLING_CALIBRATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP24_EVENT_THROTTLING_CALIBRATION_REPORT.md:48) | `0.05` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:8991) | production GP->beta conversion control | keep |
| `gp_to_beta_drive_const` | [reports/step_reports/STEP24_EVENT_THROTTLING_CALIBRATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP24_EVENT_THROTTLING_CALIBRATION_REPORT.md:50) | `1.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:8982) | production GP->beta conversion control | keep |
| `gp_to_beta_max_events_per_check` | legacy | `1` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1238) | production GP->beta conversion control | keep |
| `gp_to_beta_barrier_mode` | legacy | "cnt_simple" | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1161) | production GP->beta conversion control | keep |
| `gp_to_beta_drive_mode` | [reports/step_reports/STEP24_EVENT_THROTTLING_CALIBRATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP24_EVENT_THROTTLING_CALIBRATION_REPORT.md:49) | "local_simple" | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1165) | production GP->beta conversion control | keep |
| `gp_to_beta_mass_mode` | legacy | "report" | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1157) | production GP->beta conversion control | keep |
| `gp_to_beta_eta_deplete_mode` | legacy | "multiply_1_minus_hphi_seed" | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1169) | production GP->beta conversion control | keep |
| `gp_to_beta_phi_insert_mode` | legacy | "max" | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1173) | production GP->beta conversion control | keep |
| `gp_to_beta_conversion_mass_audit_enabled` | [reports/step_reports/STEP17_CONVERSION_MASS_AUDIT_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP17_CONVERSION_MASS_AUDIT_REPORT.md:21) | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:9132) | conversion audit hook | archive |
| `gp_to_beta_stop_after_conversion_audit` | [reports/step_reports/STEP17_CONVERSION_MASS_AUDIT_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP17_CONVERSION_MASS_AUDIT_REPORT.md:22) | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:9677) | conversion audit hook | archive |
| `gp_to_beta_conversion_audit_prefix` | [reports/step_reports/STEP17_CONVERSION_MASS_AUDIT_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP17_CONVERSION_MASS_AUDIT_REPORT.md:23) | "gp_to_beta_conversion_mass_audit" | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:10203) | conversion audit hook | archive |
| `gp_to_beta_feasibility_gate_enabled` | [reports/step_reports/STEP22_FEASIBILITY_GATE_VALIDATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP22_FEASIBILITY_GATE_VALIDATION_REPORT.md:26) | `1` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:9373) | production GP->beta conversion control | keep |
| `gp_to_beta_min_shell_capacity_factor` | [reports/step_reports/STEP22_FEASIBILITY_GATE_VALIDATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP22_FEASIBILITY_GATE_VALIDATION_REPORT.md:27) | `1.05` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1242) | production GP->beta conversion control | keep |
| `gp_to_beta_reject_if_infeasible` | [reports/step_reports/STEP22_FEASIBILITY_GATE_VALIDATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP22_FEASIBILITY_GATE_VALIDATION_REPORT.md:28) | `1` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:9413) | production GP->beta conversion control | keep |
| `gp_to_beta_allow_seed_amplitude_scaling` | [reports/step_reports/STEP22_FEASIBILITY_GATE_VALIDATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP22_FEASIBILITY_GATE_VALIDATION_REPORT.md:29) | `1` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:9377) | production GP->beta conversion control | keep |
| `gp_to_beta_min_seed_amplitude` | [reports/step_reports/STEP22_FEASIBILITY_GATE_VALIDATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP22_FEASIBILITY_GATE_VALIDATION_REPORT.md:30) | `0.05` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1246) | production GP->beta conversion control | keep |
| `gp_to_beta_event_cooldown_steps` | [reports/step_reports/STEP22_FEASIBILITY_GATE_VALIDATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP22_FEASIBILITY_GATE_VALIDATION_REPORT.md:31) | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1251) | production GP->beta conversion control | keep |
| `gp_to_beta_min_event_spacing` | [reports/step_reports/STEP22_FEASIBILITY_GATE_VALIDATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP22_FEASIBILITY_GATE_VALIDATION_REPORT.md:32) | `0.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1255) | production GP->beta conversion control | keep |
| `gp_to_beta_event_exclusion_radius` | [reports/step_reports/STEP23_EVENT_THROTTLING_VALIDATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP23_EVENT_THROTTLING_VALIDATION_REPORT.md:31) | `0.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1259) | production GP->beta conversion control | keep |
| `gp_to_beta_max_events_global` | [reports/step_reports/STEP23_EVENT_THROTTLING_VALIDATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP23_EVENT_THROTTLING_VALIDATION_REPORT.md:32) | `1000000000` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1263) | production GP->beta conversion control | keep |
| `gp_to_beta_max_events_per_window` | [reports/step_reports/STEP23_EVENT_THROTTLING_VALIDATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP23_EVENT_THROTTLING_VALIDATION_REPORT.md:33) | `1000000000` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1267) | production GP->beta conversion control | keep |
| `gp_to_beta_event_window_steps` | [reports/step_reports/STEP23_EVENT_THROTTLING_VALIDATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP23_EVENT_THROTTLING_VALIDATION_REPORT.md:34) | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1271) | production GP->beta conversion control | keep |
| `post_conversion_y_update_audit_enabled` | [reports/step_reports/STEP18_POST_CONVERSION_Y_AUDIT_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP18_POST_CONVERSION_Y_AUDIT_REPORT.md:18) | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:9974) | validated audit hook | archive |
| `post_conversion_y_update_audit_steps` | [reports/step_reports/STEP18_POST_CONVERSION_Y_AUDIT_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP18_POST_CONVERSION_Y_AUDIT_REPORT.md:19) | `5` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:9975) | validated audit hook | archive |
| `post_conversion_y_update_audit_prefix` | [reports/step_reports/STEP18_POST_CONVERSION_Y_AUDIT_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP18_POST_CONVERSION_Y_AUDIT_REPORT.md:20) | "post_conversion_y_update_audit" | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:10209) | validated audit hook | archive |
| `y_update_k0_audit_enabled` | [reports/step_reports/STEP19_Y_UPDATE_K0_AUDIT_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP19_Y_UPDATE_K0_AUDIT_REPORT.md:19) | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:9976) | validated audit hook | archive |
| `y_update_k0_audit_steps` | [reports/step_reports/STEP19_Y_UPDATE_K0_AUDIT_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP19_Y_UPDATE_K0_AUDIT_REPORT.md:20) | `5` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1275) | validated audit hook | archive |
| `y_update_k0_audit_prefix` | [reports/step_reports/STEP19_Y_UPDATE_K0_AUDIT_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP19_Y_UPDATE_K0_AUDIT_REPORT.md:21) | "y_update_k0_audit" | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:10215) | validated audit hook | archive |
| `y_update_mass_projection_enabled` | [reports/step_reports/STEP19_Y_UPDATE_K0_AUDIT_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP19_Y_UPDATE_K0_AUDIT_REPORT.md:22) | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:9978) | projection audit / correction hook | archive |
| `y_update_mass_projection_report_enabled` | [reports/step_reports/STEP20_Y_MASS_PROJECTION_VALIDATION_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP20_Y_MASS_PROJECTION_VALIDATION_REPORT.md:36) | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1279) | projection audit / correction hook | archive |
| `y_update_mass_projection_max_iter` | [reports/step_reports/STEP19_Y_UPDATE_K0_AUDIT_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP19_Y_UPDATE_K0_AUDIT_REPORT.md:23) | `30` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1284) | projection audit / correction hook | archive |
| `y_update_mass_projection_tol` | [reports/step_reports/STEP19_Y_UPDATE_K0_AUDIT_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP19_Y_UPDATE_K0_AUDIT_REPORT.md:24) | `1.0e-12` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1288) | projection audit / correction hook | archive |
| `y_update_mass_projection_target_mode` | [reports/step_reports/STEP19_Y_UPDATE_K0_AUDIT_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP19_Y_UPDATE_K0_AUDIT_REPORT.md:25) | "pre_Y_update" | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1197) | projection audit / correction hook | archive |
| `gp_obs_target_radius_nm` | legacy | `1.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1293) | production observed-GP init control | keep |
| `gp_obs_eta_peak` | legacy | `0.2` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1297) | production observed-GP init control | keep |
| `gp_obs_iface_width_nm` | [reports/step_reports/STEP37_GP_ETA_RHS_PHYSICS_AUDIT_REPORT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP37_GP_ETA_RHS_PHYSICS_AUDIT_REPORT.md:11) | `0.2` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1301) | production observed-GP init control | keep |
| `gp_obs_profile_type` | legacy | "tanh" | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1133) | production observed-GP init control | keep |
| `gp_obs_match_mode` | legacy | "match_integral_h_volume" | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1137) | production observed-GP init control | keep |
| `gp_obs_compensation_mode` | legacy | "smooth_radial_depletion" | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1141) | production observed-GP init control | keep |
| `gp_obs_depletion_radius_factor` | legacy | `3.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1305) | production observed-GP init control | keep |
| `gp_obs_depletion_smooth_width_factor` | legacy | `0.5` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1309) | production observed-GP init control | keep |
| `gp_obs_min_xB_alpha` | legacy | `0.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1313) | production observed-GP init control | keep |
| `gp_obs_max_xB_alpha` | legacy | `1.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1317) | production observed-GP init control | keep |
| `gp_raw_reaction_drive_only` | [STEP38G_DGBULK_DETA_AUDIT.md](/Users/heng/Documents/GitHub/CUDA_STO_PF/reports/step_reports/STEP38G_DGBULK_DETA_AUDIT.md:16) | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:9995) | one-off raw-drive study / cleanup candidate | archive |
| `gp_debug_freeze_eta` | legacy | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:10020) | legacy debug residue | delete |
| `gp_debug_zero_divJ` | legacy | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:10021) | legacy debug residue | delete |
| `gp_debug_flip_flux_sign` | legacy | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:10022) | legacy debug residue | delete |
| `gp_debug_disable_storage_exact_clip` | legacy | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:10023) | legacy debug residue | delete |
| `gp_debug_mu_mode` | legacy | "full" | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1189) | legacy debug residue | delete |
| `gp_debug_mobility_mode` | legacy | "full" | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:79) | legacy debug residue | delete |
| `gp_debug_xB_ref` | legacy | `0.03` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:10024) | legacy debug residue | delete |
| `gp_debug_mu_linear_slope` | legacy | `0.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:10025) | legacy debug residue | delete |
| `gp_debug_mobility_scale` | legacy | `1.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:80) | legacy debug residue | delete |
| `gp_debug_constant_mobility` | legacy | `1.0e-3` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:81) | legacy debug residue | delete |
| `diag_vtk_enabled` | legacy | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:10087) | production observability / diagnostics | keep |
| `diag_elastic_bulk_penalty_enabled` | legacy | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:10088) | production observability / diagnostics | keep |
| `minimize_full_model` | legacy | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:1801) | production minimize-mode control | keep |
| `minimize_max_iter` | legacy | `2000` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:11446) | production minimize-mode control | keep |
| `minimize_dt` | legacy | `P->dt` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:11447) | production minimize-mode control | keep |
| `minimize_V0` | legacy | `0.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:11869) | production minimize-mode control | keep |
| `minimize_resample_elastic_every` | legacy | `100` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:4970) | production minimize-mode control | keep |
| `minimize_rms_dphi_threshold` | legacy | `1.0e-6` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:11870) | production minimize-mode control | keep |
| `minimize_rms_dY_threshold` | legacy | `5.0e-5` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:11871) | production minimize-mode control | keep |
| `minimize_energy_diff_rel_threshold` | legacy | `1.0e-9` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:11872) | production minimize-mode control | keep |
| `minimize_rms_res_for_energy_plateau` | legacy | `1.0e-4` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:4975) | production minimize-mode control | keep |
| `minimize_rms_res_threshold` | legacy | `1.0e-4` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:11873) | production minimize-mode control | keep |
| `minimize_vol_err_rel_threshold` | legacy | `1.0e-4` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:11874) | production minimize-mode control | keep |
| `minimize_convergence_steps` | legacy | `10` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:11875) | production minimize-mode control | keep |
| `minimize_dt_safety_limit` | legacy | `1.0e-6` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:4979) | production minimize-mode control | keep |
| `minimize_xB_max_safe` | legacy | `0.07` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:2052) | production minimize-mode control | keep |
| `minimize_post_projection_iters` | legacy | `1` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:11877) | production minimize-mode control | keep |
| `minimize_continue_from_vtk` | legacy | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:11449) | production minimize-mode control | keep |
| `scheduled_nuc_enabled` | legacy | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3785) | legacy test harness | archive |
| `scheduled_nuc_source_dyn_dir` | legacy | `<unset>` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3647) | legacy test harness | archive |
| `scheduled_nuc_profile_dir` | legacy | `<unset>` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:10620) | legacy test harness | archive |
| `scheduled_nuc_source_step` | legacy | "latest" | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:10625) | legacy test harness | archive |
| `scheduled_nuc_source_phi_vtk` | legacy | `<unset>` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:10630) | legacy test harness | archive |
| `scheduled_nuc_source_xB_vtk` | legacy | `<unset>` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:10635) | legacy test harness | archive |
| `scheduled_nuc_steps_csv` | legacy | `<unset>` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:10645) | legacy test harness | archive |
| `scheduled_nuc_centers_nm` | legacy | `<unset>` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:10650) | legacy test harness | archive |
| `scheduled_nuc_source_case_label` | legacy | "T400_xB0p030_no_strain" | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:4011) | legacy test harness | archive |
| `scheduled_nuc_xB_edge_mode` | legacy | "sample-current-background-shell" | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:10655) | legacy test harness | archive |
| `scheduled_nuc_edge_sample_stat` | legacy | "mean" | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3861) | legacy test harness | archive |
| `scheduled_nuc_edge_sample_inner_nm` | legacy | `0.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3848) | legacy test harness | archive |
| `scheduled_nuc_edge_sample_outer_nm` | legacy | `3.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3809) | legacy test harness | archive |
| `scheduled_nuc_local_comp_inner_nm` | legacy | `0.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3931) | legacy test harness | archive |
| `scheduled_nuc_local_comp_outer_nm` | legacy | `20.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3932) | legacy test harness | archive |
| `scheduled_nuc_local_comp_taper_nm` | legacy | `5.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3934) | legacy test harness | archive |
| `scheduled_nuc_source_dx_nm` | legacy | `0.1` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3807) | legacy test harness | archive |
| `scheduled_nuc_source_lambda_nm` | legacy | `0.6` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3812) | legacy test harness | archive |
| `scheduled_nuc_target_lambda_nm` | legacy | `4.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3812) | legacy test harness | archive |
| `scheduled_nuc_scale_geometry` | legacy | `1.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3816) | legacy test harness | archive |
| `scheduled_nuc_scale_interface_width` | legacy | `0.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3810) | legacy test harness | archive |
| `scheduled_nuc_scale_xB_profile_width` | legacy | `0.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3813) | legacy test harness | archive |
| `scheduled_nuc_alpha_interface` | legacy | `0.25` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3900) | legacy test harness | archive |
| `scheduled_nuc_xB_min` | legacy | `1.0e-8` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3968) | legacy test harness | archive |
| `scheduled_nuc_xB_max` | legacy | `0.035` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3969) | legacy test harness | archive |
| `scheduled_nuc_phi_matrix_threshold` | legacy | `0.05` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3903) | legacy test harness | archive |
| `scheduled_nuc_W_comp_threshold` | legacy | `1.0e-3` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3925) | legacy test harness | archive |
| `scheduled_nuc_mass_iters` | legacy | `10` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3950) | legacy test harness | archive |
| `scheduled_nuc_mass_tol` | legacy | `1.0e-9` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:3975) | legacy test harness | archive |
| `scheduled_nuc_write_event_vtk` | legacy | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:4038) | legacy test harness | archive |
| `scheduled_nuc_fallback_analytic_sphere` | legacy | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:11478) | legacy test harness | archive |
| `dynamics_mass_diag_enabled` | legacy | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:11835) | production observability / diagnostics | keep |
| `dynamics_mass_diag_interval` | legacy | `1` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:11836) | production observability / diagnostics | keep |
| `enable_Y_rhs_previous_time_level` | legacy | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:11838) | Y RHS audit / convergence study knob | archive |
| `disable_Y_rhs_gamma_term` | legacy | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:14779) | Y RHS audit / convergence study knob | archive |
| `Y_rhs_term_h_scale` | legacy | `1.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:14798) | Y RHS audit / convergence study knob | archive |
| `enable_Y_rhs_picard` | legacy | `0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:13306) | Y RHS audit / convergence study knob | archive |
| `Y_rhs_picard_iters` | legacy | `1` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:2497) | Y RHS audit / convergence study knob | archive |
| `Y_rhs_picard_omega` | legacy | `1.0` | [main_cuda.cu](/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:2497) | Y RHS audit / convergence study knob | archive |

## Totals

- keep: 78
- archive: 52
- delete: 10
- total: 140

## 不在 triage 范围

以下家族明确排除在本 triage 之外，因为它们是核心物理参数、单位换算量、PDE 系数或 CLI-only / 常规运行控制，而不是诊断旋钮：

- `W`, `kappa_phi`, `L_phi`, `W_eta`, `kappa_eta`, `L_eta`, `D_alpha`
- `dx`, `dy`, `dz`, `dt`, `t_real_unit`, `temperature_C`, `lambda_sm`, `mu_reference_scale`, `Vm_*`
- `gp_gamma_alpha_gp`, `gp_l_eta_nm`, `gp_D_ratio`, `gp_M_int_eta`, `gp_M_GP`, `gp_M_beta`, `gp_eps_iso`
- `ic_*`, `eps_*`, `S_*`, `S_p_*`, `E0_*`
- 常规网格与输出控制：`Nx`, `Ny`, `Nz`, `dimension`, `nsteps`, `out_every`, `csv_out_every`, `seed`, `model_mode`
- CLI / workflow-only flags 与路径参数

## Notes

- 本页只统计 `pf_params.h` 中的诊断 / 调试 / 审计 / mode-control 字段，合计 140 项。
- `keep` 条目全部独立列出；`archive` / `delete` 也逐项列出，方便后续清理。
