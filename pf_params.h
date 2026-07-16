#ifndef PF_PARAMS_H
#define PF_PARAMS_H

#define PF_PARAMS_SCHEMA_VERSION 2

typedef struct {
    int pf_params_schema_version;
    // 网格参数
    int Nx, Ny, Nz;
    double dx, dy, dz;
    double dt;
    // 物理时间尺度：dt=1.0 所对应的真实时间（例如：秒），用于 1D 监测输出
    double t_real_unit;
    int nsteps;
    int out_every;      // VTK文件输出间隔
    int csv_out_every;  // CSV文件输出间隔
    int dimension;
    unsigned long seed;
    char model_mode[32]; // "two_phase" (default) or "gp_zone"
    double gp_xB_fixed;
    double gp_delta_g0;
    double gp_delta_g_stab;
    // GP mechanical-mixture stabilization shift [J/mol]
    // default 0.0; calibrate against xB~0.03 experiment
    double gp_W_eta;
    double gp_kappa_eta;
    double gp_L_eta;
    double gp_W_eta_phys_input;
    double gp_kappa_eta_phys_input;
    double gp_L_eta_phys_input;
    double gp_W_eta_code_input;
    double gp_kappa_eta_code_input;
    double gp_L_eta_code_input;
    int gp_W_eta_legacy_specified;
    int gp_kappa_eta_legacy_specified;
    int gp_L_eta_legacy_specified;
    int gp_W_eta_phys_specified;
    int gp_kappa_eta_phys_specified;
    int gp_L_eta_phys_specified;
    int gp_W_eta_code_specified;
    int gp_kappa_eta_code_specified;
    int gp_L_eta_code_specified;
    int gp_W_eta_input_mode_resolved;
    int gp_kappa_eta_input_mode_resolved;
    double gp_gamma_alpha_gp;
    double gp_l_eta_nm;
    double gp_D_ratio;
    double gp_M_int_eta;
    double gp_eps_iso;
    double gp_M_GP;
    double gp_M_beta;
    int gp_barrier_only_mode;
    int enable_legacy_gp_storage_coupling;
    int gp_elastic_enabled;
    int gp_elastic_active_eta;
    int gp_elastic_active_phi;
    double gp_elastic_derivative_scale;
    int gp_nuc_enabled;
    int gp_nuc_check_interval;
    double gp_nuc_phi_threshold;
    double gp_nuc_eta_threshold;
    double gp_nuc_h_alpha_threshold;
    double gp_nuc_J0;
    double gp_nuc_gamma;
    double gp_nuc_seed_radius;
    double gp_nuc_seed_peak;
    double gp_nuc_seed_iface_width;
    double gp_nuc_patch_radius;
    double gp_nuc_shell_inner_radius;
    double gp_nuc_shell_outer_radius;
    int gp_nuc_max_events_per_step;
    char gp_nuc_mass_mode[32];
    int gp_to_beta_enabled;
    int gp_to_beta_check_interval;
    double gp_to_beta_eta_threshold;
    double gp_to_beta_radius_threshold;
    double gp_to_beta_xB_threshold;
    double gp_to_beta_seed_radius;
    double gp_to_beta_seed_peak;
    double gp_to_beta_seed_iface_width;
    double gp_to_beta_patch_radius;
    double gp_to_beta_shell_inner_radius;
    double gp_to_beta_shell_outer_radius;
    int gp_to_beta_max_events_per_step;
    int gp_to_beta_stochastic_enabled;
    double gp_to_beta_J0_site;
    double gp_to_beta_gamma;
    double gp_to_beta_drive_const;
    int gp_to_beta_max_events_per_check;
    char gp_to_beta_barrier_mode[32];
    char gp_to_beta_drive_mode[32];
    char gp_to_beta_mass_mode[32];
    char gp_to_beta_eta_deplete_mode[64];
    char gp_to_beta_phi_insert_mode[32];
    int gp_to_beta_conversion_mass_audit_enabled;
    int gp_to_beta_stop_after_conversion_audit;
    char gp_to_beta_conversion_audit_prefix[128];
    int gp_to_beta_feasibility_gate_enabled;
    double gp_to_beta_min_shell_capacity_factor;
    int gp_to_beta_reject_if_infeasible;
    int gp_to_beta_allow_seed_amplitude_scaling;
    double gp_to_beta_min_seed_amplitude;
    int gp_to_beta_event_cooldown_steps;
    double gp_to_beta_min_event_spacing;
    double gp_to_beta_event_exclusion_radius;
    int gp_to_beta_max_events_global;
    int gp_to_beta_max_events_per_window;
    int gp_to_beta_event_window_steps;

    // ============================================
    // GP-assisted beta nucleation debug mode
    // Explicit one-site, scheduled, mass-conserving path.
    // Defaults are inert; this does not use GP eta/free-energy.
    // ============================================
    int enable_gp_assisted_beta_nucleation;
    int gp_assisted_debug_scheduled;
    int gp_debug_scheduled_site_id;
    int gp_debug_scheduled_step;
    int gp_debug_site_ix;
    int gp_debug_site_iy;
    int gp_debug_site_iz;
    char gp_site_mode[32];
    int gp_n_sites;
    double gp_site_spacing;
    unsigned long gp_seed;
    char gp_initial_mass_mode[64];
    char gp_release_mode[64];
    char gp_release_kernel[64];
    char gp_site_file[4096];
    double gp_release_radius_nm;
    double gp_site_B_mass_equiv;
    double gp_site_S_factor;
    double gp_marker_core_radius_nm;
    double gp_marker_influence_radius_nm;
    double gp_depletion_radius_nm;
    double gp_xB_floor;
    char gp_depletion_kernel[64];
    double gp_debug_beta_seed_radius;
    double gp_debug_beta_seed_iface_width;
    double gp_debug_xB_min;
    double gp_debug_xB_max;
    int gp_debug_mass_ledger;
    int gp_event_log_enabled;
    int gp_stochastic_enabled;
    double gp_stochastic_k0;
    double gp_stochastic_S_GP;
    double gp_stochastic_deltaG_homo_kBT;
    double gp_stochastic_xB_sensitivity;
    int gp_literature_model_enabled;
    char gp_birth_model[64];
    double gp_literature_A_m5;
    double gp_literature_B_eff_J3_m6;
    double gp_literature_D0_m2_s;
    double gp_literature_Q_J_mol;
    double gp_literature_xAg_default;
    char gp_literature_xAg_mode[64];
    double gp_literature_L_alpha0_J_mol;
    double gp_literature_L_alpha1_J_mol_K;
    double gp_literature_a_PbTe_m;
    double gp_literature_xeq_guard;
    char gp_birth_candidate_volume_model[64];
    int gp_birth_dt_uses_physical_time;
    int gp_birth_max_events_per_step;
    int gp_birth_max_total_sites;
    char gp_birth_position_mode[64];
    unsigned long gp_birth_rng_seed;
    int gp_birth_connect_to_smooth_local_depletion;
    char gp_birth_inventory_policy[64];
    int gp_literature_birth_requires_post_Y_projection;
    int gp_birth_debug_force_single_event;
    int gp_birth_debug_force_step;
    char gp_birth_debug_force_position_mode[64];
    int gp_birth_debug_disable_poisson_randomness;
    int gp_birth_debug_max_events_total;
    int gp_birth_max_total_new_births_for_debug;
    int gp_birth_debug_stop_after_step;
    int gp_birth_debug_freeze_dynamics_after_birth;
    int gp_birth_debug_disable_CH_dynamics_after_birth;
    int gp_birth_debug_force_rebuild_Y_after_birth;
    int gp_post_birth_mass_probe_enabled;
    char gp_population_source[64];
    int gp_literature_JGP_override_enabled;
    double gp_literature_JGP_override_m3_s;
    int gp_smooth_depletion_enabled;
    int gp_static_marker_enabled;
    int gp_initial_population_enabled;
    char gp_initial_population_source[64];
    double gp_initial_xB_tot;
    double gp_initial_rho_m3;
    double gp_initial_xAg_far;
    double gp_initial_xAg_GP;
    char gp_initial_radius_distribution[64];
    double gp_initial_radius_mean_target_nm;
    double gp_initial_radius_std_nm;
    double gp_initial_radius_min_nm;
    double gp_initial_radius_max_nm;
    char gp_initial_radius_renormalization[64];
    char gp_initial_count_mode[64];
    char gp_initial_position_mode[64];
    double gp_initial_min_center_spacing_factor;
    unsigned long gp_initial_rng_seed;
    int gp_growth_enabled;
    int gp_radius_evolution_enabled;
    int gp_inventory_growth_enabled;
    int gp_beta_selection_enabled;
    char gp_overlap_saturation_mode[64];
    int enable_gp_runtime_library_nucleation;
    char gp_runtime_barrier_library_path[4096];
    char gp_runtime_nucleus_catalog_path[4096];
    char gp_runtime_barrier_mode[64];
    char gp_runtime_temperature_unit[8];
    char gp_runtime_s_gp_mode[32];
    double gp_runtime_s_gp_scalar;
    char gp_runtime_nucleation_mode[64];
    int gp_runtime_reject_invalid_barrier_cases;
    int gp_runtime_log_candidates;
    int gp_runtime_log_candidate_full_rows;
    int gp_runtime_log_candidate_summary;
    int gp_ranked_hazard_full_log_enabled;
    int gp_runtime_log_accepted_events;
    int gp_runtime_disable_scheduled_when_active;
    double gp_runtime_catalog_T_tol_C;
    double gp_runtime_catalog_xB_tol;
    int gp_runtime_catalog_strain_mode_strict;
    int gp_runtime_catalog_allow_fallback;
    int enable_dynamic_continue_bridge;
    char dynamic_continue_bridge_catalog_path[4096];
    char gp_runtime_bridge_missing_policy[64];
    double gp_runtime_min_rseed_over_dx;
    int gp_runtime_enable_delayed_insertion_queue;
    int gp_runtime_log_bridge_queue;
    int gp_runtime_allow_immediate_fallback_debug;
    int enable_runtime_nucleus_library;
    char gp_runtime_nucleus_library_path[4096];
    char gp_runtime_profile_cache_root[4096];
    int gp_runtime_force_first_selector_event;
    int gp_runtime_force_event_step;

    // ============================================
    // Physical CNT-like beta nucleation rate model
    // Runtime selector may use this instead of surrogate hazard.
    // ============================================
    char beta_rate_model[64];
    int beta_rate_use_physical_dt;
    int beta_rate_use_gp_barrier_modifier;
    char beta_rate_D_B_alpha_model[64];
    double beta_rate_D_B_alpha_D0_m2_s;
    double beta_rate_D_B_alpha_Q_J_mol;
    int beta_rate_D_B_alpha_use_xB_factor;
    double beta_rate_Omega_g_m3;
    char beta_rate_Omega_g_source[256];
    double beta_rate_Omega_site_m3;
    double beta_rate_N_site_m3;
    char beta_rate_site_model[64];
    char beta_rate_gp_capture_volume_model[64];
    char beta_rate_Z_type[64];
    char beta_rate_Z_r_fallback_mode[64];
    char beta_rate_Z_r_source[64];
    int beta_rate_Z_r_required;
    int beta_rate_Z_r_debug_fallback_enabled;
    int beta_rate_allow_runtime_Zn_from_Zr;
    int beta_rate_scale_Z_with_sGP;
    char beta_rate_deltaV_nuc_mode[64];
    double beta_rate_deltaV_nuc_m3;
    double beta_rate_debug_rate_multiplier;
    double beta_rate_phi_threshold;
    double beta_rate_xB_min;
    int beta_rate_transient_enabled;
    double beta_rate_tau_inc_s;
    int beta_debug_force_single_event;
    int beta_debug_force_step;
    char beta_debug_position_mode[64];
    char beta_debug_inventory_mode[64];
    double beta_debug_capacity_fraction;
    char beta_debug_draw_radius_mode[64];
    char beta_debug_draw_radius_list_nm[256];
    int beta_debug_do_not_reduce_requested_mass;
    double beta_debug_matrix_draw_radius_nm;
    char beta_debug_GP_capture_mode[64];
    double beta_debug_GP_capture_radius_nm;
    char beta_debug_GP_capture_consume_order[64];
    int beta_debug_max_events_total;
    char beta_handoff_policy[64];
    int beta_capacity_gate_enabled;
    double beta_capacity_gate_matrix_draw_radius_nm;
    char beta_capacity_gate_GP_capture_mode[64];
    double beta_capacity_gate_GP_capture_radius_nm;
    double beta_capacity_gate_max_reasonable_radius_nm;
    double beta_capacity_gate_allow_direct_if_capacity_ratio_ge;
    int beta_staged_conversion_enabled;
    char beta_staged_conversion_target[64];
    char beta_staged_conversion_initial_inventory_mode[64];
    char beta_staged_conversion_release_mode[64];
    int beta_staged_conversion_insert_when_capacity_reached;
    int beta_staged_conversion_max_subgrid_steps;
    double beta_staged_conversion_mass_tolerance_rel;
    int beta_staged_accumulation_enabled;
    int beta_staged_accumulation_interval_steps;
    double beta_staged_accumulation_GP_capture_radius_nm;
    double beta_staged_accumulation_matrix_draw_radius_nm;
    double beta_staged_accumulation_max_fraction_per_step;
    double beta_staged_accumulation_max_inventory_per_step;
    int beta_staged_insert_when_target_reached;
    int beta_staged_debug_accelerated_accumulation;
    double beta_staged_debug_accumulation_rate_multiplier;
    int beta_staged_debug_stop_after_resolved_insert;
    char resolved_handoff_xB_write_mode[64];

    // Diagnostic-only required-supply matrix-halo source engine.
    // This is not a GP thermodynamic release law: it conservatively moves
    // existing GP reservoir inventory into local matrix alpha storage near a
    // resolved beta seed to test PF-side seed stability versus xBcrit.
    int diagnostic_rsmd_enabled;
    double diagnostic_rsmd_T_only;
    double diagnostic_rsmd_xB_halo_target;
    double diagnostic_rsmd_R_exchange_nm;
    double diagnostic_rsmd_chi_rel;
    double diagnostic_rsmd_kernel_radius_dx;
    // `gp_centered_kernel` is the legacy source geometry.  The optional
    // `seed_interface_alpha_shell` mode is a diagnostic relay only: it moves
    // bounded eligible-GP inventory into the matrix-side shell of the seed.
    char diagnostic_rsmd_delivery_mode[64];
    double diagnostic_rsmd_interface_shell_width_nm;
    char diagnostic_rsmd_interface_shell_kernel[64];
    // A host-side RSMD source is an external Y/xB transaction.  When enabled,
    // reset the lagged dY/dt state after a nonzero source write so the next
    // PF step does not combine that new state with a pre-source derivative.
    int diagnostic_rsmd_reset_Y_history_after_source;
    int diagnostic_rsmd_history_restart_mode;  // 0=stale, 1=mutate history, 2=one-step RHS mask
    int diagnostic_rsmd_interface_diag_enabled;
    int diagnostic_rsmd_interface_diag_every;
    double diagnostic_rsmd_h_src_max;
    double diagnostic_rsmd_f_max_per_step;
    // Numerical scenario-coupling controls; these are not GP kinetics.
    char diagnostic_rsmd_operator_split[32];
    char diagnostic_rsmd_source_integrator[32];
    double diagnostic_rsmd_source_substep_dt_code;
    int diagnostic_rsmd_headroom_weighted;
    char diagnostic_rsmd_control_mode[40];
    // Diagnostic-only PF baseline operator decomposition after a resolved seed exists.
    // full | frozen_phi | transport_no_projection | projection_only | phi_only
    char pf_baseline_control_mode[40];
    // Numerical composition update. The default retains the historical solver;
    // storage_exact advances conserved two-phase storage in the same step.
    char pf_y_update_mode[40];  // lagged_rhs | storage_exact | x_transport_projection_split | q_transport_projection_split
    // PF-only conservative composition architecture. "legacy" leaves the
    // historical Y/x/q paths untouched. The conservative modes own their
    // storage coordinate and do not use physical global Y projection.
    char pf_composition_mode[48];  // legacy | ctot_conservative_split | qalpha_conservative_local_transaction
    char pf_conservative_flux_strategy[32];  // pairwise_limited | pairwise_backward_euler
    double pf_conservative_bound_tol;
    double pf_conservative_mass_tol;
    double pf_conservative_beta_support_eps;
    int pf_conservative_max_subcycles;
    int pf_conservative_one_step_replay;
    // Production Ctot candidates. This selector is independent of legacy L/X/Q
    // and the earlier conservative diagnostic selectors above.
    char composition_evolution_mode[48];  // legacy_lagged_y | ctot_mimetic_be | ctot_fv_be | ctot_spectral_be(test-only)
    // Default-off multifidelity research contract.  These fields alter no
    // legacy path unless PF_RESEARCH_MODEL selects the coarse4 model.
    char PF_RESEARCH_MODEL[96];
    // Physics and time-integration contracts are intentionally independent.
    // "legacy" preserves every existing Ctot path. The two versioned values
    // below are default-off production-research candidates.
    char ctot_numerics_contract[96];
    char ctot_split_defect_policy[40];  // OFF | polish policies | LIE/IMEX no-polish policies
    int ctot_max_coupling_correctors;
    double ctot_split_defect_skip_threshold;
    double ctot_split_defect_hard_cap;
    double ctot_split_defect_scale;
    char PHASE_KINETICS_MODE[64];  // FINITE_LPHI_BE | QUASI_EQUILIBRIUM_FAST_INTERFACE_V1
    char coarse_interface_mobility_mode[64];  // off | INTERFACE_BAND_BOOST_V1
    double coarse_interface_mobility_a_M;
    char GP_population_mode[64];  // OFF | FIXED_POPULATION_DEPLETION_ONLY
    // Checkpoint provenance for the coarse surrogate.  The three hashes and
    // uncertainty version are mandatory when the coarse4 model is selected.
    char coarse_model_name[96];
    char coarse_model_version[32];
    char fine_reference_hash[65];
    char coarse_calibration_hash[65];
    char coarse_uncertainty_version[64];
    // Coarse4-only mechanical precision/acceptance provenance.  The legacy
    // solver and gate remain selected by default; the FP32-aware contract is
    // admitted only after its external qualification evidence is frozen.
    char mechanics_precision_mode[48];
    char mechanics_acceptance_mode[64];
    char eta_floor_version[64];
    double eta_accept;
    char double_oracle_contract_hash[65];
    char residual_normalization_version[64];
    // Default-off small-grid evidence capture.  This never changes a solve;
    // it only records the first completed mechanics state of a run.
    int mechanics_fp32_diagnostics_enabled;
    int mechanics_fp32_dump_first_solve_fields;
    int mechanics_fp32_diagnostic_solve_index;
    int ctot_nonlinear_max_iter;
    double ctot_residual_abs_tol;
    double ctot_residual_rel_tol;
    double ctot_line_search_min;
    // Nonlinear coordinate only; the conservative FV operator is unchanged.
    // legacy_logit_newton is the historical default. The adaptive mode starts
    // from that coordinate and switches to authoritative Ctot only when the
    // logit line search fails; it never clips physical mass.
    char ctot_transport_nonlinear_coordinate[48];
    // Default-off outer fixed-point acceleration. Raw x/Y are never mixed.
    char ctot_outer_acceleration[32];
    // Acceptance/reporting policy only.  The default preserves the historical
    // zero-reject qualification; the bounded-retry policy never changes a
    // physical operator, nonlinear tolerance, or retry algorithm.
    char ctot_retry_acceptance_contract[64];
    int ctot_step_max_retries;
    double ctot_retry_shrink_factor;
    double ctot_dt_min_ratio;
    int ctot_automatic_dt_growth;
    int ctot_debug_force_first_attempt_reject;
    // Default-off rollback oracle for an attempt that selected BDF2.
    int ctot_debug_force_first_bdf2_attempt_reject;
    // Default-off active-set crossing guard and atomic Lie-BE event fallback.
    // These flags alter only the time integrator transaction around a detected
    // nonsmooth capacity event; all physical operators and tolerances remain
    // unchanged.
    int bdf2_event_preflight_v1;
    int bdf2_event_be_subcycling_v1;
    // Default-off tiny-grid residual-floor archaeology. This diagnostic may
    // replay fixed-phi transport solves but never changes the accepted state.
    int ctot_debug_transport_floor_audit;
    int ctot_elastic_validation_enabled;
    int ctot_debug_force_elastic_post_phi_reject;
    double ctot_matrix_support_eps;
    int ctot_phase_constraint_enabled;
    int ctot_phase_semismooth_pdas_enabled;
    int ctot_phase_restart_solver_migration_allowed;
    int ctot_phase_linear_max_iter;
    double ctot_phase_linear_rel_tol;
    double ctot_phase_fd_rel_step;
    int ctot_diagnostics_enabled;
    // Engineering-only P0/P1 profiler. Default-off and excluded from the
    // accepted-state/checkpoint numerical contract.
    int ctot_performance_profile_enabled;
    int ctot_initialization_dry_run;
    int ctot_phase_only_dry_run;
    double ctot_preconditioner_a_ref;
    double ctot_preconditioner_D_ref_multiplier;
    int ctot_spectral_fv_warm_start_diagnostic;
    int ctot_nonadjoint_hybrid_test_only;
    int ctot_finite_interface_antitrapping_enabled;
    double finite_interface_calibration_min_points;
    double finite_interface_calibration_max_points;
    double finite_interface_production_min_points;
    int finite_interface_resolution_test_override;
    int finite_interface_violation_diagnostics_enabled;
    double ctot_solver_k0_max_abs_shift;
    double ctot_energy_rel_tol;
    double ctot_energy_balance_rel_tol;
    double ctot_energy_abs_tol;
    int ctot_outer_max_iter;
    double ctot_outer_rel_tol;
    double ctot_outer_abs_tol;
    double ctot_outer_C_scale;
    double ctot_outer_phi_scale;
    double ctot_outer_sigma_scale;
    double ctot_outer_displacement_scale;
    double pf_matrix_storage_floor;
    // Numerical add/subtract stabilizer, independent of the matrix support rule.
    double pf_composition_stabilizer_Dalpha_multiplier;
    int diagnostic_rsmd_release_window_steps;
    double diagnostic_rsmd_seed_R_eff_h_nm;
    char diagnostic_rsmd_provenance[64];

    int post_conversion_y_update_audit_enabled;
    int post_conversion_y_update_audit_steps;
    char post_conversion_y_update_audit_prefix[128];
    int audit_post_insertion_drift_enabled;
    int audit_post_insertion_drift_steps;
    char audit_post_insertion_drift_prefix[128];
    int y_update_k0_audit_enabled;
    int y_update_k0_audit_steps;
    char y_update_k0_audit_prefix[128];
    // Optional conservative correction after each Y update:
    // apply a scalar shift Y <- Y + lambda so storage_exact total mass
    // matches a chosen target without changing the Y-update RHS itself.
    int y_update_mass_projection_enabled;
    // Write per-step projection CSV only when explicitly requested
    // (or when a Y-update audit is already active).
    int y_update_mass_projection_report_enabled;
    int y_update_mass_projection_max_iter;
    double y_update_mass_projection_tol;
    // Allowed:
    // - "pre_Y_update" (recommended)
    // - "post_conversion_baseline" (debug/test mode)
    char y_update_mass_projection_target_mode[64];
    char gp_C_mode[32];
    char gp_eps_mode[32];
    char gp_init_mode[32];
    char gp_init_mass_mode[32];
    double gp_eta_seed_radius;
    double gp_eta_seed_peak;
    double gp_eta_seed_center_x;
    double gp_eta_seed_center_y;
    double gp_eta_seed_center_z;
    double gp_eta_iface_width;
    // Observed-GP diffuse initialization:
    // represent an experimentally observed GP zone by matching the h(eta)
    // effective volume to a target physical radius, while applying a smooth,
    // broad composition compensation cloud that preserves total xBtot.
    double gp_obs_target_radius_nm;
    double gp_obs_eta_peak;
    double gp_obs_iface_width_nm;
    char gp_obs_profile_type[32];
    char gp_obs_match_mode[64];
    char gp_obs_compensation_mode[64];
    double gp_obs_depletion_radius_factor;
    double gp_obs_depletion_smooth_width_factor;
    double gp_obs_min_xB_alpha;
    double gp_obs_max_xB_alpha;
    int gp_raw_reaction_drive_only;
    int gp_raw_reaction_drive_use_raw_units_debug;
    double gp_reaction_nu_A;
    double gp_reaction_nu_B;
    double gp_xB_eq_alpha_for_eta;
    char gp_L_eta_mode[32];
    double gp_M_eta_phys;
    double gp_M_eta_ratio_to_crit;
    int gp_kinetic_ref_enabled;
    int gp_kinetic_ref_apply;
    int phi_eta_step_delta_diag_enabled;
    int phi_eta_step_delta_diag_every;
    int phi_eta_step_delta_diag_max_steps;
    char phi_eta_step_delta_diag_prefix[128];
    int phi_eta_rhs_attribution_diag_enabled;
    int phi_eta_rhs_attribution_diag_every;
    int phi_eta_rhs_attribution_diag_max_steps;
    char phi_eta_rhs_attribution_diag_prefix[128];
    double gp_h_alpha_eps;
    char gp_eta_mass_limiter[32];
    char gp_y_update_mode[32];               // old_rhs | conservative_y_rhs | picard_storage | storage_exact
    int gp_y_picard_iters;

    // 界面能相关参数
    double W;
    double kappa_phi;

    // 相场动力学参数
    double L_phi;
    char L_phi_calibration_mode[64];
    double L_phi_physical_value;
    double L_phi_code_value;
    double zeta_phi;
    double zeta0_phi;
    double D_beta_for_calibration;
    double zeta_eta;
    double zeta0_eta;
    char thermodynamic_backend_hash[65];
    char calibration_script_hash[65];

    // 化学扩散系数
    double D_alpha;       // 基体相 (matrix) 扩散系数
    double D_compound;    // 化合物相扩散系数

    // 化学势模型控制参数
    double temperature_C;
    int thermo_convex_extrapolation_enabled;
    double mu_reference_scale;

    // 化学计量系数
    double v_A;
    double v_B;

    // 化合物相参考物性
    double mu0_compound;
    double Vm_compound;
    double Vm_alpha_0;
    double dVm_alpha_dxB;

    // Logit x_B 数值方案相关参数
    double Y_clip;
    double xB_eps;
    double xB_s_floor;

    // === 初始化场构造所需参数 ===
    double ic_vf_init_phi;      // 初始化合物(φ≈1)体积分数 ∈(0,1)
    double ic_vf_target_phi;    // 目标化合物体积分数 ∈(0,1)
    int    ic_phi_num_seeds;    // 种子数量 N
    double ic_phi_iface_w;      // tanh 界面宽度
    double ic_xB_width_factor;  // xB 界面宽度因子 (相对于 phi 宽度)
    double ic_phi_seed_radius;  // 反求得到的半径(由初始化写回)
    double ic_xB_eq_matrix;     // 矩阵平衡溶解度 x_{B,eq}
    double ic_1d_half_width_ratio; // 1D 基准测试：界面半宽 / (Nx*dx) 的比例，可由 main.c 调节

    // 可选：若未提供种子中心则在体内均匀生成
    int    ic_phi_centers_max;
    double *ic_phi_centers;     // 长度 3*N

    // === 破对称 / 多初值测试相关初始化参数 ===
    // init_shape_mode: -1 = legacy 行为（保持旧逻辑）;
    //                   0 = 球 (Rx=Ry=Rz=R);
    //                   1 = 椭球 (Rx, Ry, Rz 由 R 和 axis ratio 决定)
    int    init_shape_mode;
    // 轴向比例：当 init_shape_mode==1 时，Rx = R*init_axis_ratio_rx 等；
    // 若未显式指定（保持默认 1,1,1），则退化为球形
    double init_axis_ratio_rx;
    double init_axis_ratio_ry;
    double init_axis_ratio_rz;
    // 椭球短轴法向方向的极角（度）：theta, phi
    // theta=0,phi=0 时短轴沿 z 轴
    double init_tilt_theta_deg;
    double init_tilt_phi_deg;
    // 几何中心基础上的物理偏移（单位与 dx 相同）
    double init_center_shift_x;
    double init_center_shift_y;
    double init_center_shift_z;
    // phi 初始化后叠加的均匀随机噪声幅度：phi <- clamp01(phi + eta), eta∈[-amp,amp]
    double init_phi_noise_amp;
    unsigned long init_phi_noise_seed;
    // 自动测试 preset 编号；<0 表示关闭，>=0 时根据预定义方案设置一组初始化参数
    int    init_test_id;
    // 是否在 init_phi_kernel 中使用旋转椭球（由 tilt 或 preset / flag 控制）
    int    init_use_rotation;

    /* ==== 1D 基准测试相关开关和参数 ==== */
    int    oneD_test_mode;   /* =1 时运行 1D 测试（使用 1D slab 初始化） */
    double ic_xB_out;        /* 1D 情形：外部（φ≈0）区域初始 xB */
    double ic_23d_xB_out;    /* 2D/3D 情形：外部（φ≈0）区域初始 xB，若 >0 则忽略 ic_vf_target_phi，直接使用该值 */

    // === 输出控制 ===
    int diag_vtk_enabled;   // 诊断VTK输出开关：=1 时输出除 phi/xB/xBtot 外的其它VTK场
    int diag_elastic_bulk_penalty_enabled;  // 弹性 bulk 惩罚诊断开关：=1 时计算并输出弹性 bulk 能量密度诊断（需要 elastic_enabled=1）

    // === 物理量标定（用于把无量纲能量/化学势转换到物理单位）===
    // w_phys = 12*gamma/lambda_sm (J/m^3)
    double gamma_Jm2;       // 界面能 γ (J/m^2)
    double lambda_sm_m;     // 界面厚度参数 λ_sm (m)
    int elastic_gel_is_dimless; // 1: gel/gel_hat 为无量纲；0: gel 已是 J/m^3
    double Vm_alpha_0_phys_m3mol; // 基体相有量纲摩尔体积 (m^3/mol)，用于弹性 bulk 惩罚诊断：Delta_mu_el = E_el_bulk_Jm3 * Vm_alpha_0_phys_m3mol

    // ============================================
    // 弹性计算相关参数
    // ============================================

    // 弹性计算控制
    int elastic_enabled;     // 是否启用弹性计算（0/1）
    int elastic_iter_max;    // 弹性弛豫最大迭代次数（类似SDV_Poly.c的total）
    // 无量纲的弹性 shift 能量密度（加在 delta_mu 上）；仅在 elastic_enabled=1 时有效
    double elastic_shift_dimless;

    // 基体弹性刚度矩阵（21个独立分量，Voigt记号）
    double S_11, S_12, S_13, S_14, S_15, S_16;
    double S_22, S_23, S_24, S_25, S_26;
    double S_33, S_34, S_35, S_36;
    double S_44, S_45, S_46;
    double S_55, S_56;
    double S_66;

    // 弹性常数perturbation（析出相相对基体的弹性常数差，21个分量）
    double S_p_11, S_p_12, S_p_13, S_p_14, S_p_15, S_p_16;
    double S_p_22, S_p_23, S_p_24, S_p_25, S_p_26;
    double S_p_33, S_p_34, S_p_35, S_p_36;
    double S_p_44, S_p_45, S_p_46;
    double S_p_55, S_p_56;
    double S_p_66;

    // 外部应变（6个分量，Voigt记号：xx, yy, zz, yz, xz, xy）
    double E0_xx, E0_yy, E0_zz, E0_yz, E0_xz, E0_xy;

    // Wu & Ji 模型中的 stress-free transformation strain ε^00_ij（Voigt顺序）
    double eps_xx00, eps_yy00, eps_zz00, eps_yz00, eps_xz00, eps_xy00;

    // 各向同性化学膨胀参数：eps_iso_over_vB = ε_iso / v_B（标量，用户可输入）
    double eps_iso_over_vB;

    // 化学膨胀参考成分 x_B^0，用于定义 V_ref = V_m^alpha(x_B^0)
    double xB_ref_for_eps_c;

    // ============================================
    // Energy minimization mode (quasi-static)
    // ============================================
    // mode: 0 = dynamics (default), 1 = minimize (energy minimization)
    // Minimization(full-model): enable chem+diffusion with volume Lagrange constraint
    int mode;
    // 0 = legacy phi-only minimization (no chemical/transport)
    // 1 = full-model minimization (chemistry + diffusion + volume constraint)
    int minimize_full_model;
    // minimize iteration control
    int    minimize_max_iter;
    double minimize_dt;       // dt for gradient flow / semi-implicit update
    double minimize_V0;       // target volume fraction <h(phi)>; if <=0 use initial mean_h
    int    minimize_resample_elastic_every;  // true residual diagnostic interval; N<=0 disables
    // 智能收敛判据参数
    double minimize_rms_dphi_threshold;        // phi 收敛判据：rms_dphi < threshold (default: 1e-6)
    double minimize_rms_dY_threshold;          // Y 收敛判据（full-model）：rms_dY < threshold (default: 5e-5)
    double minimize_energy_diff_rel_threshold; // 平台判据：能量相对变化率 < threshold (default: 1e-9)
    double minimize_rms_res_for_energy_plateau; // 仅保留为诊断：能量停滞时的 rms_res 阈值
    double minimize_rms_res_threshold;        // Euler-Lagrange/KKT 残差判据：rms_res < threshold (default: 1e-4)
    double minimize_vol_err_rel_threshold;    // 体积约束相对误差：vol_err_rel < threshold (default: 1e-4)，V0<=0 时以第一步体积为参考
    int    minimize_convergence_steps;        // 连续满足判据的步数阈值 (default: 10)
    double minimize_dt_safety_limit;          // 预留：dt 安全下限（当前不再用于能量上升自适应）
    double eta_lambda_vol;                    // lambda_vol under-relaxation 阻尼系数 (default: 0.2, range: [0,1])
    double minimize_xB_max_safe;              // minimize 模式下热力学调用前的 xB 上限（pre-thermo clamp, default: 0.07）
    int    minimize_post_projection_iters;    // 后投影修正子步数 (default: 1, n>=0)，抑制 vol 慢漂
    int    minimize_continue_from_vtk;        // =1 时从已有 VTK 场恢复，而非重新初始化
    char   continue_phi_vtk_path[4096];       // continuation: 必需的 phi VTK 路径
    char   continue_xB_vtk_path[4096];        // continuation(full-model): 可选 xB VTK 路径
    int    init_mode_raw_fields;               // =1 时从 Python 生成的 raw 场读取
    char   init_phi_raw_path[4096];            // raw_fields: phi_init.raw
    char   init_xB_raw_path[4096];             // raw_fields: xB_init.raw
    char   init_ctot_raw_path[4096];           // raw_fields: optional authoritative Ctot checkpoint
    char   init_ctot_nm1_raw_path[4096];       // raw_fields: optional BDF2 Ctot history
    char   init_phi_nm1_raw_path[4096];        // raw_fields: optional BDF2 phi history
    char   init_eta_raw_path[4096];            // raw_fields: optional eta_init.raw
    char   init_meta_path[4096];               // raw_fields: init_meta.json

    // ============================================
    // Scheduled nucleation test mode (explicit test-only feature)
    // ============================================
    int    scheduled_nuc_enabled;              // =1 only with --enable-scheduled-nucleation-test
    int    scheduled_nuc_use_manual_nucleus;   // =1 keeps legacy manual profile/source dirs; default selector path is 0
    int    scheduled_nuc_selector_active;       // runtime diagnostic: selector resolved the insertion template
    char   scheduled_nuc_source_dyn_dir[4096]; // source no-strain dynamic-continue directory
    char   scheduled_nuc_profile_dir[4096];    // extracted faceted profile directory
    char   scheduled_nuc_selector_script[4096]; // nucleus_selector.py path
    char   scheduled_nuc_catalog_json[4096];   // nucleus_catalog.json path
    char   scheduled_nuc_selected_json[4096];  // selected_nucleus.json path
    char   scheduled_nuc_selection_log[4096];  // optional selected_nucleus_log.csv path
    char   scheduled_nuc_selected_shape_type[64];
    double scheduled_nuc_selected_rc_nm;
    double scheduled_nuc_selected_energy_barrier_kBT;
    double scheduled_nuc_selected_input_xB;
    double scheduled_nuc_selected_input_strain;
    char   scheduled_nuc_source_step[64];      // "latest" or numeric label, diagnostic only for v1
    char   scheduled_nuc_source_phi_vtk[4096]; // optional explicit source phi VTK name/path
    char   scheduled_nuc_source_xB_vtk[4096];  // optional explicit source xB VTK name/path
    char   scheduled_nuc_steps_csv[1024];      // e.g. "100,300,600"
    char   scheduled_nuc_centers_nm[2048];     // e.g. "100,100,100;180,100,100"
    char   scheduled_nuc_seed_metadata_json[4096];
    double scheduled_nuc_t_nuc_code;
    double scheduled_nuc_t_nuc_s;
    char   scheduled_nuc_library_entry_id[256];
    double scheduled_nuc_library_r_seed_nm;
    double scheduled_nuc_library_r_seed_grid;
    double scheduled_nuc_library_tau_bridge_s;
    double scheduled_nuc_library_tau_bridge_code_time;
    double scheduled_nuc_library_dt_code;
    double scheduled_nuc_library_dt_s;
    double scheduled_nuc_library_t_real_unit_s;
    double scheduled_nuc_library_t_insert_code;
    double scheduled_nuc_library_t_insert_s;
    char   scheduled_nuc_source_case_label[256];
    char   scheduled_nuc_xB_edge_mode[128];    // sample-current-background-shell
    char   scheduled_nuc_edge_sample_stat[32]; // mean|median (median falls back to sampled sort)
    double scheduled_nuc_edge_sample_inner_nm;
    double scheduled_nuc_edge_sample_outer_nm;
    double scheduled_nuc_local_comp_inner_nm;
    double scheduled_nuc_local_comp_outer_nm;
    double scheduled_nuc_local_comp_taper_nm;
    double scheduled_nuc_source_dx_nm;
    double scheduled_nuc_source_lambda_nm;
    double scheduled_nuc_target_lambda_nm;
    double scheduled_nuc_scale_geometry;
    double scheduled_nuc_scale_interface_width;
    double scheduled_nuc_scale_xB_profile_width;
    double scheduled_nuc_alpha_interface;
    double scheduled_nuc_xB_min;
    double scheduled_nuc_xB_max;
    double scheduled_nuc_phi_matrix_threshold;
    double scheduled_nuc_W_comp_threshold;
    int    scheduled_nuc_mass_iters;
    double scheduled_nuc_mass_tol;
    int    scheduled_nuc_write_event_vtk;
    int    scheduled_nuc_fallback_analytic_sphere;

    // ============================================
    // Dynamics xBtot mass-drift diagnostics (explicit opt-in)
    // ============================================
    int    dynamics_mass_diag_enabled;        // =1 only with explicit flag
    int    dynamics_mass_diag_interval;       // per-step interval for CSV diagnostics
    int    enable_Y_rhs_previous_time_level;  // =1 only with explicit flag; evaluate Y RHS explicit terms using phi^n / Y^n
    int    disable_Y_rhs_gamma_term;          // =1 only with explicit flag; diagnostic mode only
    double Y_rhs_term_h_scale;                // diagnostic sensitivity scaling for term_h, default 1.0
    int    enable_Y_rhs_picard;               // =1 only with explicit flag; Picard iterate gamma*dYdt coupling
    int    Y_rhs_picard_iters;                // number of Picard iterations; default 1
    double Y_rhs_picard_omega;                // Picard under-relaxation; default 1.0

    // 当前初始化 case 的标签，用于结果子目录命名
    char   init_case_tag[256];
} PFParams;

#endif // PF_PARAMS_H
