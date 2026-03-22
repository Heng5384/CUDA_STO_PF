#ifndef PF_PARAMS_H
#define PF_PARAMS_H

typedef struct {
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

    // 界面能相关参数
    double W;
    double kappa_phi;

    // 相场动力学参数
    double L_phi;

    // 化学扩散系数
    double D_alpha;       // 基体相 (matrix) 扩散系数
    double D_compound;    // 化合物相扩散系数

    // 化学势模型控制参数
    double temperature_C;
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

    // 当前初始化 case 的标签，用于结果子目录命名
    char   init_case_tag[256];
} PFParams;

#endif // PF_PARAMS_H
