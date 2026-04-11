# Jobs Workflow

这个目录统一存放项目的运行入口脚本，分成两类：

- `run_*.sh`: 本地或交互式 GPU 环境运行
- `submit_*.sbatch`: Slurm 批量提交

当前脚本职责如下：

- `run_dynamics_local.sh`: 本地运行 dynamics 模式
- `run_minimize_init_cases_local.sh`: 本地跑通用多初值最小化
- `run_minimize_caseb_elastic_init_cases_local.sh`: 本地跑 Case B 弹性多初值最小化
- `run_minimize_rotated_oblate_local.sh`: 本地跑旋转扁椭球初始核的单案例
- `submit_minimize_init_cases.sbatch`: Slurm 上提交固定半径的多初值最小化
- `submit_minimize_caseb_elastic_radius_sweep.sbatch`: Slurm 上提交 Case B 半径扫描
- `submit_minimize_fullmodel_radius_sweep_512_uvip.sbatch`: Slurm 上提交 512^3、400C、xB_out=0.03 的 full-model 半径扫描
- `submit_rotate_oblate_vs_sphere_radius_sweep.sbatch`: Slurm 上提交旋转扁椭球与球形对照扫描

标准模板：

- `template_run_local_single.sh`: 本地单案例模板
- `template_run_local_loop.sh`: 本地循环模板
- `template_run_local_continue.sh`: 本地 continue 模板
- `template_submit_slurm_single.sbatch`: Slurm 单作业模板
- `template_submit_slurm_loop.sbatch`: Slurm 循环/扫描模板
- `template_submit_slurm_continue.sbatch`: Slurm continue 模板

推荐复制方式：

- 想新建本地单案例脚本：从 `template_run_local_single.sh` 复制
- 想新建本地 sweep/case-loop 脚本：从 `template_run_local_loop.sh` 复制
- 想新建本地 continue 脚本：从 `template_run_local_continue.sh` 复制
- 想新建单个 sbatch 提交入口：从 `template_submit_slurm_single.sbatch` 复制
- 想新建 radius/temperature/case 扫描脚本：从 `template_submit_slurm_loop.sbatch` 复制
- 想新建 continue 提交脚本：从 `template_submit_slurm_continue.sbatch` 复制

标准章节顺序：

- `A. 启动与项目定位`
- `B. 默认参数区`
- `C. 生成 PF 参数文件` 或 `C. 循环执行`
- `D. main_cuda 调用`

也就是说，后续无论人还是 AI 改脚本，优先只改：

- `#SBATCH` 资源区
- 默认参数区
- 循环变量区
- `main_cuda` 命令区

规范约定：

- 所有模拟结果都写到仓库根目录下的 `Results/`
- Slurm 日志统一写到 `jobs/logs/`
- 本地 `run_*.sh` 脚本无论从仓库根目录还是 `jobs/` 目录启动，都能自动定位项目根目录
- `submit_*.sbatch` 现在支持从仓库根目录或 `jobs/` 目录提交
- `submit_*.sbatch` 会在运行时自动把 stdout/stderr 重定向到 `${PROJECT_ROOT}/jobs/logs/`，不再依赖提交目录解释相对路径

物理参数覆盖流程：

- `physical_inputs.example.json` 是基准物理参数输入
- `run_*.sh` 和 `submit_*.sbatch` 会先根据环境变量生成 `jobs/generated/physical_inputs_<tag>.json`
- 然后调用 `Unit_Psedobinary.py` 生成 `jobs/generated/pf_params_<tag>.params`
- 最后用 `./main_cuda --pf-param-file jobs/generated/pf_params_<tag>.params` 真正启动模拟

因此，在脚本或提交命令里设置的物理参数会覆盖基准 JSON，再进入相场：

- `TEMP_C` -> `temperature_C`
- `DX_M` -> `dx`（单位 `m`）
- `GAMMA_JM2` -> `gamma`
- `LAMBDA_SM_M` -> `lambda_sm`
- `VF_INIT` -> `vf_init`
- `VF_TARGET` -> `vf_target`
- `EPS_ISO` -> `eps_iso`

continue 运行补充约定：

- 现在支持 `minimize-continue` 和 `dynamics-continue`
- 每个结果目录现在会额外保存：
  - `output_root/pf_input.params`
  - `case_output_dir/pf_input.params`
- continue 时如果 `--continue-phi-vtk` 所在目录下存在 `pf_input.params`，程序会优先使用这份参数快照
- 如果该目录下没有 `pf_input.params`，则回退到脚本传入的 `--pf-param-file`
- full-model continue 若未提供 `--continue-xB-vtk`，程序会按初始化阶段同样的质量守恒逻辑从 `phi` 重建 `xB/Y`
- `--mode=dynamics-continue` 适合“读取临界核的 `phi_final/xB_final` 后继续跑动态生长”的场景；
  `--mode=minimize-continue` 适合从已收敛的场继续做能量最小化

如果需要一次覆盖多项复杂字段，可用：

- `PHYSICAL_OVERRIDE_JSON='{\"temperature_C\":410.0,\"vf_target\":0.12}'`
- `PHYSICAL_OVERRIDE_FILE=/path/to/override.json`

需要区分：

- `TEMP_C`、`DX_M`、`VF_TARGET` 这类属于物理输入，会先进 `physical_inputs -> .params`
- `RADIUS`、`XB_OUT`、`NSTEPS`、`MIN_DT` 这类仍然是运行控制参数，直接走 `main_cuda` 命令行
