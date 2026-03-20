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
- `submit_rotate_oblate_vs_sphere_radius_sweep.sbatch`: Slurm 上提交旋转扁椭球与球形对照扫描

规范约定：

- 所有模拟结果都写到仓库根目录下的 `Results/`
- Slurm 日志统一写到 `jobs/logs/`
- 本地 `run_*.sh` 脚本无论从仓库根目录还是 `jobs/` 目录启动，都能自动定位项目根目录
- `submit_*.sbatch` 建议从仓库根目录提交，例如 `sbatch jobs/submit_minimize_init_cases.sbatch`，以保证 `#SBATCH --output/--error` 相对路径与约定一致

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

如果需要一次覆盖多项复杂字段，可用：

- `PHYSICAL_OVERRIDE_JSON='{\"temperature_C\":410.0,\"vf_target\":0.12}'`
- `PHYSICAL_OVERRIDE_FILE=/path/to/override.json`

需要区分：

- `TEMP_C`、`DX_M`、`VF_TARGET` 这类属于物理输入，会先进 `physical_inputs -> .params`
- `RADIUS`、`XB_OUT`、`NSTEPS`、`MIN_DT` 这类仍然是运行控制参数，直接走 `main_cuda` 命令行
