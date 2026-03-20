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
- 脚本无论从仓库根目录还是 `jobs/` 目录启动，都能自动定位项目根目录
