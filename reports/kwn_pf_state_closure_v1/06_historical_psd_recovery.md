# Historical 12 h PSD 恢复

## 状态

`HISTORICAL_12H_PSD_NOT_RECOVERED`。

## 已完成的只读搜寻与发现

本轮检查了可见 local worktrees、Git history/Git objects、workstation 既有 outputs，以及 run manifest/provenance。搜寻找到历史 campaign 的 commit/manifest 线索，主要为 `1c08f9ee011b31e0cd4d82749e58a8f69ebd2204`；也找到 legacy runtime 与 exact-candidate 的热力学记录。

| 搜寻对象 | 具体发现 | 对本轮 historical comparison 的资格 |
|---|---|---|
| P1 downstream full-radius list | base worktree 的 `reports/p1_pf_discrete_psd_true_area_transport_v1_20260806/pf_particle_geometry_manifest.csv`，SHA-256 `87dc051849308e2411ef09a03659e3911bed6bbea550c87ad49fa7bb4a7f5e42`；step 21798/12 h 的 001 REF_BROAD_N512_A 有 121 条、004 NARROW_N512_A 有 165 条；019/020/021 也分别有 125/119/124 条 | 可读 PSD evidence，但不合格为 authority：P1 `formal_authority_pending=true`，001/004 均为 `CONDITIONAL_NOT_FORMALLY_REGISTERED`，004 还有 periodic/lineage block |
| 14-case independent downstream list | `reports/pf_400cube_14case_hourly_transport_audit_v1_20260809_complete/data/case001_psd_long.csv`（SHA-256 `a66ff4bc42c3f69fb931f234bde92436962964d458d8687db45e6fc89970d565`）与 `case004_psd_long.csv`（`6a948fe31f0426d0632cf2925dd72e14f6825177668ecd2fafb85fa4681fdd84`）在 12.00005750730298 h 给出相同 121/165 个 `R_i_nm`，最大差分别 `7.1e-15`/`2.6e-9` nm | 不合格：14-case report 明确 `ROOT_AUTHORITY_PASS=NOT_CLAIMED`、`RETRYABLE_CASE_FAILURE`，只允许 conditional read-only diagnostic |
| raw checkpoint / as-run contract | P1 `checkpoint_authority_manifest.csv`（SHA-256 `356666777e1aae3e32b321186d02de3699cdd52d068b4b708415e4373704e236`）记录外部 001/004 `step_21798.chk` paths、checkpoint hash chain、fixture 与 binary identity；当前 workstation 上这两个 raw files 缺失，且该 conditional record 不是 authority-qualified 的唯一完整 runtime contract | 不合格 |

因此，**下游 full PSD lists 已恢复为 conditional evidence，raw authoritative checkpoint 与唯一 as-run contract 未恢复**。本轮没有把这些候选重命名为 authoritative historical comparison，也没有根据 `N`、`Rmean`、`f_beta` 或 moments 重构伪 PSD，更没有运行 cluster/400³ 作业。

历史 beta-only comparison 继续为 `HISTORICAL_BETA_ONLY_COMPARISON_NOT_RUN`。该缺口不阻止 96³ validation-control，但禁止把其结果称为 historical production comparison。
