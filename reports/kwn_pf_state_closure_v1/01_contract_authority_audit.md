# 合同权威审计

## 结论

`HISTORICAL_AS_RUN_AUTHORITY_UNRECOVERED`。

本轮的 `PF_KWN_VALIDATION_CONTRACT_V1` 是限定于六颗粒 96³ fixture 的技术验证合同；它不能替代、重命名或回写历史 246³/400³ production 的 as-run 合同。因而本轮任何 PASS 都只构成 validation-control 证据。

## 已完成的只读证据搜寻

本轮只读检查覆盖当前可见的 local worktrees、Git history/Git objects、workstation 既有 outputs、build/run manifests 和 provenance 记录。检查到的证据如下：

| 对象 | 已见证据 | 能否作为唯一 as-run authority |
|---|---|---|
| 历史 runtime 线索 | 多个 conditional candidate manifest/provenance 指向 `1c08f9ee011b31e0cd4d82749e58a8f69ebd2204`，并保留 fixture、binary/checkpoint-hash-chain 线索；冻结审计记录 legacy `L(T)=41212.9-18.05T` J mol⁻¹ | 否；这些线索仍是 `SNAPSHOT_ONLY_NO_IDENTITY_KINETICS`/conditional evidence，不能消除 legacy 与 exact-candidate 的 authority conflict，也未形成 formal as-run contract |
| exact candidate | 只读的热力学冻结材料记录 `ΔH=41504.29119633958` J mol⁻¹、`ΔS=18.469276826409214` J mol⁻¹ K⁻¹ | 否；它是 validation exact candidate，不是历史 production 的已恢复 as-run 身份 |
| 下游 400³ full PSD candidate | base worktree 的 `reports/p1_pf_discrete_psd_true_area_transport_v1_20260806/pf_particle_geometry_manifest.csv`（SHA-256 `87dc051849308e2411ef09a03659e3911bed6bbea550c87ad49fa7bb4a7f5e42`）含 step 21798 的 001 REF_BROAD_N512_A 121 条与 004 NARROW_N512_A 165 条 full-radius rows；14-case 逐粒子表可数值复核同一列表 | 否；它是 downstream conditional diagnostic，不是原始 checkpoint/as-run authority |
| downstream route status | P1 manifest 为 `CONDITIONAL_PASS_TRUE_AREA_OR_AUTHORITY_PENDING`、`formal_authority_pending=true`；authority manifest 的 001/004 为 `CONDITIONAL_NOT_FORMALLY_REGISTERED`、root 为 `RETRYABLE_CASE_FAILURE`，004 另有 unresolved periodic overlap/lineage | 否；这些状态禁止将其升级为 historical production authority |
| raw checkpoint 与完整 as-run contract | manifests 记录 001/004 的外部 `step_21798.chk` paths，并记录 checkpoint hash chain、fixture 与 binary identity；但原始 checkpoint bytes 在本机不存在，且这些 records 本身不构成 authority-qualified 的唯一完整 as-run runtime contract | 否 |

这些可读的 downstream PSD 被保留为其原有来源的条件性历史证据，未重命名、未用作本轮 historical comparison，也未回写旧 metadata。没有根据 moments、`N`、`Rmean`、`fβ` 或随机 lognormal 伪造完整 PSD，也没有提交 cluster 作业。

## 本轮验证合同的边界

本轮合同位于 `contracts/pf_kwn_validation_contract_v1.json`，源基线为 `f228cd45638279413ea62edd774d5b2315dbcbf6`。它只绑定：

- 六颗粒 96³ validation fixture；
- 同合同的 KWN host control；
- frozen、storage-only 的四库存 handoff/checkpoint 控制。

历史 12 h PSD 与 historical beta-only comparison 继续分别保持 `HISTORICAL_12H_PSD_NOT_RECOVERED` 和 `HISTORICAL_BETA_ONLY_COMPARISON_NOT_RUN`。在这两个 authority 条件恢复前，不能把 validation 结果表述为既有 production consistency。
