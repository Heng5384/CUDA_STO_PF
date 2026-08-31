# Four-bucket automated controls

## 结论

输出摘要的精确状态为 `PASS_FOUR_BUCKET_HOST_STORAGE_CONTROL_NOT_CUDA`；它满足本轮的 host-only `PASS_PF_FOUR_BUCKET_STORAGE` gate。两者都限定为 `HOST_STORAGE_CONTROL_NOT_CUDA`。这些控制验证 field-space inverse、compact PSD ledger 和 C++ V6 checkpoint persistence；它们不是 CUDA PF dynamics smoke。

| Gate | host-side 结果 |
|---|---|
| S1 zero-aux identity | `phi` 未变；source `xB_alpha` 未变；重映射 `xB` 最大差 `4.336808689942018e-18` |
| S2 nonzero frozen storage-only | local `phi`/`xB`/matrix/resolved 保持不变；GP `1.309926818847619e-21` mol、subgrid `6.549634094238095e-22` mol 只进入完整四库存；relative residual `1.4833316144724465e-16` |
| S2E fixture-conditioned rebalanced control | 固定总库存时 matrix baseline 映射的最大差 `9.330867835967947e-05`；resolved β 仍只计一次；four-bucket residual `0` |
| S3 matrix→GP→matrix | `ΔQ=1.309926818847619e-21` mol；transfer relative residual `3.862297964393128e-14`；forward matrix inverse residual `3.712771878204502e-16`；reverse inverse residual `3.675644159422457e-16`；roundtrip `max |ΔxB|=2.6020852139652106e-18`；未使用 clipping |
| S4 checkpoint/restart dependency | `PASS_PF_ZERO_MODE_CHECKPOINT_PROVENANCE_V2_TO_V6_AUX`：V6 zero/nonzero aux PSD＋package identity roundtrip、hash mismatch reject 与 V2–V5 backward read 均通过；V5 active auxiliary restart 因无 package identity 被拒绝 |

S3 的 matrix capacity、bound margins、transfer map 与字段 hashes 见 `outputs/kwn_pf_state_closure_v1/four_bucket_storage_control_summary.json`；逐库存行见 `outputs/kwn_pf_state_closure_v1/four_bucket_test_ledger.csv`。

补充的 compact sidecar host materializer 也通过 `PASS_PF_AUXILIARY_HANDOFF_V2_HOST_MATERIALIZER`：它检验四项 identity、ledger 闭合、PSD bin inventory 与 `m⁻4 → m⁻3` 转换。这是 S1–S4 之外的 parser/materialization check，仍为 host-only。

## 证据边界

S1–S4 没有启动 CUDA PF。因此它们没有测量 A–E 的 0/6/48 h inventories、local dynamics、free-energy continuity、matrix pulse、seed loss、NaN/Inf、GPU restart trajectory 或 frozen auxiliary state 的时间不变性。实际 96³ smoke 仍是 `NOT_RUN_NO_CUDA_OR_PF_BINARY`；不得把本 host gate 扩展为 `PASS_96CUBE_STORAGE_SMOKE`。
