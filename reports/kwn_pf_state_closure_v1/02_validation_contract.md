# PF–KWN validation contract

## 冻结身份

合同文件为 `contracts/pf_kwn_validation_contract_v1.json`：

- schema：`PF_KWN_VALIDATION_CONTRACT_V1`；
- `historical_as_run_claim=false`；
- canonicalization：UTF-8 JSON、`sort_keys=true`、`separators=(',', ':')`、`allow_nan=false`、IEEE-754 binary64；
- canonical SHA-256：`d0ff02973ab0f737043e1a40d4f69893a469cbfe2bc4cd22f9e6a410bd0b1333`。

`tools/generate_pf_contract_header.py` 从该 JSON 生成 `generated/pf_kwn_validation_contract_v1.h`；生成头不可人工编辑。`src/kwn_mvp/contract.py` 对同一 JSON 执行 canonical hash、字段、单位及数值结构校验。需要已有 hash 的读取方以相等性 gate 处理不一致，而不是 warning 后继续。

## 已绑定的数值视图

- Python KWN 直接读取该 JSON；PF host thermo probe 经生成头与 `thermo_utils.h` 的活动 wrapper 读取同一套量。
- 合同包括 alpha/beta 热力学、planar solvus、`D(T)`、molar volumes、`gamma`、`lambda`、`h(phi)`、96³ 数值视图和固定的 validation-only 假设。
- convex extrapolation 的合同值为 `enabled=false`；生成头同时给出该开关、`xB_limit=0.09` 和 `penalty=5000 J mol⁻¹`，避免 host wrapper 使用一套手写默认值。
- KWN checkpoint metadata、v2 handoff metadata 和 PF V6 checkpoint provenance 的 host 代码路径以该 hash 绑定。生成的 validation manifest 位于 `outputs/kwn_pf_state_closure_v1/contracts/validation_contract_manifest.json`。

## 弹性记录的解释边界

合同现在记录 `C_alpha_voigt_GPa` 和 `C_beta_voigt_GPa` 两个 6×6 Voigt stiffness tensor，以及 fixture 的 eigenstrain/boundary statement。它们是 validation-contract records，供字段完整性、来源与 fixture 身份审计使用。

这不等于已经完成 CUDA 弹性求解的重放、也不等于 KWN beta-only control 含有 PF elastic self-energy。合同中 KWN 的 elastic penalty 仍明确为零，并标为 spherical mean-field validation-control 假设。当前工作站没有经本合同编译和执行的 CUDA PF binary，因此没有可报告的 CUDA elastic replay。

## 运行时证据边界

生成头与 host-side probe 已可由普通 C++ 编译器检查；这证明的是合同生成与 host wrapper 的一致性。PF binary 的 startup hash、GPU runtime log 和 CUDA checkpoint trajectory 尚未取得，不能由 source path 或 host test 代替。
