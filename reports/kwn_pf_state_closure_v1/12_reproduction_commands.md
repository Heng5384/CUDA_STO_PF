# Reproduction commands

以下命令在 repository root 执行，且只复现本轮 host-side contracts/controls。`outputs/kwn_pf_state_closure_v1/` 必须是新的或空的目标目录；v2 writer 有意拒绝覆盖既有 package，旧 v1 outputs 不得改写。

```bash
python3 tools/generate_pf_contract_header.py --check
PYTHONPATH=src python3 -m unittest tests.kwn.test_pf_validation_contract -v
make test_pf_thermo_probe
PYTHONPATH=src python3 scripts/run_thermo_cross_language_parity.py
make test_pf_zero_mode_checkpoint
make test_pf_auxiliary_handoff_v2
PYTHONPATH=src python3 -m unittest tests.coupling.test_fixture_conditioned_handoff_v2 -v
PYTHONPATH=src python3 -m unittest tests.coupling.test_materialize_fixture_conditioned_handoff_v2_raw_init -v
PYTHONPATH=src python3 scripts/build_fixture_conditioned_kwn_pf_handoff_v2.py --no-synthetic-fallback
PYTHONPATH=src python3 scripts/materialize_fixture_conditioned_kwn_pf_handoff_v2_raw_init.py \
  --handoff-dir outputs/kwn_pf_state_closure_v1/kwn_pf_handoff_fixture_conditioned_v2 \
  --out /tmp/kwn_pf_fixture_conditioned_v2_raw_init
PYTHONPATH=src python3 scripts/run_four_bucket_storage_controls.py
PYTHONPATH=src python3 scripts/run_beta_only_same_contract_control.py
PYTHONPATH=src python3 scripts/write_validation_contract_manifest.py
PYTHONPATH=src python3 scripts/write_pf_smoke_not_run_status.py
PYTHONPATH=src python3 scripts/make_kwn_pf_state_closure_figures.py
```

raw-init materialization 只写入调用方显式指定的新目录；不要把生成的 dense `*.raw.f64` fields 加入 Git。其成功状态仍是 `NOT_RUN_RAW_INITIALIZATION_MATERIALIZATION_ONLY`，不是 PF run。

预期的 host gates 包括 `PASS_THERMO_VALIDATION_HASH_BIND` 和 `PASS_PF_FOUR_BUCKET_STORAGE`（host-only）。`run_beta_only_same_contract_control.py` 当前会在 strict positivity guard 的 `0.39317699499770825 h` 未完成状态非零退出；该退出应记录为 `PARTIAL_BETA_ONLY_DIRECTION_T0_ONLY_KWN_RUNTIME_INCOMPLETE`，不得忽略或用 shell 成功状态掩盖。

这些命令不会启动 CUDA PF。actual A–E smoke 只能在同时具备 CUDA toolchain、用本合同编译的受控 PF binary、以及同一 fixture/source/package provenance 的工作站上运行；在该环境准备完成前，`PF_96CUBE_SMOKE_GATE` 必须保持 `NOT_RUN_NO_CUDA_OR_PF_BINARY`。
