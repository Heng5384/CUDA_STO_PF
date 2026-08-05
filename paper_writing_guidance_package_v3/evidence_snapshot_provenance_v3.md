# Case 001–003 conditional evidence snapshot provenance

## Identity

The two CSV snapshots in this package are writing-audit extracts, not new production authority. They preserve values needed by the V3 claim matrix and point back to immutable checkpoints, local reports and remote derived artifacts.

## Local sources

- `reports/pf_400cube_v2_case_pair_audit_v1/audit_report.md`, SHA-256 `df8cef2ea4d096c2949f00f7e3f1b2c885fe0d13cf6fe16765f6159785188346`.
- `reports/pf_400cube_v2_case_pair_audit_v1/audit.json`, SHA-256 `8c2abe53873c5ee4d88573adf2c76baada9deef5d8372eba24a0997d5c4fe9a7`.
- frozen V2 transport module `scripts/pf_full_psd_no_dislocation_transport_v1.py`, SHA-256 `a42f933c43d0adbd4fc9a4d86a7b74bc08d4efe64517cc3fbf1ba177b62dcf66`.
- frozen transport contract `data/qualification/pf_full_psd_no_dislocation_transport_v1/transport_parameter_contract.json`, SHA-256 `d16e948dd34403129b2deecb90efbea937251e23ae9c128c353b6f66bbee1ae0`.
- interface calibration `reports/sheskin_pf_interface_strain_blind_prediction_v1/interface_model_6h_calibration.csv`, SHA-256 `251fe3c9c4c83931431514472111e18198d332eedbf1e9efe8bdacf4f0768063`.

## Remote complete-checkpoint conditional sources

Root: `/data/home/luozhiheng/tmp/pf_400cube_psd_ladder_postprocess_v2_20260805/cases/`

| Case | tracker-low observables SHA-256 | tracker-low particle lineage SHA-256 |
|---|---|---|
| 001 | `a0684d2dd81865f06928487654cc94e5110a360de80a41274bf08f230043a4d4` | `01d983c9d248ac2eebb7f47fdd22d86701475d2f3cc93812310f62f9706c2589` |
| 002 | `694ea30d8bc8d236bf2a80511880a845f2cb46efa1f487d4dcd7a22bd7e766b1` | `b0cc9278e848fd1dcc267ca9e9c398812a0bacbd840e1d834b361eaf6b59c8a1` |
| 003 | `23b7a9189a8da31d87ce76d0027b01a5e404ac50e5b2dbbbf81be438ff130ca3` | `bce2017b3746b5ffbd061c9225b620970246087b47635165053617734ba073d8` |

Case 003 48 h checkpoint: `/data/home/luozhiheng/tmp/pf_400cube_psd_spatial_density_ladder_production_v1_20260803/attempts/003/gpu_uvip_74964_3/checkpoints/step_152585.chk`, SHA-256 `9465656663a6fc5b5954f8e197467b9621752d1bf8d1474dab82abdf1be7f4fa`.

## Failure/authority boundary

The original Case 001–003 drivers reached step 152585 and passed segment-level conservation/elastic audits, then ended `RETRYABLE_CASE_FAILURE` because the old anchor path reported `wrong fixture schema`. The later Case 003 merge-aware audit failed closed on `overlapping raw lineage families at one step`; its status and first-failure SHA-256 values are `68666f940d8ecfe50415cf90debb906dcdcde2ded0555247a57ffeb398b1a` and `d825fc15d3cbd14777316d6ae3bdca3079bd06ccb857dde88197ad7cad535851`.

The Case 003 MI value was reproduced read-only with `scripts/analyze_pf_400cube_psd_ladder_transport_v2.py` (SHA-256 `babde335df7ee1b0914caf4044171d9a53fd861c4be0fe8239d8306f853efcac`) using the tracker-low snapshots and frozen V2 inputs. The temporary endpoint output SHA-256 was `4dc9ecbe746941108459e91be1b5e1a517a944a23a53f7248fc99edb30433631`. No PF, checkpoint, raw result or existing transport artifact was changed.

## Permitted use

`COMPLETE_CHECKPOINT_CONDITIONAL_ANALYSIS` in internal writing and authority-pending figures. Prohibited: final production number, converged ensemble statistic, quantitative experimental validation or designed spatial-control evidence.
