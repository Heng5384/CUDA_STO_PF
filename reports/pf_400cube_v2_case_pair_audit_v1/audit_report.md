# 400³ Case 001/002 transport-V2 endpoint audit

Status: `CONDITIONAL_V2_M0_MI_ENDPOINT_AUDIT_V1`.

Frozen V2 equations and 18 AQ background/interface members were applied without refitting. `A_N=1.5`, `S11=S13=0`, and Yu 0.1172768 scaling are not used. The endpoint particle inputs come from the completed 400³ checkpoints; `MS/MIS` are not calculated because no same-time accepted-field strain replay is available.

| case | matrix | model | κ(6 h) at 573.15 K | κ(48 h) | Δκ | relative |
|---|---|---|---:|---:|---:|---:|
| 001 | pf_time_varying_matrix | M0 | 0.873537403518 | 0.871782361358 | -0.00175504216 | -0.200912079% |
| 001 | pf_time_varying_matrix | MI | 0.817801507044 | 0.851335564385 | +0.0335340573408 | +4.10051303% |
| 001 | fixed_6h_matrix | M0 | 0.873537403518 | 0.872401313994 | -0.001136089524 | -0.130056197% |
| 001 | fixed_6h_matrix | MI | 0.817801507044 | 0.85194342763 | +0.0341419205859 | +4.17484197% |
| 002 | pf_time_varying_matrix | M0 | 0.873537780208 | 0.871954860087 | -0.00158292012033 | -0.181207975% |
| 002 | pf_time_varying_matrix | MI | 0.817801869357 | 0.852210866306 | +0.0344089969491 | +4.20749796% |
| 002 | fixed_6h_matrix | M0 | 0.873537780208 | 0.872413446185 | -0.00112433402244 | -0.128710406% |
| 002 | fixed_6h_matrix | MI | 0.817801869357 | 0.852661517331 | +0.0348596479747 | +4.26260312% |

Input endpoint descriptors:

| case | 6 h particles | 48 h particles | 6 h mean R (nm) | 48 h mean R (nm) | 6 h Sv (m⁻¹) | 48 h Sv (m⁻¹) | 6 h xAg | 48 h xAg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 001 | 512 | 21 | 8.878359 | 24.561093 | 7.97764e+06 | 2.63269e+06 | 0.006202331 | 0.006308959 |
| 002 | 512 | 19 | 8.878359 | 25.402071 | 7.97764e+06 | 2.53966e+06 | 0.006202266 | 0.006281227 |

Interpretation: M0 isolates AQ background + host + full-PSD precipitate scattering; MI adds the frozen V2 interface term. These are conditional model diagnostics, not absolute experimental κ reproduction and not production authority because the endpoint-only lineage inputs are marked BLOCKED until the full checkpoint-chain audit is complete.
