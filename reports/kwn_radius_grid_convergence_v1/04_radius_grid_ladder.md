# Radius-grid ladder

Fixture status: `FAIL_KWN_RADIUS_GRID_CONVERGENCE`; smooth status: `PASS_KWN_RADIUS_GRID_CONVERGENCE`.

| Scenario | Pair | Largest 48 h error | Largest full-time error | Six primary 48 h errors |
| --- | --- | --- | --- | --- |
| fixture | 100 vs 200 | 21.150% | 22.177% | N_m0_m3=17.352%; Rmean_m=5.521%; Rmean3_m3=21.150%; Sv_m_inv=7.016%; f_beta=0.128%; matrix_xB=0.628% |
| fixture | 200 vs 400 | 15.994% | 18.096% | N_m0_m3=13.698%; Rmean_m=4.357%; Rmean3_m3=15.994%; Sv_m_inv=5.353%; f_beta=0.105%; matrix_xB=0.511% |
| fixture | 400 vs 800 | 12.297% | 15.153% | N_m0_m3=10.874%; Rmean_m=3.470%; Rmean3_m3=12.297%; Sv_m_inv=4.130%; f_beta=0.086%; matrix_xB=0.417% |
| fixture | 800 vs 1600 | 9.910% | 13.273% | N_m0_m3=8.950%; Rmean_m=2.882%; Rmean3_m3=9.910%; Sv_m_inv=3.314%; f_beta=0.073%; matrix_xB=0.354% |
| fixture | 1600 vs 3200 | 8.394% | 12.046% | N_m0_m3=7.684%; Rmean_m=2.507%; Rmean3_m3=8.394%; Sv_m_inv=2.782%; f_beta=0.065%; matrix_xB=0.314% |
| smooth | 100 vs 200 | 16.475% | 16.475% | N_m0_m3=14.060%; Rmean_m=4.189%; Rmean3_m3=16.475%; Sv_m_inv=5.779%; f_beta=0.098%; matrix_xB=0.483% |
| smooth | 200 vs 400 | 9.362% | 9.362% | N_m0_m3=8.507%; Rmean_m=2.418%; Rmean3_m3=9.362%; Sv_m_inv=3.443%; f_beta=0.059%; matrix_xB=0.289% |
| smooth | 400 vs 800 | 5.093% | 5.093% | N_m0_m3=4.815%; Rmean_m=1.324%; Rmean3_m3=5.093%; Sv_m_inv=1.934%; f_beta=0.033%; matrix_xB=0.161% |
| smooth | 800 vs 1600 | 2.676% | 2.676% | N_m0_m3=2.589%; Rmean_m=0.698%; Rmean3_m3=2.676%; Sv_m_inv=1.036%; f_beta=0.018%; matrix_xB=0.086% |
| smooth | 1600 vs 3200 | 1.375% | 1.375% | N_m0_m3=1.347%; Rmean_m=0.359%; Rmean3_m3=1.375%; Sv_m_inv=0.538%; f_beta=0.009%; matrix_xB=0.044% |

The P5 endpoint decision always uses the six 48 h metrics, while the full-time maxima are reported without replacing that gate. All inter-grid PSD distances use M0/M3-conservative remapping to a shared 3200-bin analysis grid.
