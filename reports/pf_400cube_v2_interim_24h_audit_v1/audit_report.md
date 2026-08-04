# 400 nm任务的transport-V2中期审计：6–24 h

当前状态：`INTERIM_PASS_400CUBE_V2_M0_MI_24H_MS_MIS_PENDING_V1`。

本报告把不可变的`TRANSPORT_V2_SHESKIN_BACKGROUND_INTERFACE_SCALAR_STRAIN_FROZEN`方程与18个已冻结AQ背景/界面参数成员用于新的400 nm PF输入。没有修改或重拟合V2；没有读取24 h实验热导来定参；`A_N=1.5`、`S11=S13=0`，未使用Yu的`0.1172768`缩放，也不声称绝对实验热导率复现。

## PF与谱系门

- 精确24 h端点为step `65393`，checkpoint SHA-256为`217f1b0130e136f70b02e154723752bff6987ec91e28c8abcebdf1821a244760`；该段zero-mode、restart provenance与弹性收敛均PASS，stderr为空。
- step 0至65393的19个逐小时/科学快照在cluster CPU作业`73974`中完成三阈值审计：low/medium/strong tracker均为`PASS_PERIODIC_OVERLAP_PARTICLE_LINEAGE_V1`，merge-aware的18项gate全部PASS。
- 6–24 h内有17个初始身份在逐小时区间内合法消失，1个持久强核merge group；没有未解析split、身份重现或新颗粒。最终为46个几何分量、47个仍存活初始身份。
- 最大质量相对误差为`1.6878645207422476e-11`。

## 结构变化

| 描述量 | 6 h | 24 h | 变化 |
|---|---:|---:|---:|
| 颗粒分量数 | 64 | 46 | −28.1% |
| 平均等体积半径 (nm) | 17.8671 | 18.6655 | +4.47% |
| β体积分数 | 0.0239296 | 0.0235944 | −1.40% |
| PF far-field matrix xAg | 0.00620436 | 0.00654147 | +5.43% |
| 周期true-MC面积 (nm²) | 268477.21 | 228040.54 | −15.06% |
| true-MC `Sv` (m⁻¹) | 4.19496e6 | 3.56313e6 | −15.06% |

周期面积使用V2相同的`N+1` wrapped endpoint lattice、`phi=0.5` marching-cubes合同；6 h输入φ哈希为`d9c3256c...e695c`，24 h输入checkpoint哈希为`217f1b01...4760`。

## V2数值结果

`M0`为冻结AQ背景 + 无位错Yu host + full-PSD直接求和；`MI`在M0上加入V2界面项。表中为18个冻结成员的非概率算术均值。

| T | matrix处理 | M0: 6→24 h (Δκ) | MI: 6→24 h (Δκ) |
|---:|---|---:|---:|
| 303.15 K | PF时变matrix | 1.221091→1.217651 (−0.003440) | 1.161499→1.166462 (**+0.004963**, +0.427%) |
| 303.15 K | 固定6 h matrix | 1.221091→1.221431 (+0.000340) | 1.161499→1.170112 (**+0.008613**, +0.742%) |
| 573.15 K | PF时变matrix | 0.872366→0.870544 (−0.001822) | 0.840744→0.843407 (**+0.002664**, +0.317%) |
| 573.15 K | 固定6 h matrix | 0.872366→0.872493 (+0.000127) | 0.840744→0.845309 (**+0.004566**, +0.543%) |

单位均为W m⁻¹ K⁻¹。完整七温度结果见`interim_m0_mi_summary.csv`。

结论很清楚：full PSD粗化本身的热导增量极小；PF预测的matrix xAg上升会增强点缺陷散射，并把M0的净变化推成轻微负值。真正使V2预测转为正增长的是true interface area/Sv下降15.1%，但到24 h的增幅仍只有约0.3–0.7%，远小于实验时间效应的量级。因此当前400 nm结果支持“界面粗化给出正确正号，但已解析通道的幅度仍小”这一V2判断。

## 尚未关闭的门

`MS`和`MIS`需要与checkpoint当前φ/xAg严格同时间层的accepted-field mechanics replay，不能直接使用checkpoint内属于`n−1`源场的warm displacement。400³生产峰值显存约18.1 GiB，超过workstation的16.3 GiB；因此没有以近似场代替。重放已按队列规则登记为`73850 → 73851 → 73979 → 73980`，写入新的只读分析根；生产作业73850/73851未修改、取消或重排。在重放PASS前，不把MI冒充为完整MIS结果。

## Provenance

- V2 frozen manifest SHA-256：`1c815a6277d46ec4c5537966de402f0e190545b6433761291bf46c08ce725856`
- V2 calibration CSV SHA-256：`251fe3c9c4c83931431514472111e18198d332eedbf1e9efe8bdacf4f0768063`
- V1 full-PSD transport script SHA-256：`a42f933c43d0adbd4fc9a4d86a7b74bc08d4efe64517cc3fbf1ba177b62dcf66`
- hourly observables SHA-256：`399925cd5b057b0849ec316d1e9d1e7e298dea8a7a97ea4803942be1000a35cb`
- hourly particle lineage SHA-256：`e1e17d6730e69b0a3282152c0a42b0e20aaa684a9487de1db898207c87ef1ec1`
- cluster lineage job：`73974`, `COMPLETED`, exit `0:0`
- replay target/tail：`73979`/`73980`

