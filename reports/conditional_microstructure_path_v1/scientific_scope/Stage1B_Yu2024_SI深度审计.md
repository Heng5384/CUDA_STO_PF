# Stage 1B — Yu et al. (2024) Supporting Information 深度审计

## 1. 审计结论

Supporting Information（SI）补齐了主文最关键的三类信息：

1. Debye–Callaway 各散射项、参数和拟合自由度；
2. PbTe/Ag2Te 取向、界面结构和界面能的详细证据；
3. 退火后 Ag 装饰位错作为 pipe-diffusion 通道并促进 Ostwald ripening 的直接陈述。

SI 使本项目的边界更加清楚：

> 当前相场模型适合作为 resolved Ag2Te 群体在 bulk-diffusion、coherent/near-coherent 极限下的三维演化基线；它不能直接作为 Yu 2024 AQ→48 h 实验过程的完整数字孪生。

本项目相对于 Yu 2024 的稳固增量仍然是：

- 受库存约束的条件初始化；
- 三维正向群体演化；
- 动力学可达的 PSD 和界面面积轨迹；
- 初始群体不确定性传播；
- 组织描述符压缩及连续 κL(t) 预测。

---

## 2. SI 对粗化机制的加强证据

Figure S5 明确指出：

- 48 h 退火样品中部分颗粒由 Ag-rich dislocations 连接；
- 位错是 Ag “pipe diffusion”的通道；
- 位错促进 Ostwald ripening；
- 大 Ag2Te 附近不再观察到 Ag-rich nanoparticles，因为大颗粒已消耗附近 matrix Ag。

这比主文的表述更强。它说明实验中的粗化至少包含：

\[
\text{bulk matrix diffusion}
+
\text{small-object dissolution}
+
\text{dislocation pipe diffusion}.
\]

因此，当前不含位错和未解析小对象的 PF 模型：

- 可以给出 bulk-diffusion baseline；
- 可以研究 resolved particles 的质量守恒竞争；
- 不能无条件预测真实 48 h 粗化速率；
- 不能把实验–模拟时间差简单归因于扩散系数误差；
- 不应通过任意放大 \(D_{\rm Ag}\) 强行拟合。

实验快于模型时，优先解释为缺失的 pipe diffusion 或小对象供料通道。

---

## 3. 本征应变与实验 OR 的一致性

Yu 2024 给出的三个晶格失配为：

\[
+4.6\%,\qquad -2.2\%,\qquad -1.7\%.
\]

当前项目使用的本征应变主值为：

\[
[+0.046,\,-0.022,\,-0.017].
\]

二者在数值大小上逐项一致。这是非常重要的正面审计结果，说明当前本征应变不是任意参数，而与实验 OR 的晶格失配具有直接溯源关系。

对应关系为：

- OR-a：\(+0.046\)，高失配方向，实验 lath 最短尺寸方向；
- OR-b：\(-0.022\)，中等失配方向；
- OR-c：\(-0.017\)，最低失配方向，实验 lath 最长尺寸方向。

仍需代码级确认：

1. 三个主值在代码中的轴排序；
2. 本征应变从 Ag2Te 晶体坐标到 PbTe 模拟坐标的旋转矩阵；
3. 正负号约定；
4. 是否只固定了一个 crystallographic variant；
5. 当前弹性常数和失配是否对应室温单斜相，还是退火温度下的有效高温相。

---

## 4. 界面结构和界面能

SI 对 72 个 \((002)_{\rm PbTe}/(200)_{\rm Ag2Te}\) 界面平移构型进行了 DFT 筛选。

界面自由能范围约为：

\[
168\text{–}495\ {\rm mJ\,m^{-2}}.
\]

最低值出现在：

\[
[0.5,0.25],\qquad [0.5,0.75],
\]

分别约为：

\[
168,\qquad169\ {\rm mJ\,m^{-2}}.
\]

这些最低能界面与显微镜观察到的取向关系一致。

对当前 PF 的意义：

- 168–169 mJ m\(^{-2}\) 是特定取向、特定原子堆垛、0 K、低温单斜 Ag2Te 模型的界面能；
- 不能直接把它当作 380–400 °C 所有界面的各向同性有效 \(\gamma\)；
- 必须与当前 PF 使用的 surface energy 在定义、温度、面积归一化和界面取向上逐项对比；
- 若当前 PF 只使用单一 \(\gamma\)，则形貌各向异性主要来自 elasticity，而不是显式 \(\gamma(\mathbf n)\)；
- 若要使用 SI 的数据构建界面能各向异性，还缺少其他界面法向的完整能量面。

---

## 5. Debye–Callaway 模型完整结构

论文使用：

\[
\kappa_{\rm lat}
=
\frac{k_B}{2\pi^2v}
\left(\frac{k_BT}{\hbar}\right)^3
\int_0^{\Theta_D/T}
\tau_{\rm tot}(x)
\frac{x^4e^x}{(e^x-1)^2}\,dx.
\]

总散射率：

\[
\tau_{\rm tot}^{-1}
=
\tau_{U+N}^{-1}
+
\tau_{GB}^{-1}
+
\tau_{PD}^{-1}
+
\tau_{Pre}^{-1}
+
\tau_{DC}^{-1}
+
\tau_{DS}^{-1}.
\]

### 5.1 声子–声子散射

\[
\tau_U^{-1}+\tau_N^{-1}
=
A_N\frac{2(6\pi^2)^{1/3}k_B\bar V^{1/3}\gamma^2\omega^2T}
{\bar M v^3}.
\]

其中 \(A_N=1.5\) 是明确标注的 fitted factor。

因此该模型不是 parameter-free prediction，而是含经验标定的 reduced Debye–Callaway model。

### 5.2 晶界散射

\[
\tau_{GB}^{-1}=v/d.
\]

使用：

- AQ grain size：10.4 µm；
- 48 h：12.1 µm。

### 5.3 点缺陷散射

\[
\tau_{PD}^{-1}
=
\frac{\bar V\omega^4}{4\pi v^3}\Gamma.
\]

质量和尺寸失配均被计入，但只把 interstitial Ag 作为点缺陷输入。

使用：

- \(x_i=0.0069\)（AQ）；
- \(x_i=0.0034\)（annealed）；
- \(\epsilon=65\)。

这些 \(x_i\) 是实验 Ag atomic fraction，不等同于当前伪二元 PF 的 \(x_B^\alpha\)。

### 5.4 析出物散射

\[
\tau_{Pre}^{-1}
=
vN_P
\left(\sigma_S^{-1}+\sigma_l^{-1}\right)^{-1},
\]

短波/几何截面：

\[
\sigma_S=2\pi R^2,
\]

长波/Rayleigh 截面：

\[
\sigma_l
=
\frac{4}{9}\pi R^2
\left(\frac{\Delta D}{D_M}\right)^2
\left(\frac{\omega R}{v}\right)^4.
\]

即：

\[
\sigma_l\propto R^6\omega^4.
\]

AQ 的 small 和 big population 分别计算，再按 Matthiessen 规则相加。

### 5.5 位错散射

位错芯：

\[
\tau_{DC}^{-1}
=
N_D\frac{\bar V^{4/3}}{v^2}\omega^3.
\]

位错应变场：

\[
\tau_{DS}^{-1}\propto
C B_D^2N_D(\gamma+\gamma')^2\omega.
\]

Ag 偏聚通过额外 Grüneisen 参数：

\[
\gamma'=2.62
\]

增强位错应变场散射。

---

## 6. SI 中的关键热输运参数

| 参数 | 数值 |
|---|---:|
| \(A_N\) | 1.5，fitted |
| \(\gamma\) | 1.96 |
| \(\bar M\) | \(2.784\times10^{-25}\) kg |
| \(v\) | 1770 m s\(^{-1}\) |
| \(v_L\) | 3590 m s\(^{-1}\) |
| \(v_T\) | 1610 m s\(^{-1}\) |
| \(\Theta_D\) | 136 K |
| AQ grain size | 10.4 µm |
| 48 h grain size | 12.1 µm |
| AQ matrix Ag fraction | 0.0069 |
| 48 h matrix Ag fraction | 0.0034 |
| small radius | 2 nm |
| AQ big effective radius | 30 nm |
| 48 h big effective radius | 50 nm |
| matrix density | 8.383 g cm\(^{-3}\) |
| density difference | 0.063 g cm\(^{-3}\) |
| AQ \(N_{P,\rm small}\) | \(7.5\times10^{24}\) m\(^{-3}\) |
| 48 h \(N_{P,\rm small}\) | \(2.5\times10^{19}\) m\(^{-3}\) |
| AQ \(N_{P,\rm big}\) | \(1.9\times10^{21}\) m\(^{-3}\) |
| 48 h \(N_{P,\rm big}\) | \(9.2\times10^{18}\) m\(^{-3}\) |
| 48 h dislocation density | \(3.5\times10^{11}\) cm\(^{-2}\) |
| Burgers vector | \(4.56\times10^{-10}\) m |
| dislocation prefactor \(C\) | 0.96 |
| Poisson ratio | 0.218 |
| Ag near dislocations | 4 at.% |
| bulk modulus | \(4.1\times10^{10}\) Pa |
| annealing temperature in SI | 655 K |
| \(\gamma'\) | 2.62 |

注意：

- 主文写 653 K，而 SI 参数表使用 655 K，属于小的取整/录入差异；
- Table S2 中 \(V_m,V_i\) 的单位写成 m\(^{-3}\)，从物理量定义看应为 m\(^3\)/atom，属于明显单位排版问题；
- 实现代码时不能照抄错误单位。

---

## 7. 对“界面面积主曲线”假设的理论修正

由 SI 的析出物散射公式可直接推出：

### 短波/几何极限

\[
\tau_{Pre}^{-1}
\propto N_PR^2.
\]

对于球形颗粒：

\[
S_v=4\pi N_PR^2.
\]

所以在几何极限：

\[
\tau_{Pre}^{-1}\propto S_v.
\]

此时 interface area density 可能是充分描述符。

### 长波/Rayleigh 极限

\[
\tau_{Pre}^{-1}
\propto N_PR^6\omega^4.
\]

对多分散颗粒群：

\[
\tau_{Pre}^{-1}
\propto
\int R^6n(R)\,dR.
\]

此时 \(S_v\sim\int R^2n(R)dR\) 不足，必须知道 PSD 的高阶矩甚至完整 PSD。

因此最科学的问题不是预设：

\[
\kappa_L=\mathcal F(S_v),
\]

而是检验：

\[
\boxed{
\text{在哪些频率/尺寸区间 }S_v\text{ 足够，}
\text{在哪些区间必须保留 }M_6\text{ 或完整 PSD？}
}
\]

这为“最低充分组织描述符”提供了直接理论基础。

使用 SI 参数：

\[
\Delta D/D_M\approx0.063/8.383\approx7.5\times10^{-3}.
\]

令 \(\sigma_S=\sigma_l\)，可估算 crossover 条件：

\[
\omega R/v\approx16.8.
\]

相应频率约为：

- \(R=2\) nm：\(f_c\approx2.4\) THz；
- \(R=30\) nm：\(f_c\approx0.16\) THz；
- \(R=50\) nm：\(f_c\approx0.095\) THz。

因此：

- 大颗粒在绝大部分热声子频段更接近几何散射；
- 2 nm 小对象跨越 Rayleigh–几何过渡区，对 PSD 高频谱特别敏感；
- 第一篇不显式模拟 1–5 nm 对象时，无法完整重现 AQ 的频谱散射。

---

## 8. Yu 2024 输运模型的简化和局限

### 8.1 使用平均半径，不使用完整 PSD

模型只使用：

- small \(R=2\) nm；
- big \(R=30\) 或 50 nm；
- 对应 number density。

没有输入实际完整 PSD。

因此“动力学生成的 full PSD → κL(t)”仍是本项目的真实增量。

### 8.2 plate/lath 被等效为球形半径

SI 的 \(\sigma_S\) 和 \(\sigma_l\) 是球形截面。

尽管实验析出物是 plate/lath，模型仍使用单一平均半径：

- AQ big：30 nm；
- annealed big：50 nm。

所以 Yu 2024 没有真正计算 morphology/orientation-dependent phonon scattering。

### 8.3 键合失配没有被显式写进 precipitate cross-section

文章叙事强调：

- PbTe 为 metavalent bonding；
- Ag2Te 为 iono-covalent bonding；
- 键合和声子谱失配增强界面散射。

但 SI 的析出物散射截面只显式使用：

\[
\Delta D/D_M
\]

即密度失配，没有独立的界面透射率、力常数失配或 acoustic impedance 参数。

因此不能把该 Debye–Callaway 拟合视为“显式包含键合失配的预测模型”。

本项目若沿用 S7–S10，只能声称采用 density-contrast precipitate scattering；不能声称已定量模拟 metavalent/iono-covalent bonding mismatch。

### 8.4 背景模型含经验参数

\(A_N=1.5\) 被拟合；\(\epsilon=65\) 和 \(C=0.96\) 取自文献。

所以输运模块适合：

- 趋势和机制分解；
- 半定量预测；
- snapshot-to-trajectory extension。

不适合宣称无参数的绝对 \(\kappa_L\) 预测。

---

## 9. 对本项目的修订后定位

### 可做

1. 仅对 resolved Ag2Te population 做守恒三维 PF；
2. 从 PF 提取：
   \[
   N_v(t),n(R,t),S_v(t),M_6(t),f_p(t),g(r,t);
   \]
3. 将 full PSD 代入 S7–S10 的粒径积分版本；
4. 对比：
   - \(N_v\)；
   - \(\langle R\rangle\)；
   - \(S_v\)；
   - \(S_v+\mathrm{CV}\)；
   - \(M_6\)；
   - full PSD；
5. 得到 resolved-precipitate-only 的 \(\kappa_L(t)\) 贡献；
6. 将实验残差归因于：
   - 未解析 1–5 nm 对象；
   - 位错和 Cottrell atmospheres；
   - point defects；
   - 共格应变释放。

### 不可做

1. 不显式模拟 1–5 nm 对象却声称复现 AQ 全部声子散射；
2. 不模拟位错却声称完整预测 48 h \(\kappa_L\)；
3. 用 Sheskin 6 h/48 h matrix plateau 与 Yu AQ/48 h 组织静默拼接；
4. 将 DFT 168 mJ m\(^{-2}\) 直接视为 380–400 °C 各向同性 \(\gamma\)；
5. 用室温单斜 Ag2Te OR 无审计地代表高温退火相；
6. 仅使用 \(S_v\) 而不检验 PSD 高阶矩；
7. 把 Yu 2024 的 fitted Debye–Callaway 当作 parameter-free model。

---

## 10. 下一阶段硬门

1. 核对当前 eigenstrain 轴排序与 OR-a/b/c；
2. 核对当前 Ag2Te elastic constants 对应的相结构和温度；
3. 比较当前 PF \(\gamma\) 与 168–169 mJ m\(^{-2}\) 的定义；
4. 实现 Yu S1–S16 和 Table S2 的可复现基准；
5. 先复现 Yu 的 AQ/annealed \(\kappa_{\rm lat}(T)\) 曲线；
6. 将 average-radius 模型扩展为：
   \[
   \tau_{Pre}^{-1}
   =
   v\int n(R)
   \left(\sigma_S^{-1}+\sigma_l^{-1}\right)^{-1}dR;
   \]
7. 用合成 PSD 验证平均半径、\(S_v\)、\(M_6\) 与 full PSD 的误差；
8. 将 PF 输出转换成 transport-ready PSD；
9. 第一篇把位错项设为：
   - OFF：析出物-only prediction；
   - measured Yu \(N_D\)：实验边界场景；
   不能由 PF 自行生成；
10. 生产计算前明确选择：
    - Sheskin 6 h→48 h 平台课题；
    - 或 Yu AQ→48 h 多缺陷课题；
    两者不能共用一套初始化和验证合同。
