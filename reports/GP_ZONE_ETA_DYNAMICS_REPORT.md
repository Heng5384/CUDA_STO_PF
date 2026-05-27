# GP Zone 中 `eta / Y / phi` 迭代公式与动力学参数报告

## 1. 报告目的

这份报告面向当前 `gp_zone` 模型，说明：

1. `eta`（GP zone 序参量）当前实际使用的迭代公式是什么；
2. `xB_alpha / Y` 当前实际使用的 Cahn-Hilliard 型输运链是什么；
3. `phi` 当前使用的 Allen-Cahn 型方程是什么；
4. 这些方程对应的动力学参数、数值离散方式、代码默认值、以及最近一批 `Step 13–32` 中实际采用的验证基线是什么。

报告完全基于当前代码，不引入外部假设。

---

## 2. 当前 `gp_zone` 模型的主变量

当前 `gp_zone` 模型中有三个主场：

1. `phi`
   - `beta` 相（Ag2Te）序参量
   - `phi = 0` 表示非 `beta`
   - `phi = 1` 表示 `beta`

2. `eta`
   - GP zone 序参量
   - `eta = 0` 表示无 GP
   - `eta = 1` 表示完全 GP

3. `Y`
   - `xB_alpha` 的 logit 变量
   - `xB_alpha = sigmoid(Y)`
   - 实际扩散链不是直接更新 `xB_alpha`，而是更新 `Y`

辅助定义：

- `xB_GP = gp_xB_fixed`
- 当前通常取固定 GP 组成 `xB_GP ≈ 0.3529`

---

## 3. 三相权重函数

代码位置：

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/phase_functions.h`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu` 中 `phase_fractions_gp(...)`

当前平滑插值函数：

\[
h(s)=s^3(6s^2-15s+10)
\]

\[
h'(s)=30s^2(1-s)^2
\]

双阱函数导数在代码中通过 `g_prime_of_phi(...)` 使用，对应常见：

\[
g(s)=s^2(1-s)^2,\qquad g'(s)=2s(1-s)(1-2s)
\]

`gp_zone` 三相分数定义为：

\[
h_\beta = h(\phi)
\]

\[
h_{GP} = (1-h(\phi))\,h(\eta)
\]

\[
h_\alpha = (1-h(\phi))(1-h(\eta))
\]

因此守恒关系是：

\[
h_\alpha + h_{GP} + h_\beta = 1
\]

当前 `gp_zone` 的总 Ag 组成写成：

\[
xB_{tot}^{gp}=h_\alpha xB_\alpha + h_{GP}xB_{GP} + h_\beta
\]

因为 `beta` 相在当前写法里取纯 B 端，所以 `h_\beta` 前面的系数是 `1`。

---

## 4. `eta` 的 Allen-Cahn 方程

代码位置：

- `compute_eta_rhs_kernel(...)`
  - `/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:765`
- `eta_semi_implicit_update_kernel(...)`
  - `/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:835`
- 时间推进调用
  - `/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:11574`

### 4.1 实际使用的连续形式

当前代码中的 `eta` 驱动力写成：

\[
\frac{\partial \eta}{\partial t}
=-L_\eta\left(
\frac{\delta F}{\delta \eta}
\right)
\]

其中

\[
\frac{\delta F}{\delta \eta}
=
\underbrace{c_{ref}(1-h_\phi)h'_\eta\,(g_{GP}-g_\alpha)}_{\text{bulk chemical}}
+
\underbrace{W_\eta g'(\eta)}_{\text{double-well}}
+
\underbrace{\frac{\partial g_{el}}{\partial \eta}}_{\text{elastic, optional}}
-
\underbrace{\kappa_\eta \nabla^2 \eta}_{\text{gradient term}}
\]

代码里对应量为：

- `h_phi = h_of_phi(phi)`
- `hp_eta = h_prime_of_phi(eta)`（与 `h_prime_of_eta` 相同）
- `g_gp = (1-xB_GP) muA_GP + xB_GP muB_GP - gp_delta_g0`
- `g_alpha = (1-xB_alpha) muA_alpha + xB_alpha muB_alpha`
- `c_ref = 1 / Vm_alpha_of_xB(xB_alpha, Vm_alpha_0, dVm_alpha_dxB)`

因此显式 RHS 是：

\[
\text{rhs}_\eta
=
c_{ref}(1-h_\phi)h'_\eta(g_{GP}-g_\alpha)
W_\eta g'(\eta)
\left(\frac{\partial g_{el}}{\partial \eta}\right)
\]

代码中这一行是：

```cpp
rhs_r[idx] = dgbulk_deta + gp_W_eta * gp_eta + elastic_part;
```

这里 `rhs_r` 还不含 `-\kappa_\eta \nabla^2 \eta`，梯度项放在半隐式分母里处理。

### 4.2 `eta` 的半隐式离散

在 Fourier 空间里，代码实际更新是：

\[
\hat{\eta}^{\,n+1}
=
\frac{
\hat{\eta}^{\,n}-L_\eta\Delta t\,\widehat{\text{rhs}_\eta}
}{
1+L_\eta \Delta t \,\kappa_\eta k^2
}
\]

这正对应：

```cpp
double L_dt = L_eta * dt;
double denom = 1.0 + L_dt * kappa_eta * k2[idx];
eta_k_new = (eta_k_old - L_dt * rhs_k) / denom;
```

inverse FFT 回实空间后，会做：

1. `1/N` 归一化
2. 强制截断到 `[0,1]`

因此当前 `eta` 的数值更新是：

- **显式处理**：
  - bulk chemical
  - double-well
  - elastic
- **隐式处理**：
  - `\kappa_\eta \nabla^2 \eta`

---

## 5. `phi` 的 Allen-Cahn 方程

代码位置：

- `compute_phi_rhs_kernel(...)`
  - `/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:377`
- `phi_semi_implicit_update_kernel(...)`
  - `/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:741`
- 时间推进调用
  - `/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:11488`

### 5.1 实际使用的连续形式

当前 `phi` 的动力学也是 Allen-Cahn：

\[
\frac{\partial \phi}{\partial t}
=
-L_\phi \left(\frac{\delta F}{\delta \phi}\right)
\]

代码中的 `\delta F / \delta \phi` 由三部分组成：

\[
\frac{\delta F}{\delta \phi}
=
\underbrace{W g'(\phi)}_{\text{double-well}}
+
\underbrace{\partial_\phi g_{bulk}}_{\text{chemical bulk}}
+
\underbrace{\partial_\phi g_{el}}_{\text{elastic}}
-
\underbrace{\kappa_\phi \nabla^2 \phi}_{\text{gradient term}}
\]

其中 bulk 化学项在代码中是：

\[
\partial_\phi g_{bulk}
=
c_{bulk} h'(\phi)
\left[
\Delta \mu
-c_{bulk}\mu_{tot}
\left(
V_m^{comp}-V_m^\alpha+\frac{dV_m^\alpha}{dx_B}(x_B-v_B)
\right)
\right]
\]

对应代码：

```cpp
double partial_g_bulk = c_bulk * hp *
    (delta_mu - c_bulk * mu_total * volume_term);
f_phi += partial_g_bulk;
```

### 5.2 `phi` 的半隐式离散

在 Fourier 空间里更新是：

\[
\hat{\phi}^{\,n+1}
=
\frac{
\hat{\phi}^{\,n}-L_\phi \Delta t\,\widehat{\text{rhs}_\phi}
}{
1+L_\phi \Delta t\,\kappa_\phi k^2
}
\]

inverse FFT 后：

- 做 `1/N` 归一化
- 截断到近似 `[0,1]` 的安全范围

所以 `phi` 和 `eta` 在数值结构上是同类的：

- 同为 Allen-Cahn
- 同为“显式 bulk + 隐式梯度”

---

## 6. `xB_alpha / Y` 的 Cahn-Hilliard 型输运链

这是当前 `gp_zone` 最关键的部分。

代码位置：

- `compute_mu_C_gp_kernel(...)`
  - `/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:1116`
- `compute_flux_single_component_gp_kernel(...)`
  - `/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:1809`
- `compute_Y_rhs_gp_kernel(...)`
  - `/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:2054`
- `compute_Y_rhs_gp_conservative_kernel(...)`
  - `/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu`
- `gp_storage_exact_Y_update_kernel(...)`
  - `/Users/heng/Documents/GitHub/CUDA_STO_PF/cuda_kernels.cu:2356`
- 主时间推进
  - `/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:11766–12175`

### 6.1 基本思想

当前 `gp_zone` 里守恒量不是直接用 `xB_alpha` 显式扩散，而是：

1. 用 `Y = logit(xB_alpha)` 作为数值变量；
2. 先算 `\mu_C`
3. 再算 `J`
4. 再算 `divJ`
5. 最后根据所选 `gp_y_update_mode`，要么直接更新 `Y`，要么通过 `storage_exact` 代数重建 `xB_alpha`。

因此当前 `gp_zone` 的扩散主链始终是：

> `mu_C -> grad(mu_C) -> flux -> divJ -> (Y update or storage reconstruction)`

真正的差异发生在 **Y 更新阶段**。

### 6.2 化学势 `\mu_C`

当前代码里的主形式是：

\[
\mu_C
=
\frac{1}{V_m^\alpha(x_B)}\left(\mu_B-\mu_A\right)
\]

若启用弹性，则再加：

\[
\mu_C^{el}
=
-\epsilon'_{iso}\,\text{tr}(\sigma)
\]

代码实现：

```cpp
mu_C_val = c_ref * (muB - muA);
mu_C_val += -eps_iso_over_vB * sigma_hydro;
```

### 6.3 有效迁移率与通量

当前相依赖有效迁移率：

\[
M_{eff}
=
h_\alpha M_\alpha + h_{GP} M_{GP} + h_\beta M_\beta
\]

其中：

\[
M_\alpha = \frac{D_\alpha}{G_\alpha}
\]

`G_alpha` 由 `gamma_thermo_nonlinear(...)` 给出，是热力学因子。

代码：

```cpp
double M_eff = h_alpha * M_alpha + h_GP * gp_M_GP + h_beta * gp_M_beta;
```

通量链在当前实现中使用：

\[
J = M_{eff}\nabla \mu_C
\]

随后在 Fourier 空间取散度，得到 `divJ`。

### 6.4 `gp_y_update_mode` 的四种模式

当前代码支持四种 `gp_y_update_mode`：

1. `old_rhs`
2. `conservative_y_rhs`
3. `picard_storage`
4. `storage_exact`

其中：

- `old_rhs` 和 `conservative_y_rhs`
  - 都走 Fourier 半隐式 `Y` 更新链
- `picard_storage` 和 `storage_exact`
  - 都走本地点 storage-consistent 更新链

### 6.5 旧 `old_rhs` 更新模式

旧模式下，代码先写 `Y` 的 RHS：

\[
\text{rhs}_Y
=
divJ
-
\underbrace{\dot{h}_{GP}(xB_{GP}-x_B)+\dot{h}_\beta(1-x_B)}_{\text{storage source}}
-
\underbrace{\overline{D_Y}\nabla^2 Y}_{\text{semi-implicit correction}}
-
\underbrace{\gamma_{local}\,\dot{Y}_{prev}}_{\text{gamma term}}
\]

然后在 Fourier 空间做：

\[
\hat{Y}^{n+1}
=
\frac{\hat{Y}^{n}+\Delta t\,\widehat{\text{rhs}_Y}}
{1+\Delta t\,\overline{D_Y}k^2}
\]

这就是 `compute_Y_rhs_gp_kernel(...)` + `Y_semi_implicit_update_kernel(...)`。

### 6.6 新增 `conservative_y_rhs` 更新模式

`conservative_y_rhs` 是 `Step 32` 新增的可选模式，目标是避免 `storage_exact` 在

\[
h_\alpha \to 0
\]

时出现的代数奇异性。

从 GP storage identity

\[
xB_{tot}
=
h_\alpha xB_\alpha + h_{GP}xB_{GP} + h_\beta
\]

出发，对时间求导：

\[
h_\alpha \frac{dxB_\alpha}{dt}
=
\operatorname{div}J
- \dot{h}_{GP}(xB_{GP}-xB_\alpha)
- \dot{h}_\beta(1-xB_\alpha)
\]

再用

\[
xB_\alpha = \sigma(Y), \qquad q = xB_\alpha(1-xB_\alpha),
\]

得到

\[
h_\alpha q \frac{dY}{dt}
=
\operatorname{div}J
- \dot{h}_{GP}(xB_{GP}-xB_\alpha)
- \dot{h}_\beta(1-xB_\alpha)
\]

当前实现**不直接除以** `h_alpha q`，而是改写成 paper-style source form：

\[
\frac{dY}{dt}
=
\overline{D_Y}\nabla^2 Y + f_Y
\]

其中：

\[
f_Y
=
\operatorname{div}J
- \dot{h}_{GP}(xB_{GP}-xB_\alpha)
- \dot{h}_\beta(1-xB_\alpha)
- \overline{D_Y}\nabla^2 Y
- (h_\alpha q - 1)\dot{Y}_{lagged}
\]

代码里对应量是：

- `storage_source = dh_GP_dt * (xB_GP - xB) + dh_beta_dt * (1.0 - xB)`
- `hq = h_alpha_explicit * q`
- `gamma_local = hq - 1.0`
- `term_gamma = gamma_local * lagged_dYdt`
- `term_lap = mean_DY * lapY`
- `fY = divJ - storage_source - term_lap - term_gamma`

然后在 Fourier 空间更新：

\[
\hat{Y}^{\,n+1}
=
\frac{\hat{Y}^{\,n} + \Delta t\,\hat{f}_Y}
{1 + \Delta t\,\overline{D_Y}k^2}
\]

inverse FFT 后：

1. `1/N` 归一化
2. 按现有 `Y` 安全范围截断
3. 用 `sigmoid(Y)` 更新 `xB_alpha`

这个模式的关键点是：

> **它不再通过**
> \[
> xB_\alpha = \frac{xB_{tot}^{target}-h_{GP}xB_{GP}-h_\beta}{h_\alpha}
> \]
> **来代数重建 `xB_alpha`。**

### 6.7 `picard_storage`

`picard_storage` 仍然是 storage-consistent 更新，但它不是一次性代数精确重构，而是用 Picard 方式做本地点自洽迭代。

它的意义主要是：

- 在保持 storage consistency 的同时，尝试缓和一次性代数更新的刚性
- 但它仍然属于 “local storage-consistent update” 一类，不属于 `Y` Fourier RHS 一类

### 6.8 当前推荐主路径：`storage_exact`

当前项目中真正验证通过并推荐的 `gp_zone` 更新方式是：

```text
gp_y_update_mode = storage_exact
```

这一模式不再依赖上面的近似 RHS 闭合，而是直接用守恒关系代数求解。

其核心步骤是：

1. 用旧相分数和旧 `xB_alpha` 算：
   \[
   xB_{tot}^{old}
   =
   xB_\alpha^{old}
   +
   h_{GP}^{old}(xB_{GP}-xB_\alpha^{old})
   +
   h_\beta^{old}(1-xB_\alpha^{old})
   \]

2. 用扩散散度推进守恒量：
   \[
   xB_{tot}^{target}
   =
   xB_{tot}^{old}+\Delta t\,divJ
   \]

3. 在新相分数下直接解出：
   \[
   xB_\alpha^{n+1}
   =
   \frac{xB_{tot}^{target}-h_{GP}^{new}xB_{GP}-h_\beta^{new}}
   {h_\alpha^{new}}
   \]

4. 再通过数值安全约束得到：
   \[
   Y^{n+1} = \text{logit}(xB_\alpha^{n+1})
   \]

代码就是：

```cpp
double xBtot_target = xBtot_old + dt * divJ_r[idx];
double numerator = xBtot_target - h_GP_new * xB_GP - h_beta_new;
double xB_unclipped = numerator / h_alpha_new;
```

这一步是当前 `gp_zone` 质量守恒表现最好的主链。

但它也有一个已知弱点：

\[
xB_\alpha^{n+1}
=
\frac{xB_{tot}^{target}-h_{GP}^{new}xB_{GP}-h_\beta^{new}}
{h_\alpha^{new}}
\]

当 `eta -> 1` 且 `h_alpha -> 0` 时，这个重建会变得非常刚，甚至在 mature GP case 里表现为早期 blow-up。

这正是 `conservative_y_rhs` 被引入的原因。

---

## 7. 当前数值离散结构：Allen-Cahn + Cahn-Hilliard 的组合

当前代码可以总结为：

### `phi`
- 方程类型：Allen-Cahn
- 变量：`phi`
- 时间推进：Fourier 半隐式
- 显式项：bulk 化学 + double-well + elastic
- 隐式项：`\kappa_\phi \nabla^2 \phi`

### `eta`
- 方程类型：Allen-Cahn
- 变量：`eta`
- 时间推进：Fourier 半隐式
- 显式项：GP bulk 化学 + double-well + elastic
- 隐式项：`\kappa_\eta \nabla^2 \eta`

### `Y/xB_alpha`
- 方程类型：Cahn-Hilliard 型守恒输运链
- 变量：`Y = logit(xB_alpha)`
- 主链：`\mu_C -> J -> divJ -> Y/storage update`
- 可选更新模式：
  - `old_rhs`
  - `conservative_y_rhs`
  - `picard_storage`
  - `storage_exact`
- 目前已验证最强的两条代表性路线：
  - mass conservation 主链：`storage_exact`
  - `h_alpha \to 0` singularity 回避候选：`conservative_y_rhs`

---

## 8. 参数表：代码默认值

下面是 `params_default(...)` 中与 `gp_zone` 动力学直接相关的代码默认值。

代码位置：

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/main_cuda.cu:4635–4785`

### 8.1 GP 专用参数默认值

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `model_mode` | `two_phase` | 模型模式，`gp_zone` 需显式开启 |
| `gp_xB_fixed` | `0.3529` | 固定 GP 组成 |
| `gp_delta_g0` | `0.0` | GP 额外稳定化能量偏置 |
| `gp_W_eta` | `0.0` | `eta` 双阱系数 |
| `gp_kappa_eta` | `0.0` | `eta` 梯度能系数 |
| `gp_L_eta` | `0.0` | `eta` Allen-Cahn 动力学系数 |
| `gp_eps_iso` | `0.0` | GP 各向同性本征应变 |
| `gp_M_GP` | `0.0` | GP 相迁移率分量 |
| `gp_M_beta` | `0.0` | beta 相迁移率分量 |
| `gp_elastic_enabled` | `0` | 是否启用 GP elastic |
| `gp_elastic_active_eta` | `0` | 是否把 `dgel/deta` 喂回 `eta` RHS |
| `gp_elastic_active_phi` | `0` | 是否把 `dgel/dphi` 喂回 `phi` RHS |
| `gp_elastic_derivative_scale` | `1.0` | elastic 变分项缩放 |
| `gp_h_alpha_eps` | `1e-8` | `storage_exact` 中的小 `h_alpha` 阈值 |
| `gp_eta_mass_limiter` | `off` | `eta` 质量限制器默认关闭 |
| `gp_y_update_mode` | `old_rhs` | 默认仍是旧模式，支持 `old_rhs / conservative_y_rhs / picard_storage / storage_exact` |
| `gp_y_picard_iters` | `3` | `picard_storage` 迭代次数 |

### 8.2 数值稳定参数默认值

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `Y_clip` | `20.0` | `Y` 下界裁剪和 `logit/sigmoid` 数值安全参数 |
| `xB_eps` | `1e-8` | `xB_alpha` 的安全上下界 |
| `xB_s_floor` | `2e-4` | 额外数值下限保护 |
| `y_update_mass_projection_enabled` | `0` | 守恒 scalar projection，默认关闭 |
| `y_update_mass_projection_target_mode` | `pre_Y_update` | projection 目标模式 |
| `y_update_mass_projection_max_iter` | `30` | projection 最大迭代数 |
| `y_update_mass_projection_tol` | `1e-12` | projection 容差 |

### 8.3 `phi / Y` 主链的代码默认值

注意：下面这些不是“可直接运行的物理默认值”，而是**哨兵值**，必须由 `--pf-param-file` 提供。

| 参数 | 代码默认值 | 说明 |
|---|---:|---|
| `W` | `-1.0` | 必须由外部物理输入提供 |
| `kappa_phi` | `-1.0` | 必须由外部物理输入提供 |
| `L_phi` | `-1.0` | 必须由外部物理输入提供 |
| `D_alpha` | `-1.0` | 必须由外部物理输入提供 |
| `D_compound` | `-1.0` | 必须由外部物理输入提供 |
| `temperature_C` | `-1.0` | 必须由外部物理输入提供 |
| `mu_reference_scale` | `-1.0` | 必须由外部物理输入提供 |
| `v_A, v_B` | `-1.0` | 必须由外部物理输入提供 |
| `Vm_compound` | `-1.0` | 必须由外部物理输入提供 |
| `Vm_alpha_0` | `-1.0` | 必须由外部物理输入提供 |
| `dVm_alpha_dxB` | `-1.0` | 必须由外部物理输入提供 |
| `eps_iso_over_vB` | `-1.0` | 必须由外部物理输入提供 |

---

## 9. 示例物理输入下的主动力学参数

我用仓库里的示例输入：

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/physical_inputs.example.json`

通过：

```bash
python3 Unit_Psedobinary.py \
  --input-json physical_inputs.example.json \
  --output-pf-param-file /tmp/step_eta_report_example.params
```

换算出了当前示例物理参数文件。

### 9.1 示例输入下的主参数

| 参数 | 数值 |
|---|---:|
| `dt` | `1.0e-2` |
| `dx` | `0.1` nm |
| `temperature_C` | `400` |
| `mu_reference_scale` | `1.3779024e+04` |
| `W` | `1.0` |
| `kappa_phi` | `4.5` |
| `L_phi` | `2.3958248488` |
| `D_alpha` | `3.6e+03` |
| `D_compound` | `3.6e+01` |
| `Vm_alpha_0` | `1.0` |
| `Vm_compound` | `1.0` |
| `dVm_alpha_dxB` | `0.0` |
| `eps_iso_over_vB` | `2.33e-03` |
| `v_A` | `0.0` |
| `v_B` | `1.0` |

### 9.2 这个示例输入的意义

这组值主要说明：

1. `phi` 的 Allen-Cahn 双阱和梯度项目前示例是
   - `W = 1`
   - `kappa_phi = 4.5`
   - `L_phi ≈ 2.396`

2. `Y/xB` 扩散链的基准扩散系数目前示例是
   - `D_alpha = 3600`
   - `D_compound = 36`

3. 当前 `gp_zone` 中 `eta` 的系数并**不会**自动从 `Unit_Psedobinary.py` 生成，
   而是仍由 `gp_*` 参数单独指定。

---

## 10. 当前项目中真正验证过的 `gp_zone` 运行基线

在 `Step 13–32` 这批工作里，实际反复验证过的 `gp_zone` 基线不是代码默认值，而是下面这组。

脚本参考：

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/run_step26b_gp_growth_diagnostics.sh`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/run_step27_observed_gp_zone_diagnostics.sh`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/run_step29_observed_gp_diffuse_init.sh`
- `/Users/heng/Documents/GitHub/CUDA_STO_PF/run_step31b_eta1_box_depletion_validation.sh`

### 10.1 近期常用 `gp_zone` 覆盖项

| 参数 | 近期常用值 |
|---|---:|
| `model_mode` | `gp_zone` |
| `gp_nuc_enabled` | `0`（做纯 GP 诊断时） |
| `gp_to_beta_enabled` | `0`（做纯 GP 诊断时） |
| `y_update_mass_projection_enabled` | `0` |
| `gp_y_update_mode` | `storage_exact`（当前主守恒基线） |
| `gp_elastic_enabled` | `1` |
| `gp_elastic_active_eta` | `1` |
| `gp_elastic_active_phi` | `0` |
| `temperature_C` | `27` |
| `gp_delta_g0` | `0.0` |
| `dt` | `1e-5` |
| `dx` | `0.1` nm |

### 10.2 对这组基线的解释

这意味着当前我们真正用来判断 GP physics 的主方程链，不是：

- `old_rhs`
- 也不是 projection-on rescue mode

而是：

1. `phi`：Allen-Cahn，半隐式
2. `eta`：Allen-Cahn，半隐式
3. `Y/xB_alpha`：`storage_exact` 守恒重建链
4. `elastic`：对 `eta` 开启、对 `phi` 关闭

这是当前最有代表性的 `gp_zone` 评估主链。

### 10.3 `Step 32` 的新增验证候选

`Step 32` 新增的不是默认基线，而是一个**可选诊断候选**：

| 参数 | 值 |
|---|---:|
| `gp_y_update_mode` | `conservative_y_rhs` |
| `model_mode` | `gp_zone` |
| `gp_nuc_enabled` | `0` |
| `gp_to_beta_enabled` | `0` |
| `y_update_mass_projection_enabled` | `0` |

它的定位不是替代 `storage_exact` 的默认主链，而是回答：

> 在 mature GP、`eta \to 1`、`h_alpha \to 0` 的代表性 case 上，  
> 直接更新 `Y` 能否避免 `storage_exact` 的代数 blow-up。

### 10.4 `Step 32` representative single-case 已验证结果

当前已经完成了一个最有代表性的 mature GP 单 case：

| 参数 | 值 |
|---|---:|
| `model_mode` | `gp_zone` |
| `gp_y_update_mode` | `conservative_y_rhs` |
| `gp_nuc_enabled` | `0` |
| `gp_to_beta_enabled` | `0` |
| `y_update_mass_projection_enabled` | `0` |
| `xB_background` | `0.03` |
| `R_GP` | `1 nm` |
| `eta_peak` | `1.0` |
| `depletion_radius_factor` | `2` |
| `grid` | `96^3` |
| `dx` | `0.1 nm` |
| `dt` | `1e-5` |
| `steps` | `5000` |

这个 case 的结果是：

- 完整跑满 `5000` 步
- 无 `NaN/Inf`
- 无 `xB -> 0/1` clipping
- `phi` 全程保持 `0`
- `total_relative_drift \approx -1.738\times 10^{-5}`

更关键的是，这条 case 没有重演同配置下 `storage_exact` 的 early blow-up。

用 `h`-体积定义的 GP 体积和有效半径比较：

\[
V_h^{init}=4.1887902117\ \text{nm}^3,\qquad
V_h^{final}=4.1883494637\ \text{nm}^3
\]

\[
\frac{\Delta V_h}{V_h}\approx -1.052\times 10^{-4}
\]

\[
R_{eff,h}^{init}\approx 1.0000000005\ \text{nm},\qquad
R_{eff,h}^{final}\approx 0.9999649257\ \text{nm}
\]

因此，这条 representative mature GP case 在当前 backend 下更准确的分类是：

> **stable mature GP with extremely weak relaxation/shrinkage**

也就是：

- 没有明显增长
- 没有明显溶解
- 没有 `storage_exact` 的 `h_\alpha \to 0` 早期爆炸

---

## 11. 当前最值得监控的动力学参数

从最近多轮诊断看，当前 `gp_zone` 的稳定性最敏感的不是单个自由能公式本身，而是下面这组组合量：

### 11.1 `eta` 侧

- `gp_L_eta`
- `gp_kappa_eta`
- `gp_W_eta`
- `gp_delta_g0`

这四个决定：

1. `eta` 是快速松弛还是慢演化；
2. mature GP / embryo GP 是否会走向 plateau、衰减或增长。

### 11.2 `Y/xB` 侧

- `D_alpha`
- `gp_M_GP`
- `gp_M_beta`
- `Y_clip`
- `xB_eps`
- `gp_h_alpha_eps`
- `gp_y_update_mode`

这组决定：

1. `divJ` 强度；
2. `storage_exact` 反解时的可行性；
3. 是否容易触发 `xB` clipping。

### 11.3 实际运行时必须重点看四个诊断

1. `max_abs(dt*divJ)`
   - 这是当前最直观的 transport aggression 指标

2. `xB_clip_count_high / low`
   - 是否把 `xB_alpha` 顶到上下界

3. `total_relative_drift`
   - 全局质量漂移

4. `gp_closure_error`
   - `storage_exact` 闭合误差

如果当前运行的是 `conservative_y_rhs`，还必须额外看：

5. `min_h_alpha_for_Y_rhs`
6. `min_h_alpha_q_for_Y_rhs`
7. `max_abs_fY`
8. `max_abs_lagged_dYdt`

这四项用来判断：

- 是否真的进入 `h_alpha q \ll 1` 的危险区
- lagged `dY/dt` 修正项是否过大
- `fY` 是否已经被局部 source term 拉到非物理量级

---

## 12. 当前模型的可评估结论

基于当前代码，可以把 `gp_zone` 的动力学理解成：

### `phi`
- 标准 Allen-Cahn 型非守恒序参量
- 梯度项半隐式

### `eta`
- 标准 Allen-Cahn 型非守恒序参量
- 真正控制项是：
  - `gp_L_eta`
  - `gp_kappa_eta`
  - `gp_W_eta`
  - `gp_delta_g0`
  - `dgel/deta`

### `Y/xB_alpha`
- 不是简单的显式 `xB` 扩散
- 而是一个守恒输运链：
  - `mu_C`
  - `grad_mu`
  - `flux`
  - `divJ`
  - `Y` 更新或 `storage_exact` 重建

也就是说，当前 `gp_zone` 的数学结构是：

> **两个 Allen-Cahn 场 (`phi`, `eta`) + 一个 storage-constrained Cahn-Hilliard 型守恒输运场 (`Y/xB_alpha`)**

更准确一点说，`Y/xB_alpha` 当前有两类数值闭合：

1. **storage-reconstruction 类**
   - `picard_storage`
   - `storage_exact`

2. **direct-Y RHS 类**
   - `old_rhs`
   - `conservative_y_rhs`

---

## 13. 当前热力学 backend 的实际口径

代码位置：

- `/Users/heng/Documents/GitHub/CUDA_STO_PF/thermo_utils.h`

当前代码里，低浓度区域确实使用 regular-solution / CALPHAD 形式的化学势：

\[
\mu_{PbTe}^{calphad}(T,x_B),\qquad
\mu_{Ag_2Te}^{calphad}(T,x_B)
\]

但 solver 实际调用的不是全域“纯 regular solution”，而是：

- `mu_PbTe_raw(T, xB)`
- `mu_Ag2Te_raw(T, xB)`

这两个 `raw` 函数在

\[
x_B > X\_LIMIT\_CONVEX
\]

时，会切到凸化外推。当前代码常量是：

\[
X\_LIMIT\_CONVEX = 0.09
\]

外推形式是：

\[
\mu(x)=\mu(x_c)+\mu'(x_c)(x-x_c)+K_{penalty}(x-x_c)^2
\]

其中：

- `x_c = 0.09`
- `K_penalty = 5000`

这意味着当前所有已经完成的 `Step 27–32` mature GP 结果，包括：

- `storage_exact` mature-GP blow-up
- `conservative_y_rhs` mature-GP stable 5000-step single-case

都应解释成：

> **在“convexified thermodynamics backend”下的结果**

而不是“无凸化 true regular-solution backend”下的最终物理判定。

如果后面要严格评估：

- `T = 653 K`
- `no convex extrapolation`
- `true regular-solution mature GP stability`

那还需要新增一个最小的 thermodynamics mode 开关；这一步目前还没进入默认主链。

---

## 14. 对“现在用的参数是什么”的直接回答

如果只问“当前代码里写死的默认值是什么”，答案是：

- `gp_xB_fixed = 0.3529`
- `gp_delta_g0 = 0`
- `gp_W_eta = 0`
- `gp_kappa_eta = 0`
- `gp_L_eta = 0`
- `gp_M_GP = 0`
- `gp_M_beta = 0`
- `gp_elastic_enabled = 0`
- `gp_elastic_active_eta = 0`
- `gp_elastic_active_phi = 0`
- `gp_y_update_mode = old_rhs`

但这组默认值**不代表当前实际运行基线**，因为：

1. `W / kappa_phi / L_phi / D_alpha` 在代码默认中只是哨兵 `-1`
2. 真正运行必须依赖 `--pf-param-file`
3. 近期所有有意义的 `gp_zone` 主验证，实际都改成了：
   - `gp_y_update_mode = storage_exact`
   - `gp_elastic_enabled = 1`
   - `gp_elastic_active_eta = 1`
   - `gp_elastic_active_phi = 0`
   - `temperature_C = 27`
   - `dt = 1e-5`

同时，`Step 32` 新增了一个正在单独验证的代表性候选：

- `gp_y_update_mode = conservative_y_rhs`
- 不作为默认主链
- 仅用于测试 mature GP `eta \to 1` case 是否能回避 `storage_exact` 的 `h_alpha \to 0` 奇异性

所以如果要做“可评估”的模型审查，应该同时看：

1. **代码结构默认值**
2. **物理输入生成的 `phi/Y` 参数**
3. **`gp_zone` 实际覆写参数**

缺任何一层都会误判。

---

## 15. 建议的评估清单

如果你要评估当前 `gp_zone` 是否合理，建议最低限度同时检查：

1. `gp_y_update_mode`
   - 当前到底是 `storage_exact` 还是 `conservative_y_rhs`

2. `gp_L_eta`, `gp_kappa_eta`, `gp_W_eta`, `gp_delta_g0`
   - 它们决定 `eta` 是 embryo-like 还是 mature-like

3. `D_alpha`, `gp_M_GP`, `gp_M_beta`
   - 它们决定 transport 链是否过强

4. `max_abs(dt*divJ)`
   - 当前最敏感的稳定性指标

5. `xB_clip_count`
   - 是否存在局部不可行性

6. `total_relative_drift` 和 `gp_closure_error`
   - 是否真的守恒

7. 若是 `conservative_y_rhs`
   - `min_h_alpha_q_for_Y_rhs`
   - `max_abs_fY`
   - `max_abs_lagged_dYdt`
   - `mean_xBtot_gp_before_Y` vs `mean_xBtot_gp_after_Y_update`

8. 若在讨论“真实 regular-solution mature GP”
   - 必须先确认当前 backend 是否仍在使用 `xB > 0.09` 的 convex extrapolation
   - 否则不能把当前数值结果直接解释成“true thermodynamics”结论

---

## 16. 总结

当前 `gp_zone` 里：

1. `phi` 和 `eta` 都是 **Allen-Cahn**；
2. `xB_alpha` 通过 `Y=logit(xB_alpha)` 走 **Cahn-Hilliard 型守恒输运链**；
3. 当前最推荐、也最经验证的更新方式是：
   - `gp_y_update_mode = storage_exact`
4. `Step 32` 新增了一个可选候选：
   - `gp_y_update_mode = conservative_y_rhs`
   - 它的作用是回避 `h_alpha \to 0` 时的代数重建奇异性
   - 它还不是默认主链，但 representative single-case 已经验证：
     - `eta_peak = 1.0`
     - `R = 1 nm`
     - `xB = 0.03`
     - `dep = 2`
     - `96^3`
     可以稳定跑满 `5000` 步
5. 代码里的 `gp_*` 默认值多数只是占位，不代表实际验证基线；
6. 真正可评估的参数应当以：
   - `--pf-param-file` 生成的 `W / kappa_phi / L_phi / D_alpha`
   - 再加上 `gp_zone` 覆写项
   为准。
7. 当前所有 mature-GP 物理解读还必须区分：
   - **数值更新模式**：`storage_exact` vs `conservative_y_rhs`
   - **初始化模式**：hard seed / observed_gp_diffuse / profile-bundle
   - **热力学 backend**：当前默认仍是 convexified backend，而不是无凸化真 regular-solution

如果后续要进一步做“模型是否支持 mature GP、embryo GP、还是必须加额外稳定化”的科学判断，这份报告里的方程、`gp_y_update_mode` 选项和参数表可以作为统一基线。
