#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D


def detect_second_group(df):
    """
    只保留第二组（大间隔）数据。

    优先级：
    1) 如果已有显式分组列，则直接用
    2) 否则根据 r 的步长自动识别“大间隔组”
    """
    df = df.sort_values("r").reset_index(drop=True).copy()

    # ------------------------------------------------------------
    # 方案 A：已有显式分组列
    # ------------------------------------------------------------
    possible_group_cols = ["group", "data_group", "Group", "DataGroup", "dataset", "set_id"]
    for col in possible_group_cols:
        if col in df.columns:
            unique_vals = sorted(df[col].dropna().unique())
            # 常见情况：group=1/2
            if 2 in unique_vals:
                return df[df[col] == 2].copy()
            # 或 second / large_step / coarse 之类
            for target in ["second", "group2", "large_step", "coarse", "AR"]:
                mask = df[col].astype(str).str.lower() == target.lower()
                if mask.any():
                    return df[mask].copy()

    # ------------------------------------------------------------
    # 方案 B：按半径步长自动识别
    # 思路：第一组通常是小步长，第二组通常是大步长
    # ------------------------------------------------------------
    if len(df) < 4:
        return pd.DataFrame(columns=df.columns)

    r_vals = df["r"].to_numpy()
    dr = np.diff(r_vals)

    if len(dr) == 0:
        return pd.DataFrame(columns=df.columns)

    # 取最小正步长作为“细间隔”参考
    positive_dr = dr[dr > 1e-12]
    if len(positive_dr) == 0:
        return pd.DataFrame(columns=df.columns)

    dr_min = np.min(positive_dr)

    # 认为“大间隔”至少要比最小步长大 1.5 倍
    # 你如果知道两组间隔差更明显，可以改成 2.0
    threshold = 1.5 * dr_min

    # 找到第一个明显进入“大间隔”的位置
    split_idx = None
    for i, d in enumerate(dr):
        if d >= threshold:
            split_idx = i + 1
            break

    if split_idx is None:
        # 没找到明显的大间隔组，则返回空
        return pd.DataFrame(columns=df.columns)

    df_group2 = df.iloc[split_idx:].copy()

    # 再做一个安全检查：第二组至少要有两个点
    if len(df_group2) < 2:
        return pd.DataFrame(columns=df.columns)

    return df_group2


def process_and_plot_group2_only(input_folder='.'):
    output_dir = "Radius_analysis_result"
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # 读取所有      文件
    # ============================================================
    all_files = glob.glob(os.path.join(input_folder, "summary_T*_xB*.csv"))
    if not all_files:
        print("未找到任何 summary_T*_xB*.csv 文件。")
        return

    df_list = []
    for file in all_files:
        try:
            temp_df = pd.read_csv(file)
            temp_df["source_file"] = os.path.basename(file)
            df_list.append(temp_df)
        except Exception as e:
            print(f"读取 {file} 失败: {e}")

    df_all = pd.concat(df_list, ignore_index=True)

    # ============================================================
    # 自动识别目标列名
    # ============================================================
    candidate_cols = [
        "Len_x_Len_z",
        "LenX_LenZ",
        "len_x_len_z",
        "aspect_ratio_xz",
        "AR_xz"
    ]
    y_col = None
    for col in candidate_cols:
        if col in df_all.columns:
            y_col = col
            break

    if y_col is None:
        print("错误：没有找到 Len_x_Len_z 对应的列。")
        print("当前列名：", df_all.columns.tolist())
        return

    for col in ["T", "xB", "r"]:
        if col not in df_all.columns:
            print(f"错误：缺少必需列 {col}")
            return

    # 统一精度
    df_all["T_round"] = df_all["T"].round(6)
    df_all["xB_round"] = df_all["xB"].round(6)

    temperatures = sorted(df_all["T_round"].unique())
    xB_values = sorted(df_all["xB_round"].unique())

    # ============================================================
    # 绘图风格
    # ============================================================
    TITLE_FS = 16
    LABEL_FS = 16
    TICK_FS = 12
    LEGEND_FS = 12

    GLOBAL_T_MIN = 300.0
    GLOBAL_T_MAX = 500.0
    norm_T = mcolors.Normalize(vmin=GLOBAL_T_MIN, vmax=GLOBAL_T_MAX)
    cmap_T = plt.cm.coolwarm

    XB_MARKER_MAP = {
        0.030: 'o',
        0.035: 's',
        0.040: '^',
        0.045: 'D',
        0.050: 'P'
    }

    norm_xB = mcolors.Normalize(vmin=min(xB_values), vmax=max(xB_values))
    cmap_xB = plt.cm.viridis

    all_export_records = []

    # ============================================================
    # 1) 每个 xB 一张图：不同温度
    # ============================================================
    for xB_val in xB_values:
        fig, ax = plt.subplots(figsize=(10, 7))
        plotted_any = False
        export_records_this = []

        for T_val in temperatures:
            df_sub = df_all[
                (df_all["T_round"] == T_val) &
                (df_all["xB_round"] == xB_val)
            ].copy()

            if df_sub.empty:
                continue

            # 只保留 r <= 19
            df_sub = df_sub[df_sub["r"] <= 19].copy()
            if df_sub.empty:
                continue

            # 只取第二组
            df_group2 = detect_second_group(df_sub)
            if df_group2.empty:
                print(f"跳过 xB={xB_val:.3f}, T={T_val:.1f}: 没识别到第二组数据。")
                continue

            color = cmap_T(norm_T(T_val))

            ax.plot(
                df_group2["r"], df_group2[y_col],
                linestyle='--',
                color=color,
                linewidth=1.6,
                alpha=0.6,
                zorder=2
            )
            ax.plot(
                df_group2["r"], df_group2[y_col],
                linestyle='',
                marker='o',
                markerfacecolor=color,
                markeredgecolor='k',
                markeredgewidth=0.8,
                markersize=8,
                alpha=1.0,
                zorder=3
            )

            plotted_any = True

            for _, row in df_group2.iterrows():
                rec = {
                    "view_type": "by_xB",
                    "T_Celsius": T_val,
                    "xB": xB_val,
                    "r": row["r"],
                    y_col: row[y_col]
                }
                export_records_this.append(rec)
                all_export_records.append(rec)

        if plotted_any:
            ax.set_title(rf'AR vs. Radius $r$ ($x_B={xB_val:.3f}$)', fontsize=TITLE_FS, pad=15)
            ax.set_xlabel(r'Radius $r$ (nm)', fontsize=LABEL_FS)
            ax.set_ylabel('AR', fontsize=LABEL_FS)
            ax.set_xlim(2.5, 20)

            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=TICK_FS)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)

            temp_handles = []
            for T_val in temperatures:
                color = cmap_T(norm_T(T_val))
                temp_handles.append(
                    Line2D(
                        [0], [0],
                        linestyle='--',
                        color=color,
                        marker='o',
                        markerfacecolor=color,
                        markeredgecolor='k',
                        markeredgewidth=0.8,
                        markersize=10,
                        linewidth=1.5,
                        label=f'T = {T_val:.1f} $^\\circ$C'
                    )
                )

            ax.legend(
                handles=temp_handles,
                title='Temperature',
                loc='upper right',
                fontsize=LEGEND_FS,
                frameon=True,
                framealpha=0.9,
                edgecolor='#cccccc'
            )

            plt.tight_layout()

            xB_tag = f"{xB_val:.3f}".replace('.', 'p')
            fig_filename = os.path.join(output_dir, f"AR_vs_r_xB_{xB_tag}.png")
            plt.savefig(fig_filename, dpi=600, bbox_inches='tight')
            plt.close(fig)
            print(f"已保存图像: {fig_filename}")

            if export_records_this:
                csv_filename = os.path.join(output_dir, f"AR_vs_r_xB_{xB_tag}_group2_only.csv")
                pd.DataFrame(export_records_this).to_csv(csv_filename, index=False)
                print(f"已保存数据: {csv_filename}")
        else:
            plt.close(fig)
            print(f"xB = {xB_val:.3f} 没有符合条件的第二组数据可绘制。")

    # ============================================================
    # 2) 每个 T 一张图：不同饱和度
    # ============================================================
    for T_val in temperatures:
        fig, ax = plt.subplots(figsize=(10, 7))
        plotted_any = False
        export_records_this = []

        for xB_val in xB_values:
            df_sub = df_all[
                (df_all["T_round"] == T_val) &
                (df_all["xB_round"] == xB_val)
            ].copy()

            if df_sub.empty:
                continue

            # 只保留 r <= 19
            df_sub = df_sub[df_sub["r"] <= 19].copy()
            if df_sub.empty:
                continue

            # 只取第二组
            df_group2 = detect_second_group(df_sub)
            if df_group2.empty:
                print(f"跳过 T={T_val:.1f}, xB={xB_val:.3f}: 没识别到第二组数据。")
                continue

            color = cmap_xB(norm_xB(xB_val))
            marker = XB_MARKER_MAP.get(round(xB_val, 3), 'o')

            ax.plot(
                df_group2["r"], df_group2[y_col],
                linestyle='--',
                color=color,
                linewidth=1.6,
                alpha=0.6,
                zorder=2
            )
            ax.plot(
                df_group2["r"], df_group2[y_col],
                linestyle='',
                marker=marker,
                markerfacecolor=color,
                markeredgecolor='k',
                markeredgewidth=0.8,
                markersize=8,
                alpha=1.0,
                zorder=3
            )

            plotted_any = True

            for _, row in df_group2.iterrows():
                rec = {
                    "view_type": "by_temperature",
                    "T_Celsius": T_val,
                    "xB": xB_val,
                    "r": row["r"],
                    y_col: row[y_col]
                }
                export_records_this.append(rec)
                all_export_records.append(rec)

        if plotted_any:
            ax.set_title(rf'AR vs. Radius $r$ ($T={T_val:.1f}\,^{{\circ}}\mathrm{{C}}$)', fontsize=TITLE_FS, pad=15)
            ax.set_xlabel(r'Radius $r$ (nm)', fontsize=LABEL_FS)
            ax.set_ylabel('AR', fontsize=LABEL_FS)
            ax.set_xlim(2.5, 20)

            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=TICK_FS)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)

            xb_handles = []
            for xB_val in xB_values:
                color = cmap_xB(norm_xB(xB_val))
                marker = XB_MARKER_MAP.get(round(xB_val, 3), 'o')
                xb_handles.append(
                    Line2D(
                        [0], [0],
                        linestyle='--',
                        color=color,
                        marker=marker,
                        markerfacecolor=color,
                        markeredgecolor='k',
                        markeredgewidth=0.8,
                        markersize=10,
                        linewidth=1.5,
                        label=f'xB = {xB_val:.3f}'
                    )
                )

            ax.legend(
                handles=xb_handles,
                title='Saturation',
                loc='upper right',
                fontsize=LEGEND_FS,
                frameon=True,
                framealpha=0.9,
                edgecolor='#cccccc'
            )

            plt.tight_layout()

            T_tag = f"{T_val:.1f}".replace('.', 'p')
            fig_filename = os.path.join(output_dir, f"AR_vs_r_T_{T_tag}C.png")
            plt.savefig(fig_filename, dpi=600, bbox_inches='tight')
            plt.close(fig)
            print(f"已保存图像: {fig_filename}")

            if export_records_this:
                csv_filename = os.path.join(output_dir, f"AR_vs_r_T_{T_tag}C.csv")
                pd.DataFrame(export_records_this).to_csv(csv_filename, index=False)
                print(f"已保存数据: {csv_filename}")
        else:
            plt.close(fig)
            print(f"T = {T_val:.1f} °C 没有符合条件的第二组数据可绘制。")

    # ============================================================
    # 汇总导出
    # ============================================================
    if all_export_records:
        csv_all_filename = os.path.join(output_dir, f"AR_vs_r_all_views_group2_only.csv")
        pd.DataFrame(all_export_records).to_csv(csv_all_filename, index=False)
        print(f"已保存总数据: {csv_all_filename}")

    print("-" * 60)
    print("完成：只绘制第二组（AR 用的大间隔）数据。")
    print("不再绘制用于 nucleation barrier / critical radius 的第一组数据。")


if __name__ == "__main__":
    process_and_plot_group2_only()