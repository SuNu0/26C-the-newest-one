from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from q234_solve import DynamicQuantileParameters, Parameters, dynamic_net_forecast, load_inputs


OUT = ROOT / "论文图片"
RESULTS = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

FONT_CANDIDATES = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]
AVAILABLE_FONTS = {f.name for f in font_manager.fontManager.ttflist}
ZH_FONT = next((f for f in FONT_CANDIDATES if f in AVAILABLE_FONTS), "DejaVu Sans")

plt.rcParams.update(
    {
        "font.family": [ZH_FONT, "Times New Roman"],
        "axes.unicode_minus": False,
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "figure.dpi": 120,
    }
)

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"


def style_axis(ax: plt.Axes, grid: str | None = "y") -> None:
    if grid:
        ax.grid(axis=grid, color="#D7DCE2", linewidth=0.55, linestyle="--", alpha=0.75)
    ax.set_axisbelow(True)


def panel_labels(axes) -> None:
    axes_arr = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes_arr):
        ax.text(
            -0.08,
            1.04,
            f"({chr(97 + i)})",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
        )


def save(fig: plt.Figure, number: int, caption: str) -> None:
    path = OUT / f"图{number}_{caption}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)
    print(f"saved: {path.name}")


def hour_ticks(ax: plt.Axes) -> None:
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.set_xlabel("时刻 / h")


params = Parameters()
input_data = load_inputs(params)
dates = pd.DatetimeIndex(input_data.dates)
formal_mask = dates >= pd.Timestamp("2025-02-01")
formal_dates = dates[formal_mask]
load_kw = input_data.load_kwh / params.dt_hours
pv_kw = input_data.pv_kwh / params.dt_hours
net_kw = load_kw - pv_kw
var_price = input_data.variable_price

detail_path = RESULTS / "问题2至4_逐时段完整结果.csv"
detail = pd.read_csv(detail_path, parse_dates=["日期"])
detail = detail.loc[detail["日期"] >= pd.Timestamp("2025-02-01")].copy()
q2 = detail.loc[detail["策略"] == "问题2_固定价"].copy()
q3 = detail.loc[detail["策略"] == "问题3_固定价"].copy()
q42 = detail.loc[detail["策略"] == "问题4-2_波动价"].copy()
q43 = detail.loc[detail["策略"] == "问题4-3_波动价"].copy()

q1 = pd.read_csv(RESULTS / "问题1_逐时段调度.csv")
q1_eff = pd.read_csv(RESULTS / "问题1_效率敏感性.csv")
folds = pd.read_csv(RESULTS / "问题二至四模型检验" / "五折时间顺序预测检验.csv")
voi8 = pd.read_csv(RESULTS / "问题3_新增预报时点_VOI.csv", parse_dates=["日期"])
voi8 = voi8.loc[voi8["日期"] >= pd.Timestamp("2025-02-01")].copy()


# 图2：净负荷分布与工作日差异
formal_net = net_kw[formal_mask].ravel()
daily_load = input_data.load_kwh[formal_mask].sum(axis=1)
weekday = np.array([d.weekday() < 5 for d in formal_dates])
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
axes[0].hist(formal_net, bins=60, density=True, color=SKY, edgecolor="white", linewidth=0.25)
axes[0].axvline(formal_net.mean(), color=VERMILLION, linestyle="--", linewidth=1.25, label=f"均值 {formal_net.mean():.0f} kW")
axes[0].axvline(np.median(formal_net), color=BLUE, linestyle=":", linewidth=1.4, label=f"中位数 {np.median(formal_net):.0f} kW")
axes[0].set_xlabel("净负荷功率 / kW")
axes[0].set_ylabel("概率密度")
axes[0].set_title("十分钟净负荷分布")
axes[0].legend(frameon=False, loc="upper left")
style_axis(axes[0])

rng = np.random.default_rng(20260912)
groups = [daily_load[weekday] / 1000, daily_load[~weekday] / 1000]
bp = axes[1].boxplot(groups, positions=[1, 2], widths=0.48, showfliers=False, patch_artist=True,
                     medianprops={"color": "black", "linewidth": 1.1})
for patch, color in zip(bp["boxes"], [BLUE, ORANGE]):
    patch.set_facecolor(color)
    patch.set_alpha(0.25)
    patch.set_edgecolor(color)
for pos, vals, color in zip([1, 2], groups, [BLUE, ORANGE]):
    jitter = rng.normal(0, 0.045, size=len(vals))
    axes[1].scatter(np.full(len(vals), pos) + jitter, vals, s=8, alpha=0.35, color=color, edgecolors="none")
axes[1].set_xticks([1, 2], [f"工作日\n(n={len(groups[0])})", f"非工作日\n(n={len(groups[1])})"])
axes[1].set_ylabel("日负荷电量 / MWh")
axes[1].set_title("工作日与非工作日日负荷")
style_axis(axes[1])
panel_labels(axes)
save(fig, 2, "净负荷分布与工作日差异")


# 图3：主要变量月度变化与相关关系
daily = pd.DataFrame(
    {
        "日期": formal_dates,
        "负荷": input_data.load_kwh[formal_mask].sum(axis=1) / 1000,
        "光伏": input_data.pv_kwh[formal_mask].sum(axis=1) / 1000,
        "净负荷": (input_data.load_kwh[formal_mask] - input_data.pv_kwh[formal_mask]).sum(axis=1) / 1000,
        "电价": var_price[formal_mask].mean(axis=1),
    }
)
daily["月份"] = daily["日期"].dt.month
monthly = daily.groupby("月份")[["负荷", "光伏", "净负荷", "电价"]].mean()
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)
for col, color, marker, ls in [
    ("负荷", BLUE, "o", "-"),
    ("光伏", ORANGE, "s", "--"),
    ("净负荷", GREEN, "^", "-."),
]:
    axes[0].plot(monthly.index, monthly[col], color=color, marker=marker, linestyle=ls, linewidth=1.4, markersize=3.5, label=col)
axes[0].set_xlabel("月份")
axes[0].set_ylabel("日均电量 / MWh")
axes[0].set_xticks(monthly.index)
axes[0].set_title("负荷、光伏与净负荷的月度变化")
axes[0].legend(frameon=False, ncol=3, loc="upper center")
style_axis(axes[0])

corr_data = np.column_stack(
    [load_kw[formal_mask].ravel(), pv_kw[formal_mask].ravel(), var_price[formal_mask].ravel(), net_kw[formal_mask].ravel()]
)
corr = np.corrcoef(corr_data, rowvar=False)
im = axes[1].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
labels = ["负荷", "光伏", "波动电价", "净负荷"]
axes[1].set_xticks(range(4), labels, rotation=25, ha="right")
axes[1].set_yticks(range(4), labels)
for i in range(4):
    for j in range(4):
        axes[1].text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                     color="white" if abs(corr[i, j]) > 0.55 else "black")
cb = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
cb.set_label("Pearson 相关系数")
axes[1].set_title("主要变量的逐时相关关系")
panel_labels(axes)
save(fig, 3, "主要变量月度变化与相关关系")


# 图4：问题一两阶段MILP流程
fig, ax = plt.subplots(figsize=(7.2, 2.35), constrained_layout=True)
ax.set_xlim(0, 12)
ax.set_ylim(0, 3)
ax.axis("off")
box_texts = [
    "数据与量纲统一\n10 min，kW→kWh",
    "建立物理层MILP\n平衡、SOC、二元互斥",
    "第一层经济目标\n严格最小化购电费",
    "锁定最优费用\n" + r"$C\leq C^*+\varepsilon$",
    "第二层运行目标\n最小化充放电吞吐",
    "回代与检验\n残差、边界、成本对账",
]
xs = np.linspace(0.2, 10.2, 6)
colors = ["#E8F1F8", "#E8F1F8", "#FFF2CC", "#F3E8FF", "#E8F6EF", "#EEF0F2"]
for i, (x, txt, color) in enumerate(zip(xs, box_texts, colors)):
    box = FancyBboxPatch((x, 1.0), 1.55, 1.0, boxstyle="round,pad=0.05,rounding_size=0.08",
                         linewidth=1.0, edgecolor=BLUE if i < 3 else GRAY, facecolor=color)
    ax.add_patch(box)
    ax.text(x + 0.775, 1.5, txt, ha="center", va="center", fontsize=7.4)
    if i < 5:
        ax.add_patch(FancyArrowPatch((x + 1.56, 1.5), (xs[i + 1] - 0.04, 1.5), arrowstyle="-|>",
                                     mutation_scale=9, linewidth=1.0, color=GRAY))
ax.text(6.0, 2.55, "费用优先、运行规范性次之；第二层不改变题目规定的最优费用", ha="center", va="center",
        fontsize=8.5, color=VERMILLION)
save(fig, 4, "问题一两阶段混合整数规划求解流程")


# 图5：问题一SOC与动作
x_q1 = np.arange(len(q1)) / 6 + 1 / 6
fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.3), sharex=True, constrained_layout=True)
axes[0].bar(x_q1, q1["充电量_kWh"], width=0.13, color=BLUE, alpha=0.85, label="充电")
axes[0].bar(x_q1, -q1["放电量_kWh"], width=0.13, color=ORANGE, alpha=0.85, label="放电（负向显示）")
axes[0].axhline(0, color="black", linewidth=0.6)
axes[0].set_ylabel("交流侧电量 / kWh")
axes[0].set_title("储能充放电动作")
axes[0].legend(frameon=False, ncol=2, loc="upper right")
style_axis(axes[0])
axes[1].plot(x_q1, q1["期末储电量_kWh"], color=GREEN, linewidth=1.6, label="期末SOC")
axes[1].axhline(1200, color=GRAY, linestyle="--", linewidth=0.9, label="上下界")
axes[1].axhline(10800, color=GRAY, linestyle="--", linewidth=0.9)
axes[1].fill_between(x_q1, 1200, 10800, color=GREEN, alpha=0.06)
axes[1].set_ylabel("电池内部电量 / kWh")
axes[1].set_title("储能状态轨迹")
axes[1].legend(frameon=False, loc="upper right")
style_axis(axes[1])
hour_ticks(axes[1])
panel_labels(axes)
save(fig, 5, "问题一储能状态与充放电行为")


# 图6：问题一有无储能策略与成本
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
axes[0].plot(x_q1, q1["无储能购电量_kWh"], color=GRAY, linestyle="--", linewidth=1.1, label="无储能")
axes[0].plot(x_q1, q1["购电量_kWh"], color=BLUE, linewidth=1.35, label="有储能")
axes[0].set_ylabel("购电量 / kWh")
axes[0].set_title("逐时购电量")
axes[0].legend(frameon=False)
hour_ticks(axes[0])
style_axis(axes[0])
costs = [q1["无储能购电费_元"].sum(), q1["购电费_元"].sum()]
bars = axes[1].bar([0, 1], costs, color=[GRAY, BLUE], width=0.58)
axes[1].set_xticks([0, 1], ["无储能", "有储能"])
axes[1].set_ylabel("全天购电费用 / 元")
axes[1].set_ylim(0, max(costs) * 1.18)
axes[1].set_title("全天成本对比")
for b, v in zip(bars, costs):
    axes[1].text(b.get_x() + b.get_width() / 2, v + max(costs) * 0.025, f"{v:,.0f}", ha="center", va="bottom")
axes[1].text(0.5, max(costs) * 1.08, "节约 12,925.10 元（26.90%）", ha="center", color=VERMILLION)
style_axis(axes[1])
panel_labels(axes)
save(fig, 6, "问题一有无储能购电策略对比")


# 图7：问题一效率敏感性
rt_eff = q1_eff["往返效率"].to_numpy() * 100
cost = q1_eff["最优购电费_元"].to_numpy()
saving_rate = (q1["无储能购电费_元"].sum() - cost) / q1["无储能购电费_元"].sum() * 100
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
axes[0].plot(rt_eff, cost / 10000, color=BLUE, marker="o", linewidth=1.4)
axes[0].scatter([81], [35126.94859928963 / 10000], color=VERMILLION, marker="D", s=32, zorder=3, label="基准效率")
axes[0].set_xlabel("往返效率 / %")
axes[0].set_ylabel("最优购电费用 / 万元")
axes[0].set_title("效率与购电费用")
axes[0].legend(frameon=False)
style_axis(axes[0])
axes[1].plot(rt_eff, saving_rate, color=GREEN, marker="s", linestyle="--", linewidth=1.4)
axes[1].scatter([81], [26.8981217421], color=VERMILLION, marker="D", s=32, zorder=3)
axes[1].set_xlabel("往返效率 / %")
axes[1].set_ylabel("相对无储能节约率 / %")
axes[1].set_title("效率与经济收益")
style_axis(axes[1])
panel_labels(axes)
save(fig, 7, "问题一储能效率敏感性")


# 重新构造问题二逐日动态净负荷预测（使用结果文件记录的当日分位数）
alpha_by_date = q2.groupby("日期")["选择净负荷分位数"].first()
cfg = DynamicQuantileParameters()
forecasts = []
actuals = []
forecast_dates = []
for day_i, day in enumerate(dates):
    if day < pd.Timestamp("2025-02-01"):
        continue
    alpha = float(alpha_by_date.loc[day])
    forecasts.append(dynamic_net_forecast(input_data, day_i, alpha, cfg))
    actuals.append(input_data.load_kwh[day_i] - input_data.pv_kwh[day_i])
    forecast_dates.append(day)
forecasts = np.asarray(forecasts)
actuals = np.asarray(actuals)
forecast_dates = pd.DatetimeIndex(forecast_dates)
errors = actuals - forecasts
daily_rmse = np.sqrt(np.mean(errors**2, axis=1))
representative_i = int(np.argmin(np.abs(daily_rmse - np.median(daily_rmse))))
print(
    "q2 forecast audit:",
    forecast_dates[representative_i].date(),
    "RMSE=", float(np.sqrt(np.mean(errors**2))),
    "MAE=", float(np.mean(np.abs(errors))),
)


# 图8：实际净负荷与预测误差
x = np.arange(144) / 6 + 1 / 6
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
axes[0].plot(x, actuals[representative_i], color=BLUE, linewidth=1.35, label="实际净负荷")
axes[0].plot(x, forecasts[representative_i], color=ORANGE, linestyle="--", linewidth=1.25, label="日前预测")
axes[0].axhline(0, color=GRAY, linewidth=0.7)
axes[0].set_ylabel("十分钟电量 / kWh")
axes[0].set_title(f"代表日：{forecast_dates[representative_i]:%Y-%m-%d}")
axes[0].legend(frameon=False, loc="upper right")
hour_ticks(axes[0])
style_axis(axes[0])
axes[1].hist(errors.ravel(), bins=70, density=True, color=SKY, edgecolor="white", linewidth=0.25)
axes[1].axvline(0, color=GRAY, linewidth=0.8)
axes[1].axvline(errors.mean(), color=VERMILLION, linestyle="--", linewidth=1.2, label=f"均值 {errors.mean():.1f} kWh")
axes[1].set_xlabel("实际净负荷－日前预测 / kWh")
axes[1].set_ylabel("概率密度")
axes[1].set_title("正式评价期预测误差分布")
axes[1].legend(frameon=False)
style_axis(axes[1])
panel_labels(axes)
save(fig, 8, "实际净负荷与日前预测误差")


# 图9：动态分位数与应急购电月度分布
daily_q2 = q2.groupby("日期").agg(
    分位数=("选择净负荷分位数", "first"),
    紧急购电=("紧急购电量_kWh", "sum"),
)
monthly_q2 = daily_q2.resample("MS").agg({"分位数": "mean", "紧急购电": "sum"})
fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.25), constrained_layout=True)
axes[0].step(daily_q2.index, daily_q2["分位数"], where="post", color=BLUE, linewidth=1.15)
axes[0].scatter(daily_q2.index[::7], daily_q2["分位数"].iloc[::7], color=BLUE, s=9)
axes[0].set_ylim(0.47, 0.98)
axes[0].set_ylabel("选择分位数")
axes[0].set_title("按历史运行成本滚动选择的分位数")
axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
style_axis(axes[0])
bars = axes[1].bar(monthly_q2.index.month, monthly_q2["紧急购电"] / 1000, color=ORANGE, width=0.68)
axes[1].set_xticks(monthly_q2.index.month)
axes[1].set_xlabel("月份")
axes[1].set_ylabel("紧急购电量 / MWh")
axes[1].set_title("问题二紧急购电的月度分布")
style_axis(axes[1])
panel_labels(axes)
save(fig, 9, "动态分位数与应急购电月度分布")


# 图10：时间顺序样本外预测检验
folds["系列"] = folds["变量"].map({"净负荷": "日前净负荷", "负荷": "滚动负荷", "光伏": "滚动光伏"})
series_order = ["日前净负荷", "滚动负荷", "滚动光伏"]
means = folds.groupby("系列")[["RMSE_kWh", "周滞后基线RMSE_kWh"]].mean().reindex(series_order)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
pos = np.arange(3)
w = 0.34
axes[0].bar(pos - w / 2, means["RMSE_kWh"], width=w, color=BLUE, label="本文预测器")
axes[0].bar(pos + w / 2, means["周滞后基线RMSE_kWh"], width=w, color=GRAY, label="周滞后基线")
axes[0].set_xticks(pos, series_order)
axes[0].set_ylabel("五折平均 RMSE / kWh")
axes[0].set_title("预测误差对比")
axes[0].legend(frameon=False)
style_axis(axes[0])
for i, label in enumerate(series_order):
    sub = folds.loc[folds["系列"] == label]
    axes[1].scatter(np.full(len(sub), i) + rng.normal(0, 0.035, len(sub)), sub["相对基线RMSE改善_百分比"],
                    color=[ORANGE, GREEN, PURPLE][i], s=20, alpha=0.75, label=label)
    axes[1].plot([i - 0.16, i + 0.16], [sub["相对基线RMSE改善_百分比"].mean()] * 2, color="black", linewidth=1.2)
axes[1].axhline(0, color=GRAY, linewidth=0.8, linestyle="--")
axes[1].set_xticks(pos, series_order)
axes[1].set_ylabel("相对基线 RMSE 改善 / %")
axes[1].set_title("各时间折的相对改善（横线为均值）")
style_axis(axes[1])
panel_labels(axes)
save(fig, 10, "时间顺序样本外预测检验")


# 图11：四时点滚动优化机制
fig, ax = plt.subplots(figsize=(7.2, 3.35), constrained_layout=True)
ax.set_xlim(-0.8, 24.8)
ax.set_ylim(-2.35, 2.2)
ax.axis("off")
issue_hours = [0, 6, 12, 18, 24]
segment_colors = [BLUE, ORANGE, GREEN, PURPLE]
for i in range(4):
    ax.plot([issue_hours[i], issue_hours[i + 1]], [1.05, 1.05], color=segment_colors[i], linewidth=5, solid_capstyle="butt")
    ax.text((issue_hours[i] + issue_hours[i + 1]) / 2, 1.38, f"执行 {issue_hours[i]}:00--{issue_hours[i + 1]}:00", ha="center", fontsize=7.4)
for h in issue_hours:
    ax.scatter(h, 1.05, s=42, color="white", edgecolor=GRAY, linewidth=1.2, zorder=3)
    ax.text(h, 0.70, f"{h}:00", ha="center", fontsize=7.5)
for h in issue_hours[:-1]:
    ax.text(h, 1.72, "获取最新预测\n继承实际SOC", ha="center", va="center", fontsize=7.1,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#EEF5FA", "edgecolor": BLUE, "linewidth": 0.8})

flow_texts = ["冻结已执行决策", "优化剩余时域", "执行至下一节点", "SOC与信息回写"]
flow_x = [1.2, 7.2, 13.2, 19.2]
for i, (x0, txt) in enumerate(zip(flow_x, flow_texts)):
    box = FancyBboxPatch((x0, -1.35), 3.6, 0.8, boxstyle="round,pad=0.05,rounding_size=0.06",
                         linewidth=0.9, edgecolor=GRAY, facecolor="#F5F6F7")
    ax.add_patch(box)
    ax.text(x0 + 1.8, -0.95, txt, ha="center", va="center", fontsize=7.6)
    if i < 3:
        ax.add_patch(FancyArrowPatch((x0 + 3.65, -0.95), (flow_x[i + 1] - 0.08, -0.95), arrowstyle="-|>",
                                     mutation_scale=9, linewidth=1.0, color=GRAY))
ax.add_patch(FancyArrowPatch((22.8, -1.5), (1.25, -1.5), connectionstyle="arc3,rad=-0.10", arrowstyle="-|>",
                             mutation_scale=9, linewidth=0.9, color=BLUE))
ax.text(12, -2.05, "预测更新—重新优化—执行—状态回写；历史区间始终不可回改", ha="center", fontsize=8.1, color=VERMILLION)
save(fig, 11, "四时点滚动优化的信息更新机制")


# 图12：问题三代表日滚动调整
typical_date = pd.Timestamp("2025-09-23")
typical = q3.loc[q3["日期"] == typical_date].sort_values("时段序号")
tx = np.arange(len(typical)) / 6 + 1 / 6
fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.5), sharex=True, constrained_layout=True)
axes[0].plot(tx, typical["计划购电量_kWh"], color=GRAY, linestyle="--", linewidth=1.1, label="日前计划")
axes[0].plot(tx, typical["调整后购电量_kWh"], color=BLUE, linewidth=1.3, label="滚动调整后")
axes[0].set_ylabel("购电量 / kWh")
axes[0].set_title(f"{typical_date:%Y-%m-%d} 购电计划")
axes[0].legend(frameon=False, ncol=2, loc="upper right")
style_axis(axes[0])
axes[1].bar(tx, typical["上调量_kWh"], width=0.13, color=GREEN, label="上调")
axes[1].bar(tx, -typical["下调量_kWh"], width=0.13, color=PURPLE, label="下调（负向显示）")
axes[1].plot(tx, typical["紧急购电量_kWh"], color=VERMILLION, linewidth=1.1, label="紧急购电")
axes[1].axhline(0, color="black", linewidth=0.6)
axes[1].set_ylabel("调整或补购量 / kWh")
axes[1].set_title("调整量与紧急购电")
axes[1].legend(frameon=False, ncol=3, loc="upper right")
style_axis(axes[1])
axes[2].plot(tx, typical["期末SOC_kWh"], color=ORANGE, linewidth=1.45)
axes[2].axhline(1200, color=GRAY, linestyle="--", linewidth=0.8)
axes[2].axhline(10800, color=GRAY, linestyle="--", linewidth=0.8)
axes[2].set_ylabel("期末SOC / kWh")
axes[2].set_title("实际SOC继承轨迹")
style_axis(axes[2])
hour_ticks(axes[2])
panel_labels(axes)
save(fig, 12, "日前计划与四时点滚动调整的典型日对比")


# 图13：四时点与八时点信息价值
q3_daily = q3.groupby("日期")["总成本_元"].sum()
voi8_daily = voi8.groupby("日期")["总成本_元"].sum()
aligned = pd.concat([q3_daily.rename("四时点"), voi8_daily.rename("八时点")], axis=1).dropna()
daily_voi = aligned["四时点"] - aligned["八时点"]
totals = aligned.sum()
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
bars = axes[0].bar([0, 1], totals.values / 1e6, color=[BLUE, ORANGE], width=0.58)
axes[0].set_xticks([0, 1], ["四时点", "八时点代理"])
axes[0].set_ylabel("正式评价期总成本 / 百万元")
axes[0].set_ylim(0, max(totals.values / 1e6) * 1.17)
axes[0].set_title("更新频率与总成本")
for b, v in zip(bars, totals.values / 1e6):
    axes[0].text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.3f}", ha="center")
axes[0].text(0.5, max(totals.values / 1e6) * 1.08, "毛信息价值 = -7.116万元", ha="center", color=VERMILLION)
style_axis(axes[0])
axes[1].hist(daily_voi / 1000, bins=45, color=SKY, edgecolor="white", linewidth=0.25)
axes[1].axvline(0, color=GRAY, linestyle="--", linewidth=1.0)
axes[1].axvline(daily_voi.mean() / 1000, color=VERMILLION, linewidth=1.2, label=f"日均 {daily_voi.mean()/1000:.3f} 千元")
axes[1].set_xlabel("逐日毛信息价值 / 千元")
axes[1].set_ylabel("天数")
axes[1].set_title("逐日信息价值分布")
axes[1].legend(frameon=False)
style_axis(axes[1])
panel_labels(axes)
save(fig, 13, "不同预报更新频率下的信息价值比较")


# 图14：固定价与波动价
fixed_formal = np.tile(input_data.fixed_price, formal_mask.sum())
variable_formal = var_price[formal_mask].ravel()
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
bins = np.linspace(min(variable_formal.min(), fixed_formal.min()), max(variable_formal.max(), fixed_formal.max()), 70)
axes[0].hist(fixed_formal, bins=bins, density=True, histtype="step", linewidth=1.4, color=BLUE, label="固定分时电价")
axes[0].hist(variable_formal, bins=bins, density=True, histtype="stepfilled", alpha=0.28, color=ORANGE, label="波动电价")
axes[0].set_xlabel("电价 / (元·kWh$^{-1}$)")
axes[0].set_ylabel("概率密度")
axes[0].set_title("正式评价期电价分布")
axes[0].legend(frameon=False)
style_axis(axes[0])
fixed_profile = input_data.fixed_price
variable_profile = var_price[formal_mask].mean(axis=0)
axes[1].plot(x, fixed_profile, color=BLUE, linewidth=1.25, label="固定分时电价")
axes[1].plot(x, variable_profile, color=ORANGE, linestyle="--", linewidth=1.25, label="波动电价日均")
axes[1].set_ylabel("电价 / (元·kWh$^{-1}$)")
axes[1].set_title("平均日内价格路径")
axes[1].legend(frameon=False)
hour_ticks(axes[1])
style_axis(axes[1])
panel_labels(axes)
save(fig, 14, "固定电价与波动电价分布对比")


# 图15：四种策略月度成本
fig, ax = plt.subplots(figsize=(7.2, 3.45), constrained_layout=True)
strategy_data = [
    (q2, "问题二：固定价日前", BLUE, "o", "-"),
    (q3, "问题三：固定价滚动", GREEN, "s", "--"),
    (q42, "问题四：波动价日前", ORANGE, "^", "-."),
    (q43, "问题四：波动价滚动", PURPLE, "D", ":"),
]
for df, label, color, marker, ls in strategy_data:
    monthly_cost = df.groupby(df["日期"].dt.month)["总成本_元"].sum() / 1e6
    ax.plot(monthly_cost.index, monthly_cost.values, color=color, marker=marker, linestyle=ls,
            linewidth=1.3, markersize=3.7, label=label)
ax.set_xticks(range(2, 13))
ax.set_xlabel("月份")
ax.set_ylabel("月度总成本 / 百万元")
ax.legend(frameon=False, ncol=2, loc="upper center")
style_axis(ax)
save(fig, 15, "四种调度策略的月度成本比较")


# 图16：波动电价下滚动代表日
typical = q43.loc[q43["日期"] == typical_date].sort_values("时段序号")
tx = np.arange(len(typical)) / 6 + 1 / 6
fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.45), sharex=True, constrained_layout=True)
axes[0].plot(tx, typical["电价_元每kWh"], color=VERMILLION, linewidth=1.35)
axes[0].set_ylabel("元/kWh")
axes[0].set_title(f"{typical_date:%Y-%m-%d} 波动电价路径")
style_axis(axes[0])
axes[1].plot(tx, typical["计划购电量_kWh"], color=GRAY, linestyle="--", linewidth=1.05, label="日前计划")
axes[1].plot(tx, typical["调整后购电量_kWh"], color=BLUE, linewidth=1.25, label="滚动调整后")
axes[1].plot(tx, typical["紧急购电量_kWh"], color=ORANGE, linewidth=1.0, label="紧急购电")
axes[1].set_ylabel("电量 / kWh")
axes[1].set_title("购电动作")
axes[1].legend(frameon=False, ncol=3, loc="upper right")
style_axis(axes[1])
axes[2].bar(tx, typical["充电量_kWh"], width=0.13, color=BLUE, alpha=0.8, label="充电")
axes[2].bar(tx, -typical["放电量_kWh"], width=0.13, color=ORANGE, alpha=0.8, label="放电（负向显示）")
axes[2].axhline(0, color="black", linewidth=0.6)
axes[2].set_ylabel("交流侧电量 / kWh")
axes[2].set_title("储能动作")
axes[2].legend(frameon=False, ncol=2, loc="upper right")
style_axis(axes[2])
hour_ticks(axes[2])
panel_labels(axes)
save(fig, 16, "波动电价下滚动策略的典型日调度")


# 图17：正式评价期总体成本
total_costs = pd.Series(
    {
        "问题二\n固定价日前": q2["总成本_元"].sum(),
        "问题三\n固定价滚动": q3["总成本_元"].sum(),
        "问题四\n波动价日前": q42["总成本_元"].sum(),
        "问题四\n波动价滚动": q43["总成本_元"].sum(),
    }
).sort_values()
fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
colors = [GREEN, BLUE, PURPLE, ORANGE]
bars = ax.barh(np.arange(len(total_costs)), total_costs.values / 1e6, color=colors, height=0.58)
ax.set_yticks(np.arange(len(total_costs)), total_costs.index)
ax.set_xlabel("正式评价期总成本 / 百万元")
ax.set_xlim(0, total_costs.max() / 1e6 * 1.18)
for b, v in zip(bars, total_costs.values / 1e6):
    ax.text(v + 0.08, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
style_axis(ax, grid="x")
save(fig, 17, "不同问题与调度策略的全年总成本比较")


print(f"font: {ZH_FONT}")
print("figure generation completed")
