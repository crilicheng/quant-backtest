"""
绩效分析与可视化
计算常用的量化策略评估指标，画净值曲线和回撤图。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Optional

from config import RISK_FREE_RATE

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti SC", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 绩效指标计算
# ============================================================

def max_drawdown(nav: pd.Series) -> float:
    """
    最大回撤（百分比）
    回撤 = (当前净值 - 历史最高净值) / 历史最高净值
    """
    peak = nav.expanding().max()
    drawdown = (nav - peak) / peak
    return float(drawdown.min())


def annualized_return(nav: pd.Series) -> float:
    """
    年化收益率
    (最终净值/初始净值)^(252/交易天数) - 1
    """
    days = len(nav)
    if days < 2:
        return 0.0
    total_return = nav.iloc[-1] / nav.iloc[0]
    return total_return ** (252 / days) - 1


def annualized_volatility(daily_returns: pd.Series) -> float:
    """年化波动率 = 日收益率标准差 * sqrt(252)"""
    return float(daily_returns.std() * np.sqrt(252))


def sharpe_ratio(daily_returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """
    夏普比率 = (年化收益 - 无风险利率) / 年化波动率
    衡量每单位风险获得的超额回报
    """
    ann_ret = annualized_return(daily_returns + 1)  # 从日收益反推有问题
    # 直接用净值算
    return 0.0  # placeholder, 下面 calculate_stats 里重算


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    信息比率 = mean(超额收益) / std(超额收益) * sqrt(252)
    衡量策略相对于基准的超额收益稳定性
    """
    excess = strategy_returns - benchmark_returns
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(252))


def win_rate(daily_returns: pd.Series) -> float:
    """日胜率：日收益 > 0 的比例"""
    return float((daily_returns > 0).mean())


def calculate_stats(
    nav: pd.DataFrame,
) -> dict:
    """
    综合计算所有绩效指标。

    参数:
        nav: DataFrame with columns: portfolio_nav, benchmark_nav (optional), portfolio_value

    返回:
        dict of stats
    """
    port_nav = nav["portfolio_nav"]
    port_ret = port_nav.pct_change().dropna()

    # 基准数据
    has_benchmark = "benchmark_nav" in nav.columns and nav["benchmark_nav"].notna().any()
    bench_nav = nav["benchmark_nav"].dropna() if has_benchmark else None
    bench_ret = bench_nav.pct_change().dropna() if has_benchmark else None

    ann_ret = annualized_return(port_nav)
    ann_vol = annualized_volatility(port_ret)
    max_dd = max_drawdown(port_nav)
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0.0
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    stats = {
        "累计收益率": f"{port_nav.iloc[-1] - 1:.2%}",
        "年化收益率": f"{ann_ret:.2%}",
        "年化波动率": f"{ann_vol:.2%}",
        "夏普比率": f"{sharpe:.2f}",
        "最大回撤": f"{max_dd:.2%}",
        "卡玛比率": f"{calmar:.2f}",
        "日胜率": f"{win_rate(port_ret):.2%}",
        "交易天数": len(port_ret),
    }

    if bench_ret is not None:
        bench_ann_ret = annualized_return(bench_nav)
        bench_ann_vol = annualized_volatility(bench_ret)
        bench_max_dd = max_drawdown(bench_nav)
        bench_sharpe = (bench_ann_ret - RISK_FREE_RATE) / bench_ann_vol if bench_ann_vol > 0 else 0.0
        ir = information_ratio(port_ret, bench_ret)

        stats.update({
            "基准年化收益": f"{bench_ann_ret:.2%}",
            "基准年化波动": f"{bench_ann_vol:.2%}",
            "基准最大回撤": f"{bench_max_dd:.2%}",
            "基准夏普比率": f"{bench_sharpe:.2f}",
            "超额年化收益": f"{ann_ret - bench_ann_ret:.2%}",
            "信息比率": f"{ir:.2f}",
        })

    return stats


# ============================================================
# 可视化
# ============================================================

def plot_results(
    nav: pd.DataFrame,
    stats: dict,
    save_path: Optional[str] = None,
):
    """
    画三张图：
    1. 策略 vs 基准 净值曲线
    2. 回撤曲线
    3. 月度收益热力图
    """
    port_nav = nav["portfolio_nav"]
    has_benchmark = "benchmark_nav" in nav.columns and nav["benchmark_nav"].notna().any()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    # ---- 图1: 净值曲线 ----
    ax1 = axes[0]
    ax1.plot(port_nav.index, port_nav.values, label="多因子策略", color="#1f77b4", linewidth=1.5)
    if has_benchmark:
        bench = nav["benchmark_nav"].dropna()
        ax1.plot(bench.index, bench.values, label="沪深300", color="#ff7f0e", linewidth=1.0, alpha=0.8)

    # 标注调仓日附近的 vertical line（用净值变化较大的日子简化标注）
    ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax1.set_title("多因子选股策略 回测结果", fontsize=14, fontweight="bold")
    ax1.set_ylabel("净值", fontsize=11)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # ---- 图2: 回撤曲线 ----
    ax2 = axes[1]
    dd = (port_nav - port_nav.expanding().max()) / port_nav.expanding().max()
    ax2.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3, label="回撤")
    ax2.plot(dd.index, dd.values, color="red", linewidth=0.8)
    ax2.set_ylabel("回撤", fontsize=11)
    ax2.set_xlabel("日期", fontsize=11)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2.grid(True, alpha=0.3)

    # 在图上标注最大回撤
    max_dd_idx = dd.idxmin()
    max_dd_val = dd.min()
    ax2.annotate(f"最大回撤: {max_dd_val:.1%}",
                 xy=(max_dd_idx, max_dd_val),
                 xytext=(max_dd_idx, max_dd_val * 0.5),
                 arrowprops=dict(arrowstyle="->", color="darkred"),
                 fontsize=10, color="darkred")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] 图表已保存: {save_path}")
    else:
        plt.show()

    plt.close()

    # ---- 图3: 绩效指标表 ----
    print("\n" + "=" * 70)
    print("  绩效指标")
    print("=" * 70)
    for key, value in stats.items():
        print(f"  {key:15s}: {value}")
    print("=" * 70)
