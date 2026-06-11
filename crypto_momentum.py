"""
加密货币动量轮动策略
- 每周调仓，选近期动量最强的币
- 牛市追涨，熊市空仓（靠 BTC 均线判定牛熊）
- 365 天年化（币圈无休市）

用法:
    python crypto_momentum.py
    python crypto_momentum.py --quick  # 快速模式（20个币）
"""

import argparse
import time
import numpy as np
import pandas as pd

from data_loader import (
    get_crypto_pool, get_all_us_stocks_data, get_crypto_benchmark,
)
from config import DATA_CACHE_DIR

# ============================================================
# 币圈专用配置
# ============================================================
MOMENTUM_WINDOW = 30        # 动量看 30 天
VOLATILITY_WINDOW = 30      # 波动率看 30 天
REBALANCE_DAYS = 1          # 每天检查信号
HOLDING_COUNT = 7           # 山寨币持仓数（另加 BTC）
INITIAL_CAPITAL = 100_000   # 初始 10 万 USDT
TRADING_DAYS_PER_YEAR = 365 # 币圈全年无休
RISK_FREE_RATE = 0.03       # 3% 年化（USDC 理财利率）
TRANSACTION_COST = 0.002    # 0.2%（交易所手续费 + 滑点）


def generate_weekly_dates(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """生成每周调仓日（每 7 个自然日）"""
    sorted_dates = sorted(dates)
    rebalance_dates = []
    last_date = None
    for d in sorted_dates:
        if last_date is None or (d - last_date).days >= REBALANCE_DAYS:
            rebalance_dates.append(d)
            last_date = d
    return rebalance_dates


def is_bull_market(btc_data: pd.DataFrame, as_of_date: pd.Timestamp) -> bool:
    """
    牛熊判定：BTC 价格高于 50 日均线 → 牛市，否则熊市。
    50 日线比 200 日线更灵敏，适合币圈快节奏。
    """
    df = btc_data[btc_data["date"] <= as_of_date]
    if len(df) < 50:
        return False
    ma50 = df["close"].tail(50).mean()
    current = df["close"].iloc[-1]
    return current > ma50


def calculate_momentum(df: pd.DataFrame, window: int = MOMENTUM_WINDOW) -> float:
    """N 日动量（百分比）"""
    if len(df) < window:
        return np.nan
    return (df["close"].iloc[-1] / df["close"].iloc[-window] - 1) * 100


def calculate_volatility(df: pd.DataFrame, window: int = VOLATILITY_WINDOW) -> float:
    """N 日年化波动率（365 天）"""
    if len(df) < window:
        return np.nan
    returns = df["close"].pct_change().dropna().tail(window)
    return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100)


def calculate_sharpe_20d(df: pd.DataFrame) -> float:
    """20 日夏普比率（简化版，用于质量筛选）"""
    if len(df) < 20:
        return np.nan
    returns = df["close"].pct_change().dropna().tail(20)
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def run_crypto_backtest(quick: bool = False):
    t0 = time.time()
    pool_size = 20 if quick else 50

    # ---- 1. 数据 ----
    print("[Crypto] 获取加密货币数据...")
    tickers = get_crypto_pool(top_n=pool_size)
    crypto_data = get_all_us_stocks_data(tickers)
    btc_benchmark = get_crypto_benchmark()

    if len(crypto_data) < 10:
        print("[Crypto] 数据不足，退出")
        return

    # 提取 BTC 用于牛熊判定
    btc_data = crypto_data.get("BTC-USD")
    if btc_data is None:
        print("[Crypto] 无法获取 BTC 数据，退出")
        return

    # 交易日历（取所有币的日期交集，选有数据的日期）
    all_dates = pd.DatetimeIndex(sorted(btc_data["date"].unique()))
    rebalance_dates = generate_weekly_dates(all_dates)

    # 过滤掉前 50 天（MA50 需要）
    if len(all_dates) > 50:
        min_date = all_dates[50]
        rebalance_dates = [d for d in rebalance_dates if d >= min_date]

    print(f"[Crypto] {len(rebalance_dates)} 个调仓日, {len(crypto_data)} 个币种")

    # ---- 2. 回测循环 ----
    cash = float(INITIAL_CAPITAL)
    position = {}     # {symbol: quantity}
    nav_history = []   # [{date, total_value, is_invested}]

    for today in all_dates:
        # --- 调仓日操作 ---
        if today in rebalance_dates:
            bull = is_bull_market(btc_data, today)

            if bull:
                # 牛市：全仓 BTC
                target = {"BTC-USD": 1.0}  # 100% BTC

                # 清仓不在目标中的
                for sym in list(position.keys()):
                    if sym not in target:
                        df_sym = crypto_data.get(sym)
                        if df_sym is not None:
                            today_prices = df_sym[df_sym["date"] == today]
                            if len(today_prices) > 0:
                                cash += position[sym] * today_prices["close"].iloc[0] * (1 - TRANSACTION_COST)
                        del position[sym]

                # 买入 BTC（如果用现金）
                if "BTC-USD" not in position and cash > 0:
                    df_btc = crypto_data.get("BTC-USD")
                    if df_btc is not None:
                        today_prices = df_btc[df_btc["date"] == today]
                        if len(today_prices) > 0:
                            price = today_prices["close"].iloc[0]
                            qty = cash * (1 - TRANSACTION_COST) / price
                            position["BTC-USD"] = qty
                            cash = 0
            else:
                # 熊市：清仓，全持 USDT
                for sym in list(position.keys()):
                    df_sym = crypto_data.get(sym)
                    if df_sym is not None:
                        today_prices = df_sym[df_sym["date"] == today]
                        if len(today_prices) > 0:
                            cash += position[sym] * today_prices["close"].iloc[0] * (1 - TRANSACTION_COST)
                    del position[sym]

        # --- 每日估值 ---
        portfolio_value = 0.0
        for sym, qty in position.items():
            df_sym = crypto_data.get(sym)
            if df_sym is not None:
                today_prices = df_sym[df_sym["date"] == today]
                if len(today_prices) > 0:
                    portfolio_value += qty * today_prices["close"].iloc[0]

        total = cash + portfolio_value

        # BTC 基准
        btc_price = np.nan
        btc_today = btc_data[btc_data["date"] == today]
        if len(btc_today) > 0:
            btc_price = btc_today["close"].iloc[0]

        nav_history.append({
            "date": today,
            "portfolio": total,
            "btc_price": btc_price,
            "is_invested": len(position) > 0,
        })

    # ---- 3. 绩效计算 ----
    nav_df = pd.DataFrame(nav_history).set_index("date")
    btc_init = nav_df["btc_price"].dropna().iloc[0]
    nav_df["btc_nav"] = nav_df["btc_price"] / btc_init
    nav_df["portfolio_nav"] = nav_df["portfolio"] / INITIAL_CAPITAL
    nav_df["portfolio_return"] = nav_df["portfolio_nav"].pct_change()
    nav_df["btc_return"] = nav_df["btc_nav"].pct_change()

    final_nav = nav_df["portfolio_nav"].iloc[-1]
    btc_final = nav_df["btc_nav"].dropna().iloc[-1]

    # 年化
    total_days = len(nav_df)
    ann_return = final_nav ** (TRADING_DAYS_PER_YEAR / total_days) - 1
    btc_ann_return = btc_final ** (TRADING_DAYS_PER_YEAR / total_days) - 1

    # 波动率
    strat_vol = float(nav_df["portfolio_return"].std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    btc_vol = float(nav_df["btc_return"].dropna().std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    # 夏普
    sharpe = (ann_return - RISK_FREE_RATE) / strat_vol if strat_vol > 0 else 0
    btc_sharpe = (btc_ann_return - RISK_FREE_RATE) / btc_vol if btc_vol > 0 else 0

    # 最大回撤
    peak = nav_df["portfolio_nav"].expanding().max()
    dd = (nav_df["portfolio_nav"] - peak) / peak
    max_dd = float(dd.min())

    btc_peak = nav_df["btc_nav"].dropna().expanding().max()
    btc_dd = (nav_df["btc_nav"].dropna() - btc_peak) / btc_peak
    btc_max_dd = float(btc_dd.min())

    # 胜率
    win_rate = float((nav_df["portfolio_return"] > 0).mean())

    # 牛熊统计
    bull_days = sum(1 for v in nav_history if v["is_invested"])
    bear_days = len(nav_history) - bull_days

    # ---- 4. 输出 ----
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti SC", "PingFang SC"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax1 = axes[0]
    ax1.plot(nav_df.index, nav_df["portfolio_nav"], label="动量轮动策略", color="#1f77b4", linewidth=1.5)
    ax1.plot(nav_df.index, nav_df["btc_nav"], label="BTC 持有", color="#ff7f0e", linewidth=1.0, alpha=0.8)
    # 标注牛熊区域
    in_bull = False
    bull_start = None
    for v in nav_history:
        if v["is_invested"] and not in_bull:
            bull_start = v["date"]
            in_bull = True
        elif not v["is_invested"] and in_bull:
            ax1.axvspan(bull_start, v["date"], alpha=0.08, color="green")
            in_bull = False
            bull_start = None
    if in_bull and bull_start is not None:
        ax1.axvspan(bull_start, nav_history[-1]["date"], alpha=0.08, color="green")

    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("加密货币动量轮动策略", fontsize=14, fontweight="bold")
    ax1.set_ylabel("净值 (USDT)", fontsize=11)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3, label="策略回撤")
    max_dd_idx = dd.idxmin()
    ax2.annotate(f"最大回撤: {max_dd:.1%}", xy=(max_dd_idx, max_dd),
                 xytext=(max_dd_idx, max_dd * 0.5),
                 arrowprops=dict(arrowstyle="->", color="darkred"),
                 fontsize=10, color="darkred")
    ax2.set_ylabel("回撤", fontsize=11)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("assets/crypto_result.png", dpi=150, bbox_inches="tight")
    print("[Crypto] 图表已保存: assets/crypto_result.png")

    print(f"\n{'='*70}")
    print(f"  加密货币动量轮动策略 · 绩效报告")
    print(f"{'='*70}")
    print(f"  初始资金:        ${INITIAL_CAPITAL:,.0f}")
    print(f"  最终资金:        ${nav_df['portfolio'].iloc[-1]:,.0f}")
    print(f"  累计收益率:       {(final_nav - 1)*100:.1f}%")
    print(f"  年化收益率:       {ann_return*100:.1f}%")
    print(f"  年化波动率:       {strat_vol*100:.1f}%")
    print(f"  夏普比率:         {sharpe:.2f}")
    print(f"  最大回撤:         {max_dd*100:.1f}%")
    print(f"  日胜率:           {win_rate*100:.1f}%")
    print(f"  换手次数:         {len(rebalance_dates)}")
    print(f"  ─────────────────────────────────────")
    print(f"  BTC 年化收益:     {btc_ann_return*100:.1f}%")
    print(f"  BTC 夏普:         {btc_sharpe:.2f}")
    print(f"  BTC 最大回撤:     {btc_max_dd*100:.1f}%")
    print(f"  ─────────────────────────────────────")
    print(f"  超额收益:         {(ann_return - btc_ann_return)*100:.1f}%")
    print(f"  牛市占比:         {bull_days/len(nav_history)*100:.0f}%")
    print(f"{'='*70}")

    elapsed = time.time() - t0
    print(f"\n  Total: {elapsed:.1f}s")

    # 保存
    nav_df.to_csv("assets/crypto_nav.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="加密货币动量轮动策略")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式（20个币）")
    args = parser.parse_args()
    run_crypto_backtest(quick=args.quick)
