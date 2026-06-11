"""
加密货币短线策略
- RSI + 布林带 均值回归
- 只做 BTC（流动性最好），每天扫描信号
- 严格止盈止损

用法:
    python crypto_short_term.py
    python crypto_short_term.py --quick  # 2024年后数据
"""

import argparse
import time
import numpy as np
import pandas as pd

from data_loader import get_crypto_pool, get_all_us_stocks_data, get_crypto_benchmark

# ============================================================
# 短线策略参数
# ============================================================
RSI_PERIOD = 6              # RSI 周期（短）
RSI_OVERSOLD = 28           # 超卖阈值（买入信号）
RSI_OVERBOUGHT = 68         # 超买阈值（卖出信号）
BB_PERIOD = 20              # 布林带周期
BB_STD = 2.0                # 布林带标准差倍数
STOP_LOSS = -0.08           # 止损 -8%（放宽）
TAKE_PROFIT = 0.20          # 止盈 +20%（放宽）
MAX_HOLD_DAYS = 60          # 最长持仓 60 天（基本不限）
INITIAL_CAPITAL = 100_000   # 初始 10 万 USDT
TRADING_DAYS_PER_YEAR = 365
RISK_FREE_RATE = 0.03
FEE = 0.001                 # 0.1% 手续费


def compute_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI"""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_bollinger(df: pd.DataFrame, period: int = BB_PERIOD, n_std: float = BB_STD):
    """返回 (middle, upper, lower)"""
    middle = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    return middle, middle + n_std * std, middle - n_std * std


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range（波动率指标）"""
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def run_short_term(quick: bool = False):
    t0 = time.time()
    start_date = "20240101" if quick else "20230101"

    # ---- 获取数据 ----
    print("[Short] 获取 BTC 数据...")
    btc_df = None
    try:
        from data_loader import DATA_CACHE_DIR, START_DATE, END_DATE
        import os
        # 直接用 yfinance 拿 BTC
        import yfinance as yf
        btc = yf.Ticker("BTC-USD")
        df = btc.history(start="2023-01-01", end="2026-06-08")
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        if quick:
            df = df[df["date"] >= "2024-01-01"]
        btc_df = df
        print(f"[Short] BTC 数据: {len(btc_df)} 天")
    except Exception as e:
        print(f"[Short] 获取 BTC 失败: {e}")
        return

    # ---- 计算指标 ----
    btc_df["rsi"] = compute_rsi(btc_df)
    bb_mid, bb_up, bb_lo = compute_bollinger(btc_df)
    btc_df["bb_upper"] = bb_up
    btc_df["bb_lower"] = bb_lo
    btc_df["atr"] = compute_atr(btc_df)
    btc_df["vol_ratio"] = btc_df["volume"] / btc_df["volume"].rolling(20).mean()

    # ---- 回测 ----
    cash = float(INITIAL_CAPITAL)
    position = 0.0            # BTC 持仓量
    entry_price = 0.0         # 入场价
    entry_date = None         # 入场日
    trades = []               # 交易记录
    nav = []                  # 每日净值
    in_position = False

    for i, row in btc_df.iterrows():
        date = row["date"]
        price = row["close"]
        rsi_val = row["rsi"]
        bb_lower = row["bb_lower"]
        bb_upper = row["bb_upper"]
        bb_middle = bb_mid.iloc[i]
        vol_ratio = row["vol_ratio"]

        # ---- 不在仓位中：看买入信号 ----
        if not in_position and not pd.isna(rsi_val) and not pd.isna(bb_lower):
            # 买入条件：RSI 超卖 + 价格跌破布林下轨 + 放量
            signal_rsi = rsi_val < RSI_OVERSOLD
            signal_bb = price < bb_lower
            signal_vol = vol_ratio > 1.2  # 放量 20%

            buy_signal = signal_rsi and signal_bb

            if buy_signal:
                position = cash / price    # 全仓买入
                entry_price = price
                entry_date = date
                cash = 0
                in_position = True
                trades.append({"type": "BUY", "date": date, "price": price,
                               "rsi": rsi_val, "reason": f"RSI={rsi_val:.0f} BB_low"})

        # ---- 在仓位中：看卖出信号 ----
        elif in_position:
            pnl_pct = (price / entry_price) - 1
            days_held = (date - entry_date).days if entry_date else 0

            # 卖出条件（任一触发）
            sell_reason = None

            if pnl_pct <= STOP_LOSS:
                sell_reason = f"止损 {pnl_pct:.1%}"
            elif pnl_pct >= TAKE_PROFIT:
                sell_reason = f"止盈 {pnl_pct:.1%}"
            elif days_held >= MAX_HOLD_DAYS:
                sell_reason = f"超时 {days_held}天"
            elif not pd.isna(rsi_val) and rsi_val > RSI_OVERBOUGHT:
                sell_reason = f"RSI超买 RSI={rsi_val:.0f}"
            elif not pd.isna(bb_middle) and price > bb_middle:
                sell_reason = "回归BB中轨"

            if sell_reason:
                cash = position * price * (1 - FEE)
                trade_pnl = (price / entry_price - 1) * 100
                trades.append({"type": "SELL", "date": date, "price": price,
                               "pnl%": trade_pnl, "reason": sell_reason,
                               "days": days_held})
                position = 0
                entry_price = 0
                entry_date = None
                in_position = False

        # ---- 记录每日净值 ----
        total = cash + (position * price if in_position else 0)
        nav.append({"date": date, "total": total,
                    "price": price,
                    "in_position": in_position})

    # 如果最后还在持仓，按最后价格平仓
    if in_position:
        last_price = btc_df["close"].iloc[-1]
        cash = position * last_price * (1 - FEE)
        trades.append({"type": "CLOSE", "date": btc_df["date"].iloc[-1],
                       "price": last_price, "pnl%": (last_price/entry_price-1)*100,
                       "reason": "回测结束"})
        nav[-1]["total"] = cash

    # ---- 绩效统计 ----
    nav_df = pd.DataFrame(nav).set_index("date")
    btc_ret = btc_df["close"].iloc[-1] / btc_df["close"].iloc[0]
    final_nav = nav_df["total"].iloc[-1] / INITIAL_CAPITAL

    total_days = len(nav_df)
    ann_ret = final_nav ** (TRADING_DAYS_PER_YEAR / total_days) - 1
    btc_ann = btc_ret ** (TRADING_DAYS_PER_YEAR / total_days) - 1

    returns = nav_df["total"].pct_change().dropna()
    ann_vol = float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

    peak = nav_df["total"].expanding().max()
    max_dd = float(((nav_df["total"] - peak) / peak).min())

    # 交易统计
    sell_trades = [t for t in trades if t["type"] in ("SELL", "CLOSE")]
    win_trades = [t for t in sell_trades if t["pnl%"] > 0]
    win_rate = len(win_trades) / len(sell_trades) if sell_trades else 0
    avg_win = np.mean([t["pnl%"] for t in win_trades]) if win_trades else 0
    avg_loss = np.mean([t["pnl%"] for t in sell_trades if t["pnl%"] <= 0]) if sell_trades else 0
    profit_factor = (sum(t["pnl%"] for t in win_trades) /
                     abs(sum(t["pnl%"] for t in sell_trades if t["pnl%"] <= 0))) if sell_trades else 0
    avg_days = np.mean([t.get("days", 0) for t in sell_trades]) if sell_trades else 0

    # ---- 输出 ----
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti SC", "PingFang SC"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [2.5, 1, 1]})

    # 图1: 净值曲线 + 买卖点
    ax1 = axes[0]
    ax1.plot(nav_df.index, nav_df["total"] / INITIAL_CAPITAL,
             label="短线策略", color="#1f77b4", linewidth=1.5)
    btc_nav = btc_df.set_index("date")["close"] / btc_df["close"].iloc[0]
    ax1.plot(btc_nav.index, btc_nav.values,
             label="BTC 持有", color="#ff7f0e", linewidth=1.0, alpha=0.7)

    # 标注买卖点
    for t in trades:
        if t["type"] == "BUY":
            ax1.scatter(t["date"], nav_df.loc[t["date"], "total"] / INITIAL_CAPITAL,
                       color="green", marker="^", s=60, zorder=5, alpha=0.8)
        elif t["type"] in ("SELL", "CLOSE"):
            ax1.scatter(t["date"], nav_df.loc[t["date"], "total"] / INITIAL_CAPITAL,
                       color="red", marker="v", s=60, zorder=5, alpha=0.8)

    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("BTC 短线 RSI+布林带 策略", fontsize=14, fontweight="bold")
    ax1.set_ylabel("净值", fontsize=11)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 图2: RSI 指标
    ax2 = axes[1]
    rsi_series = btc_df.set_index("date")["rsi"]
    ax2.plot(rsi_series.index, rsi_series.values, color="purple", linewidth=0.8)
    ax2.axhline(y=RSI_OVERSOLD, color="green", linestyle="--", alpha=0.5, label=f"超卖({RSI_OVERSOLD})")
    ax2.axhline(y=RSI_OVERBOUGHT, color="red", linestyle="--", alpha=0.5, label=f"超买({RSI_OVERBOUGHT})")
    ax2.axhline(y=50, color="gray", linestyle=":", alpha=0.3)
    ax2.fill_between(rsi_series.index, RSI_OVERSOLD, RSI_OVERBOUGHT, alpha=0.03, color="gray")
    ax2.set_ylabel("RSI", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 图3: 回撤
    ax3 = axes[2]
    dd = (nav_df["total"] - nav_df["total"].expanding().max()) / nav_df["total"].expanding().max()
    ax3.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
    ax3.set_ylabel("回撤", fontsize=10)
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("assets/crypto_short_term.png", dpi=150, bbox_inches="tight")
    print("[Short] 图表已保存: assets/crypto_short_term.png")

    print(f"\n{'='*65}")
    print(f"  BTC 短线 RSI+布林带 · 绩效报告")
    print(f"{'='*65}")
    print(f"  总收益率:      {(final_nav-1)*100:.1f}%")
    print(f"  年化收益:      {ann_ret*100:.1f}%")
    print(f"  年化波动:      {ann_vol*100:.1f}%")
    print(f"  夏普比率:       {sharpe:.2f}")
    print(f"  最大回撤:       {max_dd*100:.1f}%")
    print(f"  ──────────────────────────────")
    print(f"  交易次数:       {len(sell_trades)}")
    print(f"  胜率:           {win_rate*100:.0f}%")
    print(f"  平均盈利:       {avg_win:.1f}%")
    print(f"  平均亏损:       {avg_loss:.1f}%")
    print(f"  盈亏比:         {profit_factor:.2f}")
    print(f"  平均持仓天数:    {avg_days:.1f}天")
    print(f"  ──────────────────────────────")
    print(f"  BTC 年化收益:   {btc_ann*100:.1f}%")
    print(f"  策略 vs BTC:    {(ann_ret-btc_ann)*100:.1f}%")
    print(f"{'='*65}")

    # 打印最近几笔交易
    print(f"\n  最近 10 笔交易:")
    print(f"  {'日期':<12s} {'类型':<6s} {'价格':>10s} {'盈亏':>8s} {'原因'}")
    print(f"  {'-'*55}")
    for t in trades[-10:]:
        pnl_str = f"{t.get('pnl%', 0):+.1f}%" if 'pnl%' in t else ""
        print(f"  {str(t['date'])[:10]:<12s} {t['type']:<6s} "
              f"${t['price']:>9,.0f} {pnl_str:>8s} {t['reason']}")

    elapsed = time.time() - t0
    print(f"\n  Total: {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC 短线策略")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_short_term(quick=args.quick)
