"""
BTC 杠杆超短线策略
- 5x 杠杆做 BTC 永续合约
- RSI 极端超卖抄底，快速止盈
- 紧止损，靠胜率吃饭

用法:
    python crypto_scalping.py
    python crypto_scalping.py --quick
"""

import argparse, time
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti SC", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 策略参数
# ============================================================
LEVERAGE = 5                # 5 倍杠杆
CAPITAL = 10_000            # 初始本金 $10,000
RISK_PER_TRADE = 0.02       # 每笔风险 = 本金的 2%
TP_PRICE_PCT = 0.012        # 止盈：价格涨 1.2%（×5 = 仓位赚 6%）
SL_PRICE_PCT = 0.005        # 止损：价格跌 0.5%（×5 = 仓位亏 2.5%）
RSI_PERIOD = 4              # 超短 RSI（4 根 K 线）
RSI_ENTRY = 18              # 极端超卖才进
MIN_VOL_SPIKE = 1.5         # 成交量要放大 1.5 倍（恐慌抛售确认）
TRAILING_STOP = 0.003       # 移动止盈：从最高点回撤 0.3% 就锁定利润
MAX_HOLD_BARS = 3           # 最长持仓 3 天
FEE = 0.0004                # 永续合约手续费 0.04%
SLIPPAGE = 0.0002           # 滑点 0.02%
TRADING_DAYS = 365


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def run_scalping(quick: bool = False):
    t0 = time.time()

    # ---- 数据 ----
    print("[Scalp] 获取 BTC 数据...")
    df = yf.Ticker("BTC-USD").history(start="2023-01-01", end="2026-06-08")
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    if quick:
        df = df[df["date"] >= "2025-01-01"]

    # ---- 指标 ----
    df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)
    df["atr14"] = (df["high"] - df["low"]).rolling(14).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]
    df["prev_low"] = df["low"].shift(1)

    # ---- 回测 ----
    equity = float(CAPITAL)                     # 账户权益
    position_value = 0.0                        # 仓位名义价值
    entry_price = 0.0
    highest_since_entry = 0.0
    bars_held = 0
    in_position = False
    trades = []
    nav = []

    for i, row in df.iterrows():
        if i < 30:
            nav.append({"date": row["date"], "equity": equity})
            continue

        date = row["date"]
        price = row["close"]
        price_high = row["high"]
        price_low = row["low"]
        rsi_val = row["rsi"]
        vol_ratio = row["vol_ratio"]

        # ==== 持仓中：检查平仓 ====
        if in_position:
            bars_held += 1
            # 用日内最高价更新（模拟盯盘）
            highest_since_entry = max(highest_since_entry, price_high)

            # 未实现盈亏（价格变化）
            unreal_pnl_pct = (price - entry_price) / entry_price
            leveraged_pnl = unreal_pnl_pct * LEVERAGE

            exit_price = None
            exit_reason = ""

            # ① 硬止损
            if unreal_pnl_pct <= -SL_PRICE_PCT:
                exit_price = entry_price * (1 - SL_PRICE_PCT)
                exit_reason = "止损"
            # ② 硬止盈
            elif unreal_pnl_pct >= TP_PRICE_PCT:
                exit_price = entry_price * (1 + TP_PRICE_PCT)
                exit_reason = "止盈"
            # ③ 移动止盈：从最高点回落
            elif highest_since_entry > entry_price:
                drawdown_from_peak = (highest_since_entry - price) / entry_price
                if drawdown_from_peak > TRAILING_STOP and unreal_pnl_pct > 0:
                    exit_price = price
                    exit_reason = "移动止盈"
            # ④ 超时
            elif bars_held >= MAX_HOLD_BARS:
                exit_price = price
                exit_reason = f"超时{bars_held}天"
            # ⑤ RSI 过热
            elif not pd.isna(rsi_val) and rsi_val > 75:
                exit_price = price
                exit_reason = f"RSI过热"

            if exit_price is not None:
                exit_slip = exit_price * (1 - SLIPPAGE)
                pnl_pct = (exit_slip / entry_price - 1) * LEVERAGE
                pnl_dollar = position_value * pnl_pct - position_value * FEE * 2
                equity += pnl_dollar
                if equity <= 0:
                    equity = 0.01  # 爆仓保护

                trades.append({
                    "entry_date": entry_date, "exit_date": date,
                    "entry": entry_price, "exit": exit_slip,
                    "pnl_pct": pnl_pct * 100, "pnl_$": pnl_dollar,
                    "reason": exit_reason, "bars": bars_held,
                })
                position_value = 0
                in_position = False

        # ==== 空仓中：看入场信号 ====
        if not in_position and not pd.isna(rsi_val):
            # 入场：RSI 极端超卖 + 放量（恐慌抛售 = 抄底机会）
            panic_sell = rsi_val < RSI_ENTRY and vol_ratio > MIN_VOL_SPIKE
            # 确认：当前价高于当日最低价（有反弹迹象）
            reversal = price > price_low * 1.002

            if panic_sell and reversal:
                # 仓位计算：风险 = 本金 × 2%，止损价差 = entry × SL_PRICE_PCT
                risk_amount = equity * RISK_PER_TRADE
                price_risk = price * SL_PRICE_PCT * LEVERAGE
                if price_risk > 0:
                    position_value = risk_amount / (SL_PRICE_PCT * LEVERAGE)
                else:
                    position_value = equity * LEVERAGE * 0.5

                # 不能超过可用杠杆
                max_position = equity * LEVERAGE
                position_value = min(position_value, max_position)

                entry_price = price * (1 + SLIPPAGE)  # 买入滑点
                highest_since_entry = entry_price
                entry_date = date
                bars_held = 0
                in_position = True

        # ==== 记录权益 ====
        if in_position:
            mark_price = price
            unreal = (mark_price / entry_price - 1) * LEVERAGE
            total_equity = equity + position_value * unreal
        else:
            total_equity = equity

        nav.append({"date": date, "equity": total_equity,
                    "in_position": in_position, "price": price})

    # ---- 绩效 ----
    nav_df = pd.DataFrame(nav).set_index("date")
    final_eq = nav_df["equity"].iloc[-1]
    total_ret = (final_eq / CAPITAL) - 1
    total_days = len(nav_df)
    ann_ret = (final_eq / CAPITAL) ** (TRADING_DAYS / total_days) - 1

    returns = nav_df["equity"].pct_change().dropna()
    ann_vol = float(returns.std() * np.sqrt(TRADING_DAYS))
    sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 0 else 0

    peak = nav_df["equity"].expanding().max()
    max_dd = float(((nav_df["equity"] - peak) / peak).min())

    btc_init = nav_df["price"].iloc[0]
    btc_final = nav_df["price"].iloc[-1]
    btc_ann = (btc_final / btc_init) ** (TRADING_DAYS / total_days) - 1

    sell_trades = [t for t in trades]
    win_trades = [t for t in sell_trades if t["pnl_$"] > 0]
    win_rate = len(win_trades) / len(sell_trades) if sell_trades else 0
    total_pnl = sum(t["pnl_$"] for t in trades)
    avg_win = np.mean([t["pnl_pct"] for t in win_trades]) / LEVERAGE if win_trades else 0
    avg_loss = np.mean([t["pnl_pct"] for t in sell_trades if t["pnl_$"] <= 0]) / LEVERAGE if sell_trades else 0
    avg_bars = np.mean([t["bars"] for t in sell_trades]) if sell_trades else 0
    total_fees = sum(position_value * FEE * 2 for _ in trades)
    # 近似
    total_trades = len(trades)

    # ---- 画图 ----
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [2.5, 1, 1]})

    ax1 = axes[0]
    ax1.plot(nav_df.index, nav_df["equity"] / CAPITAL,
             label=f"杠杆超短线 (5x)", color="#1f77b4", linewidth=1.5)
    btc_nav = nav_df["price"] / btc_init
    ax1.plot(nav_df.index, btc_nav, label="BTC 现货持有",
             color="#ff7f0e", linewidth=1.0, alpha=0.7)
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    # 标注交易区间
    for t in trades:
        ax1.axvspan(t["entry_date"], t["exit_date"], alpha=0.1,
                    color="green" if t["pnl_$"] > 0 else "red")
    ax1.set_title("BTC 杠杆超短线 (5x) · RSI 恐慌抄底", fontsize=14, fontweight="bold")
    ax1.set_ylabel("净值", fontsize=11)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(range(len(trades)), [t["pnl_pct"] for t in trades],
            color=["green" if t["pnl_$"] > 0 else "red" for t in trades], alpha=0.7)
    ax2.axhline(y=0, color="gray", linewidth=0.5)
    ax2.set_ylabel("每笔盈亏 %", fontsize=10)
    ax2.set_title(f"每笔交易盈亏（{len(trades)}笔，胜率{win_rate*100:.0f}%）", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")

    ax3 = axes[2]
    dd = (nav_df["equity"] - nav_df["equity"].expanding().max()) / nav_df["equity"].expanding().max()
    ax3.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
    ax3.set_ylabel("回撤", fontsize=10)
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("crypto_scalping.png", dpi=150, bbox_inches="tight")
    print("[Scalp] 图表已保存: crypto_scalping.png")

    print(f"\n{'='*60}")
    print(f"  BTC 杠杆超短线 (5x) · 绩效报告")
    print(f"{'='*60}")
    print(f"  初始本金:       ${CAPITAL:,.0f}")
    print(f"  最终权益:       ${final_eq:,.0f}")
    print(f"  总收益率:        {total_ret*100:.1f}%")
    print(f"  年化收益:        {ann_ret*100:.1f}%")
    print(f"  年化波动:        {ann_vol*100:.1f}%")
    print(f"  夏普比率:         {sharpe:.2f}")
    print(f"  最大回撤:         {max_dd*100:.1f}%")
    print(f"  ──────────────────────────────")
    print(f"  交易次数:         {total_trades}")
    print(f"  胜率:             {win_rate*100:.0f}%")
    print(f"  总盈亏:           ${total_pnl:,.0f}")
    print(f"  平均价格盈利:      {avg_win*100:.2f}%")
    print(f"  平均价格亏损:      {avg_loss*100:.2f}%")
    print(f"  平均持仓:          {avg_bars:.1f} 天")
    print(f"  ──────────────────────────────")
    print(f"  BTC 现货年化:     {btc_ann*100:.1f}%")
    print(f"  策略 vs BTC:      {(ann_ret-btc_ann)*100:.1f}%")
    print(f"{'='*60}")

    print(f"\n  最近 10 笔交易:")
    for t in trades[-10:]:
        emoji = "🟢" if t["pnl_$"] > 0 else "🔴"
        print(f"  {emoji} {str(t['entry_date'])[:10]} → {str(t['exit_date'])[:10]}  "
              f"${t['entry']:,.0f} → ${t['exit']:,.0f}  "
              f"{t['pnl_pct']:+.1f}%  {t['reason']}")

    elapsed = time.time() - t0
    print(f"\n  Total: {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC 杠杆超短线")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_scalping(quick=args.quick)
