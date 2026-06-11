"""
ETH 杠杆超短线策略
- 10x 杠杆做 ETH 永续合约（波动比 ETH 大，更适合短线）
- RSI 极端超卖抄底 + 超买做空
- 限价单入场，省手续费
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
LEVERAGE = 10               # 10 倍杠杆
CAPITAL = 10_000            # 初始本金 $10,000
RISK_PER_TRADE = 0.05       # 每笔风险 = 本金的 5%
TP_PRICE_PCT = 0.015        # 止盈：价格涨 1.5%（×10 = 赚 15%）
SL_PRICE_PCT = 0.004        # 止损：价格跌 0.4%（×10 = 亏 4%）
RSI_PERIOD = 5              # RSI 周期
RSI_ENTRY_LONG = 25         # 超卖做多阈值
RSI_ENTRY_SHORT = 78        # 超买做空阈值
MIN_VOL_SPIKE = 1.2         # 放量确认
TRAILING_STOP = 0.004       # 移动止盈
MAX_HOLD_BARS = 3
MAKER_FEE = 0.0002          # 限价单手续费 0.02%（比市价单便宜一半）
TAKER_FEE = 0.0004           # 市价单手续费 0.04%（没成交时的 fallback）
LIMIT_OFFSET = 0.0005        # 限价单挂单偏移 0.05%（挂低买、挂高卖）
LIMIT_FILL_PROB = 0.85       # 限价单当日成交概率
SLIPPAGE = 0.0001            # 限价单滑点极小
WICK_RISK_PCT = 0.08         # 插针概率
EVENT_FILTER = False          # 事件日历过滤（设为 True 则 FOMC/CPI 日不开仓）
TRADING_DAYS = 365

# ============================================================
# 事件日历（高影响力宏观事件日）
# ============================================================
import datetime as _dt

def _date(s): return _dt.datetime.strptime(s, "%Y-%m-%d").date()

HIGH_IMPACT_EVENTS = set()
# FOMC 利率决议日（2023-2026，包含前后各1天缓冲）
_fomc = [
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26",
    "2023-09-20","2023-11-01","2023-12-13",
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31",
    "2024-09-18","2024-11-07","2024-12-18",
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30",
    "2025-09-17","2025-11-06","2025-12-10",
    "2026-01-28","2026-03-18",
]
for d in _fomc:
    base = _date(d)
    for offset in [-1, 0, 1]:  # 前一天、当天、后一天
        HIGH_IMPACT_EVENTS.add((base + _dt.timedelta(days=offset)).strftime("%Y-%m-%d"))

# CPI 发布日（每月中旬，加前后缓冲）
_cpi_months = [(y, m) for y in range(2023, 2027) for m in range(1, 13)]
for y, m in _cpi_months:
    try:
        d = _date(f"{y}-{m:02d}-12")
        HIGH_IMPACT_EVENTS.add(d.strftime("%Y-%m-%d"))
        HIGH_IMPACT_EVENTS.add((d - _dt.timedelta(days=1)).strftime("%Y-%m-%d"))
        HIGH_IMPACT_EVENTS.add((d + _dt.timedelta(days=1)).strftime("%Y-%m-%d"))
    except:
        pass

# 币圈专属大事件
_crypto_events = [
    "2024-01-10","2024-01-11","2024-01-12",  # ETF 批准
    "2024-04-19","2024-04-20",                # BTC 减半
    "2024-11-05","2024-11-06",                # 美国大选
    "2025-01-20","2025-01-21",                # 川普就职
    "2025-08-05","2025-08-06",                # 加密市场闪崩
]
for d in _crypto_events:
    HIGH_IMPACT_EVENTS.add(d)

print(f"[Event] 事件日历: {len(HIGH_IMPACT_EVENTS)} 个高风险日")


def is_event_day(date: pd.Timestamp) -> bool:
    """检查是否为高影响力事件日"""
    return EVENT_FILTER and date.strftime("%Y-%m-%d") in HIGH_IMPACT_EVENTS


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
    np.random.seed(7)   # 固定随机种子（可改为任意整数，不同种子结果不同）
    print("[Scalp] 获取 ETH 数据...")
    df = yf.Ticker("ETH-USD").history(start="2023-01-01", end="2026-06-08")
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
    entry_fee_rate = MAKER_FEE
    highest_since_entry = 0.0
    bars_held = 0
    in_position = False
    wick_count = 0
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
            highest_since_entry = max(highest_since_entry, price_high)
            lowest_since_entry = min(lowest_since_entry, price_low)

            # 价格变化（做多正向，做空反向）
            price_change = ((price - entry_price) / entry_price)
            if direction == "short":
                price_change = -price_change

            exit_price = None
            exit_reason = ""

            # ⚡ 插针检查：当日振幅 > 3倍 ATR，可能跳过止损
            atr_pct = row["atr14"] / price
            daily_range = (price_high - price_low) / price
            is_wick_day = daily_range > 1.5 * atr_pct  # 高波动日
            wick_hit = False

            if is_wick_day and np.random.random() < WICK_RISK_PCT:
                # 插针深度：2-5倍正常止损距离
                wick_depth = np.random.uniform(2, 5)
                if direction == "long" and price_low < entry_price * (1 - SL_PRICE_PCT * wick_depth):
                    exit_price = entry_price * (1 - SL_PRICE_PCT * wick_depth)
                    exit_reason = "⚡插针"
                    wick_hit = True
                    wick_count += 1
                elif direction == "short" and price_high > entry_price * (1 + SL_PRICE_PCT * wick_depth):
                    exit_price = entry_price * (1 + SL_PRICE_PCT * wick_depth)
                    exit_reason = "⚡插针"
                    wick_hit = True
                    wick_count += 1

            if not wick_hit:
                if direction == "long":
                    # 做多平仓
                    if price_change >= TP_PRICE_PCT:
                        exit_price = entry_price * (1 + TP_PRICE_PCT)
                        exit_reason = "止盈"
                    elif price_change <= -SL_PRICE_PCT:
                        exit_price = entry_price * (1 - SL_PRICE_PCT)
                        exit_reason = "止损"
                    elif highest_since_entry > entry_price:
                        dd = (highest_since_entry - price) / entry_price
                        if dd > TRAILING_STOP and price_change > 0:
                            exit_price = price
                            exit_reason = "移动止盈"
                    elif bars_held >= MAX_HOLD_BARS:
                        exit_price = price
                        exit_reason = f"超时{bars_held}天"
                    elif not pd.isna(rsi_val) and rsi_val > 70:
                        exit_price = price
                        exit_reason = "RSI回落"
                else:
                    # 做空平仓
                    if price_change >= TP_PRICE_PCT:
                        exit_price = entry_price * (1 - TP_PRICE_PCT)
                        exit_reason = "止盈"
                    elif price_change <= -SL_PRICE_PCT:
                        exit_price = entry_price * (1 + SL_PRICE_PCT)
                        exit_reason = "止损"
                    elif lowest_since_entry < entry_price:
                        rally = (price - lowest_since_entry) / entry_price
                        if rally > TRAILING_STOP and price_change > 0:
                            exit_price = price
                            exit_reason = "移动止盈"
                    elif bars_held >= MAX_HOLD_BARS:
                        exit_price = price
                        exit_reason = f"超时{bars_held}天"
                    elif not pd.isna(rsi_val) and rsi_val < 35:
                        exit_price = price
                        exit_reason = "RSI回升"

            if exit_price is not None:
                # 限价卖出价 = 平仓价 × (1 + 偏移)（挂高卖）
                exit_fill = exit_price * (1 + LIMIT_OFFSET if direction == "long" else 1 - LIMIT_OFFSET)
                if direction == "long":
                    pnl_pct = (exit_fill / entry_price - 1) * LEVERAGE
                else:
                    pnl_pct = (1 - exit_fill / entry_price) * LEVERAGE
                pnl_dollar = position_value * pnl_pct - position_value * (MAKER_FEE + entry_fee_rate)
                equity += pnl_dollar
                if equity <= 0:
                    equity = 0.01
                    break

                trades.append({
                    "dir": direction, "entry_date": entry_date, "exit_date": date,
                    "entry": entry_price, "exit": exit_fill,
                    "pnl_pct": pnl_pct * 100, "pnl_$": pnl_dollar,
                    "reason": exit_reason, "bars": bars_held,
                })
                position_value = 0
                in_position = False

        # ==== 空仓中：看入场信号 ====
        if not in_position and not pd.isna(rsi_val):
            # 事件日不开新仓（FOMC/CPI/重大消息面）
            if is_event_day(date):
                nav.append({"date": date, "equity": equity,
                           "in_position": False, "price": price})
                continue

            direction = None

            # 做多：超卖 + 放量 + 反弹确认
            long_sig = (rsi_val < RSI_ENTRY_LONG and vol_ratio > MIN_VOL_SPIKE
                       and price > price_low * 1.003)
            # 做空：超买 + 放量 + 回落确认
            short_sig = (rsi_val > RSI_ENTRY_SHORT and vol_ratio > MIN_VOL_SPIKE
                        and price < price_high * 0.997)

            if long_sig:
                direction = "long"
            elif short_sig:
                direction = "short"

            if direction:
                # 限价单：挂低 0.05% 买入，85% 概率成交
                if np.random.random() < LIMIT_FILL_PROB:
                    risk_amount = equity * RISK_PER_TRADE
                    price_risk = SL_PRICE_PCT * LEVERAGE
                    position_value = risk_amount / (SL_PRICE_PCT * LEVERAGE) if price_risk > 0 else equity * LEVERAGE * 0.5
                    max_position = equity * LEVERAGE
                    position_value = min(position_value, max_position)

                    # 限价买入价 = 市价 × (1 - 偏移)（挂低买）
                    entry_price = price * (1 - LIMIT_OFFSET if direction == "long" else 1 + LIMIT_OFFSET)
                    entry_fee_rate = MAKER_FEE  # 限价单 = maker，手续费低

                    highest_since_entry = entry_price
                    lowest_since_entry = entry_price
                    entry_date = date
                    bars_held = 0
                    in_position = True
                # 15% 概率没成交 → 等下一个信号

        # ==== 记录权益 ====
        if in_position:
            if direction == "long":
                unreal = (price / entry_price - 1) * LEVERAGE
            else:
                unreal = (1 - price / entry_price) * LEVERAGE
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
    total_trades = len(trades)

    # ---- 画图 ----
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [2.5, 1, 1]})

    ax1 = axes[0]
    ax1.plot(nav_df.index, nav_df["equity"] / CAPITAL,
             label=f"杠杆超短线 (5x)", color="#1f77b4", linewidth=1.5)
    btc_nav = nav_df["price"] / btc_init
    ax1.plot(nav_df.index, btc_nav, label="ETH 现货持有",
             color="#ff7f0e", linewidth=1.0, alpha=0.7)
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    # 标注交易区间
    for t in trades:
        ax1.axvspan(t["entry_date"], t["exit_date"], alpha=0.1,
                    color="green" if t["pnl_$"] > 0 else "red")
    ax1.set_title("ETH 杠杆超短线 (5x) · RSI 恐慌抄底", fontsize=14, fontweight="bold")
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
    plt.savefig("assets/crypto_scalping.png", dpi=150, bbox_inches="tight")
    print("[Scalp] 图表已保存: assets/crypto_scalping.png")

    print(f"\n{'='*60}")
    print(f"  ETH 杠杆超短线 (10x) · 绩效报告")
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
    print(f"  插针次数:         {wick_count}")
    print(f"  胜率:             {win_rate*100:.0f}%")
    print(f"  总盈亏:           ${total_pnl:,.0f}")
    print(f"  平均价格盈利:      {avg_win*100:.2f}%")
    print(f"  平均价格亏损:      {avg_loss*100:.2f}%")
    print(f"  平均持仓:          {avg_bars:.1f} 天")
    print(f"  ──────────────────────────────")
    print(f"  ETH 现货年化:     {btc_ann*100:.1f}%")
    print(f"  策略 vs ETH:      {(ann_ret-btc_ann)*100:.1f}%")
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
    parser = argparse.ArgumentParser(description="ETH 杠杆超短线")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_scalping(quick=args.quick)
