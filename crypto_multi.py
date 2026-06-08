"""
多币种并发杠杆超短线
- 同时监控 ETH/SOL/BNB/AVAX/DOGE
- 哪个出 RSI 极端信号就开哪个
- 最多同时持有 3 个仓位，资金分配 30%/仓
- 每个币独立止盈止损

用法:
    python crypto_multi.py
"""

import time, argparse
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
LEVERAGE = 10
CAPITAL = 10_000
RISK_PER_TRADE = 0.03       # 每笔风险 3% 本金（并发降低单仓风险）
TP_PRICE_PCT = 0.015        # 止盈 1.5%（×10 = 15%）
SL_PRICE_PCT = 0.004        # 止损 0.4%（×10 = 4%）
TRAILING_STOP = 0.004
RSI_PERIOD = 5
RSI_LONG = 25
RSI_SHORT = 78
MIN_VOL = 1.2
MAX_HOLD = 3
MAX_POSITIONS = 3
MAX_POSITION_SIZE = 500_000 # 单仓名义价值上限 $50万（模拟流动性限制）
MAKER_FEE = 0.0002
LIMIT_OFFSET = 0.0005
LIMIT_FILL = 0.85
WICK = 0.08
TRADING_DAYS = 365

# 交易标的
COINS = ["ETH-USD", "SOL-USD", "BNB-USD", "AVAX-USD", "DOGE-USD"]
BENCHMARK = "ETH-USD"


def compute_rsi(close, period):
    d = close.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    return 100 - 100 / (1 + g.ewm(alpha=1/period, adjust=False).mean() /
                         l.ewm(alpha=1/period, adjust=False).mean().replace(0, 1e-10))


def run_multi(quick=False):
    t0 = time.time()
    np.random.seed(7)

    # ---- 加载所有币的数据 ----
    print(f"[Multi] 加载 {len(COINS)} 个币种数据...")
    coin_data = {}
    for sym in COINS:
        try:
            df = yf.Ticker(sym).history(start="2023-01-01", end="2026-06-08")
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            if len(df) > 200:
                # 计算指标
                df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)
                df["atr14"] = (df["high"] - df["low"]).rolling(14).mean()
                df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
                coin_data[sym] = df
                print(f"  {sym}: {len(df)} 天")
        except Exception as e:
            print(f"  {sym}: 失败 - {e}")

    # 找到所有币种共同的日期范围，取最大的日期集合
    all_dates = set()
    for df in coin_data.values():
        all_dates.update(df["date"].tolist())
    all_dates = sorted(all_dates)

    # ---- 回测 ----
    equity = float(CAPITAL)
    # 每个仓位: {symbol, direction, entry_price, position_value, entry_date, bars, highest, lowest}
    positions = []
    trades = []
    nav = []

    for date in all_dates:
        # ==== 逐仓检查平仓 ====
        for pos in list(positions):
            sym = pos["symbol"]
            df = coin_data.get(sym)
            if df is None:
                continue
            row = df[df["date"] == date]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            price = row["close"]
            ph, pl = row["high"], row["low"]

            pos["bars"] += 1
            pos["highest"] = max(pos["highest"], ph)
            pos["lowest"] = min(pos["lowest"], pl)

            pc = ((price - pos["entry_price"]) / pos["entry_price"]
                  if pos["direction"] == "long"
                  else (pos["entry_price"] - price) / pos["entry_price"])

            atr_p = row["atr14"] / price
            dr = (ph - pl) / price
            is_wick = dr > 1.5 * atr_p and np.random.random() < WICK

            exit_price = None
            reason = ""

            if is_wick:
                wd = np.random.uniform(2, 5)
                if pos["direction"] == "long" and pl < pos["entry_price"] * (1 - SL_PRICE_PCT * wd):
                    exit_price = pos["entry_price"] * (1 - SL_PRICE_PCT * wd)
                    reason = "插针"
                elif pos["direction"] == "short" and ph > pos["entry_price"] * (1 + SL_PRICE_PCT * wd):
                    exit_price = pos["entry_price"] * (1 + SL_PRICE_PCT * wd)
                    reason = "插针"

            if not exit_price:
                if pc >= TP_PRICE_PCT:
                    exit_price = pos["entry_price"] * (1 + TP_PRICE_PCT if pos["direction"] == "long" else 1 - TP_PRICE_PCT)
                    reason = "止盈"
                elif pc <= -SL_PRICE_PCT:
                    exit_price = pos["entry_price"] * (1 - SL_PRICE_PCT if pos["direction"] == "long" else 1 + SL_PRICE_PCT)
                    reason = "止损"
                elif pos["direction"] == "long" and pos["highest"] > pos["entry_price"]:
                    dd = (pos["highest"] - price) / pos["entry_price"]
                    if dd > TRAILING_STOP and pc > 0:
                        exit_price = price; reason = "移动止盈"
                elif pos["direction"] == "short" and pos["lowest"] < pos["entry_price"]:
                    rally = (price - pos["lowest"]) / pos["entry_price"]
                    if rally > TRAILING_STOP and pc > 0:
                        exit_price = price; reason = "移动止盈"
                elif pos["bars"] >= MAX_HOLD:
                    exit_price = price; reason = "超时"
                elif pos["direction"] == "long" and row["rsi"] > 70:
                    exit_price = price; reason = "RSI回落"
                elif pos["direction"] == "short" and row["rsi"] < 35:
                    exit_price = price; reason = "RSI回升"

            if exit_price is not None:
                ef = exit_price * (1 + LIMIT_OFFSET if pos["direction"] == "long" else 1 - LIMIT_OFFSET)
                if pos["direction"] == "long":
                    pnl = (ef / pos["entry_price"] - 1) * LEVERAGE
                else:
                    pnl = (1 - ef / pos["entry_price"]) * LEVERAGE
                pnl_dollar = pos["position_value"] * pnl - pos["position_value"] * MAKER_FEE * 2
                equity += pnl_dollar
                trades.append({
                    "symbol": sym, "dir": pos["direction"],
                    "entry_date": pos["entry_date"], "exit_date": date,
                    "entry": pos["entry_price"], "exit": ef,
                    "pnl_pct": pnl * 100, "pnl_$": pnl_dollar,
                    "reason": reason, "bars": pos["bars"],
                })
                positions.remove(pos)

        # ==== 空仓位时扫描入场信号 ====
        if len(positions) < MAX_POSITIONS:
            for sym in COINS:
                if len(positions) >= MAX_POSITIONS:
                    break
                # 已经有这个币的仓位就不重复开
                if any(p["symbol"] == sym for p in positions):
                    continue
                df = coin_data.get(sym)
                if df is None:
                    continue
                row = df[df["date"] == date]
                if len(row) == 0:
                    continue
                row = row.iloc[0]
                price = row["close"]
                rsi_v = row["rsi"]
                vol_r = row["vol_ratio"]
                ph, pl = row["high"], row["low"]

                if pd.isna(rsi_v):
                    continue

                direction = None
                if rsi_v < RSI_LONG and vol_r > MIN_VOL and price > pl * 1.003:
                    direction = "long"
                elif rsi_v > RSI_SHORT and vol_r > MIN_VOL and price < ph * 0.997:
                    direction = "short"

                if direction and np.random.random() < LIMIT_FILL:
                    risk_amount = equity * RISK_PER_TRADE
                    pos_value = risk_amount / (SL_PRICE_PCT * LEVERAGE)
                    pos_value = min(pos_value, equity * LEVERAGE, MAX_POSITION_SIZE)
                    entry_price = price * (1 - LIMIT_OFFSET if direction == "long" else 1 + LIMIT_OFFSET)
                    positions.append({
                        "symbol": sym,
                        "direction": direction,
                        "entry_price": entry_price,
                        "position_value": pos_value,
                        "entry_date": date,
                        "bars": 0,
                        "highest": entry_price,
                        "lowest": entry_price,
                    })

        # ==== 记录每日权益 ====
        total_equity = equity
        for pos in positions:
            df = coin_data.get(pos["symbol"])
            if df is not None:
                row = df[df["date"] == date]
                if len(row) > 0:
                    p = row["close"].iloc[0]
                    if pos["direction"] == "long":
                        unreal = (p / pos["entry_price"] - 1) * LEVERAGE
                    else:
                        unreal = (1 - p / pos["entry_price"]) * LEVERAGE
                    total_equity += pos["position_value"] * unreal

        nav.append({"date": date, "equity": total_equity, "num_pos": len(positions)})

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

    sell_trades = [t for t in trades if t["reason"] != ""]
    win_trades = [t for t in sell_trades if t["pnl_$"] > 0]
    win_rate = len(win_trades) / len(sell_trades) if sell_trades else 0
    wick_count = sum(1 for t in trades if t["reason"] == "插针")

    # 按币种统计
    by_coin = {}
    for t in trades:
        s = t["symbol"]
        if s not in by_coin:
            by_coin[s] = {"trades": 0, "pnl": 0, "wins": 0}
        by_coin[s]["trades"] += 1
        by_coin[s]["pnl"] += t["pnl_$"]
        if t["pnl_$"] > 0:
            by_coin[s]["wins"] += 1

    # ---- 画图 ----
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [2.5, 1, 1]})

    ax1 = axes[0]
    ax1.plot(nav_df.index, nav_df["equity"] / CAPITAL, label="多币种并发", color="#1f77b4", linewidth=1.5)
    # 叠加每个币的单独表现（净值归一）
    for sym in COINS:
        df = coin_data.get(sym)
        if df is not None:
            df = df.set_index("date")
            ax1.plot(df.index, df["close"] / df["close"].iloc[0],
                     linewidth=0.5, alpha=0.4, label=sym.replace("-USD", ""))
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title(f"多币种并发 · {len(COINS)} 币 · 10x 杠杆", fontsize=14, fontweight="bold")
    ax1.set_ylabel("净值", fontsize=11)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for t in trades:
        color = "green" if t["pnl_$"] > 0 else "red"
        ax2.bar(t["exit_date"], t["pnl_pct"], color=color, alpha=0.6, width=1.5)
    ax2.axhline(y=0, color="gray", linewidth=0.5)
    ax2.set_ylabel("每笔盈亏 %", fontsize=10)
    ax2.set_title(f"交易盈亏分布（{len(trades)}笔，胜率{win_rate*100:.0f}%）", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")

    ax3 = axes[2]
    dd = (nav_df["equity"] - nav_df["equity"].expanding().max()) / nav_df["equity"].expanding().max()
    ax3.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
    ax3.set_ylabel("回撤", fontsize=10)
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("crypto_multi.png", dpi=150, bbox_inches="tight")
    print("[Multi] 图表: crypto_multi.png")

    print(f"\n{'='*60}")
    print(f"  多币种并发 · 绩效报告")
    print(f"{'='*60}")
    print(f"  交易币种:       {', '.join(c.replace('-USD','') for c in COINS)}")
    print(f"  初始本金:       ${CAPITAL:,.0f}")
    print(f"  最终权益:       ${final_eq:,.0f}")
    print(f"  总收益率:        {total_ret*100:.1f}%")
    print(f"  年化收益:        {ann_ret*100:.1f}%")
    print(f"  年化波动:        {ann_vol*100:.1f}%")
    print(f"  夏普比率:         {sharpe:.2f}")
    print(f"  最大回撤:         {max_dd*100:.1f}%")
    print(f"  ──────────────────────────────")
    print(f"  总交易次数:       {len(trades)}")
    print(f"  插针次数:         {wick_count}")
    print(f"  胜率:             {win_rate*100:.0f}%")
    print(f"  总盈亏:           ${sum(t['pnl_$'] for t in trades):,.0f}")
    print(f"  ──────────────────────────────")
    for sym, stats in sorted(by_coin.items()):
        wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
        print(f"  {sym.replace('-USD',''):<6s}  {stats['trades']:3d}笔  胜率{wr:.0f}%  PnL ${stats['pnl']:,.0f}")
    print(f"{'='*60}")

    print(f"\n  最近 10 笔交易:")
    for t in trades[-10:]:
        emoji = "🟢" if t["pnl_$"] > 0 else "🔴"
        print(f"  {emoji} {t['symbol']:<8s} {str(t['entry_date'])[:10]} → {str(t['exit_date'])[:10]}  "
              f"{t['pnl_pct']:+.1f}%  {t['reason']}")

    elapsed = time.time() - t0
    print(f"\n  Total: {elapsed:.1f}s")


if __name__ == "__main__":
    run_multi()
