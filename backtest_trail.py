"""
RSI 均值回归策略回测 — 对比移动止盈 vs 固定止盈止损
用法: python backtest_trail.py
"""
import numpy as np, pandas as pd, yfinance as yf

# === 与实盘一致的参数 ===
RSI_P = 5; RSI_L = 25; RSI_S = 78; MIN_VOL = 1.2
TP = 0.015          # 止盈 1.5%
SL = 0.004          # 硬止损 / 移动止盈距离 0.4%
ACTIVATE = 0.006    # 浮盈超过 0.6% 才启动移动止盈
PERIOD = "3mo"      # 回测周期
INTERVAL = "1h"     # K线周期
COINS = ["ETH-USD","SOL-USD","BNB-USD","AVAX-USD","DOGE-USD"]

def compute_rsi(close, period=RSI_P):
    d = close.diff().dropna()
    if len(d) < period: return 50.0
    g = d.clip(lower=0); l = (-d).clip(lower=0)
    ag = g.ewm(alpha=1/period, adjust=False).mean()
    al = l.ewm(alpha=1/period, adjust=False).mean()
    if al.iloc[-1] == 0: return 100.0 if ag.iloc[-1] > 0 else 50.0
    return float(100 - 100 / (1 + ag.iloc[-1] / al.iloc[-1]))

def backtest(df, use_trail=False):
    """返回 trades 列表, 每笔 {'pnl%','reason'}"""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    trades = []
    pos = None  # {'dir','entry','tp','sl','best'}

    for i in range(50, len(df)-1):
        price = float(c.iloc[i])
        hi = float(h.iloc[i]); lo = float(l.iloc[i])

        # ── 检查离场 ──
        if pos:
            if pos['dir'] == 'long':
                # 止盈
                if hi >= pos['tp']:
                    trades.append({'pnl%': TP, 'reason': 'tp'})
                    pos = None; continue
                # 更新最佳价
                if hi > pos['best']: pos['best'] = hi

                if use_trail:
                    # 浮盈超过激活门槛后才启动移动止盈
                    float_pnl = (pos['best'] - pos['entry']) / pos['entry']
                    if float_pnl >= ACTIVATE:
                        # 已激活：止损线跟随最高价 -0.4%
                        sl = pos['best'] * (1 - SL)
                        if lo <= sl:
                            pnl = (sl - pos['entry']) / pos['entry']
                            trades.append({'pnl%': pnl, 'reason': 'trail'})
                            pos = None; continue
                    else:
                        # 未激活：保留硬止损
                        if lo <= pos['sl']:
                            trades.append({'pnl%': -SL, 'reason': 'sl'})
                            pos = None; continue
                else:
                    # 无移动止盈：固定止损
                    if lo <= pos['sl']:
                        trades.append({'pnl%': -SL, 'reason': 'sl'})
                        pos = None; continue
            else:  # short
                if lo <= pos['tp']:
                    trades.append({'pnl%': TP, 'reason': 'tp'})
                    pos = None; continue
                if lo < pos['best']: pos['best'] = lo

                if use_trail:
                    float_pnl = (pos['entry'] - pos['best']) / pos['entry']
                    if float_pnl >= ACTIVATE:
                        sl = pos['best'] * (1 + SL)
                        if hi >= sl:
                            pnl = (pos['entry'] - sl) / pos['entry']
                            trades.append({'pnl%': pnl, 'reason': 'trail'})
                            pos = None; continue
                    else:
                        if hi >= pos['sl']:
                            trades.append({'pnl%': -SL, 'reason': 'sl'})
                            pos = None; continue
                else:
                    if hi >= pos['sl']:
                        trades.append({'pnl%': -SL, 'reason': 'sl'})
                        pos = None; continue

        # ── 检查入场 ──
        if pos is None:
            rsi = compute_rsi(c.iloc[max(0,i-50):i+1])
            vm = v.iloc[max(0,i-20):i+1].mean()
            vr = float(v.iloc[i] / vm) if vm > 0 else 1

            if rsi < RSI_L and vr > MIN_VOL and price > lo * 1.003:
                tp_price = price * (1 + TP)
                sl_price = price * (1 - SL)
                pos = {'dir': 'long', 'entry': price, 'tp': tp_price, 'sl': sl_price, 'best': price}
            elif rsi > RSI_S and vr > MIN_VOL and price < hi * 0.997:
                tp_price = price * (1 - TP)
                sl_price = price * (1 + SL)
                pos = {'dir': 'short', 'entry': price, 'tp': tp_price, 'sl': sl_price, 'best': price}

    return trades


def stats(trades):
    """汇总统计"""
    if not trades: return {}
    wins = [t['pnl%'] for t in trades if t['pnl%'] > 0]
    losses = [t['pnl%'] for t in trades if t['pnl%'] <= 0]
    return {
        'count': len(trades),
        'wr': len(wins)/len(trades)*100,
        'avg_win': np.mean(wins)*100 if wins else 0,
        'avg_loss': np.mean(losses)*100 if losses else 0,
        'net': sum(t['pnl%'] for t in trades)*100,
        'rr': abs(np.mean(wins)/np.mean(losses)) if wins and losses else 0,
        'tp': sum(1 for t in trades if t['reason']=='tp'),
        'sl': sum(1 for t in trades if t['reason']=='sl'),
        'trail': sum(1 for t in trades if t['reason']=='trail'),
    }


if __name__ == "__main__":
    print("=" * 90)
    print(f"  RSI 均值回归回测 | {PERIOD} {INTERVAL} | TP={TP:.1%} SL={SL:.1%} 激活={ACTIVATE:.1%}")
    print("=" * 90)
    print(f"{'币种':<10} {'模式':<8} {'交易数':>5} {'胜率':>6} {'均盈':>7} {'均亏':>7} {'净利':>7} {'盈亏比':>6}  {'明细'}")
    print("-" * 90)

    all_no = []; all_yes = []

    for symbol in COINS:
        try:
            df = yf.Ticker(symbol).history(period=PERIOD, interval=INTERVAL)
            if len(df) < 100:
                print(f"{symbol:<10} 数据不足")
                continue

            for label, use_trail in [("无移动", False), ("移止", True)]:
                t = backtest(df, use_trail)
                if not t: continue
                s = stats(t)
                detail = f"止盈{s['tp']}/止损{s['sl']}"
                if s['trail']: detail += f"/移止{s['trail']}"
                print(f"{symbol:<10} {label:<8} {s['count']:>5} {s['wr']:>5.0f}% {s['avg_win']:>6.2f}% {s['avg_loss']:>6.2f}% {s['net']:>6.2f}% {s['rr']:>5.1f}  {detail}")

                if use_trail: all_yes.extend(t)
                else: all_no.extend(t)
        except Exception as e:
            print(f"{symbol:<10} 错误: {e}")

    # 合计
    print("-" * 90)
    for label, trades in [("无移动止盈", all_no), ("移动止盈  ", all_yes)]:
        if not trades: continue
        s = stats(trades)
        detail = f"止盈{s['tp']}/止损{s['sl']}"
        if s['trail']: detail += f"/移止{s['trail']}"
        emoji = "⭐" if "移" in label else "  "
        print(f"{emoji} {'合计':<8} {label:<8} {s['count']:>5} {s['wr']:>5.0f}% {s['avg_win']:>6.2f}% {s['avg_loss']:>6.2f}% {s['net']:>6.2f}% {s['rr']:>5.1f}  {detail}")

    print()
    print("  硬止损 = -0.4% 始终有效")
    print("  移止 = 浮盈 > 0.6% 激活，回撤 0.4% 平仓保利")
