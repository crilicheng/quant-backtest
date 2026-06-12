"""
RSI 均值回归回测 — 独立复核版
- 信号用已收 K → 下K open入场 → SL优先 → 同K不移止 → 0.08%费
- 组合回测：最多3仓，权益复利，<$20停开
用法: python backtest_trail.py
"""
import numpy as np, pandas as pd, yfinance as yf

FEE = 0.0008; MAX_POS = 3; MIN_EQUITY = 20
PERIOD = "1y"; INTERVAL = "1h"
RSI_P = 5
COINS = ["ETH-USD","SOL-USD","BNB-USD","AVAX-USD","DOGE-USD"]

# === 三个候选方案 ===
CONFIGS = {
    "A-稳健": {
        "RSI_L": 20, "RSI_S": 82, "MIN_VOL": 1.5,
        "TP": 0.025, "SL": 0.006, "WICK": 0.003,
        "activate": 0.006, "trail": 0.003,
    },
    "B-激进": {
        "RSI_L": 25, "RSI_S": 78, "MIN_VOL": 1.2,
        "TP": 0.022, "SL": 0.004, "WICK": 0.003,
        "activate": 0.010, "trail": 0.0025,
    },
    "当前": {
        "RSI_L": 25, "RSI_S": 78, "MIN_VOL": 1.2,
        "TP": 0.015, "SL": 0.004, "WICK": 0.003,
        "activate": 0.006, "trail": 0.003,
    },
}

def compute_rsi(close, period=RSI_P):
    d = close.diff().dropna()
    if len(d) < period: return 50.0
    g = d.clip(lower=0); l = (-d).clip(lower=0)
    ag = g.ewm(alpha=1/period, adjust=False).mean()
    al = l.ewm(alpha=1/period, adjust=False).mean()
    if al.iloc[-1] == 0: return 100.0 if ag.iloc[-1] > 0 else 50.0
    return float(100 - 100 / (1 + ag.iloc[-1] / al.iloc[-1]))


def run_portfolio(df_dict, p, use_trail=False):
    """p = param dict from CONFIGS"""
    trades = []
    equity = 100.0
    nav = [100.0]
    dd = [0.0]
    peak = 100.0
    positions = []
    max_len = max(len(df) for df in df_dict.values())

    for i in range(50, max_len - 2):
        # === 离场 ===
        survived = []
        for pos in positions:
            sym = pos['sym']
            if i >= len(df_dict[sym]) - 1:
                continue
            hi = float(df_dict[sym]["High"].iloc[i])
            lo = float(df_dict[sym]["Low"].iloc[i])

            if pos['dir'] == 'long':
                pos['_prev_best'] = pos.get('_curr_best', pos['best'])
                pos['_curr_best'] = max(pos['best'], hi)
            else:
                pos['_prev_best'] = pos.get('_curr_best', pos['best'])
                pos['_curr_best'] = min(pos['best'], lo)

            exit_px = None; exit_reason = None

            # SL 优先
            if pos['dir'] == 'long':
                if lo <= pos['sl']: exit_px = pos['sl']; exit_reason = 'sl'
            else:
                if hi >= pos['sl']: exit_px = pos['sl']; exit_reason = 'sl'

            # TP
            if exit_px is None:
                if pos['dir'] == 'long':
                    if hi >= pos['tp']: exit_px = pos['tp']; exit_reason = 'tp'
                else:
                    if lo <= pos['tp']: exit_px = pos['tp']; exit_reason = 'tp'

            # 移止（用 prev_best 避免同K先涨后跌）
            if exit_px is None and use_trail:
                prev_best = pos.get('_prev_best', pos['best'])
                if pos['dir'] == 'long':
                    fp = (prev_best - pos['entry_px']) / pos['entry_px']
                    if fp >= p['activate']:
                        ts = prev_best * (1 - p['trail'])
                        if lo <= ts: exit_px = ts; exit_reason = 'trail'
                else:
                    fp = (pos['entry_px'] - prev_best) / pos['entry_px']
                    if fp >= p['activate']:
                        ts = prev_best * (1 + p['trail'])
                        if hi >= ts: exit_px = ts; exit_reason = 'trail'

            if exit_px:
                if pos['dir'] == 'long':
                    gross = (exit_px - pos['entry_px']) / pos['entry_px']
                else:
                    gross = (pos['entry_px'] - exit_px) / pos['entry_px']
                pnl = gross - FEE
                trades.append({
                    'sym': sym, 'dir': pos['dir'],
                    'entry_px': pos['entry_px'], 'exit_px': exit_px,
                    'pnl_net': pnl, 'reason': exit_reason,
                    'half': 1 if i < max_len // 2 else 2,
                })
                equity *= (1 + pnl)
            else:
                pos['best'] = pos.get('_prev_best', pos['best'])
                survived.append(pos)

        positions = survived
        nav.append(equity)
        if equity > peak: peak = equity
        dd.append((equity - peak) / peak)

        # === 入场 ===
        if len(positions) >= MAX_POS or equity < MIN_EQUITY:
            continue

        for sym, df in df_dict.items():
            if len(positions) >= MAX_POS or equity < MIN_EQUITY:
                break
            if i >= len(df) - 2: continue
            if any(p['sym'] == sym for p in positions): continue

            c_win = df["Close"].iloc[max(0,i-50):i+1]
            price = float(df["Close"].iloc[i])
            hi = float(df["High"].iloc[i])
            lo = float(df["Low"].iloc[i])
            v_win = df["Volume"].iloc[max(0,i-20):i+1]

            rsi = compute_rsi(c_win)
            vm = v_win.mean()
            vr = float(v_win.iloc[-1] / vm) if vm > 0 else 1

            sig = None
            # Wick filter: close must be beyond low/high by wick%
            if rsi < p['RSI_L'] and vr > p['MIN_VOL'] and price > lo * (1 + p['WICK']):
                sig = 'long'
            elif rsi > p['RSI_S'] and vr > p['MIN_VOL'] and price < hi * (1 - p['WICK']):
                sig = 'short'

            if sig:
                entry_px = float(df["Open"].iloc[i + 1])
                if sig == 'long':
                    sl_px = entry_px * (1 - p['SL'])
                    tp_px = entry_px * (1 + p['TP'])
                else:
                    sl_px = entry_px * (1 + p['SL'])
                    tp_px = entry_px * (1 - p['TP'])

                positions.append({
                    'sym': sym, 'dir': sig,
                    'entry_px': entry_px, 'sl': sl_px, 'tp': tp_px,
                    'best': entry_px,
                })

    # 计算 max_dd
    peak = 100.0; max_dd = 0.0
    for v in nav:
        if v > peak: peak = v
        d = (v - peak) / peak
        if d < max_dd: max_dd = d

    return trades, nav, max_dd


def summary(label, trades, nav, max_dd):
    if not trades: return None
    wins = [t['pnl_net'] for t in trades if t['pnl_net'] > 0]
    losses = [t['pnl_net'] for t in trades if t['pnl_net'] <= 0]
    from collections import Counter
    rc = Counter(t['reason'] for t in trades)

    t1 = [t for t in trades if t['half'] == 1]
    t2 = [t for t in trades if t['half'] == 2]
    half1 = sum(t['pnl_net'] for t in t1) * 100 if t1 else 0
    half2 = sum(t['pnl_net'] for t in t2) * 100 if t2 else 0

    return {
        'name': label,
        'count': len(trades),
        'wr': len(wins)/len(trades)*100,
        'avg_win': np.mean(wins)*100 if wins else 0,
        'avg_loss': np.mean(losses)*100 if losses else 0,
        'net': sum(t['pnl_net'] for t in trades)*100,
        'final': nav[-1],
        'max_dd': max_dd * 100,
        'half1': half1,
        'half2': half2,
        'tp': rc.get('tp', 0), 'sl': rc.get('sl', 0), 'trail': rc.get('trail', 0),
    }


if __name__ == "__main__":
    # 加载数据
    df_dict = {}
    for sym in COINS:
        df = yf.Ticker(sym).history(period=PERIOD, interval=INTERVAL)
        if len(df) > 100:
            df_dict[sym] = df
    print(f"数据: {PERIOD} {INTERVAL}, {len(df_dict)} coins\n")

    header = f"{'方案':<10} {'交易':>5} {'胜率':>6} {'均盈':>7} {'均亏':>7} {'净利':>8} {'终值':>7} {'最大回撤':>7} {'前半':>7} {'后半':>7}"
    print(header)
    print("-" * 90)

    for name, p in CONFIGS.items():
        for use_trail, trail_label in [(False, ""), (True, "+移止")]:
            label = f"{name}{trail_label}"
            trades, nav, max_dd = run_portfolio(df_dict, p, use_trail)
            s = summary(label, trades, nav, max_dd)
            if s:
                print(f"  {s['name']:<10} {s['count']:>5} {s['wr']:>5.0f}% {s['avg_win']:>6.2f}% {s['avg_loss']:>6.2f}% {s['net']:>7.2f}% {s['final']:>6.1f} {s['max_dd']:>6.2f}% {s['half1']:>+6.2f}% {s['half2']:>+6.2f}%  TP{s['tp']}/SL{s['sl']}/移{s['trail']}")

    print(f"\n  标准: 信号已收K → 下K open入场 → SL优先 → 同K不移止 → 0.08%费 → wick过滤")
