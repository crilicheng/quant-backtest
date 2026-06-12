"""
RSI 回测 — 复核 Codex 分币种移止优化
用法: python backtest_trail.py
"""
import numpy as np, pandas as pd, yfinance as yf
from collections import Counter

FEE = 0.0008; MAX_POS = 3; MIN_EQUITY = 20
PERIOD = "1y"; INTERVAL = "1h"; RSI_P = 5

# 固定入场参数（方案A）
P = {
    "RSI_L": 20, "RSI_S": 82, "MIN_VOL": 1.5,
    "TP": 0.025, "SL": 0.006, "WICK": 0.003,
}

COINS = ["ETH-USD","SOL-USD","BNB-USD","AVAX-USD","DOGE-USD"]

# 两套移止对比
TRAIL_UNIFIED = {s: {"activate": 0.006, "trail": 0.003} for s in COINS}
TRAIL_PERCOIN = {
    "ETH-USD":  {"activate": 0.006, "trail": 0.003},
    "SOL-USD":  {"activate": 0.015, "trail": 0.002},
    "BNB-USD":  {"activate": 0.004, "trail": 0.002},
    "AVAX-USD": {"activate": 0.006, "trail": 0.002},
    "DOGE-USD": {"activate": 0.012, "trail": 0.002},
}

def compute_rsi(close, period=RSI_P):
    d = close.diff().dropna()
    if len(d) < period: return 50.0
    g = d.clip(lower=0); l = (-d).clip(lower=0)
    ag = g.ewm(alpha=1/period, adjust=False).mean()
    al = l.ewm(alpha=1/period, adjust=False).mean()
    if al.iloc[-1] == 0: return 100.0 if ag.iloc[-1] > 0 else 50.0
    return float(100 - 100 / (1 + ag.iloc[-1] / al.iloc[-1]))

def run_portfolio(df_dict, trail_cfg, use_trail=False):
    trades = []; equity = 100.0; peak = 100.0; max_dd = 0
    positions = []; nav_equity = []
    max_len = max(len(df) for df in df_dict.values())

    for i in range(50, max_len - 2):
        survived = []
        for pos in positions:
            sym = pos['sym']
            if i >= len(df_dict[sym]) - 1: continue
            hi = float(df_dict[sym]["High"].iloc[i])
            lo = float(df_dict[sym]["Low"].iloc[i])

            if pos['dir'] == 'long':
                pos['_prev_best'] = pos.get('_curr_best', pos['best'])
                pos['_curr_best'] = max(pos['best'], hi)
            else:
                pos['_prev_best'] = pos.get('_curr_best', pos['best'])
                pos['_curr_best'] = min(pos['best'], lo)

            exit_px = None; exit_reason = None

            if pos['dir'] == 'long':
                if lo <= pos['sl']: exit_px = pos['sl']; exit_reason = 'sl'
            else:
                if hi >= pos['sl']: exit_px = pos['sl']; exit_reason = 'sl'

            if exit_px is None:
                if pos['dir'] == 'long':
                    if hi >= pos['tp']: exit_px = pos['tp']; exit_reason = 'tp'
                else:
                    if lo <= pos['tp']: exit_px = pos['tp']; exit_reason = 'tp'

            if exit_px is None and use_trail:
                prev_best = pos.get('_prev_best', pos['best'])
                cfg = pos.get('trail_cfg', {})
                if pos['dir'] == 'long':
                    fp = (prev_best - pos['entry_px']) / pos['entry_px']
                    if fp >= cfg.get('activate', 0.006):
                        ts = prev_best * (1 - cfg.get('trail', 0.003))
                        if lo <= ts: exit_px = ts; exit_reason = 'trail'
                else:
                    fp = (pos['entry_px'] - prev_best) / pos['entry_px']
                    if fp >= cfg.get('activate', 0.006):
                        ts = prev_best * (1 + cfg.get('trail', 0.003))
                        if hi >= ts: exit_px = ts; exit_reason = 'trail'

            if exit_px:
                gross = (exit_px-pos['entry_px'])/pos['entry_px'] if pos['dir']=='long' else (pos['entry_px']-exit_px)/pos['entry_px']
                pnl = gross - FEE
                trades.append({'sym':sym, 'pnl_net':pnl, 'reason':exit_reason, 'half': 1 if i < max_len//2 else 2})
                equity *= (1+pnl)
            else:
                pos['best'] = pos.get('_prev_best', pos['best'])
                survived.append(pos)

        positions = survived
        if equity > peak: peak = equity
        dd = (equity-peak)/peak
        if dd < max_dd: max_dd = dd

        if len(positions) >= MAX_POS or equity < MIN_EQUITY: continue

        for sym, df in df_dict.items():
            if len(positions) >= MAX_POS or equity < MIN_EQUITY: break
            if i >= len(df)-2: continue
            if any(p['sym']==sym for p in positions): continue

            c_win = df["Close"].iloc[max(0,i-50):i+1]
            price = float(df["Close"].iloc[i])
            hi = float(df["High"].iloc[i])
            lo = float(df["Low"].iloc[i])
            v_win = df["Volume"].iloc[max(0,i-20):i+1]

            rsi = compute_rsi(c_win)
            vm = v_win.mean()
            vr = float(v_win.iloc[-1]/vm) if vm>0 else 1

            sig = None
            if rsi < P['RSI_L'] and vr > P['MIN_VOL'] and price > lo*(1+P['WICK']):
                sig = 'long'
            elif rsi > P['RSI_S'] and vr > P['MIN_VOL'] and price < hi*(1-P['WICK']):
                sig = 'short'

            if sig:
                entry_px = float(df["Open"].iloc[i+1])
                sl_px = entry_px*(1-P['SL']) if sig=='long' else entry_px*(1+P['SL'])
                tp_px = entry_px*(1+P['TP']) if sig=='long' else entry_px*(1-P['TP'])
                positions.append({'sym':sym, 'dir':sig, 'entry_px':entry_px, 'sl':sl_px, 'tp':tp_px, 'best':entry_px, 'trail_cfg': trail_cfg.get(sym) if use_trail else None})

    return trades, equity, max_dd

def stats(trades, equity, max_dd):
    if not trades: return {}
    wins = [t['pnl_net'] for t in trades if t['pnl_net']>0]
    losses = [t['pnl_net'] for t in trades if t['pnl_net']<=0]
    rc = Counter(t['reason'] for t in trades)
    t1 = [t for t in trades if t['half']==1]
    t2 = [t for t in trades if t['half']==2]
    return {
        'count': len(trades), 'wr': len(wins)/len(trades)*100,
        'aw': np.mean(wins)*100 if wins else 0,
        'al': np.mean(losses)*100 if losses else 0,
        'net': sum(t['pnl_net'] for t in trades)*100,
        'final': equity, 'max_dd': max_dd*100,
        'half1': sum(t['pnl_net'] for t in t1)*100 if t1 else 0,
        'half2': sum(t['pnl_net'] for t in t2)*100 if t2 else 0,
        'tp': rc.get('tp',0), 'sl': rc.get('sl',0), 'trail': rc.get('trail',0),
    }

if __name__ == "__main__":
    df_dict = {}
    for sym in COINS:
        df = yf.Ticker(sym).history(period=PERIOD, interval=INTERVAL)
        if len(df)>100: df_dict[sym]=df

    print(f"{'配置':<14} {'交易':>5} {'胜率':>6} {'均盈':>7} {'均亏':>7} {'净利':>8} {'终值':>7} {'最大回撤':>7} {'前半':>7} {'后半':>7}  {'明细'}")
    print("-"*102)

    configs = [
        ("无移止", TRAIL_UNIFIED, False),
        ("统一移止", TRAIL_UNIFIED, True),
        ("分币种移止", TRAIL_PERCOIN, True),
    ]

    for label, tcfg, use_trail in configs:
        trades, equity, max_dd = run_portfolio(df_dict, tcfg, use_trail)
        s = stats(trades, equity, max_dd)
        detail = f"TP{s['tp']}/SL{s['sl']}/移{s['trail']}"
        print(f"  {label:<12} {s['count']:>5} {s['wr']:>5.1f}% {s['aw']:>6.2f}% {s['al']:>6.2f}% {s['net']:>7.2f}% {s['final']:>6.1f} {s['max_dd']:>6.2f}% {s['half1']:>+7.2f}% {s['half2']:>+7.2f}%  {detail}")

    # 分币种单币对比
    print(f"\n{'币种':<10} {'统一移止':>10} {'分币种移止':>10} {'差值':>8}")
    print("-"*42)
    for sym in COINS:
        # Single-coin backtest with per-coin configs
        t_u, eq_u, dd_u = run_portfolio({sym: df_dict[sym]}, TRAIL_UNIFIED, True)
        t_p, eq_p, dd_p = run_portfolio({sym: df_dict[sym]}, TRAIL_PERCOIN, True)
        nu = sum(t['pnl_net'] for t in t_u)*100 if t_u else 0
        np_ = sum(t['pnl_net'] for t in t_p)*100 if t_p else 0
        marker = "⭐" if np_ > nu+2 else "✅" if np_ > nu else "  "
        print(f"  {sym:<10} {nu:>+9.2f}% {np_:>+9.2f}% {np_-nu:>+7.2f}% {marker}")
