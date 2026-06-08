"""分析 ETH 超短线策略的三次插针"""
import numpy as np
import pandas as pd
import yfinance as yf

LEVERAGE = 10
RISK = 0.05
SL = 0.004
TP = 0.015
RSI_P = 5
RSI_L = 25
RSI_S = 78
MAKER = 0.0002
LIMIT_O = 0.0005
LIMIT_F = 0.85
TRAIL = 0.004
MAX_H = 3
WICK = 0.08

def rsi(c, p):
    d = c.diff()
    g = d.clip(lower=0)
    l = (-d).clip(lower=0)
    return 100 - 100 / (1 + g.ewm(alpha=1/p, adjust=False).mean() /
                         l.ewm(alpha=1/p, adjust=False).mean().replace(0, 1e-10))

np.random.seed(42)

df = yf.Ticker('ETH-USD').history(start='2023-01-01', end='2026-06-08')
df = df.reset_index()
df.columns = [c.lower() for c in df.columns]
df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
df['rsi'] = rsi(df['close'], RSI_P)
df['atr14'] = (df['high'] - df['low']).rolling(14).mean()
df['vol_ma20'] = df['volume'].rolling(20).mean()
df['vol_ratio'] = df['volume'] / df['vol_ma20']

eq = 10000.0
pos_val = 0.0
entry_p = 0.0
bars = 0
in_pos = False
direction = ''
highest = 0.0
lowest = 0.0
entry_fee_rate = MAKER
cnt = 0
wick_trades = []
all_trades = []

for i, row in df.iterrows():
    if i < 30:
        continue
    d = row['date']
    p = row['close']
    ph = row['high']
    pl = row['low']
    rsi_v = row['rsi']
    vol_r = row['vol_ratio']

    if in_pos:
        bars += 1
        highest = max(highest, ph)
        lowest = min(lowest, pl)
        pc = ((p - entry_p) / entry_p) if direction == 'long' else ((entry_p - p) / entry_p)

        atr_p = row['atr14'] / p
        dr = (ph - pl) / p
        is_wick = dr > 1.5 * atr_p and np.random.random() < WICK

        exit_p = None
        reason = ''

        if is_wick:
            wd = np.random.uniform(2, 5)
            if direction == 'long' and pl < entry_p * (1 - SL * wd):
                exit_p = entry_p * (1 - SL * wd)
                reason = '插针'
            elif direction == 'short' and ph > entry_p * (1 + SL * wd):
                exit_p = entry_p * (1 + SL * wd)
                reason = '插针'

        if not exit_p:
            if pc >= TP:
                exit_p = entry_p * (1 + TP if direction == 'long' else 1 - TP)
                reason = '止盈'
            elif pc <= -SL:
                exit_p = entry_p * (1 - SL if direction == 'long' else 1 + SL)
                reason = '止损'
            elif highest > entry_p and direction == 'long':
                dd = (highest - p) / entry_p
                if dd > TRAIL and pc > 0:
                    exit_p = p
                    reason = '移动止盈'
            elif lowest < entry_p and direction == 'short':
                rally = (p - lowest) / entry_p
                if rally > TRAIL and pc > 0:
                    exit_p = p
                    reason = '移动止盈'
            elif bars >= MAX_H:
                exit_p = p
                reason = '超时'
            elif direction == 'long' and not pd.isna(rsi_v) and rsi_v > 70:
                exit_p = p
                reason = 'RSI回落'
            elif direction == 'short' and not pd.isna(rsi_v) and rsi_v < 35:
                exit_p = p
                reason = 'RSI回升'

        if exit_p is not None:
            ef = exit_p * (1 + LIMIT_O if direction == 'long' else 1 - LIMIT_O)
            if direction == 'long':
                pnl = (ef / entry_p - 1) * LEVERAGE
            else:
                pnl = (1 - ef / entry_p) * LEVERAGE
            eq += pos_val * pnl - pos_val * (MAKER + entry_fee_rate)
            cnt += 1

            trade = {
                'cnt': cnt, 'dir': direction, 'entry_date': entry_date, 'exit_date': d,
                'entry_price': entry_p, 'exit_price': ef,
                'pnl_pct': pnl * 100, 'reason': reason, 'bars': bars,
                'daily_range': (ph - pl) / p * 100,
                'atr_pct': row['atr14'] / p * 100,
                'rsi_at_entry': entry_rsi,
                'price_low': pl, 'price_high': ph,
            }
            all_trades.append(trade)
            if reason == '插针':
                wick_trades.append(trade)
            pos_val = 0
            in_pos = False

    if not in_pos and not pd.isna(rsi_v):
        direction = None
        ls = rsi_v < RSI_L and vol_r > 1.2 and p > pl * 1.003
        ss = rsi_v > RSI_S and vol_r > 1.2 and p < ph * 0.997
        if ls:
            direction = 'long'
        elif ss:
            direction = 'short'
        if direction and np.random.random() < LIMIT_F:
            pos_val = min(eq * RISK / (SL * LEVERAGE), eq * LEVERAGE)
            entry_p = p * (1 - LIMIT_O if direction == 'long' else 1 + LIMIT_O)
            entry_fee_rate = MAKER
            highest = entry_p
            lowest = entry_p
            bars = 0
            entry_date = d
            entry_rsi = rsi_v
            in_pos = True

print(f'总交易: {cnt} 笔, 其中插针: {len(wick_trades)} 次\n')

for j, w in enumerate(wick_trades):
    arrow = "📈 做多" if w['dir'] == 'long' else "📉 做空"
    normal_sl_loss = -SL * LEVERAGE * 100
    print("=" * 60)
    print(f"  ⚡ 插针 #{j+1}")
    print("=" * 60)
    print(f"  方向:         {arrow}")
    print(f"  入场时间:     {str(w['entry_date'])[:10]}")
    print(f"  插针时间:     {str(w['exit_date'])[:10]}")
    print(f"  持仓:         {w['bars']} 天")
    print(f"  入场价:       ${w['entry_price']:,.2f}")
    print(f"  被针到:       ${w['exit_price']:,.2f}")
    print(f"  实际亏损:     {w['pnl_pct']:.1f}%")
    print(f"  正常止损应亏:  {normal_sl_loss:.1f}%")
    print(f"  多亏了:       {abs(w['pnl_pct']) - abs(normal_sl_loss):.1f}%")
    print(f"  ──────────────────────────────")
    print(f"  当日振幅:     {w['daily_range']:.1f}%  (正常 ATR: {w['atr_pct']:.1f}%)")
    print(f"  入场时 RSI:   {w['rsi_at_entry']:.1f}")
    print(f"  当日最低:     ${w['price_low']:,.2f}")
    print(f"  当日最高:     ${w['price_high']:,.2f}")
    print()
    print(f"  原因分析:")
    if w['dir'] == 'long':
        sl_price = w['entry_price'] * (1 - SL)
        print(f"    预设止损价: ${sl_price:,.2f}")
        print(f"    实际最低:   ${w['price_low']:,.2f}")
        print(f"    价格直接穿透止损，在更低的位置才成交")
    else:
        sl_price = w['entry_price'] * (1 + SL)
        print(f"    预设止损价: ${sl_price:,.2f}")
        print(f"    实际最高:   ${w['price_high']:,.2f}")
        print(f"    价格直接突破止损，在更高位置才成交")
    print()
