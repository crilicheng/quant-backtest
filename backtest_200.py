"""回测：$200本金 2025-01-01 至今 当前实盘策略参数"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import time

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti SC", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

LEVERAGE_MAP = {"ETH-USD":100,"SOL-USD":100,"BNB-USD":75,"AVAX-USD":75,"DOGE-USD":75}
RISK = 0.03; SL = 0.004; TP = 0.015; TRAIL = 0.004
RSI_P = 5; RSI_L = 25; RSI_S = 78; MIN_VOL = 1.2
MAX_POSITIONS = 3; MAX_HOLD = 3; WICK = 0.08
MAKER = 0.0002; LIMIT_FILL = 0.85; LIMIT_O = 0.0005
CAPITAL = 200; START = "2025-01-01"; END = "2026-06-08"
COINS = ["ETH-USD","SOL-USD","BNB-USD","AVAX-USD","DOGE-USD"]

def compute_rsi_series(close, period):
    d = close.diff()
    g = d.clip(lower=0); l = (-d).clip(lower=0)
    ag = g.ewm(alpha=1/period, adjust=False).mean()
    al = l.ewm(alpha=1/period, adjust=False).mean()
    return 100 - 100 / (1 + ag / al.replace(0, 1e-10))

np.random.seed(7)

print("加载数据...")
coin_data = {}
all_dates_set = set()
for sym in COINS:
    df = yf.Ticker(sym).history(start=START, end=END)
    if len(df) > 100:
        df = df.reset_index(); df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.sort_values("date").reset_index(drop=True)
        df["rsi"] = compute_rsi_series(df["close"], RSI_P)
        df["atr14"] = (df["high"]-df["low"]).rolling(14).mean()
        df["vol_ma20"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)
        coin_data[sym] = df
        all_dates_set.update(df["date"].tolist())
        print(f"  {sym}: {len(df)}天")

all_dates = sorted(all_dates_set)
date_idx = {sym: coin_data[sym].set_index("date") for sym in coin_data}

eq = CAPITAL; pos_list = []; trades = []; nav = []
for i, date in enumerate(all_dates):
    if i < 50:  # 跳过头50天让指标稳定
        nav.append({"date":date,"equity":eq}); continue

    # 检查平仓
    for pos in list(pos_list):
        sym = pos["symbol"]
        if date not in date_idx[sym].index: continue
        row = date_idx[sym].loc[date]
        price,ph,pl,rsi_v,atr14 = float(row["close"]),float(row["high"]),float(row["low"]),float(row["rsi"]),float(row["atr14"])
        pos["bars"]+=1; pos["highest"]=max(pos["highest"],ph); pos["lowest"]=min(pos["lowest"],pl)
        pc = ((price-pos["entry_price"])/pos["entry_price"]) if pos["dir"]=="long" else ((pos["entry_price"]-price)/pos["entry_price"])
        
        exit_p,reason=None,""
        atr_p=atr14/price if atr14>0 else 0.01
        dr=(ph-pl)/price
        if dr>1.5*atr_p and np.random.random()<WICK:
            wd=np.random.uniform(2,5)
            if pos["dir"]=="long" and pl<pos["entry_price"]*(1-SL*wd):
                exit_p=pos["entry_price"]*(1-SL*wd);reason="插针"
            elif pos["dir"]=="short" and ph>pos["entry_price"]*(1+SL*wd):
                exit_p=pos["entry_price"]*(1+SL*wd);reason="插针"
        if not exit_p:
            if pc>=TP: exit_p=pos["entry_price"]*(1+TP if pos["dir"]=="long" else 1-TP);reason="止盈"
            elif pc<=-SL: exit_p=pos["entry_price"]*(1-SL if pos["dir"]=="long" else 1+SL);reason="止损"
            elif pos["dir"]=="long" and pos["highest"]>pos["entry_price"]:
                dd=(pos["highest"]-price)/pos["entry_price"]
                if dd>TRAIL and pc>0: exit_p=price;reason="移动止盈"
            elif pos["dir"]=="short" and pos["lowest"]<pos["entry_price"]:
                rally=(price-pos["lowest"])/pos["entry_price"]
                if rally>TRAIL and pc>0: exit_p=price;reason="移动止盈"
            elif pos["bars"]>=MAX_HOLD: exit_p=price;reason="超时"
            elif pos["dir"]=="long" and rsi_v>70: exit_p=price;reason="RSI回落"
            elif pos["dir"]=="short" and rsi_v<35: exit_p=price;reason="RSI回升"
        
        if exit_p:
            ef=exit_p*(1+LIMIT_O if pos["dir"]=="long" else 1-LIMIT_O)
            pnl = ((ef-pos["entry_price"])/pos["entry_price"] if pos["dir"]=="long" else (pos["entry_price"]-ef)/pos["entry_price"])
            pnl_d = pos["notional"]*pnl - pos["notional"]*MAKER*2
            eq+=pnl_d
            trades.append({"sym":sym,"dir":pos["dir"],"pnl":pnl_d,"reason":reason,"date":date})
            pos_list.remove(pos)

    # 扫描新信号
    if len(pos_list) < MAX_POSITIONS and eq > 20:
        for sym in COINS:
            if len(pos_list) >= MAX_POSITIONS: break
            if any(p["symbol"]==sym for p in pos_list): continue
            if date not in date_idx[sym].index: continue
            row = date_idx[sym].loc[date]
            price,ph,pl = float(row["close"]),float(row["high"]),float(row["low"])
            rsi_v, vr = float(row["rsi"]), float(row["vol_ratio"])
            if pd.isna(rsi_v) or pd.isna(vr): continue
            
            direction = None
            if rsi_v<RSI_L and vr>MIN_VOL and price>pl*1.003 and np.random.random()<LIMIT_FILL:
                direction="long"
            elif rsi_v>RSI_S and vr>MIN_VOL and price<ph*0.997 and np.random.random()<LIMIT_FILL:
                direction="short"
            
            if direction:
                risk=eq*RISK; notional=risk/SL
                entry_p=price*(1-LIMIT_O if direction=="long" else 1+LIMIT_O)
                pos_list.append({"symbol":sym,"dir":direction,"entry_price":entry_p,
                                "notional":notional,"bars":0,"highest":entry_p,"lowest":entry_p,"entry_date":date})

    # 记录权益
    total=eq
    for p in pos_list:
        sym=p["symbol"]
        if date in date_idx[sym].index:
            row=date_idx[sym].loc[date]
            cur=float(row["close"])
            unreal = ((cur-p["entry_price"])/p["entry_price"] if p["dir"]=="long" else (p["entry_price"]-cur)/p["entry_price"])
            total+=p["notional"]*unreal
    nav.append({"date":date,"equity":total})

# 绩效
nav_df=pd.DataFrame(nav).set_index("date")
final=nav_df["equity"].iloc[-1]
days=len(nav_df)
ann=(final/CAPITAL)**(365/days)-1 if final>0 else -1
ret=nav_df["equity"].pct_change().dropna()
vol=float(ret.std()*np.sqrt(365)) if len(ret)>0 else 0
sharpe=(ann-0.03)/vol if vol>0 else 0
peak=nav_df["equity"].expanding().max()
mdd=float(((nav_df["equity"]-peak)/peak).min()) if peak.iloc[-1]>0 else 0
wins=sum(1 for t in trades if t["pnl"]>0)
wr=wins/len(trades) if trades else 0
wicks=sum(1 for t in trades if t["reason"]=="插针")

# 图
btc=yf.Ticker("BTC-USD").history(start=START,end=END)
btc=btc.reset_index(); btc.columns=[c.lower() for c in btc.columns]
btc["date"]=pd.to_datetime(btc["date"]).dt.tz_localize(None)
btc=btc.set_index("date")
btc_init=btc["close"].iloc[0]

fig,axes=plt.subplots(2,1,figsize=(14,7),gridspec_kw={"height_ratios":[3,1]})
ax1=axes[0]
ax1.plot(nav_df.index,nav_df["equity"]/CAPITAL,label=f"策略 ${final:,.0f}",color="#1f77b4",linewidth=1.5)
ax1.plot(btc.index,btc["close"]/btc_init,label=f"BTC ${btc['close'].iloc[-1]:,.0f}",color="#ff7f0e",linewidth=1,alpha=0.7)
ax1.axhline(y=1,color="gray",linestyle="--",alpha=0.5)
ax1.set_title(f"回测 2025-01~2026-06 | $200 → ${final:,.0f} | 年化{ann*100:.1f}% | {len(trades)}笔",fontsize=14,fontweight="bold")
ax1.set_ylabel("净值");ax1.legend(loc="upper left");ax1.grid(True,alpha=0.3)

ax2=axes[1]
dd=(nav_df["equity"]-nav_df["equity"].expanding().max())/nav_df["equity"].expanding().max()
ax2.fill_between(dd.index,dd.values,0,color="red",alpha=0.3)
ax2.set_ylabel("回撤");ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0));ax2.grid(True,alpha=0.3)

plt.tight_layout();plt.savefig("assets/backtest_200.png",dpi=150,bbox_inches="tight")
print(f"\n$200 → ${final:,.0f} | 年化{ann*100:.1f}% | 夏普{sharpe:.2f} | 回撤{mdd*100:.1f}% | {len(trades)}笔 | 胜率{wr*100:.0f}% | 插针{wicks}次")
print("图: assets/backtest_200.png")
