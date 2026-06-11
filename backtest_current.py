"""回测当前实盘策略——小时K线，预计算所有指标"""
import numpy as np, pandas as pd, yfinance as yf
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
plt.rcParams["font.sans-serif"]=["Arial Unicode MS","Heiti SC","PingFang SC"]
plt.rcParams["axes.unicode_minus"]=False

RISK=0.03;SL=0.004;TP=0.015;RSI_P=5;RSI_L=25;RSI_S=78;MIN_VOL=1.2
MAX_POSITIONS=3;MAX_HOLD=72;MAKER=0.0002
CAPITAL=200;START="2025-01-01";END="2026-06-08"
COINS=["ETH-USD","SOL-USD","BNB-USD","AVAX-USD","DOGE-USD"]
np.random.seed(7)

def precompute_rsi(close,period=5):
    d=close.diff();g=d.clip(lower=0);l=(-d).clip(lower=0)
    ag=g.ewm(alpha=1/period,adjust=False).mean()
    al=l.ewm(alpha=1/period,adjust=False).mean()
    return 100-100/(1+ag/al.replace(0,1e-10))

print("加载+预计算小时K线...")
coin_data={}
for sym in COINS:
    df=yf.Ticker(sym).history(start=START,end=END,interval="1h")
    if len(df)>200:
        df=df.reset_index();df.columns=[c.lower() for c in df.columns]
        dc=[c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        df["date"]=pd.to_datetime(df[dc[0]]).dt.tz_localize(None) if dc else pd.to_datetime(df.index)
        df=df.sort_values("date").reset_index(drop=True)
        df["rsi"]=precompute_rsi(df["close"],RSI_P)
        df["vol_ma"]=df["volume"].rolling(20).mean()
        df["vol_ratio"]=df["volume"]/df["vol_ma"].replace(0,np.nan)
        coin_data[sym]=df;print(f"  {sym}: {len(df)}根")

all_ts=sorted(set().union(*[set(d["date"]) for d in coin_data.values()]))
date_idx={sym:coin_data[sym].set_index("date") for sym in coin_data}

eq=CAPITAL;pos_list=[];nav=[(all_ts[0],eq)];trades=[]

for ts in all_ts:
    # 检查平仓
    for pos in list(pos_list):
        sym=pos["symbol"]
        if ts not in date_idx[sym].index:continue
        row=date_idx[sym].loc[ts];price=float(row["close"]);ph,pl=float(row["high"]),float(row["low"])
        pos["bars"]+=1
        exit_p,reason=None,""
        hit_tp=ph>=pos["tp"];hit_sl=pl<=pos["sl"]
        if pos["dir"]=="short":hit_tp=pl<=pos["tp"];hit_sl=ph>=pos["sl"]
        if hit_tp and hit_sl:
            po=float(row.get("open",price))
            if abs(pos["tp"]-po)<abs(pos["sl"]-po):hit_sl=False
            else:hit_tp=False
        if hit_tp:exit_p=pos["tp"];reason="止盈"
        elif hit_sl:exit_p=pos["sl"];reason="止损"
        elif pos["bars"]>=MAX_HOLD:exit_p=price;reason="超时"
        if exit_p:
            pnl=((exit_p-pos["entry"])/pos["entry"] if pos["dir"]=="long" else (pos["entry"]-exit_p)/pos["entry"])
            eq+=pos["notional"]*pnl-pos["notional"]*MAKER*2
            trades.append({"date":ts,"pnl":pos["notional"]*pnl,"reason":reason,"sym":sym,"dir":pos["dir"]})
            nav.append((ts,eq))
            pos_list.remove(pos)

    # 扫描信号
    if len(pos_list)<MAX_POSITIONS and eq>20:
        for sym in COINS:
            if len(pos_list)>=MAX_POSITIONS:break
            if any(p["symbol"]==sym for p in pos_list):continue
            if ts not in date_idx[sym].index:continue
            row=date_idx[sym].loc[ts];price=float(row["close"]);ph,pl=float(row["high"]),float(row["low"])
            rsi_v=row.get("rsi");vr=row.get("vol_ratio")
            if pd.isna(rsi_v) or pd.isna(vr):continue
            rsi_v=float(rsi_v);vr=float(vr)
            direction=None
            if rsi_v<RSI_L and vr>MIN_VOL and price>pl*1.001 and np.random.random()<0.85:direction="long"
            elif rsi_v>RSI_S and vr>MIN_VOL and price<ph*0.999 and np.random.random()<0.85:direction="short"
            if direction:
                risk=eq*RISK;notional=risk/SL
                entry_p=price*(1-0.0005 if direction=="long" else 1+0.0005)
                tp_p=price*(1+TP if direction=="long" else 1-TP)
                sl_p=price*(1-SL if direction=="long" else 1+SL)
                pos_list.append({"symbol":sym,"dir":direction,"entry":entry_p,"notional":notional,"bars":0,"tp":tp_p,"sl":sl_p})

    total=eq
    for p in pos_list:
        if ts in date_idx[p["symbol"]].index:
            cur=float(date_idx[p["symbol"]].loc[ts]["close"])
            unreal=((cur-p["entry"])/p["entry"] if p["dir"]=="long" else (p["entry"]-cur)/p["entry"])
            total+=p["notional"]*unreal
    nav.append((ts,total))

nav_df=pd.DataFrame(nav,columns=["date","equity"]).set_index("date")
final=nav_df["equity"].iloc[-1];days=len(nav_df)
ann=(final/CAPITAL)**(365*24/days)-1 if final>0 else -1
ret=nav_df["equity"].pct_change().dropna()
vol=float(ret.std()*np.sqrt(365*24)) if len(ret)>0 else 0
sharpe=(ann-0.03)/vol if vol>0 else 0
peak=nav_df["equity"].expanding().max();mdd=float(((nav_df["equity"]-peak)/peak).min())
wins=sum(1 for t in trades if t["pnl"]>0);wr=wins/len(trades) if trades else 0
tp_hits=sum(1 for t in trades if t["reason"]=="止盈");sl_hits=sum(1 for t in trades if t["reason"]=="止损")
avg_win=np.mean([t["pnl"] for t in trades if t["pnl"]>0]) if wins>0 else 0
avg_loss=np.mean([t["pnl"] for t in trades if t["pnl"]<=0]) if len(trades)-wins>0 else 0

print(f"\n{'='*60}")
print(f"  $200 -> ${final:,.0f} | 年化{ann*100:.1f}% | 夏普{sharpe:.2f} | 回撤{mdd*100:.1f}%")
print(f"  {len(trades)}笔 | 止盈{tp_hits} 止损{sl_hits} | 胜率{wr*100:.0f}%")
print(f"  avg盈利${avg_win:.0f} avg亏损${avg_loss:.0f}")

fig,axes=plt.subplots(2,1,figsize=(14,7),gridspec_kw={"height_ratios":[3,1]})
ax1=axes[0]
ax1.plot(nav_df.index,nav_df["equity"]/CAPITAL,label=f"策略 ${final:,.0f}",color="#1f77b4",linewidth=1)
btc=yf.Ticker("BTC-USD").history(start=START,end=END);btc_init=btc["Close"].iloc[0]
ax1.plot(btc.index,btc["Close"]/btc_init,label="BTC",color="#ff7f0e",linewidth=1,alpha=0.5)
ax1.axhline(y=1,color="gray",linestyle="--",alpha=0.5)
ax1.set_title(f"当前策略 小时K线 | $200->${final:,.0f} | {len(trades)}笔 胜率{wr*100:.0f}%",fontsize=13,fontweight="bold")
ax1.set_ylabel("净值");ax1.legend(loc="upper left");ax1.grid(True,alpha=0.3)
ax2=axes[1]
dd=(nav_df["equity"]-nav_df["equity"].expanding().max())/nav_df["equity"].expanding().max()
ax2.fill_between(dd.index,dd.values,0,color="red",alpha=0.3)
ax2.set_ylabel("回撤");ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0));ax2.grid(True,alpha=0.3)
plt.tight_layout();plt.savefig("assets/backtest_current.png",dpi=150,bbox_inches="tight")
print("图: assets/backtest_current.png")
