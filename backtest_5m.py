"""5分钟线回测——最近60天"""
import numpy as np, pandas as pd, yfinance as yf
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
plt.rcParams["font.sans-serif"]=["Arial Unicode MS","Heiti SC","PingFang SC"]
plt.rcParams["axes.unicode_minus"]=False

RISK=0.03;SL=0.004;TP=0.015;MAX_POSITIONS=3;MAX_HOLD=576  # 48小时
MAKER=0.0002;LIMIT_FILL=0.85;LIMIT_O=0.0005
CAPITAL=200;COINS=["ETH-USD","SOL-USD","BNB-USD","AVAX-USD","DOGE-USD"]

def rsi(c,p=5):
    d=c.diff();g=d.clip(lower=0);l=(-d).clip(lower=0)
    ag=g.ewm(alpha=1/p,adjust=False).mean();al=l.ewm(alpha=1/p,adjust=False).mean()
    return 100-100/(1+ag/al.replace(0,1e-10))

np.random.seed(7)
print("加载5分钟线(最近60天)...")
coin_data={}
for sym in COINS:
    df=yf.Ticker(sym).history(period="60d",interval="5m")
    if len(df)>100:
        df=df.reset_index();df.columns=[c.lower() for c in df.columns]
        # find date column
        date_col=[c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        if date_col:df["date"]=pd.to_datetime(df[date_col[0]]).dt.tz_localize(None)
        else:df["date"]=df.index
        df=df.sort_values("date").reset_index(drop=True)
        # 预计算指标
        df["rsi"]=rsi(df["close"])
        df["atr14"]=(df["high"]-df["low"]).rolling(14).mean()
        df["vol_ma"]=df["volume"].rolling(20).mean()
        df["vol_ratio"]=df["volume"]/df["vol_ma"].replace(0,np.nan)
        coin_data[sym]=df;print(f"  {sym}: {len(df)}根")
all_ts=sorted(set().union(*[set(d["date"]) for d in coin_data.values()]))
date_idx={sym:coin_data[sym].set_index("date") for sym in coin_data}

eq=CAPITAL;pos_list=[];trades=[]
for i,ts in enumerate(all_ts):
    if i<300:continue
    for pos in list(pos_list):
        sym=pos["symbol"]
        if ts not in date_idx[sym].index:continue
        row=date_idx[sym].loc[ts];price=float(row["close"])
        ph,pl=float(row["high"]),float(row["low"])
        pos["bars"]+=1
        exit_p,reason=None,""
        hit_tp=hit_sl=False
        if pos["dir"]=="long":
            if ph>=pos["tp"]:hit_tp=True
            if pl<=pos["sl"]:hit_sl=True
            if hit_tp and hit_sl:
                po=float(row.get("open",price))
                if pos["tp"]-po<po-pos["sl"]:hit_sl=False
                else:hit_tp=False
            if hit_tp:exit_p=pos["tp"];reason="止盈"
            elif hit_sl:exit_p=pos["sl"];reason="止损"
            elif pos["bars"]>=MAX_HOLD:exit_p=price;reason="超时"
        else:
            if pl<=pos["tp"]:hit_tp=True
            if ph>=pos["sl"]:hit_sl=True
            if hit_tp and hit_sl:
                po=float(row.get("open",price))
                if pos["tp"]-po<pos["sl"]-po:hit_sl=False
                else:hit_tp=False
            if hit_tp:exit_p=pos["tp"];reason="止盈"
            elif hit_sl:exit_p=pos["sl"];reason="止损"
            elif pos["bars"]>=MAX_HOLD:exit_p=price;reason="超时"

        if exit_p:
            pnl=((exit_p-pos["entry_price"])/pos["entry_price"] if pos["dir"]=="long" else (pos["entry_price"]-exit_p)/pos["entry_price"])
            pnl_d=pos["notional"]*pnl-pos["notional"]*MAKER*2
            eq+=pnl_d
            trades.append({"date":ts,"pnl":pnl_d,"reason":reason})
            pos_list.remove(pos)

    if i%12==0 and len(pos_list)<MAX_POSITIONS and eq>20:
        for sym in COINS:
            if len(pos_list)>=MAX_POSITIONS:break
            if any(p["symbol"]==sym for p in pos_list):continue
            if ts not in date_idx[sym].index:continue
            row=date_idx[sym].loc[ts]
            price=float(row["close"]);ph=float(row["high"]);pl=float(row["low"])
            if pd.isna(row.get("rsi")) or pd.isna(row.get("vol_ratio")):continue
            rsi_v=float(row["rsi"]);vr=float(row["vol_ratio"])
            direction=None
            if rsi_v<25 and vr>1.2 and price>pl*1.001 and np.random.random()<LIMIT_FILL:direction="long"
            elif rsi_v>78 and vr>1.2 and price<ph*0.999 and np.random.random()<LIMIT_FILL:direction="short"
            if direction:
                risk=eq*RISK;notional=risk/SL
                entry_p=price*(1-LIMIT_O if direction=="long" else 1+LIMIT_O)
                tp_p=price*(1+TP if direction=="long" else 1-TP)
                sl_p=price*(1-SL if direction=="long" else 1+SL)
                pos_list.append({"symbol":sym,"dir":direction,"entry_price":entry_p,"notional":notional,"bars":0,"tp":tp_p,"sl":sl_p})

total_eq=eq
for p in pos_list:
    if ts in date_idx[p["symbol"]].index:
        cur=float(date_idx[p["symbol"]].loc[ts]["close"])
        unreal=((cur-p["entry_price"])/p["entry_price"] if p["dir"]=="long" else (p["entry_price"]-cur)/p["entry_price"])
        total_eq+=p["notional"]*unreal

final=total_eq;days=len(all_ts)/288  # 288根5m线=1天
wins=sum(1 for t in trades if t["pnl"]>0)
wr=wins/len(trades) if trades else 0
wins_tp=sum(1 for t in trades if t["reason"]=="止盈")
wins_sl=sum(1 for t in trades if t["reason"]=="止损")
pnl_series=pd.Series([t["date"] for t in trades],[t["pnl"] for t in trades])

print(f"\n{'='*60}")
print(f"  5分钟线回测 最近60天")
print(f"  $200 → ${final:,.0f} | {len(trades)}笔")
print(f"  止盈{wins_tp}次 | 止损{wins_sl}次 | 胜率{wr*100:.0f}%")
print(f"  总盈亏 ${sum(t['pnl'] for t in trades):,.0f}")

# 净值重建
nav_pts=[(all_ts[0],CAPITAL)]
cur_eq=CAPITAL
for t in sorted(trades,key=lambda x:x["date"]):
    cur_eq+=t["pnl"];nav_pts.append((t["date"],cur_eq))
nav_df=pd.DataFrame(nav_pts,columns=["date","equity"]).set_index("date")

fig,axes=plt.subplots(2,1,figsize=(14,7),gridspec_kw={"height_ratios":[3,1]})
ax1=axes[0]
ax1.plot(nav_df.index,nav_df["equity"]/CAPITAL,label=f"策略 ${final:,.0f}",color="#1f77b4",linewidth=1.5,drawstyle='steps-post')
btc=yf.Ticker("BTC-USD").history(period="60d")
btc_init=btc["Close"].iloc[0]
ax1.plot(btc.index,btc["Close"]/btc_init,label=f"BTC",color="#ff7f0e",linewidth=1,alpha=0.5)
ax1.axhline(y=1,color="gray",linestyle="--",alpha=0.5)
ann=(final/CAPITAL)**(365/max(days,1))-1 if final>0 else -1
ax1.set_title(f"5分钟线 60天 | $200→${final:,.0f} | 年化{ann*100:.0f}% | {len(trades)}笔 胜率{wr*100:.0f}%",fontsize=13,fontweight="bold")
ax1.set_ylabel("净值");ax1.legend(loc="upper left");ax1.grid(True,alpha=0.3)

ax2=axes[1]
colors=["green" if t["pnl"]>0 else "red" for t in trades]
ax2.bar(range(len(trades)),[t["pnl"] for t in trades],color=colors,alpha=0.7)
ax2.axhline(y=0,color="gray",linewidth=0.5)
ax2.set_ylabel("每笔盈亏$");ax2.set_xlabel("交易序号")
ax2.set_title(f"每笔交易盈亏（止盈{wins_tp} 止损{wins_sl}）")
ax2.grid(True,alpha=0.3,axis='y')

plt.tight_layout();plt.savefig("backtest_5m.png",dpi=150,bbox_inches="tight")
print("图: backtest_5m.png")
