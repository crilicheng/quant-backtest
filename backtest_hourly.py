"""日内回测——用小时K线，先到TP还是SL，不偏不倚"""
import numpy as np, pandas as pd, yfinance as yf
import matplotlib.pyplot as plt, matplotlib.ticker as mticker
plt.rcParams["font.sans-serif"]=["Arial Unicode MS","Heiti SC","PingFang SC"]
plt.rcParams["axes.unicode_minus"]=False

RISK=0.03;SL=0.004;TP=0.015;TRAIL=0.004
RSI_P=5;RSI_L=25;RSI_S=78;MIN_VOL=1.2
MAX_POSITIONS=3;MAX_HOLD=72  # 72小时=3天
MAKER=0.0002;LIMIT_FILL=0.85;LIMIT_O=0.0005
CAPITAL=200;START="2025-01-01";END="2026-06-08"
COINS=["ETH-USD","SOL-USD","BNB-USD","AVAX-USD","DOGE-USD"]

def rsi(c,p):
    d=c.diff();g=d.clip(lower=0);l=(-d).clip(lower=0)
    ag=g.ewm(alpha=1/p,adjust=False).mean();al=l.ewm(alpha=1/p,adjust=False).mean()
    return float(100-100/(1+ag.iloc[-1]/al.iloc[-1])) if al.iloc[-1]!=0 else 50

np.random.seed(7)
print("加载小时K线...")
coin_data={}
for sym in COINS:
    try:
        df=yf.Ticker(sym).history(start=START,end=END,interval="1h")
        if len(df)>200:
            df=df.reset_index();df.columns=[c.lower() for c in df.columns]
            if "date" not in df.columns:
                df["date"]=df[[c for c in df.columns if "date" in c.lower() or "time" in c.lower()][0]]
            df["date"]=pd.to_datetime(df["date"]).dt.tz_localize(None)
            df=df.sort_values("date").reset_index(drop=True)
            df["rsi"]=df["close"].rolling(100).apply(lambda x:rsi(pd.Series(x.values),RSI_P))
            df["atr14"]=(df["high"]-df["low"]).rolling(14).mean()
            df["vol_ratio"]=df["volume"]/df["volume"].rolling(20).mean()
            coin_data[sym]=df;print(f"  {sym}: {len(df)}根小时K线")
    except Exception as e:print(f"  {sym}: 失败 {e}")

if len(coin_data)<3:print("数据不足");exit()

all_ts=sorted(set().union(*[set(d["date"]) for d in coin_data.values()]))
date_idx={sym:coin_data[sym].set_index("date") for sym in coin_data}

eq=CAPITAL;pos_list=[];trades=[];nav=[]

for i,ts in enumerate(all_ts):
    if i<200:nav.append({"date":ts,"equity":eq});continue

    # 平仓检查——用小时K线的高低点，先到谁就触发谁
    for pos in list(pos_list):
        sym=pos["symbol"]
        if ts not in date_idx[sym].index:continue
        row=date_idx[sym].loc[ts];price=float(row["close"])
        ph,pl=float(row["high"]),float(row["low"])
        pos["bars"]+=1

        exit_p,reason=None,""
        hit_tp=False;hit_sl=False

        if pos["dir"]=="long":
            if ph>=pos["tp"]:hit_tp=True
            if pl<=pos["sl"]:hit_sl=True
            if hit_tp and hit_sl:
                # 用开盘价判断哪个更近
                po=float(row.get("open",price))
                tp_dist=pos["tp"]-po;sl_dist=po-pos["sl"]
                if tp_dist<sl_dist:hit_sl=False  # TP更近,先到TP
                else:hit_tp=False  # SL更近,先到SL
            if hit_tp:exit_p=pos["tp"];reason="止盈"
            elif hit_sl:exit_p=pos["sl"];reason="止损"
            elif pos["bars"]>=MAX_HOLD:exit_p=price;reason="超时"
            elif "rsi" in row and not pd.isna(row["rsi"]) and row["rsi"]>70:exit_p=price;reason="RSI"
        else:
            # 做空：TP在下方，SL在上方
            if pl<=pos["tp"]:hit_tp=True
            if ph>=pos["sl"]:hit_sl=True
            if hit_tp and hit_sl:
                po=float(row.get("open",price))
                tp_dist=po-pos["tp"];sl_dist=pos["sl"]-po
                if tp_dist<sl_dist:hit_sl=False
                else:hit_tp=False
            if hit_tp:exit_p=pos["tp"];reason="止盈"
            elif hit_sl:exit_p=pos["sl"];reason="止损"
            elif pos["bars"]>=MAX_HOLD:exit_p=price;reason="超时"
            elif "rsi" in row and not pd.isna(row["rsi"]) and row["rsi"]<35:exit_p=price;reason="RSI"

        if exit_p:
            pnl=((exit_p-pos["entry_price"])/pos["entry_price"] if pos["dir"]=="long" else (pos["entry_price"]-exit_p)/pos["entry_price"])
            pnl_d=pos["notional"]*pnl-pos["notional"]*MAKER*2
            eq+=pnl_d
            trades.append({"date":ts,"pnl":pnl_d,"reason":reason,"sym":sym,"dir":pos["dir"]})
            pos_list.remove(pos)

    # 扫描新信号（只用1/4的频率，节省计算）
    if i%4==0 and len(pos_list)<MAX_POSITIONS and eq>20:
        for sym in COINS:
            if len(pos_list)>=MAX_POSITIONS:break
            if any(p["symbol"]==sym for p in pos_list):continue
            if ts not in date_idx[sym].index:continue
            row=date_idx[sym].loc[ts]
            price=float(row["close"]);ph=float(row["high"]);pl=float(row["low"])
            if pd.isna(row.get("rsi",np.nan)) or pd.isna(row.get("vol_ratio",np.nan)):continue
            rsi_v=float(row["rsi"]);vr=float(row["vol_ratio"])
            direction=None
            if rsi_v<RSI_L and vr>MIN_VOL and price>pl*1.003 and np.random.random()<LIMIT_FILL:direction="long"
            elif rsi_v>RSI_S and vr>MIN_VOL and price<ph*0.997 and np.random.random()<LIMIT_FILL:direction="short"
            if direction:
                risk=eq*RISK;notional=risk/SL
                entry_p=price*(1-LIMIT_O if direction=="long" else 1+LIMIT_O)
                tp_p=price*(1+TP if direction=="long" else 1-TP)
                sl_p=price*(1-SL if direction=="long" else 1+SL)
                pos_list.append({"symbol":sym,"dir":direction,"entry_price":entry_p,"notional":notional,
                                "bars":0,"tp":tp_p,"sl":sl_p})

    total=eq
    for p in pos_list:
        if ts in date_idx[p["symbol"]].index:
            cur=float(date_idx[p["symbol"]].loc[ts]["close"])
            unreal=((cur-p["entry_price"])/p["entry_price"] if p["dir"]=="long" else (p["entry_price"]-cur)/p["entry_price"])
            total+=p["notional"]*unreal
    nav.append({"date":ts,"equity":total})

# 绩效
nav_df=pd.DataFrame(nav).set_index("date")
final=nav_df["equity"].iloc[-1]
days=len(nav_df);ann=(final/CAPITAL)**(365*24/days)-1 if final>0 else 0
ret=nav_df["equity"].pct_change().dropna()
vol=float(ret.std()*np.sqrt(365*24)) if len(ret)>0 else 0
sharpe=(ann-0.03)/vol if vol>0 else 0
peak=nav_df["equity"].expanding().max()
mdd=float(((nav_df["equity"]-peak)/peak).min()) if peak.max()>0 else 0
wins=sum(1 for t in trades if t["pnl"]>0)
wr=wins/len(trades) if trades else 0
wins_sl=sum(1 for t in trades if t["reason"]=="止盈")
loss_sl=sum(1 for t in trades if t["reason"]=="止损")

print(f"\n{'='*60}")
print(f"  日内回测（小时K线，先到先触发）")
print(f"  $200 → ${final:,.0f} | 年化{ann*100:.1f}%")
print(f"  夏普{sharpe:.2f} | 回撤{mdd*100:.1f}% | {len(trades)}笔")
print(f"  止盈{wins_sl}次 止损{loss_sl}次 | 胜率{wr*100:.0f}%")

# 画图
fig,axes=plt.subplots(2,1,figsize=(14,7),gridspec_kw={"height_ratios":[3,1]})
ax1=axes[0]
ax1.plot(nav_df.index,nav_df["equity"]/CAPITAL,label=f"策略 ${final:,.0f}",color="#1f77b4",linewidth=1)
btc=yf.Ticker("BTC-USD").history(start=START,end=END)
btc_init=btc["Close"].iloc[0]
ax1.plot(btc.index,btc["Close"]/btc_init,label=f"BTC",color="#ff7f0e",linewidth=1,alpha=0.5)
ax1.axhline(y=1,color="gray",linestyle="--",alpha=0.5)
ax1.set_title(f"日内回测 2025-01~2026-06 | $200→${final:,.0f} | 年化{ann*100:.0f}% | {len(trades)}笔 胜率{wr*100:.0f}%",fontsize=13,fontweight="bold")
ax1.set_ylabel("净值");ax1.legend(loc="upper left");ax1.grid(True,alpha=0.3)
ax2=axes[1]
dd=(nav_df["equity"]-nav_df["equity"].expanding().max())/nav_df["equity"].expanding().max()
ax2.fill_between(dd.index,dd.values,0,color="red",alpha=0.3)
ax2.set_ylabel("回撤");ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0));ax2.grid(True,alpha=0.3)
plt.tight_layout();plt.savefig("assets/backtest_hourly.png",dpi=150,bbox_inches="tight")
print("图: assets/backtest_hourly.png")
