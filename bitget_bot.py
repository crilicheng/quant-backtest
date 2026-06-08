"""
Bitget 实盘机器人 —— 原生 HTTP 版

用法:
    python bitget_bot.py --dry-run    # 模拟
    python bitget_bot.py --live       # 实盘
"""

import os, time, json, hmac, base64, hashlib, argparse, logging
import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("bot")

# ==== 策略参数 ====
LEVERAGE = 10
RISK = 0.03
TP = 0.015
SL = 0.004
TRAIL = 0.004
RSI_P = 5
RSI_L = 25
RSI_S = 78
MIN_VOL = 1.2
MAX_HOLD_BARS = 3
MAX_POSITIONS = 3
COINS = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT", "DOGEUSDT"]
SCAN_INTERVAL = 300
MAX_DAILY_LOSS = 0.15

# ==== Bitget HTTP 客户端 ====
class Bitget:
    BASE = "https://api.bitget.com"

    def __init__(self, dry=True):
        self.dry = dry
        if not dry:
            self.key = os.getenv("BITGET_KEY")
            self.secret = os.getenv("BITGET_SECRET")
            self.pw = os.getenv("BITGET_PASSPHRASE")
            logger.info(f"Bitget 实盘模式")

    def _sign(self, method, path, body=""):
        ts = str(int(time.time() * 1000))
        prehash = ts + method + path + body
        sig = base64.b64encode(hmac.new(self.secret.encode(), prehash.encode(), hashlib.sha256).digest()).decode()
        h = {"ACCESS-KEY": self.key, "ACCESS-SIGN": sig, "ACCESS-TIMESTAMP": ts,
             "ACCESS-PASSPHRASE": self.pw, "Content-Type": "application/json"}
        return h

    def _get(self, path):
        h = self._sign("GET", path)
        r = __import__("requests").get(self.BASE + path, headers=h, timeout=15)
        return r.json()

    def _post(self, path, body):
        b = json.dumps(body)
        h = self._sign("POST", path, b)
        r = __import__("requests").post(self.BASE + path, headers=h, data=b, timeout=15)
        return r.json()

    def balance(self):
        if self.dry: return {"available": 0, "equity": 0}
        d = self._get("/api/v2/mix/account/accounts?productType=USDT-FUTURES")
        if d.get("code") == "00000":
            a = d["data"][0]
            return {"available": float(a.get("available", 0)),
                    "equity": float(a.get("equity", 0))}
        return {"available": 0, "equity": 0}

    def candles(self, symbol, bar="1H", limit=100):
        if self.dry:
            return self._mock_candles(symbol)
        d = self._get(f"/api/v2/mix/market/candles?symbol={symbol}&productType=USDT-FUTURES&granularity={bar}&limit={limit}")
        if d.get("code") == "00000":
            rows = []
            for r in d["data"]:
                # [ts, open, high, low, close, baseVol, quoteVol]
                rows.append({"ts": int(r[0]), "open": float(r[1]), "high": float(r[2]),
                             "low": float(r[3]), "close": float(r[4]), "vol": float(r[5])})
            return pd.DataFrame(rows)
        return pd.DataFrame()

    def _mock_candles(self, symbol):
        try:
            import yfinance as yf
            sym = symbol.replace("USDT", "-USD")
            df = yf.Ticker(sym).history(period="5d", interval="15m")
            df = df.reset_index(); df.columns = [c.lower() for c in df.columns]
            df["ts"] = df["date"].astype(int)//10**9
            return df.rename(columns={"date":"ts_d","volume":"vol"})
        except:
            n=100; p=100+np.cumsum(np.random.normal(0,1,n))
            return pd.DataFrame({"ts":range(n),"open":p,"high":p*1.01,"low":p*0.99,"close":p,"vol":np.random.lognormal(10,1,n)})

    def positions(self):
        if self.dry: return []
        d = self._get("/api/v2/mix/position/all-position?productType=USDT-FUTURES&marginCoin=USDT")
        if d.get("code")=="00000":
            return [p for p in d["data"] if float(p.get("total","0"))>0]
        return []

    def set_leverage(self, symbol, lev=LEVERAGE):
        if self.dry: return
        self._post("/api/v2/mix/account/set-leverage",
                   {"symbol":symbol,"marginCoin":"USDT","leverage":str(lev)})

    def market_order(self, symbol, side, size, tp=None, sl_price=None):
        """side: 'buy'开多 'sell'开空"""
        body = {"symbol":symbol,"marginCoin":"USDT","size":str(size),
                "side":side,"orderType":"market"}
        if tp: body["presetTakeProfitPrice"] = str(tp)
        if sl_price: body["presetStopLossPrice"] = str(sl_price)
        if self.dry:
            logger.info(f"  [模拟] {side} {size} @ {symbol}")
            return {"code":"00000"}
        return self._post("/api/v2/mix/order/place-order", body)

    def close_position(self, symbol, side, size):
        """平仓"""
        close_side = "sell" if side == "long" else "buy"
        body = {"symbol":symbol,"marginCoin":"USDT","side":close_side,
                "orderType":"market","size":str(size),"reduceOnly":"YES"}
        if self.dry:
            logger.info(f"  [模拟] 平仓 {symbol} {side}")
            return {"code":"00000"}
        return self._post("/api/v2/mix/order/place-order", body)


# ==== 策略逻辑 ====
def compute_rsi(close, period=RSI_P):
    d = close.diff().dropna()
    if len(d)<period: return 50.0
    g=d.clip(lower=0); l=(-d).clip(lower=0)
    ag=g.ewm(alpha=1/period,adjust=False).mean()
    al=l.ewm(alpha=1/period,adjust=False).mean()
    if al.iloc[-1]==0: return 100.0 if ag.iloc[-1]>0 else 50.0
    return float(100-100/(1+ag.iloc[-1]/al.iloc[-1]))

def check_signal(api, symbol):
    df = api.candles(symbol)
    if len(df)<50: return None
    c,h,l,v = df["close"],df["high"],df["low"],df["vol"]
    price = float(c.iloc[-1])
    rsi = compute_rsi(c)
    vm = v.rolling(20).mean()
    vr = float(v.iloc[-1]/vm.iloc[-1]) if vm.iloc[-1]>0 else 1
    if pd.isna(rsi): return None
    if rsi<RSI_L and vr>MIN_VOL and price>float(l.iloc[-1])*1.003:
        logger.info(f"📈 {symbol} LONG  RSI={rsi:.0f} Vol={vr:.1f}x Price=${price:.2f}")
        return {"dir":"long","price":price,"rsi":rsi}
    if rsi>RSI_S and vr>MIN_VOL and price<float(h.iloc[-1])*0.997:
        logger.info(f"📉 {symbol} SHORT RSI={rsi:.0f} Vol={vr:.1f}x Price=${price:.2f}")
        return {"dir":"short","price":price,"rsi":rsi}
    return None


# ==== 主循环 ====
def run(dry=True):
    api = Bitget(dry=dry)
    bal = api.balance()
    if not dry and bal["available"] < 50:
        logger.warning(f"余额 ${bal['available']:.0f} 不足，建议至少 $50")
    logger.info(f"余额: ${bal['available']:.0f} | 持仓: {len(api.positions())} | 扫描间隔: {SCAN_INTERVAL}s")
    count = 0

    while True:
        try:
            pos = api.positions()
            bal = api.balance()

            # 扫描信号
            if len(pos) < MAX_POSITIONS:
                for coin in COINS:
                    if len(pos) >= MAX_POSITIONS: break
                    if any(p.get("symbol")==coin for p in pos): continue
                    sig = check_signal(api, coin)
                    if sig:
                        price = sig["price"]
                        size = round(bal["available"]*RISK/(SL*LEVERAGE)/price, 4)
                        if size > 0:
                            api.set_leverage(coin)
                            tp_price = price*(1+TP if sig["dir"]=="long" else 1-TP)
                            sl_price = price*(1-SL if sig["dir"]=="long" else 1+SL)
                            side = "buy" if sig["dir"]=="long" else "sell"
                            r = api.market_order(coin, side, size, tp=tp_price, sl_price=sl_price)
                            if r.get("code")=="00000":
                                logger.info(f"✅ 已开仓 {coin} {sig['dir']} size={size}")
                            else:
                                logger.error(f"❌ 开仓失败: {r}")
                        time.sleep(1)

            count+=1
            if count%6==0:
                b=api.balance()
                logger.info(f"💼 权益 ${b['equity']:.0f} | 持仓 {len(pos)} | 扫描#{count}")
            time.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            logger.info("用户停止"); break
        except Exception as e:
            logger.error(f"异常: {e}"); time.sleep(SCAN_INTERVAL)

    logger.info(f"机器人已停")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dry-run",action="store_true",default=True)
    p.add_argument("--live",action="store_true")
    a=p.parse_args()
    dry=not a.live
    if not dry:
        for k in ["BITGET_KEY","BITGET_SECRET","BITGET_PASSPHRASE"]:
            if not os.getenv(k):
                logger.error(f"请设置环境变量 {k}"); exit(1)
        print("\n⚠️ 实盘模式！输入 YES 继续:")
        if input()!="YES": print("取消"); exit()
    run(dry=dry)
