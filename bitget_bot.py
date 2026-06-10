"""
Bitget 实盘机器人

用法:
    python bitget_bot.py --dry-run
    python bitget_bot.py --live
"""

import os, time, json, hmac, base64, hashlib, argparse, logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("bot")

# ==== 策略参数 ====
LEVERAGE = 100
RISK = 0.03
TP = 0.015
SL = 0.004
RSI_P = 5; RSI_L = 25; RSI_S = 78; MIN_VOL = 1.2
MAX_POSITIONS = 3
COINS = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT", "DOGEUSDT"]
MAX_LEVERAGE = {"ETHUSDT":150,"SOLUSDT":100,"BNBUSDT":75,"AVAXUSDT":75,"DOGEUSDT":75}
PREC_MAP = {"BTCUSDT":1,"ETHUSDT":2,"SOLUSDT":3,"BNBUSDT":2,"AVAXUSDT":3,"DOGEUSDT":5}
SCAN_INTERVAL = 60
DAILY_LOSS_LIMIT = 0.20    # 日亏20%停机
CONSECUTIVE_FAILS_MAX = 3  # 连续开仓失败停机
CONSECUTIVE_ERRORS_MAX = 5 # 连续 API 错误停机

cooldown = {}  # {symbol: last_entry_ts} 防重复开仓

# ==== Bitget ====
class Bitget:
    BASE = "https://api.bitget.com"

    def __init__(self, dry=True):
        self.dry = dry
        if not dry:
            self.key = os.getenv("BITGET_KEY")
            self.secret = os.getenv("BITGET_SECRET")
            self.pw = os.getenv("BITGET_PASSPHRASE")

    def _sign(self, method, path, body=""):
        ts = str(int(time.time() * 1000))
        prehash = ts + method + path + body
        sig = base64.b64encode(hmac.new(self.secret.encode(), prehash.encode(), hashlib.sha256).digest()).decode()
        return {"ACCESS-KEY": self.key, "ACCESS-SIGN": sig, "ACCESS-TIMESTAMP": ts,
                "ACCESS-PASSPHRASE": self.pw, "Content-Type": "application/json"}

    def _get(self, path):
        r = __import__("requests").get(self.BASE + path, headers=self._sign("GET", path), timeout=15)
        return r.json()

    def _post(self, path, body):
        b = json.dumps(body)
        r = __import__("requests").post(self.BASE + path, headers=self._sign("POST", path, b), data=b, timeout=15)
        return r.json()

    def balance(self):
        if self.dry: return {"available": 0, "equity": 0}
        d = self._get("/api/v2/mix/account/accounts?productType=USDT-FUTURES")
        if d.get("code") == "00000":
            a = d["data"][0]
            return {"available": float(a.get("available", 0)), "equity": float(a.get("equity", 0))}
        return {"available": 0, "equity": 0}

    def candles(self, symbol, bar="1H", limit=100):
        if self.dry: return self._mock_candles(symbol)
        d = self._get(f"/api/v2/mix/market/candles?symbol={symbol}&productType=USDT-FUTURES&granularity={bar}&limit={limit}")
        if d.get("code") == "00000":
            rows = [{"ts": int(r[0]), "open": float(r[1]), "high": float(r[2]),
                     "low": float(r[3]), "close": float(r[4]), "vol": float(r[5])} for r in d["data"]]
            return pd.DataFrame(rows)
        return pd.DataFrame()

    def _mock_candles(self, symbol):
        try:
            import yfinance as yf
            df = yf.Ticker(symbol.replace("USDT", "-USD")).history(period="5d", interval="15m")
            df = df.reset_index(); df.columns = [c.lower() for c in df.columns]
            df["ts"] = df["date"].astype(int)//10**9
            return df.rename(columns={"date":"ts_d","volume":"vol"})
        except:
            n=100; p=100+np.cumsum(np.random.normal(0,1,n))
            return pd.DataFrame({"ts":range(n),"open":p,"high":p*1.01,"low":p*0.99,"close":p,"vol":np.random.lognormal(10,1,n)})

    def positions(self):
        if self.dry: return []
        d = self._get("/api/v2/mix/position/all-position?productType=USDT-FUTURES&marginCoin=USDT")
        if d.get("code") == "00000":
            return [p for p in d["data"] if float(p.get("total", 0)) > 0]
        return []

    def open_orders(self, symbol=None):
        """查挂单（限价单），防重复开仓"""
        if self.dry: return []
        path = "/api/v2/mix/order/orders-pending?productType=USDT-FUTURES"
        if symbol: path += f"&symbol={symbol}"
        d = self._get(path)
        if d.get("code") == "00000":
            data = d.get("data", {})
            lst = data.get("entrustedList") if isinstance(data, dict) else data
            return lst if isinstance(lst, list) else []
        return []

    def set_leverage(self, symbol, lev):
        if self.dry: return
        self._post("/api/v2/mix/account/set-position-mode", {"productType":"USDT-FUTURES","posMode":"one_way_mode"})
        self._post("/api/v2/mix/account/set-leverage",
                   {"symbol":symbol,"marginCoin":"USDT","leverage":str(lev),
                    "productType":"USDT-FUTURES","marginMode":"crossed"})

    def open_with_tpsl(self, symbol, side, size, limit_price, tp, sl_price):
        """限价单开仓 → 确认成交 → 挂TP/SL。失败则平仓返回False"""
        if self.dry:
            logger.info(f"  [模拟] {side} {size}@{limit_price} {symbol}")
            return True

        # 1. 下限价单 (v2 用 force 不是 timeInForceValue)
        body = {"symbol":symbol,"marginCoin":"USDT","size":str(size),
                "side":side,"orderType":"limit","price":str(limit_price),
                "productType":"USDT-FUTURES","marginMode":"crossed","force":"gtc"}
        r = self._post("/api/v2/mix/order/place-order", body)
        if r.get("code") != "00000":
            logger.error(f"限价单失败 {symbol}: {r.get('code')} {r.get('msg','')}")
            return False

        order_id = r.get("data", {}).get("orderId", "")
        if not order_id: logger.error("未获取订单ID"); return False

        # 2. 轮询成交 (GET 接口 + productType 查询参数)
        filled = False
        for _ in range(12):
            time.sleep(5)
            check = self._get(f"/api/v2/mix/order/detail?symbol={symbol}&productType=USDT-FUTURES&orderId={order_id}")
            if check.get("code") == "00000":
                if check.get("data", {}).get("state") == "filled":
                    filled = True; break

        # 3. 未成交 → 撤单 → 检查撤单结果 → 市价补
        if not filled:
            logger.info(f"{symbol} 限价单60秒未成交，撤单")
            cancel_r = self._post("/api/v2/mix/order/cancel-order",
                                  {"symbol":symbol,"marginCoin":"USDT","orderId":str(order_id),
                                   "productType":"USDT-FUTURES"})
            if cancel_r.get("code") != "00000":
                # 撤单失败 → 可能已成交 → 重新查一次
                time.sleep(2)
                check2 = self._get(f"/api/v2/mix/order/detail?symbol={symbol}&productType=USDT-FUTURES&orderId={order_id}")
                if check2.get("code") == "00000" and check2.get("data", {}).get("state") == "filled":
                    filled = True
                else:
                    logger.error(f"{symbol} 撤单失败且未成交，跳过")
                    return False

        if not filled:
            # 市价补单
            time.sleep(1)
            del body["price"]; del body["force"]
            body["orderType"] = "market"
            r = self._post("/api/v2/mix/order/place-order", body)
            if r.get("code") != "00000":
                logger.error(f"{symbol} 市价单失败")
                return False
            time.sleep(2)

        # 4. 确认持仓
        time.sleep(3)
        pos_r = self._get("/api/v2/mix/position/all-position?productType=USDT-FUTURES&marginCoin=USDT")
        hold_side = "long" if side == "buy" else "short"
        actual_size = size; pos_exists = False
        if pos_r.get("code") == "00000":
            for p in pos_r.get("data", []):
                if p["symbol"] == symbol and float(p.get("total", 0)) > 0:
                    actual_size = p["total"]; hold_side = p.get("holdSide", hold_side)
                    pos_exists = True; break

        if not pos_exists:
            logger.error(f"{symbol} 开仓后无持仓！")
            return False

        # 5. 挂 TP/SL — 只用必要字段
        close_side = "sell" if hold_side == "long" else "buy"
        tpsl_body = {"symbol":symbol,"marginCoin":"USDT","size":str(actual_size),
                     "holdSide":"buy" if hold_side == "long" else "sell",
                     "productType":"USDT-FUTURES","planType":"loss_plan"}

        sl_r = self._post("/api/v2/mix/order/place-tpsl-order",
                         {**tpsl_body,"triggerPrice":str(sl_price),"planType":"loss_plan"})
        tpsl_body["planType"] = "profit_plan"
        tp_r = self._post("/api/v2/mix/order/place-tpsl-order",
                         {**tpsl_body,"triggerPrice":str(tp)})

        sl_ok = sl_r.get("code") == "00000"
        tp_ok = tp_r.get("code") == "00000"

        if sl_ok and tp_ok:
            logger.info(f"✅ TP/SL已挂 {symbol}")
            return True
        else:
            # 挂不上 → 立即平仓
            logger.error(f"❌ {symbol} TP/SL失败 止盈{tp_r.get('code')} 止损{sl_r.get('code')} → 平仓!")
            close = self._post("/api/v2/mix/order/place-order",
                               {"symbol":symbol,"marginCoin":"USDT","side":close_side,
                                "orderType":"market","size":str(actual_size),"reduceOnly":"YES",
                                "productType":"USDT-FUTURES","marginMode":"crossed"})
            if close.get("code") != "00000":
                logger.error(f"⚠️ {symbol} 平仓也失败！请手动平仓！")
            return False


# ==== 信号 ====
def compute_rsi(close, period=RSI_P):
    d = close.diff().dropna()
    if len(d) < period: return 50.0
    g = d.clip(lower=0); l = (-d).clip(lower=0)
    ag = g.ewm(alpha=1/period, adjust=False).mean()
    al = l.ewm(alpha=1/period, adjust=False).mean()
    if al.iloc[-1] == 0: return 100.0 if ag.iloc[-1] > 0 else 50.0
    return float(100 - 100 / (1 + ag.iloc[-1] / al.iloc[-1]))

def price_fmt(p):
    if p < 0.01: return f"${p:.8f}"
    elif p < 1: return f"${p:.6f}"
    elif p < 10: return f"${p:.5f}"
    elif p < 100: return f"${p:.4f}"
    elif p < 1000: return f"${p:.3f}"
    else: return f"${p:.2f}"

def check_signal(api, symbol):
    df = api.candles(symbol)
    if len(df) < 50: return None
    c, h, l, v = df["close"], df["high"], df["low"], df["vol"]
    price = float(c.iloc[-1])
    rsi = compute_rsi(c)
    vm = v.rolling(20).mean()
    vr = float(v.iloc[-1] / vm.iloc[-1]) if vm.iloc[-1] > 0 else 1
    if pd.isna(rsi): return None
    if rsi < RSI_L and vr > MIN_VOL and price > float(l.iloc[-1]) * 1.003:
        logger.info(f"📈 {symbol} LONG  RSI={rsi:.0f} Vol={vr:.1f}x {price_fmt(price)}")
        return {"dir":"long","price":price,"rsi":rsi}
    if rsi > RSI_S and vr > MIN_VOL and price < float(h.iloc[-1]) * 0.997:
        logger.info(f"📉 {symbol} SHORT RSI={rsi:.0f} Vol={vr:.1f}x {price_fmt(price)}")
        return {"dir":"short","price":price,"rsi":rsi}
    return None


# ==== 主循环 ====
def run(dry=True):
    api = Bitget(dry=dry)
    bal = api.balance()
    start_equity = bal["available"]
    logger.info(f"余额: ${bal['available']:.0f} | 持仓: {len(api.positions())} | 扫描: {SCAN_INTERVAL}s")
    count = 0; fail_streak = 0; error_streak = 0

    while True:
        try:
            pos = api.positions()
            bal = api.balance()

            # === 风控 kill switch ===
            if not dry:
                daily_dd = (bal["available"] - start_equity) / start_equity if start_equity > 0 else 0
                if daily_dd < -DAILY_LOSS_LIMIT:
                    logger.error(f"日亏 {daily_dd:.1%} > {DAILY_LOSS_LIMIT:.0%} 停机"); break
                if fail_streak >= CONSECUTIVE_FAILS_MAX:
                    logger.error(f"连续开仓失败 {fail_streak} 次 停机"); break
                if error_streak >= CONSECUTIVE_ERRORS_MAX:
                    logger.error(f"连续 API 错误 {error_streak} 次 停机"); break

            # === 扫描信号 ===
            if len(pos) < MAX_POSITIONS:
                # 查挂单（防重复）
                pending_syms = set()
                for o in api.open_orders():
                    pending_syms.add(o.get("symbol", ""))

                for coin in COINS:
                    if len(pos) >= MAX_POSITIONS: break
                    if any(p.get("symbol") == coin for p in pos): continue
                    if coin in pending_syms: continue  # 有挂单没成交，不重复开

                    # per-symbol cooldown (30秒)
                    now = time.time()
                    if coin in cooldown and now - cooldown[coin] < 30: continue

                    sig = check_signal(api, coin)
                    if sig:
                        price = sig["price"]
                        risk_dollar = bal["available"] * RISK
                        notional = risk_dollar / SL
                        size = round(notional / price, 4)

                        if size > 0:
                            lev = min(LEVERAGE, MAX_LEVERAGE.get(coin, 75))
                            api.set_leverage(coin, lev)
                            side = "buy" if sig["dir"] == "long" else "sell"
                            tp_price = price * (1 + TP if sig["dir"] == "long" else 1 - TP)
                            sl_price = price * (1 - SL if sig["dir"] == "long" else 1 + SL)
                            prec = PREC_MAP.get(coin, 2)
                            tp_price = round(tp_price, prec)
                            sl_price = round(sl_price, prec)
                            limit_price = round(price * (1 - 0.0005 if sig["dir"] == "long" else 1 + 0.0005), prec)

                            ok = api.open_with_tpsl(coin, side, size, limit_price, tp_price, sl_price)
                            if ok:
                                fail_streak = 0
                                margin = notional / lev
                                logger.info(f"✅ {coin} {sig['dir']} 名义${notional:.0f} {lev}x保证金${margin:.0f} 风险${risk_dollar:.0f} {price_fmt(price)}→止盈{price_fmt(tp_price)} 止损{price_fmt(sl_price)}")
                                # 立即刷新持仓 防同轮重复开
                                pos = api.positions()
                            else:
                                fail_streak += 1
                                logger.error(f"❌ {coin} 开仓失败 (连续{fail_streak}次)")
                                cooldown[coin] = time.time()
                        time.sleep(1)

            count += 1; error_streak = 0
            if count % 6 == 0:
                b = api.balance()
                logger.info(f"💼 权益 ${b['available']:.0f} | 持仓 {len(pos)} | #{count}")

        except KeyboardInterrupt:
            logger.info("用户停止"); break
        except Exception as e:
            error_streak += 1
            logger.error(f"异常#{error_streak}: {e}")
            time.sleep(SCAN_INTERVAL)

        time.sleep(SCAN_INTERVAL)

    logger.info("机器人已停")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--live", action="store_true")
    a = p.parse_args()
    dry = not a.live
    if not dry:
        for k in ["BITGET_KEY","BITGET_SECRET","BITGET_PASSPHRASE"]:
            if not os.getenv(k): logger.error(f"请设置环境变量 {k}"); exit(1)
        logger.info("⚠️ 实盘模式启动，3秒后开始交易...")
        time.sleep(3)
    run(dry=dry)
