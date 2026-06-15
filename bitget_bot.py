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
TP = 0.025
SL = 0.006
WICK = 0.003          # 影线过滤：收盘价必须超过最低/最高价 0.3%
TRAIL_CONFIG = {      # 分币种激活门槛，统一 0.3% 回撤距离
    "ETHUSDT":  {"activate": 0.006, "trail": 0.003},
    "SOLUSDT":  {"activate": 0.015, "trail": 0.003},
    "BNBUSDT":  {"activate": 0.004, "trail": 0.003},
    "AVAXUSDT": {"activate": 0.006, "trail": 0.003},
    "DOGEUSDT": {"activate": 0.012, "trail": 0.003},
}
RSI_P = 5; RSI_L = 20; RSI_S = 82; MIN_VOL = 1.5
MAX_POSITIONS = 3
COINS = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT", "DOGEUSDT"]
MAX_LEVERAGE = {"ETHUSDT":150,"SOLUSDT":100,"BNBUSDT":75,"AVAXUSDT":75,"DOGEUSDT":75}
PREC_MAP = {"BTCUSDT":1,"ETHUSDT":2,"SOLUSDT":3,"BNBUSDT":2,"AVAXUSDT":3,"DOGEUSDT":5}
SCAN_INTERVAL = 60
DAILY_LOSS_LIMIT = 0.20    # 日亏20%停机（不可自动重启）
CONSECUTIVE_ERRORS_MAX = 5 # 连续 API 错误停机（自动重启）

MIN_TRAIL_STEP = 0.001  # SL 改善 < 0.1% 不修改，避免频繁改单
cooldown = {}          # {symbol: last_entry_ts} 防重复开仓
trailing_state = {}    # {symbol: {side, entry, best, sl_order_id, tp_order_id, last_sl}}

# ==== Bitget ====
class Bitget:
    BASE = "https://api.bitget.com"

    def __init__(self, dry=True):
        self.dry = dry
        self.stop_trading = False
        self._tpsl_oids = {}  # {symbol: {"sl": orderId, "tp": orderId}}
        if not dry:
            self.key = os.getenv("BITGET_KEY")
            self.secret = os.getenv("BITGET_SECRET")
            self.pw = os.getenv("BITGET_PASSPHRASE")

    def recover_tpsl_oids(self):
        """重启后从 pending plan orders 恢复 sl_order_id/tp_order_id"""
        if self.dry:
            return {}
        recovered = {}
        d = self._get("/api/v2/mix/order/orders-plan-pending?productType=USDT-FUTURES")
        if d.get("code") != "00000":
            return {}
        # 兼容两种返回结构：entrustedList / 直接数组
        data = d.get("data", {})
        orders = data if isinstance(data, list) else data.get("entrustedList", [])
        for o in orders:
            sym = o.get("symbol", "")
            plan = o.get("planType", "")
            oid = o.get("orderId", "")
            trigger = o.get("triggerPrice")
            if not sym or not oid:
                continue
            entry = recovered.get(sym, {"sl": "", "tp": "", "last_sl_hint": 0})
            if "loss" in plan:
                entry["sl"] = oid
                if trigger:
                    entry["last_sl_hint"] = float(trigger)
            elif "profit" in plan:
                entry["tp"] = oid
            recovered[sym] = entry
        return recovered

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
        raise RuntimeError(f"balance API 失败: {d.get('code')} {d.get('msg','')}")

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
        raise RuntimeError(f"positions API 失败: {d.get('code')} {d.get('msg','')}")

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
        raise RuntimeError(f"open_orders API 失败: {d.get('code')} {d.get('msg','')}")

    def set_leverage(self, symbol, lev):
        if self.dry: return True
        r1 = self._post("/api/v2/mix/account/set-position-mode", {"productType":"USDT-FUTURES","posMode":"one_way_mode"})
        r2 = self._post("/api/v2/mix/account/set-leverage",
                        {"symbol":symbol,"marginCoin":"USDT","leverage":str(lev),
                         "productType":"USDT-FUTURES","marginMode":"crossed"})
        if r1.get("code") != "00000" or r2.get("code") != "00000":
            logger.error(f"设杠杆失败 {symbol}: mode={r1.get('code')} lev={r2.get('code')}")
            return False
        return True

    def _position_for_symbol(self, symbol):
        """返回当前真实持仓数量和方向。用于紧急保护，避免用成交回报猜仓位。"""
        pos_r = self._get("/api/v2/mix/position/all-position?productType=USDT-FUTURES&marginCoin=USDT")
        if pos_r.get("code") == "00000":
            for p in pos_r.get("data", []):
                if p.get("symbol") == symbol and float(p.get("total", 0)) > 0:
                    return float(p.get("total", 0)), p.get("holdSide", "")
        return 0.0, ""

    def _order_state(self, symbol, order_id):
        d = self._get(f"/api/v2/mix/order/detail?symbol={symbol}&productType=USDT-FUTURES&orderId={order_id}")
        if d.get("code") != "00000":
            return "", 0.0
        data = d.get("data", {})
        return data.get("state", ""), float(data.get("baseVolume", 0))

    def _emergency_close(self, symbol, entry_side, fallback_size=0):
        actual_size, hold_side = self._position_for_symbol(symbol)
        if actual_size <= 0:
            actual_size = float(fallback_size or 0)
            hold_side = "long" if entry_side == "buy" else "short"
        if actual_size <= 0:
            logger.error(f"{symbol} 紧急平仓失败：未能确认持仓数量")
            return False
        close_side = "sell" if hold_side == "long" else "buy"
        close_r = self._post("/api/v2/mix/order/place-order",
                             {"symbol":symbol,"marginCoin":"USDT","side":close_side,
                              "orderType":"market","size":str(actual_size),
                              "reduceOnly":"YES","productType":"USDT-FUTURES",
                              "marginMode":"crossed"})
        logger.error(f"{symbol} 紧急平仓: {close_r.get('code')} {close_r.get('msg','')}")
        return close_r.get("code") == "00000"

    def open_with_tpsl(self, symbol, side, size, tp, sl_price):
        """市价单开仓 → 确认成交 → 挂TP/SL。失败则平仓返回False"""
        if self.dry:
            logger.info(f"  [模拟] {side} {size} 市价 {symbol}")
            return True

        # 1. 下市价单
        body = {"symbol":symbol,"marginCoin":"USDT","size":str(size),
                "side":side,"orderType":"market",
                "productType":"USDT-FUTURES","marginMode":"crossed"}
        r = self._post("/api/v2/mix/order/place-order", body)
        if r.get("code") != "00000":
            logger.error(f"市价单失败 {symbol}: {r.get('code')} {r.get('msg','')}")
            return False

        # 2. 等成交（市价单通常秒成交，最多等15秒）
        order_id = r.get("data", {}).get("orderId", "")
        if not order_id:
            logger.error(f"{symbol} 市价单未获取订单ID，执行保护性紧急平仓")
            self._emergency_close(symbol, side, size)
            return False
        filled_qty = 0.0
        for _ in range(5):
            time.sleep(3)
            check = self._get(f"/api/v2/mix/order/detail?symbol={symbol}&productType=USDT-FUTURES&orderId={order_id}")
            if check.get("code") == "00000":
                data = check.get("data", {})
                if data.get("state") in ("filled", "partially_filled"):
                    filled_qty = float(data.get("baseVolume", 0))
                    break
        # 订单详情查不到时，直接用持仓反查（市价单可能已成交但 detail 延迟）
        if filled_qty <= 0:
            real_qty, real_side = self._position_for_symbol(symbol)
            if real_qty > 0:
                filled_qty = real_qty
                logger.info(f"{symbol} order/detail 未返回，但持仓已存在 {real_qty} 张")
        if filled_qty <= 0:
            logger.error(f"{symbol} 市价单成交状态未知，执行保护性紧急平仓")
            self._emergency_close(symbol, side, size)
            return False

        # 3. 确认持仓（重试 + 备用查询）
        actual_size, hold_side = self._position_for_symbol(symbol)
        if actual_size <= 0:
            # 重试一次 position/all-position
            time.sleep(3)
            pos_r = self._get("/api/v2/mix/position/all-position?productType=USDT-FUTURES&marginCoin=USDT")
            if pos_r.get("code") == "00000":
                for p in pos_r.get("data", []):
                    if p.get("symbol") == symbol and float(p.get("total", 0)) > 0:
                        actual_size = float(p.get("total", 0))
                        hold_side = p.get("holdSide", "long")
                        break
        if actual_size <= 0:
            # 仍无持仓 → 立即平掉可能的残留
            logger.error(f"{symbol} 开仓后无持仓，紧急平仓检查")
            self._emergency_close(symbol, side, size)
            return False

        # 5. 挂 TP/SL — 只用必要字段
        close_side = "sell" if hold_side == "long" else "buy"
        tpsl_body = {"symbol":symbol,"marginCoin":"USDT","size":str(actual_size),
                     "holdSide": hold_side,
                     "productType":"USDT-FUTURES","planType":"loss_plan",
                     "triggerType":"mark_price","executePrice":"0"}

        sl_r = self._post("/api/v2/mix/order/place-tpsl-order",
                         {**tpsl_body,"triggerPrice":str(sl_price),"planType":"loss_plan"})
        tpsl_body["planType"] = "profit_plan"
        tp_r = self._post("/api/v2/mix/order/place-tpsl-order",
                         {**tpsl_body,"triggerPrice":str(tp)})

        sl_ok = sl_r.get("code") == "00000"
        tp_ok = tp_r.get("code") == "00000"

        if sl_ok and tp_ok:
            sl_oid = sl_r.get("data", {}).get("orderId", "")
            tp_oid = tp_r.get("data", {}).get("orderId", "")
            self._tpsl_oids[symbol] = {"sl": sl_oid, "tp": tp_oid}
            logger.info(f"✅ TP/SL已挂 {symbol} sl={sl_oid} tp={tp_oid}")
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

    def update_trailing_stop(self, symbol, entry_side, entry_price, best_price,
                             sl_order_id, last_sl):
        """移动止盈：用 modify-tpsl-order 修改止损单，本地 last_sl 节流"""
        if self.dry:
            return False, best_price, last_sl

        cfg = TRAIL_CONFIG.get(symbol, {"activate": 0.006, "trail": 0.003})
        activate = cfg["activate"]
        trail_dist = cfg["trail"]

        # 查当前持仓 & 标记价
        pos_r = self._get(
            f"/api/v2/mix/position/all-position?productType=USDT-FUTURES&marginCoin=USDT"
        )
        if pos_r.get("code") != "00000":
            return False, best_price, last_sl

        mark_price = None; size = None
        for p in pos_r.get("data", []):
            if p.get("symbol") == symbol and float(p.get("total", 0)) > 0:
                mark_price = float(p.get("markPrice", 0))
                size = p.get("total")
                break
        if mark_price is None:
            return False, best_price, last_sl

        # 始终更新最佳价
        if entry_side == "buy":
            new_best = max(best_price, mark_price)
            float_pnl = (mark_price - entry_price) / entry_price
        else:
            new_best = min(best_price, mark_price)
            float_pnl = (entry_price - mark_price) / entry_price

        # 浮盈不足激活门槛 → 不移动
        if float_pnl < activate:
            return False, new_best, last_sl

        # 计算新止损价（按币种价格精度取整）
        prec = PREC_MAP.get(symbol, 2)
        if entry_side == "buy":
            new_sl = round(new_best * (1 - trail_dist), prec)
        else:
            new_sl = round(new_best * (1 + trail_dist), prec)

        # 新止损不能比硬止损差
        if entry_side == "buy" and new_sl <= entry_price * (1 - SL):
            return False, new_best, last_sl
        if entry_side == "sell" and new_sl >= entry_price * (1 + SL):
            return False, new_best, last_sl

        # 节流：用本地 last_sl 判断改善幅度，不调 API
        if last_sl > 0:
            if entry_side == "buy":
                improvement = (new_sl - last_sl) / last_sl
            else:
                improvement = (last_sl - new_sl) / last_sl
            if improvement < MIN_TRAIL_STEP:
                return False, new_best, last_sl

        # 没有 orderId 时不退化为 place（避免多止损单），仅报警
        if not sl_order_id:
            logger.warning(f"  ⚠️ {symbol} 缺少 sl_order_id，跳过移动止盈")
            return False, new_best, last_sl

        # 修改止损单
        modify_r = self._post("/api/v2/mix/order/modify-tpsl-order",
                              {"symbol": symbol, "marginCoin": "USDT",
                               "productType": "USDT-FUTURES",
                               "planType": "loss_plan",
                               "orderId": str(sl_order_id),
                               "triggerPrice": str(new_sl),
                               "executePrice": "0",
                               "triggerType": "mark_price",
                               "size": str(size or "")})
        if modify_r.get("code") == "00000":
            logger.info(f"  🔒 {symbol} 移动止损 → {new_sl} (浮盈{float_pnl:+.2%})")
            return True, new_best, new_sl
        else:
            logger.error(f"  ⚠️ {symbol} 移动止损失败: {modify_r.get('code')} {modify_r.get('msg','')}")
            # 失败后更新 last_sl 为新值，避免每分钟重试同一价格
            return False, new_best, new_sl


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
    # 用 iloc[-2] 取上一根已收完的 K 线，避免当前未完成 K 线的假信号
    c, h, l, v = df["close"], df["high"], df["low"], df["vol"]
    idx = -2 if len(df) >= 2 else -1
    price = float(c.iloc[idx])
    rsi = compute_rsi(c.iloc[:idx+1])  # RSI 也只用到已完成 K
    vm = v.rolling(20).mean()
    vr = float(v.iloc[idx] / vm.iloc[idx]) if vm.iloc[idx] > 0 else 1
    if pd.isna(rsi): return None
    if rsi < RSI_L and vr > MIN_VOL and price > float(l.iloc[idx]) * (1 + WICK):
        logger.info(f"📈 {symbol} LONG  RSI={rsi:.0f} Vol={vr:.1f}x {price_fmt(price)}")
        return {"dir":"long","price":price,"rsi":rsi}
    if rsi > RSI_S and vr > MIN_VOL and price < float(h.iloc[idx]) * (1 - WICK):
        logger.info(f"📉 {symbol} SHORT RSI={rsi:.0f} Vol={vr:.1f}x {price_fmt(price)}")
        return {"dir":"short","price":price,"rsi":rsi}
    return None


# ==== 主循环 ====
def run(dry=True):
    api = Bitget(dry=dry)
    bal = api.balance()
    start_equity_for_dd = bal["available"]  # 当日初始余额，每天 0 点刷新
    today = time.localtime().tm_yday        # 一年中的第几天
    logger.info(f"余额: ${bal['available']:.0f} | 持仓: {len(api.positions())} | 扫描: {SCAN_INTERVAL}s")
    count = 0; error_streak = 0
    restart_delay = 60  # 非致命停机后等待 60s 自动重启

    while True:
        try:
            pos = api.positions()
            bal = api.balance()

            # === 日亏重置（每天 0 点刷新基准） ===
            if not dry:
                cur_day = time.localtime().tm_yday
                if cur_day != today:
                    start_equity_for_dd = bal["available"]
                    today = cur_day
                    logger.info(f"📅 新的一天，日亏基准重置为 ${start_equity_for_dd:.0f}")

            # === 风控 kill switch ===
            if not dry:
                if api.stop_trading:
                    logger.error("触发停机保护，停止开新仓")
                    return  # 不可重启
                daily_dd = (bal["available"] - start_equity_for_dd) / start_equity_for_dd if start_equity_for_dd > 0 else 0
                if daily_dd < -DAILY_LOSS_LIMIT:
                    logger.error(f"日亏 {daily_dd:.1%} > {DAILY_LOSS_LIMIT:.0%} 停机")
                    return  # 不可重启
                if error_streak >= CONSECUTIVE_ERRORS_MAX:
                    logger.error(f"连续 API 错误 {error_streak} 次，{restart_delay}s 后继续尝试")
                    time.sleep(restart_delay)
                    error_streak = 0
                    continue

            # === 移动止盈检查 ===
            if not dry:
                global trailing_state
                # 重启后恢复：从 pending plan orders 找回 orderId
                if not hasattr(api, '_recovered'):
                    api._recovered = True
                    recovered = api.recover_tpsl_oids()
                    open_syms = {p.get("symbol") for p in pos}
                    for sym, oids in recovered.items():
                        if sym not in open_syms:
                            # 已平仓但有残留计划单 → 清理
                            for oid_key, plan_type in [("sl","loss_plan"), ("tp","profit_plan")]:
                                oid = oids.get(oid_key, "")
                                if oid:
                                    api._post("/api/v2/mix/order/cancel-plan-order",
                                             {"symbol": sym, "marginCoin": "USDT",
                                              "productType": "USDT-FUTURES",
                                              "orderIdList": [{"orderId": str(oid)}],
                                              "planType": plan_type})
                            logger.info(f"🧹 {sym} 残留计划单已清理")
                            continue
                        api._tpsl_oids[sym] = {"sl": oids.get("sl",""), "tp": oids.get("tp",""),
                                               "last_sl_hint": oids.get("last_sl_hint", 0)}
                        if sym in trailing_state:
                            trailing_state[sym]["sl_order_id"] = oids.get("sl", "")
                            trailing_state[sym]["tp_order_id"] = oids.get("tp", "")
                            hint = oids.get("last_sl_hint", 0)
                            if hint > 0:
                                trailing_state[sym]["last_sl"] = hint
                    if recovered:
                        logger.info(f"🔄 恢复TPSL orderId: {list(recovered.keys())}")
                # 初始化新持仓状态
                for p in pos:
                    sym = p.get("symbol", "")
                    hold_side = p.get("holdSide", "")
                    if hold_side not in ("long", "short"):
                        continue
                    if sym not in trailing_state:
                        open_price = float(p.get("openPriceAvg", 0) or p.get("openPrice", 0))
                        if open_price <= 0:
                            continue
                        oids = api._tpsl_oids.get(sym, {})
                        hard_sl = open_price * (1 - SL if hold_side == "long" else 1 + SL)
                        last_sl = float(oids.get("last_sl_hint", 0) or hard_sl)
                        trailing_state[sym] = {
                            "side": "buy" if hold_side == "long" else "sell",
                            "entry": open_price,
                            "best": open_price,
                            "sl_order_id": oids.get("sl", ""),
                            "tp_order_id": oids.get("tp", ""),
                            "last_sl": last_sl,
                        }
                        logger.info(f"📋 {sym} 新持仓追踪 entry={open_price} hard_sl={hard_sl:.4f} last_sl={last_sl:.4f}")
                # 已平仓的 → 清理残留 TP 计划单 + 移除状态
                open_syms = {p.get("symbol") for p in pos}
                for sym in list(trailing_state.keys()):
                    if sym not in open_syms:
                        state = trailing_state.pop(sym)
                        logger.info(f"📋 {sym} 已平仓，清理追踪状态 (entry={state.get('entry')} last_sl={state.get('last_sl')})")
                        # 清理残留 TP 计划单
                        tp_oid = state.get("tp_order_id", "")
                        if tp_oid:
                            r = api._post("/api/v2/mix/order/cancel-plan-order",
                                         {"symbol": sym, "marginCoin": "USDT",
                                          "productType": "USDT-FUTURES",
                                          "orderIdList": [{"orderId": str(tp_oid)}],
                                          "planType": "profit_plan"})
                            if r.get("code") != "00000":
                                logger.warning(f"  ⚠️ {sym} TP清理失败 code={r.get('code')}")
                            else:
                                fl = r.get("data", {}).get("failureList", [])
                                if fl:
                                    logger.warning(f"  ⚠️ {sym} TP清理残留: {fl}")
                        sl_oid = state.get("sl_order_id", "")
                        if sl_oid:
                            r = api._post("/api/v2/mix/order/cancel-plan-order",
                                         {"symbol": sym, "marginCoin": "USDT",
                                          "productType": "USDT-FUTURES",
                                          "orderIdList": [{"orderId": str(sl_oid)}],
                                          "planType": "loss_plan"})
                            if r.get("code") != "00000":
                                logger.warning(f"  ⚠️ {sym} SL清理失败 code={r.get('code')}")
                            else:
                                fl = r.get("data", {}).get("failureList", [])
                                if fl:
                                    logger.warning(f"  ⚠️ {sym} SL清理残留: {fl}")
                # 移动止盈
                for sym, state in list(trailing_state.items()):
                    updated, new_best, new_sl = api.update_trailing_stop(
                        sym, state["side"], state["entry"], state["best"],
                        state.get("sl_order_id", ""),
                        state.get("last_sl", 0)
                    )
                    state["best"] = new_best          # 始终更新 best
                    if updated and new_sl > 0:
                        state["last_sl"] = new_sl      # 更新本地 last_sl 用于节流

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

                    # 安全检查：不能已有该币种的追踪状态（说明上一单平仓没清干净）
                    if coin in trailing_state:
                        logger.warning(f"⚠️ {coin} trailing_state 残留，跳过开仓 (entry={trailing_state[coin].get('entry')})")
                        continue

                    sig = check_signal(api, coin)
                    if sig:
                        price = sig["price"]
                        risk_dollar = bal["available"] * RISK
                        notional = risk_dollar / SL
                        size = round(notional / price, 4)

                        if size > 0:
                            lev = min(LEVERAGE, MAX_LEVERAGE.get(coin, 75))
                            if not api.set_leverage(coin, lev):
                                logger.error(f"{coin} 设杠杆失败，跳过")
                                continue
                            side = "buy" if sig["dir"] == "long" else "sell"
                            tp_price = price * (1 + TP if sig["dir"] == "long" else 1 - TP)
                            sl_price = price * (1 - SL if sig["dir"] == "long" else 1 + SL)
                            prec = PREC_MAP.get(coin, 2)
                            tp_price = round(tp_price, prec)
                            sl_price = round(sl_price, prec)
                            ok = api.open_with_tpsl(coin, side, size, tp_price, sl_price)
                            if ok:
                                margin = notional / lev
                                logger.info(f"✅ {coin} {sig['dir']} 名义${notional:.0f} {lev}x保证金${margin:.0f} 风险${risk_dollar:.0f} {price_fmt(price)}→止盈{price_fmt(tp_price)} 止损{price_fmt(sl_price)}")
                                # 存储 TP/SL orderId 到追踪状态
                                oids = api._tpsl_oids.get(coin, {})
                                if coin in trailing_state:
                                    trailing_state[coin]["sl_order_id"] = oids.get("sl", "")
                                    trailing_state[coin]["tp_order_id"] = oids.get("tp", "")
                                # 刷新持仓 + cooldown 防重复
                                pos = api.positions()
                                cooldown[coin] = time.time()
                            else:
                                logger.error(f"❌ {coin} 开仓失败")
                                cooldown[coin] = time.time()
                        time.sleep(1)

            count += 1; error_streak = 0
            if count % 6 == 0:
                b = api.balance()
                logger.info(f"💼 权益 ${b.get('equity',b['available']):.0f} | 持仓 {len(pos)} | #{count}")

        except KeyboardInterrupt:
            logger.info("用户停止"); return
        except Exception as e:
            error_streak += 1
            logger.error(f"异常#{error_streak}: {e}")
            if error_streak >= CONSECUTIVE_ERRORS_MAX:
                logger.error(f"连续 API 异常 {error_streak} 次，{restart_delay}s 后继续")
                time.sleep(restart_delay)
                error_streak = 0
            time.sleep(SCAN_INTERVAL)
            continue

        time.sleep(SCAN_INTERVAL)

    logger.info("机器人已停（不会到达这里，return/continue 控制流程）")


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
