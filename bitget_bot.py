"""
Bitget 实盘交易机器人
基于 RSI 极端值 + 放量的杠杆超短线策略（多币种并发）

⚠️ 警告：这是实盘交易代码，使用前务必：
  1. 先用 --dry-run 模拟跑一周，确认逻辑没问题
  2. 从小资金开始（$100-500），不要一上来就大仓位
  3. 10x 杠杆极高风险，一天亏光完全可能
  4. API Key 不要在代码里硬编码

用法:
    # 模拟运行（不实际下单，安全测试）
    python bitget_bot.py --dry-run

    # 实盘运行（真金白银！）
    export BITGET_KEY="your_api_key"
    export BITGET_SECRET="your_secret"
    export BITGET_PASSPHRASE="your_passphrase"
    python bitget_bot.py --live
"""

import os
import time
import json
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bitget_bot")

# ============================================================
# 策略参数（和回测一致）
# ============================================================
LEVERAGE = 10
RISK_PER_TRADE = 0.03       # 每笔风险 3%
TP_PRICE_PCT = 0.015        # 止盈 1.5%
SL_PRICE_PCT = 0.004        # 止损 0.4%
TRAILING_STOP = 0.004       # 移动止盈
RSI_PERIOD = 5
RSI_LONG = 25
RSI_SHORT = 78
MIN_VOL_RATIO = 1.2
MAX_HOLD_BARS = 3
MAX_POSITIONS = 3

# 交易标
COINS = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT", "DOGEUSDT"]
# Bitget 产品类型
PRODUCT_TYPE = "USDT-FUTURES"
MARGIN_COIN = "USDT"

# 风控
MAX_DAILY_LOSS = 0.15       # 日亏损 15% 停机
MIN_EQUITY = 50              # 权益低于 $50 停机
SCAN_INTERVAL = 300          # 扫描间隔 5 分钟


class BitgetBot:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.client = None
        self.equity = 0.0
        self.start_equity = 0.0
        self.daily_pnl = 0.0
        self.last_day = None
        self.positions = []    # [{symbol, direction, entry_price, size, ...}]
        self.trade_history = []

        if not dry_run:
            self._init_client()

    def _init_client(self):
        """初始化 Bitget 客户端"""
        from bitget.client import Client

        api_key = os.getenv("BITGET_KEY")
        secret = os.getenv("BITGET_SECRET")
        passphrase = os.getenv("BITGET_PASSPHRASE")

        if not all([api_key, secret, passphrase]):
            raise ValueError("请设置环境变量 BITGET_KEY / BITGET_SECRET / BITGET_PASSPHRASE")

        self.client = Client(api_key, secret, passphrase, verbose=False)
        logger.info("Bitget 客户端初始化成功")

    # ============================================================
    # 数据获取
    # ============================================================
    def fetch_candles(self, symbol: str, granularity: str = "15m",
                      limit: int = 200) -> pd.DataFrame:
        """获取 K 线数据"""
        if self.dry_run:
            return self._fetch_mock_candles(symbol)

        try:
            result = self.client.mix_get_candles(
                symbol=symbol,
                granularity=granularity,
                startTime=int((datetime.now() - timedelta(hours=limit * 0.25)).timestamp() * 1000),
                endTime=int(datetime.now().timestamp() * 1000),
                limit=str(limit),
            )
            if isinstance(result, dict) and result.get("code") == "00000":
                data = result["data"]
                df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "baseVol", "quoteVol"])
                for col in ["open", "high", "low", "close", "baseVol"]:
                    df[col] = pd.to_numeric(df[col])
                df["ts"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms")
                return df
        except Exception as e:
            logger.error(f"获取 {symbol} K线失败: {e}")
        return pd.DataFrame()

    def _fetch_mock_candles(self, symbol: str) -> pd.DataFrame:
        """模拟数据（dry-run 用 yfinance）"""
        try:
            import yfinance as yf
            sym = symbol.replace("USDT", "-USD")
            df = yf.Ticker(sym).history(period="60d", interval="1h")
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            return df.rename(columns={"date": "ts"})
        except:
            # 生成随机数据
            np.random.seed(hash(symbol) % 10000)
            n = 200
            prices = 100 + np.cumsum(np.random.normal(0, 1, n))
            return pd.DataFrame({
                "ts": pd.date_range(end=datetime.now(), periods=n, freq="15min"),
                "open": prices, "high": prices * 1.005, "low": prices * 0.995,
                "close": prices, "baseVol": np.random.lognormal(10, 1, n),
            })

    # ============================================================
    # 信号生成
    # ============================================================
    def compute_rsi(self, close: pd.Series) -> float:
        """计算最新 RSI 值"""
        d = close.diff().dropna()
        if len(d) < RSI_PERIOD:
            return 50.0
        g = d.clip(lower=0)
        l = (-d).clip(lower=0)
        avg_g = g.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
        avg_l = l.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
        if avg_l.iloc[-1] == 0:
            return 100.0 if avg_g.iloc[-1] > 0 else 50.0
        return float(100 - 100 / (1 + avg_g.iloc[-1] / avg_l.iloc[-1]))

    def check_signal(self, symbol: str) -> Optional[dict]:
        """检查是否有入场信号"""
        df = self.fetch_candles(symbol)
        if len(df) < 50:
            return None

        close = df["close"]
        volume = df.get("baseVol", df.get("volume", pd.Series([0] * len(df))))
        high = df["high"]
        low = df["low"]
        current_price = float(close.iloc[-1])

        rsi_val = self.compute_rsi(close)
        vol_ma20 = volume.rolling(20).mean()
        vol_ratio = float(volume.iloc[-1] / vol_ma20.iloc[-1] if vol_ma20.iloc[-1] > 0 else 1)

        if pd.isna(rsi_val):
            return None

        # 做多信号
        if rsi_val < RSI_LONG and vol_ratio > MIN_VOL_RATIO:
            signal_low = float(low.iloc[-1])
            if current_price > signal_low * 1.003:  # 反弹确认
                logger.info(f"📈 {symbol} 做多信号 RSI={rsi_val:.0f} Vol={vol_ratio:.1f}x")
                return {"direction": "long", "symbol": symbol, "rsi": rsi_val, "price": current_price}

        # 做空信号
        if rsi_val > RSI_SHORT and vol_ratio > MIN_VOL_RATIO:
            signal_high = float(high.iloc[-1])
            if current_price < signal_high * 0.997:
                logger.info(f"📉 {symbol} 做空信号 RSI={rsi_val:.0f} Vol={vol_ratio:.1f}x")
                return {"direction": "short", "symbol": symbol, "rsi": rsi_val, "price": current_price}

        return None

    # ============================================================
    # 仓位管理
    # ============================================================
    def get_equity(self) -> float:
        """获取当前权益"""
        if self.dry_run:
            if self.equity == 0:
                self.equity = 10000.0  # 模拟起始资金
            return self.equity

        try:
            result = self.client.mix_get_accounts(productType=PRODUCT_TYPE)
            if result.get("code") == "00000" and result["data"]:
                account = result["data"][0]
                return float(account.get("equity", 0))
        except Exception as e:
            logger.error(f"获取账户失败: {e}")
        return 0.0

    def get_positions(self) -> list:
        """获取当前持仓"""
        if self.dry_run:
            return self.positions

        try:
            result = self.client.mix_get_all_positions(
                productType=PRODUCT_TYPE, marginCoin=MARGIN_COIN
            )
            if result.get("code") == "00000":
                return [p for p in result["data"] if float(p.get("available", 0)) > 0]
        except Exception as e:
            logger.error(f"获取持仓失败: {e}")
        return []

    def open_position(self, signal: dict):
        """开仓"""
        symbol = signal["symbol"]
        direction = signal["direction"]

        # 检查持仓数量
        current_positions = self.get_positions()
        if len(current_positions) >= MAX_POSITIONS:
            logger.info(f"已达最大持仓数 {MAX_POSITIONS}，跳过 {symbol}")
            return

        # 计算仓位
        equity = self.get_equity()
        risk_amount = equity * RISK_PER_TRADE
        pos_value = risk_amount / (SL_PRICE_PCT * LEVERAGE)
        pos_value = min(pos_value, equity * LEVERAGE, 500000)
        size = pos_value / signal["price"]  # 合约张数（USDT 永续按 USDT 计价）
        size = round(size, 4)

        side = "buy" if direction == "short" else "sell"  # Bitget: buy=开多, sell=开空
        # 实际上: open_long=buy, open_short=sell
        if direction == "long":
            side = "open_long"
        else:
            side = "open_short"

        logger.info(f"🚀 开仓 {symbol} {direction} 数量={size} 杠杆={LEVERAGE}x")

        if self.dry_run:
            self.positions.append({
                "symbol": symbol, "direction": direction,
                "entry_price": signal["price"], "size": size,
                "entry_time": datetime.now(),
            })
            logger.info(f"  [模拟] 已记录仓位")
            return

        # 实盘下单
        try:
            # 先设杠杆
            self.client.mix_adjust_leverage(
                symbol=symbol, marginCoin=MARGIN_COIN,
                leverage=str(LEVERAGE), holdSide="long" if direction == "long" else "short"
            )

            # 市价单
            order = self.client.mix_place_order(
                symbol=symbol, marginCoin=MARGIN_COIN,
                size=str(size), side="buy" if direction == "long" else "sell",
                orderType="market",
                # 预设止盈止损
                presetTakeProfitPrice=str(signal["price"] * (1 + TP_PRICE_PCT if direction == "long" else 1 - TP_PRICE_PCT)),
                presetStopLossPrice=str(signal["price"] * (1 - SL_PRICE_PCT if direction == "long" else 1 + SL_PRICE_PCT)),
            )
            logger.info(f"  订单: {order}")
        except Exception as e:
            logger.error(f"  下单失败: {e}")

    def close_position(self, pos: dict, reason: str):
        """平仓"""
        symbol = pos.get("symbol", "")
        direction = pos.get("direction", "long")
        logger.info(f"🏁 平仓 {symbol} {direction} 原因: {reason}")

        if self.dry_run:
            self.equity += pos.get("pnl", 0)
            self.positions = [p for p in self.positions if p != pos]
            return

        try:
            size = str(pos.get("available", pos.get("size", 0)))
            order = self.client.mix_place_order(
                symbol=symbol, marginCoin=MARGIN_COIN,
                size=size, side="sell" if direction == "long" else "buy",
                orderType="market", reduceOnly=True,
            )
            logger.info(f"  平仓单: {order}")
        except Exception as e:
            logger.error(f"  平仓失败: {e}")

    # ============================================================
    # 风险检查
    # ============================================================
    def check_risk_limits(self) -> bool:
        """检查是否触发风控限制"""
        equity = self.get_equity()

        # 最低权益
        if equity < MIN_EQUITY:
            logger.error(f"⚠️ 权益 ${equity:.2f} 低于最低 ${MIN_EQUITY}，停机")
            return False

        # 日亏损限制
        today = datetime.now().date()
        if self.last_day != today:
            self.start_equity = equity
            self.daily_pnl = 0
            self.last_day = today

        daily_change = (equity - self.start_equity) / self.start_equity if self.start_equity > 0 else 0
        if daily_change < -MAX_DAILY_LOSS:
            logger.error(f"⚠️ 日亏损 {daily_change:.1%} 超过 {MAX_DAILY_LOSS:.1%}，停机")
            return False

        return True

    # ============================================================
    # 主循环
    # ============================================================
    def run(self):
        logger.info("=" * 50)
        logger.info(f"Bitget 交易机器人启动")
        logger.info(f"模式: {'模拟' if self.dry_run else '⚠️ 实盘 ⚠️'}")
        logger.info(f"标的: {', '.join(COINS)}")
        logger.info(f"杠杆: {LEVERAGE}x | 最大持仓: {MAX_POSITIONS} | 扫描间隔: {SCAN_INTERVAL}s")
        logger.info(f"风控: 日亏损 {MAX_DAILY_LOSS:.0%} 停机 | 最低权益 ${MIN_EQUITY}")
        logger.info("=" * 50)

        if self.dry_run:
            self.equity = 10000.0
            self.start_equity = 10000.0
        else:
            self.equity = self.get_equity()
            self.start_equity = self.equity

        scan_count = 0

        while True:
            try:
                # 风控检查
                if not self.check_risk_limits():
                    logger.info("风控触发，退出")
                    break

                # 检查已有仓位是否需要平仓
                positions = self.get_positions()
                for pos in positions:
                    symbol = pos.get("symbol", "")
                    # 计算未实现盈亏（简化版，实盘应从交易所拉）
                    unreal_pnl = float(pos.get("unrealizedPL", 0))
                    entry_price = float(pos.get("openPriceAvg", pos.get("entry_price", 0)))
                    if entry_price > 0:
                        pnl_pct = unreal_pnl / (entry_price * float(pos.get("available", 1)))
                        # 这里简化处理，实际应检查详细止盈止损
                        if abs(pnl_pct) > TP_PRICE_PCT * LEVERAGE * 0.5:
                            pass  # 接近止盈，交给交易所预设单处理

                # 扫描新信号
                if len(positions) < MAX_POSITIONS:
                    for coin in COINS:
                        if len(positions) >= MAX_POSITIONS:
                            break
                        # 检查是否已有同币种持仓
                        if any(p.get("symbol") == coin for p in positions):
                            continue

                        signal = self.check_signal(coin)
                        if signal:
                            self.open_position(signal)
                            time.sleep(1)  # 避免瞬间多次下单

                scan_count += 1
                if scan_count % 12 == 0:  # 每小时汇报一次
                    equity = self.get_equity()
                    total_return = (equity / self.start_equity - 1) * 100 if self.start_equity > 0 else 0
                    logger.info(f"💼 权益: ${equity:,.0f} ({total_return:+.1f}%) | "
                                f"持仓: {len(positions)} | 扫描: {scan_count}次")

                time.sleep(SCAN_INTERVAL)

            except KeyboardInterrupt:
                logger.info("用户中断")
                break
            except Exception as e:
                logger.error(f"主循环异常: {e}")
                time.sleep(SCAN_INTERVAL)

        logger.info(f"机器人已停止。最终权益: ${self.get_equity():,.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bitget 量化交易机器人")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="模拟运行（默认），不实际下单")
    parser.add_argument("--live", action="store_true",
                        help="实盘运行（需设置 API 环境变量）")
    args = parser.parse_args()

    dry = not args.live
    bot = BitgetBot(dry_run=dry)

    if not dry:
        print("\n" + "!" * 50)
        print("  ⚠️  实盘模式 - 真金白银交易 ⚠️")
        print("  确认你已: ")
        print("  1. 在 dry-run 模式下测试过策略逻辑")
        print("  2. 了解 10x 杠杆的风险")
        print("  3. 只用闲钱交易")
        print("!" * 50)
        confirm = input("\n  输入 'YES' 确认继续: ")
        if confirm != "YES":
            print("已取消")
            exit()

    bot.run()
