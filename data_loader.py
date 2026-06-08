"""
数据获取模块
- 用 AkShare 获取 A 股行情数据
- 支持本地缓存，避免重复下载
- 返回 pandas DataFrame，统一格式
"""

import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

import akshare as ak

from config import (
    STOCK_POOL_SIZE, ST_MIN_THRESHOLD, EXCLUDE_ST,
    START_DATE, END_DATE, BENCHMARK_INDEX, DATA_CACHE_DIR,
)


def ensure_cache_dir():
    """确保缓存目录存在"""
    Path(DATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)


def get_stock_pool(top_n: int = STOCK_POOL_SIZE) -> list[str]:
    """
    获取股票池：A 股全部股票按总市值排序，取前 top_n 只。
    自动剔除 ST 股和低价股。

    返回：股票代码列表，如 ['000001', '000002', ...]
    """
    print(f"[Data] 获取 A 股股票列表（按市值取前 {top_n}）...")
    df = ak.stock_zh_a_spot_em()
    # df 列：代码, 名称, 最新价, 涨跌幅, 涨跌额, 成交量, 成交额, 总市值, ...

    # 数据清洗
    df["总市值"] = pd.to_numeric(df["总市值"], errors="coerce")
    df["最新价"] = pd.to_numeric(df["最新价"], errors="coerce")

    # 剔除 ST
    if EXCLUDE_ST:
        df = df[~df["名称"].str.contains("ST|退市", na=False)]

    # 剔除低价股
    df = df[df["最新价"] > ST_MIN_THRESHOLD]

    # 剔除市值为空的
    df = df.dropna(subset=["总市值"])

    # 按市值降序排列，取前 top_n
    df = df.sort_values("总市值", ascending=False)

    stocks = df["代码"].head(top_n).tolist()
    print(f"[Data] 股票池大小: {len(stocks)}，"
          f"市值范围: {df['总市值'].iloc[top_n-1]/1e8:.0f}亿 ~ {df['总市值'].iloc[0]/1e8:.0f}亿")

    return stocks


def get_stock_daily(symbol: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame | None:
    """
    获取单只股票的日线数据（前复权）。

    参数:
        symbol: 股票代码，如 '000001'
        start: 开始日期 'YYYYMMDD'
        end: 结束日期 'YYYYMMDD'

    返回:
        DataFrame with columns: date, open, high, low, close, volume, amount, turnover, change_pct
        失败返回 None
    """
    cache_path = os.path.join(DATA_CACHE_DIR, f"{symbol}.csv")

    # 优先读缓存
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["date"])
        return df

    # 从 AkShare 获取
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq",  # 前复权
        )
        if df is None or df.empty:
            return None

        # 统一列名
        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover",
            "涨跌幅": "change_pct",
        })

        # 保留需要的列
        cols = ["date", "open", "high", "low", "close",
                "volume", "amount", "turnover", "change_pct"]
        df = df[[c for c in cols if c in df.columns]]
        df["date"] = pd.to_datetime(df["date"])

        # 写入缓存
        df.to_csv(cache_path, index=False)
        return df

    except Exception as e:
        print(f"  ⚠ 获取 {symbol} 失败: {e}")
        return None


def get_all_stocks_data(stock_list: list[str]) -> dict[str, pd.DataFrame]:
    """
    批量获取多只股票日线数据，带限速和进度提示。

    返回: {symbol: DataFrame, ...}
    """
    ensure_cache_dir()
    result = {}
    total = len(stock_list)
    print(f"[Data] 下载 {total} 只股票日线数据（{START_DATE} ~ {END_DATE}）...")
    print(f"[Data] 使用缓存目录: {DATA_CACHE_DIR}")

    for i, symbol in enumerate(stock_list):
        df = get_stock_daily(symbol)
        if df is not None and len(df) > 100:  # 至少100个交易日
            result[symbol] = df

        # 进度提示（每20只或每10%）
        if (i + 1) % max(1, total // 10) == 0:
            print(f"  ... {i+1}/{total} (有效 {len(result)})")

        # 限速（避免被 ban），缓存命中时不限速
        cache_path = os.path.join(DATA_CACHE_DIR, f"{symbol}.csv")
        if not os.path.exists(cache_path):
            time.sleep(0.15)

    print(f"[Data] 数据下载完成: {len(result)} 只有效股票")
    return result


def get_benchmark_data(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """
    获取基准指数日线数据（沪深300）。

    返回: DataFrame with columns: date, close
    """
    cache_path = os.path.join(DATA_CACHE_DIR, f"benchmark_{BENCHMARK_INDEX}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["date"])
        return df

    try:
        df = ak.stock_zh_index_daily(symbol=f"sh{BENCHMARK_INDEX}")
        # 列可能不同，重新映射
        if "date" not in df.columns:
            df = df.rename(columns={"date": "date"})
        df = df.rename(columns={c: c.lower() for c in df.columns})
        if "date" not in df.columns:
            # 可能日期是 index
            df = df.reset_index()
        df = df[["date", "close"]]
        df["date"] = pd.to_datetime(df["date"])

        # 时间筛选
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        df.to_csv(cache_path, index=False)
        print(f"[Data] 基准数据: {len(df)} 个交易日")
        return df

    except Exception as e:
        print(f"[Data] ⚠ 获取基准指数失败: {e}")
        print("[Data] 将使用等权基准代替")
        return None


# ============================================================
# 模拟数据生成器（用于无网络/海外环境测试回测逻辑）
# ============================================================

def generate_fake_data(
    n_stocks: int = 50,
    start: str = START_DATE,
    end: str = END_DATE,
    seed: int = 42,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    生成模拟的股票日线数据 + 基准数据。
    用几何布朗运动生成价格，加入随机漂移和波动率，模拟真实市场。

    参数:
        n_stocks: 股票数量
        start, end: 日期范围
        seed: 随机种子（保证可复现）

    返回:
        (stock_data_dict, benchmark_df)
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start, end=end, freq="B")  # 仅交易日
    print(f"[FakeData] 生成 {n_stocks} 只股票的模拟数据，{len(dates)} 个交易日...")

    stock_data = {}

    for i in range(n_stocks):
        symbol = f"60{i:04d}"  # 600000, 600001, ...

        # 每只股票有不同的漂移率和波动率
        mu = np.random.uniform(-0.05, 0.20) / 252       # 年化 -5% ~ +20%
        sigma = np.random.uniform(0.20, 0.50) / np.sqrt(252)  # 年化波动 20%~50%

        # 几何布朗运动
        returns = np.random.normal(mu, sigma, len(dates))
        prices = 100.0 * np.exp(np.cumsum(returns))

        # 产生 OHLCV
        close = prices
        daily_returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
        daily_returns[0] = 0

        high = close * (1 + np.abs(np.random.normal(0, 0.015, len(dates))))
        low = close * (1 - np.abs(np.random.normal(0, 0.015, len(dates))))
        open_price = close * (1 + np.random.normal(0, 0.005, len(dates)))
        volume = np.random.lognormal(15, 1.5, len(dates)).astype(int)
        amount = close * volume

        # 确保 OHLC 关系正确
        for j in range(len(dates)):
            o, h, l, c = open_price[j], high[j], low[j], close[j]
            all_vals = [o, h, l, c]
            high[j] = max(all_vals)
            low[j] = min(all_vals)

        df = pd.DataFrame({
            "date": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
            "turnover": np.random.uniform(0.5, 5.0, len(dates)),
            "change_pct": daily_returns * 100,
        })

        stock_data[symbol] = df

    # 生成基准数据（模拟沪深300走势）
    bench_mu = 0.03 / 252           # 年化 3%
    bench_sigma = 0.20 / np.sqrt(252)  # 年化波动 20%
    bench_returns = np.random.normal(bench_mu, bench_sigma, len(dates))
    bench_prices = 1000.0 * np.exp(np.cumsum(bench_returns))
    benchmark = pd.DataFrame({
        "date": dates,
        "close": bench_prices,
    })

    print(f"[FakeData] 生成完成，{len(stock_data)} 只股票，基准 {len(benchmark)} 行")
    return stock_data, benchmark


# ============================================================
# 美股数据（yfinance，全球都能用）
# ============================================================

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def get_us_stock_pool(top_n: int = 100) -> list[str]:
    """
    获取美股股票池：标普500成分股，按市值取前 top_n。
    """
    print(f"[Data] 获取标普500成分股（取前 {top_n}）...")
    try:
        # 从 Wikipedia 获取标普500成分股
        sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = sp500["Symbol"].tolist()
        # 取前 top_n
        tickers = tickers[:top_n]
        print(f"[Data] 美股股票池: {len(tickers)} 只")
        return tickers
    except Exception:
        # 降级：手动指定一些大市值美股
        fallback = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
            "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "BAC", "DIS",
            "ADBE", "NFLX", "CRM", "AMD", "INTC", "QCOM", "TXN", "PYPL", "CSCO",
            "CMCSA", "PEP", "COST", "ABT", "CVX", "WFC", "MRK", "ABBV", "AVGO",
            "ORCL", "ACN", "TMO", "NKE", "DHR", "LLY", "PM", "UPS", "MS", "GS",
            "BLK", "CAT", "AXP", "SPGI", "T", "VZ",
        ]
        print(f"[Data] 降级使用预设列表: {len(fallback)} 只")
        return fallback[:top_n]


def get_us_stock_daily(symbol: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame | None:
    """
    用 yfinance 获取单只美股日线数据。

    返回统一的 DataFrame 格式（和 A 股接口一致）。
    """
    cache_path = os.path.join(DATA_CACHE_DIR, f"us_{symbol}.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["date"])
        return df

    # 日期格式转换: YYYYMMDD → YYYY-MM-DD
    start_fmt = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
    end_fmt = f"{end[:4]}-{end[4:6]}-{end[6:8]}"

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_fmt, end=end_fmt)

        if df.empty:
            return None

        # 统一列名（yfinance 不同版本列名大小写不同，先全转小写）
        df = df.reset_index()  # Date index → column
        df.columns = [c.lower() for c in df.columns]

        # 重命名标准列
        col_map = {
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume", "date": "date",
        }
        df = df.rename(columns=col_map)

        # 计算缺失列
        if "amount" not in df.columns or df["amount"].isna().all():
            df["amount"] = df["close"] * df["volume"]
        # 美股换手率：用 20 日均量 / 总股本（粗略估计，用成交量相对变化代替）
        if "turnover" not in df.columns or df["turnover"].isna().all():
            avg_vol = df["volume"].rolling(20).mean()
            df["turnover"] = (df["volume"] / avg_vol.replace(0, np.nan)).fillna(1.0) * 2.0  # 缩放到合理范围
        if "change_pct" not in df.columns or df["change_pct"].isna().all():
            df["change_pct"] = df["close"].pct_change() * 100

        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        # 保留标准列
        cols = ["date", "open", "high", "low", "close",
                "volume", "amount", "turnover", "change_pct"]
        df = df[[c for c in cols if c in df.columns]]

        df.to_csv(cache_path, index=False)
        return df

    except Exception as e:
        print(f"  ⚠ 获取 {symbol} 失败: {e}")
        return None


def get_all_us_stocks_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """批量获取美股数据"""
    ensure_cache_dir()
    result = {}
    total = len(tickers)
    print(f"[Data] 下载 {total} 只美股日线数据（{START_DATE} ~ {END_DATE}）...")

    for i, sym in enumerate(tickers):
        df = get_us_stock_daily(sym)
        if df is not None and len(df) > 100:
            result[sym] = df
        if (i + 1) % max(1, total // 10) == 0:
            print(f"  ... {i+1}/{total} (有效 {len(result)})")

    print(f"[Data] 美股数据下载完成: {len(result)} 只有效股票")
    return result


def get_us_benchmark(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """获取标普500指数作为基准"""
    cache_path = os.path.join(DATA_CACHE_DIR, "benchmark_sp500.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, parse_dates=["date"])

    start_fmt = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
    end_fmt = f"{end[:4]}-{end[4:6]}-{end[6:8]}"

    try:
        sp500 = yf.Ticker("^GSPC")
        df = sp500.history(start=start_fmt, end=end_fmt)
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df[["date", "close"]]
        df.to_csv(cache_path, index=False)
        print(f"[Data] 标普500基准: {len(df)} 个交易日")
        return df
    except Exception as e:
        print(f"[Data] ⚠ 获取标普500失败: {e}")
        return None


# ============================================================
# 加密货币数据（yfinance，24/7 交易）
# ============================================================

CRYPTO_LIST = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "MATIC-USD",
    "LINK-USD", "UNI-USD", "ATOM-USD", "LTC-USD", "ETC-USD",
    "FIL-USD", "APT-USD", "ARB-USD", "OP-USD", "NEAR-USD",
    "INJ-USD", "TIA-USD", "SEI-USD", "SUI-USD", "RUNE-USD",
    "STX-USD", "IMX-USD", "GRT-USD", "SAND-USD", "MANA-USD",
    "AAVE-USD", "MKR-USD", "SNX-USD", "CRV-USD", "COMP-USD",
    "ALGO-USD", "FTM-USD", "EGLD-USD", "FLOW-USD", "AXS-USD",
    "GALA-USD", "KAVA-USD", "ICP-USD", "QNT-USD", "RPL-USD",
    "LDO-USD", "ENS-USD", "PENDLE-USD", "CFX-USD", "MASK-USD",
]


def get_crypto_pool(top_n: int = 50) -> list[str]:
    """获取加密货币列表（按市值排序的前 N 个）"""
    tickers = CRYPTO_LIST[:top_n]
    print(f"[Data] 加密货币池: {len(tickers)} 个币种")
    return tickers


def get_crypto_benchmark(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame | None:
    """获取 BTC 作为加密货币基准"""
    cache_path = os.path.join(DATA_CACHE_DIR, "benchmark_btc.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, parse_dates=["date"])

    start_fmt = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
    end_fmt = f"{end[:4]}-{end[4:6]}-{end[6:8]}"

    try:
        btc = yf.Ticker("BTC-USD")
        df = btc.history(start=start_fmt, end=end_fmt)
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df[["date", "close"]]
        df.to_csv(cache_path, index=False)
        print(f"[Data] BTC基准: {len(df)} 个交易日")
        return df
    except Exception as e:
        print(f"[Data] ⚠ 获取BTC失败: {e}")
        return None
