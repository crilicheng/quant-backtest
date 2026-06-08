"""
回测引擎
实现月度调仓的多因子选股策略回测

核心逻辑:
    每月最后一个交易日：
        1. 计算所有股票因子值
        2. 截面标准化 + 加权合成打分
        3. 选得分最高的 N 只股票等权持有
        4. 持有到下月调仓日，扣除交易成本

输出:
    - 每日组合净值序列
    - 调仓记录
    - 基准净值序列（用于对比）
"""

import pandas as pd
import numpy as np
from datetime import datetime

from config import (
    START_DATE, END_DATE, HOLDING_COUNT,
    REBALANCE_FREQ, ROUND_TRIP_COST, BENCHMARK_INDEX,
)
from factors import calculate_all_factors, normalize_factors, composite_score, print_factor_summary


def generate_rebalance_dates(
    start: str, end: str, freq: str = "monthly"
) -> list[pd.Timestamp]:
    """
    生成调仓日列表。
    每月最后一个交易日 → 用 pandas 的月末频率。
    """
    dates = pd.date_range(start=start, end=end, freq="ME")  # ME = month end
    return sorted(dates.tolist())


def get_next_trading_day(date: pd.Timestamp, all_dates: pd.DatetimeIndex) -> pd.Timestamp | None:
    """
    获取 date 之后的下一个交易日。
    用于确定持仓的起始日（调仓日的下一个交易日开盘买入）。
    """
    future = all_dates[all_dates > date]
    if len(future) == 0:
        return None
    return future[0]


def backtest(
    stock_data: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame | None = None,
) -> dict:
    """
    运行多因子选股回测。

    参数:
        stock_data: {symbol: DataFrame} 股票日线数据
        benchmark: 基准日线数据 (columns: date, close)

    返回:
        dict with:
            - nav: DataFrame (date, portfolio_value, benchmark_value)
            - trades: DataFrame 调仓记录
            - stats: dict 绩效统计
    """
    # ---- 准备交易日历 ----
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df["date"].tolist())
    all_dates = pd.DatetimeIndex(sorted(all_dates))

    rebalance_dates = generate_rebalance_dates(START_DATE, END_DATE, REBALANCE_FREQ)
    # 只保留有数据的调仓日
    rebalance_dates = [d for d in rebalance_dates if d in all_dates]
    print(f"[Backtest] 共 {len(rebalance_dates)} 个调仓日")

    # ---- 初始化 ----
    cash = 1_000_000.0          # 初始资金 100 万
    position = {}               # {symbol: shares} 当前持仓
    portfolio_values = []       # 每日净值记录
    trade_log = []              # 调仓记录

    benchmark_init = None
    if benchmark is not None and len(benchmark) > 0:
        benchmark_init = benchmark["close"].iloc[0]

    # ---- 逐日模拟 ----
    for i, today in enumerate(all_dates):
        # 今天是否调仓日
        if today in rebalance_dates:
            # === 1. 计算因子，选股 ===
            factor_df = calculate_all_factors(stock_data, today)

            if len(factor_df) >= HOLDING_COUNT:
                normalized = normalize_factors(factor_df)
                scores = composite_score(normalized)
                selected = scores.head(HOLDING_COUNT).index.tolist()

                if i == 0 or (i > 0 and len(trade_log) <= 3):
                    print_factor_summary(factor_df, scores)

                # === 2. 调仓: 清空旧仓位，买入新股票 ===
                # 2a. 卖出全部旧持仓
                sell_value = 0.0
                for sym in list(position.keys()):
                    if sym in stock_data:
                        df_sym = stock_data[sym]
                        today_prices = df_sym[df_sym["date"] == today]
                        if len(today_prices) > 0:
                            sell_price = today_prices["close"].iloc[0]
                            sell_value += position[sym] * sell_price
                    del position[sym]  # 全部清仓

                # 卖出成本: 印花税(0.1%) + 佣金(0.03%) + 滑点
                sell_cost = sell_value * (ROUND_TRIP_COST / 2)
                cash += sell_value - sell_cost

                # 2b. 等权买入新选中的股票
                per_stock_cash = cash / HOLDING_COUNT
                total_buy_value = 0.0
                for sym in selected:
                    if sym in stock_data:
                        df_sym = stock_data[sym]
                        today_prices = df_sym[df_sym["date"] == today]
                        if len(today_prices) > 0:
                            buy_price = today_prices["close"].iloc[0]
                            # 整数手（100股 = 1手），确保不超过分配金额
                            target_value = per_stock_cash
                            shares = int(target_value / buy_price / 100) * 100
                            if shares > 0:
                                position[sym] = shares
                                total_buy_value += shares * buy_price

                # 买入成本: 佣金(0.03%) + 滑点（印花税卖出时才扣）
                buy_cost = total_buy_value * (ROUND_TRIP_COST / 2)
                # 实际现金变动: 扣除买入本金 + 买入手续费
                cash -= (total_buy_value + buy_cost)

                # 剩余现金 = 未花完的钱（因为取整手产生的零头）
                residual = cash

                trade_log.append({
                    "date": today,
                    "stocks": selected,
                    "num_stocks": len(selected),
                })

        # === 3. 计算当日组合净值 ===
        portfolio_value = 0.0
        used_cash = 0.0
        for sym, shares in position.items():
            if sym in stock_data:
                df_sym = stock_data[sym]
                today_prices = df_sym[df_sym["date"] == today]
                if len(today_prices) > 0:
                    price = today_prices["close"].iloc[0]
                    portfolio_value += shares * price
                    used_cash += shares * price

        total_value = portfolio_value + cash

        # 基准净值
        benchmark_value = np.nan
        if benchmark is not None and benchmark_init:
            bench_row = benchmark[benchmark["date"] == today]
            if len(bench_row) > 0:
                benchmark_value = bench_row["close"].iloc[0] / benchmark_init

        portfolio_values.append({
            "date": today,
            "portfolio_value": total_value,
            "benchmark_value": benchmark_value,
        })

    # ---- 整理结果 ----
    nav_df = pd.DataFrame(portfolio_values)
    nav_df["portfolio_nav"] = nav_df["portfolio_value"] / 1_000_000.0
    nav_df["benchmark_nav"] = nav_df["benchmark_value"]
    nav_df = nav_df.set_index("date")

    trade_df = pd.DataFrame(trade_log)

    print(f"[Backtest] 回测完成")
    print(f"[Backtest] 初始资金: ¥1,000,000")
    print(f"[Backtest] 最终净值: ¥{nav_df['portfolio_value'].iloc[-1]:,.0f}")
    print(f"[Backtest] 总换手次数: {len(trade_log)}")

    return {
        "nav": nav_df,
        "trades": trade_df,
    }
