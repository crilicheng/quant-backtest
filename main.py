"""
多因子选股回测 —— 主入口

用法:
    python main.py           # A股真实数据（需要东方财富网络）
    python main.py --us      # 美股真实数据（标普500，全球能用）
    python main.py --fake    # 模拟数据（不需要网络，验证逻辑）
    python main.py --quick   # 快速模式（50只股票）
    可以组合：python main.py --us --quick
"""

import argparse
import time

from data_loader import (
    # A股
    get_stock_pool, get_all_stocks_data, get_benchmark_data,
    # 美股
    get_us_stock_pool, get_all_us_stocks_data, get_us_benchmark,
    # 模拟
    generate_fake_data,
)
from backtest import backtest
from analysis import calculate_stats, plot_results


def main(quick: bool = False, fake: bool = False, us: bool = False):
    t0 = time.time()
    pool_size = 50 if quick else 100  # 美股100只，A股300只
    if not us:
        pool_size = 50 if quick else 300

    # ============================================================
    # 1. 获取数据
    # ============================================================
    if fake:
        print("[Main] 模拟数据模式")
        stock_data, benchmark = generate_fake_data(n_stocks=pool_size, seed=42)

    elif us:
        print("[Main] 美股数据模式（yfinance → 标普500）")
        try:
            tickers = get_us_stock_pool(top_n=pool_size)
            stock_data = get_all_us_stocks_data(tickers)
            benchmark = get_us_benchmark()
        except Exception as e:
            print(f"\n[Main] ⚠ 美股数据获取失败: {e}")
            print("[Main] 自动切换到模拟数据...")
            stock_data, benchmark = generate_fake_data(n_stocks=pool_size, seed=42)

    else:
        print("[Main] A股数据模式（AkShare → 沪深300）")
        try:
            stocks = get_stock_pool(top_n=pool_size)
            stock_data = get_all_stocks_data(stocks)
            benchmark = get_benchmark_data()
        except Exception as e:
            print(f"\n[Main] ⚠ A股数据获取失败: {e}")
            print("[Main] 自动切换到模拟数据...")
            stock_data, benchmark = generate_fake_data(n_stocks=pool_size, seed=42)

    if len(stock_data) < 20:
        print("[Error] 有效股票数不足。")
        return

    # ============================================================
    # 2. 运行回测
    # ============================================================
    print(f"\n[Main] 开始回测（{len(stock_data)} 只股票）...")
    result = backtest(stock_data, benchmark)
    nav = result["nav"]
    trades = result["trades"]

    # ============================================================
    # 3. 绩效 & 可视化
    # ============================================================
    stats = calculate_stats(nav)
    plot_results(nav, stats, save_path="backtest_result.png")

    if len(trades) > 0:
        print(f"\n{'='*70}")
        print(f"  最近 5 期调仓记录")
        print(f"{'='*70}")
        for _, row in trades.tail(5).iterrows():
            print(f"  {row['date'].strftime('%Y-%m-%d')}: "
                  f"持仓 {row['num_stocks']} 只")

    elapsed = time.time() - t0
    print(f"\n[Total] 总耗时: {elapsed:.1f}s")

    # ============================================================
    # 4. 保存
    # ============================================================
    nav.to_csv("backtest_nav.csv")
    trades.to_csv("backtest_trades.csv", index=False)
    print("[Total] 净值 → backtest_nav.csv")
    print("[Total] 调仓 → backtest_trades.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多因子选股回测系统")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式：减少股票数量，跑得快")
    parser.add_argument("--fake", action="store_true",
                        help="模拟数据（不联网也能跑）")
    parser.add_argument("--us", action="store_true",
                        help="美股模式（标普500，yfinance数据源）")
    args = parser.parse_args()
    main(quick=args.quick, fake=args.fake, us=args.us)
