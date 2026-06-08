"""
多因子选股回测 —— 主入口

用法:
    python main.py           # 真实数据回测（需要 A 股网络环境）
    python main.py --fake    # 模拟数据（任何环境都能跑，验证逻辑）
    python main.py --quick   # 快速模式（50只股票，验证流程）
"""

import argparse
import time
import traceback

from data_loader import (
    get_stock_pool, get_all_stocks_data, get_benchmark_data,
    generate_fake_data,
)
from backtest import backtest
from analysis import calculate_stats, plot_results


def main(quick: bool = False, fake: bool = False):
    t0 = time.time()

    # ============================================================
    # 1. 获取股票池
    # ============================================================
    pool_size = 50 if quick else 300

    if fake:
        print("[Main] 使用模拟数据模式（用于测试回测逻辑）")
        stock_data, benchmark = generate_fake_data(
            n_stocks=pool_size, seed=42
        )
    else:
        try:
            stocks = get_stock_pool(top_n=pool_size)
            stock_data = get_all_stocks_data(stocks)
            benchmark = get_benchmark_data()
        except Exception as e:
            print(f"\n[Main] ⚠ 真实数据获取失败: {e}")
            print(f"[Main] 自动切换到模拟数据模式...\n")
            print("[Main] 提示: 如果你在北京，一般直接跑就行，AkShare 能访问东方财富。")
            print("[Main] 如果在海外/网络受限，就用 python main.py --fake\n")
            stock_data, benchmark = generate_fake_data(
                n_stocks=pool_size, seed=42
            )

    if len(stock_data) < 20:
        print("[Error] 有效股票数不足。")
        return

    # ============================================================
    # 2. 运行回测
    # ============================================================
    print(f"\n[Main] 开始回测...")
    result = backtest(stock_data, benchmark)
    nav = result["nav"]
    trades = result["trades"]

    # ============================================================
    # 3. 计算绩效 & 可视化
    # ============================================================
    stats = calculate_stats(nav)
    plot_results(nav, stats, save_path="backtest_result.png")

    # 打印最近几期调仓记录
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
    # 4. 保存结果
    # ============================================================
    nav.to_csv("backtest_nav.csv")
    trades.to_csv("backtest_trades.csv", index=False)
    print("[Total] 净值数据已保存至 backtest_nav.csv")
    print("[Total] 调仓记录已保存至 backtest_trades.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多因子选股回测系统")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式：只用50只股票验证流程")
    parser.add_argument("--fake", action="store_true",
                        help="使用模拟数据（不需要网络，只验证逻辑）")
    args = parser.parse_args()
    main(quick=args.quick, fake=args.fake)
