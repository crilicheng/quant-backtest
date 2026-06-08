"""
因子计算模块
每个因子是一个函数：输入股票日线 DataFrame，输出截面因子值 Series

因子列表:
    1. momentum_20d    — 20日动量（过去20个交易日的涨跌幅）
    2. volatility_20d  — 20日波动率（日收益率标准差，年化）
    3. turnover_20d    — 20日平均换手率
    4. volume_ratio_5d — 5日量比（近5日均量 / 近20日均量）
    5. rsi_14d         — 14日相对强弱指标
    6. size            — 流通市值对数（ln(close * volume 中位数近似)）

设计原则:
    - 每个因子函数独立，方便增删
    - 截面标准化采用 z-score（减均值除以标准差）
    - 异常值处理：3-sigma 截尾
"""

import pandas as pd
import numpy as np

from config import FACTOR_WEIGHTS


# ============================================================
# 单因子函数
# ============================================================

def momentum_20d(df: pd.DataFrame) -> float:
    """
    20日动量：过去20个交易日涨跌幅（百分比）
    df 必须至少包含 20 行
    """
    if len(df) < 20:
        return np.nan
    return (df["close"].iloc[-1] / df["close"].iloc[-20] - 1) * 100


def volatility_20d(df: pd.DataFrame) -> float:
    """
    20日年化波动率（%）
    计算日收益率标准差，年化 × sqrt(252)
    """
    if len(df) < 20:
        return np.nan
    daily_returns = df["close"].pct_change().dropna().tail(20)
    return daily_returns.std() * np.sqrt(252) * 100


def turnover_20d(df: pd.DataFrame) -> float:
    """20日平均换手率（%）"""
    if len(df) < 20 or "turnover" not in df.columns:
        return np.nan
    return df["turnover"].tail(20).mean()


def volume_ratio_5d(df: pd.DataFrame) -> float:
    """
    5日量比 = 近5日均量 / 近20日均量
    大于1表示近期放量
    """
    if len(df) < 20 or "volume" not in df.columns:
        return np.nan
    vol_5 = df["volume"].tail(5).mean()
    vol_20 = df["volume"].tail(20).mean()
    if vol_20 == 0:
        return np.nan
    return vol_5 / vol_20


def rsi_14d(df: pd.DataFrame) -> float:
    """
    14日 RSI（相对强弱指标，0-100）
    RSI = 100 - 100/(1 + RS)，RS = 14日平均涨幅 / 14日平均跌幅
    """
    if len(df) < 15:
        return np.nan
    delta = df["close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.tail(14).mean()
    avg_loss = losses.tail(14).mean()

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def size_factor(df: pd.DataFrame) -> float:
    """
    规模因子：对数流通市值
    用 close * volume 的均值的对数近似流通市值
    """
    if len(df) < 20:
        return np.nan
    # 近期日均成交额 * 换手率倒推（粗略估计）
    avg_amount = df["amount"].tail(20).mean()
    return np.log(max(avg_amount, 1e6))


# ============================================================
# 因子注册表（加新因子只需在这里加一行）
# ============================================================
FACTOR_FUNCTIONS = {
    "momentum_20d": momentum_20d,
    "volatility_20d": volatility_20d,
    "turnover_20d": turnover_20d,
    "volume_ratio_5d": volume_ratio_5d,
    "rsi_14d": rsi_14d,
    "size": size_factor,
}


# ============================================================
# 批量计算 + 标准化 + 合成
# ============================================================

def winsorize(series: pd.Series, n_sigma: float = 3.0) -> pd.Series:
    """
    3-sigma 截尾：将超出均值 ± n_sigma 倍标准差的值替换为边界值。
    防止极端值扭曲标准化。
    """
    lower = series.mean() - n_sigma * series.std()
    upper = series.mean() + n_sigma * series.std()
    return series.clip(lower, upper)


def calculate_all_factors(
    stock_data: dict[str, pd.DataFrame],
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    在指定日期，计算所有股票的所有因子值。

    参数:
        stock_data: {symbol: DataFrame} 每个股票的日线数据
        as_of_date: 截面日期（调仓日）

    返回:
        DataFrame: index=symbol, columns=因子名
    """
    records = []

    for symbol, df in stock_data.items():
        # 截至 as_of_date 的数据
        df_cut = df[df["date"] <= as_of_date]
        if len(df_cut) < 50:  # 至少需要50个交易日的历史数据
            continue

        record = {"symbol": symbol}
        for name, func in FACTOR_FUNCTIONS.items():
            try:
                record[name] = func(df_cut)
            except Exception:
                record[name] = np.nan
        records.append(record)

    if len(records) == 0:
        return pd.DataFrame()

    factor_df = pd.DataFrame(records).set_index("symbol")

    # 删除有任何因子为空的股票
    factor_df = factor_df.dropna()

    return factor_df


def normalize_factors(factor_df: pd.DataFrame) -> pd.DataFrame:
    """
    截面标准化：每个因子先 winsorize 再 z-score 标准化。
    z-score = (raw - cross_sectional_mean) / cross_sectional_std

    标准化后所有因子在同一尺度，可以线性组合。
    """
    normalized = pd.DataFrame(index=factor_df.index)

    for col in factor_df.columns:
        raw = factor_df[col]
        # 3-sigma 截尾
        winsorized = winsorize(raw, n_sigma=3.0)
        # z-score 标准化
        if winsorized.std() > 1e-10:
            normalized[col] = (winsorized - winsorized.mean()) / winsorized.std()
        else:
            normalized[col] = 0.0

    return normalized


def composite_score(normalized_df: pd.DataFrame) -> pd.Series:
    """
    合成打分：各因子 z-score 加权求和。
    权重从 config.py 的 FACTOR_WEIGHTS 读取。

    负向因子（如波动率）权重为负 → 低波股票得分高。
    """
    scores = pd.Series(0.0, index=normalized_df.index)

    for factor_name, weight in FACTOR_WEIGHTS.items():
        if factor_name in normalized_df.columns:
            scores += normalized_df[factor_name] * weight

    return scores.sort_values(ascending=False)


def print_factor_summary(factor_df: pd.DataFrame, scores: pd.Series):
    """
    打印因子截面统计信息，帮助理解当前截面发生了什么。
    """
    print(f"  [Factor] 有效股票数: {len(factor_df)}")
    print(f"  [Factor] 得分前5: {list(scores.head(5).index)}")
    print(f"  [Factor] 得分前5: {[f'{s:.3f}' for s in scores.head(5).values]}")
    print(f"  [Factor] 得分后5: {list(scores.tail(5).index)}")
    # 每个因子的截面统计
    print(f"  [Factor] 各因子截面均值/std:")
    for col in factor_df.columns:
        print(f"    {col:20s}: mean={factor_df[col].mean():+.3f}, std={factor_df[col].std():.3f}")
