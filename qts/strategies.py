"""策略信号层：三类主流量化策略，全部向量化 pandas 实现。

1. trend      趋势跟踪（双均线）：SMA(fast) > SMA(slow) 持多，否则空仓。
              A股/港股均适用；捕捉中期行情，震荡市会被反复止损。
2. reversion  均值回归（Connors RSI2）：RSI(2)<buy_th 超卖买入，RSI(2)>exit_th 离场。
              在长期向上漂移的市场（如优质龙头股）胜率较高。
3. momentum   截面动量：每 rebalance_days 个交易日，按 lookback 日收益率对全池排序，
              做多前 top_n，等权。经典 CTA/股票多空逻辑（此处仅做多）。
"""
import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 2) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    # Wilder 平滑
    ru = up.ewm(alpha=1 / n, min_periods=n).mean()
    rd = dn.ewm(alpha=1 / n, min_periods=n).mean()
    rs = ru / rd.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # RSI(2) 极端情形：rd=0 时 RSI=100
    out = out.where(rd != 0, 100.0)
    return out


def sig_trend(df: pd.DataFrame, fast: int = 20, slow: int = 60) -> pd.Series:
    f = df["close"].rolling(fast).mean()
    s = df["close"].rolling(slow).mean()
    return (f > s).astype(float)


def sig_reversion(df: pd.DataFrame, n: int = 2, buy_th: float = 10,
                 exit_th: float = 70) -> pd.Series:
    r = rsi(df["close"], n)
    sig = pd.Series(np.nan, index=df.index)
    sig[r < buy_th] = 1.0
    sig[r > exit_th] = 0.0
    return sig.ffill().fillna(0.0)


def sig_momentum(data: dict, lookback: int = 20, top_n: int = 3,
                rebalance_days: int = 5) -> pd.DataFrame:
    """返回 DataFrame(index=交易日, columns=symbol)，值为该日各票的目标权重(和<=1)。
    仅在调仓日更新，非调仓日沿用前值（ffill）。"""
    closes = pd.DataFrame({s: df["close"] for s, df in data.items()}).sort_index()
    mom = closes.pct_change(lookback)
    rank = mom.rank(axis=1, ascending=False)
    keep = rank <= top_n
    w = keep.div(keep.sum(axis=1).replace(0, 1), axis=0).fillna(0.0)
    # 非调仓日冻结：把非调仓日置 NaN 再 ffill
    day_idx = closes.index
    is_rebal = np.arange(len(day_idx)) % rebalance_days == (len(day_idx) - 1) % rebalance_days
    w = w.where(pd.Series(is_rebal, index=day_idx))
    return w.ffill().fillna(0.0)
