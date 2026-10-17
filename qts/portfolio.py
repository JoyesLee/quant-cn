"""组合层：把三个策略子账户合成每个标的的目标权重（共享给回测与虚拟盘）。

sleeve 结构：trend 40% / reversion 30% / momentum 30%（可配）。
每个 sleeve 内部对其触发标的等权；单票总权重受 max_weight_per_stock 限制；
其余资金留现金。信号当日收盘生效（T+1 约束下当日买入次日方可卖出，
由于每日只调仓一次，天然满足）。
"""
import pandas as pd
from . import strategies as st


def sleeve_signals(data: dict, cfg: dict) -> dict:
    """返回 {'trend': {sym: sig_series}, 'reversion': {...}, 'momentum': DataFrame}"""
    scfg = cfg["strategies"]
    trend, rev = {}, {}
    for sym, df in data.items():
        trend[sym] = st.sig_trend(df, scfg["trend"]["fast"], scfg["trend"]["slow"])
        rev[sym] = st.sig_reversion(df, scfg["reversion"]["rsi_n"],
                                    scfg["reversion"]["buy_th"],
                                    scfg["reversion"]["exit_th"])
    mom = st.sig_momentum(data, scfg["momentum"]["lookback"],
                         scfg["momentum"]["top_n"], scfg["momentum"]["rebalance_days"])
    return {"trend": trend, "reversion": rev, "momentum": mom}


def targets_at(sigs: dict, date, cfg: dict) -> dict:
    """返回 {symbol: target_weight}，权重为占组合基准净值比例，和 <= 1。"""
    sc = cfg["strategies"]
    maxw = cfg["risk"]["max_weight_per_stock"]
    tw = {}
    # trend sleeve
    bulls = [s for s, ser in sigs["trend"].items()
             if date in ser.index and ser.loc[date] > 0.5]
    if bulls:
        w = min(sc["trend"]["weight"] / len(bulls), maxw)
        for s in bulls:
            tw[s] = tw.get(s, 0.0) + w
    # reversion sleeve
    oversold = [s for s, ser in sigs["reversion"].items()
               if date in ser.index and ser.loc[date] > 0.5]
    if oversold:
        w = min(sc["reversion"]["weight"] / len(oversold), maxw)
        for s in oversold:
            tw[s] = tw.get(s, 0.0) + w
    # momentum sleeve
    mom = sigs["momentum"]
    if date in mom.index:
        for s, w in mom.loc[date].items():
            if w > 0:
                tw[s] = min(tw.get(s, 0.0) + float(w), maxw)
    total = sum(tw.values())
    if total > 1.0:  # 理论上被 maxw 限制住，防御性归一
        tw = {s: w / total for s, w in tw.items()}
    return tw
