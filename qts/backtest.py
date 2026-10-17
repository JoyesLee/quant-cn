"""回测引擎：信号收盘计算 → 次日收盘成交，含手续费与整手约束。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import portfolio as pf
from .broker import rebalance_to_targets, equity_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "reports")
os.makedirs(OUT_DIR, exist_ok=True)

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei",
                                         "Droid Sans Fallback", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


def metrics(equity: pd.Series, rf: float = 0.02) -> dict:
    eq = equity.dropna()
    if len(eq) < 20:
        return {}
    ret = eq.pct_change().dropna()
    n = len(eq)
    years = n / 244
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / max(years, 1e-9)) - 1
    vol = ret.std() * np.sqrt(244)
    sharpe = (ret.mean() * 244 - rf) / (ret.std() * np.sqrt(244) + 1e-12)
    dd = eq / eq.cummax() - 1
    maxdd = dd.min()
    win = (ret > 0).mean()
    return {"total_return": total, "cagr": cagr, "vol": vol, "sharpe": sharpe,
            "max_drawdown": maxdd, "calmar": (cagr / abs(maxdd)) if maxdd else np.nan,
            "day_win_rate": win, "days": n}


def run_backtest(data, cfg, fx, start=None, bench=None, label="composite",
                sleeve=None):
    """sleeve: None=三策略合成；'trend'/'reversion'/'momentum'=单策略消融。"""
    # 日历：所有标的日期并集
    cal = sorted(set().union(*[set(df.index) for df in data.values()]))
    if start:
        cal = [d for d in cal if d >= pd.Timestamp(start)]
    market_of = {it["symbol"]: it["market"] for it in cfg["universe"]}
    lot = cfg["lot_size"]

    cfg2 = {"risk": cfg["risk"], "fees": cfg["fees"], "lot_size": lot}
    if sleeve:
        cfg2["strategies"] = {k: dict(v) for k, v in cfg["strategies"].items()}
        for k in cfg2["strategies"]:
            if k != sleeve:
                cfg2["strategies"][k]["weight"] = 0.0
    else:
        cfg2["strategies"] = cfg["strategies"]

    sigs = pf.sleeve_signals(data, cfg2)
    state = {"cash": float(cfg["initial_capital"]), "positions": {}}
    eq_hist, orders_hist = [], []
    for i, d in enumerate(cal):
        if i == 0:
            continue
        prev = cal[i - 1]
        targets = pf.targets_at(sigs, prev, cfg2)
        prices = {}
        for s, df in data.items():
            if d in df.index:
                p = float(df.loc[d, "close"])
                prices[s] = p * fx if market_of[s] == "HK" else p
        ords = rebalance_to_targets(state, targets, prices, market_of, cfg2, fx,
                                   lot=lot)
        for o in ords:
            o["date"] = d
        orders_hist += ords
        eq_hist.append((d, equity_of(state, prices)))
    eq = pd.Series(dict(eq_hist), name=label)
    m = metrics(eq)
    out = {"label": label, "equity": eq, "metrics": m,
           "orders": pd.DataFrame(orders_hist)}
    if bench is not None:
        b = bench.reindex(eq.index).ffill()
        b = b / b.iloc[0] * cfg["initial_capital"]
        out["bench_equity"] = b
        out["bench_metrics"] = metrics(b)
    return out


def plot_result(results, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    for r in results:
        eq = r["equity"] / r["equity"].iloc[0]
        ax1.plot(eq.index, eq, label=r["label"], lw=1.4)
        dd = r["equity"] / r["equity"].cummax() - 1
        ax2.fill_between(dd.index, dd, 0, alpha=0.35, label=r["label"])
    if "bench_equity" in results[0]:
        b = results[0]["bench_equity"] / results[0]["bench_equity"].iloc[0]
        ax1.plot(b.index, b, label="基准", color="grey", lw=1.2, ls="--")
    ax1.set_ylabel("净值（初始=1）")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("回撤")
    ax2.grid(alpha=0.3)
    ax1.set_title("quant-cn 组合回测（A股+港股，含手续费，收盘撮合）")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
