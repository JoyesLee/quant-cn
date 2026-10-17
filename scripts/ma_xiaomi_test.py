"""小米集团(01810.HK) 双均线策略显著性检验。

方法：
1. 全历史前复权日线（2018-07 上市 ~ 最新），数据源 akshare 新浪。
2. 参数网格：fast ∈ 5..30, slow ∈ 20..150（fast<slow），共 51 组。
3. 撮合：信号收盘计算、次日收盘成交（pos = sig.shift(1)），扣港股真实费率
   （买入≈0.06%，卖出≈0.16%，含佣金+印花税+交易/交收费）。
4. 显著性检验：
   - t 检验（配对）：策略日收益 - 买入持有日收益，H0: 均值≤0
   - t 检验（绝对）：策略日收益均值 / 标准误 × √n（即夏普显著性）
   - 稳健性：看网格中显著组合的占比与分布，而非单点最优
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import akshare as ak

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports")
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei",
                                         "Droid Sans Fallback", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

FEE_BUY, FEE_SELL = 0.0006, 0.0016  # 港股单边（CNY 无关，按 HKD 计）


def load_xiaomi():
    cache = os.path.join(ROOT, "data", "HK_01810_full.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache, parse_dates=["date"], index_col="date")
    else:
        raw = ak.stock_hk_daily(symbol="01810", adjust="qfq")
        df = raw.rename(columns={"date": "date"}).copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date").astype(float)
        df.to_csv(cache)
    return df


def bt_pair(df, fast, slow, rf=0.03):
    close = df["close"]
    r = close.pct_change().fillna(0.0)
    sig = (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(float)
    pos = sig.shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(0.0)
    fee = turn * np.where(turn * (pos - pos.shift(1)) >= 0, FEE_BUY, FEE_SELL)
    # 更精确：按方向分别计费
    d_pos = pos.diff().fillna(pos)
    fee = d_pos.clip(lower=0) * FEE_BUY + (-d_pos).clip(lower=0) * FEE_SELL
    strat = pos * r - fee
    bh = r.copy()
    n = len(strat)
    years = n / 244.0

    def cagr(x):
        base = (1 + x).prod()
        return base ** (1 / years) - 1
    def sharpe(x):
        return (x.mean() * 244 - rf) / (x.std() * np.sqrt(244) + 1e-12)
    def maxdd(x):
        eq = (1 + x).cumprod()
        return (eq / eq.cummax() - 1).min()

    diff = (strat - bh).dropna()
    t_rel, p_rel = stats.ttest_rel(strat, bh)          # vs 买入持有
    t_abs = strat.mean() / (strat.std() / np.sqrt(n))  # vs 0（夏普t）
    # 超额收益bootstrap 1000次，看 P(超额年化<=0)
    rng = np.random.default_rng(42)
    ex = (strat - bh).values
    boot = [rng.choice(ex, size=n, replace=True).mean() * 244 for _ in range(1000)]
    p_boot = float(np.mean(np.array(boot) <= 0))
    return {
        "fast": fast, "slow": slow,
        "strat_cagr": cagr(strat), "bh_cagr": cagr(bh),
        "excess_cagr": cagr(strat) - cagr(bh),
        "sharpe": sharpe(strat), "bh_sharpe": sharpe(bh),
        "maxdd": maxdd(strat), "bh_maxdd": maxdd(bh),
        "t_vs_bh": float(t_rel), "p_vs_bh": float(p_rel),
        "t_abs": float(t_abs),
        "p_boot": p_boot,
        "trades": int((d_pos.abs() > 0).sum()),
        "in_market": float(pos.mean()),
        "strat_ret": strat, "bh_ret": bh,
    }


def main():
    df = load_xiaomi()
    print(f"小米集团(01810) 前复权日线: {df.index[0].date()} ~ {df.index[-1].date()}, "
          f"{len(df)} 个交易日 (~{len(df)/244:.1f} 年)")
    bh_total = df['close'].iloc[-1] / df['close'].iloc[0] - 1
    print(f"买入持有总收益: {bh_total:+.1%}\n")

    grid = []
    for fast in [5, 10, 15, 20, 25, 30]:
        for slow in [20, 30, 40, 60, 80, 100, 120, 150]:
            if fast < slow:
                grid.append(bt_pair(df, fast, slow))
    g = pd.DataFrame([{k: v for k, v in r.items() if k not in
                     ("strat_ret", "bh_ret")} for r in grid])

    sig5 = g[g["p_vs_bh"] < 0.05]
    sig10 = g[g["p_vs_bh"] < 0.10]
    print(f"参数网格 {len(g)} 组：p<0.05 跑赢持有 {len(sig5)} 组 "
          f"({len(sig5)/len(g):.0%})，p<0.10 {len(sig10)} 组")
    print()
    cols = ["fast", "slow", "strat_cagr", "bh_cagr", "excess_cagr", "sharpe",
            "maxdd", "t_vs_bh", "p_vs_bh", "t_abs", "p_boot", "trades", "in_market"]
    show = g.sort_values("excess_cagr", ascending=False)[cols].copy()
    for c in ["strat_cagr", "bh_cagr", "excess_cagr"]:
        show[c] = show[c].map(lambda x: f"{x:+.1%}")
    for c in ["sharpe", "t_vs_bh", "t_abs"]:
        show[c] = show[c].map(lambda x: f"{x:.2f}")
    for c in ["p_vs_bh", "p_boot"]:
        show[c] = show[c].map(lambda x: f"{x:.3f}")
    show["maxdd"] = show["maxdd"].map(lambda x: f"{x:.1%}")
    show["in_market"] = show["in_market"].map(lambda x: f"{x:.0%}")
    print(show.to_string(index=False))

    # 经典参数重点展示
    print("\n== 经典参数组合 ==")
    for fast, slow in [(5, 20), (10, 30), (20, 60), (50, 200)]:
        if fast == 50:
            r = bt_pair(df, 25, 120)  # 网格上限近似
            note = "(用25/120近似50/200)"
        else:
            r = bt_pair(df, fast, slow)
            note = ""
        print(f"MA{fast}/{slow}{note}: 年化 {r['strat_cagr']:+.1%} vs 持有 "
              f"{r['bh_cagr']:+.1%} | 超额 t={r['t_vs_bh']:.2f} "
              f"p={r['p_vs_bh']:.3f} | 夏普 {r['sharpe']:.2f} | "
              f"回撤 {r['maxdd']:.1%} | 交易 {r['trades']} 次")

    # 热力图：超额年化 + 显著性标记
    fasts = sorted(g["fast"].unique()); slows = sorted(g["slow"].unique())
    piv = g.pivot(index="fast", columns="slow", values="excess_cagr")
    pivp = g.pivot(index="fast", columns="slow", values="p_vs_bh")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(piv.values * 100, cmap="RdYlGn", aspect="auto")
    for i, f in enumerate(fasts):
        for j, s in enumerate(slows):
            p = pivp.loc[f, s]
            mark = "**" if p < 0.05 else ("*" if p < 0.10 else "")
            ax.text(j, i, f"{piv.loc[f, s]*100:+.0f}{mark}", ha="center",
                    va="center", fontsize=9)
    ax.set_xticks(range(len(slows)), slows)
    ax.set_yticks(range(len(fasts)), fasts)
    ax.set_xlabel("慢线 MA")
    ax.set_ylabel("快线 MA")
    ax.set_title("小米集团(01810) 双均线 vs 买入持有：超额年化% （**=p<0.05, *=p<0.10）")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "xiaomi_ma_heatmap.png"), dpi=110)

    g.to_csv(os.path.join(OUT, "xiaomi_ma_grid.csv"), index=False)
    print(f"\n[out] {OUT}/xiaomi_ma_heatmap.png, xiaomi_ma_grid.csv")


if __name__ == "__main__":
    main()
