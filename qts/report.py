"""每日报告生成：Markdown + 净值曲线图。"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import strategies as st
from .broker import equity_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(ROOT, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei",
                                         "Droid Sans Fallback", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


def signal_snapshot(data, cfg, date):
    """每票的当日信号明细，用于报告。"""
    sc = cfg["strategies"]
    mom = st.sig_momentum(data, sc["momentum"]["lookback"],
                         sc["momentum"]["top_n"], sc["momentum"]["rebalance_days"])
    closes = pd.DataFrame({s: df["close"] for s, df in data.items()}).sort_index()
    mom_rank = closes.pct_change(sc["momentum"]["lookback"]).rank(axis=1,
                                                                ascending=False)
    rows = {}
    for s, df in data.items():
        if date not in df.index:
            continue
        trend = float(st.sig_trend(df, sc["trend"]["fast"], sc["trend"]["slow"]).loc[date])
        r2 = float(st.rsi(df["close"], sc["reversion"]["rsi_n"]).loc[date])
        rank = float(mom_rank.loc[date, s]) if date in mom_rank.index else float("nan")
        mw = float(mom.loc[date, s]) if date in mom.index else 0.0
        rows[s] = {"trend": trend, "rsi2": round(r2, 1),
                  "mom_rank": rank, "mom_w": mw}
    return rows


def plot_paper(state, path):
    eh = state["equity_history"]
    if len(eh) < 2:
        return None
    eq = pd.Series({r["date"]: r["equity"] for r in eh})
    eq.index = pd.to_datetime(eq.index)
    base = state["initial_capital"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(eq.index, eq / base, lw=1.6, label="虚拟盘净值")
    ax.axhline(1.0, color="grey", ls="--", lw=0.8)
    dd = eq / eq.cummax() - 1
    ax2 = ax.twinx()
    ax2.fill_between(dd.index, dd * 100, 0, color="red", alpha=0.15)
    ax2.set_ylabel("回撤 %", color="red")
    ax.set_ylabel("净值（初始=1）")
    ax.set_title("quant-cn 虚拟盘净值")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def daily_report(data, cfg, fx, bench_cn, bench_hk, ai_note=None):
    state = json_load()
    if not state:
        return None
    date = state["last_date"]
    d = pd.Timestamp(date)
    market_of = {it["symbol"]: it["market"] for it in cfg["universe"]}
    name_of = {it["symbol"]: it["name"] for it in cfg["universe"]}
    prices = {}
    for s, df in data.items():
        if d in df.index:
            p = float(df.loc[d, "close"])
            prices[s] = p * fx if market_of[s] == "HK" else p
    from .engine import positions_marked
    rows = positions_marked(state, prices, market_of, fx, name_of)
    eq = state["equity_history"][-1]["equity"]
    base = state["initial_capital"]
    prev_eq = (state["equity_history"][-2]["equity"]
              if len(state["equity_history"]) > 1 else base)
    day_ret = eq / prev_eq - 1
    total_ret = eq / base - 1
    todays = [t for t in state["trades"] if t["date"] == date]
    sigs = signal_snapshot(data, cfg, d)

    def pct(x):
        return f"{x * 100:+.2f}%"

    def bchg(b):
        try:
            if b is None or d not in b.index:
                return "—"
            prior = b[b.index < d]
            if prior.empty:
                return "—"
            return pct(b.loc[d] / prior.iloc[-1] - 1)
        except Exception:
            return "—"

    lines = [f"# quant-cn 虚拟盘日报 · {date}", ""]
    lines += [f"**账户**：净值 ¥{eq:,.0f}（初始 ¥{base:,.0f}，累计 {pct(total_ret)}，"
              f"当日 {pct(day_ret)}）｜现金 ¥{state['cash']:,.0f}｜汇率 1HKD={fx:.4f}CNY",
              f"**基准当日**：沪深300 {bchg(bench_cn)} ｜ 恒生指数 {bchg(bench_hk)}", ""]
    lines += ["## 当前持仓", ""]
    if rows:
        lines += ["| 标的 | 市场 | 数量 | 成本(CNY) | 现价(CNY) | 市值(CNY) | 盈亏% | 趋势 | RSI2 | 动量排名 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            sg = sigs.get(r["symbol"], {})
            tr = "↑多" if sg.get("trend") else "—"
            lines.append(f"| {r['name']}({r['symbol']}) | {r['market']} | {r['qty']} "
                        f"| {r['avg_cost']} | {r['price']} | {r['mv_cny']:,.0f} "
                        f"| {r['pnl_pct']:+.2f}% | {tr} | {sg.get('rsi2','—')} "
                        f"| {int(sg['mom_rank']) if sg.get('mom_rank') == sg.get('mom_rank') else '—'} |")
    else:
        lines.append("（空仓）")
    lines += ["", "## 今日成交", ""]
    if todays:
        lines += ["| 方向 | 标的 | 数量 | 价格(CNY) | 费用(CNY) |", "|---|---|---|---|---|"]
        for o in todays:
            nm = name_of.get(o["sym"], o["sym"])
            lines.append(f"| {o['side']} | {nm}({o['sym']}) | {o['qty']} "
                        f"| {o['price']:.3f} | {o['fee']:.2f} |")
    else:
        lines.append("（无调仓）")
    lines += ["", "## 全池信号", "",
              "| 标的 | 市场 | 趋势(20/60) | RSI2 | 20日动量排名 | 动量权重 |",
              "|---|---|---|---|---|---|"]
    for it in cfg["universe"]:
        s = it["symbol"]
        sg = sigs.get(s)
        if not sg:
            continue
        rank = sg["mom_rank"]
        rank_s = f"{int(rank)}/{len(cfg['universe'])}" if rank == rank else "—"
        lines.append(f"| {it['name']}({s}) | {it['market']} "
                     f"| {'🟢 多头' if sg['trend'] else '🔴 空头'} | {sg['rsi2']} "
                     f"| {rank_s} | {sg['mom_w']:.0%} |")
    if ai_note:
        lines += ["", "## AI 点评（vibe-trading）", "", ai_note]
    lines += ["", "> 本报告由 quant-cn 自动生成，虚拟盘模拟，不构成投资建议。"]
    md = "\n".join(lines)
    path = os.path.join(REPORT_DIR, f"daily_{date}.md")
    with open(path, "w") as f:
        f.write(md)
    plot_paper(state, os.path.join(REPORT_DIR, f"equity_{date}.png"))
    return path


def json_load():
    import json
    from .engine import STATE_FILE
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return None
