"""quant-cn 命令行入口。

用法（用 vibe-trading 的 python 运行，或本目录 venv）：
  python main.py update                 # 刷新行情缓存/汇率
  python main.py backtest               # 组合 + 单策略消融回测
  python main.py trade                  # 执行今日虚拟盘调仓
  python main.py report [--ai]         # 生成日报（--ai 调 vibe-trading 点评）
  python main.py daily [--ai]          # update + trade + report
  python main.py status                # 查看虚拟盘状态
"""
import argparse, json, os, subprocess, sys, datetime as dt
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qts import data as D
from qts import backtest as B
from qts import engine as E

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_cfg():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)


def cmd_update(cfg, refresh=False):
    fx = D.fetch_fx(cfg["fx_fallback"])
    print(f"[fx] 1 HKD = {fx:.4f} CNY")
    data = D.load_universe(cfg["universe"], cfg["start_date"], refresh=refresh)
    for s, df in data.items():
        print(f"[data] {s}: {len(df)} bars, 最新 {df.index[-1].date()}")
    with open(os.path.join(ROOT, "state", "fx.json"), "w") as f:
        json.dump({"fx": fx, "ts": dt.datetime.now().isoformat()}, f)
    return data, fx


def get_fx(cfg):
    p = os.path.join(ROOT, "state", "fx.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        if d["ts"][:10] >= (dt.date.today() - dt.timedelta(days=3)).isoformat():
            return d["fx"]
    return D.fetch_fx(cfg["fx_fallback"])


def cmd_backtest(cfg):
    fx = get_fx(cfg)
    data = D.load_universe(cfg["universe"], cfg["start_date"])
    bench = D.fetch_benchmark(cfg["benchmarks"]["cn"], cfg["start_date"])
    results = []
    comp = B.run_backtest(data, cfg, fx, bench=bench, label="三策略合成")
    results.append(comp)
    for s in ["trend", "reversion", "momentum"]:
        results.append(B.run_backtest(data, cfg, fx, sleeve=s, label=s))
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append({"策略": r["label"], "总收益": f"{m['total_return']:+.1%}",
                    "年化": f"{m['cagr']:+.1%}", "波动": f"{m['vol']:.1%}",
                    "Sharpe": f"{m['sharpe']:.2f}",
                    "最大回撤": f"{m['max_drawdown']:.1%}",
                    "Calmar": f"{m['calmar']:.2f}",
                    "日胜率": f"{m['day_win_rate']:.1%}"})
    bm = comp.get("bench_metrics", {})
    if bm:
        rows.append({"策略": "沪深300(基准)", "总收益": f"{bm['total_return']:+.1%}",
                    "年化": f"{bm['cagr']:+.1%}", "波动": f"{bm['vol']:.1%}",
                    "Sharpe": f"{bm['sharpe']:.2f}",
                    "最大回撤": f"{bm['max_drawdown']:.1%}",
                    "Calmar": f"{bm['calmar']:.2f}",
                    "日胜率": f"{bm['day_win_rate']:.1%}"})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    B.plot_result(results, os.path.join(ROOT, "reports", "backtest.png"))
    df.to_csv(os.path.join(ROOT, "reports", "backtest_summary.csv"),
              index=False, encoding="utf-8-sig")
    comp["orders"].to_csv(os.path.join(ROOT, "reports", "backtest_orders.csv"),
                         index=False, encoding="utf-8-sig")
    print("[out] reports/backtest.png, backtest_summary.csv, backtest_orders.csv")


def cmd_trade(cfg):
    fx = get_fx(cfg)
    data = D.load_universe(cfg["universe"], cfg["start_date"])
    orders, snap = E.run_paper_day(data, cfg, fx)
    if snap:
        print(f"[paper] {snap['date']} 净值 ¥{snap['equity']:,.0f} "
              f"现金 ¥{snap['cash']:,.0f} 调仓 {len(orders)} 笔")
        for o in orders:
            print(f"  {o['side']:4s} {o['sym']} x{o['qty']} @ {o['price']:.3f} "
                  f"fee {o['fee']}")
    else:
        print("[paper] 无新交易日或已执行")


def cmd_report(cfg, ai=False):
    fx = get_fx(cfg)
    data = D.load_universe(cfg["universe"], cfg["start_date"])
    try:
        bcn = D.fetch_benchmark(cfg["benchmarks"]["cn"], cfg["start_date"])
    except Exception:
        bcn = None
    try:
        bhk = D.fetch_benchmark(cfg["benchmarks"]["hk"], cfg["start_date"])
    except Exception:
        bhk = None
    note = None
    if ai:
        note = vibe_comment(cfg)
    path = None
    try:
        path = __import__("qts.report", fromlist=["daily_report"]).daily_report(
            data, cfg, fx, bcn, bhk, ai_note=note)
    except Exception as e:  # noqa: BLE001
        print(f"[report] 失败: {e}")
    if path:
        print(f"[report] {path}")
    return path


def vibe_comment(cfg):
    """调用本机 vibe-trading，对当日虚拟盘给出 AI 点评。"""
    from qts.engine import STATE_FILE
    if not os.path.exists(STATE_FILE):
        return None
    with open(STATE_FILE) as f:
        state = json.load(f)
    pos = ", ".join(f"{k}:{v['qty']}股(成本{v['avg_cost']:.2f})"
                    for k, v in state["positions"].items()) or "空仓"
    eh = state["equity_history"][-3:] if state["equity_history"] else []
    prompt = (f"你是量化风控分析师。A股+港股虚拟盘今日状态：日期{state['last_date']}，"
              f"净值{[e['equity'] for e in eh]}，现金{state['cash']:.0f}元，"
              f"持仓：{pos}。请用中文给出3条简短的风险分析与关注要点，"
              f"聚焦A股/港股市场环境与策略执行风险，不超过200字。")
    try:
        r = subprocess.run(["vibe-trading", "run", "-p", prompt],
                          capture_output=True, text=True, timeout=420)
        out = (r.stdout or "").strip()
        if out:
            return out[-1200:]
        return f"(vibe-trading 无输出: {r.stderr[-200:]})"
    except Exception as e:  # noqa: BLE001
        return f"(vibe-trading 调用失败: {e})"


def cmd_status(cfg):
    from qts.engine import STATE_FILE
    if not os.path.exists(STATE_FILE):
        print("尚未开始虚拟盘，运行: main.py trade")
        return
    with open(STATE_FILE) as f:
        s = json.load(f)
    eq = s["equity_history"][-1] if s["equity_history"] else None
    print(f"创建: {s['created']}  最后交易日: {s['last_date']}")
    if eq:
        print(f"净值: ¥{eq['equity']:,.0f} ({eq['equity']/s['initial_capital']-1:+.2%})  "
              f"现金: ¥{eq['cash']:,.0f}  交易笔数: {len(s['trades'])}")
    for k, v in s["positions"].items():
        print(f"  {k}: {v['qty']} 股 @ {v['avg_cost']:.3f}")


def main():
    ap = argparse.ArgumentParser(description="quant-cn 量化虚拟盘")
    ap.add_argument("cmd", choices=["update", "backtest", "trade", "report",
                                   "daily", "status"])
    ap.add_argument("--refresh", action="store_true", help="强制刷新行情缓存")
    ap.add_argument("--ai", action="store_true", help="调用 vibe-trading AI 点评")
    a = ap.parse_args()
    cfg = load_cfg()
    if a.cmd == "update":
        cmd_update(cfg, a.refresh)
    elif a.cmd == "backtest":
        cmd_backtest(cfg)
    elif a.cmd == "trade":
        cmd_trade(cfg)
    elif a.cmd == "report":
        cmd_report(cfg, a.ai)
    elif a.cmd == "daily":
        cmd_update(cfg)
        cmd_trade(cfg)
        cmd_report(cfg, a.ai)
    elif a.cmd == "status":
        cmd_status(cfg)


if __name__ == "__main__":
    main()
