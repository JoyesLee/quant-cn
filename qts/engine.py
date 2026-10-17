"""虚拟盘（Paper Trading）引擎：状态持久化 + 每日收盘撮合。

每日流程（收盘后运行）：
  当日收盘数据 → 三策略信号 → 目标权重 → 与当前持仓轧差 → 按收盘价虚拟成交
信号当日收盘生效，卖出的是此前持有的仓位，满足 A股 T+1。
"""
import os, json, datetime as dt
import pandas as pd
from . import portfolio as pf
from .broker import rebalance_to_targets, equity_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(ROOT, "state")
os.makedirs(STATE_DIR, exist_ok=True)
STATE_FILE = os.path.join(STATE_DIR, "paper.json")


def load_state(cfg):
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"created": dt.date.today().isoformat(),
            "initial_capital": float(cfg["initial_capital"]),
            "cash": float(cfg["initial_capital"]),
            "positions": {}, "trades": [], "equity_history": [],
            "last_date": None}


def save_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, STATE_FILE)


def run_paper_day(data, cfg, fx, trade_date=None):
    """执行一个虚拟交易日。返回 (orders, snapshot)。"""
    state = load_state(cfg)
    sigs = pf.sleeve_signals(data, cfg)
    market_of = {it["symbol"]: it["market"] for it in cfg["universe"]}
    lot = cfg["lot_size"]

    if trade_date is None:
        # 取所有标的共同的最新交易日
        common = None
        for df in data.values():
            common = set(df.index) if common is None else common & set(df.index)
        if not common:
            raise RuntimeError("无共同交易日")
        trade_date = max(common)
    trade_date = pd.Timestamp(trade_date)

    if state["last_date"] and str(trade_date.date()) <= state["last_date"]:
        print(f"[paper] {trade_date.date()} 已执行过，跳过（--force 可强制重跑当日信号查看）")
        return [], None

    # 信号用截至当日的收盘，当日收盘成交（虚拟盘约定）
    targets = pf.targets_at(sigs, trade_date, cfg)
    prices = {}
    for s, df in data.items():
        if trade_date in df.index:
            p = float(df.loc[trade_date, "close"])
            prices[s] = p * fx if market_of[s] == "HK" else p
    orders = rebalance_to_targets(state, targets, prices, market_of, cfg, fx,
                                 lot=lot)
    for o in orders:
        o["date"] = str(trade_date.date())
    state["trades"] += orders
    eq = equity_of(state, prices)
    state["equity_history"].append({"date": str(trade_date.date()),
                                   "equity": round(eq, 2),
                                   "cash": round(state["cash"], 2)})
    state["last_date"] = str(trade_date.date())
    save_state(state)
    snap = {"date": str(trade_date.date()), "equity": eq,
            "cash": state["cash"], "targets": targets, "prices": prices}
    return orders, snap


def positions_marked(state, prices_cny, market_of, fx, name_of):
    rows = []
    for s, p in state["positions"].items():
        price = prices_cny.get(s, p["avg_cost"])
        mv = p["qty"] * price
        pnl = (price / p["avg_cost"] - 1) if p["avg_cost"] else 0.0
        rows.append({"symbol": s, "name": name_of.get(s, s),
                    "market": market_of[s], "qty": p["qty"],
                    "avg_cost": round(p["avg_cost"], 3),
                    "price": round(price, 3),
                    "mv_cny": round(mv, 0), "pnl_pct": round(pnl * 100, 2)})
    rows.sort(key=lambda r: -r["mv_cny"])
    return rows
