"""账户与撮合：手续费模型 + 目标权重调仓（回测与虚拟盘共用）。

计价基准：CNY。港股价格由调用方按 fx (1 HKD = fx CNY) 折成 CNY 传入。
A股 T+1：本模块每日只按目标权重调仓一次，信号收盘计算、次日收盘成交，
当日买入最早第三日卖出，天然满足 T+1。
"""
import math


def trade_fee(market, side, turnover_cny, fees_cfg, fx):
    """单边费用（CNY）。side: 'BUY' | 'SELL'"""
    f = fees_cfg[market]
    comm = max(turnover_cny * f["commission"],
              f["min_commission"] * (fx if market == "HK" else 1.0))
    fee = comm
    if market == "CN":
        fee += turnover_cny * f.get("transfer", 0.0)
        if side == "SELL":
            fee += turnover_cny * f.get("stamp_sell", 0.0)
    else:  # HK：印花税双边 + 交易费/交收费
        fee += turnover_cny * (f.get("stamp", 0.0) + f.get("other", 0.0))
    return fee


def rebalance_to_targets(state, targets, prices_cny, market_of, cfg, fx,
                        lot=None, min_trade=None):
    """把 state(含 cash, positions{sym:{qty,avg_cost}}) 调整到 targets{sym:weight}。
    prices_cny: {sym: 价格(CNY)}。lot: int 或 {market: int}。返回订单列表。
    先卖后买，整手交易。"""
    min_trade = min_trade or cfg["risk"]["min_trade_value"]
    if lot is None:
        lot = {"CN": 100, "HK": 100}
    lot_of = lot if isinstance(lot, dict) else {k: lot for k in ("CN", "HK")}
    equity = equity_of(state, prices_cny)
    orders = []

    def price(s):
        return prices_cny.get(s)

    tgt_val = {s: equity * w for s, w in targets.items() if price(s)}
    # 1) 卖出（含清仓）
    for s in list(state["positions"].keys()):
        p = price(s)
        if not p:
            continue
        L = lot_of[market_of[s]]
        cur = state["positions"][s]["qty"] * p
        tgt = tgt_val.get(s, 0.0)
        if tgt < cur:
            qty = math.floor((cur - tgt) / (p * L)) * L
            sell_val = qty * p
            if qty > 0 and sell_val >= min_trade:
                fee = trade_fee(market_of[s], "SELL", sell_val, cfg["fees"], fx)
                state["cash"] += sell_val - fee
                orders.append({"sym": s, "side": "SELL", "qty": qty,
                             "price": p, "fee": round(fee, 2)})
                state["positions"][s]["qty"] -= qty
                if state["positions"][s]["qty"] <= 0:
                    del state["positions"][s]
    # 2) 买入（现金约束下缩量）
    for s, tgt in tgt_val.items():
        p = price(s)
        L = lot_of[market_of[s]]
        cur_qty = state["positions"].get(s, {}).get("qty", 0)
        if tgt > cur_qty * p:
            qty = math.floor((tgt - cur_qty * p) / (p * L)) * L
            buy_val = qty * p
            if qty <= 0 or buy_val < min_trade:
                continue
            fee = trade_fee(market_of[s], "BUY", buy_val, cfg["fees"], fx)
            if buy_val + fee > state["cash"]:
                qty = int((state["cash"] * 0.999) // (p * L) * L)
                if qty <= 0:
                    continue
                buy_val = qty * p
                fee = trade_fee(market_of[s], "BUY", buy_val, cfg["fees"], fx)
                if buy_val + fee > state["cash"]:
                    continue
            state["cash"] -= buy_val + fee
            avg = state["positions"].get(s, {}).get("avg_cost", p)
            tot_qty = cur_qty + qty
            state["positions"][s] = {"qty": tot_qty,
                                    "avg_cost": (avg * cur_qty + p * qty) / tot_qty}
            orders.append({"sym": s, "side": "BUY", "qty": qty,
                          "price": p, "fee": round(fee, 2)})
    return orders


def equity_of(state, prices_cny):
    return state["cash"] + sum(p["qty"] * prices_cny.get(s, p["avg_cost"])
                              for s, p in state["positions"].items())
