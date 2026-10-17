"""数据层：A股/港股日线获取与本地 CSV 缓存。
A股: akshare stock_zh_a_hist (东财, 前复权)
港股: akshare stock_hk_daily (新浪, 前复权)  —— 东财港股接口在本机网络不通，走新浪
汇率: 中行折算价 (HKD/CNY)
基准: 沪深300 / 恒生指数 (新浪)
"""
import os, time, datetime as dt
import pandas as pd
import akshare as ak

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def _retry(fn, *args, tries=3, wait=3, **kwargs):
    last = None
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(wait)
    raise last


def _normalize(df, date_col, rename_map):
    df = df.rename(columns=rename_map)[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    return df.astype(float)


def _cn_sina_symbol(symbol):
    if symbol.startswith(("6", "9", "5")):
        return "sh" + symbol
    return "sz" + symbol


def fetch_cn(symbol, start, end):
    # 主源：东财；备源：新浪（本机网络下东财偶发拒连）
    try:
        raw = _retry(ak.stock_zh_a_hist, symbol=symbol, period="daily",
                     start_date=start, end_date=end, adjust="qfq", tries=2, wait=2)
        return _normalize(raw, "日期",
                         {"日期": "date", "开盘": "open", "最高": "high",
                          "最低": "low", "收盘": "close", "成交量": "volume"})
    except Exception as e:  # noqa: BLE001
        print(f"[data] 东财 {symbol} 失败({type(e).__name__})，切换新浪")
        d = _retry(ak.stock_zh_a_daily, symbol=_cn_sina_symbol(symbol),
                   adjust="qfq", tries=2, wait=2)
        d = d.rename(columns={"date": "date"}).copy()
        d["date"] = pd.to_datetime(d["date"])
        d = d[(d["date"] >= pd.to_datetime(start)) & (d["date"] <= pd.to_datetime(end))]
        return _normalize(d, "date", {})


def fetch_hk(symbol, start):
    raw = _retry(ak.stock_hk_daily, symbol=symbol, adjust="qfq")
    df = _normalize(raw, "date", {})
    return df[df.index >= pd.to_datetime(start)]


def cache_path(market, symbol):
    return os.path.join(DATA_DIR, f"{market}_{symbol}.csv")


def load_history(item, start, end=None, refresh=False):
    """返回标准化日线 df（index=date）。带当日缓存。"""
    end = end or dt.date.today().strftime("%Y%m%d")
    path = cache_path(item["market"], item["symbol"])
    today = dt.date.today().isoformat()
    if not refresh and os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        if len(df) and str(df.index[-1].date()) >= today:
            return df
    try:
        if item["market"] == "CN":
            df = fetch_cn(item["symbol"], start, end)
        else:
            df = fetch_hk(item["symbol"], start)
        df.to_csv(path)
        return df
    except Exception as e:  # noqa: BLE001
        if os.path.exists(path):
            print(f"[data] {item['symbol']} 拉取失败({e})，使用旧缓存")
            return pd.read_csv(path, parse_dates=["date"], index_col="date")
        raise


def load_universe(universe, start, end=None, refresh=False):
    out = {}
    for i, item in enumerate(universe):
        if i:
            time.sleep(1.0)  # 节流，避免触发数据源限流
        try:
            out[item["symbol"]] = load_history(item, start, end, refresh)
        except Exception as e:  # noqa: BLE001
            print(f"[data] 跳过 {item['symbol']}: {e}")
    return out


def fetch_fx(fallback):
    """1 HKD = x CNY，取中行折算价最新值。"""
    try:
        end = dt.date.today().strftime("%Y%m%d")
        start = (dt.date.today() - dt.timedelta(days=15)).strftime("%Y%m%d")
        d = _retry(ak.currency_boc_sina, symbol="港币", start_date=start, end_date=end, tries=2)
        col = "中行折算价" if "中行折算价" in d.columns else "央行中间价"
        v = float(d[col].dropna().iloc[-1]) / 100.0
        if 0.5 < v < 1.5:
            return v
    except Exception as e:  # noqa: BLE001
        print(f"[fx] 实时汇率失败({e})，用兜底 {fallback}")
    return fallback


def fetch_benchmark(code, start):
    """code: 'sh000300' / 'HSI'"""
    if code.startswith(("sh", "sz")):
        d = _retry(ak.stock_zh_index_daily, symbol=code)
    else:
        d = _retry(ak.stock_hk_index_daily_sina, symbol=code)
    d = d.rename(columns={"date": "date"}).copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").set_index("date")
    d = d[d.index >= pd.to_datetime(start)]
    return d["close"].astype(float)
