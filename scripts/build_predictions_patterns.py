# scripts/build_predictions_patterns.py
#
# Reads:
#   - Data/Historical/*.csv  (all stocks, long history)
#   - Model/pattern_stats_T123.json (built by build_pattern_table.py)
#
# Builds:
#   - dist/stocks/<SYMBOL>.html   (per-stock multi-horizon page)
#   - dist/index.html             (landing page with strongest T+1 Up / Down)
#
# Each stock page also shows last 10 daily signals, with actual vs predicted.

from pathlib import Path
import json

import numpy as np
import pandas as pd

DATA_DIR = Path("Data")
HIST_DIR = DATA_DIR / "Historical"
MASTER_FILE = DATA_DIR / "Historical.csv"

MODEL_DIR = Path("Model")
PATTERN_FILE = MODEL_DIR / "pattern_stats_T123.json"

DIST_DIR = Path("dist")


# ---------- master builder (from per-symbol CSVs) ----------

def build_master_from_per_symbol():
    if not HIST_DIR.exists():
        raise SystemExit(f"Historical folder {HIST_DIR} does not exist.")

    files = sorted(HIST_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {HIST_DIR}.")

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=["Date"], low_memory=False)
        except Exception as e:
            print(f"Skipping {f.name} due to read error: {e!r}")
            continue

        if "Symbol" not in df.columns:
            df["Symbol"] = f.stem

        needed = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]
        if any(col not in df.columns for col in needed):
            print(f"{f.name}: missing one of {needed}, skipping.")
            continue

        if "Sector" not in df.columns:
            df["Sector"] = ""
        if "Index" not in df.columns:
            df["Index"] = "NSE"

        df = df[["Symbol", "Date", "Open", "High", "Low", "Close",
                 "Volume", "Sector", "Index"]]
        frames.append(df)

    if not frames:
        raise SystemExit("No valid symbol CSVs to build master file.")

    master = pd.concat(frames, ignore_index=True)
    master = master.sort_values(["Symbol", "Date"])
    master.to_csv(MASTER_FILE, index=False)
    print(f"Built master file {MASTER_FILE} with {len(master)} rows.")


# ---------- bucket helpers (must match build_pattern_table.py) ----------

def bucket_rsi(r):
    if pd.isna(r):
        return None
    if r < 30:
        return "RSI<30"
    elif r < 40:
        return "RSI30-40"
    elif r < 60:
        return "RSI40-60"
    elif r < 70:
        return "RSI60-70"
    else:
        return "RSI>70"


def bucket_ret(r):
    if pd.isna(r):
        return None
    if r <= -0.03:
        return "RET<=-3"
    elif r <= -0.01:
        return "RET-3--1"
    elif r < 0.01:
        return "RET-1-+1"
    elif r < 0.03:
        return "RET+1-+3"
    else:
        return "RET>=+3"


def bucket_volz(z):
    if pd.isna(z):
        return None
    if z <= -1.0:
        return "VOL<-1"
    elif z >= 1.0:
        return "VOL>+1"
    else:
        return "VOL-1-+1"


def bucket_macd(macd_hist, close):
    if pd.isna(macd_hist) or pd.isna(close):
        return None
    rel = macd_hist / (close + 1e-9)
    if rel <= -0.01:
        return "MACD_strong_neg"
    elif rel >= 0.01:
        return "MACD_strong_pos"
    else:
        return "MACD_flat"


# ---------- indicator computation per symbol ----------

def compute_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if df.empty:
        return pd.DataFrame()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float)

    # RET1
    ret1 = close.pct_change()

    # RSI14
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    roll_up = gain.rolling(14, min_periods=14).mean()
    roll_down = loss.rolling(14, min_periods=14).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi14 = 100 - (100 / (1 + rs))

    # Volume z-score 20
    vol_mean = vol.rolling(20, min_periods=20).mean()
    vol_std = vol.rolling(20, min_periods=20).std()
    vol_z = (vol - vol_mean) / (vol_std + 1e-9)

    # MACD_HIST (12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    out = df.copy()
    out["RSI14"] = rsi14
    out["RET1"] = ret1
    out["VOL_Z"] = vol_z
    out["MACD_HIST"] = macd_hist

    return out


# ---------- probability -> direction + confidence ----------

def direction_and_confidence(p_up):
    """
    p_up: probability 0..1 from pattern stats

    Returns:
      direction: "Up" or "Down"
      confidence: 0..100 (how far from 50/50)
    """
    if p_up is None:
        return "Unknown", 0.0
    if p_up >= 0.5:
        return "Up", (p_up - 0.5) * 200.0
    else:
        return "Down", (0.5 - p_up) * 200.0


# ---------- HTML helpers ----------

COMMON_STYLE = """
body{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#050816;color:#e5e5e5;margin:0;padding:24px;}
h1,h2{margin-top:0;}
.card{background:#0b1020;border-radius:16px;padding:16px;margin-bottom:16px;border:1px solid #1f2937;}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px;}
th,td{border:1px solid #272b3b;padding:6px;text-align:center;}
th{background:#111827;}
a{color:#7ee7ff;text-decoration:none;}
a:hover{text-decoration:underline;}
.tag{font-size:11px;color:#9ca3af;}
.conf-high{color:#22c55e;font-weight:600;}
.conf-mid{color:#eab308;font-weight:600;}
.conf-low{color:#f97316;font-weight:600;}
"""


def conf_class(conf):
    if conf >= 60:
        return "conf-high"
    elif conf >= 30:
        return "conf-mid"
    else:
        return "conf-low"


def render_index(up_rows, down_rows):
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>NSE Pattern Model – T+1 Signals</title>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "<style>",
        COMMON_STYLE,
        "</style></head><body>",
        "<h1>NSE Pattern Model – T+1 Signals</h1>",
        "<p class='tag'>Signals are generated from a pattern table built from your historical data. "
        "Each row shows today's pattern for that stock: hit-rate P(Up T+1), confidence (0–100) and "
        "support (how many past occurrences of that pattern exist).</p>",
        "<div class='card'>",
        "<h2>Strongest T+1 Up signals</h2>",
        "<table>",
        "<tr><th>Symbol</th><th>Last Date</th><th>Close</th>"
        "<th>P(Up T+1)%</th><th>Confidence</th><th>Support</th></tr>",
    ]

    for r in up_rows:
        cls = conf_class(r["conf_T1"])
        html.append(
            "<tr>"
            f"<td><a href='stocks/{r['symbol']}.html'>{r['symbol']}</a></td>"
            f"<td>{r['date']}</td>"
            f"<td>{r['close']:.2f}</td>"
            f"<td class='{cls}'>{r['p_up_T1']*100:0.1f}</td>"
            f"<td class='{cls}'>{r['conf_T1']:0.1f}</td>"
            f"<td>{r['support']}</td>"
            "</tr>"
        )

    html.extend([
        "</table>",
        "</div>",
        "<div class='card'>",
        "<h2>Strongest T+1 Down signals</h2>",
        "<table>",
        "<tr><th>Symbol</th><th>Last Date</th><th>Close</th>"
        "<th>P(Up T+1)%</th><th>Confidence</th><th>Support</th></tr>",
    ])

    for r in down_rows:
        cls = conf_class(r["conf_T1"])
        html.append(
            "<tr>"
            f"<td><a href='stocks/{r['symbol']}.html'>{r['symbol']}</a></td>"
            f"<td>{r['date']}</td>"
            f"<td>{r['close']:.2f}</td>"
            f"<td class='{cls}'>{r['p_up_T1']*100:0.1f}</td>"
            f"<td class='{cls}'>{r['conf_T1']:0.1f}</td>"
            f"<td>{r['support']}</td>"
            "</tr>"
        )

    html.extend([
        "</table>",
        "</div>",
        "</body></html>",
    ])
    return "\n".join(html)


def render_stock_page(row, pattern_info_today, history_rows, win_summary):
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{row['symbol']} – Pattern-based T+1/T+2/T+3</title>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "<style>",
        COMMON_STYLE,
        "</style></head><body>",
        "<a href='../index.html'>&larr; Back to all stocks</a>",
        "<div class='card'>",
        f"<h1>{row['symbol']} – {row['date']}</h1>",
        "<table><tr><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Volume</th></tr>",
        f"<tr><td>{row['open']:.2f}</td><td>{row['high']:.2f}</td>"
        f"<td>{row['low']:.2f}</td><td>{row['close']:.2f}</td><td>{int(row['volume'])}</td></tr>",
        "</table></div>",
    ]

    # Today's signal
    p_up_T2 = pattern_info_today.get("p_up_T2", 0.5)
    p_up_T3 = pattern_info_today.get("p_up_T3", 0.5)
    dir_T2, conf_T2 = direction_and_confidence(p_up_T2)
    dir_T3, conf_T3 = direction_and_confidence(p_up_T3)
    cls1 = conf_class(row["conf_T1"])
    cls2 = conf_class(conf_T2)
    cls3 = conf_class(conf_T3)

    html.append("<div class='card'><h2>Today's pattern signal</h2>")
    html.append("<table><tr><th>Horizon</th><th>Direction</th>"
                "<th>P(Up)%</th><th>Confidence</th><th>Support</th></tr>")
    html.append(
        "<tr>"
        "<td>T+1</td>"
        f"<td>{row['dir_T1']}</td>"
        f"<td class='{cls1}'>{row['p_up_T1']*100:0.2f}</td>"
        f"<td class='{cls1}'>{row['conf_T1']:0.1f}</td>"
        f"<td>{row['support']}</td>"
        "</tr>"
    )
    html.append(
        "<tr>"
        "<td>T+2</td>"
        f"<td>{dir_T2}</td>"
        f"<td class='{cls2}'>{p_up_T2*100:0.2f}</td>"
        f"<td class='{cls2}'>{conf_T2:0.1f}</td>"
        f"<td>{row['support']}</td>"
        "</tr>"
    )
    html.append(
        "<tr>"
        "<td>T+3</td>"
        f"<td>{dir_T3}</td>"
        f"<td class='{cls3}'>{p_up_T3*100:0.2f}</td>"
        f"<td class='{cls3}'>{conf_T3:0.1f}</td>"
        f"<td>{row['support']}</td>"
        "</tr>"
    )
    html.append("</table>")
    html.append("<p class='tag'>Probabilities and confidence are derived from your historical data "
                "for this exact pattern signature. Support = how many past occurrences exist.</p>")
    html.append("</div>")

    # Summary of last-10 daily signals
    html.append("<div class='card'><h2>Last 10 daily signals – T+1 performance</h2>")
    html.append("<table><tr><th>Signals</th><th>Wins</th><th>Win %</th></tr>")
    if win_summary["signals"] > 0:
        win_pct = win_summary["wins"] / win_summary["signals"] * 100.0
    else:
        win_pct = 0.0
    html.append(
        f"<tr><td>{win_summary['signals']}</td>"
        f"<td>{win_summary['wins']}</td>"
        f"<td>{win_pct:0.1f}</td></tr>"
    )
    html.append("</table>")
    html.append("<p class='tag'>Each signal uses the pattern that existed on that day, "
                "with its own probability and support from the pattern table.</p>")
    html.append("</div>")

    # Detailed last-10 daily signals table
    if history_rows:
        html.append("<div class='card'><h2>Last 10 daily signals (detailed)</h2>")
        html.append(
            "<table><tr>"
            "<th>Date</th><th>Close at signal</th>"
            "<th>Pattern</th><th>P(Up T+1)%</th><th>Conf</th><th>Support</th>"
            "<th>Pred T+1</th><th>Actual T+1</th><th>Win?</th>"
            "<th>Actual T+2</th><th>Actual T+3</th>"
            "</tr>"
        )
        for r in history_rows:
            cls = conf_class(r["conf_T1"])
            html.append(
                "<tr>"
                f"<td>{r['date']}</td>"
                f"<td>{r['close']:.2f}</td>"
                f"<td>{r['pattern_key']}</td>"
                f"<td class='{cls}'>{r['p_up_T1']*100:0.1f}</td>"
                f"<td class='{cls}'>{r['conf_T1']:0.1f}</td>"
                f"<td>{r['support']}</td>"
                f"<td>{r['dir_T1']}</td>"
                f"<td>{r['act_T1']}</td>"
                f"<td>{'Yes' if r['win_T1'] else 'No'}</td>"
                f"<td>{r['act_T2']}</td>"
                f"<td>{r['act_T3']}</td>"
                "</tr>"
            )
        html.append("</table>")
        html.append("<p class='tag'>This table shows, day by day, what pattern was seen, "
                    "what the pattern table predicted for T+1, and what actually happened "
                    "for T+1/T+2/T+3.</p>")
        html.append("</div>")

    html.append("</body></html>")
    return "\n".join(html)


# ---------- main build ----------

def main():
    # Ensure master exists
    if not MASTER_FILE.exists():
        print("Data/Historical.csv not found – building from per-symbol CSVs...")
        build_master_from_per_symbol()

    if not PATTERN_FILE.exists():
        raise SystemExit(f"Pattern file {PATTERN_FILE} not found. Run build_pattern_table.py first.")

    pattern_data = json.loads(PATTERN_FILE.read_text())
    pattern_stats = pattern_data["patterns"]

    print(f"Loaded {len(pattern_stats)} patterns from {PATTERN_FILE}")

    df = pd.read_csv(MASTER_FILE, parse_dates=["Date"], low_memory=False)
    df = df.sort_values(["Symbol", "Date"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if df.empty:
        raise SystemExit("No data after cleaning master file.")

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    stocks_dir = DIST_DIR / "stocks"
    stocks_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    # We now accept ALL patterns for the landing page
    MIN_SUPPORT_FOR_LISTING = 1

    for sym, g in df.groupby("Symbol", sort=False):
        g_ind = compute_basic_indicators(g)
        if g_ind.empty or len(g_ind) < 50:
            continue
        g_ind = g_ind.sort_values("Date").reset_index(drop=True)

        close = g_ind["Close"].astype(float)
        rsi14 = g_ind["RSI14"].astype(float)
        ret1 = g_ind["RET1"].astype(float)
        vol_z = g_ind["VOL_Z"].astype(float)
        macd_hist = g_ind["MACD_HIST"].astype(float)

        n = len(g_ind)
        pattern_keys = [None] * n

        # pattern for each eligible index
        for i in range(3, n - 3):
            r_prev = ret1.iloc[i - 1]
            r_today = ret1.iloc[i]
            rsi_today = rsi14.iloc[i]
            vol_today = vol_z.iloc[i]
            macd_today = macd_hist.iloc[i]
            price_today = close.iloc[i]

            b_rsi = bucket_rsi(rsi_today)
            b_ret_prev = bucket_ret(r_prev)
            b_ret_today = bucket_ret(r_today)
            b_vol = bucket_volz(vol_today)
            b_macd = bucket_macd(macd_today, price_today)
            if None in (b_rsi, b_ret_prev, b_ret_today, b_vol, b_macd):
                continue
            pattern_keys[i] = f"{b_rsi}|{b_macd}|{b_vol}|{b_ret_prev}|{b_ret_today}"

        g_ind["pattern_key"] = pattern_keys

        # actual future moves vs today's close
        act_T1 = []
        act_T2 = []
        act_T3 = []
        for i in range(n):
            if i + 3 >= n:
                act_T1.append(None)
                act_T2.append(None)
                act_T3.append(None)
                continue
            c_t = close.iloc[i]
            c_t1 = close.iloc[i + 1]
            c_t2 = close.iloc[i + 2]
            c_t3 = close.iloc[i + 3]
            act_T1.append("Up" if c_t1 > c_t else "Down")
            act_T2.append("Up" if c_t2 > c_t else "Down")
            act_T3.append("Up" if c_t3 > c_t else "Down")

        g_ind["act_T1"] = act_T1
        g_ind["act_T2"] = act_T2
        g_ind["act_T3"] = act_T3

        # today's row (last available)
        last = g_ind.iloc[-1]
        pk_today = last["pattern_key"]
        if pk_today is None or pk_today not in pattern_stats:
            # no valid pattern for today; skip from index
            continue

        pat_today = pattern_stats[pk_today]
        support_today = pat_today["count"]
        p_up_T1_today = pat_today["p_up_T1"]
        dir_T1_today, conf_T1_today = direction_and_confidence(p_up_T1_today)

        if support_today < MIN_SUPPORT_FOR_LISTING:
            # still keep it, because MIN_SUPPORT_FOR_LISTING=1;
            # this check is just for future if you increase the threshold
            pass

        row_info = {
            "symbol": sym,
            "date": last["Date"].strftime("%Y-%m-%d"),
            "open": float(last["Open"]),
            "high": float(last["High"]),
            "low": float(last["Low"]),
            "close": float(last["Close"]),
            "volume": float(last["Volume"]),
            "p_up_T1": float(p_up_T1_today),
            "dir_T1": dir_T1_today,
            "conf_T1": float(conf_T1_today),
            "support": int(support_today),
        }
        summary_rows.append(row_info)

        # last 10 valid daily signals: each with its own pattern
        valid_indices = [
            i for i in range(3, n - 3)
            if g_ind.loc[i, "pattern_key"] is not None
               and g_ind.loc[i, "pattern_key"] in pattern_stats
               and g_ind.loc[i, "act_T1"] in ("Up", "Down")
        ]
        last10_idx = valid_indices[-10:]

        history_rows = []
        win_summary = {"signals": 0, "wins": 0}

        for i in last10_idx:
            pk = g_ind.loc[i, "pattern_key"]
            pat = pattern_stats[pk]
            p_up = pat["p_up_T1"]
            dir_T1, conf_T1 = direction_and_confidence(p_up)
            support = pat["count"]
            act1 = g_ind.loc[i, "act_T1"]
            act2 = g_ind.loc[i, "act_T2"]
            act3 = g_ind.loc[i, "act_T3"]
            win_T1 = (dir_T1 == act1)

            history_rows.append({
                "date": g_ind.loc[i, "Date"].strftime("%Y-%m-%d"),
                "close": float(g_ind.loc[i, "Close"]),
                "pattern_key": pk,
                "p_up_T1": float(p_up),
                "conf_T1": float(conf_T1),
                "support": int(support),
                "dir_T1": dir_T1,
                "act_T1": act1,
                "win_T1": bool(win_T1),
                "act_T2": act2,
                "act_T3": act3,
            })

            win_summary["signals"] += 1
            if win_T1:
                win_summary["wins"] += 1

        stock_html = render_stock_page(row_info, pat_today, history_rows, win_summary)
        (stocks_dir / f"{sym}.html").write_text(stock_html, encoding="utf-8")

    # landing page
    # now we do NOT filter by support – show all available signals
    strong = summary_rows

    up_rows = [r for r in strong if r["dir_T1"] == "Up"]
    down_rows = [r for r in strong if r["dir_T1"] == "Down"]

    up_rows.sort(key=lambda r: (r["conf_T1"], r["p_up_T1"]), reverse=True)
    down_rows.sort(key=lambda r: (r["conf_T1"], 1 - r["p_up_T1"]), reverse=True)

    up_rows = up_rows[:200]
    down_rows = down_rows[:200]

    index_html = render_index(up_rows, down_rows)
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    (DIST_DIR / "index.html").write_text(index_html, encoding="utf-8")

    print(f"Wrote {len(summary_rows)} stock pages and index.html under {DIST_DIR}/")


if __name__ == "__main__":
    main()
