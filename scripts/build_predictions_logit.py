# scripts/build_predictions_logit.py
#
# Use the trained logistic model (Model/logit_model_T1.json)
# to compute next-day up probability for each NSE symbol and
# build a simple static website under dist/.
#
# index.html  -> table of all stocks and probabilities
# stocks/SYMBOL.html -> detail page with latest OHLC and features

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

DATA_FILE = Path("Data") / "Historical.csv"
MODEL_PATH = Path("Model") / "logit_model_T1.json"
DIST_DIR = Path("dist")


# ---------- feature engineering (same as in train_logit.py, but no target) ----------

def compute_features_block(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) < 40:
        return pd.DataFrame()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if df.empty:
        return pd.DataFrame()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float)

    ret1 = close.pct_change()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(14, min_periods=14).mean()
    roll_down = loss.rolling(14, min_periods=14).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi14 = 100 - (100 / (1 + rs))

    roc5 = close.pct_change(5)
    roc10 = close.pct_change(10)

    vol_mean = vol.rolling(20, min_periods=20).mean()
    vol_std = vol.rolling(20, min_periods=20).std()
    vol_z = (vol - vol_mean) / (vol_std + 1e-9)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()
    atr_pct = atr14 / (close + 1e-9)

    feat = pd.DataFrame({
        "Symbol": df["Symbol"],
        "Date": df["Date"],
        "Close": close,
        "Open": df["Open"],
        "High": high,
        "Low": low,
        "Volume": vol,
        "RSI14": rsi14,
        "ROC5": roc5,
        "ROC10": roc10,
        "VOL_Z": vol_z,
        "ATR_PCT": atr_pct,
        "RET1": ret1,
    })

    feat = feat.dropna(subset=["RSI14", "ROC5", "ROC10", "VOL_Z", "ATR_PCT", "RET1"])
    return feat


# ---------- helpers ----------

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def load_model():
    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing model file {MODEL_PATH}")
    data = json.loads(MODEL_PATH.read_text())
    feature_names = data["feature_names"]
    intercept = float(data["intercept"])
    coef = np.array(data["coef"], dtype=float)
    return feature_names, intercept, coef


def render_index(rows):
    # simple dark table UI
    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>NSE – Probabilistic Pattern Model</title>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "<style>",
        "body{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#050816;color:#e5e5e5;margin:0;padding:24px;}",
        "h1{margin-top:0;margin-bottom:12px;}",
        "table{width:100%;border-collapse:collapse;font-size:13px;}",
        "th,td{border:1px solid #272b3b;padding:6px;text-align:center;}",
        "th{background:#111827;}",
        "a{color:#7ee7ff;text-decoration:none;}",
        "a:hover{text-decoration:underline;}",
        ".conf-high{color:#22c55e;font-weight:600;}",
        ".conf-mid{color:#eab308;font-weight:600;}",
        ".conf-low{color:#f97316;font-weight:600;}",
        "</style></head><body>",
        "<h1>NSE – Next-day Probability (T+1)</h1>",
        "<p>This table is generated from historical OHLCV and a logistic model. Values are probabilities, not guarantees.</p>",
        "<table>",
        "<tr><th>Symbol</th><th>Last Date</th><th>Close</th><th>Prob. Up % (T+1)</th></tr>",
    ]
    for r in rows:
        cls = "conf-mid"
        if r["prob_up"] >= 0.6:
            cls = "conf-high"
        elif r["prob_up"] <= 0.4:
            cls = "conf-low"
        html.append(
            f"<tr>"
            f"<td><a href='stocks/{r['symbol']}.html'>{r['symbol']}</a></td>"
            f"<td>{r['date']}</td>"
            f"<td>{r['close']:.2f}</td>"
            f"<td class='{cls}'>{r['prob_up']*100:0.1f}</td>"
            f"</tr>"
        )
    html.extend(["</table>", "</body></html>"])
    return "\n".join(html)


def render_stock_page(row, feature_names, feature_values, beta, intercept):
    # simple per-stock detail page
    beta_abs = [abs(b * x) for b, x in zip(beta, feature_values)]
    driver_rows = sorted(
        zip(feature_names, beta, beta_abs),
        key=lambda t: t[2],
        reverse=True,
    )[:5]

    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{row['symbol']} – T+1 probability</title>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "<style>",
        "body{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#050816;color:#e5e5e5;margin:0;padding:24px;}",
        "h1,h2{margin-top:0;}",
        ".card{background:#0b1020;border-radius:16px;padding:16px;margin-bottom:16px;border:1px solid #1f2937;}",
        "table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px;}",
        "th,td{border:1px solid #272b3b;padding:6px;text-align:center;}",
        "th{background:#111827;}",
        "a{color:#7ee7ff;text-decoration:none;}",
        "a:hover{text-decoration:underline;}",
        ".tag{font-size:11px;color:#9ca3af;}",
        ".conf-high{color:#22c55e;font-weight:600;}",
        ".conf-mid{color:#eab308;font-weight:600;}",
        ".conf-low{color:#f97316;font-weight:600;}",
        "</style></head><body>",
        f"<a href='../index.html'>&larr; Back to all stocks</a>",
        "<div class='card'>",
        f"<h1>{row['symbol']} – {row['date']}</h1>",
        "<table><tr><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Volume</th></tr>",
        f"<tr><td>{row['open']:.2f}</td><td>{row['high']:.2f}</td>"
        f"<td>{row['low']:.2f}</td><td>{row['close']:.2f}</td><td>{int(row['volume'])}</td></tr>",
        "</table>",
        "</div>",
        "<div class='card'>",
        "<h2>Next-day probability (T+1)</h2>",
    ]
    cls = "conf-mid"
    if row["prob_up"] >= 0.6:
        cls = "conf-high"
    elif row["prob_up"] <= 0.4:
        cls = "conf-low"
    html.append(
        f"<p class='{cls}' style='font-size:20px;'>Prob. Up = {row['prob_up']*100:0.2f}%</p>"
    )
    html.append("<p class='tag'>Estimated from a logistic model trained on historical OHLCV.</p>")
    html.append("</div>")

    html.append("<div class='card'><h2>Top drivers today</h2>")
    html.append("<table><tr><th>Feature</th><th>β</th><th>|β·x|</th></tr>")
    for name, b, mag in driver_rows:
        html.append(
            f"<tr><td>{name}</td><td>{b:0.4f}</td><td>{mag:0.4f}</td></tr>"
        )
    html.append("</table><p class='tag'>Higher |β·x| means stronger influence on today's probability.</p></div>")

    html.append("<div class='card'><h2>Feature snapshot</h2>")
    html.append("<table><tr><th>Feature</th><th>Value</th></tr>")
    for name, val in zip(feature_names, feature_values):
        html.append(f"<tr><td>{name}</td><td>{val:0.4f}</td></tr>")
    html.append("</table></div>")

    html.append("</body></html>")
    return "\n".join(html)


# ---------- main ----------

def main():
    if not DATA_FILE.exists():
        raise SystemExit(f"Missing {DATA_FILE}")

    feature_names, intercept, beta = load_model()

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    stocks_dir = DIST_DIR / "stocks"
    stocks_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE, parse_dates=["Date"], low_memory=False)

    # clean numeric OHLCV *in memory* (bad rows like '20MICRONS.NS' will be dropped)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df = df.sort_values(["Symbol", "Date"])

    summary_rows = []

    for sym, g in df.groupby("Symbol", sort=False):
        block = compute_features_block(g)
        if block.empty:
            continue

        last = block.iloc[-1]

        # build feature vector in the same order as training
        fv = np.array([float(last[name]) for name in feature_names], dtype=float)
        score = float(intercept + np.dot(beta, fv))
        p_up = sigmoid(score)

        row_info = {
            "symbol": sym,
            "date": last["Date"].strftime("%Y-%m-%d"),
            "open": float(last["Open"]),
            "high": float(last["High"]),
            "low": float(last["Low"]),
            "close": float(last["Close"]),
            "volume": float(last["Volume"]),
            "prob_up": p_up,
        }
        summary_rows.append(row_info)

        # write per-stock page
        stock_html = render_stock_page(row_info, feature_names, fv, beta, intercept)
        (stocks_dir / f"{sym}.html").write_text(stock_html, encoding="utf-8")

    # sort summary by probability desc
    summary_rows.sort(key=lambda r: r["prob_up"], reverse=True)
    index_html = render_index(summary_rows)
    (DIST_DIR / "index.html").write_text(index_html, encoding="utf-8")

    print(f"Wrote {len(summary_rows)} stock pages and index.html under {DIST_DIR}/")


if __name__ == "__main__":
    main()
