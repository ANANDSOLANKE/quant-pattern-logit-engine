# scripts/build_predictions_logit.py
#
# Use multi-horizon logistic model (T+1, T+2, T+3) to:
#   - compute probabilities of UP for each horizon
#   - build index.html with all stocks
#   - build stocks/<SYMBOL>.html with:
#       * latest OHLC
#       * T+1/T+2/T+3 Up% and Down%
#       * backtest table for last 10 days:
#           date, close, actual move, predicted Up% (T+1,T+2,T+3)

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

DATA_FILE = Path("Data") / "Historical.csv"
MODEL_PATH = Path("Model") / "logit_model_T123.json"
DIST_DIR = Path("dist")


# ---------- feature engineering (same as in training, but no targets) ----------

def compute_features_block(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) < 60:
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
    open_ = df["Open"].astype(float)

    # base series
    ret1 = close.pct_change()

    # RSI14
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    roll_up = gain.rolling(14, min_periods=14).mean()
    roll_down = loss.rolling(14, min_periods=14).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi14 = 100 - (100 / (1 + rs))

    # ROC 5 / 10
    roc5 = close.pct_change(5)
    roc10 = close.pct_change(10)

    # volume z-score
    vol_mean = vol.rolling(20, min_periods=20).mean()
    vol_std = vol.rolling(20, min_periods=20).std()
    vol_z = (vol - vol_mean) / (vol_std + 1e-9)

    # ATR% (14)
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()
    atr_pct = atr14 / (close + 1e-9)

    # Momentum 10
    mom10 = close.pct_change(10)

    # CCI20
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(20, min_periods=20).mean()
    mad_tp = (tp - sma_tp).abs().rolling(20, min_periods=20).mean()
    cci20 = (tp - sma_tp) / (0.015 * (mad_tp + 1e-9))

    # Stochastic %K (14)
    low14 = low.rolling(14, min_periods=14).min()
    high14 = high.rolling(14, min_periods=14).max()
    stoch_k = 100 * (close - low14) / (high14 - low14 + 1e-9)

    # MACD (12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    # Money Flow Index 14
    mf_tp = tp * vol
    mf_pos = mf_tp.where(tp > tp.shift(1), 0.0)
    mf_neg = mf_tp.where(tp < tp.shift(1), 0.0)
    mf_pos_sum = mf_pos.rolling(14, min_periods=14).sum()
    mf_neg_sum = mf_neg.rolling(14, min_periods=14).sum()
    mfr = mf_pos_sum / (mf_neg_sum + 1e-9)
    mfi14 = 100 - (100 / (1 + mfr))

    # Chaikin Money Flow 20
    mfm = ((close - low) - (high - close)) / (high - low + 1e-9)
    mfv = mfm * vol
    cmf20 = mfv.rolling(20, min_periods=20).sum() / (vol.rolling(20, min_periods=20).sum() + 1e-9)

    # OBV and ADL
    direction = np.sign(close - close.shift(1))
    obv = (direction * vol).fillna(0.0).cumsum()
    adl = mfv.fillna(0.0).cumsum()
    obv_mean = obv.rolling(20, min_periods=20).mean()
    obv_std = obv.rolling(20, min_periods=20).std()
    obv_z = (obv - obv_mean) / (obv_std + 1e-9)
    adl_mean = adl.rolling(20, min_periods=20).mean()
    adl_std = adl.rolling(20, min_periods=20).std()
    adl_z = (adl - adl_mean) / (adl_std + 1e-9)

    # Candle geometry
    body = (close - open_).abs()
    candle_range = (high - low).replace(0, np.nan)
    body_pct = body / (candle_range + 1e-9)
    up_shadow_pct = (high - close.clip(upper=open_)) / (candle_range + 1e-9)
    down_shadow_pct = (open_.clip(upper=close) - low) / (candle_range + 1e-9)

    feat = pd.DataFrame({
        "Symbol": df["Symbol"],
        "Date": df["Date"],
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol,
        "RSI14": rsi14,
        "ROC5": roc5,
        "ROC10": roc10,
        "VOL_Z": vol_z,
        "ATR_PCT": atr_pct,
        "RET1": ret1,
        "MOM10": mom10,
        "CCI20": cci20,
        "STOCH_K": stoch_k,
        "MACD": macd,
        "MACD_SIGNAL": macd_signal,
        "MACD_HIST": macd_hist,
        "MFI14": mfi14,
        "CMF20": cmf20,
        "OBV_Z": obv_z,
        "ADL_Z": adl_z,
        "BODY_PCT": body_pct,
        "UP_SHADOW_PCT": up_shadow_pct,
        "DOWN_SHADOW_PCT": down_shadow_pct,
    })

    feat = feat.dropna(
        subset=[
            "RSI14", "ROC5", "ROC10", "VOL_Z", "ATR_PCT", "RET1",
            "MOM10", "CCI20", "STOCH_K",
            "MACD", "MACD_SIGNAL", "MACD_HIST",
            "MFI14", "CMF20", "OBV_Z", "ADL_Z",
            "BODY_PCT", "UP_SHADOW_PCT", "DOWN_SHADOW_PCT",
        ]
    )

    return feat


# ---------- helpers ----------

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def load_model():
    if not MODEL_PATH.exists():
        raise SystemExit(f"Missing model file {MODEL_PATH}")
    data = json.loads(MODEL_PATH.read_text())
    feature_names = data["feature_names"]
    models = data["models"]
    for k in models:
        models[k]["intercept"] = float(models[k]["intercept"])
        models[k]["coef"] = np.array(models[k]["coef"], dtype=float)
    return feature_names, models


def render_index(rows):
    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>NSE – Multi-horizon Probabilities</title>",
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
        "<h1>NSE – Next-day and 2/3-day Probabilities</h1>",
        "<p>Probabilities are estimated from historical OHLCV using logistic models (T+1, T+2, T+3). They are not guarantees.</p>",
        "<table>",
        "<tr><th>Symbol</th><th>Last Date</th><th>Close</th>",
        "<th>T+1 Up%</th><th>T+2 Up%</th><th>T+3 Up%</th></tr>",
    ]
    for r in rows:
        def cls(p):
            if p >= 0.6:
                return "conf-high"
            elif p <= 0.4:
                return "conf-low"
            return "conf-mid"

        html.append(
            "<tr>"
            f"<td><a href='stocks/{r['symbol']}.html'>{r['symbol']}</a></td>"
            f"<td>{r['date']}</td>"
            f"<td>{r['close']:.2f}</td>"
            f"<td class='{cls(r['p_T1'])}'>{r['p_T1']*100:0.1f}</td>"
            f"<td class='{cls(r['p_T2'])}'>{r['p_T2']*100:0.1f}</td>"
            f"<td class='{cls(r['p_T3'])}'>{r['p_T3']*100:0.1f}</td>"
            "</tr>"
        )
    html.extend(["</table>", "</body></html>"])
    return "\n".join(html)


def render_stock_page(row, feature_names, latest_fv, models, history_rows):
    def cls(p):
        if p >= 0.6:
            return "conf-high"
        elif p <= 0.4:
            return "conf-low"
        return "conf-mid"

    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{row['symbol']} – Multi-horizon probability</title>",
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
        "<a href='../index.html'>&larr; Back to all stocks</a>",
        "<div class='card'>",
        f"<h1>{row['symbol']} – {row['date']}</h1>",
        "<table><tr><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Volume</th></tr>",
        f"<tr><td>{row['open']:.2f}</td><td>{row['high']:.2f}</td>"
        f"<td>{row['low']:.2f}</td><td>{row['close']:.2f}</td><td>{int(row['volume'])}</td></tr>",
        "</table>",
        "</div>",
        "<div class='card'>",
        "<h2>Multi-horizon probabilities</h2>",
        "<table>",
        "<tr><th>Horizon</th><th>Up %</th><th>Down %</th></tr>",
        f"<tr><td>T+1</td><td class='{cls(row['p_T1'])}'>{row['p_T1']*100:0.2f}</td>"
        f"<td>{(1-row['p_T1'])*100:0.2f}</td></tr>",
        f"<tr><td>T+2</td><td class='{cls(row['p_T2'])}'>{row['p_T2']*100:0.2f}</td>"
        f"<td>{(1-row['p_T2'])*100:0.2f}</td></tr>",
        f"<tr><td>T+3</td><td class='{cls(row['p_T3'])}'>{row['p_T3']*100:0.2f}</td>"
        f"<td>{(1-row['p_T3'])*100:0.2f}</td></tr>",
        "</table>",
        "<p class='tag'>These are statistical probabilities based on past patterns, not certainties.</p>",
        "</div>",
    ]

    # Backtest table
    if history_rows:
        html.append("<div class='card'>")
        html.append("<h2>Last 10 bars – prediction vs actual</h2>")
        html.append("<table>")
        html.append("<tr><th>Date</th><th>Close</th>"
                    "<th>Actual T+1</th><th>P(T+1 Up)%</th>"
                    "<th>Actual T+2</th><th>P(T+2 Up)%</th>"
                    "<th>Actual T+3</th><th>P(T+3 Up)%</th></tr>")
        for r in history_rows:
            html.append(
                "<tr>"
                f"<td>{r['date']}</td>"
                f"<td>{r['close']:.2f}</td>"
                f"<td>{r['act_T1']}</td>"
                f"<td>{r['p_T1']*100:0.1f}</td>"
                f"<td>{r['act_T2']}</td>"
                f"<td>{r['p_T2']*100:0.1f}</td>"
                f"<td>{r['act_T3']}</td>"
                f"<td>{r['p_T3']*100:0.1f}</td>"
                "</tr>"
            )
        html.append("</table>")
        html.append("<p class='tag'>Backtest is computed from your uploaded historical data; no values are faked.</p>")
        html.append("</div>")

    # Feature snapshot of latest bar
    beta_T1 = models["T1"]["coef"]
    beta_abs = [abs(b * x) for b, x in zip(beta_T1, latest_fv)]
    driver_rows = sorted(
        zip(feature_names, beta_T1, beta_abs),
        key=lambda t: t[2],
        reverse=True,
    )[:7]

    html.append("<div class='card'><h2>Top drivers for T+1 today</h2>")
    html.append("<table><tr><th>Feature</th><th>β</th><th>|β·x|</th></tr>")
    for name, b, mag in driver_rows:
        html.append(
            f"<tr><td>{name}</td><td>{b:0.4f}</td><td>{mag:0.4f}</td></tr>"
        )
    html.append("</table><p class='tag'>Higher |β·x| means stronger influence on T+1 probability.</p></div>")

    html.append("</body></html>")
    return "\n".join(html)


# ---------- main ----------

def main():
    if not DATA_FILE.exists():
        raise SystemExit(f"Missing {DATA_FILE}")
    feature_names, models = load_model()

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    stocks_dir = DIST_DIR / "stocks"
    stocks_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE, parse_dates=["Date"], low_memory=False)
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

        # feature matrix for this symbol
        fv_mat = block[feature_names].values.astype(float)
        n = len(block)

        # probabilities for each horizon for all rows
        probs_T1 = []
        probs_T2 = []
        probs_T3 = []

        for fv in fv_mat:
            s1 = models["T1"]["intercept"] + float(np.dot(models["T1"]["coef"], fv))
            s2 = models["T2"]["intercept"] + float(np.dot(models["T2"]["coef"], fv))
            s3 = models["T3"]["intercept"] + float(np.dot(models["T3"]["coef"], fv))
            probs_T1.append(sigmoid(s1))
            probs_T2.append(sigmoid(s2))
            probs_T3.append(sigmoid(s3))

        block = block.copy()
        block["p_T1"] = probs_T1
        block["p_T2"] = probs_T2
        block["p_T3"] = probs_T3

        # actual movements for backtest
        close_series = block["Close"]
        act_T1 = (close_series.shift(-1) > close_series).map({True: "Up", False: "Down"})
        act_T2 = (close_series.shift(-2) > close_series).map({True: "Up", False: "Down"})
        act_T3 = (close_series.shift(-3) > close_series).map({True: "Up", False: "Down"})

        block["act_T1"] = act_T1
        block["act_T2"] = act_T2
        block["act_T3"] = act_T3

        # latest bar for summary + current prediction
        last = block.iloc[-1]
        row_info = {
            "symbol": sym,
            "date": last["Date"].strftime("%Y-%m-%d"),
            "open": float(last["Open"]),
            "high": float(last["High"]),
            "low": float(last["Low"]),
            "close": float(last["Close"]),
            "volume": float(last["Volume"]),
            "p_T1": float(last["p_T1"]),
            "p_T2": float(last["p_T2"]),
            "p_T3": float(last["p_T3"]),
        }
        summary_rows.append(row_info)

        # backtest last 10 rows where all actuals are known (need at least 3 days ahead)
        hist = block.dropna(subset=["act_T1", "act_T2", "act_T3"]).tail(10)
        history_rows = [
            {
                "date": r["Date"].strftime("%Y-%m-%d"),
                "close": float(r["Close"]),
                "act_T1": r["act_T1"],
                "act_T2": r["act_T2"],
                "act_T3": r["act_T3"],
                "p_T1": float(r["p_T1"]),
                "p_T2": float(r["p_T2"]),
                "p_T3": float(r["p_T3"]),
            }
            for _, r in hist.iterrows()
        ]

        latest_fv = fv_mat[-1]
        stock_html = render_stock_page(row_info, feature_names, latest_fv, models, history_rows)
        (stocks_dir / f"{sym}.html").write_text(stock_html, encoding="utf-8")

    summary_rows.sort(key=lambda r: r["p_T1"], reverse=True)
    index_html = render_index(summary_rows)
    (DIST_DIR / "index.html").write_text(index_html, encoding="utf-8")

    print(f"Wrote {len(summary_rows)} stock pages and index.html under {DIST_DIR}/")


if __name__ == "__main__":
    main()
