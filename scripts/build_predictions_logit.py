import json
from pathlib import Path
from math import exp

import numpy as np
import pandas as pd
from jinja2 import Template

from utils_indicators import compute_indicators_for_symbol
from patterns import add_pattern_features

DATA_FILE = Path("Data/Historical.csv")
MODEL_FILE = Path("Model/logit_model_T1.json")
TEMPLATE_FILE = Path("templates/stock_template.html")

DIST_DIR = Path("dist")
STOCK_DIR = DIST_DIR / "stocks"
DIST_DIR.mkdir(parents=True, exist_ok=True)
STOCK_DIR.mkdir(parents=True, exist_ok=True)

def load_model():
    with open(MODEL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_template():
    with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
        return Template(f.read())

def logistic(z: float) -> float:
    if z >= 0:
        ez = exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = exp(z)
        return ez / (1.0 + ez)

def predict_prob(model, x_vec):
    mean = np.array(model["mean"])
    std = np.array(model["std"])
    beta0 = model["beta0"]
    beta = np.array([model["beta"][f] for f in model["features"]])
    xn = (x_vec - mean) / std
    z = beta0 + float(np.dot(beta, xn))
    return float(logistic(z)), xn, beta

def build_index_page(summary_rows):
    rows_html = []
    for r in summary_rows:
        rows_html.append(
            f"<tr>"
            f"<td><a href='stocks/{r['Symbol']}.html'>{r['Symbol']}</a></td>"
            f"<td>{r['Date']}</td>"
            f"<td>{r['Close']:.2f}</td>"
            f"<td>{r['Conf_T1']:.1f}</td>"
            f"</tr>"
        )
    table_rows = "\n".join(rows_html)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>NSE Stock Predictions</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:#050816;
      color:#e5e5e5;
      margin:0;
      padding:24px;
    }}
    h1 {{ margin-bottom: 12px; }}
    table {{
      width:100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border:1px solid #272b3b;
      padding:6px;
      text-align:center;
    }}
    th {{ background:#111827; }}
    a {{ color:#7ee7ff; text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
  </style>
</head>
<body>
  <h1>NSE Stock Prediction Summary</h1>
  <p style="font-size:12px;color:#9ca3af;">
    Each row shows the latest close and model probability that the next trading day is up.
    Click a symbol to see full indicator and pattern details.
  </p>
  <table>
    <tr>
      <th>Symbol</th>
      <th>Date</th>
      <th>Last Close</th>
      <th>Prob. Up (T+1, %)</th>
    </tr>
    {table_rows}
  </table>
</body>
</html>
"""
    (DIST_DIR / "index.html").write_text(html, encoding="utf-8")

def main():
    if not DATA_FILE.exists():
        raise SystemExit("Missing Data/Historical.csv (run sync_nse_data.py first)")
    if not MODEL_FILE.exists():
        raise SystemExit("Missing Model/logit_model_T1.json; run train_logit.py first")

    model = load_model()
    tpl = load_template()
    feats = model["features"]

    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    df = df.sort_values(["Symbol", "Date"])

    summary = []

    for sym, g in df.groupby("Symbol", sort=False):
        g = g.sort_values("Date")
        if len(g) < 100:
            continue

        gi = compute_indicators_for_symbol(g)
        gi = add_pattern_features(gi)
        last = gi.iloc[-1]

        x_vec = np.array([last[f] for f in feats], dtype=float)
        x_vec = np.nan_to_num(x_vec, nan=0.0, posinf=0.0, neginf=0.0)

        p_up, xn, beta_arr = predict_prob(model, x_vec)
        conf_t1 = 100.0 * p_up

        today = last["Date"].strftime("%Y-%m-%d")
        ohlc = {
            "Open": float(last["Open"]),
            "High": float(last["High"]),
            "Low": float(last["Low"]),
            "Close": float(last["Close"]),
            "Volume": float(last["Volume"]),
        }

        show_keys = [
            "RSI14","MACD_HIST","STOCH_K","CCI20",
            "ROC5","ROC10","VOL_Z","ATR_PCT",
            "CANDLE_DIR","CANDLE_RANGE_PCT",
            "RSI_BULL_DIV","RSI_BEAR_DIV",
            "DOUBLE_TOP","DOUBLE_BOTTOM",
            "WEDGE_UP","WEDGE_DOWN",
        ]
        indicators = {
            key: {"label": key, "value": float(last.get(key, 0.0))}
            for key in show_keys
        }

        contrib = []
        for i, f in enumerate(feats):
            w = beta_arr[i]
            impact = abs(w * xn[i])
            contrib.append((f, w, impact))
        contrib.sort(key=lambda z: z[2], reverse=True)
        top = contrib[:5]

        html = tpl.render(
            symbol=sym,
            date=today,
            ohlc=ohlc,
            indicators=indicators,
            conf={"T1": conf_t1, "T2": None, "T3": None},
            contrib={"T1": [
                {"indicator": f, "weight": float(w), "impact": float(imp)}
                for (f, w, imp) in top
            ]},
            base_rate={"T1": 0.5, "T2": 0.5, "T3": 0.5},
        )
        out_path = STOCK_DIR / f"{sym}.html"
        out_path.write_text(html, encoding="utf-8")

        summary.append({
            "Symbol": sym,
            "Date": today,
            "Close": ohlc["Close"],
            "Conf_T1": conf_t1,
        })

    summary_sorted = sorted(summary, key=lambda r: r["Symbol"])
    build_index_page(summary_sorted)

if __name__ == "__main__":
    main()
