import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils_indicators import compute_indicators_for_symbol
from patterns import add_pattern_features

DATA_FILE = Path("Data/Historical.csv")
MODEL_DIR = Path("Model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

H = 1  # predict next day

FEATURE_COLS = [
    "RSI14", "MACD_HIST", "STOCH_K", "CCI20",
    "ROC5", "ROC10", "VOL_Z", "ATR_PCT",
    "CANDLE_DIR", "CANDLE_RANGE_PCT",
    "RSI_BULL_DIV", "RSI_BEAR_DIV",
    "DOUBLE_TOP", "DOUBLE_BOTTOM",
    "WEDGE_UP", "WEDGE_DOWN",
]

def build_dataset():
    if not DATA_FILE.exists():
        raise SystemExit("Missing Data/Historical.csv")

    raw = pd.read_csv(DATA_FILE)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values(["Symbol", "Date"]).reset_index(drop=True)

    rows = []
    for sym, g in raw.groupby("Symbol", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        if len(g) < 100:
            continue

        gi = compute_indicators_for_symbol(g)
        gi = add_pattern_features(gi)

        closes = gi["Close"].astype(float).values
        n = len(gi)

        for i in range(n - H):
            j = i + H
            ret = (closes[j] - closes[i]) / closes[i]
            y = 1 if ret > 0 else 0
            row = gi.iloc[i]
            feat = {c: float(row[c]) for c in FEATURE_COLS}
            feat["y"] = y
            rows.append(feat)

    if not rows:
        raise SystemExit("No training rows; check Historical.csv")

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def standardize(X: np.ndarray):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xn = (X - mean) / std
    return Xn, mean, std

def main():
    df = build_dataset()
    X = df[FEATURE_COLS].values.astype(float)
    y = df["y"].values.astype(int)

    Xn, mean, std = standardize(X)

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
    )
    clf.fit(Xn, y)

    coef = clf.coef_[0].tolist()
    intercept = float(clf.intercept_[0])

    model = {
        "horizon": H,
        "features": FEATURE_COLS,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "beta0": intercept,
        "beta": {FEATURE_COLS[i]: coef[i] for i in range(len(FEATURE_COLS))},
    }

    (MODEL_DIR / "logit_model_T1.json").write_text(
        json.dumps(model, indent=2),
        encoding="utf-8"
    )

if __name__ == "__main__":
    main()
