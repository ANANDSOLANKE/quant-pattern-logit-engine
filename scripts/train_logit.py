# scripts/train_logit.py
#
# Train a simple logistic regression model:
#   Target: 1 if next day's close > today's close, else 0
#   Features: RSI14, 5/10-day returns, volume z-score, ATR%
#
# Model is saved to Model/logit_model_T1.json

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA_FILE = Path("Data") / "Historical.csv"
MODEL_DIR = Path("Model")
MODEL_PATH = MODEL_DIR / "logit_model_T1.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------- feature engineering ----------

def compute_features_block(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given raw OHLCV for ONE symbol (columns: Symbol, Date, Open, High, Low, Close, Volume, ...),
    compute technical features and the next-day target y.
    """
    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) < 40:  # not enough history
        return pd.DataFrame()

    # ensure numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if df.empty:
        return pd.DataFrame()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float)

    # daily return
    ret1 = close.pct_change()

    # RSI14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(14, min_periods=14).mean()
    roll_down = loss.rolling(14, min_periods=14).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi14 = 100 - (100 / (1 + rs))

    # ROC 5 / ROC 10
    roc5 = close.pct_change(5)
    roc10 = close.pct_change(10)

    # volume z-score over 20 days
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

    # target: next-day up or not
    next_close = close.shift(-1)
    y = (next_close > close).astype(int)

    feat = pd.DataFrame({
        "Symbol": df["Symbol"],
        "Date": df["Date"],
        "RSI14": rsi14,
        "ROC5": roc5,
        "ROC10": roc10,
        "VOL_Z": vol_z,
        "ATR_PCT": atr_pct,
        "RET1": ret1,
        "y": y,
    })

    # drop first rows with NaNs / last row with y NaN
    feat = feat.dropna(subset=["RSI14", "ROC5", "ROC10", "VOL_Z", "ATR_PCT", "RET1", "y"])

    return feat


# ---------- main training ----------

def main():
    if not DATA_FILE.exists():
        raise SystemExit(f"Missing {DATA_FILE}")

    print(f"Reading {DATA_FILE} ...")
    raw = pd.read_csv(DATA_FILE, parse_dates=["Date"], low_memory=False)

    # clean numeric fields globally (in case some rows have text like '20MICRONS.NS')
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    raw = raw.sort_values(["Symbol", "Date"])

    feats_list = []
    for sym, g in raw.groupby("Symbol", sort=False):
        block = compute_features_block(g)
        if not block.empty:
            feats_list.append(block)

    if not feats_list:
        raise SystemExit("No features produced; check data.")

    feats = pd.concat(feats_list, ignore_index=True)
    feats = feats.dropna(subset=["y"])

    feature_cols = ["RSI14", "ROC5", "ROC10", "VOL_Z", "ATR_PCT", "RET1"]

    X = feats[feature_cols].values.astype(float)
    y = feats["y"].astype(int).values

    print(f"Training samples: {len(y)}; positive ratio={y.mean():.3f}")

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=1,
    )
    clf.fit(X, y)

    model_dict = {
        "feature_names": feature_cols,
        "intercept": float(clf.intercept_[0]),
        "coef": [float(c) for c in clf.coef_[0]],
    }

    MODEL_PATH.write_text(json.dumps(model_dict, indent=2))
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
