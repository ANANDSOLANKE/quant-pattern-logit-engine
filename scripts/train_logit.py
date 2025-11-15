# scripts/train_logit.py
#
# Train three logistic regression models:
#   T1: next-day (T+1) direction vs today's close
#   T2: 2-day ahead (T+2) direction vs today's close
#   T3: 3-day ahead (T+3) direction vs today's close
#
# Targets:
#   y_T1 = 1 if Close[t+1] > Close[t] else 0
#   y_T2 = 1 if Close[t+2] > Close[t] else 0
#   y_T3 = 1 if Close[t+3] > Close[t] else 0
#
# Features (per bar):
#   RSI14, ROC5, ROC10, VOL_Z, ATR_PCT, RET1,
#   MOM10, CCI20, STOCH_K,
#   MACD, MACD_SIGNAL, MACD_HIST,
#   MFI14, CMF20, OBV_Z, ADL_Z,
#   BODY_PCT, UP_SHADOW_PCT, DOWN_SHADOW_PCT

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA_FILE = Path("Data") / "Historical.csv"
MODEL_DIR = Path("Model")
MODEL_PATH = MODEL_DIR / "logit_model_T123.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------- feature engineering for one symbol ----------

def compute_features_block(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given raw OHLCV for ONE symbol (Symbol, Date, Open, High, Low, Close, Volume, ...),
    compute indicators + targets y_T1, y_T2, y_T3.
    """
    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) < 60:  # need some history for indicators
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
    open_ = df["Open"].astype(float)

    # basic returns
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

    # Momentum 10 (percentage)
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

    # MACD (12,26,9) on close
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
    mfm = ((close - low) - (high - close)) / (high - low + 1e-9)  # Money flow multiplier
    mfv = mfm * vol                                              # Money flow volume
    cmf20 = mfv.rolling(20, min_periods=20).sum() / (vol.rolling(20, min_periods=20).sum() + 1e-9)

    # OBV and Accumulation/Distribution Line
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

    # Targets: T+1, T+2, T+3 vs today's close
    next1 = close.shift(-1)
    next2 = close.shift(-2)
    next3 = close.shift(-3)
    y_T1 = (next1 > close).astype(float)
    y_T2 = (next2 > close).astype(float)
    y_T3 = (next3 > close).astype(float)

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
        "y_T1": y_T1,
        "y_T2": y_T2,
        "y_T3": y_T3,
    })

    # drop rows where we miss any key features or targets
    feat = feat.dropna(
        subset=[
            "RSI14", "ROC5", "ROC10", "VOL_Z", "ATR_PCT", "RET1",
            "MOM10", "CCI20", "STOCH_K",
            "MACD", "MACD_SIGNAL", "MACD_HIST",
            "MFI14", "CMF20", "OBV_Z", "ADL_Z",
            "BODY_PCT", "UP_SHADOW_PCT", "DOWN_SHADOW_PCT",
            "y_T1", "y_T2", "y_T3",
        ]
    )

    return feat


# ---------- main training ----------

def train_for_label(feats: pd.DataFrame, feature_cols, label: str):
    df = feats.dropna(subset=feature_cols + [label])
    X = df[feature_cols].values.astype(float)
    y = df[label].astype(int).values
    if len(y) < 1000:
        raise SystemExit(f"Not enough samples for {label}: {len(y)}")
    print(f"Training {label}: samples={len(y)}, positive_ratio={y.mean():.3f}")
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=1,
    )
    clf.fit(X, y)
    return {
        "intercept": float(clf.intercept_[0]),
        "coef": [float(c) for c in clf.coef_[0]],
    }


def main():
    if not DATA_FILE.exists():
        raise SystemExit(f"Missing {DATA_FILE}")

    print(f"Reading {DATA_FILE} ...")
    raw = pd.read_csv(DATA_FILE, parse_dates=["Date"], low_memory=False)

    # clean numeric globally
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    raw = raw.sort_values(["Symbol", "Date"])

    blocks = []
    for sym, g in raw.groupby("Symbol", sort=False):
        block = compute_features_block(g)
        if not block.empty:
            blocks.append(block)

    if not blocks:
        raise SystemExit("No features produced; check data.")

    feats = pd.concat(blocks, ignore_index=True)

    feature_cols = [
        "RSI14", "ROC5", "ROC10", "VOL_Z", "ATR_PCT", "RET1",
        "MOM10", "CCI20", "STOCH_K",
        "MACD", "MACD_SIGNAL", "MACD_HIST",
        "MFI14", "CMF20", "OBV_Z", "ADL_Z",
        "BODY_PCT", "UP_SHADOW_PCT", "DOWN_SHADOW_PCT",
    ]

    models = {
        "T1": train_for_label(feats, feature_cols, "y_T1"),
        "T2": train_for_label(feats, feature_cols, "y_T2"),
        "T3": train_for_label(feats, feature_cols, "y_T3"),
    }

    model_dict = {
        "feature_names": feature_cols,
        "models": models,
    }

    MODEL_PATH.write_text(json.dumps(model_dict, indent=2))
    print(f"Saved multi-horizon model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
