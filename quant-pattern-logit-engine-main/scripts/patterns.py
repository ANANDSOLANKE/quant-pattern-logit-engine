import numpy as np
import pandas as pd

def find_swings(high: pd.Series, low: pd.Series, left: int = 3, right: int = 3):
    n = len(high)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        window_h = high[i-left:i+right+1]
        window_l = low[i-left:i+right+1]
        if high[i] == window_h.max() and window_h.values.argmax() == left:
            swing_high[i] = True
        if low[i] == window_l.min() and window_l.values.argmin() == left:
            swing_low[i] = True
    return swing_high, swing_low

def rsi_divergence_features(df: pd.DataFrame,
                            rsi_col: str = "RSI14",
                            left: int = 3, right: int = 3,
                            price_eps: float = 0.002,
                            rsi_eps: float = 0.5):
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    rsi = df[rsi_col].astype(float)

    swing_high, swing_low = find_swings(high, low, left, right)

    bull_div = np.zeros(len(df), dtype=int)
    bear_div = np.zeros(len(df), dtype=int)

    last_sh_price = last_sh_rsi = None
    last_sl_price = last_sl_rsi = None

    for i in range(len(df)):
        if swing_high[i]:
            if last_sh_price is not None:
                if high[i] > last_sh_price * (1 + price_eps) and rsi[i] < last_sh_rsi - rsi_eps:
                    bear_div[i] = 1
            last_sh_price = high[i]
            last_sh_rsi = rsi[i]

        if swing_low[i]:
            if last_sl_price is not None:
                if low[i] < last_sl_price * (1 - price_eps) and rsi[i] > last_sl_rsi + rsi_eps:
                    bull_div[i] = 1
            last_sl_price = low[i]
            last_sl_rsi = rsi[i]

    df["RSI_BULL_DIV"] = bull_div
    df["RSI_BEAR_DIV"] = bear_div
    return df

def double_top_bottom_features(df: pd.DataFrame,
                               left: int = 3, right: int = 3,
                               tol: float = 0.01):
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    swing_high, swing_low = find_swings(high, low, left, right)

    dt = np.zeros(len(df), dtype=int)
    db = np.zeros(len(df), dtype=int)

    last_sh_price = None
    last_sl_price = None

    for i in range(len(df)):
        if swing_high[i]:
            if last_sh_price is not None:
                if abs(high[i] - last_sh_price) / last_sh_price <= tol:
                    if close[i] < min(high[i], last_sh_price):
                        dt[i] = 1
            last_sh_price = high[i]

        if swing_low[i]:
            if last_sl_price is not None:
                if abs(low[i] - last_sl_price) / last_sl_price <= tol:
                    if close[i] > max(low[i], last_sl_price):
                        db[i] = 1
            last_sl_price = low[i]

    df["DOUBLE_TOP"] = dt
    df["DOUBLE_BOTTOM"] = db
    return df

def wedge_stub(df: pd.DataFrame, window: int = 20):
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    rng = high - low
    mean_rng = rng.rolling(window).mean()
    slope = mean_rng.diff()
    close = df["Close"].astype(float)
    ret = close.pct_change(window)
    wedge_up = ((slope < 0) & (ret > 0)).astype(int)
    wedge_down = ((slope < 0) & (ret < 0)).astype(int)
    df["WEDGE_UP"] = wedge_up.fillna(0).astype(int)
    df["WEDGE_DOWN"] = wedge_down.fillna(0).astype(int)
    return df

def add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    df = rsi_divergence_features(df)
    df = double_top_bottom_features(df)
    df = wedge_stub(df)
    return df
