import numpy as np
import pandas as pd

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def stoch_kd(high: pd.Series, low: pd.Series, close: pd.Series,
             k_period: int = 14, d_period: int = 3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_period).mean()
    return k.fillna(50), d.fillna(50)

def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(period).mean()
    mad = (tp - sma_tp).abs().rolling(period).mean()
    cci_val = (tp - sma_tp) / (0.015 * mad)
    return cci_val.replace([np.inf, -np.inf], np.nan).fillna(0)

def roc(series: pd.Series, period: int = 10) -> pd.Series:
    return series.pct_change(periods=period)

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def mfi(high: pd.Series, low: pd.Series, close: pd.Series,
        volume: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    raw_mf = tp * volume
    pos_mf = raw_mf.where(tp > tp.shift(1), 0.0)
    neg_mf = raw_mf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    mr = pos_sum / neg_sum.replace(0, np.nan)
    mfi_val = 100 - (100 / (1 + mr))
    return mfi_val.replace([np.inf, -np.inf], np.nan).fillna(50)

def compute_indicators_for_symbol(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    vol = out["Volume"].astype(float)

    out["RSI14"] = rsi(close, 14)
    out["SMA5"] = sma(close, 5)
    out["SMA20"] = sma(close, 20)
    out["SMA50"] = sma(close, 50)
    out["EMA5"] = ema(close, 5)
    out["EMA10"] = ema(close, 10)
    out["EMA20"] = ema(close, 20)

    macd_line, signal_line, hist = macd(close)
    out["MACD"] = macd_line
    out["MACD_SIGNAL"] = signal_line
    out["MACD_HIST"] = hist

    out["ATR14"] = atr(high, low, close, 14)
    out["ATR_PCT"] = out["ATR14"] / close.replace(0, np.nan)

    k, d = stoch_kd(high, low, close, 14, 3)
    out["STOCH_K"] = k
    out["STOCH_D"] = d

    out["CCI20"] = cci(high, low, close, 20)
    out["ROC5"] = roc(close, 5)
    out["ROC10"] = roc(close, 10)

    out["VOL_MA20"] = vol.rolling(20).mean()
    out["VOL_STD20"] = vol.rolling(20).std()
    out["VOL_Z"] = (vol - out["VOL_MA20"]) / out["VOL_STD20"].replace(0, np.nan)

    out["OBV"] = obv(close, vol)
    out["MFI14"] = mfi(high, low, close, vol, 14)

    out["CANDLE_DIR"] = np.sign(close - out["Open"].astype(float))
    out["CANDLE_RANGE_PCT"] = (high - low) / close.replace(0, np.nan)

    return out
