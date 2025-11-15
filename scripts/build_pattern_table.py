# scripts/build_pattern_table.py
#
# Build a REAL reverse-engineering pattern table from your NSE history.
#
# 1) Ensure Data/Historical.csv exists:
#       - if missing, build it from Data/Historical/*.csv
# 2) For each day and each symbol:
#       - compute RSI14, MACD_HIST, VOL_Z, RET1
#       - build a discrete pattern_key from:
#           * today's RSI bucket
#           * today's MACD_HIST bucket
#           * today's VOL_Z bucket
#           * yesterday RET1 bucket
#           * today's RET1 bucket
#       - look at actual T+1, T+2, T+3 moves vs today's close
# 3) Aggregate counts and hit-rates per pattern_key:
#       - count, up_T1, up_T2, up_T3, p_up_T1, p_up_T2, p_up_T3
# 4) Save to Model/pattern_stats_T123.json

from pathlib import Path
import json

import numpy as np
import pandas as pd

DATA_DIR = Path("Data")
HIST_DIR = DATA_DIR / "Historical"
MASTER_FILE = DATA_DIR / "Historical.csv"

MODEL_DIR = Path("Model")
PATTERN_FILE = MODEL_DIR / "pattern_stats_T123.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------- master builder (same structure you already use) ----------

def build_master_from_per_symbol():
    """Create Data/Historical.csv by concatenating Data/Historical/*.csv."""
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


# ---------- bucketing helpers: define the PATTERN ----------

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
    # daily % return buckets
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
    # normalise MACD_HIST by price to make it dimensionless
    rel = macd_hist / (close + 1e-9)
    if rel <= -0.01:
        return "MACD_strong_neg"
    elif rel >= 0.01:
        return "MACD_strong_pos"
    else:
        return "MACD_flat"


# ---------- feature computation per symbol ----------

def compute_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic indicators used for pattern definition:
      - RSI14
      - MACD_HIST
      - VOL_Z (20-day z-score of volume)
      - RET1 (1-day return)
    """
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

    # MACD_HIST (12, 26, 9)
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


# ---------- main pattern table builder ----------

def main():
    # 1) Ensure master file
    if not MASTER_FILE.exists():
        print("Data/Historical.csv not found – building from per-symbol CSVs...")
        build_master_from_per_symbol()

    print(f"Reading {MASTER_FILE} ...")
    raw = pd.read_csv(MASTER_FILE, parse_dates=["Date"], low_memory=False)
    raw = raw.sort_values(["Symbol", "Date"])

    # convert numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if raw.empty:
        raise SystemExit("Master file empty after cleaning.")

    stats = {}  # pattern_key -> counts

    # iterate per symbol
    for sym, g in raw.groupby("Symbol", sort=False):
        g = compute_basic_indicators(g)
        if g.empty or len(g) < 50:
            continue

        g = g.sort_values("Date").reset_index(drop=True)
        close = g["Close"].astype(float)
        rsi14 = g["RSI14"].astype(float)
        ret1 = g["RET1"].astype(float)
        vol_z = g["VOL_Z"].astype(float)
        macd_hist = g["MACD_HIST"].astype(float)

        # we need at least two past days (t-1) and 3 future days (t+3)
        # so loop from i=3 to len(g)-4 inclusive
        n = len(g)
        for i in range(3, n - 3):
            # yesterday and today returns
            r_prev = ret1.iloc[i - 1]
            r_today = ret1.iloc[i]
            rsi_today = rsi14.iloc[i]
            vol_today = vol_z.iloc[i]
            macd_today = macd_hist.iloc[i]
            price_today = close.iloc[i]

            # bucket each component
            b_rsi = bucket_rsi(rsi_today)
            b_ret_prev = bucket_ret(r_prev)
            b_ret_today = bucket_ret(r_today)
            b_vol = bucket_volz(vol_today)
            b_macd = bucket_macd(macd_today, price_today)

            if None in (b_rsi, b_ret_prev, b_ret_today, b_vol, b_macd):
                continue

            pattern_key = f"{b_rsi}|{b_macd}|{b_vol}|{b_ret_prev}|{b_ret_today}"

            # actual future outcomes vs today's close
            c_t = price_today
            c_t1 = close.iloc[i + 1]
            c_t2 = close.iloc[i + 2]
            c_t3 = close.iloc[i + 3]

            up_T1 = 1 if c_t1 > c_t else 0
            up_T2 = 1 if c_t2 > c_t else 0
            up_T3 = 1 if c_t3 > c_t else 0

            rec = stats.get(pattern_key)
            if rec is None:
                rec = {
                    "count": 0,
                    "up_T1": 0,
                    "up_T2": 0,
                    "up_T3": 0,
                }
            rec["count"] += 1
            rec["up_T1"] += up_T1
            rec["up_T2"] += up_T2
            rec["up_T3"] += up_T3
            stats[pattern_key] = rec

    # compute probabilities
    patterns = {}
    for key, rec in stats.items():
        c = rec["count"]
        if c == 0:
            continue
        patterns[key] = {
            "count": c,
            "up_T1": rec["up_T1"],
            "up_T2": rec["up_T2"],
            "up_T3": rec["up_T3"],
            "p_up_T1": rec["up_T1"] / c,
            "p_up_T2": rec["up_T2"] / c,
            "p_up_T3": rec["up_T3"] / c,
        }

    out = {
        "meta": {
            "description": "Pattern table built from discrete RSI/MACD/VOL/RET buckets for T+1/T+2/T+3.",
            "min_history_per_symbol": 50,
            "pattern_components": [
                "RSI bucket (today)",
                "MACD_HIST bucket (today, relative to price)",
                "VOL_Z bucket (today)",
                "RET1 bucket (yesterday)",
                "RET1 bucket (today)",
            ],
        },
        "patterns": patterns,
    }

    PATTERN_FILE.write_text(json.dumps(out, indent=2))
    print(f"Saved pattern stats with {len(patterns)} patterns to {PATTERN_FILE}")


if __name__ == "__main__":
    main()
