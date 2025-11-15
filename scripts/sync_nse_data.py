# scripts/sync_nse_data.py
#
# Incremental NSE data sync with:
#   - Weekend + fixed holiday calendar shortcut for 2025
#   - Probe-symbol shortcut for other unexpected non-trading days
#
# Behaviour:
#   * If today is Saturday/Sunday or one of the configured NSE holidays -> exit fast.
#   * Else use one probe symbol to check if a new bar exists.
#   * If probe has no new bar -> treat as holiday and exit.
#   * Otherwise incrementally update all symbols.

from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

DATA_DIR = Path("Data")
SYMBOLS_FILE = DATA_DIR / "NSE_symbols.csv"

HIST_DIR = DATA_DIR / "Historical"
TODAY_DIR = DATA_DIR / "Today"
MASTER_FILE = DATA_DIR / "Historical.csv"

START_DATE = date(2000, 1, 1)

for d in (HIST_DIR, TODAY_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- NSE holiday calendar (2025) ----------

HOLIDAYS_NSE_2025 = {
    # 1  Mahashivratri
    date(2025, 2, 26),
    # 2  Holi
    date(2025, 3, 14),
    # 3  Id-Ul-Fitr (Ramadan Eid)
    date(2025, 3, 31),
    # 4  Shri Mahavir Jayanti
    date(2025, 4, 10),
    # 5  Dr. Baba Saheb Ambedkar Jayanti
    date(2025, 4, 14),
    # 6  Good Friday
    date(2025, 4, 18),
    # 7  Maharashtra Day
    date(2025, 5, 1),
    # 8  Independence Day / Parsi New Year
    date(2025, 8, 15),
    # 9  Shri Ganesh Chaturthi
    date(2025, 8, 27),
    # 10 Mahatma Gandhi Jayanti / Dussehra
    date(2025, 10, 2),
    # 11 Diwali Laxmi Pujan
    date(2025, 10, 21),
    # 12 Balipratipada
    date(2025, 10, 22),
    # 13 Prakash Gurpurb Sri Guru Nanak Dev
    date(2025, 11, 5),
    # 14 Christmas
    date(2025, 12, 25),
}


def is_nse_holiday(d: date) -> bool:
    """Return True if 'd' is a configured NSE non-trading day."""
    # Saturday (5) or Sunday (6)
    if d.weekday() >= 5:
        return True
    # Year-specific exchange holidays
    if d in HOLIDAYS_NSE_2025:
        return True
    return False


# ---------- helpers for data handling ----------

def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure the dataframe has a 'Date' column."""
    if df is None or df.empty:
        return df

    if "Date" in df.columns:
        return df

    for cand in ("Datetime", "DATE", "date", "Index", "index"):
        if cand in df.columns:
            return df.rename(columns={cand: "Date"})

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if "index" in df.columns:
            df = df.rename(columns={"index": "Date"})

    return df


def download_incremental(ticker: str, start: date, today: date) -> pd.DataFrame:
    """Download OHLCV from yfinance in [start, today] inclusive."""
    return yf.download(
        ticker,
        start=start.isoformat(),
        end=(today + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
    )


def download_and_update_symbol(symbol: str, today: date) -> pd.DataFrame:
    """Update one symbol. Only downloads data after last saved date."""
    ticker = f"{symbol}.NS"
    hist_path = HIST_DIR / f"{symbol}.csv"

    # -------- load existing history (if present) --------
    if hist_path.exists():
        old = pd.read_csv(hist_path, parse_dates=["Date"])
        old = _ensure_date_column(old)
        old = old.sort_values("Date")
        last_date = old["Date"].max().date()

        # if already up-to-date for this symbol, just reuse
        if last_date >= today:
            df_all = old
            print(f"{symbol}: already up to date (last date {last_date})")
        else:
            start = last_date + timedelta(days=1)
            print(f"{symbol}: updating from {start} to {today}")
            df_new = download_incremental(ticker, start, today)

            if df_new.empty:
                print(f"{symbol}: no new rows from yfinance")
                df_all = old
            else:
                df_new = df_new.reset_index()
                df_new = _ensure_date_column(df_new)

                df_new["Symbol"] = symbol
                df_new["Sector"] = ""   # optional – can be filled later
                df_new["Index"] = "NSE"

                df_new = df_new[[
                    "Symbol", "Date", "Open", "High", "Low", "Close", "Volume",
                    "Sector", "Index"
                ]]

                df_all = pd.concat([old, df_new], ignore_index=True)

    else:
        # no local history: download full series from START_DATE
        print(f"{symbol}: no local history, downloading full series from {START_DATE}")
        df_new = download_incremental(ticker, START_DATE, today)

        if df_new.empty:
            print(f"{symbol}: no data from yfinance at all, skipping")
            return pd.DataFrame()

        df_new = df_new.reset_index()
        df_new = _ensure_date_column(df_new)

        df_new["Symbol"] = symbol
        df_new["Sector"] = ""
        df_new["Index"] = "NSE"

        df_all = df_new[[
            "Symbol", "Date", "Open", "High", "Low", "Close", "Volume",
            "Sector", "Index"
        ]]

    # -------- final clean-up, save history & today's row --------
    df_all = _ensure_date_column(df_all)

    if df_all is None or df_all.empty or "Date" not in df_all.columns:
        print(f"{symbol}: skipping after update, invalid dataframe")
        return pd.DataFrame()

    df_all = df_all.sort_values("Date")
    df_all = df_all.loc[~df_all["Date"].duplicated()].copy()

    df_all.to_csv(hist_path, index=False)

    # latest row -> Today/<symbol>.csv
    last_row = df_all.iloc[-1:]
    TODAY_DIR.mkdir(parents=True, exist_ok=True)
    last_row.to_csv(TODAY_DIR / f"{symbol}.csv", index=False)

    print(f"{symbol}: {len(df_all)} total rows, last date {last_row['Date'].iloc[0].date()}")
    return df_all


def probe_non_trading_day(symbol: str, today: date) -> bool:
    """
    Check a single 'probe' symbol to decide whether today has a new bar.
    Returns True if today looks like a non-trading day (holiday/weekend),
    False otherwise.
    """
    hist_path = HIST_DIR / f"{symbol}.csv"
    ticker = f"{symbol}.NS"

    if not hist_path.exists():
        # No history yet: can't detect holiday, let normal loop handle it.
        print(f"Probe: {symbol} has no local history, skipping probe shortcut.")
        return False

    probe_df = pd.read_csv(hist_path, parse_dates=["Date"])
    probe_df = _ensure_date_column(probe_df)
    probe_df = probe_df.sort_values("Date")
    last_date = probe_df["Date"].max().date()

    # if we already have today's bar for probe, we definitely should update others
    if last_date >= today:
        print(f"Probe: {symbol} already has data for {last_date}, not a holiday.")
        return False

    start = last_date + timedelta(days=1)
    print(f"Probe: checking {symbol} from {start} to {today} for holiday detection...")
    df_new = download_incremental(ticker, start, today)

    if df_new.empty:
        print(f"Probe: no new rows for {symbol}. Assuming {today} is a non-trading day.")
        return True

    print(f"Probe: found new rows for {symbol}. Market is open, continuing full update.")
    return False


def main():
    if not SYMBOLS_FILE.exists():
        raise SystemExit(f"Missing {SYMBOLS_FILE}")

    today = date.today()

    # ---------- HARD holiday/weekend shortcut ----------
    if is_nse_holiday(today):
        print(f"{today} is configured NSE non-trading day (weekend or exchange holiday).")
        print("Skipping full NSE update.")
        return

    # ---------- load symbol universe ----------
    syms = pd.read_csv(SYMBOLS_FILE)["symbol"].dropna().astype(str).unique()
    syms = [s.strip().upper() for s in syms if s.strip()]
    if not syms:
        raise SystemExit("No symbols found in NSE_symbols.csv")

    # ---------- Probe shortcut for unexpected non-trading days ----------
    probe_symbol = syms[0]
    try:
        if probe_non_trading_day(probe_symbol, today):
            print("Probe detected non-trading day. Skipping full NSE update.")
            return
    except Exception as e:
        print(f"Probe failed with error {e!r}, continuing with full update loop.")

    # ---------- Normal incremental update loop ----------
    all_frames = []
    for s in syms:
        df = download_and_update_symbol(s, today)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        raise SystemExit("No data for any symbol – check NSE_symbols.csv / internet")

    master = pd.concat(all_frames, ignore_index=True)
    master = master.sort_values(["Symbol", "Date"])
    master.to_csv(MASTER_FILE, index=False)

    print(f"Wrote master file with {len(master)} rows to {MASTER_FILE}")


if __name__ == "__main__":
    main()
