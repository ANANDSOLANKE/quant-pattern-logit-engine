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


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Try to guarantee there is a column called 'Date'."""
    if df is None or df.empty:
        return df

    if "Date" in df.columns:
        return df

    # Common alternatives
    for cand in ("Datetime", "DATE", "date", "Index", "index"):
        if cand in df.columns:
            return df.rename(columns={cand: "Date"})

    # If index is datetime, move it to a column
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if "index" in df.columns:
            df = df.rename(columns={"index": "Date"})
    return df


def download_and_update_symbol(symbol: str) -> pd.DataFrame:
    ticker = f"{symbol}.NS"
    hist_path = HIST_DIR / f"{symbol}.csv"

    # -------- load existing history, if any --------
    if hist_path.exists():
        old = pd.read_csv(hist_path, parse_dates=["Date"])
        old = _ensure_date_column(old)
        old = old.sort_values("Date")
        last_date = old["Date"].max().date()
        start = last_date + timedelta(days=1)
    else:
        old = None
        start = START_DATE

    # -------- download new data with yfinance --------
    today = date.today()
    if start > today:
        # nothing to update
        if old is None:
            return pd.DataFrame()
        df_all = old
    else:
        df_new = yf.download(
            ticker,
            start=start.isoformat(),
            end=(today + timedelta(days=1)).isoformat(),
            auto_adjust=False,
            progress=False,
        )

        if df_new.empty and old is None:
            print(f"No data for {symbol} ({ticker})")
            return pd.DataFrame()

        if not df_new.empty:
            df_new = df_new.reset_index()
            df_new = _ensure_date_column(df_new)

            df_new["Symbol"] = symbol
            df_new["Sector"] = ""      # you can fill this later
            df_new["Index"] = "NSE"

            df_new = df_new[[
                "Symbol", "Date", "Open", "High", "Low", "Close", "Volume",
                "Sector", "Index"
            ]]

        # merge old and new
        if old is not None:
            if not df_new.empty:
                df_all = pd.concat([old, df_new], ignore_index=True)
            else:
                df_all = old
        else:
            df_all = df_new

    # -------- final clean-up & save --------
    df_all = _ensure_date_column(df_all)

    # SAFETY: if still no Date or empty, skip this symbol
    if df_all is None or df_all.empty:
        print(f"{symbol}: skipping, dataframe empty")
        return pd.DataFrame()
    if "Date" not in df_all.columns:
        print(f"{symbol}: skipping, no 'Date' column. cols={list(df_all.columns)}")
        return pd.DataFrame()

    # sort and remove duplicates without using subset=["Date"]
    try:
        df_all = df_all.sort_values("Date")
        df_all = df_all.loc[~df_all["Date"].duplicated()].copy()
    except Exception as e:
        print(f"{symbol}: error cleaning by Date: {e}, cols={list(df_all.columns)}")
        return pd.DataFrame()

    df_all.to_csv(hist_path, index=False)

    # last row for Today folder
    last_row = df_all.iloc[-1:]
    TODAY_DIR.mkdir(parents=True, exist_ok=True)
    last_row.to_csv(TODAY_DIR / f"{symbol}.csv", index=False)

    print(f"{symbol}: {len(df_all)} rows")
    return df_all


def main():
    if not SYMBOLS_FILE.exists():
        raise SystemExit(f"Missing {SYMBOLS_FILE}")

    syms = pd.read_csv(SYMBOLS_FILE)["symbol"].dropna().astype(str).unique()

    all_frames = []
    for s in syms:
        s = s.strip().upper()
        if not s:
            continue
        df = download_and_update_symbol(s)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        raise SystemExit("No data downloaded; check symbols or internet")

    master = pd.concat(all_frames, ignore_index=True)
    master = master.sort_values(["Symbol", "Date"])
    master.to_csv(MASTER_FILE, index=False)
    print(f"Wrote master file with {len(master)} rows to {MASTER_FILE}")


if __name__ == "__main__":
    main()
