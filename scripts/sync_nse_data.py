# scripts/sync_nse_data.py

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


def download_and_update_symbol(symbol: str) -> pd.DataFrame:
    """Update one symbol. Only downloads data after last saved date."""
    ticker = f"{symbol}.NS"
    hist_path = HIST_DIR / f"{symbol}.csv"

    today = date.today()

    # -------- load existing history (if present) --------
    if hist_path.exists():
        old = pd.read_csv(hist_path, parse_dates=["Date"])
        old = _ensure_date_column(old)
        old = old.sort_values("Date")
        last_date = old["Date"].max().date()

        # already up-to-date: no download, just reuse
        if last_date >= today:
            df_all = old
            print(f"{symbol}: already up to date (last date {last_date})")
        else:
            start = last_date + timedelta(days=1)
            print(f"{symbol}: updating from {start} to {today}")

            df_new = yf.download(
                ticker,
                start=start.isoformat(),
                end=(today + timedelta(days=1)).isoformat(),
                auto_adjust=False,
                progress=False,
            )

            if df_new.empty:
                print(f"{symbol}: no new rows from yfinance")
                df_all = old
            else:
                df_new = df_new.reset_index()
                df_new = _ensure_date_column(df_new)

                df_new["Symbol"] = symbol
                df_new["Sector"] = ""   # optional – fill later if you have sector data
                df_new["Index"] = "NSE"

                df_new = df_new[[
                    "Symbol", "Date", "Open", "High", "Low", "Close", "Volume",
                    "Sector", "Index"
                ]]

                df_all = pd.concat([old, df_new], ignore_index=True)

    else:
        # no local history: fall back to full download from START_DATE
        print(f"{symbol}: no local history, downloading full series from {START_DATE}")
        df_new = yf.download(
            ticker,
            start=START_DATE.isoformat(),
            end=(today + timedelta(days=1)).isoformat(),
            auto_adjust=False,
            progress=False,
        )

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
        raise SystemExit("No data for any symbol – check NSE_symbols.csv / internet")

    master = pd.concat(all_frames, ignore_index=True)
    master = master.sort_values(["Symbol", "Date"])
    master.to_csv(MASTER_FILE, index=False)

    print(f"Wrote master file with {len(master)} rows to {MASTER_FILE}")


if __name__ == "__main__":
    main()
