# Quant Pattern Logit Engine (NSE)

This repository:

- Reads `Data/NSE_symbols.csv` (list of NSE tickers, without `.NS`).
- Downloads / updates OHLCV from Jan-2000 for each symbol as `SYMBOL.NS` via yfinance.
- Stores:
  - `Data/Historical/<SYMBOL>.csv` — full history per symbol.
  - `Data/Today/<SYMBOL>.csv` — latest day only.
  - `Data/Historical.csv` — master file with all symbols combined.
- Trains a logistic regression model on technical indicators + pattern features
  (RSI divergences, double top/bottom, wedge-like structure).
- Builds static HTML pages:
  - `dist/index.html` — table of all stocks with next-day probability.
  - `dist/stocks/<SYMBOL>.html` — detailed page for each stock.

## First-time local run

```bash
pip install pandas numpy jinja2 scikit-learn yfinance

python scripts/sync_nse_data.py      # download NSE data
python scripts/train_logit.py        # train logistic model
python scripts/build_predictions_logit.py  # build HTML pages
```

Then open `dist/index.html` in your browser.

## Data format

`Data/NSE_symbols.csv`:

```csv
symbol
20MICRONS
21STCENMGM
360ONE
GOLD360
...
```

`Data/Historical.csv` will be created automatically.
