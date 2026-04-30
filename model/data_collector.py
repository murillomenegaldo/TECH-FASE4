import yfinance as yf
import pandas as pd
import os


DEFAULT_SYMBOL = "PETR4.SA"
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2024-12-31"


def download_stock_data(
    symbol: str = DEFAULT_SYMBOL,
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Download historical OHLCV data from Yahoo Finance."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)

    if df.empty:
        raise ValueError(f"No data returned for symbol '{symbol}' in the given date range.")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.sort_index(inplace=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)
        print(f"Data saved to {save_path}")

    print(f"Downloaded {len(df)} rows for {symbol} ({start_date} → {end_date})")
    return df


def load_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    return df
