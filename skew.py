"""
Rolling Skew Analyzer np_skew = "
              f"{df['skew_np'].iloc[-1]:.3f}, "
              f"moment_skew = {df['skew_moment'].iloc[-1]:.3f}")
    return results


# ---------------------------------------------------------------------------
# Optional plot
# ---------------------------------------------------------------------------

def plot_skew(results: dict[str, pd.DataFrame], window: int) -> None:
    """Plot nonparametric vs moment-based skew side by side."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"Rolling Skew  (window = {window} days)\n"
        "Left: Nonparametric |bounds (-1,1)|     "
        "Right: Moment-based |unbounded|",
        fontsize=11,
    )

    for ax_row, (ticker, df) in zip(axes, results.items()):
        ax_np, ax_m = ax_row

        # nonparametric
        ax_np.plot(df.index, df["skew_np"], color="steelblue", lw=1)
        ax_np.axhline(0, color="black", lw=0.6, ls="--")
        ax_np.axhline(1, color="red", lw=0.6, ls=":")
        ax_np.axhline(-1, color="red", lw=0.6, ls=":")
        ax_np.set_ylim(-1.1, 1.1)
        ax_np.set_title(f"{ticker} — NP skew", fontsize=9)
        ax_np.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

        # moment-based
        ax_m.plot(df.index, df["skew_moment"], color="darkorange", lw=1)
        ax_m.axhline(0, color="black", lw=0.6, ls="--")
        ax_m.set_title(f"{ticker} — Moment skew", fontsize=9)
        ax_m.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    fig.autofmt_xdate()
    plt.tight_layout()
    out = Path("rolling_skew.png")
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out.resolve()}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--window", "-w",
        type=int,
        default=20,
        metavar="N",
        help="Rolling window in trading days (default: 20)",
    )
    p.add_argument(
        "--ticker", "-t",
        nargs="+",
        metavar="TICKER",
        help="One or more tickers to analyse (default: all in stocks.csv)",
    )
    p.add_argument(
        "--period",
        default="2y",
        metavar="PERIOD",
        help="yfinance period string, e.g. 1y 2y 5y (default: 2y)",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Show and save a comparison chart (requires matplotlib)",
    )
    p.add_argument(
        "--list-tickers",
        action="store_true",
        help="Print available tickers from data/stocks.csv and exit",
    )
    p.add_argument(
        "--csv",
        action="store_true",
        help="Export results to CSV files in the current directory",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    stock_df = load_ticker_list()

    if args.list_tickers:
        print(stock_df.to_string(index=False))
        return

    tickers = args.ticker if args.ticker else stock_df["ticker"].tolist()
    window = args.window

    if window < 3:
        parser.error("--window must be at least 3")

    results = compute_rolling_skew(tickers, window=window, period=args.period)

    if not results:
        sys.exit("No data retrieved. Check tickers or network connection.")

    if args.csv:
        for ticker, df in results.items():
            fname = f"{ticker}_skew_w{window}.csv"
            df.to_csv(fname)
            print(f"  Saved {fname}")

    if args.plot:
        plot_skew(results, window)


if __name__ == "__main__":
    main()

=====================
Computes two rolling skew measures for a list of stocks:

  1. Nonparametric (Pearson median) skew  [RECOMMENDED]
        skew_np = (mean - median) / stdev

     Bounded in (-1, 1) by the Hotelling-Solomons inequality
     |mean - median| <= stdev.  Robust to outliers because the
     median does not react strongly to a single extreme return.

  2. Moment-based skew  [for comparison only]
        skew_m = mean((r - mean)^3) / stdev^3

     Unbounded and highly sensitive to outliers: one large return
     dominates the whole window and causes a sharp spike/drop that
     persists for exactly `window` days then vanishes abruptly.

Usage
-----
    python skew.py                        # 20-day window, all tickers
    python skew.py --window 10            # custom window
    python skew.py --window 60 --ticker AAPL MSFT NVDA
    python skew.py --list-tickers         # print available tickers

Requirements
------------
    pip install yfinance pandas numpy matplotlib
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
STOCKS_CSV = DATA_DIR / "stocks.csv"

# ---------------------------------------------------------------------------
# Core skew functions
# ---------------------------------------------------------------------------

def rolling_np_skew(returns: pd.Series, window: int) -> pd.Series:
    """
    Nonparametric (Pearson median) rolling skew.

        skew_np = (mean - median) / stdev

    Bounded in (-1, 1).  Robust to single extreme observations.
    Requires at least 3 observations; earlier values are NaN.
    """
    def _np_skew(x):
        if len(x) < 3:
            return np.nan
        m, med, s = x.mean(), np.median(x), x.std(ddof=1)
        return np.nan if s == 0 else (m - med) / s

    return returns.rolling(window=window, min_periods=3).apply(_np_skew, raw=True)


def rolling_moment_skew(returns: pd.Series, window: int) -> pd.Series:
    """
    Moment-based (Fisher) rolling skew.

        skew_m = mean((r - mean)^3) / stdev^3

    Unbounded.  Sensitive to outliers: a single large return
    dominates the estimate for `window` days.
    Requires at least 3 observations; earlier values are NaN.
    """
    def _moment_skew(x):
        if len(x) < 3:
            return np.nan
        s = x.std(ddof=1)
        return np.nan if s == 0 else ((x - x.mean()) ** 3).mean() / s ** 3

    return returns.rolling(window=window, min_periods=3).apply(_moment_skew, raw=True)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_ticker_list() -> pd.DataFrame:
    """Load the stock list from data/stocks.csv."""
    if not STOCKS_CSV.exists():
        raise FileNotFoundError(f"Stock list not found: {STOCKS_CSV}")
    return pd.read_csv(STOCKS_CSV)


def fetch_returns(tickers: list[str], period: str = "2y") -> pd.DataFrame:
    """
    Download adjusted close prices via yfinance and return daily log-returns.
    Falls back to pct_change if log-returns fail.
    """
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is required.  Install it with:  pip install yfinance")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw

    prices = prices[tickers] if set(tickers).issubset(prices.columns) else prices
    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    return returns


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_rolling_skew(
    tickers: list[str],
    window: int = 20,
    period: str = "2y",
