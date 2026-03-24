# rolling-skew-analyzer

Rolling skew calculator for stocks, with a **20-day default window** and a **custom rolling option**.

Computes two measures side-by-side so you can see the difference in practice.

---

## The two measures

### 1. Nonparametric skew (recommended for rolling use)

```
skew_np = (mean - median) / stdev
```

**Bounded in (−1, 1)** by the Hotelling-Solomons inequality: for any distribution,
`|mean − median| ≤ stdev`. This means the ratio can never exceed ±1.

**Why it matters for rolling windows:** the median is insensitive to a single extreme
return. When a large outlier enters or leaves the window, `skew_np` moves smoothly.

### 2. Moment-based skew (for comparison only)

```
skew_m = mean((r − mean)³) / stdev³
```

**Unbounded.** Because deviations are cubed, one extreme return dominates the entire
window. This creates a sharp spike (or drop) that persists for exactly `window` days
and then vanishes abruptly when that observation scrolls out — a pure sample-boundary
artifact with no economic content.

---

## Quick start

```bash
pip install yfinance pandas numpy matplotlib

# 20-day window, all tickers in data/stocks.csv
python skew.py

# Custom window
python skew.py --window 10
python skew.py --window 60

# Specific tickers
python skew.py --window 20 --ticker AAPL MSFT NVDA TSLA

# Save results to CSV
python skew.py --window 20 --csv

# Plot nonparametric vs moment-based side by side
python skew.py --window 20 --ticker AAPL --plot

# List available tickers
python skew.py --list-tickers
```

---

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--window` / `-w` | `20` | Rolling window in trading days |
| `--ticker` / `-t` | all in stocks.csv | One or more tickers |
| `--period` | `2y` | yfinance history period (e.g. `1y`, `5y`) |
| `--plot` | off | Save & show a comparison chart |
| `--csv` | off | Export per-ticker CSV files |
| `--list-tickers` | — | Print ticker list and exit |

---

## Repository structure

```
rolling-skew-analyzer/
├── skew.py          # main script
├── data/
│   └── stocks.csv   # 30 large-cap tickers with name & sector
├── .gitignore
└── README.md
```

---

## Requirements

```
yfinance >= 0.2
pandas >= 1.5
numpy >= 1.23
matplotlib >= 3.5   # only for --plot
```
