"""Macro G/I/P scoring (Growth / Inflation / Fiscal Policy).

Fetches monthly economic series from FRED, transforms them into indicator
signals, computes trailing z-scores, and maps scores to discrete states
and regimes
"""

import numpy as np
import pandas as pd

# pandas-datareader is only required when fetching live FRED data
# Import optional so the module can be imported for testing or static
# analysis without a network dependency; fetch_fred() will raise a
# ImportError if it is invoked and pandas-datareader is missing.
try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

# CONFIG

# How many years to use for trailing z-score standardization
LOOKBACK_YEARS_Z = 15
# Discretization thresholds applied to each smoothed score
G_THRESH = 0.25
I_THRESH = 0.25
P_THRESH = 0.25

# Convert lookback to months (used by the rolling z-score)
LOOKBACK_MONTHS = LOOKBACK_YEARS_Z * 12

# Publication lags: some series are released with a delay; allow shifting
# the raw signals conservatively to reflect these lags when computing scores.
LAG = {
    "CFNAI": 0,
    "UNRATE": 1,
    "RSAFS": 1,
    "CPIAUCSL": 1,
    "CPILFESL": 1,
    "INDPRO": 1,
    "FEDFUNDS": 0,
    "DGS10": 0,
    "TB3MS": 0,
    "BAA": 0,
    "T10YIE": 0,
} 


# HELPERS

def classify(score, thresh, up, down, neutral):
    """Discretize a continuous score into an ordinal label.

    Arguments:
    - score: numeric value to classify
    - thresh: positive threshold; values >thresh => `up`, <-thresh => `down`
    - up/down/neutral: strings returned for each bucket
    """
    if score > thresh:
        return up
    if score < -thresh:
        return down
    return neutral


def zscore(series, window):
    """Simple trailing z-score using a fixed-length rolling window.

    Uses ddof=0 to match population std; calling code handles min_periods.
    """
    mu = series.rolling(window).mean()
    sd = series.rolling(window).std(ddof=0)
    return (series - mu) / sd


# DATA DOWNLOAD (FRED)

def fetch_fred(start="1990-01-01"):
    """Download the canonical set of FRED series and return a monthly DataFrame.

    Notes:
    - We resample to month-end and take the last available observation to
      produce a consistent monthly index (matching typical macro release timing).
    - The function will raise ImportError if pandas-datareader is not installed.
    """
    if pdr is None:
        raise ImportError("pandas-datareader is required to fetch FRED data. Install it with `pip install pandas-datareader`.")

    fred = {
        "CFNAI": "CFNAI",
        "UNRATE": "UNRATE",
        "RSAFS": "RSAFS",
        "CPIAUCSL": "CPIAUCSL",
        "CPILFESL": "CPILFESL",
        "INDPRO": "INDPRO",
        "FEDFUNDS": "FEDFUNDS",
        "DGS10": "DGS10",
        "TB3MS": "TB3MS",
        "BAA": "BAA",
        "T10YIE": "T10YIE",
    }

    df = []
    for name, code in fred.items():
        # fetch each series individually (errors will surface clearly per ticker)
        s = pdr.DataReader(code, "fred", start)
        s.columns = [name]
        df.append(s)

    # concat into a single frame and align to month-end using last available value
    df = pd.concat(df, axis=1)
    df = df.resample("M").last()
    return df


# RAW SIGNALS

def build_raw_signals(df):
    """Create the raw transformed indicators used to form G/I/P scores.

    Each x_* column is designed to capture the intended economic impulse:
      - Growth: CFNAI level + short-run momentum, unemployment improvements,
        real retail momentum, industrial production YoY.
      - Inflation: YoY core CPI, short-run CPI momentum, breakevens (T10YIE).
      - Policy/FC: yield curve, real policy rate proxy, fed funds changes, credit spread.
    """
    x = pd.DataFrame(index=df.index)

    # ---- Growth ----
    # CFNAI: combine current level with a 3-month momentum term
    x["x_GROWTH"] = 0.6 * df["CFNAI"] + 0.4 * (df["CFNAI"] - df["CFNAI"].shift(3))

    # Unemployment rate: improvement is positive for growth, so negate the change
    x["x_UR"] = -(df["UNRATE"] - df["UNRATE"].shift(6))

    # Real retail sales momentum (log-difference over 6 months)
    real_rs = df["RSAFS"] / df["CPIAUCSL"]
    x["x_RS"] = np.log(real_rs) - np.log(real_rs.shift(6))

    # Industrial production YoY (log-difference over 12 months)
    x["x_IP"] = np.log(df["INDPRO"]) - np.log(df["INDPRO"].shift(12))

    # ---- Inflation ----
    # Core CPI YoY
    x["x_CPI_level"] = np.log(df["CPILFESL"]) - np.log(df["CPILFESL"].shift(12))

    # CPI 3-month annualized momentum minus 12m (captures recent acceleration)
    cpi_3m_ann = 4 * (np.log(df["CPILFESL"]) - np.log(df["CPILFESL"].shift(3)))
    x["x_CPI_mom"] = cpi_3m_ann - x["x_CPI_level"]

    # Breakeven change YoY
    x["x_BE"] = df["T10YIE"] - df["T10YIE"].shift(12)

    # ---- Policy / Financial Conditions ----
    # Nominal yield curve slope (10y minus 3m)
    x["x_YC"] = df["DGS10"] - df["TB3MS"]
    # Real policy rate proxy: fed funds minus CPI level (both in percent-ish units)
    x["x_RPR"] = df["FEDFUNDS"] - x["x_CPI_level"]
    # Recent policy tightening/loosening (6-month change)
    x["x_FF"] = df["FEDFUNDS"] - df["FEDFUNDS"].shift(6)
    # Credit spread proxy (BAA less 10y)
    x["x_CS"] = df["BAA"] - df["DGS10"]

    return x


# APPLY LAGS

def apply_lags(x):
    """Apply conservative publication lags to transformed indicators.

    Some series are published with a delay; applying a small shift prevents
    the model from using values that would not have been known at the time.
    """
    out = pd.DataFrame(index=x.index)
    for col in x.columns:
        # conservative: max lag among inputs
        if col in ["x_GROWTH"]:
            lag = 0
        elif col in ["x_UR", "x_RS", "x_IP", "x_CPI_level", "x_CPI_mom"]:
            # these indicators are based on slower-released series, use 1-month lag
            lag = 1
        elif col in ["x_RPR"]:
            lag = 1
        else:
            lag = 0
        out[col] = x[col].shift(lag)
    return out


# COMPUTE G, I, P

def compute_GIP(df):
    """High-level pipeline to compute G/I/P scores from monthly series.

    Steps:
    1. Build transformed raw signals from the FRED inputs
    2. Apply conservative publication lags
    3. Compute trailing z-scores using LOOKBACK_MONTHS
    4. Aggregate z-scores into G, I, P by simple averaging/weights
    5. Discretize using +/- thresholds
    """
    # 1) Transform inputs into indicator series
    raw = build_raw_signals(df)
    # 2) Apply small publication lags where appropriate
    avail = apply_lags(raw)

    # 3) Standardize each transformed indicator to a trailing z-score
    z = avail.apply(lambda s: zscore(s, LOOKBACK_MONTHS))

    # 4) Aggregate into composite scores (equal-weight within groups)
    G = z[["x_GROWTH", "x_UR", "x_RS", "x_IP"]].mean(axis=1)

    # Inflation mixes long-run level and short-run momentum (weights: 25/50/25)
    I = (
        0.25 * z["x_CPI_level"]
        + 0.50 * z["x_CPI_mom"]
        + 0.25 * z["x_BE"]
    )

    # Policy combines curve/inflation-adjusted policy/proc cyclical signals; sign conventions
    P = (
        -z["x_YC"]   # more negative (inverted) = tighter financial conditions
        + z["x_RPR"]
        + z["x_FF"]
        + z["x_CS"]
    ) / 4

    # 5) Build output frame and discretize
    out = pd.DataFrame(index=df.index)
    out["G"] = G
    out["I"] = I
    out["P"] = P

    # Use classify helper to bucket continuous scores into named states
    out["G_state"] = out["G"].apply(lambda v: classify(v, G_THRESH, "UP", "DOWN", "NEUTRAL"))
    out["I_state"] = out["I"].apply(lambda v: classify(v, I_THRESH, "UP", "DOWN", "NEUTRAL"))
    out["P_state"] = out["P"].apply(lambda v: classify(v, P_THRESH, "TIGHT", "EASY", "NEUTRAL"))

    return out


# RUN / quick demo

if __name__ == "__main__":
    # Small CLI/demo: fetch FRED and print the most recent 12 monthly rows.
    # Note: this requires network access and pandas-datareader to be installed.
    df = fetch_fred(start="1990-01-01")
    gip = compute_GIP(df)
    print(gip.tail(12))
