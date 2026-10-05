"""Treasury curve inputs: checked-in FRED cache, optional refresh, synthetic fallback.

The engine always runs offline. ``run_all`` reads the checked-in cache
``data/cache/fred_treasury_cmt.csv`` (U.S. Treasury constant-maturity par
yields from the Federal Reserve H.15 release, retrieved from FRED). Refreshing
the cache is an explicit, separate action::

    python -m firisk.data --refresh

If the cache file is missing, a clearly labeled *synthetic* history generated
from a seeded three-factor model is used instead so the pipeline still runs.
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from firisk.config import DATA_DIR

CACHE_FILE = DATA_DIR / "cache" / "fred_treasury_cmt.csv"
META_FILE = DATA_DIR / "cache" / "fred_treasury_cmt.meta.json"

# Tenor label -> (FRED series id, tenor in years)
SERIES: dict[str, tuple[str, float]] = {
    "1M": ("DGS1MO", 1 / 12),
    "3M": ("DGS3MO", 0.25),
    "6M": ("DGS6MO", 0.5),
    "1Y": ("DGS1", 1.0),
    "2Y": ("DGS2", 2.0),
    "3Y": ("DGS3", 3.0),
    "5Y": ("DGS5", 5.0),
    "7Y": ("DGS7", 7.0),
    "10Y": ("DGS10", 10.0),
    "20Y": ("DGS20", 20.0),
    "30Y": ("DGS30", 30.0),
}
TENOR_LABELS: list[str] = list(SERIES)
TENOR_YEARS: np.ndarray = np.array([SERIES[k][1] for k in TENOR_LABELS])

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}"


@dataclass(frozen=True)
class CurveHistory:
    """Daily par-yield history in percent, one column per tenor label."""

    frame: pd.DataFrame
    source: str  # "FRED cache" or "synthetic"

    def par_yields_on(self, when: date) -> np.ndarray:
        """Par yields (decimal) on ``when``; raises if that date is not present."""
        ts = pd.Timestamp(when)
        if ts not in self.frame.index:
            raise KeyError(f"{when} not in curve history ({self.source})")
        row = self.frame.loc[ts, TENOR_LABELS]
        if row.isna().any():
            raise ValueError(f"incomplete curve on {when}: {row[row.isna()].index.tolist()}")
        return row.to_numpy(dtype=float) / 100.0


def fetch_fred(start: str = "2016-01-01", timeout: float = 30.0) -> pd.DataFrame:
    """Download every tenor from FRED and merge on date (network required)."""
    cols = []
    for label, (sid, _) in SERIES.items():
        url = FRED_URL.format(sid=sid, start=start)
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed host
            raw = resp.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(raw), na_values=["."])
        df.columns = ["date", label]
        df["date"] = pd.to_datetime(df["date"])
        cols.append(df.set_index("date"))
    merged = pd.concat(cols, axis=1).sort_index()
    return merged.dropna(how="all")


def refresh_cache(start: str = "2016-01-01") -> Path:
    frame = fetch_fred(start)
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CACHE_FILE, index_label="date", float_format="%.2f", lineterminator="\n")
    meta = {
        "source": "Board of Governors of the Federal Reserve System, H.15 Selected Interest "
        "Rates, via FRED (Federal Reserve Bank of St. Louis)",
        "series": {k: v[0] for k, v in SERIES.items()},
        "units": "percent, constant-maturity par yield (bond-equivalent basis)",
        "start": start,
        "first_date": str(frame.index[0].date()),
        "last_date": str(frame.index[-1].date()),
        "rows": int(len(frame)),
        "retrieved_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    META_FILE.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    return CACHE_FILE


def synthetic_history(n_days: int = 2500, seed: int = 20260930) -> pd.DataFrame:
    """SYNTHETIC curve history from a seeded level/slope/curvature model.

    Used only when the FRED cache is absent. Not market data.
    """
    rng = np.random.default_rng(seed)
    tau = 2.0
    t = TENOR_YEARS
    slope_load = (1 - np.exp(-t / tau)) / (t / tau)
    curv_load = slope_load - np.exp(-t / tau)
    level, slope, curv = 4.0, -1.0, 0.5
    rows = []
    for _ in range(n_days):
        level += rng.normal(0, 0.055)
        slope += rng.normal(0, 0.035) - 0.002 * (slope + 1.0)
        curv += rng.normal(0, 0.03) - 0.003 * (curv - 0.5)
        y = level + slope * slope_load + curv * curv_load + rng.normal(0, 0.008, size=t.size)
        rows.append(np.maximum(y, 0.01))
    idx = pd.bdate_range("2016-01-04", periods=n_days)
    return pd.DataFrame(np.round(rows, 2), index=idx, columns=TENOR_LABELS)


def load_history(path: Path = CACHE_FILE) -> CurveHistory:
    if path.exists():
        frame = pd.read_csv(path, parse_dates=["date"], index_col="date")
        return CurveHistory(frame[TENOR_LABELS], "FRED cache")
    return CurveHistory(synthetic_history(), "synthetic")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="re-download the FRED cache")
    ap.add_argument("--start", default="2016-01-01")
    args = ap.parse_args()
    if args.refresh:
        print(f"wrote {refresh_cache(args.start)}")
    else:
        h = load_history()
        print(f"{h.source}: {len(h.frame)} rows, {h.frame.index[0].date()} .. {h.frame.index[-1].date()}")


if __name__ == "__main__":
    main()
