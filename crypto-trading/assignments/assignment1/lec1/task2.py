import argparse
import io
import requests
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

BASE_URL = "https://api.blockchain.info/charts"

def parse_dt_utc(s: str) -> datetime:
    """
    Accepts:
      '2020-01-01'
      '2020-01-01 13:30:00'
      '2020-01-01T13:30:00'
    Returns timezone-aware UTC datetime.
    """
    s = s.strip().replace("T", " ")
    if len(s) == 10:
        s += " 00:00:00"
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    return dt.replace(tzinfo=timezone.utc)

def fetch_chart_raw(chart: str, timespan: str, fmt: str) -> str:
    """
    Fetch raw chart data from Blockchain.com charts endpoint.
    fmt: 'json' or 'csv'
    """
    url = f"{BASE_URL}/{chart}"
    params = {
        "timespan": timespan,
        "format": fmt,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.text

def parse_chart_json_to_df(text: str) -> pd.DataFrame:
    """
    Blockchain charts JSON looks like:
      { "status": "...", "name": "...", "values": [{"x": 1234567890, "y": 123.4}, ...], ... }
    where x is unix seconds.
    """
    data = requests.models.complexjson.loads(text)
    if "values" not in data:
        raise RuntimeError(f"Unexpected JSON format. Keys: {list(data.keys())}")
    df = pd.DataFrame(data["values"])
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["x"], unit="s", utc=True)
    df = df.rename(columns={"y": "value"})[["time", "value"]]
    return df

def parse_chart_csv_to_df(text: str) -> pd.DataFrame:
    """
    CSV usually like:
      Timestamp,Value
      2012-01-01, ...
    We try to parse robustly.
    """
    df = pd.read_csv(io.StringIO(text))
    # Normalize column names
    cols_lower = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols_lower)

    # Common possibilities: 'timestamp' or 'date' + 'value'
    time_col = None
    for c in ["timestamp", "date", "time"]:
        if c in df.columns:
            time_col = c
            break
    if time_col is None or "value" not in df.columns:
        raise RuntimeError(f"Unexpected CSV columns: {df.columns.tolist()}")

    df["time"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).copy()
    df = df[["time", "value"]]
    return df

def filter_df_by_range(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if df.empty:
        return df
    if start:
        sdt = parse_dt_utc(start)
        df = df[df["time"] >= sdt]
    if end:
        edt = parse_dt_utc(end)
        df = df[df["time"] < edt]
    return df

def download_blockchain_chart(
    chart: str,
    timespan: str,
    fmt: str,
    start: Optional[str],
    end: Optional[str],
    out: str,
) -> pd.DataFrame:
    raw = fetch_chart_raw(chart=chart, timespan=timespan, fmt=fmt)

    # Parse
    if fmt == "json":
        df = parse_chart_json_to_df(raw)
    elif fmt == "csv":
        df = parse_chart_csv_to_df(raw)
    else:
        raise ValueError("fmt must be 'json' or 'csv'")

    # Optional local filtering
    df = filter_df_by_range(df, start, end)

    # Save
    df.to_csv(out, index=False)
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Download Blockchain.com Explorer chart data via API."
    )
    parser.add_argument(
        "--chart",
        type=str,
        default="market-price",
        help="Chart name from https://www.blockchain.com/explorer/charts (e.g., hash-rate, difficulty, n-transactions, market-price)",
    )
    parser.add_argument(
        "--timespan",
        type=str,
        default="all",
        help="Timespan string (e.g., all, 5years, 365days, 30days)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv"],
        default="json",
        help="API response format to request",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional start datetime (UTC): 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional end datetime (UTC): same format as start",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV filename (default auto-named).",
    )
    args = parser.parse_args()

    out = args.out or f"blockchain_{args.chart}_{args.timespan}.csv"
    df = download_blockchain_chart(
        chart=args.chart,
        timespan=args.timespan,
        fmt=args.format,
        start=args.start,
        end=args.end,
        out=out,
    )
    print(df.head())
    print(f"\nSaved {len(df)} rows to {out}")

if __name__ == "__main__":
    main()
