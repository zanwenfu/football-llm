import argparse
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"

def parse_dt_to_unix(s: str) -> int:
    """
    Accepts:
      '2020-01-01'
      '2020-01-01 13:30:00'
      '2020-01-01T13:30:00'
    Returns unix seconds (UTC).
    """
    s = s.strip().replace("T", " ")
    if len(s) == 10:
        s += " 00:00:00"
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

def fetch_kraken_ohlc_page(pair: str, interval: int, since: int) -> Dict[str, Any]:
    params = {"pair": pair, "interval": interval, "since": since}
    r = requests.get(KRAKEN_OHLC_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken API error: {data['error']}")
    return data["result"]

def download_kraken_ohlcv(
    pair: str,
    interval: int,
    start: str,
    end: Optional[str] = None,
    out_csv: str = "ohlcv_kraken.csv",
    sleep_s: float = 0.25,
) -> pd.DataFrame:
    """
    Downloads OHLCV candles from Kraken public API and saves to CSV.

    Kraken returns:
      [time, open, high, low, close, vwap, volume, count]
    Pagination:
      use returned 'last' as next since cursor.
    """
    start_unix = parse_dt_to_unix(start)
    end_unix = parse_dt_to_unix(end) if end else None

    all_rows: List[List[Any]] = []
    since = start_unix
    seen_last = None

    while True:
        result = fetch_kraken_ohlc_page(pair=pair, interval=interval, since=since)

        # The OHLC data is under a dynamic key (pair name), and 'last' cursor is separate
        last_cursor = int(result["last"])
        pair_keys = [k for k in result.keys() if k != "last"]
        if not pair_keys:
            break
        ohlc_key = pair_keys[0]
        rows = result[ohlc_key]

        if not rows:
            break

        # Filter rows to [start, end) if end provided
        if end_unix is not None:
            rows = [row for row in rows if int(row[0]) < end_unix]

        all_rows.extend(rows)

        # Stop conditions
        if end_unix is not None and all_rows and int(all_rows[-1][0]) >= end_unix:
            break

        # If cursor doesn't move, stop to avoid infinite loop
        if seen_last is not None and last_cursor == seen_last:
            break
        seen_last = last_cursor

        # Advance
        since = last_cursor
        time.sleep(sleep_s)

        # If we got filtered to empty due to end_unix, we can stop
        if end_unix is not None and not rows:
            break

    if not all_rows:
        raise RuntimeError("No OHLC data returned. Check pair/interval/date range.")

    df = pd.DataFrame(
        all_rows,
        columns=["time_unix", "open", "high", "low", "close", "vwap", "volume", "count"],
    )

    # types
    df["time_unix"] = df["time_unix"].astype(int)
    for c in ["open", "high", "low", "close", "vwap", "volume"]:
        df[c] = df[c].astype(float)
    df["count"] = df["count"].astype(int)

    df["time"] = pd.to_datetime(df["time_unix"], unit="s", utc=True)

    # neat column order
    df = df[["time", "open", "high", "low", "close", "volume", "vwap", "count"]]

    df.to_csv(out_csv, index=False)
    return df

def main():
    parser = argparse.ArgumentParser(description="Download OHLCV data via Kraken public API.")
    parser.add_argument("--pair", type=str, default="XBTUSD",
                        help="Kraken trading pair, e.g., XBTUSD, XBTUSDT, ETHUSD, SOLUSD")
    parser.add_argument("--interval", type=int, default=1440,
                        help="Candle interval in minutes. Common: 1,5,15,60,240,1440")
    parser.add_argument("--start", type=str, default="2017-01-01",
                        help="Start datetime (UTC): 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--end", type=str, default=None,
                        help="End datetime (UTC): same format as start. If omitted, downloads to latest.")
    parser.add_argument("--out", type=str, default="ohlcv_kraken.csv",
                        help="Output CSV filename")
    args = parser.parse_args()

    df = download_kraken_ohlcv(
        pair=args.pair,
        interval=args.interval,
        start=args.start,
        end=args.end,
        out_csv=args.out,
    )
    print(df.head())
    print(f"\nSaved {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
