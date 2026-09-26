#!/usr/bin/env python3
"""Download a manageable real-company master from SEC EDGAR.

Uses SEC's public company_tickers.json plus each issuer's submissions JSON.
The SEC API is public and does not require an API key, but callers should
identify themselves with a descriptive User-Agent and respect SEC rate limits.
"""
import argparse, json, time
from pathlib import Path
import requests
import pandas as pd

SEC = "https://www.sec.gov"
HEADERS_BASE = {"Accept-Encoding": "gzip, deflate"}

def get_json(url, user_agent, timeout=60):
    headers = dict(HEADERS_BASE)
    headers["User-Agent"] = user_agent
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--out", default="data/real/sec_companies.csv")
    ap.add_argument("--user-agent", required=True, help="e.g. name@example.com")
    ap.add_argument("--sleep", type=float, default=0.15)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    tickers = get_json(f"{SEC}/files/company_tickers.json", args.user_agent)
    rows = list(tickers.values())[:args.limit]
    records = []
    for i, item in enumerate(rows, 1):
        cik = int(item["cik_str"])
        sub = get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", args.user_agent)
        current = sub.get("name") or item.get("title")
        former = [x.get("name") for x in sub.get("formerNames", []) if x.get("name")]
        entity_id = f"SEC:{cik:010d}"
        records.append({
            "entity_id": entity_id,
            "canonical_name": current,
            "aliases": "|".join(dict.fromkeys(former)),
            "cik": cik,
            "ticker": item.get("ticker", ""),
            "exchange": item.get("exchange", ""),
            "source": "SEC",
        })
        if i % 100 == 0:
            print(f"downloaded {i}/{len(rows)}")
        time.sleep(args.sleep)
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"wrote {len(records)} entities -> {out}")

if __name__ == "__main__": main()
