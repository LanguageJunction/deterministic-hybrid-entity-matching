#!/usr/bin/env python3
"""Download real organization records from OpenAlex, restricted to companies."""
import argparse, time
from pathlib import Path
import requests
import pandas as pd

BASE = "https://api.openalex.org/institutions"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--per-page", type=int, default=100)
    ap.add_argument("--out", default="data/real/openalex_companies.csv")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--sleep", type=float, default=0.1)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    rows=[]; cursor="*"
    while len(rows) < args.limit:
        params={
            "filter":"type:company",
            "per-page":min(args.per_page, args.limit-len(rows)),
            "cursor":cursor,
            "select":"id,ror,display_name,display_name_alternatives,display_name_acronyms,country_code,type",
        }
        if args.api_key: params["api_key"] = args.api_key
        r=requests.get(BASE, params=params, timeout=60); r.raise_for_status()
        data=r.json(); batch=data.get("results", [])
        if not batch: break
        for x in batch:
            alts=(x.get("display_name_alternatives") or []) + (x.get("display_name_acronyms") or [])
            rows.append({
                "entity_id": x.get("ror") or x.get("id"),
                "canonical_name": x.get("display_name"),
                "aliases":"|".join(dict.fromkeys(a for a in alts if a)),
                "country_code":x.get("country_code",""),
                "source":"OpenAlex",
            })
        cursor=data.get("meta",{}).get("next_cursor")
        if not cursor: break
        time.sleep(args.sleep)
    pd.DataFrame(rows).drop_duplicates("entity_id").to_csv(out,index=False)
    print(f"wrote {len(rows)} entities -> {out}")

if __name__ == "__main__": main()
