#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path
import requests
from bs4 import BeautifulSoup

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def get(url, params, email=None, api_key=None):
    p = dict(params)
    if email: p["email"] = email
    if api_key: p["api_key"] = api_key
    r = requests.get(url, params=p, timeout=60)
    r.raise_for_status()
    return r.text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default='(Johnson Johnson OR 3M OR Pfizer OR Abbott) AND affiliation[ad]')
    ap.add_argument("--retmax", type=int, default=500)
    ap.add_argument("--out", default="data/pubmed/affiliations.jsonl")
    ap.add_argument("--email")
    ap.add_argument("--api-key")
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    xml = get(f"{BASE}/esearch.fcgi", {"db":"pubmed","term":args.query,"retmax":args.retmax,"retmode":"json"}, args.email, args.api_key)
    ids = __import__('json').loads(xml)["esearchresult"]["idlist"]
    if not ids:
        print("No PubMed IDs returned"); return
    xml = get(f"{BASE}/efetch.fcgi", {"db":"pubmed","id":",".join(ids),"retmode":"xml"}, args.email, args.api_key)
    soup = BeautifulSoup(xml, "xml")
    with out.open("w", encoding="utf-8") as f:
        for aff in soup.find_all("Affiliation"):
            text = aff.get_text(" ", strip=True)
            if text:
                f.write(json.dumps({"affiliation": text}, ensure_ascii=False) + "\n")
    print(f"Wrote {out}")

if __name__ == "__main__": main()
