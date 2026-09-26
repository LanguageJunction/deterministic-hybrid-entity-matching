#!/usr/bin/env python3
"""Create a curated, business-style challenge set.

Expected decision can be MATCH or REVIEW. REVIEW cases are intentionally
ambiguous and should not be forced into a single entity.
"""
import argparse
from pathlib import Path
import pandas as pd

CASES=[
    ("J&J","E001","MATCH","abbreviation"),
    ("J N J","E001","MATCH","spaced_abbreviation"),
    ("JNJ","E001","MATCH","ticker_like_alias"),
    ("JHONSON & JHONSON","E001","MATCH","typos"),
    ("Johnson and Johnson","E001","MATCH","punctuation"),
    ("3M LTD","E002","MATCH","legal_suffix"),
    ("3 M","E002","MATCH","spacing"),
    ("3M COMPANY","E002","MATCH","legal_suffix"),
    ("THREE M COMPANY","E002","MATCH","word_form"),
    ("PFEIZER","E003","MATCH","typo"),
    ("ABBOTT LABS","E004","MATCH","abbreviation"),
    ("MICROSOFT CORP.","E010","MATCH","legal_suffix"),
    ("IBM CORP","E009","MATCH","abbreviation"),
    ("GENERAL ELETRIC","E008","MATCH","typo"),
    ("SIEMANS","E007","MATCH","typo"),
    ("UNITED","","REVIEW","ambiguous_short_name"),
    ("DELTA","","REVIEW","ambiguous_short_name"),
    ("AMERICAN","","REVIEW","ambiguous_short_name"),
    ("LABORATORIES","","REVIEW","generic_name"),
    ("3M JOHNSON","","REVIEW","cross_entity"),
]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="data/benchmarks/difficult_cases.csv"); args=ap.parse_args()
    p=Path(args.out); p.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(CASES,columns=["query","expected_entity_id","expected_decision","case_type"]).to_csv(p,index=False)
    print(f"wrote {len(CASES)} benchmark cases -> {p}")
if __name__=="__main__": main()
