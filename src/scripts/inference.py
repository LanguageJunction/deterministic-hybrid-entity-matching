#!/usr/bin/env python3
import argparse, pandas as pd
from dhem.model import load_checkpoint
from dhem.matcher import HybridMatcher

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--master',default='data/master_entities.csv'); ap.add_argument('--query',required=True); ap.add_argument('--candidates',nargs='*'); ap.add_argument('--threshold',type=float,default=.75); args=ap.parse_args()
    model,_=load_checkpoint(args.checkpoint)
    master=pd.read_csv(args.master)
    if args.candidates:
        wanted=set(args.candidates); master=master[master.canonical_name.isin(wanted)].reset_index(drop=True)
    matcher=HybridMatcher(master,model,args.threshold)
    for row in matcher.match(args.query):
        print(f"decision      : {row['decision']}")
        print(f"entity_id     : {row['entity_id'] or '(none)'}")
        print(f"canonical_name: {row['canonical_name'] or '(none)'}")
        print(f"score         : {row['score']:.4f}")
        print(f"margin        : {row['margin']:.4f}")
        if row.get("matched_alias"):
            print(f"matched_alias : {row['matched_alias']}")
        if row.get("reason"):
            print(f"reason        : {row['reason']}")
        print()
if __name__=='__main__': main()
