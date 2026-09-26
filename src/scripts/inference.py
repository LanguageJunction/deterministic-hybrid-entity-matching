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
    for row in matcher.match(args.query): print(row)
if __name__=='__main__': main()
