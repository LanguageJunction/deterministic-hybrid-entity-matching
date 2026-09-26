#!/usr/bin/env python3
import argparse, pandas as pd
from dhem.model import load_checkpoint
from dhem.matcher import HybridMatcher

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--master',default='data/master_entities.csv')
    ap.add_argument('--benchmark',default='data/benchmarks/difficult_cases.csv')
    ap.add_argument('--threshold',type=float,default=.85)
    ap.add_argument('--margin',type=float,default=.08)
    args=ap.parse_args()
    master=pd.read_csv(args.master)
    model,_=load_checkpoint(args.checkpoint)
    matcher=HybridMatcher(master,model,args.threshold,margin=args.margin)
    bench=pd.read_csv(args.benchmark)
    rows=[]
    for r in bench.itertuples(index=False):
        result=matcher.match(r.query,top_k=3)
        top=result[0] if result else {}
        rows.append({
            'query':r.query,
            'expected_entity_id':r.expected_entity_id,
            'expected_decision':r.expected_decision,
            'predicted_entity_id':top.get('entity_id',''),
            'predicted_name':top.get('canonical_name',''),
            'predicted_decision':top.get('decision','NO_MATCH'),
            'score':top.get('score',0.0),
            'margin':top.get('margin',0.0),
            'case_type':r.case_type,
        })
    out=pd.DataFrame(rows)
    out.to_csv('artifacts/difficult_benchmark_results.csv',index=False)
    print(out.to_string(index=False))
    match=out.expected_decision.eq('MATCH')
    match_ok=match & out.predicted_entity_id.eq(out.expected_entity_id) & out.predicted_decision.eq('MATCH')
    review=out.expected_decision.eq('REVIEW')
    review_ok=review & out.predicted_decision.eq('REVIEW')
    print(f'\nMATCH cases correct: {match_ok.sum()}/{match.sum()}')
    print(f'REVIEW cases correctly held: {review_ok.sum()}/{review.sum()}')

if __name__=='__main__': main()
