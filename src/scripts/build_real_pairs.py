#!/usr/bin/env python3
"""Turn real SEC/OpenAlex organization records into positive and hard-negative pairs.

Positive labels are source-backed: aliases/former names belong to the same
stable entity_id. Negatives are sampled from different entity IDs and include
lexically similar candidates.
"""
import argparse, random, re
from pathlib import Path
import pandas as pd
from rapidfuzz.fuzz import ratio

SEED=20260925
random.seed(SEED)

def clean_aliases(v):
    if pd.isna(v) or not str(v).strip(): return []
    return [x.strip() for x in str(v).split("|") if x.strip()]

def typo_variants(s):
    out={s, s.lower(), s.upper(), s.replace("&","and"), s.replace(" & "," "), s.replace(" ","")}
    compact=re.sub(r"[^A-Za-z0-9]", "", s)
    if len(compact)>5:
        for i in [max(1,len(compact)//2), min(len(compact)-2,3)]:
            out.add(compact[:i]+compact[i+1:])
            out.add(compact[:i]+compact[i]+compact[i:])
            if i+1<len(compact): out.add(compact[:i]+compact[i+1]+compact[i]+compact[i+2:])
    return list(out)

def build(df, n_pos=8, n_neg=8):
    rows=[]
    names=df["canonical_name"].fillna("").tolist()
    for _,r in df.iterrows():
        eid=r.entity_id; cand=r.canonical_name
        base=[cand]+clean_aliases(r.get("aliases",""))
        pool=list(dict.fromkeys(v for b in base for v in typo_variants(b) if v))
        for q in random.sample(pool, min(n_pos,len(pool))):
            rows.append((q,cand,1,eid,"real_positive"))
        # hard negatives: candidates sharing tokens / high string similarity first
        others=df[df.entity_id!=eid].copy()
        scored=[]
        for _,o in others.sample(min(len(others), 500), random_state=SEED).iterrows():
            sc=ratio(str(cand),str(o.canonical_name))
            shared=len(set(str(cand).lower().split()) & set(str(o.canonical_name).lower().split()))
            scored.append((sc+20*shared,o.canonical_name,o.entity_id))
        scored.sort(reverse=True)
        chosen=scored[:max(2,n_neg//2)]
        if len(scored)>max(2,n_neg//2):
            chosen += random.sample(scored[max(2,n_neg//2):], min(n_neg-len(chosen), len(scored)-len(chosen)))
        for _,cn,_ in chosen:
            rows.append((random.choice(pool),cn,0,eid,"real_hard_negative"))
    return pd.DataFrame(rows,columns=["query","candidate","label","entity_id","source"])

def split_by_entity(df, val_fraction=.2):
    ids=df.entity_id.drop_duplicates().tolist(); random.shuffle(ids)
    cut=max(1,int(len(ids)*val_fraction)); valids=set(ids[:cut])
    return df[~df.entity_id.isin(valids)].copy(), df[df.entity_id.isin(valids)].copy()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True, help="CSV from download_sec.py or download_openalex.py")
    ap.add_argument("--out-dir", default="data/pairs")
    ap.add_argument("--positive-per-entity", type=int, default=8)
    ap.add_argument("--negative-per-entity", type=int, default=8)
    args=ap.parse_args()
    frames=[pd.read_csv(x) for x in args.input]
    df=pd.concat(frames,ignore_index=True).dropna(subset=["entity_id","canonical_name"])
    df=df.drop_duplicates("entity_id")
    pairs=build(df,args.positive_per_entity,args.negative_per_entity)
    train,val=split_by_entity(pairs)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    train.to_csv(out/"real_train.csv",index=False); val.to_csv(out/"real_val.csv",index=False)
    df.to_csv(out/"real_master.csv",index=False)
    print(f"entities={len(df)} pairs={len(pairs)} train={len(train)} val={len(val)}")

if __name__=="__main__": main()
