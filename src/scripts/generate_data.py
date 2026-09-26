#!/usr/bin/env python3
import argparse, random, re
from pathlib import Path
import pandas as pd
from rapidfuzz.fuzz import ratio

SEED=42
random.seed(SEED)

DEFAULT_MASTER = [
("E001","Johnson & Johnson","JnJ|J&J|Jhonson & Jhonson|Johnson and Johnson"),
("E002","3M","3M Ltd|3MLTD|3M Limited|Three M"),
("E003","Pfizer Inc","Pfizer|Pfizer Incorporated|Pfeizer"),
("E004","Abbott Laboratories","Abbott Labs|Abbott Laboratory|Abbot Laboratories"),
("E005","Merck & Co","Merck|Merck and Company|Merck Co"),
("E006","Bayer AG","Bayer|Bayer Aktiengesellschaft"),
("E007","Siemens AG","Siemens|Siemans AG"),
("E008","General Electric","GE|G.E.|General Electric Co"),
("E009","International Business Machines","IBM|I.B.M.|IBM Corporation"),
("E010","Microsoft Corporation","Microsoft|MSFT|Microsoft Corp"),
]

def variants(name):
    n=name
    out={n, n.replace("&","and"), n.replace(" & "," "), n.replace(" ",""), n.lower(), n.upper()}
    out.add(n.replace("Johnson", "Jhonson"))
    out.add(re.sub(r"(?i)corporation", "corp", n))
    out.add(re.sub(r"(?i)inc", "inc.", n))
    out.add(n.replace("3M", "3M Ltd"))
    # random typo: delete/duplicate/swap one character
    s=''.join(ch for ch in n if ch.isalnum() or ch==' ')
    if len(s)>5:
        i=random.randrange(1,len(s)-1)
        out.add(s[:i]+s[i+1:])
        out.add(s[:i]+s[i]+s[i:])
        if i+1<len(s): out.add(s[:i]+s[i+1]+s[i]+s[i+2:])
    return {x for x in out if x}

def load_master(path):
    if path.exists(): return pd.read_csv(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df=pd.DataFrame(DEFAULT_MASTER, columns=["entity_id","canonical_name","aliases"])
    df.to_csv(path,index=False); return df

def make_pairs(master, n_per_entity=80, negative_ratio=1.5):
    positives=[]; negatives=[]
    for _, r in master.iterrows():
        pool=list(variants(r.canonical_name)) + str(r.aliases).split('|')
        for _ in range(n_per_entity):
            q=random.choice(pool)
            positives.append((q,r.canonical_name,1,r.entity_id))
        # hard negatives: choose names with some lexical overlap, otherwise random
        others=master[master.entity_id!=r.entity_id]
        for _ in range(max(1,int(n_per_entity*negative_ratio))):
            cand=random.choice(others.canonical_name.tolist())
            negatives.append((random.choice(pool),cand,0,r.entity_id))
    df=pd.DataFrame(positives+negatives,columns=["query","candidate","label","entity_id"])
    df=df.sample(frac=1,random_state=SEED).reset_index(drop=True)
    return df

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--master',default='data/master_entities.csv')
    ap.add_argument('--out-dir',default='data/pairs')
    ap.add_argument('--n-per-entity',type=int,default=100)
    args=ap.parse_args()
    master=load_master(Path(args.master))
    df=make_pairs(master,args.n_per_entity)
    # Entity-aware split: each entity is assigned to one split, avoiding exact
    # variant leakage. In production, keep a fixed master and time-based split.
    entities=list(master.entity_id)
    random.Random(SEED).shuffle(entities)
    cut=max(1,int(.2*len(entities)))
    val_entities=set(entities[:cut])
    val=df[df.entity_id.isin(val_entities)].copy()
    train=df[~df.entity_id.isin(val_entities)].copy()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    train.to_csv(out/'train.csv',index=False); val.to_csv(out/'val.csv',index=False)
    print(f'train={len(train)} val={len(val)}')

if __name__=='__main__': main()
