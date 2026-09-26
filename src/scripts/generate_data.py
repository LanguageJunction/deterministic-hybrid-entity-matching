#!/usr/bin/env python3
import argparse, random, re
from pathlib import Path
import pandas as pd
from rapidfuzz.fuzz import ratio
from dhem.normalization import compact, normalize

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
    rows=[]
    for _, r in master.iterrows():
        aliases=[x.strip() for x in str(r.aliases).split('|') if x.strip() and x != "nan"]
        bases=list(dict.fromkeys([r.canonical_name]+aliases))
        pool=list(dict.fromkeys(q for base in bases for q in variants(base)))
        queries=bases+[random.choice(pool) for _ in range(max(0,n_per_entity-len(bases)))]
        alias_keys={normalize(alias) for alias in aliases}
        others=master[master.entity_id!=r.entity_id]
        for q in queries:
            alias_exact=int(normalize(q) in alias_keys)
            rows.append((q,r.canonical_name,1,r.entity_id,alias_exact,"synthetic_positive"))
            negative_count=int(negative_ratio)+(random.random() < negative_ratio % 1)
            for _ in range(max(1,negative_count)):
                cand=random.choice(others.canonical_name.tolist())
                rows.append((q,cand,0,r.entity_id,0,"synthetic_hard_negative"))
    return pd.DataFrame(rows,columns=["query","candidate","label","entity_id","alias_exact","source"])

def query_key(value):
    return compact(value)

def split_by_entity_queries(pairs, master, val_fraction=.2):
    pairs=pairs.copy()
    pairs["_query_key"]=pairs["query"].map(query_key)
    validation_groups=set()
    rng=random.Random(SEED)
    for _, entity in master.iterrows():
        entity_id=entity.entity_id
        groups=pairs.loc[pairs.entity_id==entity_id,"_query_key"].drop_duplicates().tolist()
        if not groups:
            continue
        aliases=[x.strip() for x in str(entity.aliases).split('|') if x.strip() and x != "nan"]
        required=query_key(aliases[0]) if aliases else groups[0]
        if required not in groups:
            required=groups[0]
        validation_groups.add((entity_id,required))
        remaining=[key for key in groups if key != required]
        rng.shuffle(remaining)
        target=min(max(1,round(len(groups)*val_fraction)),max(1,len(groups)-1))
        validation_groups.update((entity_id,key) for key in remaining[:target-1])
    is_validation=pairs.apply(lambda row:(row["entity_id"],row["_query_key"]) in validation_groups,axis=1)
    return pairs.loc[~is_validation].drop(columns="_query_key").copy(), pairs.loc[is_validation].drop(columns="_query_key").copy()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--master',default='data/master_entities.csv')
    ap.add_argument('--out-dir',default='data/pairs')
    ap.add_argument('--n-per-entity',type=int,default=100)
    args=ap.parse_args()
    master=load_master(Path(args.master))
    pairs=make_pairs(master,args.n_per_entity)
    train,val=split_by_entity_queries(pairs,master)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    train.to_csv(out/'train.csv',index=False); val.to_csv(out/'val.csv',index=False)
    print(f'train={len(train)} val={len(val)}')

if __name__=='__main__': main()
