#!/usr/bin/env python3
import argparse
from torch.utils.data import DataLoader
from dhem.dataset import PairDataset
from dhem.model import load_checkpoint

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--pairs',required=True); args=ap.parse_args()
    ds=PairDataset(args.pairs); dl=DataLoader(ds,batch_size=1024); model,_=load_checkpoint(args.checkpoint)
    import torch
    ys=[]; ps=[]
    with torch.no_grad():
        for x,y in dl: ys += y.tolist(); ps += torch.sigmoid(model(x)).tolist()
    for t in [.5,.7,.8,.9,.95]:
        pred=[p>=t for p in ps]; tp=sum(y==1 and p for y,p in zip(ys,pred)); fp=sum(y==0 and p for y,p in zip(ys,pred)); fn=sum(y==1 and not p for y,p in zip(ys,pred))
        precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1); f1=2*precision*recall/max(precision+recall,1e-9)
        print(f'threshold={t:.2f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}')
if __name__=='__main__': main()
