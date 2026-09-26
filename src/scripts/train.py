#!/usr/bin/env python3
import argparse, random, numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader
from dhem.dataset import PairDataset
from dhem.model import DHEMModel, save_checkpoint
from dhem.config import Config

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def metrics(model, loader, loss_fn):
    model.eval(); ys=[]; ps=[]; total_loss=0
    with torch.no_grad():
        for x,y in loader:
            logits=model(x); total_loss+=loss_fn(logits,y).item()*len(y)
            p=torch.sigmoid(logits); ys.extend(y.tolist()); ps.extend(p.tolist())
    preds=[int(p>=.5) for p in ps]
    tp=sum(a==1 and b==1 for a,b in zip(ys,preds)); tn=sum(a==0 and b==0 for a,b in zip(ys,preds))
    fp=sum(a==0 and b==1 for a,b in zip(ys,preds)); fn=sum(a==1 and b==0 for a,b in zip(ys,preds))
    precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1); f1=2*precision*recall/max(precision+recall,1e-9)
    acc=(tp+tn)/max(len(ys),1)
    return total_loss/max(len(ys),1),acc,precision,recall,f1

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pairs',required=True); ap.add_argument('--val',required=True); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--out',default='artifacts/best.pt')
    args=ap.parse_args(); cfg=Config(); seed_all(cfg.seed)
    train=PairDataset(args.pairs); val=PairDataset(args.val)
    tl=DataLoader(train,batch_size=cfg.batch_size,shuffle=True); vl=DataLoader(val,batch_size=cfg.batch_size)
    model=DHEMModel(cfg.hidden_dim,cfg.dropout)
    # balanced loss for noisy pair datasets
    pos=max(train.y.sum().item(),1); neg=max(len(train)-pos,1)
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg/pos]))
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    best=-1; Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,args.epochs+1):
        model.train(); total=0
        for x,y in tl:
            opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step(); total+=loss.item()*len(y)
        m=metrics(model,vl,loss_fn)
        print(f'epoch={epoch:02d} train_loss={total/len(train):.4f} val_loss={m[0]:.4f} val_acc={m[1]:.4f} p={m[2]:.4f} r={m[3]:.4f} f1={m[4]:.4f}')
        if m[4]>best:
            best=m[4]; save_checkpoint(model,args.out,{'feature_names':cfg.__dict__,'val_f1':best})
    print('saved',args.out)
if __name__=='__main__': main()
