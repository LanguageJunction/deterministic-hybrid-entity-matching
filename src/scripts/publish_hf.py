#!/usr/bin/env python3
import argparse
from huggingface_hub import HfApi

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo-id',required=True); ap.add_argument('--path',default='artifacts/best.pt'); ap.add_argument('--private',action='store_true'); args=ap.parse_args()
    api=HfApi(); api.create_repo(args.repo_id,repo_type='model',private=args.private,exist_ok=True)
    api.upload_file(path_or_fileobj=args.path,path_in_repo='best.pt',repo_id=args.repo_id,repo_type='model')
    print('uploaded to',args.repo_id)
if __name__=='__main__': main()
