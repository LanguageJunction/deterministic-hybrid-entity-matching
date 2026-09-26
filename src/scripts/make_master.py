#!/usr/bin/env python3
"""Create/merge the small demonstration master used by examples."""
from pathlib import Path
import pandas as pd

DEFAULT_MASTER=[
("E001","Johnson & Johnson","JnJ|J&J|J N J|JNJ|Jhonson & Jhonson|Johnson and Johnson"),
("E002","3M","3M Ltd|3MLTD|3M Limited|3 M|3M Company|Three M|Three M Company"),
("E003","Pfizer Inc","Pfizer|Pfizer Incorporated|Pfeizer|Pfeizer Inc"),
("E004","Abbott Laboratories","Abbott Labs|Abbott Laboratory|Abbot Laboratories"),
("E005","Merck & Co","Merck|Merck and Company|Merck Co"),
("E006","Bayer AG","Bayer|Bayer Aktiengesellschaft"),
("E007","Siemens AG","Siemens|Siemans AG|Siemans"),
("E008","General Electric","GE|G.E.|General Electric Co|General Electric Company"),
("E009","International Business Machines","IBM|I.B.M.|IBM Corporation|IBM Corp"),
("E010","Microsoft Corporation","Microsoft|MSFT|Microsoft Corp|Microsoft Corporation"),
]
p=Path('data/master_entities.csv'); p.parent.mkdir(parents=True,exist_ok=True)
pd.DataFrame(DEFAULT_MASTER,columns=['entity_id','canonical_name','aliases']).to_csv(p,index=False)
print(p)
