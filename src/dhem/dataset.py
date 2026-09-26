from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from .features import pair_features

@dataclass
class Pair:
    query: str
    candidate: str
    label: int
    entity_id: str

class PairDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        required = {"query", "candidate", "label", "entity_id"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        self.features = []
        self.labels = []
        for r in self.df.itertuples(index=False):
            self.features.append(pair_features(r.query, r.candidate, bool(getattr(r, "alias_exact", 0))))
            self.labels.append(float(r.label))
        self.x = torch.tensor(self.features, dtype=torch.float32)
        self.y = torch.tensor(self.labels, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

def load_pairs(path):
    return pd.read_csv(path)
