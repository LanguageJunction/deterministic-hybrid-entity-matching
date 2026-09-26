import torch
from torch import nn
from pathlib import Path
from .config import FEATURE_NAMES

class DHEMModel(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def save_checkpoint(model, path, metadata=None):
    torch.save({"state_dict": model.state_dict(), "metadata": metadata or {}}, path)

def load_checkpoint(path, device="cpu"):
    with torch.serialization.safe_globals([type(Path())]):
        ckpt = torch.load(path, map_location=device, weights_only=True)
    model = DHEMModel()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt.get("metadata", {})
