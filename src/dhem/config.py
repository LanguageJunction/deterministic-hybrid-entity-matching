from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    seed: int = 42
    max_len: int = 64
    hidden_dim: int = 128
    dropout: float = 0.15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 20
    threshold: float = 0.75

def default_master_for_checkpoint(checkpoint):
    if Path(checkpoint).stem.casefold().startswith("real_"):
        return Path("data/pairs/real_master.csv")
    return Path("data/master_entities.csv")

FEATURE_NAMES = [
    "exact_normalized", "alias_exact", "token_jaccard",
    "token_containment", "char_jaccard", "levenshtein_ratio",
    "prefix_ratio", "acronym_match", "length_ratio", "rule_score"
]
