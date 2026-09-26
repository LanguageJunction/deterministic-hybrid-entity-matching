#!/usr/bin/env python3
"""Publish DHEM checkpoints with model cards and per-model metadata."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi

from dhem.config import FEATURE_NAMES
from dhem.features import pair_features
from dhem.model import load_checkpoint


MODELS = {
    "best": {
        "checkpoint": Path("artifacts/best.pt"),
        "master": "data/master_entities.csv",
        "train_pairs": "data/pairs/train.csv",
        "validation_pairs": "data/pairs/val.csv",
        "training_data": "Curated master entities with synthetic name variants and approved aliases.",
        "validation_strategy": "Held-out normalized query groups; each master entity appears in both splits.",
        "positive_examples": [
            {"query": "JNJ", "candidate": "Johnson & Johnson", "alias_exact": True},
            {"query": "3MLTD", "candidate": "3M", "alias_exact": True},
        ],
        "negative_examples": [
            {"query": "JNJ", "candidate": "3M", "alias_exact": False},
            {"query": "3MLTD", "candidate": "Microsoft Corporation", "alias_exact": False},
        ],
    },
    "real_best": {
        "checkpoint": Path("artifacts/real_best.pt"),
        "master": "data/pairs/real_master.csv",
        "train_pairs": "data/pairs/real_train.csv",
        "validation_pairs": "data/pairs/real_val.csv",
        "training_data": "Company records and source-backed aliases/former names from SEC and/or OpenAlex.",
        "validation_strategy": "Entity-aware split; validation entities are held out from training.",
        "positive_examples": [
            {"query": "NVIDIA CORP/CA", "candidate": "NVIDIA CORP", "alias_exact": True},
        ],
        "negative_examples": [
            {"query": "NVIDIA CORP/CA", "candidate": "Apple Inc.", "alias_exact": False},
        ],
    },
}


def score_pair(model, query, candidate, alias_exact):
    features = pair_features(query, candidate, alias_exact)
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor([features], dtype=torch.float32))).item()


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def inspect_checkpoint(name, details):
    path = details["checkpoint"]
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model, checkpoint_metadata = load_checkpoint(path)
    model_config = checkpoint_metadata.get("feature_names", {})
    hidden_dim = model_config.get("hidden_dim", 128)
    dropout = model_config.get("dropout", 0.15)
    architecture = {
        "input_features": len(FEATURE_NAMES),
        "layers": [
            {"type": "Linear", "in_features": len(FEATURE_NAMES), "out_features": hidden_dim},
            {"type": "ReLU"},
            {"type": "LayerNorm", "normalized_shape": hidden_dim},
            {"type": "Dropout", "p": dropout},
            {"type": "Linear", "in_features": hidden_dim, "out_features": hidden_dim // 2},
            {"type": "ReLU"},
            {"type": "Dropout", "p": dropout},
            {"type": "Linear", "in_features": hidden_dim // 2, "out_features": 1},
        ],
        "output": "Squeeze final dimension to one binary match logit per query-candidate pair.",
    }
    examples = {}
    for kind in ("positive", "negative"):
        examples[kind] = []
        for definition in details[f"{kind}_examples"]:
            example = dict(definition)
            if kind == "positive" and example["alias_exact"]:
                example["score"] = 1.0
            else:
                example["score"] = score_pair(
                    model, example["query"], example["candidate"], example["alias_exact"]
                )
            example["expected_result"] = "MATCH" if kind == "positive" else "NO_MATCH"
            examples[kind].append(example)
    return {
        "name": name,
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "validation_f1": checkpoint_metadata.get("val_f1"),
        "architecture": "DHEMModel feed-forward binary pair classifier",
        "architecture_details": architecture,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "features": FEATURE_NAMES,
        "model_config": model_config,
        "master_catalog": details["master"],
        "training_pairs": details["train_pairs"],
        "validation_pairs": details["validation_pairs"],
        "training_data": details["training_data"],
        "validation_strategy": details["validation_strategy"],
        "examples": examples,
    }


def make_model_card(repo_id, metadata):
    model_sections = []
    inference_commands = []
    for item in metadata:
        config = item["model_config"]
        architecture = item["architecture_details"]
        layer_lines = "\n".join(
            f"{index}. `{layer['type']}`"
            + (f" ({layer['in_features']} -> {layer['out_features']})" if "in_features" in layer else "")
            + (f" (normalized_shape={layer['normalized_shape']})" if "normalized_shape" in layer else "")
            + (f" (p={layer['p']})" if "p" in layer else "")
            for index, layer in enumerate(architecture["layers"], start=1)
        )
        model_sections.append(f"""### `{item['file']}`

- **Catalog:** `{item['master_catalog']}`
- **Training data:** {item['training_data']}
- **Validation:** {item['validation_strategy']}
- **Validation F1:** {item['validation_f1']}
- **Parameters:** {item['parameter_count']:,}
- **Configuration:** hidden dimension {config.get('hidden_dim', 'unknown')}, dropout {config.get('dropout', 'unknown')}, learning rate {config.get('learning_rate', 'unknown')}, weight decay {config.get('weight_decay', 'unknown')}, batch size {config.get('batch_size', 'unknown')}, epochs {config.get('epochs', 'unknown')}
- **SHA-256:** `{item['sha256']}`

**Architecture** (input: {architecture['input_features']} numeric pair features):

{layer_lines}

Output: {architecture['output']}

Training pairs: `{item['training_pairs']}`
Validation pairs: `{item['validation_pairs']}`

#### Pair examples

| Use case | Query | Candidate | Pair score | Expected result |
|---|---|---|---:|---|
{chr(10).join(f"| Positive, approved alias | `{example['query']}` | `{example['candidate']}` | {example['score']:.4f} | `{example['expected_result']}` |" for example in item['examples']['positive'])}
{chr(10).join(f"| Negative, different entity | `{example['query']}` | `{example['candidate']}` | {example['score']:.4f} | `{example['expected_result']}` |" for example in item['examples']['negative'])}
""")
        inference_commands.append(" ".join([
            "python src/scripts/inference.py",
            f"--checkpoint artifacts/{item['file']}",
            f"--master {item['master_catalog']}",
            f"--query {json.dumps(item['examples']['positive'][0]['query'])}",
        ]))

    return f"""---
library_name: pytorch
tags:
- entity-resolution
- entity-matching
- pytorch
---

# DHEM Entity Matching

PyTorch pair scorers for matching noisy organization names against a supplied
master catalog. Models return pairwise scores; use the catalog's canonical name
as the resolved entity name and retain review handling for ambiguous matches.

## Checkpoints

{''.join(model_sections)}

## Input features

The model consumes ten deterministic pair features, in order: normalized
equality, approved-alias equality, token Jaccard, token containment, character
n-gram Jaccard, edit similarity, prefix similarity, acronym match, length ratio,
and rule score. Full feature names are included in `model_details.json`.

## Inference

Use each checkpoint with its corresponding master CSV:

```bash
{chr(10).join(inference_commands)}
```

The `best.pt` checkpoint targets the curated catalog. `real_best.pt` targets
the real SEC/OpenAlex catalog. Do not interchange their master files: an alias
present in one catalog may be absent from the other. For hierarchical business
decisions, use `src/scripts/hierarchical_inference.py`.

## Limitations

Pair scores are model probabilities, not legal-identity guarantees. Validation
scores are specific to each dataset and split and are not directly comparable
estimates of performance on every production catalog. The real-data catalog
can contain related but legally distinct organizations; use authoritative
entity IDs and business rules, and route uncertain or related-entity cases for
review.

## Repository

Source code, dataset-generation scripts, and evaluation utilities: [project repository](https://huggingface.co/{repo_id}).
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repository, e.g. user/dhem-models")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS), help="Checkpoints to publish (default: both)")
    parser.add_argument("--private", action="store_true", help="Create or update a private model repository")
    args = parser.parse_args()

    load_dotenv()
    metadata = [inspect_checkpoint(name, MODELS[name]) for name in args.models]
    card = make_model_card(args.repo_id, metadata)
    details_json = json.dumps(metadata, indent=2, ensure_ascii=False, default=json_safe)

    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    for item in metadata:
        api.upload_file(
            path_or_fileobj=str(MODELS[item["name"]]["checkpoint"]),
            path_in_repo=item["file"],
            repo_id=args.repo_id,
            repo_type="model",
        )
    api.upload_file(
        path_or_fileobj=io.BytesIO(card.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=io.BytesIO(details_json.encode("utf-8")),
        path_in_repo="model_details.json",
        repo_id=args.repo_id,
        repo_type="model",
    )
    print(f"Published {', '.join(item['file'] for item in metadata)} with model card to {args.repo_id}")


if __name__ == "__main__":
    main()