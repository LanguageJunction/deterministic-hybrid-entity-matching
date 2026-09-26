#!/usr/bin/env python3
"""Production inference using DHEM scoring + hierarchical business rules."""
import argparse
import json
from pathlib import Path

import pandas as pd

from dhem.model import load_checkpoint
from dhem.matcher import HybridMatcher
from dhem.decision_engine import HierarchicalDecisionEngine
from dhem.relationships import RelationshipGraph, EntityRelationship


def load_graph(path: str | None) -> RelationshipGraph | None:
    if not path or not Path(path).exists():
        return None
    graph = RelationshipGraph()
    df = pd.read_csv(path).fillna("")
    for _, r in df.iterrows():
        graph.add(EntityRelationship(
            str(r["parent_entity_id"]),
            str(r["child_entity_id"]),
            str(r["relationship_type"]),
        ))
    return graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--master", default="data/master_entities.csv")
    ap.add_argument("--relationships", default="data/entity_relationships.csv")
    ap.add_argument("--query", required=True)
    ap.add_argument("--query-entity-id", default=None)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=.75)
    ap.add_argument("--margin", type=float, default=.08)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    model, _ = load_checkpoint(args.checkpoint)
    master = pd.read_csv(args.master).fillna("")
    matcher = HybridMatcher(master, model, args.threshold, args.margin)
    candidates = matcher.match(args.query, top_k=args.top_k)

    graph = load_graph(args.relationships)
    engine = HierarchicalDecisionEngine(
        fuzzy_threshold=args.threshold,
        review_margin=args.margin,
    )
    result = engine.decide(
        args.query,
        candidates,
        query_entity_id=args.query_entity_id,
        relationship_graph=graph,
    )

    payload = result.to_dict()
    payload["candidates"] = candidates
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"decision      : {result.decision}")
        print(f"reason_code   : {result.reason_code}")
        print(f"reason        : {result.reason}")
        print(f"entity_id     : {result.entity_id}")
        print(f"canonical_name: {result.canonical_name}")
        print(f"score         : {result.score:.4f}")
        print(f"margin        : {result.margin:.4f}")
        if result.relation_type:
            print(f"relationship  : {result.relation_type}")
        print("evidence      :", json.dumps(result.evidence or {}, ensure_ascii=False))
        print("\nTop candidates:")
        for c in candidates:
            print("  ", c)


if __name__ == "__main__":
    main()
