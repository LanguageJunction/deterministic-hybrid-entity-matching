#!/usr/bin/env python3
"""
Combine SEC EDGAR and OpenAlex organization/company datasets into one
canonical master + alias table.

The script is deliberately conservative:
- preserves source IDs and provenance
- de-duplicates by normalized names and source identifiers
- does NOT automatically declare subsidiaries equivalent to parents
- optionally consumes a manually curated relationship file

Expected input examples:
  data/real/sec_companies.csv
  data/real/openalex_companies.csv

Output:
  data/combined/combined_master.csv
  data/combined/combined_aliases.csv
  data/combined/source_lineage.csv

Optional relationship CSV:
  entity_relationships.csv

Columns:
  parent_entity_id,child_entity_id,relationship_type
where relationship_type can be:
  parent, subsidiary, affiliate, dba, former_name

This file is a data-integration layer, not a legal-entity authority.
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def first(row: dict[str, str], *names: str) -> str:
    for name in names:
        if row.get(name):
            return row[name].strip()
    return ""


def aliases_from_row(row: dict[str, str]) -> list[str]:
    raw = first(row, "aliases", "alternative_names", "alt_names",
                "former_names", "previous_names")
    if not raw:
        return []
    return [x.strip() for x in re.split(r"[|;]", raw) if x.strip()]


def load_source(path: Path, source: str) -> list[dict[str, str]]:
    rows = read_csv(path)
    result = []

    for i, row in enumerate(rows):
        name = first(row, "canonical_name", "name", "company_name",
                     "organization_name", "display_name")
        if not name:
            continue

        source_id = first(
            row,
            "source_id",
            "cik",
            "entity_id",
            "id",
            "openalex_id",
            "ror",
        ) or f"{source}:{i}"

        result.append({
            "source": source,
            "source_id": source_id,
            "canonical_name": name,
            "aliases": "|".join(dict.fromkeys(aliases_from_row(row))),
            "country": first(row, "country", "country_code"),
            "website": first(row, "website", "homepage", "domain"),
            "industry": first(row, "industry", "sector"),
        })

    return result


def make_entity_id(source: str, source_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.:-]+", "_", source_id)
    return f"{source.upper()}::{safe}"


def load_relationships(path: Path) -> list[dict[str, str]]:
    return read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", default="data/real/sec_companies.csv")
    ap.add_argument("--openalex", default="data/real/openalex_companies.csv")
    ap.add_argument("--relationships", default="")
    ap.add_argument("--out-dir", default="data/combined")
    args = ap.parse_args()

    sec = load_source(Path(args.sec), "sec")
    oa = load_source(Path(args.openalex), "openalex")
    rows = sec + oa

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Stable source-specific entity IDs. We intentionally do not merge
    # SEC and OpenAlex records merely because their names match.
    master = []
    aliases = []
    lineage = []

    normalized_to_entities: defaultdict[str, set[str]] = defaultdict(set)

    for row in rows:
        entity_id = make_entity_id(row["source"], row["source_id"])
        canonical = row["canonical_name"]

        master.append({
            "entity_id": entity_id,
            "canonical_name": canonical,
            "source": row["source"],
            "source_id": row["source_id"],
            "country": row["country"],
            "website": row["website"],
            "industry": row["industry"],
        })

        all_aliases = [canonical] + [
            x for x in row["aliases"].split("|") if x
        ]

        seen = set()
        for alias in all_aliases:
            key = normalize(alias)
            if not key or key in seen:
                continue
            seen.add(key)
            aliases.append({
                "entity_id": entity_id,
                "alias": alias,
                "normalized_alias": key,
                "alias_type": "canonical" if alias == canonical else "source_alias",
                "source": row["source"],
                "source_id": row["source_id"],
            })
            normalized_to_entities[key].add(entity_id)

        lineage.append({
            "entity_id": entity_id,
            "source": row["source"],
            "source_id": row["source_id"],
            "canonical_name": canonical,
        })

    # Report cross-source collisions instead of merging them.
    collisions = []
    for normalized_name, entity_ids in sorted(normalized_to_entities.items()):
        if len(entity_ids) > 1:
            collisions.append({
                "normalized_name": normalized_name,
                "entity_ids": "|".join(sorted(entity_ids)),
                "n_entities": str(len(entity_ids)),
                "decision": "REVIEW_NOT_AUTO_MERGED",
            })

    rels = load_relationships(Path(args.relationships)) if args.relationships else []

    with (out / "combined_master.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(master[0].keys()) if master else
                                ["entity_id", "canonical_name", "source",
                                 "source_id", "country", "website", "industry"])
        writer.writeheader()
        writer.writerows(master)

    with (out / "combined_aliases.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["entity_id", "alias", "normalized_alias", "alias_type",
                  "source", "source_id"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aliases)

    with (out / "source_lineage.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["entity_id", "source", "source_id", "canonical_name"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(lineage)

    with (out / "cross_source_collisions.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["normalized_name", "entity_ids", "n_entities", "decision"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(collisions)

    if args.relationships:
        with (out / "entity_relationships.csv").open("w", newline="", encoding="utf-8") as f:
            fields = ["parent_entity_id", "child_entity_id", "relationship_type"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rels)

    print(f"SEC records:      {len(sec):,}")
    print(f"OpenAlex records: {len(oa):,}")
    print(f"Combined entities:{len(master):,}")
    print(f"Aliases:          {len(aliases):,}")
    print(f"Cross-source collisions requiring review: {len(collisions):,}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
