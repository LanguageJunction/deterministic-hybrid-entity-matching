# Parent/Subsidiary-aware matching

## Why this matters

String similarity alone can produce a dangerous result:

    "Janssen Pharmaceuticals"
        vs
    "Johnson & Johnson"

They are related, but they are not necessarily the same vendor/legal entity.

The DHEM system therefore separates four concepts:

1. **Identity** — same entity.
2. **Former name / DBA** — may be identity-equivalent depending on master-data policy.
3. **Parent/subsidiary** — related entities, NOT automatically identical.
4. **Affiliate** — related entities, NOT automatically identical.

## Recommended decision policy

```text
same entity / approved alias
    -> MATCH

former name / approved DBA
    -> MATCH only if business policy allows

parent ↔ subsidiary
    -> RELATED / REVIEW

affiliate
    -> RELATED / REVIEW

generic or ambiguous name
    -> REVIEW

no sufficiently strong candidate
    -> NO_MATCH
```

## Required relationship file

Create:

`data/entity_relationships.csv`

with:

```csv
parent_entity_id,child_entity_id,relationship_type
SEC::...,SEC::...,subsidiary
SEC::...,SEC::...,former_name
OPENALEX::...,OPENALEX::...,affiliate
```

Do not infer legal-entity equivalence solely from a name match.

## Combining SEC + OpenAlex

Run:

```bash
python scripts/combine_sources.py \
  --sec data/real/sec_companies.csv \
  --openalex data/real/openalex_companies.csv \
  --relationships data/entity_relationships.csv \
  --out-dir data/combined
```

The resulting:

- `combined_master.csv`
- `combined_aliases.csv`
- `source_lineage.csv`
- `cross_source_collisions.csv`
- `entity_relationships.csv`

are intentionally provenance-preserving.

A name that occurs in both SEC and OpenAlex is NOT silently merged. It is placed in
`cross_source_collisions.csv` for review.

This is safer for procurement/vendor/customer master data because a collision
does not prove that two records represent the same legal entity.
