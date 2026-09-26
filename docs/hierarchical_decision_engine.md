# Hierarchical decision engine

The PyTorch network is a scorer, not the final business decision maker.
`dhem/decision_engine.py` applies deterministic and relationship-aware rules first.

## Decision order

```text
EXACT_CANONICAL
      ↓
APPROVED_ALIAS
      ↓
FORMER_NAME / DBA / SAME_LEGAL_ENTITY
      ↓
PARENT / SUBSIDIARY / AFFILIATE → REVIEW
      ↓
FUZZY_MATCH
      ↓
LOW_MARGIN → REVIEW
      ↓
LOW_SIMILARITY → NO_MATCH
```

Every result includes:

- `decision`: MATCH / REVIEW / NO_MATCH
- `reason_code`: machine-readable audit reason
- `reason`: human-readable explanation
- `entity_id`
- `canonical_name`
- `score`
- `margin`
- `relationship_type` when applicable
- `evidence`

## Example

```bash
python scripts/hierarchical_inference.py \
  --checkpoint artifacts/best.pt \
  --master data/master_entities.csv \
  --relationships data/entity_relationships.csv \
  --query "Jhonson & Jhonson" \
  --json
```

For a known incoming entity, pass `--query-entity-id`. This enables explicit
parent/subsidiary/affiliate relationships to override a tempting fuzzy match.

## Why this is safer

A high fuzzy score is not proof of legal identity. For example, a subsidiary can
have a very similar or related name to its parent. The engine therefore returns
`REVIEW` for explicit `parent`, `subsidiary`, and `affiliate` relationships rather
than collapsing them into one master entity.
