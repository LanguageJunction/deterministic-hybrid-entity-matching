"""
Parent/subsidiary-aware entity resolution.

Important principle:
A name match answers "are these names similar?"
A relationship model answers "are these legal/business entities the same
entity, a subsidiary, a parent, a former name, a DBA, or merely related?"

Never collapse these relationships automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict


@dataclass(frozen=True)
class EntityRelationship:
    parent_entity_id: str
    child_entity_id: str
    relationship_type: str
    confidence: float = 1.0


VALID_RELATIONSHIPS = {
    "parent",
    "subsidiary",
    "affiliate",
    "dba",
    "former_name",
    "same_legal_entity",
}


class RelationshipGraph:
    def __init__(self) -> None:
        self.parents: defaultdict[str, set[str]] = defaultdict(set)
        self.children: defaultdict[str, set[str]] = defaultdict(set)
        self.types: dict[tuple[str, str], str] = {}

    def add(self, relationship: EntityRelationship) -> None:
        if relationship.relationship_type not in VALID_RELATIONSHIPS:
            raise ValueError(
                f"Unsupported relationship: {relationship.relationship_type}"
            )

        p, c = relationship.parent_entity_id, relationship.child_entity_id
        self.parents[c].add(p)
        self.children[p].add(c)
        self.types[(p, c)] = relationship.relationship_type

    def parents_of(self, entity_id: str) -> set[str]:
        return set(self.parents.get(entity_id, set()))

    def children_of(self, entity_id: str) -> set[str]:
        return set(self.children.get(entity_id, set()))

    def relationship(self, parent: str, child: str) -> str | None:
        return self.types.get((parent, child))

    def is_same_legal_entity(self, a: str, b: str) -> bool:
        if a == b:
            return True

        rel = self.relationship(a, b) or self.relationship(b, a)

        # Only these relationships can support identity-equivalence logic.
        # A subsidiary/affiliate is deliberately NOT considered the same entity.
        return rel == "former_name" or rel == "dba"

    def relation_aware_adjustment(self, candidate: str, query_entity: str) -> float:
        """
        Score adjustment used after normal similarity scoring.

        Returns:
          0.00  same entity / alias relationship: no penalty
         -0.10  parent/subsidiary
         -0.15  affiliate
         -0.20  unknown but name collision
        """
        if candidate == query_entity:
            return 0.0

        rel = self.relationship(candidate, query_entity)
        reverse = self.relationship(query_entity, candidate)

        if rel in {"former_name", "dba"} or reverse in {"former_name", "dba"}:
            return 0.0

        if rel == "subsidiary" or reverse == "subsidiary":
            return -0.10

        if rel == "affiliate" or reverse == "affiliate":
            return -0.15

        return 0.0
