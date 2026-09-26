"""Hierarchical, explainable decision layer for DHEM.

The PyTorch model remains a *scorer*. This module owns business decisions and
keeps deterministic identity rules above fuzzy similarity.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable

from .normalization import normalize


@dataclass
class Decision:
    query: str
    entity_id: str | None
    canonical_name: str | None
    decision: str
    reason_code: str
    reason: str
    score: float = 0.0
    margin: float = 0.0
    relation_type: str | None = None
    evidence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HierarchicalDecisionEngine:
    """Apply business-safe rules in front of / around fuzzy model scoring.

    Priority:
      EXACT_CANONICAL
      APPROVED_ALIAS
      FORMER_NAME_DBA
      SAME_LEGAL_ENTITY
      PARENT_SUBSIDIARY / AFFILIATE -> REVIEW
      FUZZY_MATCH
      REVIEW / NO_MATCH

    A subsidiary is never silently converted into the parent entity.
    """

    def __init__(
        self,
        exact_threshold: float = 0.90,
        fuzzy_threshold: float = 0.75,
        review_margin: float = 0.08,
    ):
        self.exact_threshold = exact_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.review_margin = review_margin

    @staticmethod
    def _aliases(row: Any) -> list[str]:
        raw = row.get("aliases", "") if hasattr(row, "get") else ""
        return [x.strip() for x in str(raw).split("|") if x.strip() and x != "nan"]

    @staticmethod
    def _candidate_relationship(
        candidate_id: str,
        query_entity_id: str | None,
        relationship_graph: Any | None,
    ) -> str | None:
        if not relationship_graph or not query_entity_id:
            return None
        return (
            relationship_graph.relationship(candidate_id, query_entity_id)
            or relationship_graph.relationship(query_entity_id, candidate_id)
        )

    def decide(
        self,
        query: str,
        ranked_candidates: Iterable[dict[str, Any]],
        *,
        query_entity_id: str | None = None,
        relationship_graph: Any | None = None,
    ) -> Decision:
        candidates = list(ranked_candidates)
        if not candidates:
            return Decision(
                query=query,
                entity_id=None,
                canonical_name=None,
                decision="NO_MATCH",
                reason_code="NO_CANDIDATES",
                reason="No candidate entity was retrieved.",
                evidence={},
            )

        if candidates[0].get("reason_code") == "NO_CANDIDATES":
            return Decision(
                query=query,
                entity_id=None,
                canonical_name=None,
                decision="REVIEW",
                reason_code="NO_CANDIDATES",
                reason="No master candidates were retrieved; manual review is required.",
                evidence={},
            )

        qn = normalize(query)

        # 1. Exact canonical name. If several entities share the exact name,
        # ambiguity wins over automatic matching.
        exact = [c for c in candidates if normalize(c["canonical_name"]) == qn]
        if len(exact) == 1:
            c = exact[0]
            return Decision(
                query=query, entity_id=c["entity_id"], canonical_name=c["canonical_name"],
                decision="MATCH", reason_code="EXACT_CANONICAL",
                reason="Normalized query exactly matches the canonical entity name.",
                score=float(c.get("score", 1.0)), margin=float(c.get("margin", 1.0)),
                evidence={"normalized_query": qn},
            )
        if len(exact) > 1:
            return self._review(query, exact[0], "AMBIGUOUS_EXACT", 
                                "More than one candidate has the same normalized canonical name.",
                                evidence={"candidate_count": len(exact)})

        # 2. Approved aliases. HybridMatcher marks alias_exact on candidates.
        aliases = [c for c in candidates if c.get("alias_exact")]
        if len(aliases) == 1:
            c = aliases[0]
            return Decision(
                query=query, entity_id=c["entity_id"], canonical_name=c["canonical_name"],
                decision="MATCH", reason_code="APPROVED_ALIAS",
                reason="Normalized query exactly matches an approved master-data alias.",
                score=float(c.get("score", 1.0)), margin=float(c.get("margin", 1.0)),
                evidence={"alias": c.get("matched_alias", query)},
            )
        if len(aliases) > 1:
            return self._review(query, aliases[0], "AMBIGUOUS_ALIAS",
                                "The query is an approved alias for multiple entities.",
                                evidence={"candidate_count": len(aliases)})

        # 3-5. Explicit relationship evidence.
        top = candidates[0]
        rel = self._candidate_relationship(top["entity_id"], query_entity_id, relationship_graph)
        if rel in {"former_name", "dba", "same_legal_entity"}:
            return Decision(
                query=query, entity_id=top["entity_id"], canonical_name=top["canonical_name"],
                decision="MATCH", reason_code=rel.upper(),
                reason=f"Relationship data identifies the candidate as {rel.replace('_', ' ')}.",
                score=float(top.get("score", 1.0)), margin=float(top.get("margin", 0.0)),
                relation_type=rel,
                evidence={"query_entity_id": query_entity_id},
            )
        if rel in {"parent", "subsidiary", "affiliate"}:
            return self._review(
                query, top, "RELATED_ENTITY", 
                f"Candidate is related as {rel}, but related entities are not automatically the same vendor/legal entity.",
                relation_type=rel,
                evidence={"query_entity_id": query_entity_id},
            )

        # 6. Fuzzy model decision.
        score = float(top.get("score", 0.0))
        margin = float(top.get("margin", 0.0))
        if score >= self.fuzzy_threshold and margin >= self.review_margin:
            return Decision(
                query=query, entity_id=top["entity_id"], canonical_name=top["canonical_name"],
                decision="MATCH", reason_code="FUZZY_MATCH",
                reason="PyTorch/DHEM similarity score and candidate margin exceed configured thresholds.",
                score=score, margin=margin,
                evidence={"threshold": self.fuzzy_threshold, "margin_threshold": self.review_margin},
            )

        if score >= self.fuzzy_threshold:
            return self._review(
                query, top, "LOW_MARGIN",
                "Similarity is high, but the top candidates are too close to safely auto-match.",
                evidence={"threshold": self.fuzzy_threshold, "margin_threshold": self.review_margin},
            )

        return Decision(
            query=query, entity_id=top.get("entity_id"), canonical_name=top.get("canonical_name"),
            decision="NO_MATCH", reason_code="LOW_SIMILARITY",
            reason="No deterministic identity rule matched and similarity is below the match threshold.",
            score=score, margin=margin,
            evidence={"threshold": self.fuzzy_threshold},
        )

    @staticmethod
    def _review(query, candidate, code, reason, *, relation_type=None, evidence=None):
        return Decision(
            query=query,
            entity_id=candidate.get("entity_id"),
            canonical_name=candidate.get("canonical_name"),
            decision="REVIEW",
            reason_code=code,
            reason=reason,
            score=float(candidate.get("score", 0.0)),
            margin=float(candidate.get("margin", 0.0)),
            relation_type=relation_type,
            evidence=evidence or {},
        )
