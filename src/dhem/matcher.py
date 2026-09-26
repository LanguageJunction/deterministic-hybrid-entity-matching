from collections import defaultdict
import math
from .normalization import normalize, compact
from .features import pair_features

try:
    import ahocorasick
except ImportError:
    ahocorasick = None

class HybridMatcher:
    def __init__(self, master_df, model=None, threshold=0.75, margin=0.08):
        self.master = master_df.reset_index(drop=True)
        self.model = model
        self.threshold = threshold
        self.margin = margin
        self.alias_map = {}
        self.token_index = defaultdict(set)
        for i, r in self.master.iterrows():
            values = [r["canonical_name"]] + str(r.get("aliases", "")).split("|")
            for alias in values:
                if not alias or alias == "nan": continue
                self.alias_map[normalize(alias)] = i
                for tok in normalize(alias).split():
                    self.token_index[tok].add(i)
        self.automaton = None
        if ahocorasick:
            a = ahocorasick.Automaton()
            for alias, idx in self.alias_map.items():
                if alias: a.add_word(alias, (alias, idx))
            a.make_automaton()
            self.automaton = a

    def candidates(self, query, top_k=100):
        n = normalize(query)
        ids = set()
        if n in self.alias_map: ids.add(self.alias_map[n])
        for tok in n.split(): ids.update(self.token_index.get(tok, set()))
        # fallback: compact substring search; for millions of rows replace this
        # with a persistent n-gram index/vectorized retrieval service.
        if not ids:
            q = compact(query)
            for i, r in self.master.iterrows():
                c = compact(r["canonical_name"])
                if q[:4] in c or c[:4] in q:
                    ids.add(i)
                if len(ids) >= top_k: break
        return list(ids)[:top_k]

    def score(self, query, idx):
        r = self.master.iloc[idx]
        aliases = [x for x in [r["canonical_name"]] + str(r.get("aliases", "")).split("|") if x and x != "nan"]
        normalized_query = normalize(query)
        alias_exact = normalized_query in {normalize(x) for x in aliases}
        feat = pair_features(query, r["canonical_name"], alias_exact)
        if self.model is None:
            return max(feat[0], feat[1], feat[2], feat[4], feat[5], feat[9])
        import torch
        with torch.no_grad():
            logit = self.model(torch.tensor([feat], dtype=torch.float32)).item()
            return 1 / (1 + math.exp(-logit))

    def match(self, query, top_k=5):
        rows = []
        for idx in self.candidates(query, top_k=max(top_k, 100)):
            score = self.score(query, idx)
            r = self.master.iloc[idx]
            aliases = [x for x in [r["canonical_name"]] + str(r.get("aliases", "")).split("|") if x and x != "nan"]
            normalized_query = normalize(query)
            matched_alias = next((a for a in aliases if normalize(a) == normalized_query and normalize(a) != normalize(r["canonical_name"])), None)
            rows.append({"entity_id": r["entity_id"], "canonical_name": r["canonical_name"], "score": score,
                         "alias_exact": bool(matched_alias), "matched_alias": matched_alias})
        rows.sort(key=lambda x: x["score"], reverse=True)
        for rank, x in enumerate(rows):
            next_score = rows[rank + 1]["score"] if rank + 1 < len(rows) else 0.0
            x["margin"] = x["score"] - next_score
            if x["score"] < self.threshold:
                x["decision"] = "NO_MATCH"
            elif rank > 0 and x["margin"] < self.margin:
                x["decision"] = "REVIEW"
            else:
                x["decision"] = "MATCH"
        return rows[:top_k]
