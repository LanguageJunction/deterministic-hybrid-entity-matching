from rapidfuzz.fuzz import ratio
from .normalization import normalize, compact, canonical_key
from .tokenizer import content_tokens, char_ngrams, acronym
from .rules import rule_normalize, rule_score

FEATURE_NAMES = [
    "exact_normalized", "alias_exact", "token_jaccard",
    "token_containment", "char_jaccard", "levenshtein_ratio",
    "prefix_ratio", "acronym_match", "length_ratio", "rule_score"
]

def _jaccard(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def _containment(a, b):
    if not a or not b: return 0.0
    return len(a & b) / min(len(a), len(b))

def pair_features(query: str, candidate: str, alias_exact: bool = False):
    qn, cn = normalize(query), normalize(candidate)
    rq, rc = rule_normalize(query), rule_normalize(candidate)
    qt, ct = set(content_tokens(qn)), set(content_tokens(cn))
    qg, cg = char_ngrams(compact(query)), char_ngrams(compact(candidate))
    qcompact, ccompact = compact(query), compact(candidate)
    longest = max(len(qcompact), len(ccompact), 1)
    prefix = 0
    for x, y in zip(qcompact, ccompact):
        if x != y: break
        prefix += 1
    prefix_ratio = prefix / longest
    acr = 1.0 if acronym(query) and acronym(query) == acronym(candidate) else 0.0
    length_ratio = min(len(qcompact), len(ccompact)) / longest
    return [
        float(qn == cn or rq == rc),
        float(alias_exact),
        _jaccard(qt, ct),
        _containment(qt, ct),
        _jaccard(qg, cg),
        ratio(qcompact, ccompact) / 100.0,
        prefix_ratio,
        acr,
        length_ratio,
        rule_score(query, candidate),
    ]
