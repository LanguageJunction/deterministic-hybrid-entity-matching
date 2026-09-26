import re
from .normalization import normalize, compact, canonical_key

# Versioned, auditable rules. In production, move organization-specific
# aliases into the master-data table or a governed YAML/CSV rules file.
ABBREVIATIONS = {
    "jnj": "johnson and johnson",
    "j and j": "johnson and johnson",
    "j&j": "johnson and johnson",
    "3m ltd": "3m",
    "3m limited": "3m",
    "3mltd": "3m",
    "3m company": "3m",
    "three m": "3m",
    "three m company": "3m",
    "ibm": "international business machines",
    "ge": "general electric",
    "msft": "microsoft",
}

LEGAL_EQUIV = {
    "inc": "incorporated", "corp": "corporation", "co": "company",
    "ltd": "limited", "pvt": "private",
}

_SINGLE_INITIALS = re.compile(r"^(?:[a-z]\s+){1,}[a-z]$")

def _collapse_initials(n: str) -> str:
    toks=n.split()
    if len(toks) >= 2 and all(len(t)==1 for t in toks):
        return "".join(toks)
    return n

def rule_normalize(text: str) -> str:
    n = normalize(text)
    collapsed = _collapse_initials(n)
    c = compact(text)
    if collapsed in ABBREVIATIONS:
        return ABBREVIATIONS[collapsed]
    if n in ABBREVIATIONS:
        return ABBREVIATIONS[n]
    if c in ABBREVIATIONS:
        return ABBREVIATIONS[c]
    return n

def rule_score(a: str, b: str) -> float:
    ra, rb = rule_normalize(a), rule_normalize(b)
    if ra == rb:
        return 1.0
    if canonical_key(ra) == canonical_key(rb):
        return 0.95
    return 0.0
