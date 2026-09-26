import re
import unicodedata

LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "plc", "pvt", "private", "holdings",
    "holding", "group", "ag", "sa", "nv", "gmbh"
}

AMP = re.compile(r"\b(and)\b|&", re.I)
NON_ALNUM = re.compile(r"[^a-z0-9]+")

def normalize(text: str) -> str:
    text = text or ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("&", " and ")
    text = re.sub(r"(?<=\d)(?=[a-z])", " ", text)
    text = re.sub(r"(?<=[a-z])(?=\d)", " ", text)
    text = NON_ALNUM.sub(" ", text)
    return " ".join(text.split())

def compact(text: str) -> str:
    return normalize(text).replace(" ", "")

def remove_legal_suffixes(text: str) -> str:
    toks = normalize(text).split()
    while toks and toks[-1] in LEGAL_SUFFIXES:
        toks.pop()
    return " ".join(toks)

def canonical_key(text: str) -> str:
    return remove_legal_suffixes(text)
