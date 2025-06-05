"""
    This is the implementation of N-gram statistical stemmer.

    The implementation has been developed by Rabeya Sadia “N-gram Statistical Stemmer for Bangla Corpus” https://arxiv.org/pdf/1912.11612 https://github.com/shaoncsecu/Bangla_n-gram_Stemmer .
    And was adapted by E.P.Ķerķe for the batchelor thesis "Statistical stemming for the latvian language".
    
""" 

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
_STEM_TXT = BASE_DIR / "stems" / "ngram_latvian.txt"

stem_map = {}
try:
    with _STEM_TXT.open(encoding="utf-8") as fh:
        for line in fh:
            stem, tok = line.rstrip("\n").split("\t")
            stem_map[tok] = stem
except FileNotFoundError:
    raise FileNotFoundError(f"stem file not found: {_STEM_TXT}")

def ngram_tokens(tokens):
    return [stem_map.get(t, t) for t in tokens]
