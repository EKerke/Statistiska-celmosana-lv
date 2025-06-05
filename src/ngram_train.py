"""
    This is the implementation of N-gram statistical stemmer.

    The implementation has been developed by Rabeya Sadia “N-gram Statistical Stemmer for Bangla Corpus” https://arxiv.org/pdf/1912.11612 https://github.com/shaoncsecu/Bangla_n-gram_Stemmer .
    And was adapted by E.P.Ķerķe for the batchelor thesis "Statistical stemming for the latvian language".
    
""" 

import argparse
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import AffinityPropagation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LEXICON_FILE = DATA_DIR / "lvwiki-20250421-cirrussearch-content.all_tokens.txt"

STEMS_DIR = PROJECT_ROOT / "stems"
STEMS_DIR.mkdir(exist_ok=True)
STEM_TXT = STEMS_DIR / "ngram_latvian.txt"

def ngram_set(word: str, n: int) -> set[str]:
    if len(word) < n:
        return set()
    return {word[i : i + n] for i in range(len(word) - n + 1)}

def dice_similarity(w1: str, w2: str, n: int) -> float:
    if w1 == w2:
        return 1.0
    g1, g2 = ngram_set(w1, n), ngram_set(w2, n)
    if not g1 or not g2:
        return 0.0
    overlap = len(g1 & g2)
    return (2.0 * overlap) / (len(g1) + len(g2))

def load_tokens(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            tok = line.rstrip("\n")
            if tok:
                yield tok

def main() -> None:
    ap = argparse.ArgumentParser(description="Train n-gram AP stemmer")
    ap.add_argument("--n", type=int, default=2, metavar="N", help="character n-gram size for similarity (default 2)")
    ap.add_argument("--min-freq", type=int, default=2, metavar="F", help="skip tokens with frequency < F (default 2)")
    ap.add_argument("--max-tokens", type=int, default=75_000, metavar="T", help="hard cap on tokens to cluster (default 75 000)")
    ap.add_argument("--damping", type=float, default=0.5, help="AffinityPropagation damping (default 0.5)")
    args = ap.parse_args()

    print("Reading", LEXICON_FILE)
    counts = Counter(load_tokens(LEXICON_FILE))
    vocab = [tok for tok, f in counts.items() if f >= args.min_freq]
    vocab.sort()
    if len(vocab) > args.max_tokens:
        vocab = vocab[: args.max_tokens]
    print(f"Clustering {len(vocab):,} tokens " f"(n={args.n}, min_freq ≥ {args.min_freq}, max {args.max_tokens})")

    token_arr = np.asarray(vocab)
    v_size = len(token_arr)

    print("Building similarity matrix")
    sim = np.zeros((v_size, v_size), dtype=np.float32)
    for i, j in itertools.combinations_with_replacement(range(v_size), 2):
        s = dice_similarity(token_arr[i], token_arr[j], args.n)
        sim[i, j] = sim[j, i] = s

    print("Running AffinityPropagationn")
    ap_cl = AffinityPropagation(affinity="precomputed", damping=args.damping, verbose=True)
    ap_cl.fit(sim)

    centres = token_arr[ap_cl.cluster_centers_indices_]
    stem_map = {tok: centres[label]
                for tok, label in zip(token_arr, ap_cl.labels_)}

    print("Writing", STEM_TXT, "with", len(stem_map), "entries")
    with STEM_TXT.open("w", encoding="utf-8") as fh:
        for tok, stem in stem_map.items():
            fh.write(f"{stem}\t{tok}\n")
    print("Done")

if __name__ == "__main__":
    main()
