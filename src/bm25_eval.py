from __future__ import annotations
import argparse
import csv
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List
from tqdm import tqdm
from latvian_stemmer import stem_tokens as latvian_stem
from ngram_stemmer import ngram_tokens
from stemmer_lv import stemlv_tokens
from grass_stemmer import grass_tokens
import yass_stemmer

K1: float = 1.2
B: float = 0.75
DATA_DIR = Path(r"C:\Users\eveli\OneDrive - Rīgas Tehniskā Universitāte\thesis\Stemming\data")
DOC_PKL = DATA_DIR / "la_lv_documents.pkl"
INDEX_PKL = DATA_DIR / "la_lv_inverted_index.pkl"
QUERY_CSV = DATA_DIR / "la_lv_queries.csv"

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BM25")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--stem", action="store_true", help="LatvianStemmer")
    group.add_argument("--stem-lv", action="store_true", help="StemmerLV")
    group.add_argument("--ngram", type=int, choices=range(2,6), metavar="[2,6]", help="character n-grams")
    group.add_argument("--grass", action="store_true", help="GRAS")
    group.add_argument("--yass", action="store_true", help="YASS")
    return p

def label(args: argparse.Namespace) -> str:
    if args.stem: return "LatvianStemmer"
    if args.stem_lv: return "StemmerLV"
    if args.ngram: return f"{args.ngram}-gram"
    if args.grass: return "GRAS"
    if args.yass: return "YASS"
    return "raw"

def resolve_normaliser(args: argparse.Namespace) -> Callable[[List[str]], List[str]]:
    if args.stem: return latvian_stem
    if args.stem_lv: return stemlv_tokens
    if args.ngram: return lambda toks: ngram_tokens(toks, args.ngram)
    if args.grass: return grass_tokens
    if args.yass: return yass_stemmer.yass_tokens
    return lambda toks: toks

def load_pickle(path: Path):
    try: return pickle.loads(path.read_bytes())
    except FileNotFoundError:
        sys.exit(f"file not found: {path}")

def build_index(raw: Dict[str, List[int]], normalise) -> Dict[str, List[int]]:
    idx: Dict[str, List[int]] = defaultdict(list)
    for token, postings in raw.items():
        for nt in normalise([token]):
            idx[nt].extend(postings)
    return {tok: sorted(set(lst)) for tok, lst in idx.items()}

def bm25_score(freq: int, dl: int, avgdl: float, idf: float) -> float:
    return idf * (freq * (K1 + 1)) / (freq + K1 * (1 - B + B * dl / avgdl))

def main() -> None:
    args = build_parser().parse_args()
    normalise = resolve_normaliser(args)
    documents = load_pickle(DOC_PKL)
    raw_index = load_pickle(INDEX_PKL)
    index = build_index(raw_index, normalise)
    tokf: Dict[int, Counter] = {}
    doclen: Dict[int, int] = {}
    bar_fmt = "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    for did, doc in tqdm(documents.items(), bar_format=bar_fmt):
        toks = normalise(doc["snippet"].lower().split())
        tokf[did] = Counter(toks)
        doclen[did] = len(toks)
    df = {tok: len(p) for tok, p in index.items()}
    avgdl = sum(doclen.values()) / len(doclen)
    N = len(doclen)
    queries: Dict[int, List[str]] = {}
    with QUERY_CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            qid = int(row["doc_id"])
            queries[qid] = normalise(row["query"].split())
    hits = mrr_sum = 0.0
    for qid, qtokens in tqdm(queries.items(), bar_format=bar_fmt):
        scores: Dict[int, float] = defaultdict(float)
        for t in qtokens:
            if t not in df:
                continue
            idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
            for doc_id in index[t]:
                scores[doc_id] += bm25_score(tokf[doc_id][t], doclen[doc_id], avgdl, idf)
        ranked = sorted(scores, key=scores.get, reverse=True)
        if qid in ranked:
            r = ranked.index(qid) + 1
            mrr_sum += 1 / r
            if r == 1:
                hits += 1
    recall = hits / len(queries)
    mrr = mrr_sum / len(queries)
    print(f"\nResults ({label(args)})")
    print(f"Recall@1: {hits}/{len(queries)} → {recall:.3f}")
    print(f"MRR: {mrr:.3f}")

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt:
        sys.exit("aborted")
