"""
    This is the implementation of YASS statistical stemmer.

    The implementation has been developed by Dhyanil Mehta https://github.com/DhyanilMehta/IT550-Information-Retrieval .
    And was translated to python by E.P.Ķerķe for the batchelor thesis "Statistical stemming for the latvian language".
    
""" 

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TOKEN_FILE = DATA_DIR / "lvwiki-20250421-cirrussearch-content.all_tokens.txt"
STEMS_DIR = PROJECT_ROOT / "stems"
STEMS_DIR.mkdir(exist_ok=True)
MAPPING_JSON = STEMS_DIR / "yass_clusters.json"

class DistanceMeasure:
    def __init__(self) -> None:
        self._sum_1_to_30 = float(np.sum([1.0 / (2**i) for i in range(30)]))

    @staticmethod
    def _penalty_indexes(a: str, b: str) -> List[int]:
        a_low, b_low = a.lower(), b.lower()
        length = max(len(a_low), len(b_low))
        a_pad, b_pad = a_low.ljust(length, "*"), b_low.ljust(length, "*")
        return [i for i, (x, y) in enumerate(zip(a_pad, b_pad)) if x != y]

    def d2(self, s1: str, s2: str) -> float:
        indexes = self._penalty_indexes(s1, s2)
        if not indexes:
            return float("inf")
        m = indexes[0]
        if m == 0:
            return float("inf")
        diff = max(len(s1), len(s2)) - m
        sub = float(np.sum([1.0 / (2**i) for i in range(diff, 30)]))
        return (1.0 / m) * (self._sum_1_to_30 - sub)

def build_yass_mapping(
    thresh: float,
    index_path: Optional[Path] = None,
    min_df: int = 10,
    batch_size: int = 1000,
    metric: str = "d2",
) -> Dict[str, str]:
    if index_path:
        with open(index_path, "rb") as fh:
            inverted = pickle.load(fh)
        tokens = [tok for tok, docs in inverted.items() if len(docs) >= min_df]
    else:
        tokens = [w.strip() for w in TOKEN_FILE.read_text(encoding="utf-8").splitlines() if w.strip()]
    tokens = sorted(set(tokens))

    n_tokens = len(tokens)
    print(f"[i] clustering {n_tokens:,} tokens with YASS (threshold={thresh}, min_df={min_df})")

    X = np.arange(n_tokens).reshape(-1, 1)
    dm = DistanceMeasure()
    nbrs = NearestNeighbors(
        radius=thresh,
        metric=lambda a, b: getattr(dm, metric)(tokens[int(a[0])], tokens[int(b[0])]),
        algorithm="brute",
        n_jobs=-1,
    ).fit(X)

    rows, cols = [], []
    for start in tqdm(range(0, n_tokens, batch_size), desc="building neighbors"):
        end = min(n_tokens, start + batch_size)
        block = X[start:end]
        neighbors = nbrs.radius_neighbors(block, radius=thresh, return_distance=False)
        for offset, neigh in enumerate(neighbors):
            i = start + offset
            for j in neigh:
                if i != j:
                    rows.append(i)
                    cols.append(j)

    adjacency_matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_tokens, n_tokens))
    n_clusters, labels = connected_components(csgraph=adjacency_matrix, directed=False)
    print(f"[i] extracted {n_clusters:,} clusters")

    clusters: Dict[int, List[str]] = {}
    for token, label in zip(tokens, labels):
        clusters.setdefault(label, []).append(token)

    root_token = {label: min(words, key=len) for label, words in clusters.items()}
    mapping = {tok: root_token[labels[i]] for i, tok in enumerate(tokens)}

    MAPPING_JSON.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    print(f"[✓] saved YASS mapping to {MAPPING_JSON}")
    return mapping

_cluster_map: Optional[Dict[str, str]] = None

def _load_mapping() -> Dict[str, str]:
    global _cluster_map
    if _cluster_map is None:
        if not MAPPING_JSON.exists():
            raise FileNotFoundError(
                "YASS mapping not found – run:\n"
                "    python src/yass_stemmer.py --yass-thresh 0.3 --index /path/to/la_lv_inverted_index.pkl"
            )
        _cluster_map = json.loads(MAPPING_JSON.read_text(encoding="utf-8"))
    return _cluster_map

def yass_tokens(tokens: List[str]) -> List[str]:
    mapping = _load_mapping()
    return [mapping.get(t, t) for t in tokens]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build YASS clusters")
    parser.add_argument("--yass-thresh", type=float, required=True, help="distance threshold for clustering")
    parser.add_argument("--index", type=str, default=None, help="path to la_lv_inverted_index.pkl")
    parser.add_argument("--min-df", type=int, default=10, help="minimum document frequency for a token to be clustered")
    parser.add_argument("--batch-size", type=int, default=1000, help="how many rows to process per batch")
    parser.add_argument("--metric", choices=["d2"], default="d2", help="distance measure to use")
    args = parser.parse_args()

    index_path = Path(args.index) if args.index else None
    build_yass_mapping(
        thresh=args.yass_thresh,
        index_path=index_path,
        min_df=args.min_df,
        batch_size=args.batch_size,
        metric=args.metric,
    )
