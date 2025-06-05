"""
    This is the implementation of GRAS statistical stemmer. 

    The java implementation has been developed by Kyumars Sheykh Esmaili, 
    a member of the Kurdish Language Processing Project (KLPP) team at University of Kurdistan, Sanandaj, Iran. 
    And was translated to python by E.P.Ķerķe for the batchelor thesis "Statistical stemming for the latvian language".
    
""" 
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set

LANDA = 6
ALPHA = 6
GAMMA = 0.5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LEXICON_FILE_PATH = DATA_DIR / "lvwiki-20250421-cirrussearch-content.all_tokens.txt"
STEMS_DIR = PROJECT_ROOT / "stems"
STEMS_DIR.mkdir(exist_ok=True)
OUTPUT_FILE_PATH = STEMS_DIR / "grass_clusters.txt"

def longest_common_substring(left: str, right: str) -> str:
    common = left[:LANDA]
    limit = min(len(left), len(right))
    for i in range(LANDA, limit):
        if left[i] != right[i]:
            break
        common += left[i]
    return common

def _unordered_pair_key(a: str, b: str) -> str:
    hi, lo = sorted((a, b), reverse=True)
    return f"{hi}:{lo}"

def create_key_for_suffix_pairs(a: str, b: str) -> str:
    return _unordered_pair_key(a, b)

def create_key(a: str, b: str) -> str:
    return _unordered_pair_key(a, b)

suffix_pair_map: Dict[str, int] = {}

def _process_group_for_suffix_pairs(group: List[str]) -> None:
    for a, b in combinations(group, 2):
        lcs = longest_common_substring(a, b)
        sa, sb = a[len(lcs):], b[len(lcs):]
        if sa != sb:
            k = create_key_for_suffix_pairs(sa, sb)
            suffix_pair_map[k] = suffix_pair_map.get(k, 0) + 1

def compute_frequent_suffix_pairs() -> None:
    terms = [
        t.strip()
        for t in Path(LEXICON_FILE_PATH).read_text(encoding="utf-8").splitlines()
        if t.strip() and len(t.strip()) >= LANDA
    ]
    terms.sort()
    group, current_prefix = [], None
    for term in terms:
        pref = term[:LANDA]
        if pref != current_prefix:
            if group:
                _process_group_for_suffix_pairs(group)
            group, current_prefix = [term], pref
        elif term not in group:
            group.append(term)
    if group:
        _process_group_for_suffix_pairs(group)

_seen_clusters: Set[str] = set()

def _build_node_pairs(group: List[str]) -> Dict[str, int]:
    pairs: Dict[str, int] = {}
    for a, b in combinations(group, 2):
        lcs = longest_common_substring(a, b)
        sa, sb = a[len(lcs):], b[len(lcs):]
        if sa != sb:
            sp_key = create_key_for_suffix_pairs(sa, sb)
            weight = suffix_pair_map.get(sp_key, 0)
            if weight >= ALPHA:
                pairs[create_key(a, b)] = weight
    return pairs

def compute_cohesion(left: List[str], right: List[str]) -> float:
    if not right:
        return 0.0
    shared = sum(1 for x in left if x in right)
    return (1 + shared) / len(right)

def convert_to_graph_and_cluster(np_map: Dict[str, int], writer) -> None:
    adjacency: Dict[str, List[str]] = {}
    for key in np_map:
        a, b = key.split(":")
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    for n, nbrs in adjacency.items():
        adjacency[n] = sorted(nbrs, key=lambda x: np_map.get(create_key(n, x), 0), reverse=True)
    while adjacency:
        root = max(adjacency, key=lambda k: len(adjacency[k]))
        cluster = [root]
        while True:
            nxt = next((z for z in adjacency[root] if z not in cluster), None)
            if not nxt:
                break
            if compute_cohesion(adjacency[root], adjacency.get(nxt, [])) >= GAMMA:
                cluster.append(nxt)
            else:
                adjacency[root].remove(nxt)
                adjacency[nxt].remove(root)
        if len(cluster) > 1:
            key = cluster[0] + "|" + ",".join(sorted(cluster[1:]))
            if key not in _seen_clusters:
                _seen_clusters.add(key)
                writer.write(cluster[0] + "\n")
                writer.write(",".join(cluster) + "\n")
                writer.write("----------\n")
        for z in cluster:
            adjacency.pop(z, None)
        for z in list(adjacency.keys()):
            adjacency[z] = [x for x in adjacency[z] if x not in cluster]

def form_the_classes() -> None:
    terms = sorted(
        {
            t.strip()
            for t in Path(LEXICON_FILE_PATH).read_text(encoding="utf-8").splitlines()
            if t.strip() and len(t.strip()) >= LANDA
        }
    )
    with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as out:
        group, current_prefix = [], None
        for term in terms:
            pref = term[:LANDA]
            if pref != current_prefix:
                if group:
                    convert_to_graph_and_cluster(_build_node_pairs(group), out)
                group, current_prefix = [term], pref
            elif term not in group:
                group.append(term)
        if group:
            convert_to_graph_and_cluster(_build_node_pairs(group), out)

_cluster_map: Dict[str, str] | None = None

def _initialize() -> None:
    global _cluster_map
    if _cluster_map is not None:
        return
    compute_frequent_suffix_pairs()
    form_the_classes()
    mapping: Dict[str, str] = {}
    lines = [
        ln.strip()
        for ln in Path(OUTPUT_FILE_PATH).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    i = 0
    while i + 1 < len(lines):
        root = lines[i]
        members = lines[i + 1].split(",")
        for m in members:
            mapping[m] = root
        i += 3
    _cluster_map = mapping

def grass_tokens(tokens: List[str]) -> List[str]:
    _initialize()
    return [_cluster_map.get(t, t) for t in tokens]
