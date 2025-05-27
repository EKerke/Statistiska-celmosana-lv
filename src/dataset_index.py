import json, pickle, re, sys, argparse, shutil, subprocess
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
from tqdm import tqdm
try:
    import py7zr
except ImportError:
    py7zr = None

parser = argparse.ArgumentParser(description="la.lv inverted-index utility")
parser.add_argument("--rebuild", action="store_true", help="Rebuild index")
args = parser.parse_args()

ARCHIVE_PATH = Path(r"C:\Users\eveli\OneDrive - Rīgas Tehniskā Universitāte\thesis\Stemming\data\la.lv.7z")
EXTRACT_DIR = ARCHIVE_PATH.with_suffix("")
INDEX_PATH = ARCHIVE_PATH.parent / "la_lv_inverted_index.pkl"
DOC_TABLE_PATH = ARCHIVE_PATH.parent / "la_lv_documents.pkl"

SENT_SPLIT_RE = re.compile(r"""(?<!\d)(?<=[.!?…])(?:\s+|\n+)(?=["“”‘’(\[]?[A-ZĀČĒĢĪĶĻŅŌŖŠŪŽ])""",re.VERBOSE | re.UNICODE,)
TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
NUM_RE = re.compile(r"^\d+$")
LATIN_RE = re.compile(r"^[A-Za-zĀČĒĢĪĶĻŅŠŪŽāčēģīķļņšūž]+$")
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
URL_RE = re.compile(r"https?://\S+")
GALLERY_RE = re.compile(r'"items"\s*:\s*\[[^\]]*?\]', re.DOTALL)
CURLY_RE = re.compile(r"\{[^{}]*\}")

def extract_archive(archive: Path, target_dir: Path) -> None:
    if target_dir.exists(): 
        print(f"Archive already extracted to {target_dir}")
        return
    if py7zr:
        print(f"Extracting with py7zr → {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
        with py7zr.SevenZipFile(archive, mode="r") as z:
            z.extractall(path=target_dir)
    elif shutil.which("7z"):
        print(f"Extracting with system 7z → {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["7z", "x", str(archive), f"-o{target_dir}"], check=True)
    else:
        sys.exit(f"Cannot extract archive to {target_dir}")

def strip_byline(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    while lines and not re.search(r"[.!?…]$", lines[0]):
        lines.pop(0)
    return "\n".join(lines)

def normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def first_n_sentences(text, n: int = 4) -> str:
    if text is None:
        return ""
    raw = text if isinstance(text, str) else " ".join(map(str, text))
    txt = strip_byline(raw)
    txt = GALLERY_RE.sub(" ", txt)
    txt = CURLY_RE.sub(" ", txt)
    txt = URL_RE.sub(" ", txt)
    sentences = SENT_SPLIT_RE.split(txt, maxsplit=n)[:n]
    snippet = " ".join(sentences)
    snippet = " ".join(tok for tok in snippet.split() if not CYRILLIC_RE.search(tok))
    snippet = re.sub(r"(,\s*){2,}", " ", snippet)
    return normalise_whitespace(snippet)

def to_plain_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(x) for x in value)
    return normalise_whitespace(str(value))

def query_tokens(title: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(title)
            if tok[0].islower() and not NUM_RE.match(tok) and LATIN_RE.match(tok)]

def load_documents(text_dir: Path) -> Dict[int, Dict[str, str]]:
    documents = {}
    txt_files = sorted(text_dir.rglob("*.txt"))
    for doc_id, txt_path in enumerate(tqdm(txt_files, desc="Loading articles")):
        with open(txt_path, "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError as e:
                print(f"JSON error in {txt_path}: {e}")
                continue
        title = to_plain_text(data.get("title"))
        snippet = first_n_sentences(data.get("content") or data.get("summary"))
        if not title or not snippet:
            continue
        documents[doc_id] = {
            "title": title, 
            "snippet": snippet, 
            "query_terms": " ".join(query_tokens(title))
        }
    return documents

def build_inverted_index(documents: Dict[int, Dict[str, str]]) -> Dict[str, List[int]]:
    index = defaultdict(set)
    for doc_id, doc in documents.items():
        for token in TOKEN_RE.findall(doc["snippet"].lower()):
            if LATIN_RE.match(token) and '"' not in token and ':' not in token and ',' not in token:
                index[token].add(doc_id)
    return {token: sorted(doc_ids) for token, doc_ids in index.items()}

def save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    print(f"Saved {path.relative_to(Path.cwd())}")

def load_pickle(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)

def main() -> None:
    extract_archive(ARCHIVE_PATH, EXTRACT_DIR)
    have_cache = INDEX_PATH.exists() and DOC_TABLE_PATH.exists()
    if not args.rebuild and have_cache:
        print("Loading index…")
        index = load_pickle(INDEX_PATH)
        documents = load_pickle(DOC_TABLE_PATH)
    else:
        print("Building index…")
        documents = load_documents(EXTRACT_DIR)
        index = build_inverted_index(documents)
        save_pickle(index, INDEX_PATH)
        save_pickle(documents, DOC_TABLE_PATH)

if __name__ == "__main__":
    main()
