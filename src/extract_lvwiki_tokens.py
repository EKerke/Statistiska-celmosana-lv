import json
import re
from pathlib import Path
from collections import Counter
from tqdm import tqdm

INPUT_PATH = Path(
    r"C:\Users\eveli\OneDrive - Rīgas Tehniskā Universitāte\thesis\DS\lvwiki-20250421-cirrussearch-content.json"
)
TOKENS_OUT = INPUT_PATH.with_suffix(".all_tokens.txt")
FREQ_OUT = INPUT_PATH.with_suffix(".token_freq.tsv")

TOKEN_PATTERN = re.compile(r"[A-Za-zĀČĒĢĪĶĻŅŠŪŽāčēģīķļņšūž]+", re.UNICODE)
TRIPLE_LETTER = re.compile(r"(.)\1\1")
INVALID_CHARS = set("xyw")

def extract_text(document: dict) -> str:
    pieces = []
    for field_name in ("text", "opening_text", "auxiliary_text"):
        field = document.get(field_name)
        if isinstance(field, list):
            pieces.extend(field)
        elif isinstance(field, str) and field.strip():
            pieces.append(field)
    return " ".join(pieces)

def is_valid(token: str) -> bool:
    if len(token) <= 1:
        return False
    if INVALID_CHARS & set(token):
        return False
    if TRIPLE_LETTER.search(token):
        return False
    return True

def main():
    token_counts = Counter()
    total_tokens = 0

    with INPUT_PATH.open(encoding="utf-8") as infile, \
         TOKENS_OUT.open("w", encoding="utf-8") as tokfile:
        for line_no, line in enumerate(tqdm(infile, desc="Scanning NDJSON")):
            if line_no % 2 == 0:
                continue
            doc = json.loads(line)
            full_text = extract_text(doc).lower()

            for tok in TOKEN_PATTERN.findall(full_text):
                if not is_valid(tok):
                    continue
                token_counts[tok] += 1
                total_tokens += 1
                tokfile.write(f"{tok}\n")

    with FREQ_OUT.open("w", encoding="utf-8") as freqfile:
        for token, count in token_counts.most_common():
            freqfile.write(f"{token}\t{count}\n")

    print("Done.")
    print(f"Lines written to {TOKENS_OUT.name}:  {total_tokens:,}")
    print(f"Lines written to {FREQ_OUT.name}: {len(token_counts):,}")

if __name__ == "__main__":
    main()

# Lines written to lvwiki-20250421-cirrussearch-content.all_tokens.txt:  49,197,339
# Lines written to lvwiki-20250421-cirrussearch-content.token_freq.tsv: 1,025,349  (unique token types)