import csv, pickle
from pathlib import Path

DATA_DIR = Path(r"C:\Users\eveli\OneDrive - Rīgas Tehniskā Universitāte\thesis\Stemming\data")
DOC_PKL = DATA_DIR / "la_lv_documents.pkl"
CSV_OUT = DATA_DIR / "la_lv_queries.csv"

def main():
    print("Loading document table …")
    with DOC_PKL.open("rb") as fh:
        docs = pickle.load(fh)

    with CSV_OUT.open("w", encoding="utf-8", newline="") as csv_fh:
        writer = csv.writer(csv_fh)
        writer.writerow(["doc_id", "query"])
        for doc_id, d in docs.items():
            q = d.get("query_terms", "")
            if q:
                writer.writerow([doc_id, q])

    print(f"Wrote {CSV_OUT.name}  ({CSV_OUT.stat().st_size/1024:.1f} KB)")

if __name__ == "__main__":
    main()
