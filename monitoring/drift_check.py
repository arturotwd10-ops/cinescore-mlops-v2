import json
import os
import pandas as pd

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
OUT_PATH = "monitoring/drift_report.json"


def main() -> None:
    ref = pd.read_csv(TRAIN_PATH)
    cur = pd.read_csv(TEST_PATH)

    report = {}
    for col in ["release_year", "imdb_rating_count", "imdb_rating"]:
        if col in ref.columns and col in cur.columns:
            ref_mean = float(ref[col].mean())
            cur_mean = float(cur[col].mean())
            rel_diff = abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-9)
            report[col] = {
                "ref_mean": round(ref_mean, 4),
                "cur_mean": round(cur_mean, 4),
                "rel_diff": round(rel_diff, 4),
                "drift_detected": bool(rel_diff > 0.15),
            }

    if "genre" in ref.columns and "genre" in cur.columns:
        ref_top = str(ref["genre"].mode()[0])
        cur_top = str(cur["genre"].mode()[0])
        report["genre"] = {
            "ref_top_genre": ref_top,
            "cur_top_genre": cur_top,
            "drift_detected": bool(ref_top != cur_top),
        }

    os.makedirs("monitoring", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Reporte guardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
