import ast
import json
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/movies.csv"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
RATING_THRESHOLD = 7.0


def parse_primary_genre(val) -> str:
    if pd.isna(val):
        return "other"
    text = str(val).strip()
    if not text:
        return "other"
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list) and len(parsed) > 0:
                return str(parsed[0]).strip().lower()
            if isinstance(parsed, str) and parsed.strip():
                return parsed.strip().lower()
        except Exception:
            continue
    cleaned = text.strip("[]\"' ").split(",")[0].strip(" \"'").lower()
    return cleaned if cleaned else "other"


def extract_year(val) -> int | None:
    if pd.isna(val):
        return None
    match = re.search(r"\b(19\d{2}|20\d{2})\b", str(val))
    if not match:
        return None
    year = int(match.group(1))
    return year if 1900 <= year <= 2026 else None


def extract_director(credit_val) -> str:
    if pd.isna(credit_val):
        return "Desconocido"
    text = str(credit_val).strip()
    if not text:
        return "Desconocido"

    data = None
    for parser in (json.loads, ast.literal_eval):
        try:
            data = parser(text)
            break
        except Exception:
            continue

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                role_blob = json.dumps(item, ensure_ascii=False).lower()
                if "director" in role_blob:
                    names = item.get("names")
                    if isinstance(names, list) and names and isinstance(names[0], dict):
                        name = names[0].get("name") or names[0].get("displayName")
                        if name:
                            return str(name).strip()
                    for key in ("name", "director", "actor"):
                        if item.get(key):
                            return str(item[key]).strip()
        for item in data:
            if isinstance(item, dict):
                names = item.get("names")
                if isinstance(names, list) and names and isinstance(names[0], dict):
                    name = names[0].get("name") or names[0].get("displayName")
                    if name:
                        return str(name).strip()

    match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
    if match:
        return match.group(1).strip()
    return "Desconocido"


def main() -> None:
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Coloca tu CSV en {RAW_PATH}")

    usecols = [
        "title",
        "genres",
        "credit",
        "details_release_date",
        "imdb_rating",
        "imdb_rating_count",
        "poster_url",
    ]
    df = pd.read_csv(RAW_PATH, usecols=lambda c: c in usecols)
    df = df.dropna(subset=["title", "imdb_rating"]).copy()

    df["imdb_rating"] = pd.to_numeric(df["imdb_rating"], errors="coerce")
    df["imdb_rating_count"] = pd.to_numeric(
        df.get("imdb_rating_count", 0), errors="coerce"
    ).fillna(0.0)
    df = df.dropna(subset=["imdb_rating"])

    df["release_year"] = df.get("details_release_date", pd.Series(index=df.index)).apply(extract_year)
    median_year = int(df["release_year"].dropna().median()) if df["release_year"].notna().any() else 2015
    df["release_year"] = df["release_year"].fillna(median_year).astype(int)

    df["genre"] = df["genres"].apply(parse_primary_genre)
    df["director"] = df.get("credit", pd.Series(index=df.index)).apply(extract_director)
    df["poster_url"] = df.get("poster_url", pd.Series(index=df.index)).fillna("")
    df["high_rating"] = (df["imdb_rating"] > RATING_THRESHOLD).astype(int)

    if len(df) > 25000:
        df = df.sample(n=25000, random_state=42)

    cols_keep = [
        "title",
        "release_year",
        "genre",
        "director",
        "imdb_rating",
        "imdb_rating_count",
        "poster_url",
        "high_rating",
    ]
    df = df[cols_keep].reset_index(drop=True)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["high_rating"]
    )

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"Listo (> {RATING_THRESHOLD}) -> train: {len(train_df)}, test: {len(test_df)}")


if __name__ == "__main__":
    main()
