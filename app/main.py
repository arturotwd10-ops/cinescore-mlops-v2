import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

MODEL_PATH = "models/champion_model.joblib"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
RATING_THRESHOLD = 7.0

app = FastAPI(title="CineScore MLOps v2 - Peliculas y Prediccion")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


class MovieInput(BaseModel):
    release_year: int
    genre: str
    director: str = ""
    imdb_rating_count: float = 1000.0


def _load_catalog() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in (TRAIN_PATH, TEST_PATH) if os.path.exists(p)]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
        df["decade"] = (df["release_year"] // 10) * 10
    if "imdb_rating" in df.columns:
        df["imdb_rating"] = pd.to_numeric(df["imdb_rating"], errors="coerce")
    return df


def _estimate_rating(prob: float, release_year: int, genre: str, director: str, catalog: pd.DataFrame) -> float:
    target_decade = (int(release_year) // 10) * 10
    model_implied = 5.0 + prob * 4.0

    genre_mean = 6.5
    decade_genre_mean = genre_mean

    if not catalog.empty and "imdb_rating" in catalog.columns:
        valid_df = catalog.dropna(subset=["imdb_rating"])
        if not valid_df.empty:
            genre_mean = float(valid_df["imdb_rating"].mean())

        g_df = valid_df[valid_df["genre"].astype(str).str.lower() == genre.strip().lower()]
        if len(g_df) >= 5:
            genre_mean = float(g_df["imdb_rating"].mean())
            if "decade" in g_df.columns:
                dg_df = g_df[g_df["decade"] == target_decade]
                decade_genre_mean = float(dg_df["imdb_rating"].mean()) if len(dg_df) >= 3 else genre_mean
            else:
                decade_genre_mean = genre_mean
        else:
            decade_genre_mean = genre_mean

        # Base ponderada dando mas peso al ano/decada y genero que solo al clasificador
        base_score = 0.35 * model_implied + 0.35 * decade_genre_mean + 0.30 * genre_mean

        d_query = director.strip().lower()
        if d_query and "director" in valid_df.columns:
            d_df = valid_df[valid_df["director"].astype(str).str.lower().str.contains(d_query, na=False)]
            if not d_df.empty:
                dir_mean = float(d_df["imdb_rating"].mean())
                decade_means = d_df.dropna(subset=["decade"]).groupby("decade")["imdb_rating"].mean()
                best_decade_mean = float(decade_means.max()) if not decade_means.empty else dir_mean

                if target_decade in decade_means.index:
                    dec_score = float(decade_means.loc[target_decade])
                    # Compara la decada consultada contra el promedio del director y su mejor epoca
                    diff_vs_avg = dec_score - dir_mean
                    diff_vs_peak = dec_score - best_decade_mean
                    era_adjustment = 0.65 * diff_vs_avg + 0.35 * diff_vs_peak
                    director_score = dec_score + era_adjustment
                else:
                    # Si no hay peliculas del director en esa decada, ajusta segun cercania a su mejor decada
                    if not decade_means.empty:
                        best_dec = float(decade_means.idxmax())
                        proximity = max(0.0, 1.0 - abs(target_decade - best_dec) / 30.0)
                        peak_bonus = (best_decade_mean - dir_mean) * 0.4 * proximity
                        director_score = dir_mean + peak_bonus
                    else:
                        director_score = dir_mean

                # El director y su epoca tienen un peso fuerte en el estimado final
                base_score = 0.55 * director_score + 0.25 * decade_genre_mean + 0.20 * model_implied

        return round(float(max(1.0, min(10.0, base_score))), 2)

    return round(float(max(1.0, min(10.0, 0.5 * model_implied + 0.5 * genre_mean))), 2)


def _find_similar_movies(release_year: int, genre: str, director: str, catalog: pd.DataFrame, limit: int = 4) -> list[dict]:
    if catalog.empty:
        return []
    df = catalog.copy()
    dq = director.strip().lower()
    gq = genre.strip().lower()

    if dq:
        d_df = df[df["director"].astype(str).str.lower().str.contains(dq, na=False)]
        if not d_df.empty:
            df = d_df
        elif gq and gq != "all":
            g_df = df[df["genre"].astype(str).str.lower() == gq]
            if not g_df.empty:
                df = g_df
    elif gq and gq != "all":
        g_df = df[df["genre"].astype(str).str.lower() == gq]
        if not g_df.empty:
            df = g_df

    df = df.copy()
    df["year_diff"] = (pd.to_numeric(df["release_year"], errors="coerce").fillna(2015) - release_year).abs()
    top = df.sort_values(by=["year_diff", "imdb_rating_count"], ascending=[True, False]).head(limit)

    return [
        {
            "title": str(r.get("title", "Sin titulo")),
            "release_year": int(r.get("release_year", 2020)) if pd.notna(r.get("release_year")) else 2020,
            "genre": str(r.get("genre", "other")),
            "director": str(r.get("director", "Desconocido")),
            "imdb_rating": round(float(r.get("imdb_rating", 0.0)), 2) if pd.notna(r.get("imdb_rating")) else 0.0,
        }
        for _, r in top.iterrows()
    ]


@app.get("/")
def home():
    return FileResponse("app/static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH), "rating_threshold": RATING_THRESHOLD}


@app.get("/genres")
def list_genres():
    catalog = _load_catalog()
    if catalog.empty or "genre" not in catalog.columns:
        return {"genres": ["drama", "documentary", "comedy", "action", "horror", "romance", "sci-fi", "other"]}
    top = catalog["genre"].dropna().astype(str).str.lower().value_counts().head(20).index.tolist()
    return {"genres": sorted(set(top))}


@app.post("/predict")
def predict(data: MovieInput):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail="Modelo no encontrado. Ejecuta src/train.py primero.")

    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([{
        "release_year": data.release_year,
        "imdb_rating_count": float(np.log1p(max(0.0, data.imdb_rating_count))),
        "genre": data.genre.strip().lower(),
    }])

    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0][1])
    catalog = _load_catalog()
    est_rating = _estimate_rating(prob, data.release_year, data.genre, data.director, catalog)
    similar = _find_similar_movies(data.release_year, data.genre, data.director, catalog, limit=4)

    is_high = prob >= 0.5
    return {
        "high_rating": int(is_high),
        "prediction_label": f"Alta probabilidad (> {RATING_THRESHOLD:.1f})" if is_high else f"Baja probabilidad (> {RATING_THRESHOLD:.1f})",
        "high_rating_probability": round(prob, 4),
        "estimated_rating": est_rating,
        "similar_movies": similar,
    }
