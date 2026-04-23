"""
Feature Engineering — CinePredict Backend
==========================================
Preprocessing, feature extraction, and inference vector builder.
"""

import json, ast, re
import numpy as np
import pandas as pd

ALL_GENRES = [
    "Action","Adventure","Animation","Comedy","Crime","Documentary",
    "Drama","Family","Fantasy","Foreign","History","Horror","Music",
    "Mystery","Romance","Science Fiction","TV Movie","Thriller","War","Western"
]

# ── Parsing helpers ──────────────────────────────────────────────────────────
def _parse_json(val):
    if pd.isna(val) or str(val).strip() in ("", "[]", "{}", "nan"):
        return []
    try:
        return json.loads(val)
    except Exception:
        try:
            return ast.literal_eval(str(val))
        except Exception:
            return []

def extract_genre_names(val):
    return [i.get("name","") for i in _parse_json(val) if isinstance(i, dict) and i.get("name")]

def extract_cast_popularity(val, top_n=3):
    items = _parse_json(val)
    pops  = [float(i.get("popularity",0)) for i in items if isinstance(i,dict)][:top_n]
    return float(np.mean(pops)) if pops else 0.0

def extract_cast_size(val):
    return float(len(_parse_json(val)))

def extract_director_popularity(val):
    for i in _parse_json(val):
        if isinstance(i, dict) and i.get("job") == "Director":
            return float(i.get("popularity", 0))
    return 0.0

def extract_keywords_count(val):
    return float(len(_parse_json(val)))

def extract_production_companies_count(val):
    return float(len(_parse_json(val)))

# ── Full preprocessing pipeline ──────────────────────────────────────────────
def preprocess_movies(movies_df: pd.DataFrame, credits_df: pd.DataFrame = None) -> pd.DataFrame:
    df = movies_df.copy()

    if credits_df is not None:
        credits_df = credits_df.rename(columns={"movie_id": "id"})
        df = df.merge(credits_df[["id","cast","crew"]], on="id", how="left")

    # Numeric coercion
    for col in ["budget","revenue","runtime","popularity","vote_average","vote_count"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    # Remove rows with zero budget/revenue
    df = df[(df["budget"] > 10_000) & (df["revenue"] > 10_000)].copy()

    # Release date
    df["release_date"]      = pd.to_datetime(df.get("release_date"), errors="coerce")
    df["release_month"]     = df["release_date"].dt.month.fillna(6).astype(float)
    df["release_year"]      = df["release_date"].dt.year.fillna(2000).astype(float)
    df["release_dayofweek"] = df["release_date"].dt.dayofweek.fillna(4).astype(float)
    df["is_summer"]         = df["release_month"].isin([6,7,8]).astype(float)
    df["is_holiday"]        = df["release_month"].isin([11,12]).astype(float)
    df["is_weekend_release"]= df["release_dayofweek"].isin([4,5]).astype(float)

    # Genre encoding
    df["genre_list"] = df.get("genres", pd.Series([[]]*len(df))).apply(extract_genre_names)
    for g in ALL_GENRES:
        col = f"genre_{g.lower().replace(' ','_').replace('-','_')}"
        df[col] = df["genre_list"].apply(lambda gl: float(g in gl))
    df["genre_count"] = df["genre_list"].apply(len).astype(float)

    # Cast / crew
    df["cast_popularity"]    = df.get("cast", pd.Series([""]*len(df))).apply(extract_cast_popularity)
    df["cast_size"]          = df.get("cast", pd.Series([""]*len(df))).apply(extract_cast_size)
    df["director_popularity"]= df.get("crew", pd.Series([""]*len(df))).apply(extract_director_popularity)

    # Keywords / companies
    df["keywords_count"]             = df.get("keywords", pd.Series([""]*len(df))).apply(extract_keywords_count)
    df["production_companies_count"] = df.get("production_companies", pd.Series([""]*len(df))).apply(extract_production_companies_count)

    # Transforms
    df["log_budget"]        = np.log1p(df["budget"])
    df["log_revenue"]       = np.log1p(df["revenue"])
    df["budget_per_minute"] = df.apply(lambda r: r["budget"]/r["runtime"] if r["runtime"]>0 else 0, axis=1)

    # Label
    df["roi"]               = df["revenue"] / df["budget"].replace(0, np.nan)
    df["performance_label"] = df["roi"].apply(lambda r: "Hit" if r>=2 else ("Average" if r>=1 else "Flop"))

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {
        "id","title","original_title","tagline","homepage","status","overview",
        "genres","keywords","production_companies","production_countries",
        "spoken_languages","cast","crew","release_date","genre_list",
        "revenue","log_revenue","performance_label","roi","budget",
    }
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


# ── Inference vector ─────────────────────────────────────────────────────────
def build_feature_vector(budget, runtime, release_month, release_year, genres,
                         popularity, cast_score, director_score, vote_average,
                         sentiment_score, nlp_features, feature_cols):
    row = {
        "budget":                       float(budget),
        "runtime":                      float(runtime),
        "popularity":                   float(popularity),
        "vote_average":                 float(vote_average),
        "vote_count":                   500.0,
        "cast_popularity":              float(cast_score),
        "director_popularity":          float(director_score),
        "cast_size":                    20.0,
        "log_budget":                   float(np.log1p(budget)),
        "budget_per_minute":            float(budget / runtime) if runtime > 0 else 0.0,
        "release_month":                float(release_month),
        "release_year":                 float(release_year),
        "release_dayofweek":            4.0,
        "is_summer":                    float(release_month in [6,7,8]),
        "is_holiday":                   float(release_month in [11,12]),
        "is_weekend_release":           1.0,
        "genre_count":                  float(len(genres)),
        "keywords_count":               5.0,
        "production_companies_count":   2.0,
        "sentiment_score":              float(sentiment_score),
    }
    for g in ALL_GENRES:
        col = f"genre_{g.lower().replace(' ','_').replace('-','_')}"
        row[col] = float(g in genres or g.replace(" ","_") in genres or g.replace(" ","") in genres)

    if nlp_features:
        row.update(nlp_features)

    df_row = pd.DataFrame([row])
    if feature_cols is not None:
        for c in feature_cols:
            if c not in df_row.columns:
                df_row[c] = 0.0
        df_row = df_row[[c for c in feature_cols if c in df_row.columns]]
    return df_row
