"""
CinePredict — FastAPI Backend
==============================
REST API for movie box office prediction.

Endpoints:
  POST /predict          — Revenue + classification prediction
  GET  /health           — Health check
  GET  /model/metrics    — Model performance metrics
  GET  /model/features   — Feature importance
  GET  /data/stats       — Dataset statistics
  GET  /data/genre-stats — Genre performance stats

Run:
  uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
"""

import os, json, pickle, warnings
from typing import List, Optional

# Add project root to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CinePredict API",
    description="AI-powered movie box office prediction system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")

# ── Load models at startup ───────────────────────────────────────────────────
_reg_model    = None
_clf_model    = None
_feature_cols = None
_label_enc    = None
_metrics      = {}
_feature_imp  = []

def _load_artifacts():
    global _reg_model, _clf_model, _feature_cols, _label_enc, _metrics, _feature_imp
    for name, fname, store in [
        ("reg",   "regression_model.pkl",    "_reg_model"),
        ("clf",   "classification_model.pkl","_clf_model"),
        ("fcols", "feature_columns.pkl",     "_feature_cols"),
        ("le",    "label_encoder.pkl",       "_label_enc"),
    ]:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                globals()[store] = pickle.load(f)
            print(f"  Loaded: {fname}")
        else:
            print(f"  Missing: {fname} — run train_model.py first")

    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            _metrics = json.load(f)

    fi_path = os.path.join(MODEL_DIR, "feature_importance.json")
    if os.path.exists(fi_path):
        with open(fi_path) as f:
            _feature_imp = json.load(f)

_load_artifacts()

# ── Request / Response schemas ───────────────────────────────────────────────
class PredictRequest(BaseModel):
    title:              str   = Field(..., example="The Final Horizon")
    budget:             float = Field(..., example=150_000_000, description="Budget in USD")
    runtime:            float = Field(120,  example=128)
    release_month:      int   = Field(7,    example=7, ge=1, le=12)
    release_year:       int   = Field(2025, example=2025)
    genres:             List[str] = Field(["Action","Adventure"])
    cast_popularity:    float = Field(50.0, example=68.0)
    director_popularity:float = Field(30.0, example=52.0)
    popularity:         float = Field(50.0, example=60.0)
    vote_average:       float = Field(7.0,  example=7.0, ge=0, le=10)
    plot_overview:      Optional[str] = Field("", example="An epic adventure...")


class PredictResponse(BaseModel):
    title:              str
    predicted_revenue:  float
    classification:     str
    probabilities:      dict
    roi:                float
    sentiment_score:    float
    feature_importance: dict
    model_used:         str


# ── Helper: build feature vector ─────────────────────────────────────────────
def _build_features(req: PredictRequest):
    from backend.utils.feature_engineering import build_feature_vector
    from backend.utils.nlp_utils import get_sentiment_score, extract_nlp_features

    sentiment  = get_sentiment_score(req.plot_overview or "")
    nlp_feats  = extract_nlp_features(req.plot_overview or "")

    return build_feature_vector(
        budget=req.budget,
        runtime=req.runtime,
        release_month=req.release_month,
        release_year=req.release_year,
        genres=req.genres,
        popularity=req.popularity,
        cast_score=req.cast_popularity,
        director_score=req.director_popularity,
        vote_average=req.vote_average,
        sentiment_score=sentiment,
        nlp_features=nlp_feats,
        feature_cols=_feature_cols,
    ), sentiment


# ── Helper: simulate if model missing ────────────────────────────────────────
def _simulate(req: PredictRequest):
    from backend.utils.nlp_utils import get_sentiment_score

    gm = {"Action":1.42,"Adventure":1.38,"Animation":1.52,"Comedy":1.08,
          "Crime":1.10,"Drama":0.82,"Family":1.35,"Fantasy":1.30,
          "Horror":1.85,"Mystery":1.05,"Romance":0.90,"Science Fiction":1.48,
          "Sci-Fi":1.48,"Thriller":1.12,"War":0.95,"Western":0.80}
    mb = [0.84,0.81,0.89,0.94,1.06,1.26,1.32,1.14,0.91,0.96,1.12,1.22]

    genres    = req.genres if req.genres else ["Drama"]
    g_mult    = np.mean([gm.get(g, 1.0) for g in genres])
    m_bonus   = mb[req.release_month - 1]
    cast_f    = 0.65 + req.cast_popularity / 150
    dir_f     = 0.85 + req.director_popularity / 200
    sent      = get_sentiment_score(req.plot_overview or "")
    revenue   = req.budget * g_mult * m_bonus * cast_f * dir_f * (1.7 + sent * 0.15)
    roi       = revenue / req.budget

    label = "Hit" if roi >= 2 else ("Average" if roi >= 1 else "Flop")
    if label == "Hit":
        ph, pa = 0.65, 0.22
    elif label == "Average":
        ph, pa = 0.20, 0.58
    else:
        ph, pa = 0.08, 0.22
    pf = max(0, 1 - ph - pa)

    return {
        "predicted_revenue": float(revenue),
        "classification":    label,
        "probabilities":     {"Hit": round(ph,3), "Average": round(pa,3), "Flop": round(pf,3)},
        "roi":               round(roi, 4),
        "sentiment_score":   round(sent, 4),
        "feature_importance":{"budget":0.82,"genre":0.68,"cast_popularity":0.55,
                               "release_month":0.44,"director":0.38,"popularity":0.31},
        "model_used":        "simulation (train model to enable real predictions)"
    }


# ── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "models_loaded": _reg_model is not None and _clf_model is not None,
        "version":      "1.0.0"
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Use simulation if models not trained yet
    if _reg_model is None or _clf_model is None:
        sim = _simulate(req)
        return PredictResponse(title=req.title, **sim)

    try:
        feat_df, sentiment = _build_features(req)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {e}")

    # Revenue prediction (log-space → back-transform)
    log_rev    = float(_reg_model.predict(feat_df)[0])
    revenue    = float(np.expm1(log_rev))
    roi        = revenue / req.budget

    # Classification
    clf_pred   = _clf_model.predict(feat_df)[0]
    clf_proba  = _clf_model.predict_proba(feat_df)[0]
    classes    = (_label_enc.inverse_transform(_clf_model.classes_)
                  if _label_enc is not None
                  else _clf_model.classes_)
    label      = (_label_enc.inverse_transform([clf_pred])[0]
                  if _label_enc is not None else str(clf_pred))
    probs      = {str(c): round(float(p), 4) for c, p in zip(classes, clf_proba)}

    # Feature importance (top 8)
    fi_top = {item["feature"]: item["importance"]
              for item in _feature_imp[:8]} if _feature_imp else {}

    return PredictResponse(
        title=req.title,
        predicted_revenue=round(revenue, 2),
        classification=label,
        probabilities=probs,
        roi=round(roi, 4),
        sentiment_score=round(sentiment, 4),
        feature_importance=fi_top,
        model_used=_metrics.get("reg_best_model", "unknown")
    )


@app.get("/model/metrics")
def model_metrics():
    if not _metrics:
        return {"message": "No metrics found. Run train_model.py first."}
    return _metrics


@app.get("/model/features")
def model_features(top_n: int = 20):
    if not _feature_imp:
        return {"message": "Feature importance not available. Run train_model.py first."}
    return {"features": _feature_imp[:top_n]}


@app.get("/data/stats")
def data_stats():
    import pandas as pd
    processed = os.path.join(BASE_DIR, "data", "processed_movies.csv")
    if not os.path.exists(processed):
        return {"message": "Processed data not found. Run train_model.py first."}
    df = pd.read_csv(processed)
    valid = df[(df["budget"] > 0) & (df["revenue"] > 0)]
    return {
        "total_movies":     int(len(df)),
        "valid_movies":     int(len(valid)),
        "avg_budget":       round(float(valid["budget"].mean()), 2),
        "avg_revenue":      round(float(valid["revenue"].mean()), 2),
        "avg_roi":          round(float((valid["revenue"] / valid["budget"]).mean()), 3),
        "hit_count":        int((valid.get("performance_label","") == "Hit").sum()),
        "average_count":    int((valid.get("performance_label","") == "Average").sum()),
        "flop_count":       int((valid.get("performance_label","") == "Flop").sum()),
    }


@app.get("/data/genre-stats")
def genre_stats():
    import pandas as pd
    processed = os.path.join(BASE_DIR, "data", "processed_movies.csv")
    if not os.path.exists(processed):
        return {"message": "Processed data not found. Run train_model.py first."}
    df = pd.read_csv(processed)
    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    results = []
    for col in genre_cols:
        sub = df[(df[col] == 1) & (df["revenue"] > 0)]
        if len(sub) < 10:
            continue
        results.append({
            "genre":       col.replace("genre_", "").replace("_", " ").title(),
            "count":       int(len(sub)),
            "avg_revenue": round(float(sub["revenue"].mean()), 2),
            "avg_roi":     round(float((sub["revenue"] / sub["budget"].replace(0, np.nan)).mean()), 3),
        })
    return {"genres": sorted(results, key=lambda x: x["avg_revenue"], reverse=True)}
