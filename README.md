# 🎬 CinePredict — Box Office Intelligence

> Full-stack AI system: Cinema-themed frontend + FastAPI ML backend

---

## Project Structure

```
cinepredict/
│
├── frontend/
│   └── index.html          ← Complete UI (open in browser)
│
├── backend/
│   ├── api.py              ← FastAPI REST endpoints
│   ├── train_model.py      ← Full ML training pipeline
│   ├── utils/
│   │   ├── feature_engineering.py
│   │   └── nlp_utils.py
│   ├── data/
│   │   └── raw/            ← Place TMDB CSVs here
│   └── models/             ← Auto-generated after training
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')"
```

### 2. Download Dataset
- Go to: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
- Download `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`
- Place both in `backend/data/raw/`

### 3. Train Models
```bash
python backend/train_model.py
```

### 4. Start API
```bash
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open Frontend
```
Open frontend/index.html in your browser
```

> The frontend works in simulation mode even without the API running.
> Connect to the live API for real ML predictions.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Revenue + classification prediction |
| GET | `/health` | API health check |
| GET | `/model/metrics` | Training metrics (R², F1, etc.) |
| GET | `/model/features` | Feature importance ranking |
| GET | `/data/stats` | Dataset statistics |
| GET | `/data/genre-stats` | Genre performance data |

### Predict Request Body
```json
{
  "title": "The Final Horizon",
  "budget": 150000000,
  "runtime": 128,
  "release_month": 7,
  "release_year": 2025,
  "genres": ["Action", "Adventure"],
  "cast_popularity": 68.0,
  "director_popularity": 52.0,
  "popularity": 60.0,
  "vote_average": 7.5,
  "plot_overview": "A fearless astronaut discovers a wormhole..."
}
```

---

## Classification Thresholds

| Label | Condition |
|-------|-----------|
| **Hit** | Revenue ≥ 2× Budget |
| **Average** | 1× ≤ Revenue < 2× Budget |
| **Flop** | Revenue < Budget |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, Chart.js, Vanilla JS |
| API | FastAPI, Pydantic, Uvicorn |
| ML | scikit-learn, XGBoost, LightGBM |
| NLP | NLTK VADER sentiment |
| Data | pandas, numpy |
| AutoML | PyCaret (optional) |
