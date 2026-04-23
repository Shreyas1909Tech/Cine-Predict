"""
NLP Utilities — CinePredict Backend
=====================================
VADER sentiment + keyword-based NLP features for movie plot overviews.
"""

import re
import numpy as np

_POS = {"hero","save","love","hope","triumph","epic","magic","courage","discover",
        "amazing","spectacular","incredible","legendary","ultimate","adventure",
        "breathtaking","sacrifice","future","humanity","journey","powerful","victory"}
_NEG = {"death","war","evil","dark","terror","disaster","ruin","horror","kill",
        "threat","despair","doom","apocalypse","destroy","murder","chaos","danger"}
_ACTION   = {"fight","battle","war","mission","agent","soldier","weapon","attack","defend","escape","chase","explosive"}
_COMEDY   = {"funny","laugh","comedy","joke","hilarious","silly","humor","quirky","awkward","ridiculous"}
_ROMANCE  = {"love","romance","relationship","heart","couple","marry","date","passion","kiss","fall"}
_THRILLER = {"mystery","suspense","thriller","secret","conspiracy","tension","twist","investigation","clue"}
_SCIFI    = {"space","alien","robot","future","technology","science","galaxy","planet","quantum","universe","wormhole"}


def get_sentiment_score(text: str) -> float:
    if not text or not isinstance(text, str):
        return 0.0
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        try:
            sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            sia = SentimentIntensityAnalyzer()
        return float(sia.polarity_scores(text)["compound"])
    except Exception:
        tl = text.lower()
        pos = sum(1 for w in _POS if w in tl)
        neg = sum(1 for w in _NEG if w in tl)
        return float(np.clip((pos - neg) / 6.0, -1.0, 1.0))


def extract_nlp_features(text: str) -> dict:
    if not text or not isinstance(text, str):
        return _zero()
    words = re.findall(r"\b[a-z]+\b", text.lower())
    wset  = set(words)
    n     = max(len(words), 1)

    return {
        "nlp_word_count":           float(n),
        "nlp_pos_ratio":            sum(1 for w in words if w in _POS) / n,
        "nlp_neg_ratio":            sum(1 for w in words if w in _NEG) / n,
        "nlp_sentiment_polarity":   get_sentiment_score(text),
        "nlp_action_score":         sum(1 for w in words if w in _ACTION) / n,
        "nlp_comedy_score":         sum(1 for w in words if w in _COMEDY) / n,
        "nlp_romance_score":        sum(1 for w in words if w in _ROMANCE) / n,
        "nlp_thriller_score":       sum(1 for w in words if w in _THRILLER) / n,
        "nlp_scifi_score":          sum(1 for w in words if w in _SCIFI) / n,
        "nlp_unique_ratio":         len(wset) / n,
        "nlp_avg_word_length":      float(np.mean([len(w) for w in words])) if words else 0.0,
    }


def _zero():
    return {k: 0.0 for k in [
        "nlp_word_count","nlp_pos_ratio","nlp_neg_ratio","nlp_sentiment_polarity",
        "nlp_action_score","nlp_comedy_score","nlp_romance_score","nlp_thriller_score",
        "nlp_scifi_score","nlp_unique_ratio","nlp_avg_word_length"
    ]}


def add_nlp_features(df):
    import pandas as pd
    overviews = df.get("overview", pd.Series([""]*len(df))).fillna("")
    feats     = overviews.apply(extract_nlp_features).apply(pd.Series)
    sentiment = overviews.apply(get_sentiment_score).rename("sentiment_score")
    return pd.concat([df, feats, sentiment], axis=1)
