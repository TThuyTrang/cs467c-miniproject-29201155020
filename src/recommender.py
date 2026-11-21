"""Popularity, content, user-based CF, and sentiment-aware hybrid recommendations."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

try:
    from .nlp_pipeline import ActivityTextIndex, analyze_sentiment
except ImportError:
    from nlp_pipeline import ActivityTextIndex, analyze_sentiment


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("popularity", "content", "collaborative", "hybrid")


def load_data():
    return pd.read_csv(ROOT / "data/items.csv"), pd.read_csv(ROOT / "data/ratings.csv")


class Recommender:
    """Use only the supplied interaction history; held-out feedback must stay outside."""

    def __init__(self, items: pd.DataFrame, ratings: pd.DataFrame):
        self.items = items.copy().reset_index(drop=True)
        self.ratings = ratings.copy()
        if self.items["item_id"].duplicated().any():
            raise ValueError("Activity IDs must be unique.")
        if self.ratings.duplicated(["user_id", "item_id"]).any():
            raise ValueError("User-item ratings must be unique.")
        if not self.ratings["item_id"].isin(self.items["item_id"]).all():
            raise ValueError("A rating references an unknown activity.")
        if not self.ratings["rating"].isin([1, 2, 3, 4, 5]).all():
            raise ValueError("Ratings must be integers from 1 to 5.")
        self.text = ActivityTextIndex(self.items)
        self.positions = {key: index for index, key in enumerate(self.items["item_id"])}
        self.ratings["sentiment"] = self.ratings["journal_feedback"].fillna("").map(
            lambda text: analyze_sentiment(text)["score"]).astype(float)
        stats = self.ratings.groupby("item_id")["rating"].agg(["mean", "count"])
        stats = stats.reindex(self.items["item_id"])
        self.counts = stats["count"].fillna(0).to_numpy(dtype=int)
        self.popularity = stats["mean"].where(stats["count"] >= 2).to_numpy()
        sentiment = self.ratings.groupby("item_id")["sentiment"].mean()
        self.sentiment = sentiment.reindex(self.items["item_id"]).to_numpy()
        self.matrix = self.ratings.pivot(index="user_id", columns="item_id", values="rating")

    def history(self, user_id):
        return self.ratings[self.ratings["user_id"] == user_id]

    def profile(self, user_id, sentiment_aware: bool = False):
        liked = self.history(user_id)
        liked = liked[liked["rating"] >= 4]
        if liked.empty:
            return None
        weights = (liked["rating"] - 3).to_numpy(dtype=float)
        if sentiment_aware:
            weights *= 1 + 0.25 * liked["sentiment"].to_numpy()
        positions = [self.positions[key] for key in liked["item_id"]]
        vector = csr_matrix(weights.reshape(1, -1)) @ self.text.matrix[positions]
        return vector / weights.sum()

    def collaborative_predictions(self, user_id):
        """Positive Pearson neighbors on co-rated items, shrunk by n/(n+2).

        At least two co-ratings are required. Use the five nearest positive
        neighbors and a mean-centered, similarity-weighted rating prediction.
        Missing ratings remain missing; they are never treated as zero ratings.
        """
        predictions = np.full(len(self.items), np.nan)
        if user_id not in self.matrix.index:
            return predictions
        target = self.matrix.loc[user_id]
        neighbors = []
        for other_id, other in self.matrix.iterrows():
            if other_id == user_id:
                continue
            common = target.notna() & other.notna()
            n_common = int(common.sum())
            if n_common < 2:
                continue
            left = target[common].to_numpy()
            right = other[common].to_numpy()
            left, right = left - left.mean(), right - right.mean()
            denominator = np.linalg.norm(left) * np.linalg.norm(right)
            similarity = float(left @ right / denominator) if denominator else 0.0
            similarity *= n_common / (n_common + 2)
            if similarity > 0:
                neighbors.append((similarity, other_id))
        neighbors.sort(key=lambda pair: (-pair[0], pair[1]))
        for position, item_id in enumerate(self.items["item_id"]):
            numerator = denominator = 0.0
            for similarity, other_id in neighbors[:5]:
                other = self.matrix.loc[other_id]
                rating = other.get(item_id, np.nan)
                if pd.notna(rating):
                    numerator += similarity * (rating - other.mean())
                    denominator += similarity
            if denominator:
                predictions[position] = np.clip(target.mean() + numerator / denominator, 1, 5)
        return predictions

    def recommend(self, method="hybrid", user_id=None, query=None, similar_to=None, k=5):
        """Return supported unseen candidates with scores, keywords, and fallback reasons."""
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        reference = None
        excluded = set(self.history(user_id)["item_id"])
        if similar_to is not None:
            if similar_to not in self.positions:
                raise ValueError(f"Unknown activity: {similar_to}")
            reference = self.text.matrix[self.positions[similar_to]]
            excluded.add(similar_to)
        elif query is not None:
            reference = self.text.transform(query)
        elif user_id is not None:
            reference = self.profile(user_id, sentiment_aware=(method == "hybrid"))

        # A query with no vocabulary match has no content evidence to explain.
        if query is not None and (reference is None or reference.nnz == 0):
            return pd.DataFrame()
        content = self.text.similarities(reference) if reference is not None else None
        cf = self.collaborative_predictions(user_id) if method in {"collaborative", "hybrid"} else None
        no_history = self.history(user_id).empty
        rows = []
        for position, item in self.items.iterrows():
            if item["item_id"] in excluded:
                continue
            pop = self.popularity[position]
            cf_value = cf[position] if cf is not None else np.nan
            cf_fallback = pd.isna(cf_value)
            if cf_fallback:
                cf_value = pop
            explanation = ""
            if method == "popularity" or (method == "content" and reference is None) or (
                method == "hybrid" and no_history and reference is None
            ):
                if pd.isna(pop):
                    continue
                score = float(pop) if method == "popularity" else (float(pop) - 1) / 4
                explanation = f"Mean rating {pop:.2f}/5 from {self.counts[position]} observed reviews"
                if method != "popularity":
                    explanation = "No usable personal history; popularity fallback. " + explanation
            elif method == "content":
                score = float(content[position])
                if score <= 0:
                    continue
                explanation = "TF-IDF cosine similarity to query, activity, or liked history"
            elif method == "collaborative":
                if pd.isna(cf_value):
                    continue
                score = float(cf_value)
                explanation = ("No usable neighbor rating; popularity fallback" if cf_fallback
                               else "Mean-centered rating prediction from similar users")
            else:
                components = []
                if content is not None:
                    components.append((0.45, float(content[position])))
                if pd.notna(cf_value):
                    components.append((0.40, (float(cf_value) - 1) / 4))
                if pd.notna(self.sentiment[position]):
                    components.append((0.15, (float(self.sentiment[position]) + 1) / 2))
                if not components or (self.counts[position] == 0 and (
                    content is None or content[position] <= 0
                )):
                    continue
                score = sum(weight * value for weight, value in components) / sum(
                    weight for weight, _ in components)
                explanation = "Weighted content, collaborative, and observed journal sentiment"
                if cf_fallback and pd.notna(pop):
                    explanation += "; CF component uses popularity fallback"
                if self.counts[position] == 0:
                    explanation = "Zero-rating activity: available content weight renormalized to 1"
            keywords = (self.text.matching_keywords(reference, position)
                        if reference is not None else [])
            rows.append({**item.to_dict(), "score": score,
                         "matching_keywords": ", ".join(keywords) or "(no shared content terms)",
                         "rating_count": int(self.counts[position]), "reason": explanation})
        if not rows:
            return pd.DataFrame()
        return (pd.DataFrame(rows).sort_values(["score", "item_id"], ascending=[False, True])
                .head(k).reset_index(drop=True))
