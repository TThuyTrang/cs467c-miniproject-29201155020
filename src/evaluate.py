"""One reproducible user-wise holdout evaluation with two publication-ready plots."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .recommender import Recommender, load_data
except ImportError:
    from recommender import Recommender, load_data


SEED = 467
K = 5
METHODS = ("popularity", "content", "hybrid")


def split_ratings(ratings):
    """Hold out four random observations per user; do not select by rating value."""
    rng = np.random.default_rng(SEED)
    test_indices = []
    for _, history in ratings.groupby("user_id", sort=True):
        if len(history) < 5:
            raise ValueError("This protocol requires at least five observations per user.")
        test_indices.extend(rng.choice(history.index.to_numpy(), size=4, replace=False))
    test = ratings.loc[test_indices].copy()
    train = ratings.drop(index=test_indices).copy()
    train_pairs = set(zip(train["user_id"], train["item_id"]))
    test_pairs = set(zip(test["user_id"], test["item_id"]))
    assert train_pairs.isdisjoint(test_pairs)
    assert len(train) + len(test) == len(ratings)
    return train, test


def main():
    items, ratings = load_data()
    train, test = split_ratings(ratings)
    # Only catalogue metadata and TRAIN interactions enter the recommender.
    # Held-out ratings and journals are accessed below solely for metric labels.
    engine = Recommender(items, train)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    details = []
    for user_id, held_out in test.groupby("user_id", sort=True):
        relevant = set(held_out.loc[held_out["rating"] >= 4, "item_id"])
        seen = set(train.loc[train["user_id"] == user_id, "item_id"])
        for method in METHODS:
            recommendations = engine.recommend(method, user_id=user_id, k=K)
            predicted = recommendations["item_id"].tolist() if not recommendations.empty else []
            assert not set(predicted) & seen
            hits = len(set(predicted) & relevant)
            details.append({
                "user_id": user_id, "method": method, "hits": hits,
                "precision_at_5": hits / K, "recommendation_count": len(predicted),
                "relevant_test_count": len(relevant),
                "recommended_ids": ";".join(predicted),
                "relevant_test_ids": ";".join(sorted(relevant)),
            })
    per_user = pd.DataFrame(details)
    summary = (per_user.groupby("method", sort=False)
               .agg(precision_at_5=("precision_at_5", "mean"), hits=("hits", "sum"),
                    users=("user_id", "count"), recommendations=("recommendation_count", "sum"))
               .reindex(METHODS).reset_index())
    summary["precision_at_5"] = summary["precision_at_5"].round(6)
    summary.to_csv(output / "evaluation.csv", index=False, lineterminator="\n")
    per_user.to_csv(output / "per_user_precision.csv", index=False, lineterminator="\n")
    membership = pd.concat([
        train[["user_id", "item_id"]].assign(split="train"),
        test[["user_id", "item_id"]].assign(split="test"),
    ]).sort_values(["user_id", "item_id"])
    membership.to_csv(output / "holdout_split.csv", index=False, lineterminator="\n")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    counts = ratings["rating"].value_counts().reindex(range(1, 6), fill_value=0)
    bars = ax.bar(counts.index, counts.values, color="#3574AD", width=0.65)
    ax.bar_label(bars, padding=4)
    ax.set(title="Synthetic rating distribution (all 240 observations)",
           xlabel="Explicit rating", ylabel="Number of observations", xticks=range(1, 6))
    ax.set_ylim(0, counts.max() * 1.18)
    fig.savefig(output / "rating_distribution.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    bars = ax.bar(["Popularity", "Content-based", "Hybrid"], summary["precision_at_5"],
                  color=["#8C98A5", "#3574AD", "#23856D"], width=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set(title="Precision@5 on a fixed synthetic holdout", ylabel="Mean Precision@5")
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.96, f"15 users | 12 train + 4 test ratings per user | seed {SEED}",
            transform=ax.transAxes, va="top", fontsize=9)
    fig.savefig(output / "precision_at_5.png", dpi=200)
    plt.close(fig)

    users_without_relevant = sum(
        not (history["rating"] >= 4).any() for _, history in test.groupby("user_id"))
    ceiling = sum(min(K, (history["rating"] >= 4).sum()) / K
                  for _, history in test.groupby("user_id")) / test["user_id"].nunique()
    report = (
        "CS467C: Synthetic recommendation evaluation\n"
        f"Split seed: {SEED}; train ratings: {len(train)}; test ratings: {len(test)}\n"
        "Protocol: four randomly held-out ratings per user; no selection by rating value.\n"
        "Relevance: held-out rating >= 4; rank all catalogue activities not seen in training.\n"
        "Precision@5 = held-out relevant hits / 5, averaged over all 15 users.\n"
        "Unjudged items count as no hit, not as known dislikes; fewer than five results are penalized.\n"
        f"Users with no relevant test items: {users_without_relevant} (included with zero hits).\n"
        f"Maximum measurable mean Precision@5 with this holdout: {ceiling:.6f}.\n"
        "TF-IDF uses catalogue metadata; all profiles, neighbors, means and sentiment use train only.\n\n"
        + summary.to_string(index=False)
        + "\n\nLimitations: 15 synthetic users, templated rating-conditioned journals, sparse judgments, "
        "one split, and no evidence of effectiveness on real users. Weights were fixed before evaluation; "
        "no test-set tuning or significance claim is made.\n"
    )
    (output / "evaluation_summary.txt").write_text(report, encoding="utf-8")
    print(report)
    print("Saved evaluation tables, holdout membership, and two PNG charts to results/.")


if __name__ == "__main__":
    main()
