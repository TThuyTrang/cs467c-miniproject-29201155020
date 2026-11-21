"""Terminal demonstration of the CS467C wellness recommendation mini-project."""

import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path

import pandas as pd

try:
    from .nlp_pipeline import analyze_sentiment
    from .recommender import METHODS, Recommender, load_data
except ImportError:
    from nlp_pipeline import analyze_sentiment
    from recommender import METHODS, Recommender, load_data


ROOT = Path(__file__).resolve().parents[1]
MOOD_QUERY = "short evening breathing exercise to reduce exam stress"


def display_recommendations(title, recommendations):
    print(f"\n{title}")
    if recommendations.empty:
        print("No supported recommendations. Try a query containing catalogue terms.")
        return
    for rank, (_, row) in enumerate(recommendations.iterrows(), start=1):
        print(f"{rank}. {row['habit_name']} [{row['item_id']}] | score={row['score']:.4f}")
        print(f"   Keywords: {row['matching_keywords']}")
        print(f"   Evidence: {row['reason']}")


def run_demo_cases(items, ratings):
    """Run three required cases with assertions; demo additions stay in memory."""
    print("CS467C - Wellness Habit and Mindfulness Activity Recommender")
    print("Synthetic data demonstration; scores are ranking signals, not health outcomes.")
    engine = Recommender(items, ratings)

    print("\nTEST 1: Mood query")
    print(f'Query: "{MOOD_QUERY}"')
    mood = engine.recommend("content", query=MOOD_QUERY)
    assert len(mood) == 5 and (mood["score"] > 0).all()
    assert not mood["matching_keywords"].eq("(no shared content terms)").any()
    display_recommendations("Content-based Top 5 (cosine similarity, 0-1)", mood)
    print("TEST 1 PASSED")

    print("\nTEST 2: Personal history")
    print("In-memory user DEMO_STUDENT likes morning stretching and journaling.")
    print("The catalogue has no literal morning-stretching item; Desk Mobility Break")
    print("is an explicit movement proxy. No existing user is relabeled.")
    print("A second liked movement session makes the fictional active preference explicit.")
    profile = pd.DataFrame([
        {"user_id": "DEMO_STUDENT", "item_id": "I002", "rating": 5,
         "journal_feedback": "I loved this mobility break as a short morning stretching pause. It was helpful and refreshing."},
        {"user_id": "DEMO_STUDENT", "item_id": "I003", "rating": 5,
         "journal_feedback": "I enjoyed this comfortable movement session before class. It was helpful and refreshing."},
        {"user_id": "DEMO_STUDENT", "item_id": "I011", "rating": 5,
         "journal_feedback": "I enjoyed journaling about good moments. Writing felt rewarding and helped me reflect."},
    ])
    print("History: I002 Desk Mobility Break = 5; I003 Beginner Bodyweight Session = 5;")
    print("         I011 Three Good Moments Journal = 5")
    personal = Recommender(items, pd.concat([ratings, profile], ignore_index=True))
    recommendations = personal.recommend("content", user_id="DEMO_STUDENT")
    assert len(recommendations) == 5
    assert not set(recommendations["item_id"]) & {"I002", "I003", "I011"}
    assert {"movement", "journaling"}.issubset(set(recommendations["category"]))
    display_recommendations("Personal-history Top 5 (cosine similarity, 0-1)", recommendations)
    sentiment = analyze_sentiment(profile.iloc[0]["journal_feedback"])
    print(f"Example journal sentiment: {sentiment['label']}, score={sentiment['score']:.3f}")
    print(f"Lexicon explanation: {sentiment['explanation']}")
    print("TEST 2 PASSED")

    print("\nTEST 3: Zero-rating activity cold start")
    new_item = {
        "item_id": "I041", "habit_name": "Evening Breathing and Exam Stress Pause",
        "category": "mindfulness", "duration_min": 5,
        "description": "Take a short evening breathing exercise before reviewing exam notes. Breathe naturally while noticing the present moment, then return to coursework at an unhurried pace.",
        "mood_goal_tags": "calm;stress_relief;breathing;evening;mindfulness",
    }
    cold = Recommender(pd.concat([items, pd.DataFrame([new_item])], ignore_index=True), ratings)
    print("I041 exists only in memory; the original CSV catalogue remains unchanged.")
    print(f"Observed ratings for I041: {cold.counts[cold.positions['I041']]}")
    print(f'Query: "{MOOD_QUERY}"')
    assert cold.counts[cold.positions["I041"]] == 0
    for method in ("content", "hybrid"):
        recommendations = cold.recommend(method, query=MOOD_QUERY)
        assert len(recommendations) == 5 and "I041" in set(recommendations["item_id"])
        display_recommendations(f"{method.title()} Top 5 (0-1)", recommendations)
    print("TEST 3 PASSED")
    print("\nAll three required test cases passed. Demo profiles and new items were not saved.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--user", help="User ID, for example U004; unknown users use a popularity fallback.")
    action.add_argument("--query", help="Free-text activity or mood query.")
    action.add_argument("--similar", help="Find activities similar to a catalogue item, for example I006.")
    action.add_argument("--demo-tests", action="store_true", help="Run and save all three required cases (also the default).")
    parser.add_argument("--method", choices=METHODS, help="Default: hybrid for users; content for text queries.")
    args = parser.parse_args()
    if args.query is not None and args.method not in (None, "content", "hybrid"):
        parser.error("Text queries support --method content or hybrid.")
    if args.similar is not None and args.method not in (None, "content"):
        parser.error("Similar-activity lookup uses --method content.")
    items, ratings = load_data()
    if args.demo_tests or (args.user is None and args.query is None and args.similar is None):
        output = io.StringIO()
        with redirect_stdout(output):
            run_demo_cases(items, ratings)
        text = output.getvalue()
        destination = ROOT / "results/sample_outputs.txt"
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        print(text, end="")
        print("Saved actual demo output to results/sample_outputs.txt")
        return
    engine = Recommender(items, ratings)
    method = args.method or ("hybrid" if args.user is not None else "content")
    if args.user is not None and engine.history(args.user).empty:
        print(f"No history for {args.user}; recommendations will use available population evidence.")
    print("Scores: content/hybrid 0-1; popularity/CF 1-5. Fallbacks are identified below.")
    try:
        recommendations = engine.recommend(method, user_id=args.user, query=args.query, similar_to=args.similar)
    except ValueError as exc:
        parser.error(str(exc))
    display_recommendations(f"{method.title()} Top 5", recommendations)


if __name__ == "__main__":
    main()
