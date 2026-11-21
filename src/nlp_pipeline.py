"""Small, explainable text processing and lexicon sentiment utilities."""

import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text: str) -> str:
    """Lowercase, replace punctuation/underscores with spaces, and collapse spaces."""
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", str(text).lower()).split())


def tokenize(text: str) -> list[str]:
    """Tokenize English words and remove scikit-learn's built-in stop words."""
    return [word for word in clean_text(text).split()
            if len(word) > 1 and word not in ENGLISH_STOP_WORDS]


class ActivityTextIndex:
    """Fit catalogue text once; transform histories and queries in the same space."""

    def __init__(self, items):
        text = (items["habit_name"] + " " + items["description"]
                + " " + items["mood_goal_tags"]).tolist()
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize, token_pattern=None, lowercase=False,
        )
        self.matrix = self.vectorizer.fit_transform(text)
        self.terms = self.vectorizer.get_feature_names_out()

    def transform(self, query: str):
        return self.vectorizer.transform([query])

    def similarities(self, reference) -> np.ndarray:
        return cosine_similarity(reference, self.matrix).ravel()

    def matching_keywords(self, reference, item_index: int, limit: int = 5) -> list[str]:
        """Return shared terms, ranked by their actual TF-IDF dot-product contribution."""
        contribution = self.matrix[item_index].multiply(reference).toarray().ravel()
        indices = sorted(np.flatnonzero(contribution),
                         key=lambda index: (-contribution[index], self.terms[index]))
        return [str(self.terms[index]) for index in indices[:limit]]


POSITIVE = set("""
    helpful enjoyable enjoyed enjoy loved love excellent wonderful delighted
    rewarding refreshing calmer calm pleasant pleased satisfying satisfied
    worthwhile reassuring supported connected positive welcome glad happily
    good great ideal favorite enthusiastic eagerly highlight appreciated useful
    worth comfortable confidence
""".split())
NEGATIVE = set("""
    disliked dislike disappointing disappointed unpleasant worse unhelpful
    regretted frustrating frustrated unhappy bad poor annoying dissatisfied
    reluctant unwelcome draining drained awkward impatient inconvenient pressure
    tiring difficult struggled forced distracting interruption unpleasant
""".split())
NEGATORS = {"not", "no", "never", "neither", "nor", "without"}


def analyze_sentiment(text: str) -> dict:
    """Score matched opinion words; negate a word after a negator within three tokens.

    Scope stops at sentence punctuation. Score = signed evidence / evidence count,
    bounded to [-1, 1]. This transparent heuristic is not a trained classifier.
    """
    evidence = []
    expanded = str(text).lower().replace("n't", " not")
    for sentence in re.split(r"[.!?;]+", expanded):
        words = clean_text(sentence).split()
        for index, word in enumerate(words):
            value = 1 if word in POSITIVE else -1 if word in NEGATIVE else 0
            if value:
                negated = any(token in NEGATORS for token in words[max(0, index - 3):index])
                if negated:
                    value *= -1
                evidence.append((f"not {word}" if negated else word, value))
    score = sum(value for _, value in evidence) / len(evidence) if evidence else 0.0
    label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    explanation = ", ".join(f"{word} ({value:+d})" for word, value in evidence)
    return {"score": score, "label": label,
            "explanation": explanation or "No opinion words matched the lexicon."}
