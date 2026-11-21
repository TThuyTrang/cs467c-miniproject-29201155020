"""Generate and validate synthetic student activity data using the standard library.

This script does not perform NLP, train models, or implement recommendations.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SEED = 4672026
ITEM_FIELDS = ["item_id", "habit_name", "category", "duration_min", "description", "mood_goal_tags"]
RATING_FIELDS = ["user_id", "item_id", "rating", "journal_feedback"]
CATEGORIES = (
    "movement", "mindfulness", "journaling", "study_recovery",
    "sleep_routine", "nutrition", "social_connection", "creative_relaxation",
)

# Five original activities per category; no external data or real people are used.
# Fields: name, category, minutes, description, semicolon-separated mood tags.
ITEM_SPECS = [
    ("Campus Loop Walk", "movement", 15,
     "Walk at a comfortable pace along a familiar campus route. Notice changes in the surroundings and leave coursework aside for a short break.",
     "fresh_air;energy;study_break"),
    ("Desk Mobility Break", "movement", 5,
     "Step away from the screen and gently move your shoulders, wrists, and ankles within a comfortable range. Return to the desk without rushing.",
     "comfort;movement;study_break"),
    ("Beginner Bodyweight Session", "movement", 15,
     "Try a short sequence of comfortable movements, such as chair squats and wall push-ups. Choose an easy pace and pause whenever needed.",
     "energy;confidence;movement"),
    ("Easy Outdoor Stroll", "movement", 20,
     "Take an unhurried walk on a level outdoor path between classes. Let the route be a change of scene rather than a fitness challenge.",
     "fresh_air;calm;movement"),
    ("Favorite Song Dance Break", "movement", 5,
     "Play a favorite song at a considerate volume and move freely in a private space. Focus on enjoying the music rather than following a routine.",
     "joy;energy;self_expression"),
    ("Comfortable Breathing Pause", "mindfulness", 5,
     "Sit comfortably and notice the natural rhythm of your breathing without holding or forcing it. When attention wanders, gently return to the next breath.",
     "calm;awareness;stress_relief"),
    ("Seated Body Awareness", "mindfulness", 10,
     "Notice how your feet, hands, and back feel while seated. Observe sensations with curiosity and adjust your position if anything feels uncomfortable.",
     "awareness;comfort;calm"),
    ("Five Senses Observation", "mindfulness", 5,
     "Pause in a familiar place and notice details you can see, hear, smell, taste, or touch. Choose whichever senses feel comfortable to explore.",
     "present_moment;awareness;focus"),
    ("Mindful Tea Break", "mindfulness", 10,
     "Prepare a familiar drink and pay attention to its warmth, aroma, and flavor. Keep the phone aside while taking a few unhurried sips.",
     "calm;present_moment;screen_break"),
    ("Quiet Outdoor Listening", "mindfulness", 10,
     "Sit in a safe outdoor space and listen to nearby and distant sounds. Let each sound pass without needing to identify or judge it.",
     "fresh_air;awareness;calm"),
    ("Three Good Moments Journal", "journaling", 5,
     "Write about three small moments you appreciated today. Include ordinary experiences, such as a useful conversation or finishing a difficult task.",
     "gratitude;reflection;positive_mood"),
    ("Unfiltered Thought Notes", "journaling", 10,
     "Write freely about whatever is occupying your attention. Leave grammar and structure aside, and finish when the scheduled time is over.",
     "reflection;mental_clarity;self_expression"),
    ("Small Win Reflection", "journaling", 5,
     "Describe one manageable task you completed and what helped you finish it. Keep the reflection specific rather than comparing yourself with others.",
     "confidence;reflection;motivation"),
    ("Values Check-In", "journaling", 15,
     "Choose a personal value, such as curiosity or kindness, and describe how it appeared in your week. Note one small action that could express it tomorrow.",
     "purpose;reflection;self_awareness"),
    ("Kind Letter to Yourself", "journaling", 10,
     "Write a brief, supportive note to yourself about a current academic challenge. Use the considerate language you would offer a classmate.",
     "self_compassion;reflection;calm"),
    ("Screen-Free Study Interval", "study_recovery", 5,
     "Set coursework and digital devices aside for a few minutes between study blocks. Look around the room and allow the break to remain unstructured.",
     "screen_break;focus;study_break"),
    ("Desk Reset", "study_recovery", 10,
     "Clear a small section of your study area and put away materials you no longer need. Leave the resources for your next task within easy reach.",
     "organization;focus;mental_clarity"),
    ("Single-Task Planning Pause", "study_recovery", 5,
     "Identify one realistic next step for an assignment and write it on a short task list. Close unrelated tabs before returning to that step.",
     "focus;organization;motivation"),
    ("Window View Break", "study_recovery", 5,
     "Move away from the screen and spend a few minutes looking at a comfortable distant view. Let your attention rest on the scene without studying it.",
     "screen_break;calm;study_break"),
    ("Between-Class Reset Walk", "study_recovery", 10,
     "Use a gap between classes for a gentle walk to a quieter part of campus. Return with enough time to settle before the next lecture.",
     "study_break;fresh_air;focus"),
    ("Evening Screen Wind-Down", "sleep_routine", 20,
     "Put coursework and optional screens aside during a planned evening interval. Choose a quiet, familiar activity as a transition toward bedtime.",
     "wind_down;screen_break;routine"),
    ("Tomorrow Preparation Ritual", "sleep_routine", 10,
     "Pack your bag and arrange the materials needed for the next morning. Finish with a short note about the first task of the day.",
     "routine;organization;wind_down"),
    ("Quiet Bedtime Reading", "sleep_routine", 15,
     "Read a few pages of a familiar, low-pressure book away from coursework. Stop at a convenient point rather than setting a reading target.",
     "wind_down;calm;screen_break"),
    ("Comfortable Bedroom Reset", "sleep_routine", 10,
     "Spend a few minutes arranging bedding and reducing avoidable clutter or noise. Adapt the routine to a shared room and available space.",
     "comfort;routine;wind_down"),
    ("Consistent Evening Check-In", "sleep_routine", 5,
     "Notice the time, your current energy, and what remains on your task list. Choose a realistic stopping point for optional work this evening.",
     "self_awareness;routine;wind_down"),
    ("Water Refill Pause", "nutrition", 5,
     "Refill your water bottle before the next study session. Use the brief pause to notice whether you would like a drink without setting an intake target.",
     "hydration;routine;study_break"),
    ("Simple Snack Preparation", "nutrition", 10,
     "Prepare an affordable snack using foods that suit your preferences and dietary needs. Put away the ingredients before returning to your next activity.",
     "nourishment;routine;self_care"),
    ("Mindful Lunch Opening", "nutrition", 10,
     "Spend the first part of lunch noticing the food's textures and flavors. Choose a comfortable pace and pause scrolling while beginning the meal.",
     "nourishment;present_moment;screen_break"),
    ("Budget-Friendly Meal Planning", "nutrition", 15,
     "Plan a few simple meals around your schedule, budget, and available cooking facilities. Make a short shopping list using familiar ingredients.",
     "organization;nourishment;routine"),
    ("Breakfast Preparation", "nutrition", 10,
     "Set out suitable ingredients or supplies for a convenient breakfast. Keep the plan flexible enough for early classes and shared kitchen access.",
     "routine;nourishment;self_care"),
    ("Classmate Check-In", "social_connection", 10,
     "Ask a classmate how their week is going during a mutually convenient break. Listen attentively without needing to solve their problems.",
     "connection;belonging;kindness"),
    ("Shared Campus Walk", "social_connection", 20,
     "Invite a friend for a comfortable walk on campus if both schedules allow. Let the conversation move beyond assignments when it feels natural.",
     "connection;fresh_air;positive_mood"),
    ("Appreciation Message", "social_connection", 5,
     "Send a brief, sincere message thanking someone for a specific helpful action. Avoid expecting an immediate response.",
     "gratitude;connection;kindness"),
    ("Casual Club Conversation", "social_connection", 20,
     "Join a low-pressure conversation at a student club or common area. Stay only as long as feels comfortable and respect other people's availability.",
     "belonging;connection;confidence"),
    ("Supportive Study Partner Break", "social_connection", 15,
     "Take a planned break with a willing study partner and exchange one manageable challenge from the week. Give each person time to speak without interruption.",
     "connection;study_break;encouragement"),
    ("Freeform Doodle Page", "creative_relaxation", 10,
     "Fill part of a page with simple lines, shapes, or patterns. Treat the activity as exploration rather than an artwork that needs to be evaluated.",
     "creativity;self_expression;calm"),
    ("Favorite Playlist Pause", "creative_relaxation", 10,
     "Listen to a few familiar songs at a comfortable volume. Set other tasks aside and notice the parts of the music that hold your attention.",
     "joy;calm;screen_break"),
    ("Short Creative Writing", "creative_relaxation", 15,
     "Write a small scene, poem, or fictional conversation unrelated to coursework. Stop without editing and leave the piece unfinished if needed.",
     "creativity;self_expression;mental_clarity"),
    ("Everyday Photography Walk", "creative_relaxation", 20,
     "Look for interesting colors or textures during a short walk and photograph objects or scenery. Respect privacy and avoid photographing people without permission.",
     "creativity;fresh_air;awareness"),
    ("Simple Paper Folding", "creative_relaxation", 10,
     "Follow an easy paper-folding pattern using spare paper. Work at a relaxed pace and focus on the process rather than a perfect result.",
     "creativity;focus;calm"),
]

# Three users per group; category scores follow CATEGORIES and are not learned.
PROFILES = [
    ("Active students", (5, 3, 2, 4, 3, 4, 3, 2)),
    ("Quiet reflectors", (2, 5, 5, 3, 4, 3, 2, 4)),
    ("Focused learners", (3, 4, 3, 5, 4, 3, 2, 2)),
    ("Social explorers", (4, 2, 2, 3, 3, 4, 5, 4)),
    ("Routine builders", (3, 4, 3, 4, 5, 5, 2, 2)),
]

# Context clauses are fictional experiences, not claims of demonstrated efficacy.
CONTEXT = {
    "movement": (
        "Moving around felt refreshing after sitting through lectures.",
        "The movement felt tiring, and I struggled to settle back into studying.",
        "The change of pace was pleasant, but I did not notice much difference afterward.",
    ),
    "mindfulness": (
        "Paying attention to the present moment helped me feel calmer before coursework.",
        "Directing my attention this way felt frustrating, and my thoughts kept drifting.",
        "I noticed a few details, but maintaining my attention still took effort.",
    ),
    "journaling": (
        "Putting my thoughts into words helped me understand my day more clearly.",
        "Writing about my day felt forced, and I finished feeling more frustrated.",
        "Writing gave me something to reflect on, although the exercise felt a little repetitive.",
    ),
    "study_recovery": (
        "The pause gave me a clear transition back to my next assignment.",
        "The interruption broke my concentration, and returning to the assignment was difficult.",
        "The pause was manageable, but my concentration was about the same when I returned.",
    ),
    "sleep_routine": (
        "The evening routine felt reassuring and made it easier to put coursework aside.",
        "The extra evening task felt inconvenient and added pressure to an already busy night.",
        "The routine was easy enough to follow, but it did not noticeably change my evening.",
    ),
    "nutrition": (
        "Taking care of a practical food or drink routine made my study day feel more organized.",
        "The preparation felt like an unwelcome chore and did not fit my schedule.",
        "The practical task was useful, although it was less enjoyable than other breaks.",
    ),
    "social_connection": (
        "The friendly exchange left me feeling supported and more connected to campus life.",
        "The interaction felt awkward and left me more drained than before.",
        "The exchange was polite, but I felt only slightly more connected afterward.",
    ),
    "creative_relaxation": (
        "Making room for a creative activity was enjoyable and gave my mind a welcome break.",
        "The creative task felt distracting, and I became impatient with the activity.",
        "The creative break was mildly enjoyable, although I would not choose it every day.",
    ),
}

# Eight alternatives per score allow unique journals without rejection sampling.
# Ratings 1-2 are negative, 3 is mixed/neutral, and 4-5 are positive by design.
FEEDBACK_TEMPLATES = {
    1: [
        'I strongly disliked "{name}". {negative} I would avoid repeating it.',
        '"{name}" was very disappointing. {negative} It was a poor use of my break.',
        'I had an unpleasant experience with "{name}". {negative} It did not suit me at all.',
        '"{name}" made this break feel worse. {negative} I do not want to try it again.',
        'I found "{name}" extremely unhelpful. {negative} I regretted choosing it today.',
        '"{name}" was frustrating from start to finish. {negative} I would choose a different activity next time.',
        'I was very unhappy with "{name}". {negative} It left me wishing I had skipped it.',
        '"{name}" was a bad fit for me. {negative} I found almost nothing enjoyable about it.',
    ],
    2: [
        'I did not enjoy "{name}". {negative} It did not feel worth the time today.',
        '"{name}" was disappointing overall. {negative} I would prefer a different break.',
        'I found "{name}" more annoying than helpful. {negative} I am reluctant to repeat it.',
        '"{name}" did not work well for me. {negative} I was dissatisfied with the experience.',
        'I felt let down by "{name}". {negative} It would be low on my list next time.',
        '"{name}" was not an enjoyable choice today. {negative} I would rather spend the time elsewhere.',
        'I struggled to appreciate "{name}". {negative} The experience was mostly unpleasant.',
        '"{name}" fell short of my expectations. {negative} I would not recommend it for my own routine.',
    ],
    3: [
        '"{name}" was acceptable. {mixed} I might try it again in a different setting.',
        'I had mixed feelings about "{name}". {mixed} It was neither especially helpful nor unpleasant.',
        '"{name}" was an average break for me. {mixed} I am undecided about repeating it.',
        'I felt fairly neutral about "{name}". {mixed} It was reasonable but not memorable.',
        '"{name}" was fine for today. {mixed} I would consider it again without making it a priority.',
        'I can see some value in "{name}". {mixed} My overall impression was moderate.',
        '"{name}" had both useful and less engaging moments. {mixed} I would need another try to decide.',
        'My experience with "{name}" was ordinary. {mixed} It was a possible option rather than a favorite.',
    ],
    4: [
        'I enjoyed "{name}". {positive} It was a worthwhile break from coursework.',
        '"{name}" was helpful and pleasant. {positive} I would happily choose it again.',
        'I had a good experience with "{name}". {positive} It fitted well into my day.',
        '"{name}" was a satisfying choice. {positive} I would like to keep it in my routine.',
        'I was pleased with "{name}". {positive} The time felt well spent.',
        '"{name}" worked well for me today. {positive} I would recommend trying it during a study break.',
        'I found "{name}" very enjoyable. {positive} It was a positive part of my afternoon.',
        '"{name}" gave me a welcome pause. {positive} I would be glad to repeat the experience.',
    ],
    5: [
        'I loved "{name}". {positive} It was one of the most helpful breaks in my week.',
        '"{name}" was excellent. {positive} I would gladly make it a regular habit.',
        'I was delighted with "{name}". {positive} It suited my preferences extremely well.',
        '"{name}" was a wonderful choice today. {positive} I am enthusiastic about doing it again.',
        'I found "{name}" exceptionally rewarding. {positive} It has become a favorite option.',
        '"{name}" exceeded my expectations. {positive} I would eagerly choose it again.',
        'I thoroughly enjoyed "{name}". {positive} It was a highlight of my day.',
        '"{name}" felt like an ideal break for me. {positive} I was extremely satisfied with the experience.',
    ],
}


def build_dataset() -> tuple[list[dict], list[dict]]:
    """Construct balanced exposure and preference-based ratings with a local RNG."""
    rng = random.Random(SEED)
    items = [dict(zip(ITEM_FIELDS, (f"I{index:03d}", *spec), strict=True))
             for index, spec in enumerate(ITEM_SPECS, start=1)]
    category_items = {
        category: [item for item in items if item["category"] == category]
        for category in CATEGORIES
    }
    for candidates in category_items.values():
        rng.shuffle(candidates)
    offsets = {category: rng.randrange(5) for category in CATEGORIES}
    template_pools: dict[tuple[str, int], list[str]] = {}
    ratings = []
    for user_index in range(15):
        user_id = f"U{user_index + 1:03d}"
        _, preferences = PROFILES[user_index // 3]
        time_budget = (10, 15, 20)[user_index % 3]
        for category_index, category in enumerate(CATEGORIES):
            candidates = category_items[category]
            # Each of five cyclic positions is visited three times by 15 users.
            # Selecting two positions gives each activity exactly six ratings.
            position = (user_index + offsets[category]) % 5
            selected = [candidates[position], candidates[(position + 2) % 5]]
            for item in selected:
                noise = rng.choice((-1, 0, 0, 0, 1))
                penalty = int(item["duration_min"] > time_budget)
                score = max(1, min(5, preferences[category_index] + noise - penalty))
                key = (item["item_id"], score)
                if key not in template_pools:
                    template_pools[key] = rng.sample(FEEDBACK_TEMPLATES[score], 8)
                positive, negative, mixed = CONTEXT[category]
                feedback = template_pools[key].pop().format(
                    name=item["habit_name"], positive=positive, negative=negative, mixed=mixed,
                )
                ratings.append(dict(zip(
                    RATING_FIELDS, (user_id, item["item_id"], score, feedback), strict=True,
                )))
    ratings.sort(key=lambda row: (row["user_id"], row["item_id"]))
    return items, ratings


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    """Write UTF-8 CSV with stable fields and platform-independent line endings."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == fields, f"Unexpected schema in {path.name}.")
        rows = list(reader)
    for line, row in enumerate(rows, start=2):
        require(set(row) == set(fields), f"Unexpected fields in {path.name}, line {line}.")
        require(all(isinstance(row[field], str) and row[field].strip() for field in fields),
                f"Missing required value in {path.name}, line {line}.")
        require(all(row[field].isascii() for field in fields),
                f"Expected English ASCII content in {path.name}, line {line}.")
    return rows


def validate_dataset() -> None:
    """Read the saved CSV files once and validate structure, coverage, and signals."""
    items = read_csv(DATA_DIR / "items.csv", ITEM_FIELDS)
    ratings = read_csv(DATA_DIR / "ratings.csv", RATING_FIELDS)
    require(len(items) == 40, "Expected exactly 40 activities.")
    require(len(ratings) == 240, "Expected exactly 240 ratings.")
    item_ids = {row["item_id"] for row in items}
    require(item_ids == {f"I{i:03d}" for i in range(1, 41)}, "Invalid or duplicate item IDs.")
    require(len({row["habit_name"].casefold() for row in items}) == 40, "Duplicate activity names.")
    require(Counter(row["category"] for row in items) == Counter(dict.fromkeys(CATEGORIES, 5)),
            "Expected five activities in each category.")
    for row in items:
        require(row["duration_min"].isdigit() and 1 <= int(row["duration_min"]) <= 60,
                f"Invalid duration for {row['item_id']}.")
        tags = row["mood_goal_tags"].split(";")
        require(len(tags) == len(set(tags)) and all(
            tag and tag == tag.lower() and tag.replace("_", "").isalpha() for tag in tags
        ), f"Invalid mood tags for {row['item_id']}.")
    users = {row["user_id"] for row in ratings}
    require(users == {f"U{i:03d}" for i in range(1, 16)}, "Expected 15 valid user IDs.")
    pairs = {(row["user_id"], row["item_id"]) for row in ratings}
    require(len(pairs) == len(ratings), "Duplicate user-item ratings.")
    require(len({tuple(row[field] for field in RATING_FIELDS) for row in ratings}) == 240,
            "Duplicate rating rows.")
    require(len({row["journal_feedback"] for row in ratings}) == 240, "Duplicate journal feedback.")
    require(all(row["item_id"] in item_ids for row in ratings), "Unknown item reference.")
    require(all(row["rating"] in {"1", "2", "3", "4", "5"} for row in ratings),
            "Ratings must be integers from 1 to 5.")
    require(set(Counter(row["user_id"] for row in ratings).values()) == {16},
            "Each user must rate 16 different activities.")
    item_counts = Counter(row["item_id"] for row in ratings)
    require(set(item_counts) == item_ids and set(item_counts.values()) == {6},
            "Each activity must have six ratings.")
    sparsity = 1 - Fraction(len(pairs), len(users) * len(item_ids))
    require(sparsity == Fraction(3, 5), "The user-item matrix must be exactly 60% missing.")
    require(set(Counter(row["rating"] for row in ratings)) == {"1", "2", "3", "4", "5"},
            "All five rating levels must be represented.")
    item_lookup = {row["item_id"]: row for row in items}
    group_scores: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in ratings:
        item = item_lookup[row["item_id"]]
        require(f'"{item["habit_name"]}"' in row["journal_feedback"], "Feedback references the wrong activity.")
        score = int(row["rating"])
        positive, negative, mixed = CONTEXT[item["category"]]
        expected = positive if score >= 4 else negative if score <= 2 else mixed
        require(expected in row["journal_feedback"], "Feedback does not match rating sentiment.")
        group_index = (int(row["user_id"][1:]) - 1) // 3
        group_scores[group_index][item["category"]].append(score)
    for group_index, (name, preferences) in enumerate(PROFILES):
        scores = group_scores[group_index]
        preferred = [score for category, base in zip(CATEGORIES, preferences, strict=True)
                     if base == 5 for score in scores[category]]
        less_preferred = [score for category, base in zip(CATEGORIES, preferences, strict=True)
                          if base == 2 for score in scores[category]]
        require(sum(preferred) / len(preferred) > sum(less_preferred) / len(less_preferred),
                f"Preference signal is missing for {name}.")
    print("Dataset validation passed.")
    print(f"Seed: {SEED}")
    print("Activities: 40 across 8 categories")
    print("Users: 15 across 5 preference groups")
    print("Ratings: 240; 16 distinct activities per user; 6 ratings per activity")
    print("User-item matrix: 600 cells; 240 observed; 360 missing; sparsity 60.00%")
    print("Duplicates: 0; missing required values: 0; invalid ratings: 0")
    print("Rating distribution: " + ", ".join(
        f"{score}={sum(row['rating'] == str(score) for row in ratings)}" for score in range(1, 6)
    ))
    print("Feedback alignment and preference signals: passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true", help="Explicitly replace existing generated CSV files.")
    mode.add_argument("--validate-only", action="store_true", help="Validate saved CSV files without writing.")
    args = parser.parse_args()
    if not args.validate_only:
        targets = [DATA_DIR / "items.csv", DATA_DIR / "ratings.csv"]
        if any(path.exists() for path in targets) and not args.overwrite:
            parser.error("Dataset files already exist; use --overwrite to regenerate them.")
        items, ratings = build_dataset()
        DATA_DIR.mkdir(exist_ok=True)
        write_csv(targets[0], ITEM_FIELDS, items)
        write_csv(targets[1], RATING_FIELDS, ratings)
        print("Created data/items.csv and data/ratings.csv (synthetic data).")
    validate_dataset()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        sys.exit(1)
