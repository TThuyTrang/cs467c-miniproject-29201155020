# Wellness Habit and Mindfulness Activity Recommender

An individual project for **CS467C - Advanced Artificial Intelligence**. The planned system will recommend student-oriented wellness habits and mindfulness activities using activity descriptions, explicit ratings, and journal feedback. Later stages may explore NLP, TF-IDF, sentiment analysis, and collaborative filtering.

The current deliverables are the development environment and a **fully synthetic dataset**. No NLP pipeline, recommendation engine, project model training, or website has been implemented. All project artifacts, code comments, documentation, analyses, and demo outputs must be written in English.

## Development environment

Use **Python 3.11, 64-bit** on Windows. The existing environment uses Python **3.11.9**. Exact versions of installed packages, dependencies, and environment tools are recorded in `requirements.txt`. Core packages are NumPy, pandas, scikit-learn, SciPy, Matplotlib, NLTK, Jupyter, and joblib. PyTorch, TensorFlow, and CUDA are not required.

Activate the existing environment in Windows PowerShell:

```powershell
Set-Location 'D:\01_Projects\cs467-miniproject-29201155020'
.\.venv\Scripts\Activate.ps1
```

Activation is optional: the commands below call the environment's Python directly. If PowerShell blocks activation, either use those direct commands or allow the activation script for the current session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

For a fresh checkout only, first ensure that Python 3.11 64-bit is installed. Create `.venv` only when it does not already exist, and restore the pinned packages:

```powershell
py -3.11 -m venv .venv
New-Item -ItemType Directory -Force .tmp | Out-Null
$env:TEMP = Join-Path (Get-Location) '.tmp'
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m pip --isolated --cache-dir .cache\pip install -r requirements.txt
```

The existing `check_environment.py` checks interpreter selection, imports, and temporary TF-IDF/joblib serialization. It is separate from dataset validation and is not needed to regenerate the data. No additional packages or NLTK language resources are needed for this dataset stage.

## Synthetic dataset

Both CSV files use UTF-8 encoding, a header row, comma delimiters, and English text. They contain no real participants, journal records, credentials, or scraped material. Activity descriptions were authored specifically for this project; ratings and journals simulate fictional student experiences.

### Activity catalogue: `data/items.csv`

The catalogue contains **40 unique activities**, with five activities in each of eight categories: `movement`, `mindfulness`, `journaling`, `study_recovery`, `sleep_routine`, `nutrition`, `social_connection`, and `creative_relaxation`. Durations range from 5 to 20 minutes. Activities consider student schedules, limited budgets, shared spaces, and low-pressure participation.

| Column | Meaning |
| --- | --- |
| `item_id` | Stable activity identifier, `I001` through `I040`. |
| `habit_name` | Unique, readable English activity name. |
| `category` | One of the eight category identifiers above. |
| `duration_min` | Positive integer indicating the planned duration in minutes. |
| `description` | An English description suitable for future text-based experiments. |
| `mood_goal_tags` | Semicolon-separated lowercase tags, such as `calm;awareness;stress_relief`. Tags describe intended themes, not demonstrated health outcomes. |

### Interaction history: `data/ratings.csv`

The interaction history contains **240 ratings** from **15 fictional users**. Each user rates **16 distinct activities**, and each activity receives **six ratings**. There are 600 possible user-item pairs, of which 240 are observed and 360 are unobserved: **40% density and exactly 60% sparsity**. An unobserved pair is not a negative rating.

| Column | Meaning |
| --- | --- |
| `user_id` | Fictional identifier, `U001` through `U015`; no demographic identity is implied. |
| `item_id` | Foreign key referencing `items.csv`. |
| `rating` | Integer from 1 (strong dislike) to 5 (strong enjoyment). |
| `journal_feedback` | Fictional English feedback consistent with the assigned rating and activity category. |

Ratings of 1-2 have clearly negative feedback, 3 has mixed or neutral feedback, and 4-5 have clearly positive feedback. These are generation rules, not independently annotated sentiment ground truth. Feedback is assembled from eight templates per rating and category-specific context; all 240 full feedback strings are distinct.

### Fictional preference groups

Three users belong to each group. Membership is defined in the generator rather than added as an extra CSV column.

| User IDs | Group | Strongest preferences | Less preferred categories |
| --- | --- | --- | --- |
| U001-U003 | Active students | Movement | Journaling, creative relaxation |
| U004-U006 | Quiet reflectors | Mindfulness, journaling | Movement, social connection |
| U007-U009 | Focused learners | Study recovery | Social connection, creative relaxation |
| U010-U012 | Social explorers | Social connection | Mindfulness, journaling |
| U013-U015 | Routine builders | Sleep routines, nutrition | Social connection, creative relaxation |

Each group has a category preference score vector. Within each group, users have different time budgets of 10, 15, and 20 minutes. A duration above the user's budget reduces the rating by one point. Seeded noise drawn from `[-1, 0, 0, 0, 1]` introduces variation before ratings are clipped to the range 1-5. The generator does not fit or learn these preferences.

Each user encounters two activities per category. A seeded shuffle and cyclic assignment provide overlapping histories while ensuring exactly six ratings per activity. This deliberately balanced exposure helps early collaborative filtering experiments compare preferences across shared items.

## Generate and validate the data

The generator uses only the Python standard library and a local `random.Random` instance with fixed seed **4672026**. Activity definitions, category ordering, templates, output ordering, and CSV line endings are fixed in the source. Running the same source with Python 3.11 reproduces the dataset without accessing the network or training a model.

When the CSV files do not yet exist:

```powershell
.\.venv\Scripts\python.exe .\src\generate_dataset.py
```

The script writes both CSV files and then performs one validation pass over the saved data. Existing files are preserved unless replacement is requested explicitly:

```powershell
.\.venv\Scripts\python.exe .\src\generate_dataset.py --overwrite
```

To validate existing files without generating or changing them:

```powershell
.\.venv\Scripts\python.exe .\src\generate_dataset.py --validate-only
```

Validation checks exact schemas, required values, unique activity IDs and names, valid durations and tags, valid foreign keys, unique user-item pairs and journal strings, all five rating levels, per-user and per-item counts, exactly 60% sparsity, feedback-context alignment, and higher average ratings for each group's strongest preferences than for its least preferred categories. Console output reports counts and the rating distribution. Structural checks do not constitute an independent assessment of linguistic quality or real-world validity.

## Limitations and responsible interpretation

- This is a small, fictional educational dataset. It does not establish recommendation accuracy, student preferences, wellness benefits, or clinical effectiveness in a real population.
- Category preferences, balanced exposure, limited time budgets, and rating noise are hand-designed assumptions. Six ratings per item and two exposures per category per user are more regular than typical real-world interactions.
- Template-based feedback has repetitive language and unusually clear alignment with ratings. Sentiment or recommendation metrics obtained from it may be optimistic and should not be presented as evidence of generalization to real journals.
- Feedback is generated after its rating. Using that same feedback to predict its own rating would introduce target leakage; it should not be treated as information available before an activity is experienced.
- The files contain no timestamps, longitudinal habit adherence, real user diversity, or independent sentiment labels. Future evaluation needs an appropriate user-item split and, where feasible and consented, independent evidence.
- Descriptions and fictional positive or negative experiences are activity ideas, not medical advice or measured health outcomes. The dataset contains no real personal or sensitive information.

## Project structure

```text
cs467-miniproject-29201155020/
|-- .git/                       Local Git repository
|-- .venv/                      Existing Python 3.11 environment; ignored by Git
|-- .cache/                     Local caches; ignored by Git
|-- .tmp/                       Temporary files; ignored by Git
|-- data/
|   |-- .gitkeep
|   |-- items.csv               40 synthetic activities
|   `-- ratings.csv             240 synthetic ratings and journals
|-- src/
|   |-- .gitkeep
|   `-- generate_dataset.py     Deterministic generation and dataset validation
|-- notebooks/.gitkeep          Reserved for future experiments
|-- results/.gitkeep            Reserved for future evaluation outputs
|-- models/.gitkeep             Reserved for genuinely fitted artifacts
|-- .gitignore
|-- check_environment.py        Existing environment smoke check
|-- requirements.txt            Pinned environment packages
`-- README.md
```

The `data/`, `results/`, and `models/` directories are intentionally eligible for version control. Only genuinely fitted models or vectorizers should later be saved in `models/`; no project model is stored there now. Review files for sensitive information before staging them, because `.gitignore` cannot detect all confidential content. The repository is kept local; this stage does not push to GitHub.
