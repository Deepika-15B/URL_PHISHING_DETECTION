"""IEEE phishing-detection feature selection using Random Forest importance.

This module consumes the completed preprocessing output unchanged.  It ranks
all available feature columns with permutation importance and persists the
Top 20 and Top 14 names for later model-training stages.

Run from the project root:
    python utils/feature_selection.py
"""

from __future__ import annotations

import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

# Use a non-interactive backend so the chart is created reliably in scripts/CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


# Resolve paths from this file, so execution works from any current directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATASET_PATH = _PROJECT_ROOT / "data" / "dataset_small_preprocessed.csv"
_MODELS_DIR = _PROJECT_ROOT / "models"
_REPORTS_DIR = _PROJECT_ROOT / "reports"
_TARGET_COLUMN = "phishing"

_TOP20_PATH = _MODELS_DIR / "top20_features.pkl"
_TOP14_PATH = _MODELS_DIR / "top14_features.pkl"
_RANKING_PATH = _REPORTS_DIR / "feature_ranking.csv"
_FIGURE_PATH = _REPORTS_DIR / "feature_importance.png"
_REPORT_PATH = _REPORTS_DIR / "feature_selection_report.txt"

# The requested reproducible settings.  No feature names are encoded here.
_RANDOM_STATE = 42
_PERMUTATION_REPEATS = 10
_TOP20_COUNT = 20
_TOP14_COUNT = 14


def load_preprocessed_data(path: Path = _DATASET_PATH) -> pd.DataFrame:
    """Load the completed preprocessing output without altering its values.

    The target is deliberately retained in the returned frame.  It is removed
    only when the feature matrix is separated in ``select_features``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found: {path}")

    dataframe = pd.read_csv(path)
    if _TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Required target column '{_TARGET_COLUMN}' is missing.")
    if dataframe.empty:
        raise ValueError("The preprocessed dataset contains no rows.")

    print(f"Loaded {path.name}: {dataframe.shape[0]:,} rows x {dataframe.shape[1]} columns")
    return dataframe


def train_random_forest(
    features: pd.DataFrame,
    target: pd.Series,
    random_state: int = _RANDOM_STATE,
) -> RandomForestClassifier:
    """Fit the IEEE feature-ranking Random Forest on the supplied features.

    No scaling is performed: the input is already the final preprocessed,
    scaled dataset.  The classifier uses sklearn defaults except for the
    required fixed random state, preserving reproducibility.
    """
    classifier = RandomForestClassifier(random_state=random_state)
    classifier.fit(features, target)
    return classifier


def compute_permutation_importance(
    model: RandomForestClassifier,
    features: pd.DataFrame,
    target: pd.Series,
    n_repeats: int = _PERMUTATION_REPEATS,
    random_state: int = _RANDOM_STATE,
) -> pd.DataFrame:
    """Measure the score drop caused by permuting each feature independently.

    Higher mean importance means disrupting that feature harms Random Forest
    classification more, making it more useful for phishing detection.
    """
    result = permutation_importance(
        estimator=model,
        X=features,
        y=target,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,  # Parallel scoring; it does not alter the methodology or seeds.
    )

    return pd.DataFrame(
        {
            "Feature": features.columns,
            "Importance": result.importances_mean,
            "ImportanceStd": result.importances_std,
        }
    )


def rank_features(importance_table: pd.DataFrame) -> pd.DataFrame:
    """Sort all feature importances descending and add one-based rank values."""
    ranking = importance_table.sort_values(
        by=["Importance", "Feature"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    ranking.insert(0, "Rank", ranking.index + 1)
    return ranking


def save_selected_features(
    top20_features: list[str],
    top14_features: list[str],
    top20_path: Path = _TOP20_PATH,
    top14_path: Path = _TOP14_PATH,
) -> None:
    """Persist ordered selections so later training uses the exact same schema."""
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with top20_path.open("wb") as file:
        pickle.dump(top20_features, file, protocol=pickle.HIGHEST_PROTOCOL)
    with top14_path.open("wb") as file:
        pickle.dump(top14_features, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved Top 20 feature names: {top20_path}")
    print(f"Saved Top 14 feature names: {top14_path}")


def _generate_feature_plot(ranking: pd.DataFrame, output_path: Path = _FIGURE_PATH) -> None:
    """Create the required horizontal bar chart for the Top 20 ranked features."""
    top20 = ranking.head(min(_TOP20_COUNT, len(ranking))).sort_values("Importance")
    figure, axis = plt.subplots(figsize=(11, 8))
    axis.barh(top20["Feature"], top20["Importance"], color="#2f6f9f")
    axis.set_title("Top 20 Features by Permutation Importance")
    axis.set_xlabel("Mean decrease in Random Forest accuracy")
    axis.set_ylabel("Feature")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def generate_report(
    ranking: pd.DataFrame,
    top20_features: list[str],
    top14_features: list[str],
    execution_time_seconds: float,
    random_forest_parameters: dict[str, Any],
    report_path: Path = _REPORT_PATH,
) -> None:
    """Write the auditable text report and CSV/chart ranking artifacts."""
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # The CSV keeps exactly the columns requested for downstream review.
    ranking.loc[:, ["Rank", "Feature", "Importance"]].to_csv(_RANKING_PATH, index=False)
    _generate_feature_plot(ranking)

    original_count = len(ranking)
    selected_count = len(top14_features)
    reduction_percentage = (1 - selected_count / original_count) * 100
    stats = ranking["Importance"].describe()

    lines = [
        "=" * 72,
        "IEEE PHISHING DETECTION PROJECT - FEATURE SELECTION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 72,
        "",
        "INPUT AND SELECTION SUMMARY",
        "-" * 72,
        f"Original feature count : {original_count}",
        f"Selected feature count : {selected_count}",
        f"Reduction percentage   : {reduction_percentage:.2f}%",
        f"Execution time         : {execution_time_seconds:.2f} seconds",
        "",
        "RANDOM FOREST PARAMETERS",
        "-" * 72,
        *[f"{name}: {value}" for name, value in sorted(random_forest_parameters.items())],
        "",
        "PERMUTATION IMPORTANCE PARAMETERS",
        "-" * 72,
        f"n_repeats: {_PERMUTATION_REPEATS}",
        f"random_state: {_RANDOM_STATE}",
        "",
        "FEATURE IMPORTANCE STATISTICS",
        "-" * 72,
        f"Maximum importance: {stats['max']:.8f}",
        f"Mean importance   : {stats['mean']:.8f}",
        f"Median importance : {stats['50%']:.8f}",
        f"Minimum importance: {stats['min']:.8f}",
        f"Std. deviation    : {stats['std']:.8f}",
        "",
        "TOP 20 FEATURES",
        "-" * 72,
    ]
    lines.extend(
        f"{int(row.Rank):>2}. {row.Feature}: {row.Importance:.8f}"
        for row in ranking.head(len(top20_features)).itertuples(index=False)
    )
    lines.extend(["", "TOP 14 SELECTED FEATURES", "-" * 72])
    lines.extend(f"{position:>2}. {feature}" for position, feature in enumerate(top14_features, 1))
    lines.extend(
        [
            "",
            "WHY THESE FEATURES MATTER FOR PHISHING DETECTION",
            "-" * 72,
            "Permutation importance selects features whose disruption most reduces",
            "Random Forest classification accuracy. The selected features therefore",
            "provide the strongest observed signal for separating phishing from",
            "legitimate websites in this IEEE preprocessing output. Reducing the",
            "training matrix to these high-impact variables removes weaker or",
            "redundant signals while retaining the features most useful to the",
            "classifier. No feature names are manually prescribed; the ranking is",
            "derived entirely from the current preprocessed dataset.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved complete ranking: {_RANKING_PATH}")
    print(f"Saved Top 20 importance chart: {_FIGURE_PATH}")
    print(f"Saved feature selection report: {report_path}")


def select_features() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Run the complete, reusable Random Forest feature-selection pipeline."""
    start_time = time.perf_counter()

    # Load once, then split strictly by the target-column name; this adapts to
    # any future preprocessing output feature count without hardcoded names.
    dataframe = load_preprocessed_data()
    features = dataframe.drop(columns=[_TARGET_COLUMN])
    target = dataframe[_TARGET_COLUMN]
    if features.empty:
        raise ValueError("No feature columns remain after removing the target.")

    print(f"Training Random Forest on {features.shape[1]} processed features...")
    model = train_random_forest(features, target)

    print(f"Computing permutation importance ({_PERMUTATION_REPEATS} repeats)...")
    importance_table = compute_permutation_importance(model, features, target)
    ranking = rank_features(importance_table)

    # Both selections are computed from the ranking so no feature identity is
    # hardcoded. min() keeps the module valid for any future smaller dataset.
    top20_features = ranking.head(min(_TOP20_COUNT, len(ranking)))["Feature"].tolist()
    top14_features = ranking.head(min(_TOP14_COUNT, len(ranking)))["Feature"].tolist()
    save_selected_features(top20_features, top14_features)

    elapsed_seconds = time.perf_counter() - start_time
    generate_report(
        ranking=ranking,
        top20_features=top20_features,
        top14_features=top14_features,
        execution_time_seconds=elapsed_seconds,
        random_forest_parameters=model.get_params(),
    )

    print("\nFeature selection complete.")
    print("Top 14 features selected from the current preprocessing output:")
    for rank, feature in enumerate(top14_features, 1):
        print(f"  {rank:>2}. {feature}")
    print(
        "\nThese features are important because permuting them causes the greatest "
        "loss of Random Forest accuracy, indicating the strongest phishing-versus-"
        "legitimate discrimination in the IEEE dataset."
    )

    return ranking, top20_features, top14_features


if __name__ == "__main__":
    select_features()

