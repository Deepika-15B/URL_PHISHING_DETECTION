"""
utils/data_analysis.py
======================
Dataset Analysis Module — IEEE Phishing Detection Project
----------------------------------------------------------
Performs comprehensive exploratory data analysis (EDA) on the two phishing-
detection datasets (dataset_small.csv and dataset_full.csv) as described in
the accompanying IEEE research paper.

This module is intentionally analysis-only:
  * It does NOT preprocess, modify, or impute the raw data.
  * It does NOT perform feature selection or engineering.
  * It does NOT implement or train any machine-learning model.
  * It does NOT start a web server or interact with Flask.

Outputs
-------
  reports/dataset_analysis.txt   — Full text report
  reports/class_distribution.png — Bar chart of label counts per dataset
  reports/missing_values.png     — Heatmap of missing / NaN coverage
  reports/correlation_heatmap.png— Pearson-correlation heatmap (full dataset)
  reports/constant_columns.csv   — Columns with zero variance
  reports/sentinel_columns.csv   — Columns containing sentinel value −1

Usage
-----
    python utils/data_analysis.py

    # Or import and call individual functions:
    from utils.data_analysis import analyze_dataset, compare_datasets

Author  : Phishing Detection IEEE Team
Version : 1.0.0
"""

# ─── Standard library ──────────────────────────────────────────────────────────
import os
import sys
import textwrap
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

# ─── Third-party ───────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")                          # Non-interactive backend (no GUI needed)

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ─── Resolve project root so the script can be run from any working directory ──
# Force UTF-8 output on Windows so box/line characters print correctly
import io as _io
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_HERE        = Path(__file__).resolve().parent          # .../utils/
_PROJECT_ROOT = _HERE.parent                            # .../phishing_detection_ieee/
_DATA_DIR    = _PROJECT_ROOT / "data"
_REPORTS_DIR = _PROJECT_ROOT / "reports"
_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Dataset paths ─────────────────────────────────────────────────────────────
SMALL_CSV = _DATA_DIR / "dataset_small.csv"
FULL_CSV  = _DATA_DIR / "dataset_full.csv"

# ─── Target column (binary label: 1 = phishing, 0 = legitimate) ───────────────
TARGET_COL   = "phishing"
SENTINEL_VAL = -1          # Value used by the dataset authors to flag "no data"

# ─── Plot styling ──────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
PALETTE = {"small": "#4C72B0", "full": "#DD8452"}   # Blue / Orange


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SINGLE-DATASET ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_dataset(df: pd.DataFrame, name: str) -> dict[str, Any]:
    """
    Perform a comprehensive analysis of a single dataset.

    Parameters
    ----------
    df   : pd.DataFrame
        The loaded dataset (unchanged — read-only analysis).
    name : str
        Human-readable label used in report headings (e.g. "Small Dataset").

    Returns
    -------
    dict
        A structured dictionary containing every metric computed.
        Keys used by compare_datasets(), generate_visualizations(), and
        save_report() are documented inside the function body.
    """

    sep = "-" * 72                          # Section separator for the text report
    results: dict[str, Any] = {"name": name}

    # ── 1. Shape ────────────────────────────────────────────────────────────────
    results["n_rows"]    = df.shape[0]
    results["n_cols"]    = df.shape[1]

    # ── 2. Column names ─────────────────────────────────────────────────────────
    results["columns"]   = df.columns.tolist()

    # ── 3. Data types — count per dtype family ──────────────────────────────────
    results["dtypes"]    = df.dtypes                    # Full Series (col → dtype)
    results["dtype_counts"] = df.dtypes.value_counts()  # e.g. int64: 111, float64: 1

    # ── 4. Missing values ───────────────────────────────────────────────────────
    # The datasets have no NaN values; missing information is encoded as -1.
    missing_per_col              = df.isnull().sum()
    results["missing_per_col"]   = missing_per_col
    results["total_missing"]     = int(missing_per_col.sum())
    results["cols_with_missing"] = missing_per_col[missing_per_col > 0].index.tolist()

    # ── 5. Duplicate rows ───────────────────────────────────────────────────────
    results["n_duplicates"] = int(df.duplicated().sum())

    # ── 6 & 7 & 8. Class distribution and percentages ──────────────────────────
    class_counts = df[TARGET_COL].value_counts().sort_index()
    class_pct    = (class_counts / len(df) * 100).round(2)

    results["class_counts"]      = class_counts        # {0: N_legit, 1: N_phish}
    results["class_pct"]         = class_pct           # percentage series
    results["n_phishing"]        = int(class_counts.get(1, 0))
    results["n_legitimate"]      = int(class_counts.get(0, 0))
    results["pct_phishing"]      = float(class_pct.get(1, 0.0))
    results["pct_legitimate"]    = float(class_pct.get(0, 0.0))

    # Imbalance ratio (minority : majority)
    minority = min(results["n_phishing"], results["n_legitimate"])
    majority = max(results["n_phishing"], results["n_legitimate"])
    results["imbalance_ratio"]   = round(minority / majority, 4) if majority else 0.0

    # ── 9. Memory usage ─────────────────────────────────────────────────────────
    mem_bytes                = df.memory_usage(deep=True).sum()
    results["memory_bytes"]  = int(mem_bytes)
    results["memory_mb"]     = round(mem_bytes / 1_048_576, 3)

    # ── 10. Basic statistics (pandas describe) ──────────────────────────────────
    results["describe"] = df.describe(percentiles=[0.25, 0.50, 0.75])

    # ── 12. Constant columns (zero variance) ────────────────────────────────────
    constant_mask              = df.nunique() == 1
    results["constant_cols"]   = df.columns[constant_mask].tolist()
    results["n_constant_cols"] = len(results["constant_cols"])

    # Collect constant value for each constant column (for the CSV report)
    results["constant_col_values"] = {
        col: df[col].iloc[0] for col in results["constant_cols"]
    }

    # ── 13 & 14. Sentinel value (−1) detection ──────────────────────────────────
    sentinel_counts_all = {
        col: int((df[col] == SENTINEL_VAL).sum())
        for col in df.select_dtypes(include=[np.number]).columns
        if (df[col] == SENTINEL_VAL).any()
    }
    results["sentinel_cols"]       = list(sentinel_counts_all.keys())
    results["sentinel_counts"]     = sentinel_counts_all          # col → row count
    results["n_sentinel_cols"]     = len(sentinel_counts_all)

    # Sentinel coverage (%) per affected column
    results["sentinel_pct"] = {
        col: round(cnt / len(df) * 100, 2)
        for col, cnt in sentinel_counts_all.items()
    }

    # ── 15. Preprocessing recommendations ──────────────────────────────────────
    recommendations = _generate_recommendations(results)
    results["recommendations"] = recommendations

    return results


def _generate_recommendations(r: dict[str, Any]) -> list[str]:
    """
    Derive a prioritised list of preprocessing recommendations from the
    analysis metrics.  Called internally by analyze_dataset().

    Parameters
    ----------
    r : dict
        The results dict produced by analyze_dataset() (partially populated).

    Returns
    -------
    list[str]
        Ordered list of human-readable recommendation strings.
    """
    recs: list[str] = []

    # Sentinel value handling
    if r["n_sentinel_cols"] > 0:
        # Group columns by their sentinel-row count to suggest strategies
        high_sentinel = [
            col for col, cnt in r["sentinel_counts"].items()
            if cnt / r["n_rows"] > 0.50
        ]
        low_sentinel  = [
            col for col, cnt in r["sentinel_counts"].items()
            if cnt / r["n_rows"] <= 0.50
        ]
        if high_sentinel:
            recs.append(
                f"[CRITICAL] {len(high_sentinel)} column(s) have >50 % sentinel "
                f"(-1) coverage (e.g. params-level and directory-level features). "
                f"Consider: (a) replacing -1 with NaN and imputing, or (b) creating "
                f"a binary 'has_<segment>' indicator column and zeroing the original."
            )
        if low_sentinel:
            recs.append(
                f"[HIGH] {len(low_sentinel)} column(s) have ≤50 % sentinel "
                f"(-1) coverage (e.g. time_response, asn_ip, ttl_hostname). "
                f"Strategy: replace -1 with NaN → median imputation, then add "
                f"a boolean missingness-indicator feature."
            )

    # Constant / zero-variance columns
    if r["n_constant_cols"] > 0:
        recs.append(
            f"[HIGH] {r['n_constant_cols']} constant column(s) detected "
            f"(all values identical). Drop these columns before training — "
            f"they carry zero predictive information and waste memory."
        )

    # Duplicate rows
    if r["n_duplicates"] > 0:
        recs.append(
            f"[MEDIUM] {r['n_duplicates']:,} duplicate row(s) found. "
            f"Remove duplicates before splitting into train/test sets to avoid "
            f"data leakage between folds."
        )

    # Class imbalance
    pct_min = min(r["pct_phishing"], r["pct_legitimate"])
    pct_max = max(r["pct_phishing"], r["pct_legitimate"])
    if pct_min < 40.0:
        recs.append(
            f"[MEDIUM] Class imbalance detected (minority class: {pct_min:.1f} %). "
            f"Consider SMOTE oversampling or class-weight adjustment during model "
            f"training.  Evaluation should prioritise F1-score and AUC-ROC, not "
            f"raw accuracy."
        )
    else:
        recs.append(
            f"[INFO] Class balance is acceptable "
            f"(phishing: {r['pct_phishing']:.1f} % / legitimate: "
            f"{r['pct_legitimate']:.1f} %).  Standard stratified k-fold "
            f"cross-validation is sufficient."
        )

    # Missing NaN values
    if r["total_missing"] > 0:
        recs.append(
            f"[MEDIUM] {r['total_missing']:,} genuine NaN values detected. "
            f"Impute with median (numeric) or most-frequent (categorical) strategy."
        )
    else:
        recs.append(
            "[INFO] No genuine NaN values found.  All missing information "
            "is encoded using the sentinel value -1 (see above)."
        )

    # Feature scaling recommendation — MinMaxScaler is mandated by the IEEE paper
    # for normalising numerical features before feeding them into deep learning models.
    recs.append(
        "[INFO] All features are numeric (int64). Apply MinMaxScaler "
        "normalization (scales each feature to the [0, 1] range) as specified "
        "by the IEEE paper before training deep learning models.  "
        "Tree-based ensembles (Random Forest, XGBoost, LightGBM) do not "
        "require scaling and may be trained on the raw feature values."
    )

    return recs


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CROSS-DATASET COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def compare_datasets(r_small: dict[str, Any], r_full: dict[str, Any]) -> dict[str, Any]:
    """
    Compare two analyzed datasets and identify structural differences.

    Parameters
    ----------
    r_small : dict
        Results dict returned by analyze_dataset() for dataset_small.csv.
    r_full : dict
        Results dict returned by analyze_dataset() for dataset_full.csv.

    Returns
    -------
    dict
        Comparison metrics including schema equality, column set differences,
        and per-metric deltas.
    """
    cmp: dict[str, Any] = {}

    # ── Schema check: identical column sets and order? ──────────────────────────
    cols_s = r_small["columns"]
    cols_f = r_full["columns"]

    cmp["schemas_identical"]    = (cols_s == cols_f)
    cmp["same_column_set"]      = (set(cols_s) == set(cols_f))
    cmp["same_column_order"]    = (cols_s == cols_f)

    # Columns present in one dataset but not the other
    cmp["only_in_small"] = list(set(cols_s) - set(cols_f))
    cmp["only_in_full"]  = list(set(cols_f) - set(cols_s))

    # ── Dtype consistency ───────────────────────────────────────────────────────
    shared_cols = list(set(cols_s) & set(cols_f))
    dtype_mismatches = {
        col: {
            "small": str(r_small["dtypes"][col]),
            "full":  str(r_full["dtypes"][col]),
        }
        for col in shared_cols
        if r_small["dtypes"][col] != r_full["dtypes"][col]
    }
    cmp["dtype_mismatches"]   = dtype_mismatches
    cmp["n_dtype_mismatches"] = len(dtype_mismatches)

    # ── Constant column agreement ────────────────────────────────────────────────
    cmp["constant_cols_agree"] = (
        set(r_small["constant_cols"]) == set(r_full["constant_cols"])
    )
    cmp["constant_only_small"] = list(
        set(r_small["constant_cols"]) - set(r_full["constant_cols"])
    )
    cmp["constant_only_full"]  = list(
        set(r_full["constant_cols"]) - set(r_small["constant_cols"])
    )

    # ── Sentinel column agreement ────────────────────────────────────────────────
    cmp["sentinel_cols_agree"] = (
        set(r_small["sentinel_cols"]) == set(r_full["sentinel_cols"])
    )
    cmp["sentinel_only_small"] = list(
        set(r_small["sentinel_cols"]) - set(r_full["sentinel_cols"])
    )
    cmp["sentinel_only_full"]  = list(
        set(r_full["sentinel_cols"]) - set(r_small["sentinel_cols"])
    )

    # ── Row-count delta (how many rows does the full dataset add?) ───────────────
    cmp["row_delta"]     = r_full["n_rows"] - r_small["n_rows"]
    cmp["row_scale"]     = round(r_full["n_rows"] / r_small["n_rows"], 3)

    # ── Class-distribution delta ─────────────────────────────────────────────────
    cmp["phishing_pct_delta"]   = round(
        r_full["pct_phishing"] - r_small["pct_phishing"], 2
    )
    cmp["legitimate_pct_delta"] = round(
        r_full["pct_legitimate"] - r_small["pct_legitimate"], 2
    )

    return cmp


# ══════════════════════════════════════════════════════════════════════════════
# 3.  VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

def generate_visualizations(
    df_small: pd.DataFrame,
    df_full: pd.DataFrame,
    r_small: dict[str, Any],
    r_full: dict[str, Any],
) -> None:
    """
    Produce and save all required analysis charts.

    Charts generated
    ----------------
    1. reports/class_distribution.png  — side-by-side bar chart of label counts
    2. reports/missing_values.png      — heatmap of NaN / sentinel coverage
    3. reports/correlation_heatmap.png — Pearson correlation on the full dataset

    Parameters
    ----------
    df_small, df_full : pd.DataFrame
        Raw (unmodified) dataframes.
    r_small, r_full   : dict
        Analysis result dicts from analyze_dataset().
    """

    # ─── Chart 1: Class Distribution ──────────────────────────────────────────
    _plot_class_distribution(r_small, r_full)

    # ─── Chart 2: Missing / Sentinel Value Coverage ───────────────────────────
    _plot_sentinel_heatmap(df_small, df_full, r_small, r_full)

    # ─── Chart 3: Pearson Correlation Heatmap (full dataset, excl. constants) ──
    _plot_correlation_heatmap(df_full, r_full)


def _plot_class_distribution(
    r_small: dict[str, Any],
    r_full: dict[str, Any],
) -> None:
    """
    Side-by-side grouped bar chart comparing phishing vs legitimate counts
    across both datasets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    fig.suptitle(
        "Class Distribution — Phishing vs Legitimate URLs",
        fontsize=15, fontweight="bold", y=1.01,
    )

    datasets = [
        (r_small, "Small Dataset\n(58,645 URLs)",  axes[0], PALETTE["small"]),
        (r_full,  "Full Dataset\n(88,647 URLs)",   axes[1], PALETTE["full"]),
    ]

    for r, title, ax, color in datasets:
        labels = ["Legitimate (0)", "Phishing (1)"]
        counts = [r["n_legitimate"], r["n_phishing"]]
        colors = [color, "#C44E52"]   # blue/orange for legit, red for phishing

        bars = ax.bar(labels, counts, color=colors, edgecolor="white",
                      linewidth=1.2, width=0.55)

        # Annotate each bar with count and percentage
        for bar, cnt, pct in zip(
            bars,
            counts,
            [r["pct_legitimate"], r["pct_phishing"]],
        ):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{cnt:,}\n({pct:.1f} %)",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

        ax.set_title(title, fontsize=12, fontweight="semibold")
        ax.set_ylabel("Number of Records")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x):,}"
        ))
        ax.set_ylim(0, max(counts) * 1.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = _REPORTS_DIR / "class_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {out}")


def _plot_sentinel_heatmap(
    df_small: pd.DataFrame,
    df_full: pd.DataFrame,
    r_small: dict[str, Any],
    r_full: dict[str, Any],
) -> None:
    """
    Heatmap showing the percentage of rows carrying the sentinel value (-1)
    for each affected column, across both datasets.

    The heatmap uses only the union of sentinel columns identified in both
    datasets — columns with zero sentinel occurrences are excluded to keep
    the chart readable.
    """
    # Collect all sentinel columns (union)
    all_sentinel_cols = sorted(
        set(r_small["sentinel_cols"]) | set(r_full["sentinel_cols"])
    )

    if not all_sentinel_cols:
        print("  [INFO] No sentinel columns found — skipping missing_values.png")
        return

    # Build a 2-row DataFrame: rows = datasets, cols = sentinel columns (pct)
    small_pct = pd.Series(r_small["sentinel_pct"])
    full_pct  = pd.Series(r_full["sentinel_pct"])

    heat_df = pd.DataFrame(
        {
            r_small["name"]: small_pct.reindex(all_sentinel_cols, fill_value=0.0),
            r_full["name"]:  full_pct.reindex(all_sentinel_cols,  fill_value=0.0),
        }
    ).T    # shape: (2 datasets) × (N sentinel columns)

    # Dynamically size the figure so column labels don't overlap
    fig_w = max(20, len(all_sentinel_cols) * 0.38)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))

    sns.heatmap(
        heat_df,
        ax=ax,
        cmap="YlOrRd",
        annot=True,
        fmt=".1f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "% Rows with Sentinel Value (−1)"},
        vmin=0,
        vmax=100,
    )

    ax.set_title(
        "Sentinel Value (−1) Coverage by Column and Dataset (%)",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Feature Column", fontsize=10)
    ax.set_ylabel("Dataset", fontsize=10)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)

    plt.tight_layout()
    out = _REPORTS_DIR / "missing_values.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {out}")


def _plot_correlation_heatmap(
    df_full: pd.DataFrame,
    r_full: dict[str, Any],
) -> None:
    """
    Pearson correlation heatmap for the full dataset.

    Constant columns (zero variance) are excluded because their Pearson
    coefficient is undefined (division by zero std dev).  The target column
    'phishing' is retained so users can visually identify high-correlation
    features.
    """
    # Drop constant columns before computing correlations
    drop_cols  = r_full["constant_cols"]
    df_numeric = df_full.drop(columns=drop_cols, errors="ignore")

    corr = df_numeric.corr(method="pearson")

    # Reorder columns/rows by hierarchical clustering for a cleaner layout
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    # Convert correlation to dissimilarity (0 = identical, 1 = orthogonal)
    dissimilarity = 1 - corr.abs()
    dissimilarity = np.clip(dissimilarity, 0, None)   # guard against float noise

    try:
        linkage_matrix = linkage(
            squareform(dissimilarity, checks=False), method="ward"
        )
        order = leaves_list(linkage_matrix)
        corr  = corr.iloc[order, :].iloc[:, order]
    except Exception:
        pass   # Fall back to original order if clustering fails

    n_cols = corr.shape[0]
    fig_size = max(22, n_cols * 0.23)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.88))

    mask = np.triu(np.ones_like(corr, dtype=bool))   # show lower triangle only

    sns.heatmap(
        corr,
        ax=ax,
        mask=mask,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.0,
        cbar_kws={"shrink": 0.6, "label": "Pearson r"},
        xticklabels=True,
        yticklabels=True,
    )

    ax.set_title(
        "Pearson Correlation Heatmap — Full Dataset\n"
        "(constant columns excluded; lower triangle shown)",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.tick_params(axis="y", rotation=0,  labelsize=6)

    plt.tight_layout()
    out = _REPORTS_DIR / "correlation_heatmap.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def save_report(
    r_small: dict[str, Any],
    r_full: dict[str, Any],
    cmp: dict[str, Any],
) -> None:
    """
    Serialise all analysis results to:
      - reports/dataset_analysis.txt   (human-readable text report)
      - reports/constant_columns.csv   (constant columns details)
      - reports/sentinel_columns.csv   (sentinel column details)

    Parameters
    ----------
    r_small, r_full : dict
        Per-dataset analysis results.
    cmp : dict
        Cross-dataset comparison results.
    """
    _save_text_report(r_small, r_full, cmp)
    _save_constant_columns_csv(r_small, r_full)
    _save_sentinel_columns_csv(r_small, r_full)


def _save_text_report(
    r_small: dict[str, Any],
    r_full: dict[str, Any],
    cmp: dict[str, Any],
) -> None:
    """Write the full textual analysis report to reports/dataset_analysis.txt."""

    buf = StringIO()

    def p(*args, **kwargs):
        """Print to the string buffer and simultaneously to stdout."""
        print(*args, **kwargs, file=buf)
        print(*args, **kwargs)          # mirror to terminal

    sep72  = "=" * 72
    sep72m = "-" * 72

    # ── Report Header ─────────────────────────────────────────────────────────
    p(sep72)
    p("  IEEE PHISHING DETECTION PROJECT - DATASET ANALYSIS REPORT")
    p(f"  Generated : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    p(f"  Datasets  : {SMALL_CSV.name}  |  {FULL_CSV.name}")
    p(sep72)

    for r in [r_small, r_full]:
        name = r["name"]
        p()
        p("=" * 72)
        p(f"  SECTION: {name.upper()}")
        p("=" * 72)

        # 1. Shape
        p(f"\n{'-'*40}")
        p(f"  1. SHAPE")
        p(f"{'-'*40}")
        p(f"  Rows    : {r['n_rows']:,}")
        p(f"  Columns : {r['n_cols']}")

        # 2. Column names
        p(f"\n{'-'*40}")
        p(f"  2. COLUMN NAMES  ({r['n_cols']} total)")
        p(f"{'-'*40}")
        # Print in groups of 4 for readability
        cols = r["columns"]
        for i in range(0, len(cols), 4):
            chunk = cols[i : i + 4]
            p("  " + "   ".join(f"{c:<30}" for c in chunk))

        # 3. Data types
        p(f"\n{'-'*40}")
        p(f"  3. DATA TYPES")
        p(f"{'-'*40}")
        p(f"  Dtype distribution:")
        for dtype, cnt in r["dtype_counts"].items():
            p(f"    {str(dtype):<12} : {cnt:,} column(s)")

        # 4. Missing values
        p(f"\n{'-'*40}")
        p(f"  4. MISSING VALUES  (genuine NaN)")
        p(f"{'-'*40}")
        if r["total_missing"] == 0:
            p("  ✓  No NaN values found in this dataset.")
            p("     Note: Missing information is encoded as sentinel value −1.")
        else:
            p(f"  Total missing cells : {r['total_missing']:,}")
            p(f"  Affected columns    : {', '.join(r['cols_with_missing'])}")

        # 5. Duplicate rows
        p(f"\n{'-'*40}")
        p(f"  5. DUPLICATE ROWS")
        p(f"{'-'*40}")
        dup_pct = r["n_duplicates"] / r["n_rows"] * 100
        p(f"  Duplicate rows : {r['n_duplicates']:,}  ({dup_pct:.3f} %)")

        # 6 & 7 & 8. Class distribution
        p(f"\n{'-'*40}")
        p(f"  6-8. CLASS DISTRIBUTION")
        p(f"{'-'*40}")
        p(f"  {'Label':<20} {'Count':>10} {'Percentage':>12}")
        p(f"  {'-'*20} {'-'*10} {'-'*12}")
        p(f"  {'Legitimate  (0)':<20} {r['n_legitimate']:>10,} {r['pct_legitimate']:>11.2f} %")
        p(f"  {'Phishing    (1)':<20} {r['n_phishing']:>10,} {r['pct_phishing']:>11.2f} %")
        p(f"  {'-'*20} {'-'*10} {'-'*12}")
        p(f"  {'TOTAL':<20} {r['n_rows']:>10,} {'100.00 %':>12}")
        p(f"\n  Imbalance ratio (min:maj) : {r['imbalance_ratio']:.4f}")

        # 9. Memory usage
        p(f"\n{'-'*40}")
        p(f"  9. MEMORY USAGE")
        p(f"{'-'*40}")
        p(f"  {r['memory_mb']:.3f} MB  ({r['memory_bytes']:,} bytes)")

        # 10. Basic statistics
        p(f"\n{'-'*40}")
        p(f"  10. BASIC STATISTICS  (pandas describe)")
        p(f"{'-'*40}")
        p()
        p(r["describe"].to_string())

        # 12. Constant columns
        p(f"\n{'-'*40}")
        p(f"  12. CONSTANT COLUMNS  (zero variance)")
        p(f"{'-'*40}")
        p(f"  Count : {r['n_constant_cols']}")
        if r["constant_cols"]:
            p(f"  {'Column':<35} {'Constant Value':>15}")
            p(f"  {'-'*35} {'-'*15}")
            for col in r["constant_cols"]:
                val = r["constant_col_values"][col]
                p(f"  {col:<35} {str(val):>15}")

        # 13 & 14. Sentinel columns
        p(f"\n{'-'*40}")
        p(f"  13-14. SENTINEL VALUE (-1) COLUMNS")
        p(f"{'-'*40}")
        p(f"  Affected columns : {r['n_sentinel_cols']}")
        if r["sentinel_cols"]:
            p(f"\n  {'Column':<38} {'Rows w/ −1':>12} {'Coverage':>10}")
            p(f"  {'-'*38} {'-'*12} {'-'*10}")
            for col in r["sentinel_cols"]:
                cnt = r["sentinel_counts"][col]
                pct = r["sentinel_pct"][col]
                p(f"  {col:<38} {cnt:>12,} {pct:>9.2f} %")

        # 15. Recommendations
        p(f"\n{'-'*40}")
        p(f"  15. PREPROCESSING RECOMMENDATIONS")
        p(f"{'-'*40}")
        for i, rec in enumerate(r["recommendations"], 1):
            wrapped = textwrap.fill(rec, width=68, subsequent_indent="       ")
            p(f"  [{i:02d}] {wrapped}")

    # ── Cross-dataset comparison ───────────────────────────────────────────────
    p()
    p("=" * 72)
    p("  SECTION: CROSS-DATASET SCHEMA COMPARISON")
    p("=" * 72)

    p(f"\n  Schemas identical (column names + order) : {cmp['schemas_identical']}")
    p(f"  Same column set (unordered)              : {cmp['same_column_set']}")
    p(f"  Dtype mismatches                         : {cmp['n_dtype_mismatches']}")

    if cmp["only_in_small"]:
        p(f"\n  Columns ONLY in small dataset : {cmp['only_in_small']}")
    if cmp["only_in_full"]:
        p(f"  Columns ONLY in full dataset  : {cmp['only_in_full']}")

    p(f"\n  Row delta (full − small)   : +{cmp['row_delta']:,} rows")
    p(f"  Row scale factor           :  {cmp['row_scale']:.3f}×")
    p(f"  Phishing % delta           :  {cmp['phishing_pct_delta']:+.2f} pp")
    p(f"  Legitimate % delta         :  {cmp['legitimate_pct_delta']:+.2f} pp")
    p(f"  Constant cols agree        :  {cmp['constant_cols_agree']}")
    p(f"  Sentinel cols agree        :  {cmp['sentinel_cols_agree']}")

    # ── IEEE-style summary ────────────────────────────────────────────────────
    _print_ieee_summary(p, r_small, r_full, cmp)

    # ── Flush buffer to file ──────────────────────────────────────────────────
    out = _REPORTS_DIR / "dataset_analysis.txt"
    out.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\n  [SAVED] {out}")


def _print_ieee_summary(p, r_small, r_full, cmp) -> None:
    """
    Print a professional, IEEE-paper-style dataset characterisation and
    preprocessing pipeline summary.  Called at the end of _save_text_report().
    """
    p()
    p("=" * 72)
    p("  IEEE-STYLE DATASET CHARACTERISATION AND PREPROCESSING ROADMAP")
    p("=" * 72)
    p()

    summary = f"""\
    ABSTRACT - DATASET CHARACTERISATION
    -------------------------------------------------------------------------
    This study employs two variants of a URL-based phishing detection
    dataset, both derived from the same feature-extraction pipeline and
    sharing an identical schema of {r_small['n_cols']} features.

    The small dataset (D_S) comprises {r_small['n_rows']:,} labelled instances
    ({r_small['n_phishing']:,} phishing, {r_small['n_legitimate']:,} legitimate),
    representing a class imbalance ratio of {r_small['imbalance_ratio']:.4f} (minority
    to majority).  The full dataset (D_F) extends D_S by {cmp['row_delta']:,}
    additional records, yielding {r_full['n_rows']:,} instances in total
    ({r_full['n_phishing']:,} phishing, {r_full['n_legitimate']:,} legitimate)
    at a scale factor of {cmp['row_scale']:.3f}× relative to D_S.

    All {r_small['n_cols']} features are integer-typed (int64), encoding
    lexical URL characteristics (URL segment lengths, special-character
    counts), domain-level attributes (IP-based domain, SPF record,
    ASN identifier), and external service signals (DNS resolution,
    Google index status, TLS certificate presence).  The binary target
    variable 'phishing' takes value 1 for malicious URLs and 0 for
    benign ones.

    MISSING DATA AND SENTINEL VALUES
    -------------------------------------------------------------------------
    Neither dataset contains genuine NaN entries.  Instead, the dataset
    authors encode structural absence — e.g. a URL lacking a query
    string or file component — using the sentinel value −1.  In D_S,
    {r_small['n_sentinel_cols']} of {r_small['n_cols']} features carry at least one sentinel
    occurrence; in D_F, {r_full['n_sentinel_cols']} features are affected.  Notably,
    all 20 directory-level and file-level character-count features
    exhibit sentinel coverage exceeding 29 % in D_S and 53 % in D_F,
    while all 20 query-parameter features exceed 87 % sentinel coverage
    in both datasets, reflecting the high proportion of URLs that lack
    these structural components.

    ZERO-VARIANCE (CONSTANT) FEATURES
    -------------------------------------------------------------------------
    {r_small['n_constant_cols']} features exhibit zero variance across both datasets:
    {', '.join(r_small['constant_cols'][:4])} (and {max(0, r_small['n_constant_cols']-4)} more).
    These columns hold a single value (typically 0) for every record
    and convey no discriminative information; they will be dropped
    during the preprocessing phase.

    PREPROCESSING PIPELINE - PHASE 2 OVERVIEW
    -------------------------------------------------------------------------
    The following preprocessing steps will be implemented in
    utils/preprocessor.py (Phase 2) based on the findings above:

    Step 1  — DROP CONSTANT COLUMNS
              Remove all {r_small['n_constant_cols']} zero-variance features identified
              in this analysis.  Reduces the feature space from
              {r_small['n_cols']} to {r_small['n_cols'] - r_small['n_constant_cols']} input dimensions.

    Step 2  — SENTINEL IMPUTATION (directory / file / params features)
              For features with ≥ 50 % sentinel coverage: replace −1
              with 0 and create a companion boolean indicator feature
              'has_<segment>' (e.g. 'has_directory', 'has_file',
              'has_params') that flags whether the URL segment was
              present.  This preserves structural information without
              inflating the feature mean.

    Step 3  — SENTINEL IMPUTATION (network / DNS features)
              For features with < 50 % sentinel coverage
              (time_response, asn_ip, ttl_hostname, domain_spf,
              time_domain_activation, time_domain_expiration,
              qty_ip_resolved, qty_redirects): replace −1 with the
              column median computed exclusively on non-sentinel rows,
              then add a binary missingness-indicator feature.

    Step 4  — DUPLICATE REMOVAL
              Remove exact duplicate rows identified in this analysis
              to prevent data leakage across cross-validation folds.

    Step 5  — STRATIFIED TRAIN / VALIDATION / TEST SPLIT
              Partition each dataset into 70 % training, 15 %
              validation, and 15 % test subsets using stratified
              sampling to preserve class proportions in each split.

    Step 6  - FEATURE SCALING (MinMaxScaler — IEEE paper specification)
              Apply MinMaxScaler normalization, which linearly scales each
              numerical feature to the [0, 1] range, as specified by the
              IEEE paper prior to training deep learning models.  This
              normalization is applied after sentinel imputation and
              duplicate removal, and is fitted exclusively on the training
              split to prevent data leakage.  Tree-based ensembles
              (Random Forest, XGBoost, LightGBM) are trained on the
              original (unscaled) feature values.

    Step 7  — CLASS-IMBALANCE HANDLING (if required)
              Evaluate SMOTE oversampling on the training set only.
              Prioritise F1-macro and AUC-ROC as primary evaluation
              metrics rather than raw accuracy.

    All preprocessing artefacts (scalers, imputer statistics) will be
    serialised with joblib and stored in models/saved/ to ensure
    identical transformation is applied at inference time.
    -------------------------------------------------------------------------
    END OF REPORT
    """

    # Dedent and print each line
    for line in textwrap.dedent(summary).splitlines():
        p(line)


def _save_constant_columns_csv(
    r_small: dict[str, Any],
    r_full: dict[str, Any],
) -> None:
    """
    Write constant_columns.csv — one row per (dataset, column) pair
    containing the column name and its single constant value.
    """
    rows = []
    for r in [r_small, r_full]:
        for col in r["constant_cols"]:
            rows.append({
                "dataset":        r["name"],
                "column":         col,
                "constant_value": r["constant_col_values"][col],
            })

    df_out = pd.DataFrame(rows)
    out    = _REPORTS_DIR / "constant_columns.csv"
    df_out.to_csv(out, index=False)
    print(f"  [SAVED] {out}")


def _save_sentinel_columns_csv(
    r_small: dict[str, Any],
    r_full: dict[str, Any],
) -> None:
    """
    Write sentinel_columns.csv — one row per (dataset, column) pair
    with the count and percentage of sentinel (−1) occurrences.
    """
    rows = []
    for r in [r_small, r_full]:
        for col in r["sentinel_cols"]:
            rows.append({
                "dataset":       r["name"],
                "column":        col,
                "sentinel_rows": r["sentinel_counts"][col],
                "total_rows":    r["n_rows"],
                "coverage_pct":  r["sentinel_pct"][col],
            })

    df_out = pd.DataFrame(rows)
    out    = _REPORTS_DIR / "sentinel_columns.csv"
    df_out.to_csv(out, index=False)
    print(f"  [SAVED] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Orchestrate the full dataset analysis pipeline:
      1. Load both datasets (read-only).
      2. Run per-dataset analysis.
      3. Compare datasets.
      4. Generate visualisations.
      5. Persist all reports and CSVs.
    """
    print("\n" + "=" * 72)
    print("  IEEE PHISHING DETECTION - DATASET ANALYSIS MODULE")
    print("  Starting pipeline...")
    print("=" * 72 + "\n")

    # ── Step 1: Load datasets ─────────────────────────────────────────────────
    print("[1/5] Loading datasets …")
    if not SMALL_CSV.exists():
        sys.exit(f"  ERROR: Dataset not found: {SMALL_CSV}")
    if not FULL_CSV.exists():
        sys.exit(f"  ERROR: Dataset not found: {FULL_CSV}")

    df_small = pd.read_csv(SMALL_CSV)
    df_full  = pd.read_csv(FULL_CSV)
    print(f"  dataset_small.csv loaded - {df_small.shape[0]:,} rows x {df_small.shape[1]} cols")
    print(f"  dataset_full.csv  loaded - {df_full.shape[0]:,}  rows x {df_full.shape[1]} cols")

    # ── Step 2: Per-dataset analysis ─────────────────────────────────────────
    print("\n[2/5] Analysing datasets …")
    r_small = analyze_dataset(df_small, name="Small Dataset")
    r_full  = analyze_dataset(df_full,  name="Full Dataset")
    print("  Analysis complete.")

    # ── Step 3: Cross-dataset comparison ─────────────────────────────────────
    print("\n[3/5] Comparing datasets …")
    cmp = compare_datasets(r_small, r_full)
    print(f"  Schemas identical : {cmp['schemas_identical']}")
    print(f"  Dtype mismatches  : {cmp['n_dtype_mismatches']}")

    # ── Step 4: Generate visualisations ──────────────────────────────────────
    print("\n[4/5] Generating visualisations …")
    generate_visualizations(df_small, df_full, r_small, r_full)

    # ── Step 5: Save text report and CSVs ────────────────────────────────────
    print("\n[5/5] Saving report files …")
    save_report(r_small, r_full, cmp)

    print("\n" + "=" * 72)
    print("  PIPELINE COMPLETE.  Outputs written to:  reports/")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
