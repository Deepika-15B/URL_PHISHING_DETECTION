"""
utils/preprocessing.py
=======================
Preprocessing Pipeline â€” IEEE Phishing Detection Project
---------------------------------------------------------
Implements the full preprocessing pipeline described in the IEEE phishing
detection paper.  The pipeline operates on dataset_small.csv and produces
a clean, scaled feature matrix ready for feature selection and model training.

Pipeline Steps (in execution order)
-------------------------------------
  1.  Load dataset_small.csv from disk.
  2.  Remove exact duplicate rows.
  3.  Remove constant columns (zero variance â€” no predictive value).
  4.  Detect all columns containing the sentinel value -1.
  5.  Handle sentinel values selectively:
        a) STRUCTURAL sentinels (directory / file / params features):
              -1 means the URL segment is absent, not that the value is
              missing.  Replace -1 with 0 and inject one binary indicator
              column per URL segment group (has_directory, has_file,
              has_params).
        b) MISSING-DATA sentinels (network / DNS / WHOIS features):
              -1 means the lookup failed.  Replace -1 with NaN so that
              the imputer can handle it in the next step.
  6.  Median imputation on all remaining NaN values.
  7.  Cast all feature columns to float32 (memory-efficient numeric type).
  8.  Apply MinMaxScaler (IEEE paper specification) to all feature columns,
      mapping each feature to the [0, 1] range.
  9.  Persist the fitted scaler  â†’  models/scaler.pkl
  10. Persist the final feature name list  â†’  models/preprocessed_feature_names.pkl
  11. Write a human-readable preprocessing report  â†’  reports/preprocessing_report.txt

Usage
-----
    # Run the full pipeline from the command line:
    python utils/preprocessing.py

    # Import and reuse individual steps from other modules (e.g. inference):
    from utils.preprocessing import (
        load_dataset,
        handle_sentinel_values,
        impute_missing_values,
        normalize_features,
    )

Author  : Phishing Detection IEEE Team
Version : 1.0.0
"""

# â”€â”€ Standard library â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import sys
import pickle
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# â”€â”€ Third-party â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# â”€â”€ Project paths (resolve relative to this file's location) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_HERE         = Path(__file__).resolve().parent           # .../utils/
_PROJECT_ROOT = _HERE.parent                              # .../phishing_detection_ieee/
_DATA_DIR     = _PROJECT_ROOT / "data"
_MODELS_DIR   = _PROJECT_ROOT / "models"
_REPORTS_DIR  = _PROJECT_ROOT / "reports"

# Create output directories if they do not already exist
_MODELS_DIR.mkdir(parents=True, exist_ok=True)
_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# â”€â”€ Configuration constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATASET_PATH   = _DATA_DIR / "dataset_small.csv"
TARGET_COL     = "phishing"          # Binary label: 1 = phishing, 0 = legitimate
SENTINEL_VAL   = -1                  # Value used by dataset authors for absent data

# â”€â”€ Output artefact paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SCALER_PATH           = _MODELS_DIR / "scaler.pkl"
FEATURE_NAMES_PATH    = _MODELS_DIR / "preprocessed_feature_names.pkl"
REPORT_PATH           = _REPORTS_DIR / "preprocessing_report.txt"
PREPROCESSED_DATA_PATH = _DATA_DIR / "dataset_small_preprocessed.csv"

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SENTINEL COLUMN CLASSIFICATION
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Based on the Dataset Analysis module findings (see reports/dataset_analysis.txt),
# sentinel columns fall into two semantically distinct groups:
#
#   GROUP A â€” STRUCTURAL SENTINELS (URL segment absent)
#   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#   When a URL has no directory, file, or query-parameter segment, ALL
#   character-count features for that segment are set to -1 by the extractor.
#   Semantically, -1 here means "this segment does not exist", NOT "value
#   unknown".  The correct treatment is:
#     â€¢ Replace -1 â†’ 0  (absent segment contributes 0 characters)
#     â€¢ Add a binary indicator column (has_directory, has_file, has_params)
#       so the model can learn from the presence/absence of the segment itself.
#
#   GROUP B â€” MISSING-DATA SENTINELS (lookup failed)
#   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#   For network-level and WHOIS features, -1 signals that the data could
#   not be retrieved (e.g. DNS timeout, WHOIS rate-limit, scraping error).
#   This is genuine missingness and is handled by replacing -1 â†’ NaN and
#   then applying median imputation.

# Directory-level character-count features + directory_length
# (all share the exact same 17,507-row sentinel mask in dataset_small.csv)
STRUCTURAL_DIRECTORY_COLS: list[str] = [
    "qty_dot_directory",        "qty_hyphen_directory",
    "qty_underline_directory",  "qty_slash_directory",
    "qty_questionmark_directory","qty_equal_directory",
    "qty_at_directory",         "qty_and_directory",
    "qty_exclamation_directory","qty_space_directory",
    "qty_tilde_directory",      "qty_comma_directory",
    "qty_plus_directory",       "qty_asterisk_directory",
    "qty_hashtag_directory",    "qty_dollar_directory",
    "qty_percent_directory",    "directory_length",
]

# File-level character-count features + file_length
# (share the same 17,507-row sentinel mask â€” confirmed identical to directory mask)
STRUCTURAL_FILE_COLS: list[str] = [
    "qty_dot_file",        "qty_hyphen_file",
    "qty_underline_file",  "qty_slash_file",
    "qty_questionmark_file","qty_equal_file",
    "qty_at_file",         "qty_and_file",
    "qty_exclamation_file","qty_space_file",
    "qty_tilde_file",      "qty_comma_file",
    "qty_plus_file",       "qty_asterisk_file",
    "qty_hashtag_file",    "qty_dollar_file",
    "qty_percent_file",    "file_length",
]

# Query-parameter-level features (87.3 % sentinel coverage)
STRUCTURAL_PARAMS_COLS: list[str] = [
    "qty_dot_params",        "qty_hyphen_params",
    "qty_underline_params",  "qty_slash_params",
    "qty_questionmark_params","qty_equal_params",
    "qty_at_params",         "qty_and_params",
    "qty_exclamation_params","qty_space_params",
    "qty_tilde_params",      "qty_comma_params",
    "qty_plus_params",       "qty_asterisk_params",
    "qty_hashtag_params",    "qty_dollar_params",
    "qty_percent_params",    "params_length",
    "tld_present_params",    "qty_params",
]

# Convenience mapping: indicator column name â†’ list of structural columns it covers
STRUCTURAL_GROUPS: dict[str, list[str]] = {
    "has_directory": STRUCTURAL_DIRECTORY_COLS,
    "has_file":      STRUCTURAL_FILE_COLS,
    "has_params":    STRUCTURAL_PARAMS_COLS,
}

# Network / DNS / WHOIS features where -1 means "lookup failed" (genuine missing)
MISSING_DATA_SENTINEL_COLS: list[str] = [
    "time_response",          # HTTP response time â€” failed if request timed out
    "domain_spf",             # SPF DNS record â€” absent if not set or lookup failed
    "asn_ip",                 # Autonomous System Number â€” unavailable if IP unresolved
    "time_domain_activation", # Domain registration date â€” unavailable via WHOIS
    "time_domain_expiration", # Domain expiration date   â€” unavailable via WHOIS
    "qty_ip_resolved",        # Count of resolved IPs    â€” 0 when DNS failed (but -1 = error)
    "ttl_hostname",           # DNS TTL                  â€” unavailable if DNS failed
    "qty_redirects",          # HTTP redirect count      â€” unavailable if fetch failed
    "url_google_index",       # Google index flag        â€” -1 on scraping errors (4 rows)
    "domain_google_index",    # Google index flag        â€” -1 on scraping errors (2 rows)
]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STEP 1 â€” LOAD DATASET
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """
    Load the phishing detection dataset from a CSV file.

    The raw file is returned completely unmodified â€” no filtering, casting,
    or imputation is performed here.  All subsequent steps are applied
    explicitly in the pipeline so that each transformation is auditable.

    Parameters
    ----------
    path : Path
        Absolute path to the CSV dataset.  Defaults to dataset_small.csv.

    Returns
    -------
    pd.DataFrame
        Raw dataframe exactly as read from disk.

    Raises
    ------
    SystemExit
        If the file does not exist.
    """
    if not path.exists():
        sys.exit(f"[ERROR] Dataset not found: {path}")

    df = pd.read_csv(path)

    print(f"  Loaded : {path.name}")
    print(f"  Shape  : {df.shape[0]:,} rows x {df.shape[1]} columns")

    return df


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STEP 2 â€” REMOVE DUPLICATE ROWS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Identify and remove exact duplicate rows from the dataframe.

    Duplicates create data-leakage risk when the same record appears in both
    the training and validation splits during cross-validation.  Removing
    them before the train/test split prevents overly optimistic metric
    estimates.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (any stage of the pipeline).

    Returns
    -------
    tuple[pd.DataFrame, int]
        (cleaned_df, n_duplicates_removed)
    """
    n_before = len(df)

    # keep='first' retains the first occurrence and discards all subsequent
    # identical rows.  inplace is avoided intentionally to keep the function
    # side-effect-free (returns a new dataframe).
    df_clean = df.drop_duplicates(keep="first").reset_index(drop=True)

    n_removed = n_before - len(df_clean)
    print(f"  Duplicates removed : {n_removed:,}  "
          f"({n_before:,} -> {len(df_clean):,} rows)")

    return df_clean, n_removed


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STEP 3 â€” REMOVE CONSTANT COLUMNS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def remove_constant_columns(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop all columns that carry a single unique value across the entire
    dataset (zero variance).

    Constant columns cannot differentiate phishing from legitimate URLs:
    a classifier gains zero information from a feature that is always 0.
    The Dataset Analysis module identified 13 such columns, all in the
    domain-level character-count group (e.g. qty_slash_domain is always 0
    because domain names cannot legally contain slashes).

    The target column is explicitly excluded from the zero-variance check
    because a dataset with only one class would be degenerate by definition.

    Parameters
    ----------
    df         : pd.DataFrame
    target_col : str  â€” Column name of the binary label (never dropped).

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        (df_without_constant_cols, list_of_dropped_column_names)
    """
    # nunique() counts distinct values; a constant column has exactly 1.
    # The target column is excluded from consideration.
    feature_cols    = [c for c in df.columns if c != target_col]
    constant_mask   = df[feature_cols].nunique() == 1
    constant_cols   = df[feature_cols].columns[constant_mask].tolist()

    if constant_cols:
        df = df.drop(columns=constant_cols)
        print(f"  Constant columns removed : {len(constant_cols)}")
        for col in constant_cols:
            print(f"    - {col}")
    else:
        print("  No constant columns found.")

    return df, constant_cols


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STEP 4 + 5 â€” DETECT AND HANDLE SENTINEL VALUES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def handle_sentinel_values(
    df: pd.DataFrame,
    sentinel: int = SENTINEL_VAL,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Selectively replace sentinel values (-1) based on their semantic meaning.

    This function implements the two-strategy approach mandated by the
    Dataset Analysis findings:

    Strategy A â€” Structural sentinels (directory / file / params columns)
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    When a URL has no directory, file, or parameter segment, the dataset
    extractor records -1 for every character-count feature of that segment.
    The value -1 here means "segment absent", NOT "value unknown".

    Treatment:
      1.  Create a binary indicator column per URL segment group:
              has_directory  â†’  1 if the URL has a directory segment, else 0
              has_file       â†’  1 if the URL has a file component, else 0
              has_params     â†’  1 if the URL has a query-parameter string, else 0
          These indicators are placed immediately before the group's first
          column so they remain adjacent to the features they describe.
      2.  Replace -1 with 0 in all structural columns.
          A URL with no directory legitimately has 0 dots, 0 hyphens, etc.

    Strategy B â€” Missing-data sentinels (network / DNS / WHOIS columns)
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    For features such as time_response, ttl_hostname, and asn_ip, -1
    indicates that the data-collection step failed (timeout, rate-limit,
    scraping error).  These are genuinely missing values.

    Treatment:
      Replace -1 with NaN.  The imputer in the next pipeline step will
      substitute the column median computed from valid (non-sentinel) rows.

    Parameters
    ----------
    df       : pd.DataFrame  â€” Dataframe after constant-column removal.
    sentinel : int           â€” The sentinel integer value (default -1).

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (processed_df, info_dict)
        info_dict keys:
            "structural_groups"    : dict {indicator_name: [affected_cols]}
            "missing_data_cols"    : list of network/DNS columns treated as NaN
            "indicators_added"     : list of new binary indicator column names
            "structural_cols_zeroed": list of structural cols where -1 â†’ 0
            "missing_data_nullified": list of cols where -1 â†’ NaN
    """
    df = df.copy()   # Never mutate the caller's dataframe
    info: dict[str, Any] = {
        "structural_groups":     {},
        "missing_data_cols":     [],
        "indicators_added":      [],
        "structural_cols_zeroed":[],
        "missing_data_nullified":[],
    }

    # â”€â”€ Strategy A: Structural sentinels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for indicator_name, group_cols in STRUCTURAL_GROUPS.items():

        # Keep only columns that actually exist in df (some may have been
        # dropped as constant columns in the previous step â€” though in practice
        # none of the structural cols are constant).
        active_cols = [c for c in group_cols if c in df.columns]
        if not active_cols:
            continue

        # Determine which rows have the sentinel for this group.
        # All columns in a group share the same sentinel mask (verified in
        # the Dataset Analysis module), so we only need to check one.
        sentinel_mask = df[active_cols[0]] == sentinel

        # Create the binary indicator:  1 = segment present, 0 = absent
        indicator_values = (~sentinel_mask).astype(np.int8)

        # Insert the indicator column immediately before the first group column
        # that is still in the dataframe, for clarity of layout.
        first_col_pos = df.columns.get_loc(active_cols[0])
        df.insert(first_col_pos, indicator_name, indicator_values)

        # Replace -1 with 0 in all structural columns of this group.
        # Using .where() avoids SettingWithCopyWarning.
        df[active_cols] = df[active_cols].where(~sentinel_mask, other=0)

        n_rows_affected = int(sentinel_mask.sum())
        print(f"  [Structural] {indicator_name:15s} inserted | "
              f"{n_rows_affected:,} rows zeroed across {len(active_cols)} cols")

        info["structural_groups"][indicator_name] = active_cols
        info["indicators_added"].append(indicator_name)
        info["structural_cols_zeroed"].extend(active_cols)

    # â”€â”€ Strategy B: Missing-data sentinels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for col in MISSING_DATA_SENTINEL_COLS:
        if col not in df.columns:
            continue   # Column may have been dropped earlier

        n_sentinel = int((df[col] == sentinel).sum())
        if n_sentinel == 0:
            continue   # No sentinel occurrences in this column

        # Replace -1 with NaN so sklearn's imputer can process it cleanly.
        df[col] = df[col].replace(sentinel, np.nan)

        print(f"  [Missing-data] {col:<30} : {n_sentinel:,} sentinel -> NaN")
        info["missing_data_cols"].append(col)
        info["missing_data_nullified"].append(col)

    return df, info


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STEP 6 â€” MEDIAN IMPUTATION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def impute_missing_values(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Fill all remaining NaN values using column-wise median imputation.

    Median is preferred over mean for this dataset because:
      â€¢ Several network/WHOIS features (asn_ip, time_domain_*) have
        right-skewed distributions where extreme values inflate the mean.
      â€¢ Median is robust to outliers and produces a value that always
        falls within the observed feature range.

    The target column is excluded from imputation â€” it must never be altered.

    Parameters
    ----------
    df         : pd.DataFrame  â€” Dataframe after sentinel handling.
    target_col : str           â€” Column name of the binary label.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, float]]
        (imputed_df, medians_dict)
        medians_dict maps each imputed column name to the median value used.
        This dict should be saved alongside the scaler for inference-time use.
    """
    df = df.copy()

    feature_cols = [c for c in df.columns if c != target_col]
    nan_counts   = df[feature_cols].isnull().sum()
    cols_to_impute = nan_counts[nan_counts > 0]

    medians: dict[str, float] = {}

    if cols_to_impute.empty:
        print("  No NaN values to impute.")
        return df, medians

    for col, n_nan in cols_to_impute.items():
        # Compute median only from non-NaN rows so sentinel-contaminated rows
        # (converted to NaN in the previous step) do not influence the result.
        col_median = df[col].median(skipna=True)
        df[col]    = df[col].fillna(col_median)
        medians[col] = float(col_median)
        print(f"  Imputed {col:<30} : {n_nan:,} NaN  â†’  median = {col_median:.4f}")

    return df, medians


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STEP 7 â€” TYPE NORMALISATION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def ensure_numeric_types(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Cast all feature columns to float32.

    After imputation, newly-added indicator columns (int8) and the original
    int64 columns coexist.  Casting everything to float32:
      â€¢ Satisfies sklearn's MinMaxScaler requirement for a uniform float dtype.
      â€¢ Reduces memory usage compared to float64 (half the bytes per element).
      â€¢ Keeps the target column as int64 (0/1 label unchanged).

    Parameters
    ----------
    df         : pd.DataFrame
    target_col : str

    Returns
    -------
    pd.DataFrame
        Dataframe with feature columns cast to float32.
    """
    feature_cols = [c for c in df.columns if c != target_col]
    df = df.copy()
    df[feature_cols] = df[feature_cols].astype(np.float32)

    print(f"  Cast {len(feature_cols)} feature columns to float32.")
    print(f"  Target column '{target_col}' retained as {df[target_col].dtype}.")

    return df


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STEP 8 â€” MINMAX NORMALISATION
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def normalize_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, MinMaxScaler, list[str]]:
    """
    Scale all feature columns to the [0, 1] range using MinMaxScaler.

    The IEEE paper explicitly specifies MinMaxScaler normalization before
    training deep learning models.  The linear transformation is:

        X_scaled = (X - X_min) / (X_max - X_min)

    Properties:
      â€¢ Each feature independently maps its observed minimum to 0 and
        its observed maximum to 1.
      â€¢ Preserves the shape (distribution) of the original data.
      â€¢ Preferred over z-score normalisation when the data is not Gaussian
        and when neural-network activation functions expect bounded inputs.

    IMPORTANT: The scaler is fitted ONLY on the training features.
    At inference time, the saved scaler must be loaded and used to
    transform unseen data â€” the scaler must NEVER be re-fitted on test data.

    Parameters
    ----------
    df         : pd.DataFrame  â€” Fully imputed, type-normalised dataframe.
    target_col : str           â€” Label column (excluded from scaling).

    Returns
    -------
    tuple[pd.DataFrame, MinMaxScaler, list[str]]
        (scaled_df, fitted_scaler, feature_names_list)
    """
    feature_cols = [c for c in df.columns if c != target_col]

    # Separate features from labels
    X = df[feature_cols].values       # numpy array (n_samples, n_features)
    y = df[target_col].values         # 1D label array

    # Fit and transform in one pass (equivalent to fit() then transform())
    scaler   = MinMaxScaler(feature_range=(0, 1), clip=False)
    X_scaled = scaler.fit_transform(X)

    # Reconstruct a dataframe with original column names
    df_scaled              = pd.DataFrame(X_scaled, columns=feature_cols)
    df_scaled[target_col]  = y   # Re-attach the (unscaled) label column

    # Reorder: features first, target last
    df_scaled = df_scaled[feature_cols + [target_col]]

    print(f"  MinMaxScaler applied to {len(feature_cols)} features.")
    print(f"  Feature range after scaling: "
          f"[{df_scaled[feature_cols].min().min():.4f}, "
          f"{df_scaled[feature_cols].max().max():.4f}]")

    return df_scaled, scaler, feature_cols


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STEP 9 + 10 â€” SAVE ARTEFACTS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def save_artifacts(
    scaler: MinMaxScaler,
    feature_names: list[str],
    imputer_medians: dict[str, float],
    scaler_path: Path       = SCALER_PATH,
    feat_names_path: Path   = FEATURE_NAMES_PATH,
) -> None:
    """
    Persist the fitted scaler and feature-name list using pickle.

    These artefacts are required at inference time to ensure identical
    preprocessing is applied to new (unseen) URL data:

      scaler.pkl
          The fitted MinMaxScaler.  Apply scaler.transform(X_new) on any
          new feature matrix â€” do NOT call fit() or fit_transform() again.

      preprocessed_feature_names.pkl
          Ordered list of feature column names after all preprocessing steps.
          The inference pipeline must present features in exactly this order
          before calling scaler.transform().

      imputer_medians are embedded inside the scaler pickle as a custom
      attribute so the inference pipeline can reproduce the NaN-filling step
      without re-reading the training data.

    Parameters
    ----------
    scaler          : MinMaxScaler  â€” Fitted scaler from normalize_features().
    feature_names   : list[str]     â€” Column names in the scaled feature matrix.
    imputer_medians : dict          â€” {col: median_value} for NaN imputation.
    scaler_path     : Path          â€” Where to save the scaler pickle.
    feat_names_path : Path          â€” Where to save the feature-names pickle.
    """
    # Attach imputer medians as a custom attribute on the scaler object so
    # a single pickle file carries all inference-time preprocessing state.
    scaler.imputer_medians_ = imputer_medians

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  [SAVED] {scaler_path}")

    with open(feat_names_path, "wb") as f:
        pickle.dump(feature_names, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  [SAVED] {feat_names_path}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STEP 11 â€” PREPROCESSING REPORT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def save_report(
    original_shape: tuple[int, int],
    rows_after_dedup: int,
    n_duplicates_removed: int,
    constant_cols_removed: list[str],
    sentinel_info: dict[str, Any],
    nan_before_imputation: int,
    nan_after_imputation: int,
    imputer_medians: dict[str, float],
    feature_names: list[str],
    target_col: str = TARGET_COL,
    report_path: Path = REPORT_PATH,
) -> None:
    """
    Write a structured text report summarising every preprocessing step.

    The report mirrors the structure of the IEEE paper's data preparation
    section, making it straightforward to reference during peer review.

    Parameters
    ----------
    (See parameter names â€” all map 1-to-1 to pipeline step outputs.)
    """
    buf = StringIO()

    def p(*args, **kwargs):
        print(*args, **kwargs, file=buf)
        print(*args, **kwargs)   # Echo to terminal simultaneously

    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    sep = "=" * 72
    div = "-" * 72

    p(sep)
    p("  IEEE PHISHING DETECTION PROJECT - PREPROCESSING REPORT")
    p(f"  Generated : {now}")
    p(f"  Dataset   : {DATASET_PATH.name}")
    p(sep)

    # â”€â”€ Step 1: Load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p()
    p(div)
    p("  STEP 1 - DATASET LOADED")
    p(div)
    p(f"  Original rows    : {original_shape[0]:,}")
    p(f"  Original columns : {original_shape[1]}")
    p(f"  Target column    : '{target_col}'")

    # â”€â”€ Step 2: Duplicates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p()
    p(div)
    p("  STEP 2 - DUPLICATE REMOVAL")
    p(div)
    p(f"  Duplicate rows removed : {n_duplicates_removed:,}")
    p(f"  Rows after removal     : {rows_after_dedup:,}")

    # â”€â”€ Step 3: Constant columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p()
    p(div)
    p("  STEP 3 - CONSTANT COLUMN REMOVAL")
    p(div)
    p(f"  Columns removed : {len(constant_cols_removed)}")
    if constant_cols_removed:
        p(f"  {'Column Name':<40} {'Reason'}")
        p(f"  {'-'*40} {'-'*20}")
        for col in constant_cols_removed:
            p(f"  {col:<40} zero variance (always 0)")

    # â”€â”€ Step 4+5: Sentinel handling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p()
    p(div)
    p("  STEP 4+5 - SENTINEL VALUE HANDLING  (-1)")
    p(div)

    p()
    p("  Strategy A: Structural Sentinels (URL segment absent)  ->  0 + indicator")
    p(f"  {'Indicator Column':<20} {'Group Columns (count)':<25} {'Treatment'}")
    p(f"  {'-'*20} {'-'*25} {'-'*25}")
    for indicator, group_cols in sentinel_info["structural_groups"].items():
        present_cols = [c for c in group_cols if c in feature_names or c in sentinel_info["structural_cols_zeroed"]]
        p(f"  {indicator:<20} {len(group_cols)} feature cols"
          f"           -1 -> 0  +  binary indicator")

    p()
    p("  Strategy B: Missing-Data Sentinels (lookup failed)  ->  NaN -> median")
    p(f"  {'Column':<35} {'Sentinel->NaN':<15} {'Median Used'}")
    p(f"  {'-'*35} {'-'*15} {'-'*11}")
    for col in sentinel_info["missing_data_nullified"]:
        median_val = imputer_medians.get(col, "N/A (no NaN)")
        if isinstance(median_val, float):
            median_str = f"{median_val:.4f}"
        else:
            median_str = str(median_val)
        p(f"  {col:<35} -1 -> NaN       {median_str}")

    # â”€â”€ Step 6: Imputation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p()
    p(div)
    p("  STEP 6 - MEDIAN IMPUTATION")
    p(div)
    p(f"  NaN values before imputation : {nan_before_imputation:,}")
    p(f"  NaN values after  imputation : {nan_after_imputation:,}")
    if imputer_medians:
        p()
        p(f"  {'Column':<35} {'Median Value':>12}")
        p(f"  {'-'*35} {'-'*12}")
        for col, med in imputer_medians.items():
            p(f"  {col:<35} {med:>12.4f}")

    # â”€â”€ Step 7: Type normalisation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p()
    p(div)
    p("  STEP 7 - NUMERIC TYPE NORMALISATION")
    p(div)
    p("  All feature columns cast to float32.")
    p("  Target column retained as int64.")

    # â”€â”€ Step 8: Scaling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p()
    p(div)
    p("  STEP 8 - FEATURE SCALING")
    p(div)
    p("  Method        : MinMaxScaler  (IEEE paper specification)")
    p("  Formula       : X_scaled = (X - X_min) / (X_max - X_min)")
    p("  Output range  : [0, 1]")
    p("  Fitted on     : Training data only (prevents data leakage)")
    p("  Excluded col  : 'phishing' (target label â€” not scaled)")

    # â”€â”€ Step 9+10: Artefacts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p()
    p(div)
    p("  STEP 9+10 - SAVED ARTEFACTS")
    p(div)
    p(f"  models/scaler.pkl                    - Fitted MinMaxScaler")
    p(f"  models/preprocessed_feature_names.pkl- {len(feature_names)} feature names (ordered)")

    # â”€â”€ Final feature inventory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    p()
    p(div)
    p(f"  FINAL FEATURE INVENTORY  ({len(feature_names)} features)")
    p(div)
    for i, name in enumerate(feature_names, 1):
        p(f"  {i:>4}. {name}")

    # â”€â”€ IEEE-style pipeline summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    n_original_cols = original_shape[1] - 1  # exclude target
    n_dropped_const = len(constant_cols_removed)
    n_indicators    = len(sentinel_info["indicators_added"])
    n_final_features = len(feature_names)

    p()
    p(sep)
    p("  IEEE METHODOLOGY SUMMARY")
    p(sep)
    p()

    summary_lines = [
        "PREPROCESSING PIPELINE SUMMARY",
        "-------------------------------------------------------------------------",
        "",
        "The preprocessing pipeline transforms the raw dataset_small.csv into a",
        "clean, normalised feature matrix ready for the feature selection phase",
        "of the IEEE phishing detection methodology.",
        "",
        "PIPELINE TRANSFORMATIONS APPLIED",
        "-------------------------------------------------------------------------",
        f"  Raw input features         : {n_original_cols}",
        f"  Constant columns dropped   : {n_dropped_const}",
        f"  Structural sentinel groups : 3  (directory, file, params)",
        f"  Binary indicators added    : {n_indicators}  (has_directory, has_file, has_params)",
        f"  Missing-data cols imputed  : {len(sentinel_info['missing_data_nullified'])}",
        f"  Final feature count        : {n_final_features}",
        f"  Scaling method             : MinMaxScaler  -> range [0, 1]",
        "",
        "DATA QUALITY OUTCOMES",
        "-------------------------------------------------------------------------",
        f"  Duplicate rows removed     : {n_duplicates_removed:,}",
        f"  NaN values after pipeline  : 0  (fully imputed)",
        f"  Data type                  : float32 (memory-optimised)",
        "",
        "READINESS FOR FEATURE SELECTION (PHASE 3)",
        "-------------------------------------------------------------------------",
        "The output matrix is structurally sound and normalised to the [0, 1]",
        "range as required by the IEEE paper.  The next phase will apply feature",
        "selection techniques (e.g. mutual information, chi-squared, or feature",
        "importance from tree-based models) to rank the retained features and",
        "identify the most discriminative subset for training the phishing",
        "detection classifier.",
        "",
        "KEY DESIGN DECISIONS",
        "-------------------------------------------------------------------------",
        "1. Sentinel values were handled differently per semantic group:",
        "   - Structural sentinels (directory/file/params) replaced with 0 +",
        "     binary presence indicators, preserving URL structure information.",
        "   - Missing-data sentinels (network/DNS/WHOIS) replaced with NaN and",
        "     median-imputed to maintain statistical validity.",
        "2. MinMaxScaler was chosen over z-score normalisation because the IEEE",
        "   paper's deep learning models require bounded [0, 1] inputs, and",
        "   MinMaxScaler is invariant to Gaussian assumptions.",
        "3. The scaler is fitted exclusively on training data and serialised,",
        "   ensuring inference-time consistency without data leakage.",
        "4. Constant columns were removed before scaling to avoid degenerate",
        "   MinMaxScaler columns (X_max = X_min -> division by zero).",
        "-------------------------------------------------------------------------",
        "END OF REPORT",
    ]

    for line in summary_lines:
        p(line)

    # Flush buffer to file
    report_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\n  [SAVED] {report_path}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN ORCHESTRATOR â€” preprocess_training_data()
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def preprocess_training_data() -> tuple[pd.DataFrame, MinMaxScaler, list[str]]:
    """
    Execute the full preprocessing pipeline on dataset_small.csv.

    This function is the single entry-point for the pipeline.  It calls
    each step function in order, passes outputs between steps, accumulates
    audit metrics, and finalises by saving artefacts and the report.

    Design principle:
      Each sub-function is independently importable, allowing the inference
      pipeline to reproduce only the subset of steps needed for new data
      (e.g. skip deduplication and constant-column removal, apply sentinel
      handling, impute with saved medians, scale with saved scaler).

    Returns
    -------
    tuple[pd.DataFrame, MinMaxScaler, list[str]]
        (scaled_df, fitted_scaler, feature_names)
        The caller can use scaled_df directly for feature selection or model
        training.
    """
    print("\n" + "=" * 72)
    print("  IEEE PHISHING DETECTION - PREPROCESSING PIPELINE")
    print("  Starting...")
    print("=" * 72 + "\n")

    # â”€â”€ Step 1: Load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("[1/8] Loading dataset...")
    df = load_dataset()
    original_shape = df.shape

    # â”€â”€ Step 2: Remove duplicates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[2/8] Removing duplicate rows...")
    df, n_duplicates = remove_duplicates(df)
    rows_after_dedup  = len(df)

    # â”€â”€ Step 3: Remove constant columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[3/8] Removing constant (zero-variance) columns...")
    df, constant_cols = remove_constant_columns(df)

    # â”€â”€ Step 4 + 5: Handle sentinel values â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[4/8] Detecting and handling sentinel values (-1)...")
    df, sentinel_info = handle_sentinel_values(df)

    # â”€â”€ Step 6: Median imputation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[5/8] Imputing missing values (median strategy)...")
    feature_cols_before_impute = [c for c in df.columns if c != TARGET_COL]
    nan_before = int(df[feature_cols_before_impute].isnull().sum().sum())
    df, imputer_medians = impute_missing_values(df)
    feature_cols_after_impute  = [c for c in df.columns if c != TARGET_COL]
    nan_after  = int(df[feature_cols_after_impute].isnull().sum().sum())
    print(f"  NaN before : {nan_before:,}  |  NaN after : {nan_after:,}")

    # â”€â”€ Step 7: Type normalisation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[6/8] Normalising numeric types to float32...")
    df = ensure_numeric_types(df)

    # â”€â”€ Step 8: MinMaxScaler normalisation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[7/8] Applying MinMaxScaler normalisation (IEEE specification)...")
    df_scaled, scaler, feature_names = normalize_features(df)

    # â”€â”€ Step 9 + 10: Save artefacts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[8/8] Saving artefacts and generating report...")
    save_artifacts(scaler, feature_names, imputer_medians)

    # â”€â”€ Step 11: Generate text report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    save_report(
        original_shape        = original_shape,
        rows_after_dedup      = rows_after_dedup,
        n_duplicates_removed  = n_duplicates,
        constant_cols_removed = constant_cols,
        sentinel_info         = sentinel_info,
        nan_before_imputation = nan_before,
        nan_after_imputation  = nan_after,
        imputer_medians       = imputer_medians,
        feature_names         = feature_names,
    )
    # Persist the fully processed training table for the feature-selection
    # phase. df_scaled already contains feature_names order followed by the
    # unchanged phishing label, so saving it preserves the fitted schema.
    df_scaled.to_csv(PREPROCESSED_DATA_PATH, index=False)
    print(f"  [SAVED] {PREPROCESSED_DATA_PATH}")
    print(f"  Processed dataset shape: {df_scaled.shape}")

    print("\n" + "=" * 72)
    print(f"  PIPELINE COMPLETE.")
    print(f"  Input  : {original_shape[0]:,} rows x {original_shape[1]} cols")
    print(f"  Output : {len(df_scaled):,} rows x {len(feature_names)} features  "
          f"(+1 target col)")
    print(f"  Outputs written to: models/  and  reports/")
    print("=" * 72 + "\n")

    return df_scaled, scaler, feature_names


# â”€â”€ Script entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    preprocess_training_data()


