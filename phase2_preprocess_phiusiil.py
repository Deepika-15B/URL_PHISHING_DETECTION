from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

try:
    from sklearn.model_selection import train_test_split
except Exception:  # pragma: no cover
    train_test_split = None


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "data" / "phiusiil_dataset" / "Dataset_HTML.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_PATH = ROOT / "reports" / "preprocessing_phase2_report.txt"


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def verify_required_columns(df: pd.DataFrame) -> tuple[bool, str | None]:
    url_present = "URL" in df.columns
    label_present = "label" in df.columns
    if not url_present or not label_present:
        return False, "Required columns missing"
    return True, None


def handle_duplicate_urls(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int, list[str]]:
    duplicate_mask = df.duplicated(subset=["URL"], keep=False)
    duplicate_rows = df.loc[duplicate_mask].copy()
    if duplicate_rows.empty:
        return df.copy(), 0, 0, []

    label_counts = duplicate_rows.groupby("URL")["label"].nunique()
    conflicting_urls = [url for url, count in label_counts.items() if count > 1]
    if conflicting_urls:
        return df.copy(), 0, 0, conflicting_urls

    deduped = df.drop_duplicates(subset=["URL"], keep="first").copy()
    removed_count = len(df) - len(deduped)
    return deduped, removed_count, 0, []


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()
    for col in encoded.columns:
        if col == "label":
            continue
        if encoded[col].dtype == "object":
            codes, _ = pd.factorize(encoded[col], sort=True)
            encoded[col] = codes.astype(np.int64)
    return encoded


def stratified_split(df: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_test_split is not None:
        train_df, temp_df = train_test_split(
            df,
            test_size=0.30,
            stratify=df["label"],
            random_state=random_state,
            shuffle=True,
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            stratify=temp_df["label"],
            random_state=random_state,
            shuffle=True,
        )
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    # Fallback deterministic split without sklearn.
    rng = np.random.default_rng(random_state)
    train_frames = []
    val_frames = []
    test_frames = []
    for label_value in sorted(df["label"].unique()):
        subset = df[df["label"] == label_value].copy()
        subset = subset.sample(frac=1, random_state=random_state + int(label_value)).reset_index(drop=True)
        n_total = len(subset)
        n_train = int(np.floor(n_total * 0.70))
        n_val = int(np.floor(n_total * 0.15))
        n_test = n_total - n_train - n_val
        if n_test < 0:
            n_test = 0
        if n_test == 0 and n_total > 1:
            n_test = 1
        if n_train == 0 and n_total > 0:
            n_train = 1
        if n_train + n_val + n_test > n_total:
            n_test = n_total - n_train - n_val
        keep = subset.iloc[:n_train]
        val_keep = subset.iloc[n_train : n_train + n_val]
        test_keep = subset.iloc[n_train + n_val : n_train + n_val + n_test]
        train_frames.append(keep)
        val_frames.append(val_keep)
        test_frames.append(test_keep)
    train_df = pd.concat(train_frames, ignore_index=True)
    val_df = pd.concat(val_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def write_split_csvs(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    ensure_directory(PROCESSED_DIR)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "validation.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)


def write_inference_reference(df: pd.DataFrame) -> None:
    reference = df[["URL", "Domain", "Title"]].copy()
    reference.to_csv(PROCESSED_DIR / "url_domain_title_for_inference.csv", index=False)


def build_report(
    original_df: pd.DataFrame,
    final_df: pd.DataFrame,
    removed_duplicate_urls: int,
    conflicting_urls: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> str:
    lines = []
    lines.append("PHIUSIIL HTML DATASET PREPROCESSING REPORT - PHASE 2")
    lines.append("===================================================")
    lines.append("")
    lines.append("Original dataset")
    lines.append("----------------")
    lines.append(f"- Source file: {INPUT_PATH}")
    lines.append(f"- Original dataset size: {len(original_df)} rows x {len(original_df.columns)} columns")
    lines.append(f"- URL column present: {'URL' in original_df.columns}")
    lines.append(f"- Target label column present: {'label' in original_df.columns}")
    lines.append(f"- Original feature count: {len(original_df.columns) - 1}")
    lines.append("")
    lines.append("Preprocessing decisions")
    lines.append("-----------------------")
    lines.append(f"- Removed duplicate URLs with identical labels: {removed_duplicate_urls}")
    lines.append(f"- Conflicting duplicate URLs retained: {len(conflicting_urls)}")
    if conflicting_urls:
        lines.append("- Conflicting URLs:")
        for url in conflicting_urls[:20]:
            lines.append(f"  * {url}")
    lines.append("- Removed columns for model training: URL, Domain, Title")
    lines.append(f"- Final processed dataset size: {len(final_df)} rows x {len(final_df.columns)} columns")
    lines.append(f"- Remaining feature count: {len(final_df.columns) - 1}")
    lines.append(
        f"- Feature dtypes after preprocessing: {', '.join([f'{col}={final_df[col].dtype}' for col in final_df.columns])}"
    )
    lines.append("")
    lines.append("Split sizes")
    lines.append("-----------")
    lines.append(f"- Training set: {len(train_df)} rows")
    lines.append(f"- Validation set: {len(val_df)} rows")
    lines.append(f"- Test set: {len(test_df)} rows")
    lines.append("")
    lines.append("Label distribution by split")
    lines.append("---------------------------")
    for name, split_df in (("train", train_df), ("validation", val_df), ("test", test_df)):
        counts = split_df["label"].value_counts().sort_index().to_dict()
        lines.append(f"- {name}: {counts}")
    lines.append("")
    lines.append("Recommendation")
    lines.append("--------------")
    lines.append("- The next module can consume the numeric train/validation/test CSV files directly for FNN, DNN, Wide & Deep, or TabNet development.")
    lines.append("- Keep the saved URL/Domain/Title reference file for future inference and auditing.")
    lines.append("")
    lines.append("End of report")
    lines.append("============")
    return "\n".join(lines)


def main() -> None:
    ensure_directory(PROCESSED_DIR)
    ensure_directory(ROOT / "reports")

    df = pd.read_csv(INPUT_PATH)
    ok, message = verify_required_columns(df)
    if not ok:
        raise ValueError(message)

    original_df = df.copy()
    original_label_values = sorted(df["label"].dropna().unique().tolist())
    if original_label_values != [0, 1]:
        raise ValueError(f"Unexpected label values found: {original_label_values}")

    deduped_df, removed_duplicate_urls, _, conflicting_urls = handle_duplicate_urls(df)
    if conflicting_urls:
        # Keep all rows to avoid silently removing conflicting duplicate URLs.
        processed_df = df.copy()
        removed_duplicate_urls = 0
    else:
        processed_df = deduped_df.copy()

    reference_df = processed_df[["URL", "Domain", "Title"]].copy() if {"URL", "Domain", "Title"}.issubset(processed_df.columns) else df[["URL", "Domain", "Title"]].copy()
    if "URL" in processed_df.columns:
        processed_df = processed_df.drop(columns=["URL", "Domain", "Title"])

    write_inference_reference(reference_df)

    # The training-ready dataframe should contain only numeric features and the label.
    # Keep the target label unchanged and encode only categorical features.
    processed_df = encode_categorical_features(processed_df)
    if not all(pd.api.types.is_numeric_dtype(processed_df[col]) for col in processed_df.columns if col != "label"):
        raise ValueError("Non-numeric feature columns remain after preprocessing")
    processed_df["label"] = processed_df["label"].astype(int)

    train_df, val_df, test_df = stratified_split(processed_df, random_state=42)
    write_split_csvs(train_df, val_df, test_df)

    report_text = build_report(original_df, processed_df, removed_duplicate_urls, conflicting_urls, train_df, val_df, test_df)
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print("Preprocessing complete")
    print(f"Processed files written to {PROCESSED_DIR}")
    print(f"Report written to {REPORT_PATH}")
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")


if __name__ == "__main__":
    main()
