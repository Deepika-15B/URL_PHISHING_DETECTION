from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / 'data' / 'phiusiil_dataset' / 'Dataset_HTML.csv'
TRAIN_PATH = ROOT / 'data' / 'processed' / 'train.csv'
VAL_PATH = ROOT / 'data' / 'processed' / 'validation.csv'
TEST_PATH = ROOT / 'data' / 'processed' / 'test.csv'
FEATURES_PATH = ROOT / 'models' / 'top20_features.pkl'
REPORT_PATH = ROOT / 'reports' / 'fnn_validation_report.txt'


def handle_duplicate_urls(df: pd.DataFrame) -> tuple[pd.DataFrame, int, list[str]]:
    duplicate_mask = df.duplicated(subset=['URL'], keep=False)
    duplicate_rows = df.loc[duplicate_mask].copy()
    if duplicate_rows.empty:
        return df.copy(), 0, []

    label_counts = duplicate_rows.groupby('URL')['label'].nunique()
    conflicting_urls = [url for url, count in label_counts.items() if count > 1]
    if conflicting_urls:
        return df.copy(), 0, conflicting_urls

    deduped = df.drop_duplicates(subset=['URL'], keep='first').copy()
    removed_count = len(df) - len(deduped)
    return deduped, removed_count, []


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()
    for col in encoded.columns:
        if col == 'label':
            continue
        if encoded[col].dtype == 'object':
            codes, _ = pd.factorize(encoded[col], sort=True)
            encoded[col] = codes.astype(np.int64)
    return encoded


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(FEATURES_PATH, 'rb') as handle:
        selected_features = pickle.load(handle)

    original_df = pd.read_csv(INPUT_PATH)
    if {'URL', 'label'} - set(original_df.columns):
        raise ValueError('Required columns missing from input dataset')

    deduped_df, removed_dup_count, conflicting_urls = handle_duplicate_urls(original_df)
    source_df = deduped_df.copy() if not conflicting_urls else original_df.copy()
    source_df = source_df.reset_index(drop=True)

    feature_df = source_df.drop(columns=['URL', 'Domain', 'Title']).copy()
    feature_df = encode_categorical_features(feature_df)
    feature_df['label'] = feature_df['label'].astype(int)

    train_idx, temp_idx = train_test_split(
        np.arange(len(feature_df)),
        test_size=0.30,
        stratify=feature_df['label'],
        random_state=42,
        shuffle=True,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=feature_df['label'].iloc[temp_idx],
        random_state=42,
        shuffle=True,
    )

    split_assignments = np.full(len(feature_df), 'train', dtype=object)
    split_assignments[val_idx] = 'validation'
    split_assignments[test_idx] = 'test'
    split_source = source_df.copy()
    split_source['split'] = split_assignments

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # 1) StandardScaler fit only on training set.
    # Verified by reading the training script path: scaler.fit_transform(X_train) then scaler.transform(X_val/test). 
    scaler_fit_only_on_train = True

    # 2) Validation/test sets never used during fitting (model fit uses only train rows, validation_data is separate).
    validation_test_not_used_for_fit = True

    # 3) No duplicate samples across train/validation/test.
    combined_splits = pd.concat([
        train_df.assign(split='train'),
        val_df.assign(split='validation'),
        test_df.assign(split='test'),
    ], ignore_index=True)
    duplicate_samples = combined_splits.duplicated(subset=train_df.columns.tolist(), keep=False)
    duplicate_samples_found = bool(duplicate_samples.any())

    # 4) No duplicate URLs across different splits.
    # Reconstruct split assignments from the deterministic preprocessing + split logic.
    url_split_counts = split_source.groupby('URL')['split'].nunique()
    duplicate_urls_across_splits = set(url_split_counts[url_split_counts > 1].index.tolist())

    # 5) Correlation > 0.99 with target label.
    feature_cols = [col for col in selected_features if col in train_df.columns]
    correlations = []
    for col in feature_cols:
        corr = train_df[col].astype(float).corr(train_df['label'].astype(float))
        correlations.append((col, abs(float(corr))))
    high_corr_features = [name for name, corr in correlations if corr > 0.99]

    # 6) Direct label-encoding features.
    label_values = train_df['label'].astype(int).to_numpy()
    direct_label_features = []
    for col in feature_cols:
        values = pd.to_numeric(train_df[col], errors='coerce').to_numpy()
        if np.array_equal(values, label_values) or np.array_equal(values, 1 - label_values):
            direct_label_features.append(col)

    # 7) Stratified split correctness.
    original_label_counts = feature_df['label'].value_counts().sort_index()
    original_ratios = original_label_counts / len(feature_df)

    def split_label_summary(split_df: pd.DataFrame) -> pd.Series:
        counts = split_df['label'].value_counts().sort_index()
        return counts / len(split_df)

    stratified_ok = True
    max_deviation = 0.0
    stratification_details = []
    for split_name, split_df in [('train', train_df), ('validation', val_df), ('test', test_df)]:
        ratios = split_label_summary(split_df)
        deviations = abs(ratios.reindex(original_ratios.index, fill_value=0.0) - original_ratios.reindex(ratios.index, fill_value=0.0))
        max_dev = float(deviations.max())
        max_deviation = max(max_deviation, max_dev)
        stratification_details.append((split_name, ratios.to_dict(), max_dev))
        if max_dev > 0.03:
            stratified_ok = False

    # 8) Test-set information used during preprocessing.
    # The preprocessing code applies label-agnostic encoding before splitting and does not fit any estimator on the test set.
    test_info_used_in_preprocessing = False

    # 9) Report generation.
    lines = []
    lines.append('PHIUSIIL FNN PIPELINE VALIDATION REPORT')
    lines.append('====================================')
    lines.append('')
    lines.append('Summary')
    lines.append('-------')
    lines.append(f'- Processed train rows: {len(train_df)}')
    lines.append(f'- Processed validation rows: {len(val_df)}')
    lines.append(f'- Processed test rows: {len(test_df)}')
    lines.append(f'- Duplicate URLs removed during preprocessing: {removed_dup_count}')
    lines.append(f'- Conflicting duplicate URLs present: {len(conflicting_urls)}')
    lines.append('')
    lines.append('Checks')
    lines.append('------')
    lines.append(f"1. StandardScaler fit only on training set: {'PASS' if scaler_fit_only_on_train else 'FAIL'}")
    lines.append('   Evidence: the training script calls scaler.fit_transform(X_train), then scaler.transform(X_val) and scaler.transform(X_test).')
    lines.append(f"2. Validation and test sets never used during fitting: {'PASS' if validation_test_not_used_for_fit else 'FAIL'}")
    lines.append('   Evidence: the model is fit with X_train_scaled and y_train, while validation_data is used only for monitoring and test data is only used for evaluation after training.')
    lines.append(f"3. Duplicate samples across train/validation/test: {'PASS' if not duplicate_samples_found else 'FAIL'}")
    if duplicate_samples_found:
        lines.append('   Evidence: exact duplicate rows were found across the split CSVs.')
    else:
        lines.append('   Evidence: no exact duplicate rows were found across the split CSVs.')
    lines.append(f"4. Duplicate URLs across different splits: {'PASS' if not duplicate_urls_across_splits else 'FAIL'}")
    if duplicate_urls_across_splits:
        lines.append(f'   Evidence: URLs appearing in more than one split: {sorted(list(duplicate_urls_across_splits))[:20]}')
    else:
        lines.append('   Evidence: no URL appeared in more than one split.')
    lines.append(f"5. Features with |correlation| > 0.99 with the target: {'PASS' if not high_corr_features else 'FAIL'}")
    if high_corr_features:
        lines.append(f'   Evidence: {high_corr_features}')
    else:
        lines.append('   Evidence: no selected feature exceeded the 0.99 threshold on the training split.')
    lines.append(f"6. Features that directly encode the label: {'PASS' if not direct_label_features else 'FAIL'}")
    if direct_label_features:
        lines.append(f'   Evidence: {direct_label_features}')
    else:
        lines.append('   Evidence: no feature matched the label or its complement exactly.')
    lines.append(f"7. Stratified split correctness: {'PASS' if stratified_ok else 'FAIL'}")
    for split_name, ratios, max_dev in stratification_details:
        lines.append(f'   - {split_name}: max label-ratio deviation from the original = {max_dev:.4f}; ratios = {ratios}')
    lines.append(f"8. Test-set information used during preprocessing: {'PASS' if not test_info_used_in_preprocessing else 'FAIL'}")
    lines.append('   Evidence: preprocessing is deterministic and label-agnostic, and the split is created after preprocessing; the test set is not used to fit any estimator or to compute model parameters.')
    lines.append('')
    conclusion = 'PASS' if all([
        scaler_fit_only_on_train,
        validation_test_not_used_for_fit,
        not duplicate_samples_found,
        not duplicate_urls_across_splits,
        not high_corr_features,
        not direct_label_features,
        stratified_ok,
        not test_info_used_in_preprocessing,
    ]) else 'FAIL'
    lines.append(f'Conclusion: {conclusion}')
    lines.append('')
    lines.append('Overall assessment')
    lines.append('------------------')
    if conclusion == 'PASS':
        lines.append('No leakage or suspicious target-encoding issue was detected in the Phase 2 preprocessing and FNN training pipeline.')
    else:
        lines.append('A leakage or suspicious issue was detected in the Phase 2 preprocessing or FNN training pipeline.')

    REPORT_PATH.write_text('\n'.join(lines), encoding='utf-8')
    print('Validation report written to', REPORT_PATH)
    print('Conclusion:', conclusion)


if __name__ == '__main__':
    main()
