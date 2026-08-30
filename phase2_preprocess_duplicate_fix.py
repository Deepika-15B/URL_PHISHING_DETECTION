from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / 'data' / 'phiusiil_dataset' / 'Dataset_HTML.csv'
OUTPUT_DIR = ROOT / 'data' / 'processed_v2'
REPORT_PATH = ROOT / 'reports' / 'preprocessing_duplicate_fix_report.txt'


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()
    for col in encoded.columns:
        if col == 'label':
            continue
        if encoded[col].dtype == 'object':
            codes, _ = pd.factorize(encoded[col], sort=True)
            encoded[col] = codes.astype(np.int64)
        else:
            encoded[col] = pd.to_numeric(encoded[col], errors='coerce').astype(np.float64)
    return encoded


def build_report(original_count: int, removed_count: int, final_count: int, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, duplicate_feature_rows: int, duplicate_urls_across_splits: int) -> str:
    lines = []
    lines.append('PHIUSIIL DUPLICATE-FREE PREPROCESSING REPORT')
    lines.append('==========================================')
    lines.append('')
    lines.append('Input summary')
    lines.append('-------------')
    lines.append(f'- Source file: {INPUT_PATH}')
    lines.append(f'- Original sample count: {original_count}')
    lines.append(f'- Duplicate rows removed (based on all feature columns + label): {removed_count}')
    lines.append(f'- Final sample count after duplicate removal: {final_count}')
    lines.append('')
    lines.append('Split summary')
    lines.append('-------------')
    lines.append(f'- Train size: {len(train_df)}')
    lines.append(f'- Validation size: {len(val_df)}')
    lines.append(f'- Test size: {len(test_df)}')
    lines.append('')
    lines.append('Verification')
    lines.append('------------')
    lines.append(f"- Identical feature vectors across different splits: {'PASS' if duplicate_feature_rows == 0 else 'FAIL'}")
    lines.append(f"- Duplicate feature vector groups across splits: {duplicate_feature_rows}")
    lines.append(f"- Duplicate URLs across different splits: {'PASS' if duplicate_urls_across_splits == 0 else 'FAIL'}")
    lines.append(f"- Duplicate URLs across splits: {duplicate_urls_across_splits}")
    lines.append('')
    lines.append('Label distribution')
    lines.append('------------------')
    for name, split_df in [('train', train_df), ('validation', val_df), ('test', test_df)]:
        counts = split_df['label'].value_counts().sort_index().to_dict()
        lines.append(f'- {name}: {counts}')
    lines.append('')
    lines.append('Notes')
    lines.append('-----')
    lines.append('- Existing processed datasets were not modified.')
    lines.append('- This script created a new version under data/processed_v2/.')
    lines.append('- No model retraining was performed.')
    lines.append('')
    lines.append('End of report')
    lines.append('============')
    return '\n'.join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    if {'URL', 'label'} - set(df.columns):
        raise ValueError('Required columns missing from input dataset')

    original_count = len(df)
    feature_df = df.drop(columns=['URL', 'Domain', 'Title']).copy()
    feature_df = encode_categorical_features(feature_df)
    feature_df['label'] = feature_df['label'].astype(int)

    feature_columns = [col for col in feature_df.columns if col != 'label']
    dedup_mask = feature_df.duplicated(subset=feature_columns + ['label'], keep='first')
    deduped = feature_df.loc[~dedup_mask].reset_index(drop=True)
    removed_count = original_count - len(deduped)

    # Use the original metadata rows that survived deduplication.
    metadata = df.loc[~dedup_mask, ['URL', 'Domain', 'Title']].reset_index(drop=True)
    deduped_with_meta = pd.concat([metadata, deduped], axis=1)

    # Enforce URL-level uniqueness before splitting so the same URL cannot appear in multiple splits.
    duplicate_url_mask = deduped_with_meta.duplicated(subset=['URL'], keep='first')
    if deduped_with_meta['label'].groupby(deduped_with_meta['URL']).nunique().gt(1).sum() == 0:
        deduped_with_meta = deduped_with_meta.loc[~duplicate_url_mask].reset_index(drop=True)

    removed_count = original_count - len(deduped_with_meta)

    train_df, temp_df = train_test_split(
        deduped_with_meta,
        test_size=0.30,
        stratify=deduped_with_meta['label'],
        random_state=42,
        shuffle=True,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df['label'],
        random_state=42,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save the processed feature matrix and label only, matching the existing pipeline format.
    train_output = train_df.drop(columns=['URL', 'Domain', 'Title']).copy()
    val_output = val_df.drop(columns=['URL', 'Domain', 'Title']).copy()
    test_output = test_df.drop(columns=['URL', 'Domain', 'Title']).copy()

    train_output.to_csv(OUTPUT_DIR / 'train.csv', index=False)
    val_output.to_csv(OUTPUT_DIR / 'validation.csv', index=False)
    test_output.to_csv(OUTPUT_DIR / 'test.csv', index=False)

    combined_feature_rows = pd.concat([
        train_output.drop(columns=['label']).copy().assign(split='train'),
        val_output.drop(columns=['label']).copy().assign(split='validation'),
        test_output.drop(columns=['label']).copy().assign(split='test'),
    ], ignore_index=True)
    feature_key_cols = [col for col in combined_feature_rows.columns if col != 'split']
    duplicate_feature_rows = int(
        combined_feature_rows.groupby(feature_key_cols)['split'].nunique().gt(1).sum()
    )

    url_series = pd.concat([
        train_df[['URL']].assign(split='train'),
        val_df[['URL']].assign(split='validation'),
        test_df[['URL']].assign(split='test'),
    ], ignore_index=True)
    duplicate_urls_across_splits = int(url_series.groupby('URL')['split'].nunique().gt(1).sum())

    report_text = build_report(original_count, removed_count, len(deduped_with_meta), train_output, val_output, test_output, duplicate_feature_rows, duplicate_urls_across_splits)
    REPORT_PATH.write_text(report_text, encoding='utf-8')

    print('Created duplicate-free processed splits at', OUTPUT_DIR)
    print('Report written to', REPORT_PATH)
    print('Train/Validation/Test sizes:', len(train_output), len(val_output), len(test_output))
    print('Duplicate feature rows across splits:', duplicate_feature_rows)
    print('Duplicate URLs across splits:', duplicate_urls_across_splits)


if __name__ == '__main__':
    main()
