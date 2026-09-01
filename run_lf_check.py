from __future__ import annotations
import json, pickle, sys, time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score)
from tensorflow.keras import models as keras_models

ROOT = Path(r'e:/phishing_project_ieee/phishing_detection_ieee')
TRAIN_PATH = ROOT / 'data/processed_leakage_free/train.csv'
VAL_PATH   = ROOT / 'data/processed_leakage_free/validation.csv'
TEST_PATH  = ROOT / 'data/processed_leakage_free/test.csv'
FEATURES_PATH = ROOT / 'models/leakage_free/features_18.pkl'
IMPUTER_PATH  = ROOT / 'models/leakage_free/imputer.pkl'
SCALER_PATH   = ROOT / 'models/leakage_free/scaler.pkl'
LF_DIR = ROOT / 'models/leakage_free'
REPORT_DIR = ROOT / 'reports'
FIG_DIR = ROOT / 'reports/confusion_matrices'
FNN_PATH       = LF_DIR / 'fnn_leakage_free.keras'
DNN_PATH       = LF_DIR / 'dnn_leakage_free.keras'
TABNET_PATH    = LF_DIR / 'tabnet_leakage_free.zip'
WIDE_DEEP_PATH = LF_DIR / 'wide_deep_leakage_free.keras'
FORBIDDEN = {'URLSimilarityIndex', 'IsHTTPS'}

print('Existing models:')
print('  FNN:', FNN_PATH.exists(), str(FNN_PATH))
print('  DNN:', DNN_PATH.exists(), str(DNN_PATH))
print('  TabNet:', TABNET_PATH.exists(), str(TABNET_PATH))
print('  WideDeep:', WIDE_DEEP_PATH.exists(), str(WIDE_DEEP_PATH))

with open(FEATURES_PATH, 'rb') as f:
    features_18 = pickle.load(f)
feature_cols = list(features_18)
print('Features loaded:', len(feature_cols))
print('Feature list:', feature_cols)
for fe in FORBIDDEN:
    if fe in feature_cols:
        print('STOP: forbidden feature', fe, 'found!')
        sys.exit(1)
print('No forbidden features found. OK.')
