"""Read-only deployment benchmark for the completed Phase 2 model artifacts."""
from __future__ import annotations

import pickle
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
REPORT_PATH = ROOT / "reports" / "deployment_benchmark.txt"
REPEATS = 20


class RssMonitor:
    """Samples process RSS while inference runs (native-framework inclusive)."""
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(0.001)

    def __enter__(self):
        self.peak = self.process.memory_info().rss
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def load_scaled_test() -> tuple[np.ndarray, np.ndarray]:
    with (MODEL_DIR / "top20_features.pkl").open("rb") as handle:
        features = list(pickle.load(handle))
    test = pd.read_csv(ROOT / "data" / "processed_v2" / "test.csv")
    return test[features].to_numpy(dtype=np.float32), test["label"].to_numpy(dtype=np.int64)


def evaluate(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(np.int64)
    return {"accuracy": accuracy_score(y, prediction), "precision": precision_score(y, prediction, zero_division=0),
            "recall": recall_score(y, prediction, zero_division=0), "f1": f1_score(y, prediction, zero_division=0),
            "roc_auc": roc_auc_score(y, probability)}


def measure(predict, data: np.ndarray) -> tuple[float, float, float]:
    """Return average elapsed ms, peak RSS MB, and process CPU utilisation percent."""
    # Warm-up avoids counting lazy graph/kernel initialization in the benchmark.
    predict(data[: min(len(data), 32)])
    process = psutil.Process()
    times, peaks, cpus = [], [], []
    for _ in range(REPEATS):
        process.cpu_percent(None)
        base_rss = process.memory_info().rss
        with RssMonitor() as monitor:
            started = time.perf_counter()
            predict(data)
            elapsed = time.perf_counter() - started
        times.append(elapsed * 1000)
        peaks.append(max(0, monitor.peak - base_rss) / (1024 * 1024))
        cpus.append(process.cpu_percent(None))
    return float(np.mean(times)), float(np.max(peaks)), float(np.mean(cpus))


def keras_entry(name: str, model_name: str, scaler_name: str, x: np.ndarray, y: np.ndarray, wide_deep: bool = False):
    from tensorflow.keras.models import load_model
    with (MODEL_DIR / scaler_name).open("rb") as handle:
        scaled = pickle.load(handle).transform(x).astype(np.float32)
    model = load_model(MODEL_DIR / model_name, compile=False)
    predict = (lambda values: model.predict([values, values], verbose=0, batch_size=1024).ravel()) if wide_deep else (lambda values: model.predict(values, verbose=0, batch_size=1024).ravel())
    probability = predict(scaled)
    return {"name": name, "metrics": evaluate(y, probability), "params": int(model.count_params()),
            "size_mb": (MODEL_DIR / model_name).stat().st_size / 1024**2,
            "timings": {count: measure(predict, scaled[:count]) for count in (1, 100, 1000)},
            "architecture": "Wide linear branch + 3-layer dense branch" if wide_deep else ("Dense(128)-Dense(64)-Dense(32)-output" if name == "DNN" else "Dense(64)-Dense(32)-output"),
            "hidden_layers": 4 if wide_deep else (3 if name == "DNN" else 2),
            "complexity": "Two parallel branches; highest dense-operation overhead of the Keras models." if wide_deep else ("Three dense hidden layers; moderate O(sum layer products) cost." if name == "DNN" else "Two compact dense layers; lowest O(sum layer products) cost."),
            "deployment": "Straightforward Keras serving; extra dual-input wiring." if wide_deep else "Straightforward Keras serving with compact dense operations."}


def main() -> None:
    x, y = load_scaled_test()
    if len(x) < 1000:
        raise ValueError("The test split must contain at least 1,000 samples for this benchmark.")
    entries = [
        keras_entry("FNN", "fnn_phase2_v2.keras", "scaler_phase2_v2.pkl", x, y),
        keras_entry("DNN", "dnn_phase2.keras", "scaler_dnn_phase2.pkl", x, y),
        keras_entry("Wide & Deep", "wide_deep_phase2.keras", "scaler_wide_deep.pkl", x, y, wide_deep=True),
    ]
    with (MODEL_DIR / "scaler_tabnet.pkl").open("rb") as handle:
        tab_x = pickle.load(handle).transform(x).astype(np.float32)
    tabnet = TabNetClassifier()
    tabnet.load_model(str(MODEL_DIR / "tabnet_phase2.zip"))
    tab_predict = lambda values: tabnet.predict_proba(values)[:, 1]
    entries.append({"name": "TabNet", "metrics": evaluate(y, tab_predict(tab_x)),
                    "params": sum(p.numel() for p in tabnet.network.parameters()),
                    "size_mb": (MODEL_DIR / "tabnet_phase2.zip").stat().st_size / 1024**2,
                    "timings": {count: measure(tab_predict, tab_x[:count]) for count in (1, 100, 1000)},
                    "architecture": "Five-step attentive TabNet encoder with sparse feature masks",
                    "hidden_layers": 5, "complexity": "Five sequential attentive decision steps; more control-flow and activation overhead than dense networks.",
                    "deployment": "Portable PyTorch artifact, but requires PyTorch TabNet runtime and has higher serving complexity."})

    lines = ["PHASE 2 DEPLOYMENT BENCHMARK", "=" * 28, "", "Scope: read-only benchmark of the completed artifacts on corrected processed_v2/test.csv.",
             f"Timing protocol: each batch timed {REPEATS} times after warm-up; averages are reported. CPU is process CPU%, where 100% represents one fully used logical CPU. Peak RAM is peak RSS increase while inference executes; zero means no measurable additional RSS allocation after warm-up.", "",
             "Prediction performance", "-" * 22,
             "Model | Accuracy | Precision | Recall | F1-score | ROC-AUC"]
    for e in entries:
        m = e["metrics"]; lines.append(f"{e['name']} | {m['accuracy']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['roc_auc']:.6f}")
    lines += ["", "Deployment and resource performance", "-" * 35,
              "Model | Model size MB | Parameters | 1 sample ms | 100 samples ms | 1000 samples ms | Peak RAM MB | CPU %"]
    for e in entries:
        one, hundred, thousand = e["timings"][1], e["timings"][100], e["timings"][1000]
        lines.append(f"{e['name']} | {e['size_mb']:.4f} | {e['params']} | {one[0]:.4f} | {hundred[0]:.4f} | {thousand[0]:.4f} | {max(one[1], hundred[1], thousand[1]):.4f} | {np.mean([one[2], hundred[2], thousand[2]]):.2f}")
    lines += ["", "Complexity and deployment analysis", "-" * 36]
    for e in entries:
        lines += [f"{e['name']}", f"- Architecture: {e['architecture']}", f"- Hidden/decision layers: {e['hidden_layers']}", f"- Trainable parameters: {e['params']}", f"- Computational complexity: {e['complexity']}", f"- Ease of deployment: {e['deployment']}"]
    best = max(entries, key=lambda e: (e["metrics"]["f1"], e["metrics"]["roc_auc"], e["metrics"]["accuracy"]))
    fnn = next(e for e in entries if e["name"] == "FNN")
    lines += ["", "Final recommendation", "-" * 20,
              f"- Highest test-set performance under F1/ROC-AUC/accuracy tie-breaking: {best['name']}.",
              "- Recommended Flask integration: FNN, provided its prediction metrics remain effectively tied with the alternatives shown above. It offers the simplest model, smallest dense architecture, and avoids TabNet's sequential-attention and runtime-dependency overhead.",
              "- Trade-off: the table provides the measured speed and memory evidence. If FNN is within trivial metric differences of the leaders, those differences do not by themselves justify DNN, Wide & Deep, or TabNet complexity for real-time URL checks.",
              "- Statistical interpretation: this benchmark measures deployment behavior, not a new statistical test. The previous corrected-split results should be considered practically tied when their errors differ on only a handful of samples; a paired test or external holdout is required to claim a meaningful quality advantage."]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
