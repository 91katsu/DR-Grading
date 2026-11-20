"""Binary referable-DR metrics shared by eval_mtm_vt.py and tools/svm_dr.py."""
import csv
import os

import numpy as np

# Original paper results (test set, n=466)
REPORT = {
    "mtmDR": {"Accuracy": 0.9464, "Sensitivity": 0.9000, "Specificity": 0.9574},
    "vtDR": {"Accuracy": 0.9957, "Sensitivity": 0.9545, "Specificity": 0.9977},
}
TASKS = {"mtmDR": 2, "vtDR": 3}  # positive if grade >= k


def binary_metrics(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "Accuracy": (tp + tn) / len(y),
        "Sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "Specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "TP": tp, "FN": fn, "TN": tn, "FP": fp,
    }


def bootstrap_ci(y, pred, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    stats = {"Accuracy": [], "Sensitivity": [], "Specificity": []}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = binary_metrics(y[idx], pred[idx])
        for k in stats:
            stats[k].append(m[k])
    return {k: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) for k, v in stats.items()}


def youden_threshold(y, score):
    best_t, best_j = 0.5, -1
    for t in np.unique(score):
        pred = (score >= t).astype(int)
        m = binary_metrics(y, pred)
        j = m["Sensitivity"] + m["Specificity"] - 1
        if j > best_j:
            best_t, best_j = t, j
    return float(best_t)


def load_manifest(data_path):
    """<data_path>/manifest.csv (from tools/prepare_dr_data.py) -> {image name: row}; {} if absent."""
    path = os.path.join(data_path, "manifest.csv")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return {r["image"]: r for r in csv.DictReader(fh)}
